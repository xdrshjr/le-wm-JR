"""LLaVA-style WM↔LLM alignment training modules (wm-llm-alignment §2.4).

Two Lightning modules replace the Stage-1 norm-matching stub:

  ``AlignStage1Module`` — real feature alignment. Trains only the
  ``WorldModelProjector`` (WM + LLM frozen) so the projected predicted
  outcome latent ``ẑ₁`` lands where the LLM represents the *actual* result
  text. Loss = InfoNCE + ``λ`` (1 − cos) over (p_i, t_i) (spec §2.4 L1).

  ``InstructStage2Module`` — predict-conditioned instruction tuning.
  Trains projector + Qwen-LoRA + ``OutcomeHead`` (WM frozen) to write the
  gold solution while reading the WM soft tokens, plus an auxiliary
  verifier BCE. Loss = CE_LM + ``μ`` BCE (spec §2.4 L2).

The WM is frozen for both; honesty invariant holds — no candidate code is
executed here (labels were produced offline by ``build_alignment_data`` /
``gen_mbpp_traj``).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from pca.action.schema import RunTestArgs
from pca.inference.wm_reranker import serialize

_VISIBLE_SELECTOR = "visible_tests"
_TIMEOUT_SEC = 5
_MAX_OBS_CHARS = 4000
_LORA_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)

# Same chat framing as scripts/bench_humaneval / aligned_model (single
# source of distribution for problem prompts).
_SYS_MSG = (
    "You are an expert Python programmer. "
    "Complete the given function. "
    "Output ONLY the function body, no explanation, no markdown, "
    "no extra definitions. Indent every line with 4 spaces."
)


def _user_msg(problem_prompt: str) -> str:
    return (
        "Complete this Python function. "
        "Return only the function body (no `def` line, no imports, "
        "no markdown fence):\n\n"
        f"```python\n{problem_prompt}```\n"
    )


def _run_test_op() -> RunTestArgs:
    return RunTestArgs(selector=_VISIBLE_SELECTOR, timeout_sec=_TIMEOUT_SEC)


def info_nce(
    p: torch.Tensor, t: torch.Tensor, tau: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric in-batch InfoNCE + mean cosine (spec §2.4 L1)."""
    p = F.normalize(p.float(), dim=-1)
    t = F.normalize(t.float(), dim=-1)
    logits = p @ t.t() / tau
    target = torch.arange(p.size(0), device=p.device)
    loss = 0.5 * (
        F.cross_entropy(logits, target)
        + F.cross_entropy(logits.t(), target)
    )
    cos = (p * t).sum(-1).mean()
    return loss, cos


# ----- Stage-1: feature alignment -------------------------------------


class AlignStage1Module(pl.LightningModule):
    """Train the projector to speak the LLM's language (spec §2.4 L1)."""

    def __init__(
        self,
        world_model: torch.nn.Module,
        projector: torch.nn.Module,
        llm_name: str,
        optimizer_cfg: dict,
        loss_cfg: dict,
    ) -> None:
        super().__init__()
        self.world_model = world_model
        self.projector = projector
        self.llm_name = llm_name
        self.optimizer_cfg = dict(optimizer_cfg)
        self.tau = float(loss_cfg.get("tau", 0.07))
        self.lam = float(loss_cfg.get("lam", 1.0))
        self._text_model = None
        self._text_tok = None

    def _ensure_text_encoder(self) -> None:
        if self._text_model is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.llm_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModel.from_pretrained(
            self.llm_name, torch_dtype=torch.float16
        )
        for p in model.parameters():
            p.requires_grad_(False)
        self._text_tok = tok
        self._text_model = model.to(self.device).eval()

    def _text_vec(self, texts: list[str]) -> torch.Tensor:
        """Frozen LLM masked-mean sentence vector → (B, d_llm)."""
        enc = self._text_tok(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=256,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            hidden = self._text_model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        return pooled.float()

    def _predict_z1(self, batch: dict) -> torch.Tensor:
        with torch.no_grad():
            info = self.world_model.encode(batch)
            emb, act = info["emb"], info["act_emb"]
            z1 = self.world_model.predict(emb[:, :1], act[:, :1])[:, -1]
        return z1.detach().float()

    def training_step(self, batch, batch_idx):
        self._ensure_text_encoder()
        z1 = self._predict_z1(batch)
        p = self.projector(z1).mean(dim=1)  # (B, d_llm), mean over K tokens
        result_texts = [row[1] for row in batch["obs_text"]]
        t = self._text_vec(result_texts)
        nce, cos = info_nce(p, t, self.tau)
        loss = nce + self.lam * (1.0 - cos)
        self.log_dict(
            {"stage1/nce": nce.detach(), "stage1/cos": cos.detach(),
             "stage1/loss": loss.detach()},
            prog_bar=True, sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        cfg = dict(self.optimizer_cfg)
        optim_cls = getattr(torch.optim, cfg.pop("type"))
        return optim_cls(self.projector.parameters(), **cfg)


# ----- Stage-2: predict-conditioned instruction tuning ----------------


class InstructDataset(Dataset):
    """Instruction samples from ``<dir>/instruct.jsonl`` (spec §5).

    Each row = ``{problem, gold_code, draft, exec_label}``. A deterministic
    seed shuffle carves a held-out val split so ckpt selection never peeks
    at HumanEval (spec §6 red line).
    """

    def __init__(
        self, path: str, split: str = "train",
        val_frac: float = 0.1, seed: int = 0,
    ) -> None:
        super().__init__()
        rows = self._read(Path(path) / "instruct.jsonl")
        random.Random(seed).shuffle(rows)
        n_val = int(round(len(rows) * val_frac))
        self.rows = rows[n_val:] if split == "train" else rows[:n_val]

    @staticmethod
    def _read(path: Path) -> list[dict]:
        if not path.exists():
            raise FileNotFoundError(
                f"instruct jsonl not found: {path} "
                "(run scripts/build_alignment_data.py first)"
            )
        return [
            json.loads(l)
            for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


def collate_instruct(batch: list[dict]) -> dict:
    items = list(batch)
    return {
        "problem": [it["problem"] for it in items],
        "gold_code": [it["gold_code"] for it in items],
        "draft": [it.get("draft", "") for it in items],
        "exec_label": torch.tensor(
            [float(it.get("exec_label", 0.0)) for it in items],
            dtype=torch.float32,
        ),
    }


class InstructStage2Module(pl.LightningModule):
    """Predict-conditioned instruction tuning (spec §2.4 L2)."""

    def __init__(
        self,
        world_model: torch.nn.Module,
        projector: torch.nn.Module,
        llm_name: str,
        train_cfg: dict,
    ) -> None:
        super().__init__()
        self.world_model = world_model
        self.projector = projector
        self.mu = float(train_cfg.get("mu", 0.5))
        self.optimizer_cfg = dict(train_cfg.get("optimizer", {"type": "AdamW"}))
        self.tokenizer, self.llm = self._build_llm(
            llm_name, dict(train_cfg.get("lora", {}))
        )

    def _build_llm(self, llm_name: str, lora_cfg: dict):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            llm_name, torch_dtype=torch.float16, trust_remote_code=True
        )
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
        lora = LoraConfig(
            r=int(lora_cfg.get("rank", 8)),
            lora_alpha=int(lora_cfg.get("alpha", 16)),
            lora_dropout=float(lora_cfg.get("dropout", 0.05)),
            target_modules=list(lora_cfg.get("targets", _LORA_TARGETS)),
            task_type="CAUSAL_LM",
        )
        return tok, get_peft_model(base, lora)

    def _format_chat(self, prompt: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": _SYS_MSG},
             {"role": "user", "content": _user_msg(prompt)}],
            tokenize=False, add_generation_prompt=True,
        )

    def _predict_z1(
        self, prompts: list[str], drafts: list[str]
    ) -> torch.Tensor:
        texts = [
            serialize(p, d)[:_MAX_OBS_CHARS]
            for p, d in zip(prompts, drafts)
        ]
        info = {
            "obs_text": [[t] for t in texts],
            "op": [[_run_test_op()] for _ in texts],
        }
        with torch.no_grad():
            info = self.world_model.encode(info)
            emb, act = info["emb"], info["act_emb"]
            z1 = self.world_model.predict(emb[:, :1], act[:, :1])[:, -1]
        return z1.detach().float()

    def _row_embeds(self, prompt: str, gold: str, soft: torch.Tensor):
        """One (embeds, labels) row: [chat ⊕ soft ⊕ gold], CE on gold."""
        embed = self.llm.get_input_embeddings()
        p_ids = self.tokenizer(
            self._format_chat(prompt), return_tensors="pt"
        ).input_ids.to(self.device)
        g_ids = self.tokenizer(
            gold + self.tokenizer.eos_token, return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(self.device)
        s = soft.to(embed.weight.dtype)
        emb = torch.cat([embed(p_ids)[0], s, embed(g_ids)[0]], dim=0)
        lab = torch.full(
            (emb.size(0),), -100, dtype=torch.long, device=self.device
        )
        lab[p_ids.size(1) + s.size(0):] = g_ids[0]
        return emb, lab

    def _pack(self, problems, golds, soft):
        rows = [
            self._row_embeds(prob, gold, soft[i])
            for i, (prob, gold) in enumerate(zip(problems, golds))
        ]
        maxlen = max(e.size(0) for e, _ in rows)
        bsz, dim = len(rows), rows[0][0].size(1)
        embs = torch.zeros(
            bsz, maxlen, dim, device=self.device, dtype=rows[0][0].dtype
        )
        labs = torch.full(
            (bsz, maxlen), -100, dtype=torch.long, device=self.device
        )
        attn = torch.zeros(bsz, maxlen, dtype=torch.long, device=self.device)
        for i, (e, l) in enumerate(rows):
            n = e.size(0)
            embs[i, :n], labs[i, :n], attn[i, :n] = e, l, 1
        return embs, attn, labs

    def _outcome_bce(self, z1: torch.Tensor, labels: torch.Tensor):
        head = getattr(self.world_model, "outcome_head", None)
        if head is None:
            return torch.zeros((), device=self.device)
        logit = head(z1).squeeze(-1)
        return F.binary_cross_entropy_with_logits(
            logit, labels.to(logit.device, dtype=logit.dtype)
        )

    def training_step(self, batch, batch_idx):
        z1 = self._predict_z1(batch["problem"], batch["draft"])
        soft = self.projector(z1)  # (B, K, d_llm)
        embs, attn, lm_labels = self._pack(
            batch["problem"], batch["gold_code"], soft
        )
        ce = self.llm(
            inputs_embeds=embs, attention_mask=attn, labels=lm_labels
        ).loss
        bce = self._outcome_bce(z1, batch["exec_label"])
        loss = ce + self.mu * bce
        self.log_dict(
            {"stage2/ce": ce.detach(), "stage2/bce": bce.detach(),
             "stage2/loss": loss.detach()},
            prog_bar=True, sync_dist=True,
        )
        return loss

    def trainable_parameters(self):
        head = getattr(self.world_model, "outcome_head", None)
        params = list(self.projector.parameters())
        params += [p for p in self.llm.parameters() if p.requires_grad]
        if head is not None:
            params += list(head.parameters())
        return params

    def configure_optimizers(self):
        cfg = dict(self.optimizer_cfg)
        optim_cls = getattr(torch.optim, cfg.pop("type"))
        return optim_cls(self.trainable_parameters(), **cfg)
