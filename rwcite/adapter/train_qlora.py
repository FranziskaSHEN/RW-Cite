"""Citation-sentence QLoRA SFT for Qwen3-32B with hidden reasoning disabled."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import torch
import transformers
import yaml
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from rwcite.adapter import V7_ALL_TASKS
from rwcite.adapter.prompts import sample_to_messages

_ALLOWED_V6 = frozenset({"cite_bundle", "cite_one"})
_ALLOWED_DEFAULT = _ALLOWED_V6 | V7_ALL_TASKS


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _is_main() -> bool:
    return _local_rank() == 0


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_train_jsonl(
    path: Path,
    *,
    allowed_tasks: frozenset[str] | None = None,
) -> list[dict]:
    allowed = allowed_tasks if allowed_tasks is not None else _ALLOWED_DEFAULT
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            task = (row.get("task") or "").strip()
            if task not in allowed:
                raise SystemExit(
                    f"ERROR: train jsonl contains disallowed task {task!r} in {path}; "
                    f"allowed={sorted(allowed)}"
                )
            rows.append(row)
    if not rows:
        raise SystemExit(f"ERROR: empty train jsonl: {path}")
    return rows


def apply_chat(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
        "return_tensors": None,
    }
    # Qwen3: disable thinking for Cite/Decider SFT
    try:
        out = tokenizer.apply_chat_template(
            messages, enable_thinking=False, **kwargs
        )
    except TypeError as e:
        raise RuntimeError(
            "tokenizer.apply_chat_template does not accept enable_thinking=False; "
            "upgrade transformers for Qwen3 or check chat_template"
        ) from e
    # Newer transformers may return BatchEncoding instead of List[int].
    if hasattr(out, "input_ids"):
        ids = out["input_ids"]
        if ids and isinstance(ids[0], (list, tuple)):
            return list(ids[0])
        return list(ids)
    if isinstance(out, list) and out and isinstance(out[0], list):
        return list(out[0])
    return list(out)


def tokenize_messages(
    tokenizer,
    messages: list[dict[str, str]],
    max_length: int,
    mask_prompt: bool,
) -> dict:
    prompt_msgs = messages[:-1]
    full_ids = apply_chat(tokenizer, messages, add_generation_prompt=False)
    if mask_prompt:
        prompt_ids = apply_chat(
            tokenizer, prompt_msgs, add_generation_prompt=True
        )
        prompt_len = len(prompt_ids)
    else:
        prompt_len = 0
    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
    labels = list(full_ids)
    cut = min(prompt_len, len(labels))
    for i in range(cut):
        labels[i] = -100
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def smoke_decode(tokenizer, tokenized: dict) -> None:
    """Fail if non-empty thinking content leaks into SFT tokens.

    Qwen3 chat_template intentionally inserts empty ``<think>\\n\\n</think>``
    markers when ``enable_thinking=False``; those are allowed.
    """
    ids = tokenized["input_ids"]
    text = tokenizer.decode(ids, skip_special_tokens=False)
    # Strip empty disable-thinking markers, then reject residual think blocks.
    cleaned = re.sub(r"<think>\s*</think>\s*", "", text)
    if "<think>" in cleaned or "</think>" in cleaned:
        raise SystemExit(
            "ERROR: non-empty thinking content after enable_thinking=False tokenize smoke"
        )


def require_train_extras() -> None:
    missing = []
    for name in ("peft", "bitsandbytes", "accelerate"):
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise SystemExit(
            "ERROR: missing packages "
            f"{missing}; run: pip install -e '.[train]'"
        )


def train_from_config(
    config_path: Path,
    *,
    train_jsonl: Path,
    output_dir: Path,
    trainer_output_dir: Path | None = None,
    fresh: bool = True,
    resume_from_checkpoint: str | None = None,
    allowed_tasks: frozenset[str] | None = None,
    wandb_project: str | None = None,
) -> Path:
    require_train_extras()
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    config = load_yaml(config_path)
    base_id = config["inference"]["base_model"]
    if not Path(base_id).is_absolute():
        # resolve relative to RWCITE_ROOT / cwd
        root = Path(os.environ.get("RWCITE_ROOT") or Path.cwd())
        cand = root / base_id
        if cand.exists():
            base_id = str(cand)

    rank = _local_rank()
    if _is_main():
        print("base", base_id)
        print("data", train_jsonl)
        print("out", output_dir)

    rows = load_train_jsonl(train_jsonl, allowed_tasks=allowed_tasks)

    bnb = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_use_double_quant=True,
        bnb_8bit_quant_type="nf8",
        bnb_8bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    if tokenizer.chat_template is None:
        raise RuntimeError(f"{base_id} has no chat_template")
    max_length = int(config["training"]["tokenizer"]["max_length"])
    tokenizer.model_max_length = max_length
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_id,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map={"": rank},
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    qlora = config["training"]["qlora"]
    lora = LoraConfig(
        r=int(qlora["rank"]),
        lora_alpha=int(qlora["lora_alpha"]),
        target_modules=list(qlora["target_modules"]),
        lora_dropout=float(qlora["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)

    mask_prompt = bool(config["training"].get("mask_prompt_labels", True))
    tokenized_rows: list[dict] = []
    for i, sample in enumerate(rows):
        msgs = sample_to_messages(sample)
        tok = tokenize_messages(tokenizer, msgs, max_length, mask_prompt)
        if i == 0 and _is_main():
            smoke_decode(tokenizer, tok)
        tokenized_rows.append(tok)
    if _is_main():
        print(f"tokenized {len(tokenized_rows)} samples", flush=True)
    ds = Dataset.from_list(tokenized_rows)

    ta = config["training"]["trainer_args"]
    out_train = Path(
        trainer_output_dir
        or ta.get("trainer_output_dir")
        or (output_dir.parent / "trainer_outputs")
    )
    out_train.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    resume = resume_from_checkpoint
    if resume is None and not fresh:
        ckpts = sorted(
            out_train.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-", 1)[1]),
        )
        if ckpts:
            resume = str(ckpts[-1])

    os.environ.setdefault("WANDB_MODE", "offline")
    try:
        import wandb

        if _is_main():
            wandb.init(
                project=wandb_project or "rwcite-rr-adapter-v6",
                config=config,
                reinit=True,
            )
    except Exception:
        pass

    trainer = transformers.Trainer(
        model=model,
        train_dataset=ds,
        args=transformers.TrainingArguments(
            output_dir=str(out_train),
            per_device_train_batch_size=int(ta["per_device_train_batch_size"]),
            gradient_accumulation_steps=int(
                ta.get("gradient_accumulation_steps") or ta.get("index") or 8
            ),
            warmup_steps=int(ta["warmup_steps"]),
            num_train_epochs=float(ta["num_train_epochs"]),
            learning_rate=float(ta["learning_rate"]),
            lr_scheduler_type=ta.get("lr_scheduler_type") or "cosine",
            bf16=True,
            fp16=False,
            logging_steps=int(ta.get("logging_steps") or 1),
            save_steps=int(ta.get("save_steps") or 100),
            save_total_limit=2,
            report_to=["wandb"] if os.environ.get("WANDB_MODE") != "disabled" else [],
            ddp_find_unused_parameters=False,
            remove_unused_columns=False,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    trainer.train(resume_from_checkpoint=resume)
    if _is_main():
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        print(f"saved adapter → {output_dir}", flush=True)
    return output_dir
