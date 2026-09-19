"""Sessao de treino: carrega o Qwen uma vez e reaproveita.

O ciclo antigo pagava duas cargas do modelo por rodada (uma para treinar, outra para
provar) e cada carga custava mais que o proprio treino. Aqui o modelo fica na GPU do
inicio ao fim do job: treinar e provar viram chamadas no mesmo objeto.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

from agentepc.config import load
from agentepc.prompting import completion_for, prompt_for


def usa_gpu(torch) -> bool:
    """device do config: auto (GPU se houver), cuda (exige) ou cpu (forca memoria)."""
    escolha = ((load().get("model") or {}).get("device") or "auto").lower()
    if escolha == "cpu":
        return False
    if escolha == "cuda":
        return True
    return bool(torch.cuda.is_available())


class Missing(RuntimeError):
    """Torch/peft/trl nao instalados."""


def _imports():
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:  # pragma: no cover - depende do ambiente
        raise Missing(
            f"deps de treino ausentes ({exc}). Rode: "
            "uv venv .venv-train && uv pip install --python .venv-train/bin/python -r requirements-train.txt"
        ) from exc
    return locals()


class TrainSession:
    """Modelo carregado uma vez; varias rodadas de treino e provas em cima dele."""

    def __init__(self, pairs: list[dict], stop, say=lambda msg: None):
        self.pairs = pairs
        self.stop = stop
        self.say = say
        self.m = _imports()
        self.model = None
        self.tok = None
        self.rounds_done = 0

    # ---------------------------------------------------------------- carga
    def open(self, resume_from: Path | None = None) -> None:
        m, cfg = self.m, load()
        torch = m["torch"]
        base = cfg["model"]["base_hf"]
        tcfg = cfg["train"]
        self.gpu = usa_gpu(torch)
        onde = "na GPU, em 4 bits" if self.gpu else "na memoria (CPU), sem quantizar"
        self.say(f"carregando {base} {onde} — uma vez para o treino inteiro")
        self.tok = m["AutoTokenizer"].from_pretrained(base, use_fast=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        if self.gpu:
            bnb = m["BitsAndBytesConfig"](
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
            )
            model = m["AutoModelForCausalLM"].from_pretrained(base, quantization_config=bnb, device_map="auto")
            model = m["prepare_model_for_kbit_training"](model)
        else:
            # bitsandbytes so quantiza em CUDA; na CPU o jeito e float32 e modelo pequeno
            model = m["AutoModelForCausalLM"].from_pretrained(base, dtype=torch.float32, device_map="cpu")
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        if resume_from and (Path(resume_from) / "adapter_config.json").exists():
            model = m["PeftModel"].from_pretrained(model, str(resume_from), is_trainable=True)
        else:
            model = m["get_peft_model"](
                model,
                m["LoraConfig"](
                    r=int(tcfg["lora_r"]),
                    lora_alpha=int(tcfg["lora_alpha"]),
                    lora_dropout=float(tcfg.get("lora_dropout", 0.05)),
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=list(tcfg["target_modules"]),
                ),
            )
        self.model = model
        train, total = 0, 0
        for p in model.parameters():
            total += p.numel()
            if p.requires_grad:
                train += p.numel()
        self.say(f"adaptador com {train / 1e6:.1f}M pesos treinaveis em {list(tcfg['target_modules'])}")

    def add_adapter(self, path: Path, name: str) -> None:
        """Mais um conhecimento no mesmo modelo carregado (para testar varios em sequencia)."""
        self.model.load_adapter(str(path), adapter_name=name)

    def use_adapter(self, name: str) -> None:
        self.model.set_adapter(name)

    # --------------------------------------------------------------- treino
    def _dataset(self):
        rows = [
            {"prompt": prompt_for(p["user"]), "completion": completion_for(p["assistant"])}
            for p in self.pairs
        ]
        return self.m["Dataset"].from_list(rows)

    def train(self, rounds: int) -> bool:
        """Treina `rounds` rodadas. False se o usuario mandou parar no meio."""
        m, cfg = self.m, load()
        tcfg = cfg["train"]
        stop = self.stop

        class _StopNow(m["TrainerCallback"]):
            def on_step_end(self, args, state, control, **kwargs):
                if stop.is_set():
                    control.should_training_stop = True

        self.tok.padding_side = "right"
        self.model.train()
        self.model.config.use_cache = False
        args = m["SFTConfig"](
            output_dir="/tmp/agentepc-train",
            num_train_epochs=float(tcfg["epochs"]) * max(1, int(rounds)),
            per_device_train_batch_size=int(tcfg["batch_size"]),
            gradient_accumulation_steps=int(tcfg["grad_accum"]),
            learning_rate=float(tcfg["learning_rate"]),
            # sessao longa: LR constante para uma rodada valer o mesmo no inicio e no fim
            lr_scheduler_type="constant",
            logging_steps=1000,
            save_strategy="no",
            report_to=[],
            fp16=False,
            bf16=False,
            max_length=int(tcfg["max_seq_len"]),
            completion_only_loss=True,
            disable_tqdm=True,
        )
        trainer = m["SFTTrainer"](
            model=self.model,
            args=args,
            train_dataset=self._dataset(),
            processing_class=self.tok,
            callbacks=[_StopNow()],
        )
        out = trainer.train()
        del trainer
        if self.stop.is_set():
            return False
        self.rounds_done += rounds
        self.last_loss = float(out.training_loss)
        return True

    # ----------------------------------------------------------------- prova
    def answer(self, questions: list[str], max_new_tokens: int = 48) -> list[str]:
        """Responde em lote, com o adaptador ligado e sem caderno."""
        from agentepc.engine import clean_answer

        m = self.m
        torch = m["torch"]
        self.model.eval()
        self.model.config.use_cache = True
        try:
            self.model.gradient_checkpointing_disable()
        except Exception:
            pass
        self.tok.padding_side = "left"
        out: list[str] = []
        batch = max(1, int((load().get("learn") or {}).get("probe_batch") or 8))
        if not getattr(self, "gpu", True):
            batch = min(batch, 2)  # lote grande na CPU so faz esperar mais
        for i in range(0, len(questions), batch):
            if self.stop.is_set():
                break
            chunk = [prompt_for(q) for q in questions[i : i + batch]]
            enc = self.tok(chunk, return_tensors="pt", padding=True).to(self.model.device)
            with torch.no_grad():
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tok.pad_token_id,
                )
            for row in gen[:, enc["input_ids"].shape[1] :]:
                out.append(clean_answer(self.tok.decode(row, skip_special_tokens=True)))
        self.model.config.use_cache = False
        try:
            self.model.gradient_checkpointing_enable()
        except Exception:
            pass
        self.tok.padding_side = "right"
        return out

    # ----------------------------------------------------------------- disco
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        self.tok.save_pretrained(str(path))

    def close(self) -> None:
        model, self.model, self.tok = self.model, None, None
        del model
        try:
            torch = self.m["torch"]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
