from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import csv
import importlib.util
import json
import math
import os
import sys
import tempfile
import threading
import time
import traceback
import types
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from peft import PeftModel
from scipy.signal import resample_poly
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import RungeKutta4ODESolver


ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "webui/omni_realtime_v1"
OUT_ROOT = ROOT / "artifacts/omni3b_realtime_webui_v1"
MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
CACHE_DIR = ROOT / ".cache/huggingface"
ENGLISH_LORA = ROOT / "artifacts/omni3b_control_response_lora_v1/control_response_3000_e2_from_label_lora/best_lora"
ENGLISH_CODEBOOK = {"WAIT": "/W", "BACKCHANNEL": "/B", "SUPPORT": "/S"}
ADAPTER_PROFILES = {
    "english": {
        "display_name": "English 3-action LoRA",
        "language": "en",
        "peft_name": "english",
        "path": ENGLISH_LORA,
        "codebook": ENGLISH_CODEBOOK,
        "prompt_mode": "english_three_action",
        "max_new_tokens": 32,
    },
}
DEFAULT_ADAPTER = "english"
LAYER = 3
ALPHA = 4.0
POSITION = "all_tokens"
TARGET_SR = 24000
PORTABLE_MSVC = ROOT / "tools/portable-msvc/msvc"


def configure_portable_msvc() -> bool:
    if os.name != "nt" or not PORTABLE_MSVC.exists():
        return False
    msvc_versions = sorted((PORTABLE_MSVC / "VC/Tools/MSVC").glob("*"), reverse=True)
    sdk_versions = sorted((PORTABLE_MSVC / "Windows Kits/10/bin").glob("*"), reverse=True)
    if not msvc_versions or not sdk_versions:
        return False
    msvc = msvc_versions[0]
    sdk_version = sdk_versions[0].name
    sdk = PORTABLE_MSVC / "Windows Kits/10"
    path_items = [
        msvc / "bin/Hostx64/x64",
        sdk / "bin" / sdk_version / "x64",
        sdk / "bin" / sdk_version / "x64/ucrt",
    ]
    include_items = [
        msvc / "include",
        sdk / "Include" / sdk_version / "ucrt",
        sdk / "Include" / sdk_version / "shared",
        sdk / "Include" / sdk_version / "um",
        sdk / "Include" / sdk_version / "winrt",
        sdk / "Include" / sdk_version / "cppwinrt",
    ]
    lib_items = [
        msvc / "lib/x64",
        sdk / "Lib" / sdk_version / "ucrt/x64",
        sdk / "Lib" / sdk_version / "um/x64",
    ]
    os.environ["PATH"] = os.pathsep.join(str(p) for p in path_items) + os.pathsep + os.environ.get("PATH", "")
    os.environ["INCLUDE"] = os.pathsep.join(str(p) for p in include_items)
    os.environ["LIB"] = os.pathsep.join(str(p) for p in lib_items)
    os.environ["CXX"] = "cl"
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / ".cache/torchinductor"))
    return True


@torch.no_grad()
def gpu_token2wav_sample(
    self,
    conditioning_vector,
    reference_mel_spectrogram,
    quantized_code,
    num_steps=10,
    guidance_scale=0.5,
    sway_coefficient=-1.0,
):
    maximum_duration = quantized_code.shape[1] * self.repeats
    initial_state = torch.randn(
        [1, maximum_duration, self.mel_dim],
        dtype=reference_mel_spectrogram.dtype,
        device=quantized_code.device,
    )
    batch_size = reference_mel_spectrogram.shape[0]
    conditioning_vector = conditioning_vector.unsqueeze(1).repeat(1, maximum_duration, 1)
    if batch_size != 1:
        raise ValueError("Only batch size = 1 is currently supported")

    def ode_function(time_step, hidden_states):
        if guidance_scale < 1e-5:
            return self(
                hidden_states=hidden_states,
                speaker_embedding=conditioning_vector,
                condition_vector=reference_mel_spectrogram,
                quantized_code=quantized_code,
                time_step=time_step,
                drop_audio_conditioning=False,
                drop_code=False,
            )
        model_output = self(
            hidden_states=hidden_states,
            quantized_code=quantized_code,
            speaker_embedding=conditioning_vector,
            condition_vector=reference_mel_spectrogram,
            time_step=time_step,
            apply_cfg=True,
        )
        guided_prediction, null_prediction = torch.chunk(model_output, 2, dim=0)
        return guided_prediction + (guided_prediction - null_prediction) * guidance_scale

    time_embedding = torch.linspace(
        0,
        1,
        num_steps,
        device=quantized_code.device,
        dtype=conditioning_vector.dtype,
    )
    if sway_coefficient is not None:
        time_embedding += sway_coefficient * (torch.cos(torch.pi / 2 * time_embedding) - 1 + time_embedding)
    solution = RungeKutta4ODESolver(function=ode_function, initial_value=initial_state).integrate(time_embedding)
    return solution[-1].permute(0, 2, 1)


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


single = import_module(ROOT / "scripts/run_omni3b_single_token_lora_v1.py", "single_token_core_webui")
single.DATA_DIR = ROOT / "data/omni3b_delayed_backchannel_v2_qwen3tts_clean_0p6b"
single.SOURCE_DATA_DIR = ROOT / "data/omni3b_sequential_v2"
demo = import_module(ROOT / "scripts/run_control_response_lora_demo_v1.py", "control_response_demo_webui")
resp = import_module(ROOT / "scripts/run_omni3b_control_response_lora_v1.py", "response_lora_core_webui")


def now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def append_csv(path: Path, row: dict, fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def resample_audio(audio: np.ndarray, sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    if sr == target_sr:
        return audio.astype(np.float32)
    g = math.gcd(sr, target_sr)
    up = target_sr // g
    down = sr // g
    return resample_poly(audio.astype(np.float32), up, down).astype(np.float32)


def softmax_probs(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64)
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / np.maximum(np.sum(exp), 1e-12)


def parse_control_text(text: str, codebook: dict[str, str]) -> tuple[str, str]:
    cleaned = text.strip()
    for label, code in sorted(codebook.items(), key=lambda item: len(item[1]), reverse=True):
        if cleaned == code:
            return label, ""
        if cleaned.startswith(code) and cleaned[len(code) : len(code) + 1] in {" ", "\n", "\t"}:
            return label, cleaned[len(code) :].strip()
    return "", cleaned


@dataclass
class EngineConfig:
    speaker: str = "Chelsie"
    enable_talker: bool = True
    tts_backend: str = "omni"
    sapi_voice: str = "auto"
    sapi_rate: int = 2
    max_context_seconds: float = 24.0
    save_segments: bool = True
    attention_backend: str = "sdpa"
    compile_token2wav: bool = True
    warm_token2wav: bool = True


def thinker_hook_module(thinker, layer: int):
    candidates = [
        ("model", "layers"),
        ("model", "model", "layers"),
        ("base_model", "model", "model", "layers"),
    ]
    for path in candidates:
        obj = thinker
        for attr in path:
            if not hasattr(obj, attr):
                break
            obj = getattr(obj, attr)
        else:
            return obj[layer]
    names = [name for name, _ in thinker.named_modules() if name.endswith(f"model.layers.{layer}")]
    raise RuntimeError(f"Could not find Thinker layer {layer}; candidates={names[:8]}")


class ThinkerScheduler:
    def __init__(self):
        self.condition = threading.Condition()
        self.busy = False
        self.response_waiters = 0

    @contextmanager
    def slot(self, response_priority: bool = False):
        with self.condition:
            if response_priority:
                self.response_waiters += 1
            while self.busy or (not response_priority and self.response_waiters > 0):
                self.condition.wait()
            if response_priority:
                self.response_waiters -= 1
            self.busy = True
        try:
            yield
        finally:
            with self.condition:
                self.busy = False
                self.condition.notify_all()

    def state(self) -> dict:
        with self.condition:
            return {"busy": self.busy, "response_waiters": self.response_waiters}


class CancelOnEvent(StoppingCriteria):
    def __init__(self, event: threading.Event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self.event.is_set()


class ModelEngine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.status = "not_loaded"
        self.status_detail = ""
        self.lock = threading.Lock()
        self.adapter_switch_lock = threading.Lock()
        self.thinker_scheduler = ThinkerScheduler()
        self.talker_lock = threading.Lock()
        self.response_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="omni-response",
        )
        self.model = None
        self.decision_model_mode = (
            "shared_bf16_external_sapi"
            if config.tts_backend == "sapi"
            else "shared_bf16_cooperative_talker"
        )
        self.processor = None
        self.dtype = None
        self.token_ids = None
        self.adapter = None
        self.active_adapter_key = DEFAULT_ADAPTER
        self.loaded_adapter_keys: set[str] = set()
        self.input_cache: OrderedDict[str, tuple[Any, int]] = OrderedDict()
        self.runtime_stage = "idle"
        self.runtime_stage_since = time.time()
        self.component_devices: dict[str, str] = {}
        self.token2wav_compiled = False
        self.token2wav_compile_mode = "disabled"
        self.token2wav_warmup: dict[str, Any] = {}
        self.portable_msvc_configured = False
        self.load_started_at = None
        self.loaded_at = None

    def state(self) -> dict:
        active_profile = ADAPTER_PROFILES[self.active_adapter_key]
        return {
            "status": self.status,
            "detail": self.status_detail,
            "model": MODEL_ID,
            "lora": str(active_profile["path"]),
            "active_adapter": self.active_adapter_key,
            "active_adapter_name": active_profile["display_name"],
            "active_language": active_profile["language"],
            "active_labels": list(active_profile["codebook"]),
            "active_codebook": active_profile["codebook"],
            "adapters": [
                {
                    "key": key,
                    "display_name": profile["display_name"],
                    "language": profile["language"],
                    "path": str(profile["path"]),
                    "labels": list(profile["codebook"]),
                    "codebook": profile["codebook"],
                    "loaded": key in self.loaded_adapter_keys,
                    "active": key == self.active_adapter_key,
                }
                for key, profile in ADAPTER_PROFILES.items()
            ],
            "enable_talker": self.config.enable_talker,
            "enable_audio_output": self.config.tts_backend != "none",
            "tts_backend": self.config.tts_backend,
            "sapi_voice": self.config.sapi_voice if self.config.tts_backend == "sapi" else "",
            "sapi_rate": self.config.sapi_rate if self.config.tts_backend == "sapi" else None,
            "speaker": self.config.speaker,
            "decision_model_mode": self.decision_model_mode,
            "attention_backend": self.config.attention_backend,
            "runtime_stage": self.runtime_stage,
            "runtime_stage_seconds": max(0.0, time.time() - self.runtime_stage_since),
            "thinker_scheduler": self.thinker_scheduler.state(),
            "component_devices": self.component_devices,
            "token2wav_compiled": self.token2wav_compiled,
            "token2wav_compile_mode": self.token2wav_compile_mode,
            "token2wav_warmup": self.token2wav_warmup,
            "portable_msvc_configured": self.portable_msvc_configured,
            "loaded_at": self.loaded_at,
        }

    def set_runtime_stage(self, stage: str):
        self.runtime_stage = stage
        self.runtime_stage_since = time.time()

    def warm_token2wav_on_response_thread(self, model):
        speaker_params = model.speaker_map[self.config.speaker]
        for bucket in (96, 32):
            self.set_runtime_stage(f"warming_token2wav_{bucket}")
            warm_t0 = time.perf_counter()
            torch.compiler.cudagraph_mark_step_begin()
            with torch.inference_mode():
                warm_wav = model.token2wav(
                    torch.zeros((1, bucket), dtype=torch.long, device=model.device),
                    conditioning=speaker_params["cond"].to(model.device).float().contiguous(),
                    reference_mel=speaker_params["ref_mel"].to(model.device).float().contiguous(),
                    use_audio_in_video=False,
                )
            torch.cuda.synchronize()
            del warm_wav
            self.token2wav_warmup[str(bucket)] = {
                "status": "ready",
                "milliseconds": (time.perf_counter() - warm_t0) * 1000.0,
                "thread": threading.current_thread().name,
            }
        self.set_runtime_stage("idle")

    def load(self):
        with self.lock:
            if self.status == "loaded":
                return self.state()
            if self.status == "loading":
                return self.state()
            self.status = "loading"
            self.status_detail = "loading Qwen2.5-Omni-3B and LoRA"
            self.load_started_at = time.time()
        try:
            self.portable_msvc_configured = configure_portable_msvc()
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                MODEL_ID,
                cache_dir=str(CACHE_DIR),
                torch_dtype=dtype,
                device_map={"": 0} if torch.cuda.is_available() else "cpu",
                attn_implementation=self.config.attention_backend,
            )
            if not self.config.enable_talker:
                model.disable_talker()
            model.config.use_cache = True
            if hasattr(model.thinker, "config"):
                model.thinker.config.use_cache = True
            processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID, cache_dir=str(CACHE_DIR))
            for key, profile in ADAPTER_PROFILES.items():
                if not Path(profile["path"]).exists():
                    raise FileNotFoundError(f"LoRA adapter does not exist: {profile['path']}")
            model.thinker = PeftModel.from_pretrained(
                model.thinker,
                ENGLISH_LORA,
                adapter_name=ADAPTER_PROFILES["english"]["peft_name"],
                is_trainable=False,
            )
            model.thinker.set_adapter(ADAPTER_PROFILES[DEFAULT_ADAPTER]["peft_name"])
            model.eval()
            if self.config.enable_talker and torch.cuda.is_available():
                model.token2wav.to(device=torch.device("cuda:0"), dtype=torch.float32)
                model.token2wav.code2wav_dit_model.sample = types.MethodType(
                    gpu_token2wav_sample,
                    model.token2wav.code2wav_dit_model,
                )
                if self.config.compile_token2wav:
                    model.token2wav = torch.compile(
                        model.token2wav,
                        mode="reduce-overhead",
                        dynamic=False,
                    )
                    self.token2wav_compiled = True
                    self.token2wav_compile_mode = "inductor_static_cudagraph_buckets_32_96"
                    if self.config.warm_token2wav:
                        self.response_executor.submit(self.warm_token2wav_on_response_thread, model).result()
            component_devices = {
                "thinker": str(next(model.thinker.parameters()).device),
                "talker": str(next(model.talker.parameters()).device) if self.config.enable_talker else "disabled",
                "token2wav": str(next(model.token2wav.parameters()).device) if self.config.enable_talker else "disabled",
                "external_tts": "cpu/windows_sapi" if self.config.tts_backend == "sapi" else "disabled",
            }
            token_ids = {
                key: single.label_token_ids(processor.tokenizer, profile["codebook"])
                for key, profile in ADAPTER_PROFILES.items()
            }
            adapter, _ = single.load_time_adapter(LAYER)
            with self.lock:
                self.model = model
                self.processor = processor
                self.dtype = dtype
                self.token_ids = token_ids
                self.adapter = adapter
                self.active_adapter_key = DEFAULT_ADAPTER
                self.loaded_adapter_keys = set(ADAPTER_PROFILES)
                self.component_devices = component_devices
                self.status = "loaded"
                self.status_detail = f"loaded in {time.time() - self.load_started_at:.1f}s"
                self.loaded_at = datetime.now().isoformat(timespec="seconds")
            return self.state()
        except Exception as exc:
            with self.lock:
                self.status = "error"
                self.status_detail = repr(exc)
            raise

    def ensure_loaded(self):
        if self.status != "loaded":
            raise RuntimeError(f"model is not loaded: {self.status} {self.status_detail}")

    def switch_adapter(self, adapter_key: str) -> dict:
        self.ensure_loaded()
        if adapter_key not in ADAPTER_PROFILES:
            raise KeyError(f"unknown adapter: {adapter_key}")
        if adapter_key not in self.loaded_adapter_keys:
            raise RuntimeError(f"adapter is not loaded: {adapter_key}")
        with self.adapter_switch_lock:
            if adapter_key == self.active_adapter_key:
                return self.state()
            profile = ADAPTER_PROFILES[adapter_key]
            self.set_runtime_stage(f"switching_adapter_{adapter_key}")
            try:
                with self.thinker_scheduler.slot(response_priority=True), self.talker_lock:
                    self.model.thinker.set_adapter(profile["peft_name"])
                    self.active_adapter_key = adapter_key
                    self.input_cache.clear()
                    self.status_detail = f"adapter switched to {profile['display_name']}"
            finally:
                self.set_runtime_stage("idle")
        return self.state()

    def active_profile(self) -> dict:
        return ADAPTER_PROFILES[self.active_adapter_key]

    def prepare_active_inputs(self, row: dict) -> tuple[dict, int, str]:
        profile = self.active_profile()
        return single.prepare_inputs(
            self.processor,
            row,
            "audio_only",
            profile["codebook"],
            label=None,
            audio_timing_mode="row_audio",
        )

    def label_forward(self, row: dict) -> dict:
        self.ensure_loaded()
        adapter_key = self.active_adapter_key
        profile = ADAPTER_PROFILES[adapter_key]
        labels = list(profile["codebook"])
        vector = single.adapter_predict(self.adapter, [row])[0]
        prep_t0 = time.perf_counter()
        with self.thinker_scheduler.slot(response_priority=False):
            cache_key = f"{adapter_key}:{row['context_id']}:{row.get('audio_revision', row.get('context_audio_snapshot_samples', 0))}"
            cached = self.input_cache.get(cache_key)
            if cached is None:
                prompt_inputs, prompt_len, _ = self.prepare_active_inputs(row)
                self.input_cache[cache_key] = (prompt_inputs, prompt_len)
                self.input_cache.move_to_end(cache_key)
                while len(self.input_cache) > 16:
                    self.input_cache.popitem(last=False)
                input_cache_hit = False
            else:
                prompt_inputs, prompt_len = cached
                self.input_cache.move_to_end(cache_key)
                input_cache_hit = True
            thinker = self.model.thinker
            moved = single.move_inputs(prompt_inputs, thinker.device, self.dtype)
            prep_ms = (time.perf_counter() - prep_t0) * 1000.0

            hook = single.InjectionHook(vector, alpha=ALPHA, position=POSITION, enabled=True)
            handle = thinker_hook_module(thinker, LAYER).register_forward_hook(hook)
            try:
                forward_t0 = time.perf_counter()
                with torch.inference_mode():
                    outputs = thinker(**moved, use_cache=False, use_audio_in_video=False)
                forward_ms = (time.perf_counter() - forward_t0) * 1000.0
            finally:
                handle.remove()
        logits = outputs.logits[0, -1, :].detach().float().cpu().numpy()
        label_logits = np.array(
            [logits[self.token_ids[adapter_key][label]] for label in labels],
            dtype=np.float64,
        )
        probs = softmax_probs(label_logits)
        pred = labels[int(np.argmax(probs))]
        probability_map = {label: float(prob) for label, prob in zip(labels, probs)}
        logit_map = {label: float(logit) for label, logit in zip(labels, label_logits)}
        stat = hook.stats[0] if hook.stats else {}
        return {
            "prompt_len": int(prompt_len),
            "moved": moved,
            "vector": vector,
            "label": pred,
            "control_code": profile["codebook"][pred],
            "adapter_key": adapter_key,
            "adapter_name": profile["display_name"],
            "language": profile["language"],
            "label_probabilities": probability_map,
            "label_logits": logit_map,
            "p_WAIT": probability_map.get("WAIT", 0.0),
            "p_BACKCHANNEL": probability_map.get("BACKCHANNEL", 0.0),
            "p_SUPPORT": probability_map.get("SUPPORT", 0.0),
            "wait_logit": logit_map.get("WAIT", float("-inf")),
            "backchannel_logit": logit_map.get("BACKCHANNEL", float("-inf")),
            "support_logit": logit_map.get("SUPPORT", float("-inf")),
            "prep_ms": prep_ms,
            "forward_ms": forward_ms,
            "label_total_ms": prep_ms + forward_ms,
            "decision_model_mode": self.decision_model_mode,
            "input_cache_hit": input_cache_hit,
            "hook_calls": hook.calls,
            "time_vector_norm": stat.get("injected_norm", 0.0),
            "context_norm": stat.get("hidden_norm", 0.0),
            "time_context_cosine": stat.get("hidden_injected_cosine", 0.0),
        }

    def generate_text(self, moved, prompt_len: int, vector: np.ndarray) -> dict:
        self.ensure_loaded()
        profile = self.active_profile()
        response_moved = single.move_inputs(moved, self.model.device, self.dtype)
        with self.thinker_scheduler.slot(response_priority=True):
            self.set_runtime_stage("response_text")
            hook = single.InjectionHook(vector, alpha=ALPHA, position=POSITION, enabled=True)
            handle = single.hook_module(self.model, LAYER).register_forward_hook(hook)
            try:
                t0 = time.perf_counter()
                with torch.inference_mode():
                    generated = self.model.thinker.generate(
                        **response_moved,
                        max_new_tokens=profile["max_new_tokens"],
                        do_sample=False,
                        use_cache=True,
                        use_audio_in_video=False,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                    )
                gen_ms = (time.perf_counter() - t0) * 1000.0
            finally:
                handle.remove()
        text = self.processor.tokenizer.decode(
            generated[0, prompt_len:].detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        label, response_text = parse_control_text(text, profile["codebook"])
        return {
            "generated_raw": text,
            "generated_label": label,
            "generated_response": response_text,
            "text_generate_ms": gen_ms,
        }

    def generate_talker_audio(
        self,
        response_text: str,
        out_path: Path,
        cancel_event: threading.Event,
        response_label: str,
    ) -> dict:
        self.ensure_loaded()
        if self.config.tts_backend == "sapi":
            return self._generate_sapi_audio(response_text, out_path, cancel_event, response_label)
        if not self.config.enable_talker:
            return {"enabled": False, "path": "", "duration_s": 0.0, "spoken_text": ""}
        return self._generate_talker_audio_cooperative(response_text, out_path, cancel_event, response_label)

    def _generate_sapi_audio(
        self,
        response_text: str,
        out_path: Path,
        cancel_event: threading.Event,
        response_label: str,
    ) -> dict:
        import pythoncom
        import win32com.client

        total_t0 = time.perf_counter()
        if cancel_event.is_set():
            return {"enabled": False, "canceled": True, "path": "", "tts_backend": "sapi"}
        self.set_runtime_stage("sapi_tts")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pythoncom.CoInitialize()
        try:
            voice = win32com.client.Dispatch("SAPI.SpVoice")
            voices = voice.GetVoices()
            requested = self.config.sapi_voice.strip().lower()
            selected = None
            available = []
            for index in range(voices.Count):
                token = voices.Item(index)
                description = token.GetDescription()
                language = token.GetAttribute("Language")
                available.append({"description": description, "language": language})
                if requested != "auto" and requested in description.lower():
                    selected = token
                    break
                if requested == "auto" and selected is None:
                    selected = token
            if selected is None:
                raise RuntimeError(f"SAPI voice was not found; available={available}")
            voice.Voice = selected
            voice.Rate = int(self.config.sapi_rate)
            voice.Volume = 100
            stream = win32com.client.Dispatch("SAPI.SpFileStream")
            audio_format = win32com.client.Dispatch("SAPI.SpAudioFormat")
            audio_format.Type = 22  # 22.05 kHz, 16-bit, mono
            stream.Format = audio_format
            stream.Open(str(out_path), 3, False)  # SSFMCreateForWrite
            voice.AudioOutputStream = stream
            synth_t0 = time.perf_counter()
            voice.Speak(response_text)
            synthesis_ms = (time.perf_counter() - synth_t0) * 1000.0
            stream.Close()
            voice_name = selected.GetDescription()
        finally:
            pythoncom.CoUninitialize()

        post_t0 = time.perf_counter()
        audio, source_rate = sf.read(str(out_path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1).astype(np.float32)
        if int(source_rate) != TARGET_SR:
            audio = resample_audio(np.asarray(audio, dtype=np.float32), int(source_rate), TARGET_SR)
            sf.write(str(out_path), audio, TARGET_SR, subtype="PCM_16")
        post_ms = (time.perf_counter() - post_t0) * 1000.0
        total_ms = (time.perf_counter() - total_t0) * 1000.0
        meta = {
            "requested_text": response_text,
            "spoken_text": response_text,
            "path": str(out_path),
            "speaker": voice_name,
            "sample_rate": TARGET_SR,
            "source_sample_rate": int(source_rate),
            "metrics": demo.v2.v4.audio_metrics(audio, TARGET_SR),
            "source": "Windows SAPI 5 desktop voice",
            "tts_backend": "sapi",
            "sapi_synthesis_ms": synthesis_ms,
            "sapi_postprocess_ms": post_ms,
            "talker_generate_ms": total_ms,
            "response_label": response_label,
            "streaming": False,
            "enabled": True,
        }
        write_json(out_path.with_suffix(".json"), meta)
        self.set_runtime_stage("idle")
        return meta

    def _generate_talker_audio_cooperative(
        self,
        response_text: str,
        out_path: Path,
        cancel_event: threading.Event,
        response_label: str,
    ) -> dict:
        total_t0 = time.perf_counter()
        conv = [
            {"role": "system", "content": [{"type": "text", "text": demo.v2.v4.DEFAULT_SYSTEM}]},
            {"role": "user", "content": [{"type": "text", "text": demo.exact_talker_prompt(response_text)}]},
        ]
        text = self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        audios, images, videos = demo.process_mm_info(conv, use_audio_in_video=False)
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        inputs = single.move_inputs(inputs, self.model.device, self.dtype)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        speaker_params = self.model.speaker_map[self.config.speaker]

        thinker_t0 = time.perf_counter()
        self.set_runtime_stage("talker_waiting_for_thinker")
        with self.thinker_scheduler.slot(response_priority=True), torch.inference_mode():
            self.set_runtime_stage("talker_thinker_teacher_forcing")
            response_ids = self.processor.tokenizer(
                response_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].to(input_ids.device)
            if response_ids.shape[1] == 0:
                response_ids = torch.tensor(
                    [[self.processor.tokenizer.eos_token_id]],
                    dtype=torch.long,
                    device=input_ids.device,
                )
            prompt_len = input_ids.shape[1]
            teacher_input_ids = torch.cat([input_ids, response_ids], dim=1)
            if attention_mask is None:
                teacher_attention_mask = torch.ones_like(teacher_input_ids)
            else:
                teacher_attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((1, response_ids.shape[1]))],
                    dim=1,
                )
            thinker_result = self.model.thinker(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask,
                use_cache=False,
                use_audio_in_video=False,
                output_hidden_states=True,
                return_dict=True,
            )
            thinker_token_embeds = thinker_result.hidden_states[0]
            thinker_hidden_states = thinker_result.hidden_states[-1]
            thinker_embed_tokens = self.model.thinker.get_input_embeddings()
            thinker_reply_part = (
                thinker_hidden_states[:, prompt_len:, :] + thinker_token_embeds[:, prompt_len:, :]
            )
            talker_inputs_embeds = (
                thinker_hidden_states[:, :prompt_len, :] + thinker_token_embeds[:, :prompt_len, :]
            )
            text_bos = torch.tensor(
                [[speaker_params["bos_token"]]],
                dtype=torch.long,
                device=input_ids.device,
            )
            talker_inputs_embeds = torch.cat(
                [talker_inputs_embeds, thinker_embed_tokens(text_bos), thinker_reply_part[:, :1, :]],
                dim=1,
            )
            eos_embedding = thinker_embed_tokens(
                torch.tensor([[self.model.talker.text_eos_token]], dtype=torch.long, device=input_ids.device)
            )
            pad_embedding = thinker_embed_tokens(
                torch.tensor([[self.model.talker.text_pad_token]], dtype=torch.long, device=input_ids.device)
            )
            thinker_reply_part = torch.cat([thinker_reply_part[:, 1:, :], eos_embedding, pad_embedding], dim=1)
        thinker_ms = (time.perf_counter() - thinker_t0) * 1000.0
        if cancel_event.is_set():
            self.set_runtime_stage("idle")
            return {"enabled": False, "canceled": True, "path": "", "talker_thinker_ms": thinker_ms}

        talker_input_text_ids = torch.cat([input_ids, text_bos, response_ids[:, :1]], dim=-1)
        talker_input_ids = torch.cat(
            [
                torch.full_like(input_ids, fill_value=self.model.talker.codec_mask_token),
                torch.tensor([[self.model.talker.codec_pad_token]], dtype=torch.long, device=input_ids.device),
                torch.tensor([[self.model.talker.codec_bos_token]], dtype=torch.long, device=input_ids.device),
            ],
            dim=1,
        )
        talker_attention_mask = None
        if attention_mask is not None:
            talker_attention_mask = torch.cat([attention_mask, attention_mask.new_ones((1, 2))], dim=1)
        token2wav_bucket = 32 if response_label == "BACKCHANNEL" else 96
        talker_token_budget = token2wav_bucket + 1

        self.set_runtime_stage("talker_codec")
        # Compiled token2wav uses CUDA Graph capture. A concurrent Thinker forward
        # on another worker invalidates that capture, so serialize the codec and
        # waveform path with the same scheduler used by Thinker generation.
        with self.thinker_scheduler.slot(response_priority=True), self.talker_lock, torch.inference_mode():
            codec_t0 = time.perf_counter()
            talker_result = self.model.talker.generate(
                input_ids=talker_input_ids,
                input_text_ids=talker_input_text_ids,
                thinker_reply_part=thinker_reply_part,
                inputs_embeds=talker_inputs_embeds,
                attention_mask=talker_attention_mask,
                suppress_tokens=[self.model.talker.codec_bos_token],
                max_new_tokens=talker_token_budget,
                do_sample=True,
                top_k=40,
                top_p=0.8,
                temperature=0.85,
                eos_token_id=[8292, 8294],
                repetition_penalty=1.05,
                use_audio_in_video=False,
                stopping_criteria=StoppingCriteriaList([CancelOnEvent(cancel_event)]),
            )
            codec_ms = (time.perf_counter() - codec_t0) * 1000.0
            if cancel_event.is_set():
                self.set_runtime_stage("idle")
                return {
                    "enabled": False,
                    "canceled": True,
                    "path": "",
                    "talker_thinker_ms": thinker_ms,
                    "talker_codec_ms": codec_ms,
                    "talker_generate_ms": (time.perf_counter() - total_t0) * 1000.0,
                }
            raw_talker_codes = talker_result[:, talker_input_ids.shape[1] : -1]
            original_code_count = int(raw_talker_codes.shape[1])
            talker_codes = raw_talker_codes[:, :token2wav_bucket]
            if talker_codes.shape[1] == 0:
                talker_codes = torch.zeros((1, token2wav_bucket), dtype=torch.long, device=input_ids.device)
            elif talker_codes.shape[1] < token2wav_bucket:
                tail = talker_codes[:, -1:].expand(-1, token2wav_bucket - talker_codes.shape[1])
                talker_codes = torch.cat([talker_codes, tail], dim=1)
            talker_codes = talker_codes.clone(memory_format=torch.contiguous_format)
            self.set_runtime_stage("token2wav")
            wav_t0 = time.perf_counter()
            if self.model.token2wav.dtype != torch.float:
                self.model.token2wav.float()
            if self.token2wav_compiled:
                torch.compiler.cudagraph_mark_step_begin()
            wav = self.model.token2wav(
                talker_codes.to(input_ids.device).contiguous(),
                conditioning=speaker_params["cond"].to(input_ids.device).float().contiguous(),
                reference_mel=speaker_params["ref_mel"].to(input_ids.device).float().contiguous(),
                use_audio_in_video=False,
            )
            token2wav_ms = (time.perf_counter() - wav_t0) * 1000.0

        spoken_text = response_text
        audio = wav.detach().float().cpu().numpy().reshape(-1)
        valid_code_count = max(1, min(original_code_count, token2wav_bucket))
        valid_samples = max(1, int(round(audio.shape[0] * valid_code_count / token2wav_bucket)))
        audio = audio[:valid_samples]
        max_s = 2.0 if len(response_text.split()) <= 4 else 5.0
        audio = demo.v2.v4.fade_and_limit(demo.v2.v4.trim_audio(audio, TARGET_SR, threshold=0.006), TARGET_SR, max_s=max_s)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), audio, TARGET_SR)
        meta = {
            "requested_text": response_text,
            "spoken_text": spoken_text,
            "path": str(out_path),
            "speaker": self.config.speaker,
            "sample_rate": TARGET_SR,
            "metrics": demo.v2.v4.audio_metrics(audio, TARGET_SR),
            "source": "Qwen2.5-Omni-3B Talker exact-text cooperative pipeline",
            "talker_thinker_ms": thinker_ms,
            "talker_thinker_mode": "teacher_forced_exact_response",
            "teacher_forcing_tokens": int(response_ids.shape[1]),
            "talker_codec_ms": codec_ms,
            "token2wav_ms": token2wav_ms,
            "talker_token_budget": talker_token_budget,
            "original_code_count": original_code_count,
            "token2wav_bucket": token2wav_bucket,
            "response_label": response_label,
            "talker_generate_ms": (time.perf_counter() - total_t0) * 1000.0,
            "enabled": True,
        }
        write_json(out_path.with_suffix(".json"), meta)
        self.set_runtime_stage("idle")
        return meta


@dataclass
class LiveSession:
    session_id: str
    session_dir: Path
    browser_sample_rate: int = 48000
    vad_threshold: float = 0.01
    vad_min_silence_s: float = 0.0
    tick_s: float = 0.5
    max_context_seconds: float = 24.0
    store_full_audio: bool = True
    seen_voice: bool = False
    was_below: bool = False
    last_voice_time: float = 0.0
    last_tick_time: float = -999.0
    previous_tick_time: float | None = None
    utterance_start_time: float | None = None
    audio_clock: float = 0.0
    chunk_index: int = 0
    tick_index: int = 0
    event_index: int = 0
    last_label: str = "WAIT"
    inference_busy: bool = False
    response_busy: bool = False
    last_speaking: bool = False
    voice_revision: int = 0
    latest_event_index: int = -1
    last_event_voice_revision: int = -1
    closed: bool = False
    skipped_ticks: int = 0
    stream_chunks: list[np.ndarray] = field(default_factory=list)
    full_chunks: list[np.ndarray] = field(default_factory=list)
    model_chunks: list[np.ndarray] = field(default_factory=list)
    pending_silence_chunks: list[np.ndarray] = field(default_factory=list)
    model_audio_revision: int = 0
    pending_tasks: set[asyncio.Task] = field(default_factory=set)
    pending_tick: dict[str, Any] | None = None
    pending_response: dict[str, Any] | None = None
    active_response_cancel: threading.Event | None = None
    active_response_label: str | None = None
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    started_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def append_audio(self, pcm: np.ndarray):
        self.chunk_index += 1
        pcm = pcm.astype(np.float32)
        duration = len(pcm) / max(1, self.browser_sample_rate)
        self.audio_clock += duration
        self.stream_chunks.append(pcm)
        if self.store_full_audio:
            self.full_chunks.append(pcm)
        max_samples = int(self.max_context_seconds * self.browser_sample_rate)
        total = sum(len(x) for x in self.stream_chunks)
        while total > max_samples and len(self.stream_chunks) > 1:
            removed = self.stream_chunks.pop(0)
            total -= len(removed)
        rms = float(np.sqrt(np.mean(pcm * pcm))) if pcm.size else 0.0
        return duration, rms

    def current_context_24k(self) -> np.ndarray:
        if not self.model_chunks:
            return np.zeros(1, dtype=np.float32)
        audio = np.concatenate(self.model_chunks).astype(np.float32)
        return resample_audio(audio, self.browser_sample_rate, TARGET_SR)

    def update_model_audio(self, pcm: np.ndarray, speaking: bool):
        if speaking:
            if not self.seen_voice:
                self.pending_silence_chunks.clear()
            elif self.pending_silence_chunks:
                self.model_chunks.extend(self.pending_silence_chunks)
                self.pending_silence_chunks.clear()
            self.model_chunks.append(pcm.astype(np.float32))
            self.model_audio_revision += 1
            max_samples = int(self.max_context_seconds * self.browser_sample_rate)
            total = sum(len(x) for x in self.model_chunks)
            while total > max_samples and len(self.model_chunks) > 1:
                total -= len(self.model_chunks.pop(0))
        elif self.seen_voice:
            self.pending_silence_chunks.append(pcm.astype(np.float32))
            max_samples = int(self.max_context_seconds * self.browser_sample_rate)
            total = sum(len(x) for x in self.pending_silence_chunks)
            while total > max_samples and len(self.pending_silence_chunks) > 1:
                total -= len(self.pending_silence_chunks.pop(0))

    def full_audio_24k(self) -> np.ndarray:
        chunks = self.full_chunks if self.full_chunks else self.stream_chunks
        if not chunks:
            return np.zeros(1, dtype=np.float32)
        audio = np.concatenate(chunks).astype(np.float32)
        return resample_audio(audio, self.browser_sample_rate, TARGET_SR)


class AppState:
    def __init__(self, args):
        self.args = args
        tts_backend = "none" if args.no_talker else args.tts_backend
        self.engine = ModelEngine(
            EngineConfig(
                speaker=args.speaker,
                enable_talker=tts_backend == "omni",
                tts_backend=tts_backend,
                sapi_voice=args.sapi_voice,
                sapi_rate=args.sapi_rate,
                max_context_seconds=args.max_context_seconds,
                save_segments=True,
                attention_backend=args.attention_backend,
                compile_token2wav=not args.no_compile_token2wav,
                warm_token2wav=not args.no_warm_token2wav,
            )
        )
        self.load_task: asyncio.Task | None = None
        self.active_sessions: set[str] = set()


def make_app(args) -> FastAPI:
    app = FastAPI(title="Omni3B Realtime Time Adapter WebUI")
    state = AppState(args)
    app.state.rt = state
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/")
    async def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/status")
    async def status():
        return JSONResponse(state.engine.state())

    @app.post("/api/load")
    async def load_model():
        if state.engine.status == "loaded":
            return JSONResponse(state.engine.state())
        if state.load_task is None or state.load_task.done():
            loop = asyncio.get_running_loop()
            state.load_task = loop.run_in_executor(None, state.engine.load)
        return JSONResponse(state.engine.state())

    @app.post("/api/adapter/{adapter_key}")
    async def switch_adapter(adapter_key: str):
        if state.engine.status != "loaded":
            raise HTTPException(status_code=409, detail="model must be loaded before switching adapters")
        if state.active_sessions:
            raise HTTPException(
                status_code=409,
                detail="stop the active microphone session before switching adapters",
            )
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, state.engine.switch_adapter, adapter_key)
            return JSONResponse(result)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/sessions")
    async def sessions():
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        items = []
        for p in sorted(OUT_ROOT.glob("session_*"), reverse=True)[:30]:
            items.append({"name": p.name, "path": str(p), "summary": str(p / "summary.json")})
        return JSONResponse({"sessions": items})

    @app.get("/api/audio/{session_id}/{filename}")
    async def session_audio(session_id: str, filename: str):
        if not session_id.startswith("session_") or "/" in filename or "\\" in filename or not filename.endswith(".wav"):
            raise HTTPException(status_code=400, detail="invalid audio path")
        response_dir = (OUT_ROOT / session_id / "responses").resolve()
        target = (response_dir / filename).resolve()
        if response_dir not in target.parents and target.parent != response_dir:
            raise HTTPException(status_code=400, detail="invalid audio path")
        if not target.exists():
            raise HTTPException(status_code=404, detail="audio not found")
        return FileResponse(target, media_type="audio/wav", filename=filename)

    async def safe_send_json(ws: WebSocket, session: LiveSession, payload: dict) -> bool:
        if session.closed:
            return False
        try:
            async with session.send_lock:
                await ws.send_json(payload)
            return True
        except Exception as exc:
            session.closed = True
            append_jsonl(session.session_dir / "errors.jsonl", {"time": time.time(), "send_error": repr(exc), "payload_type": payload.get("type")})
            return False

    def track_task(session: LiveSession, task: asyncio.Task):
        session.pending_tasks.add(task)

        def _done(done: asyncio.Task):
            session.pending_tasks.discard(done)
            try:
                exc = done.exception()
            except asyncio.CancelledError:
                return
            if exc is not None:
                append_jsonl(session.session_dir / "errors.jsonl", {"time": time.time(), "task_error": repr(exc)})

        task.add_done_callback(_done)

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        session = None
        try:
            while True:
                msg = await ws.receive()
                if "text" in msg and msg["text"] is not None:
                    data = json.loads(msg["text"])
                    kind = data.get("type")
                    if kind == "start":
                        if state.engine.status != "loaded":
                            await ws.send_json({"type": "error", "message": "model is not loaded"})
                            continue
                        session_id = f"session_{now_id()}_{uuid.uuid4().hex[:8]}"
                        session_dir = OUT_ROOT / session_id
                        session_dir.mkdir(parents=True, exist_ok=True)
                        session = LiveSession(
                            session_id=session_id,
                            session_dir=session_dir,
                            browser_sample_rate=int(data.get("sampleRate", 48000)),
                            vad_threshold=float(data.get("vadThreshold", args.vad_threshold)),
                            vad_min_silence_s=float(data.get("vadMinSilenceSeconds", 0.0)),
                            tick_s=float(data.get("tickSeconds", args.tick)),
                            max_context_seconds=float(data.get("maxContextSeconds", args.max_context_seconds)),
                        )
                        state.active_sessions.add(session_id)
                        config = {
                            "session_id": session_id,
                            "started_at": session.started_at,
                            "browser_sample_rate": session.browser_sample_rate,
                            "target_sample_rate": TARGET_SR,
                            "vad_threshold": session.vad_threshold,
                            "vad_min_silence_s": session.vad_min_silence_s,
                            "tick_s": session.tick_s,
                            "max_context_seconds": session.max_context_seconds,
                            "model": state.engine.state(),
                            "notes": [
                                "Browser sends PCM16 microphone chunks over WebSocket.",
                                "Server computes simple RMS VAD and emits inference ticks while below threshold.",
                                "Prompt receives audio only; no transcript or explicit seconds text is sent to Omni.",
                            ],
                        }
                        write_json(session_dir / "config.json", config)
                        await ws.send_json({"type": "session_started", **config})
                    elif kind == "stop":
                        if session is not None:
                            await finalize_session(session)
                            await safe_send_json(ws, session, {"type": "session_stopped", "session_id": session.session_id, "path": str(session.session_dir)})
                            session.closed = True
                            session = None
                    elif kind == "ping":
                        await ws.send_json({"type": "pong", "time": time.time()})
                    elif kind == "set":
                        if session:
                            if "vadThreshold" in data:
                                session.vad_threshold = float(data["vadThreshold"])
                            if "vadMinSilenceSeconds" in data:
                                session.vad_min_silence_s = float(data["vadMinSilenceSeconds"])
                            if "tickSeconds" in data:
                                session.tick_s = float(data["tickSeconds"])
                            if "maxContextSeconds" in data:
                                session.max_context_seconds = float(data["maxContextSeconds"])
                            await ws.send_json({
                                "type": "settings",
                                "vadThreshold": session.vad_threshold,
                                "vadMinSilenceSeconds": session.vad_min_silence_s,
                                "tickSeconds": session.tick_s,
                                "maxContextSeconds": session.max_context_seconds,
                            })
                    continue
                if "bytes" in msg and msg["bytes"] is not None:
                    if session is None:
                        continue
                    pcm_i16 = np.frombuffer(msg["bytes"], dtype="<i2")
                    pcm = (pcm_i16.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
                    await handle_audio_chunk(ws, state.engine, session, pcm)
        except WebSocketDisconnect:
            if session is not None:
                session.closed = True
                await finalize_session(session)
        except Exception as exc:
            if session is not None:
                session.closed = True
                state.active_sessions.discard(session.session_id)
                append_jsonl(session.session_dir / "errors.jsonl", {"time": time.time(), "error": repr(exc)})
            try:
                await ws.send_json({"type": "error", "message": repr(exc)})
            except Exception:
                pass

    async def finalize_session(session: LiveSession):
        state.active_sessions.discard(session.session_id)
        audio = session.full_audio_24k()
        sf.write(str(session.session_dir / "input_audio_24k.wav"), audio, TARGET_SR)
        summary = {
            "session_id": session.session_id,
            "started_at": session.started_at,
            "ended_at": datetime.now().isoformat(timespec="seconds"),
            "duration_s": session.audio_clock,
            "chunks": session.chunk_index,
            "ticks": session.tick_index,
            "events": session.event_index,
            "skipped_ticks": session.skipped_ticks,
            "input_audio_24k": str(session.session_dir / "input_audio_24k.wav"),
        }
        write_json(session.session_dir / "summary.json", summary)

    async def handle_audio_chunk(ws: WebSocket, engine: ModelEngine, session: LiveSession, pcm: np.ndarray):
        duration, rms = session.append_audio(pcm)
        now_s = session.audio_clock
        speaking = bool(rms >= session.vad_threshold)
        session.update_model_audio(pcm, speaking)
        chunk_row = {
            "type": "chunk",
            "chunk_index": session.chunk_index,
            "clock_s": now_s,
            "duration_s": duration,
            "rms": rms,
            "vad_threshold": session.vad_threshold,
            "speaking": speaking,
            "seen_voice": session.seen_voice,
            "last_voice_time_s": session.last_voice_time,
        }
        append_jsonl(session.session_dir / "chunks.jsonl", chunk_row)
        await safe_send_json(ws, session, {
            "type": "vad",
            "clock_s": now_s,
            "rms": rms,
            "speaking": speaking,
            "silence_elapsed": max(0.0, now_s - session.last_voice_time) if session.seen_voice else 0.0,
            "threshold": session.vad_threshold,
            "chunk_index": session.chunk_index,
        })
        if speaking:
            if not session.last_speaking:
                session.voice_revision += 1
                if session.active_response_cancel is not None and session.active_response_label != "BACKCHANNEL":
                    session.active_response_cancel.set()
            session.last_speaking = True
            session.seen_voice = True
            session.last_voice_time = now_s
            if session.utterance_start_time is None:
                session.utterance_start_time = max(0.0, now_s - duration)
            session.was_below = False
            return
        session.last_speaking = False
        if not session.seen_voice:
            return
        silence_elapsed = max(0.0, now_s - session.last_voice_time)
        if silence_elapsed < session.vad_min_silence_s:
            session.was_below = True
            return
        should_tick = (not session.was_below) or (now_s - session.last_tick_time >= session.tick_s - 1e-9)
        session.was_below = True
        if not should_tick:
            return
        session.last_tick_time = now_s
        delta_t = 0.0 if session.previous_tick_time is None else now_s - session.previous_tick_time
        asr_changed = session.previous_tick_time is None or session.last_voice_time > session.previous_tick_time
        session.previous_tick_time = now_s
        utterance_elapsed = now_s if session.utterance_start_time is None else now_s - session.utterance_start_time
        request = {
            "clock_s": now_s,
            "silence_elapsed": silence_elapsed,
            "delta_t": delta_t,
            "utterance_elapsed": utterance_elapsed,
            "asr_changed": asr_changed,
            "rms": rms,
            "context_audio": session.current_context_24k().copy(),
            "audio_revision": session.model_audio_revision,
            "voice_revision": session.voice_revision,
        }
        if session.inference_busy:
            session.pending_tick = request
            session.skipped_ticks += 1
            skip = {
                "type": "skip",
                "clock_s": now_s,
                "silence_elapsed": silence_elapsed,
                "reason": "coalesced_latest_tick",
                "skipped_ticks": session.skipped_ticks,
            }
            append_jsonl(session.session_dir / "ticks.jsonl", skip)
            await safe_send_json(ws, session, skip)
            return
        start_inference_tick(ws, engine, session, request)

    def start_inference_tick(ws: WebSocket, engine: ModelEngine, session: LiveSession, request: dict):
        session.inference_busy = True
        task = asyncio.create_task(
            run_inference_tick(
                ws,
                engine,
                session,
                request["clock_s"],
                request["silence_elapsed"],
                request["delta_t"],
                request["utterance_elapsed"],
                request["asr_changed"],
                request["rms"],
                request["context_audio"],
                request["audio_revision"],
                request["voice_revision"],
            )
        )
        track_task(session, task)

    async def run_inference_tick(
        ws: WebSocket,
        engine: ModelEngine,
        session: LiveSession,
        clock_s: float,
        silence_elapsed: float,
        delta_t: float,
        utterance_elapsed: float,
        asr_changed: bool,
        rms: float,
        context_audio: np.ndarray,
        audio_revision: int,
        voice_revision: int,
    ):
        tick_id = session.tick_index
        session.tick_index += 1
        segment_dir = session.session_dir / "segments"
        segment_dir.mkdir(parents=True, exist_ok=True)
        audio_path = segment_dir / f"tick_{tick_id:05d}_{int(clock_s * 1000):08d}ms.wav"
        try:
            sf.write(str(audio_path), context_audio, TARGET_SR)
            row = {
                "id": f"{session.session_id}_tick_{tick_id:05d}",
                "context_id": session.session_id,
                "audio_path": str(audio_path),
                "segment_audio_path": str(audio_path),
                "clock_s": float(clock_s),
                "timepoint_s": float(silence_elapsed),
                "silence_seconds": float(silence_elapsed),
                "features": {
                    "silence_elapsed": float(silence_elapsed),
                    "delta_t": float(delta_t),
                    "utterance_elapsed": float(utterance_elapsed),
                    "is_user_speaking": False,
                    "asr_changed": bool(asr_changed),
                },
                "profile": "live",
                "fragment": "[live microphone audio-only; transcript withheld from Omni]",
                "label": "",
                "vad_rms": float(rms),
                "vad_threshold": float(session.vad_threshold),
                "vad_trigger": "rms_below_threshold",
                "last_voice_time_s": float(session.last_voice_time),
                "context_audio_snapshot_samples": int(context_audio.shape[0]),
                "audio_revision": int(audio_revision),
            }
            loop = asyncio.get_running_loop()
            t0 = time.perf_counter()
            result = await loop.run_in_executor(None, engine.label_forward, row)
            wall_ms = (time.perf_counter() - t0) * 1000.0
            label = result["label"]
            prev_label = session.last_label
            event = label != "WAIT" and (
                label != prev_label or voice_revision != session.last_event_voice_revision
            )
            session.last_label = label

            public = {
                "type": "tick",
                "session_id": session.session_id,
                "tick_index": tick_id,
                "clock_s": clock_s,
                "silence_elapsed": silence_elapsed,
                "delta_t": delta_t,
                "utterance_elapsed": utterance_elapsed,
                "asr_changed": asr_changed,
                "rms": rms,
                "label": label,
                "control_code": result["control_code"],
                "adapter_key": result["adapter_key"],
                "adapter_name": result["adapter_name"],
                "language": result["language"],
                "previous_label": prev_label,
                "event": event,
                "response_busy": session.response_busy,
                "p_WAIT": result["p_WAIT"],
                "p_BACKCHANNEL": result["p_BACKCHANNEL"],
                "p_SUPPORT": result["p_SUPPORT"],
                "label_probabilities": result["label_probabilities"],
                "label_logits": result["label_logits"],
                "latency_ms": {
                    "prep": result["prep_ms"],
                    "forward": result["forward_ms"],
                    "label_total": result["label_total_ms"],
                    "server_wall": wall_ms,
                    "text_generate": 0.0,
                    "talker_generate": 0.0,
                },
                "generated_label": "",
                "generated_response": "",
                "generated_raw": "",
                "hook_calls": result["hook_calls"],
                "decision_model_mode": result["decision_model_mode"],
                "input_cache_hit": result["input_cache_hit"],
                "time_vector_norm": result["time_vector_norm"],
                "context_norm": result["context_norm"],
                "time_context_cosine": result["time_context_cosine"],
                "audio_path": str(audio_path),
                "context_audio_snapshot_samples": int(context_audio.shape[0]),
            }
            append_jsonl(session.session_dir / "ticks.jsonl", public)
            append_csv(
                session.session_dir / "ticks.csv",
                public,
                [
                    "type",
                    "session_id",
                    "tick_index",
                    "clock_s",
                    "silence_elapsed",
                    "delta_t",
                    "utterance_elapsed",
                    "asr_changed",
                    "rms",
                    "label",
                    "control_code",
                    "adapter_key",
                    "adapter_name",
                    "language",
                    "previous_label",
                    "event",
                    "response_busy",
                    "p_WAIT",
                    "p_BACKCHANNEL",
                    "p_SUPPORT",
                    "label_probabilities",
                    "label_logits",
                    "generated_label",
                    "generated_response",
                    "hook_calls",
                    "time_vector_norm",
                    "context_norm",
                    "time_context_cosine",
                    "audio_path",
                    "context_audio_snapshot_samples",
                ],
            )
            await safe_send_json(ws, session, public)
            if event:
                event_index = session.event_index
                session.event_index += 1
                session.latest_event_index = event_index
                session.last_event_voice_revision = voice_revision
                event_payload = {
                    "type": "event_detected",
                    "session_id": session.session_id,
                    "event_index": event_index,
                    "tick_index": tick_id,
                    "clock_s": clock_s,
                    "silence_elapsed": silence_elapsed,
                    "label": label,
                    "control_code": result["control_code"],
                    "adapter_key": result["adapter_key"],
                    "adapter_name": result["adapter_name"],
                    "previous_label": prev_label,
                    "p_WAIT": result["p_WAIT"],
                    "p_BACKCHANNEL": result["p_BACKCHANNEL"],
                    "p_SUPPORT": result["p_SUPPORT"],
                    "label_probabilities": result["label_probabilities"],
                }
                append_jsonl(session.session_dir / "events.jsonl", event_payload)
                await safe_send_json(ws, session, event_payload)
                generation_request = {
                    "event_index": event_index,
                    "tick_index": tick_id,
                    "clock_s": clock_s,
                    "silence_elapsed": silence_elapsed,
                    "label": label,
                    "moved": result["moved"],
                    "prompt_len": result["prompt_len"],
                    "vector": result["vector"],
                    "voice_revision": voice_revision,
                }
                if session.response_busy:
                    if session.active_response_cancel is not None:
                        session.active_response_cancel.set()
                    session.pending_response = generation_request
                    queued = {
                        **event_payload,
                        "type": "generation_status",
                        "stage": "queued_latest_response",
                        "message": "newer LLM event replaces any older queued response",
                    }
                    append_jsonl(session.session_dir / "events.jsonl", queued)
                    await safe_send_json(ws, session, queued)
                else:
                    start_response_generation(ws, engine, session, generation_request)
        except Exception as exc:
            append_jsonl(session.session_dir / "errors.jsonl", {"time": time.time(), "tick_index": tick_id, "error": repr(exc)})
            await safe_send_json(ws, session, {"type": "error", "message": repr(exc), "tick_index": tick_id})
        finally:
            session.inference_busy = False
            if session.pending_tick is not None and not session.closed:
                pending = session.pending_tick
                session.pending_tick = None
                start_inference_tick(ws, engine, session, pending)

    def start_response_generation(ws: WebSocket, engine: ModelEngine, session: LiveSession, request: dict):
        cancel_event = threading.Event()
        request["cancel_event"] = cancel_event
        session.active_response_cancel = cancel_event
        session.active_response_label = request["label"]
        session.response_busy = True
        task = asyncio.create_task(run_event_generation(ws, engine, session, request))
        track_task(session, task)

    async def run_event_generation(
        ws: WebSocket,
        engine: ModelEngine,
        session: LiveSession,
        request: dict,
    ):
        event_index = request["event_index"]
        tick_index = request["tick_index"]
        clock_s = request["clock_s"]
        silence_elapsed = request["silence_elapsed"]
        label = request["label"]

        def event_row(payload: dict) -> dict:
            return {
                "session_id": session.session_id,
                "event_index": event_index,
                "tick_index": tick_index,
                "clock_s": clock_s,
                "silence_elapsed": silence_elapsed,
                "label": label,
                **payload,
            }

        def is_current() -> bool:
            timing_is_current = label == "BACKCHANNEL" or (
                not session.last_speaking and request["voice_revision"] == session.voice_revision
            )
            return (
                not session.closed
                and timing_is_current
                and event_index == session.latest_event_index
            )

        async def report_stale(stage: str):
            stale = event_row(
                {
                    "type": "generation_status",
                    "stage": stage,
                    "message": "discarded because newer live audio or a newer LLM decision exists",
                }
            )
            append_jsonl(session.session_dir / "events.jsonl", stale)
            await safe_send_json(ws, session, stale)

        try:
            if not is_current():
                await report_stale("stale_before_text")
                return
            loop = asyncio.get_running_loop()
            text_start = event_row({"type": "generation_status", "stage": "generating_text"})
            append_jsonl(session.session_dir / "events.jsonl", text_start)
            await safe_send_json(ws, session, text_start)
            text_result = await loop.run_in_executor(
                None,
                engine.generate_text,
                request["moved"],
                request["prompt_len"],
                request["vector"],
            )
            response_text = text_result.get("generated_response", "").strip()
            response_label = text_result.get("generated_label") or label
            text_payload = event_row(
                {
                    "type": "generation_status",
                    "stage": "text_ready" if response_text else "no_response_text",
                    "generated_label": text_result.get("generated_label", ""),
                    "generated_response": response_text,
                    "generated_raw": text_result.get("generated_raw", ""),
                    "latency_ms": {"text_generate": text_result.get("text_generate_ms", 0.0)},
                }
            )
            append_jsonl(session.session_dir / "events.jsonl", text_payload)
            if not is_current():
                await report_stale("stale_after_text")
                return
            await safe_send_json(ws, session, text_payload)
            if not response_text:
                return
            if not is_current():
                await report_stale("stale_before_talker")
                return

            response_path = session.session_dir / "responses" / f"{event_index:03d}_{int(clock_s * 1000):08d}ms_{label.lower()}.wav"
            talker_start = event_row({"type": "generation_status", "stage": "generating_talker", "response_text": response_text})
            append_jsonl(session.session_dir / "events.jsonl", talker_start)
            await safe_send_json(ws, session, talker_start)
            talker_meta = await loop.run_in_executor(
                engine.response_executor,
                engine.generate_talker_audio,
                response_text,
                response_path,
                request["cancel_event"],
                response_label,
            )
            audio_path = Path(talker_meta.get("path") or "")
            if not is_current():
                await report_stale("stale_after_talker")
                return
            if talker_meta.get("enabled") and audio_path.exists():
                audio_payload = event_row(
                    {
                        "type": "assistant_audio_ready",
                        "stage": "audio_ready",
                        "response_text": response_text,
                        "audio_url": f"/api/audio/{session.session_id}/{audio_path.name}",
                        "mime": "audio/wav",
                        "talker_meta": talker_meta,
                        "latency_ms": {"talker_generate": talker_meta.get("talker_generate_ms", 0.0)},
                    }
                )
                append_jsonl(session.session_dir / "events.jsonl", audio_payload)
                await safe_send_json(ws, session, audio_payload)
            else:
                no_audio = event_row(
                    {
                        "type": "generation_status",
                        "stage": "talker_disabled_or_failed",
                        "response_text": response_text,
                        "talker_meta": talker_meta,
                    }
                )
                append_jsonl(session.session_dir / "events.jsonl", no_audio)
                await safe_send_json(ws, session, no_audio)
        except Exception as exc:
            engine.set_runtime_stage("idle")
            err = event_row({"type": "generation_status", "stage": "generation_error", "message": repr(exc)})
            append_jsonl(
                session.session_dir / "errors.jsonl",
                {"time": time.time(), **err, "traceback": traceback.format_exc()},
            )
            await safe_send_json(ws, session, err)
        finally:
            if session.active_response_cancel is request.get("cancel_event"):
                session.active_response_cancel = None
                session.active_response_label = None
            session.response_busy = False
            if session.pending_response is not None and not session.closed:
                pending = session.pending_response
                session.pending_response = None
                start_response_generation(ws, engine, session, pending)

    return app


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7865)
    parser.add_argument("--speaker", choices=["Chelsie", "Ethan"], default="Chelsie")
    parser.add_argument("--tts-backend", choices=["omni", "sapi", "none"], default="omni")
    parser.add_argument("--sapi-voice", default="auto", help="SAPI voice description substring; auto selects the first installed voice.")
    parser.add_argument("--sapi-rate", type=int, default=2, help="Windows SAPI speech rate from -10 to 10.")
    parser.add_argument("--no-talker", action="store_true", help="Deprecated alias for --tts-backend none.")
    parser.add_argument("--autoload", action="store_true", help="Load model during server startup.")
    parser.add_argument("--vad-threshold", type=float, default=0.01)
    parser.add_argument("--tick", type=float, default=0.5)
    parser.add_argument("--max-context-seconds", type=float, default=24.0)
    parser.add_argument("--attention-backend", choices=["eager", "sdpa"], default="sdpa")
    parser.add_argument("--no-compile-token2wav", action="store_true")
    parser.add_argument("--no-warm-token2wav", action="store_true")
    parser.add_argument("--ws-ping-interval", type=float, default=30.0)
    parser.add_argument("--ws-ping-timeout", type=float, default=120.0)
    return parser.parse_args()


def main():
    args = parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    app = make_app(args)
    if args.autoload:
        def load_bg():
            app.state.rt.engine.load()

        threading.Thread(target=load_bg, daemon=True).start()
    print(f"WebUI: http://{args.host}:{args.port}", flush=True)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        ws_ping_interval=args.ws_ping_interval,
        ws_ping_timeout=args.ws_ping_timeout,
    )


if __name__ == "__main__":
    main()
