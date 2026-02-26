# Qwen3-TTS (Optimized Fork)

> Forked from [Qwen/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — optimized for CPU inference.

This fork contains **CPU performance optimizations** for the Qwen3-TTS speech synthesis model, used by [Vocably](https://github.com/AbdulGani11/Vocably).

## What's Changed

### Lean CodePredictor Loop (`_fast_code_predict`)

The original code calls HuggingFace's `GenerationMixin.generate()` inside every Talker step to produce 31 codec tokens. This creates massive Python overhead (StoppingCriteria, LogitsProcessors, GenerationConfig, etc.) for **6,200+ iterations per paragraph**.

Replaced with a lean direct forward + sampling loop that does the same math without the overhead.

**File:** `qwen_tts/core/models/modeling_qwen3_tts.py`

### Removed Unused Components

- `qwen_tts/core/tokenizer_25hz/` — 25Hz tokenizer (not used by 12Hz models)
- `examples/test_model_12hz_base.py` — Base model tests
- `examples/test_model_12hz_voice_design.py` — VoiceDesign tests
- `finetuning/` — Finetuning scripts

## Model Used

This fork is designed for **Qwen3-TTS-12Hz-1.7B-CustomVoice** — the model weights are downloaded automatically from [HuggingFace](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) on first run.

## Installation

```bash
pip install -e .
```

## Usage

```python
from qwen_tts import Qwen3TTSModel
import torch

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cpu",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

wav = model.generate_custom_voice(
    text="Hello world",
    speaker="Chelsie",
    instruct_text="Speak in a warm, friendly tone.",
)
```

## Original Project

- 📑 [Paper](https://arxiv.org/abs/2601.15621)
- 🤗 [HuggingFace Collection](https://huggingface.co/collections/Qwen/qwen3-tts)
- 🔗 [Original Repo](https://github.com/QwenLM/Qwen3-TTS)

## License

Apache 2.0 (same as original)
