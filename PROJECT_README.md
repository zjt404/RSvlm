# Qwen3-VL-4B Remote-Sensing Assistant

An end-to-end portfolio project for remote-sensing VLM adaptation:

```text
VRSBench + DIOR
       -> validated multi-task JSONL
       -> Qwen3-VL-4B LoRA SFT
       -> task-conditioned GRPO
       -> reproducible evaluation
       -> FastAPI + Gradio demo
```

The model files may live in the project root on the target server. This file is intentionally named
`PROJECT_README.md` so the upstream model card in `README.md` remains untouched.

## What is implemented

- VRSBench LLaVA/direct-JSON conversion for VQA and visual grounding.
- Explicit conversion of VRSBench boxes from `0-100` to the project-wide `0-1000` coordinate space.
- DIOR Pascal-VOC conversion for class counting and unambiguous eight-direction spatial QA.
- Image-level deterministic train/validation/test splits and cross-split leakage checks.
- Quota sampling for 40k SFT records and 8k GRPO prompts, including 20% zero-count examples.
- Qwen3-VL LoRA SFT and task-conditioned GRPO launch scripts for ms-swift.
- Dense count/spatial reward, strict JSON-format reward, and a count-only GRPO fallback dataset.
- VQA, counting, grounding, spatial, JSON-validity, and shuffled-image evaluation.
- Lazy-loading FastAPI inference service and Gradio client.

## Server setup

The expected server layout is:

```text
/workspace/zjt/qwen3vl/
  model-*.safetensors
  config.json
  qwenvl/
  src/
  scripts/
  data/
  outputs/
```

Install the CUDA 12.1 PyTorch wheel and project dependencies:

```bash
cd /workspace/zjt/qwen3vl
bash scripts/bootstrap_server.sh
```

The colocated vLLM path is optional. The current machine uses NVIDIA driver 535, so verify the CUDA
requirement of the selected vLLM wheel before running:

```bash
bash scripts/install_vllm_optional.sh
```

The correctness smoke tests do not require vLLM.

## Data preparation

Download and extract VRSBench:

```bash
bash scripts/download_vrsbench.sh
```

Place DIOR in `data/raw/DIOR` with Pascal VOC layout:

```text
DIOR/
  Annotations/*.xml
  JPEGImages/*.{jpg,png}
```

DIOR distribution terms require obtaining the dataset from its official source. After both datasets
are available:

```bash
bash scripts/prepare_data.sh
```

Outputs:

```text
data/processed/sft_train.jsonl
data/processed/sft_val.jsonl
data/processed/sft_test.jsonl
data/processed/grpo_train.jsonl
data/processed/grpo_count_train.jsonl
data/processed/dataset_summary.json
```

Every SFT sample follows this contract:

```json
{
  "messages": [
    {"role": "user", "content": "<image> How many airplanes are visible? Return JSON only."},
    {"role": "assistant", "content": "{\"answer\":12}"}
  ],
  "images": ["/absolute/path/image.png"],
  "task_type": "count",
  "solution": {"value": 12},
  "meta": {"source": "DIOR", "image_id": "000123", "split": "train"}
}
```

GRPO files intentionally omit the assistant turn so the reference answer cannot leak into the prompt.

## Training

Generate local smoke fixtures and run one SFT step:

```bash
python scripts/make_smoke_data.py
SMOKE=1 bash scripts/train_sft.sh
```

Run full SFT:

```bash
bash scripts/train_sft.sh
```

Select the best SFT checkpoint, then run one GRPO correctness step without vLLM:

```bash
SFT_ADAPTER=/workspace/zjt/qwen3vl/outputs/sft/checkpoint-N \
SMOKE=1 USE_VLLM=0 \
bash scripts/train_grpo.sh
```

Run full mixed-task GRPO after the vLLM compatibility check:

```bash
SFT_ADAPTER=/workspace/zjt/qwen3vl/outputs/sft/checkpoint-N \
USE_VLLM=1 \
bash scripts/train_grpo.sh
```

If the mixed run fails the acceptance gate, use the predefined count-only fallback:

```bash
SFT_ADAPTER=/workspace/zjt/qwen3vl/outputs/sft/checkpoint-N \
TRAIN_DATA=/workspace/zjt/qwen3vl/data/processed/grpo_count_train.jsonl \
OUTPUT_DIR=/workspace/zjt/qwen3vl/outputs/grpo_count \
bash scripts/train_grpo.sh
```

## Evaluation

The prediction JSONL contains `task_type`, `solution`, `prediction`, and source metadata. Evaluate one
file directly:

```bash
rs-vlm-eval outputs/evaluation/base.jsonl --output outputs/evaluation/base_metrics.json
```

Run Base, SFT, GRPO, and the GRPO shuffled-image control:

```bash
SFT_ADAPTER=/path/to/sft/checkpoint \
GRPO_ADAPTER=/path/to/grpo/checkpoint \
bash scripts/evaluate_models.sh
```

The reported metrics are:

- VQA normalized exact-match accuracy.
- Counting MAE, RMSE, exact accuracy, parse failure, and zero-target false-positive rate.
- Grounding mean IoU and Acc@IoU 0.5.
- Spatial accuracy and macro-F1.
- Global JSON-valid rate.

The shuffled-image run uses the same questions and solutions with permuted images. Its score drop is
the visual-dependence control.

## Demo

Start API and UI together:

```bash
MODEL_PATH=/workspace/zjt/qwen3vl \
ADAPTER_PATH=/path/to/accepted/adapter \
bash scripts/serve.sh
```

- API health: `GET http://SERVER:8000/health`
- API inference: `POST http://SERVER:8000/analyze`
- Gradio: `http://SERVER:7860`

## Acceptance gate

Publish the GRPO adapter in the final demo only when:

1. At least one reward-aligned metric improves over SFT.
2. JSON-valid rate improves.
3. VQA accuracy falls by no more than three percentage points.

Otherwise report the mixed-run result and use the count-only fallback. Do not replace measured values
with illustrative numbers.

## Dataset restrictions

VRSBench combines DIOR and DOTA-derived imagery. Its annotation and image terms are not identical,
and DOTA-derived images are restricted to academic/non-commercial use. This repository does not
redistribute datasets or trained weights. Review the official dataset terms before publishing a demo,
adapter, or Docker image.

