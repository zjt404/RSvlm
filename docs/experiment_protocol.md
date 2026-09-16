# Experiment protocol

## Fixed comparisons

Evaluate all three checkpoints against the same immutable `sft_test.jsonl`:

1. Base Qwen3-VL-4B-Instruct.
2. Base plus selected SFT adapter.
3. Base plus accepted GRPO adapter.

Record the model path, adapter path, git commit, package versions, random seed, sample count, image
pixel cap, and generation arguments next to every result file.

## Data integrity

- Split by `(source, image_id)` before generating or sampling questions.
- Never tune on VRSBench evaluation JSON files.
- Never compare models using different prompt templates or decoding settings.
- Keep zero-count questions in every DIOR split.
- Treat unparsable count output as a parse failure, not as zero.

## GRPO gate

Compare GRPO against SFT, not only against Base. Accept mixed GRPO only if a count or spatial metric
improves, JSON validity improves, and VQA accuracy regresses by no more than 0.03 absolute. If the
gate fails, train the predefined count-only dataset without modifying reward weights after seeing test
results.

## Visual-dependence control

Run the accepted adapter twice: once with correct images and once with images deterministically
shuffled using seed 42. Report per-task score drops. A negligible drop is evidence of language or
dataset shortcuts and must be discussed in the README.

