---
name: vectorize-triton-ascend-comparisons
description: Vectorize explicit integer comparisons in Triton-Ascend by casting bounded index vectors to float32 before tl.where or similar compute expressions. Use when an agent observes scalarized int32 or int64 comparison code on Ascend NPU, especially in LayerNorm tail handling, while load/store masks already compile efficiently.
---

# Vectorize Triton-Ascend Comparisons

## Goal

Move eligible explicit comparisons onto Ascend vector cast and compare instructions without changing mask semantics.

## Workflow

1. Identify comparisons used in compute expressions such as tl.where, not only load/store masks.
2. Prove that all compared integer values are exactly representable in float32.
3. Cast the index vector to tl.float32 immediately before the comparison.
4. Keep original integer offsets for pointer arithmetic and memory masks.
5. Inspect performance and compare results around the tail boundary.

## Implementation Pattern

~~~python
cols = tl.arange(0, BLOCK_N)
mask = cols < N
x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)

cols_cmp = cols.to(tl.float32)
centered = tl.where(cols_cmp < N, x - mean, 0.0)
variance = tl.sum(centered * centered, axis=0) / N

tl.store(Out + cols, (x - mean) / tl.sqrt(variance + eps), mask=mask)
~~~

## Ascend Guardrails

- Keep load and store masks in their natural integer form; the compiler commonly vectorizes them already.
- Cast only the comparison operands, never pointers or offsets.
- Float32 represents all integers exactly only through 2^24. Do not use this transformation when indices or bounds can exceed that range.
- Compare x values, not the X pointer, inside tl.where.
- Measure the generated kernel because compiler versions may optimize the integer form differently.

## Verification

Test N immediately below, equal to, and above BLOCK_N boundaries. Compare the transformed and original kernels across valid ranges and add a guard test for N greater than 2^24.
