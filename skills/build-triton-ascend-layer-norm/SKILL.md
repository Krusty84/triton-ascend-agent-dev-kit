---
name: build-triton-ascend-layer-norm
description: Build a fused forward LayerNorm kernel for Triton-Ascend with row-wise mean and variance reductions, float32 accumulation, affine weight and bias, and masked feature tiles. Use when an agent needs LayerNorm over the last dimension on Ascend NPU or needs a reusable two-pass reduction pattern for normalization kernels.
---

# Build Triton-Ascend LayerNorm

## Goal

Implement the forward equation y = ((x - mean) / sqrt(var + eps)) * weight + bias over the final tensor dimension.

## Workflow

1. Reshape the input logically to M by N, where N is the normalized dimension.
2. Allocate output with the original shape and allocate optional mean and reciprocal-standard-deviation buffers as float32 vectors of length M.
3. Launch one program per row.
4. Accumulate the mean in float32 across masked BLOCK_SIZE tiles.
5. Make a second pass for variance, compute rstd, then make a third pass to normalize and apply weight and bias.

## Implementation Pattern

~~~python
row = tl.program_id(0)
x_row = X + row * stride
y_row = Y + row * stride

mean_acc = tl.zeros([BLOCK_SIZE], tl.float32)
for start in range(0, N, BLOCK_SIZE):
    cols = start + tl.arange(0, BLOCK_SIZE)
    mean_acc += tl.load(x_row + cols, mask=cols < N, other=0.0).to(tl.float32)
mean = tl.sum(mean_acc, axis=0) / N

var_acc = tl.zeros([BLOCK_SIZE], tl.float32)
for start in range(0, N, BLOCK_SIZE):
    cols = start + tl.arange(0, BLOCK_SIZE)
    value = tl.load(x_row + cols, mask=cols < N, other=0.0).to(tl.float32)
    centered = tl.where(cols < N, value - mean, 0.0)
    var_acc += centered * centered
rstd = 1.0 / tl.sqrt(tl.sum(var_acc, axis=0) / N + eps)
~~~

In the final tiled pass, load weight and bias with the same mask, compute (value - mean) * rstd * weight + bias, and store to y_row.

## Ascend Guardrails

- Accumulate mean and variance in tl.float32 even for float16 or bfloat16 inputs.
- Require weight and bias to be one-dimensional, contiguous, and length N.
- Mask weight, bias, input, and output accesses in every partial tile.
- Keep BLOCK_SIZE fixed initially, such as 1024, and loop when N is larger.
- State clearly that this skill implements forward inference only; do not imply backward support because the wrapper subclasses torch.autograd.Function.
- Preserve the input layout explicitly or require the flattened rows to be contiguous.

## Verification

Compare with torch.nn.functional.layer_norm using the same eps, weight, and bias. Cover float16, bfloat16, and float32, multiple N values including non-multiples of BLOCK_SIZE, and use an explicit tolerance justified by dtype.
