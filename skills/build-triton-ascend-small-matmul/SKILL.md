---
name: build-triton-ascend-small-matmul
description: Build a single-program Triton-Ascend matrix multiplication with fused bias using two-dimensional pointer grids and tl.dot. Use when an agent needs a compact matmul-plus-bias kernel for small fixed shapes, a minimal Ascend cube-operation example, or a correctness scaffold before introducing multi-block tiling.
---

# Build Triton-Ascend Small Matmul

## Goal

Compute output[A, C] = x[A, B] @ y[B, C] + bias[A, C] in one Triton program for small compile-time shapes.

## Workflow

1. Validate x as A by B, y as B by C, and bias as A by C.
2. Create row, reduction, and column offsets with tl.arange.
3. Form the flattened pointer grids through broadcasting.
4. Load both matrices and bias, call tl.dot, and store the result.
5. Launch exactly one program only while all tiles fit the target's on-chip resources.

## Implementation Pattern

~~~python
@triton.jit
def matmul_bias(out, x, y, bias,
                A: tl.constexpr, B: tl.constexpr, C: tl.constexpr):
    rows = tl.arange(0, A)
    reduction = tl.arange(0, B)
    cols = tl.arange(0, C)
    x_offsets = rows[:, None] * B + reduction[None, :]
    y_offsets = reduction[:, None] * C + cols[None, :]
    out_offsets = rows[:, None] * C + cols[None, :]
    x_tile = tl.load(x + x_offsets)
    y_tile = tl.load(y + y_offsets)
    bias_tile = tl.load(bias + out_offsets)
    tl.store(out + out_offsets, tl.dot(x_tile, y_tile) + bias_tile)
~~~

Launch with grid=(1, 1, 1) and pass A, B, and C as compile-time values.

## Ascend Guardrails

- Use this pattern only for small exact tiles such as 16 by 16 by 16.
- For arbitrary dimensions, pad to legal tile extents and mask loads/stores or implement a multi-program tiled matmul.
- Begin with float16 inputs; confirm supported tl.dot dtype and accumulation behavior before adding other dtypes.
- Allocate all tensors on the NPU and keep their layouts contiguous unless stride-aware indexing is added.
- Do not describe this two-dimensional kernel as batched matmul unless an explicit batch axis and offsets are implemented.

## Verification

Compare with torch.matmul(x, y) + bias using torch.testing.assert_close. Verify the exact supported shape and dtype first, then add boundary and padded cases only if the kernel implements their masks.
