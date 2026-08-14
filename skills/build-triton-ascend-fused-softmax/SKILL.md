---
name: build-triton-ascend-fused-softmax
description: Build a fused row-wise softmax kernel for Triton-Ascend with stable reductions, power-of-two padding, strided rows, and masked memory operations. Use when an agent needs to fuse max, exponentiation, sum, and normalization for a two-dimensional NPU tensor or adapt this reduction pattern to a similar row-wise operator.
---

# Build Triton-Ascend Fused Softmax

## Goal

Compute softmax across the last dimension while reading each input row once and writing each output row once.

## Workflow

1. Interpret the input as n_rows by n_cols and preserve its row stride.
2. Choose BLOCK_SIZE as the next power of two greater than or equal to n_cols.
3. Let each program process rows separated by tl.num_programs(0).
4. Load a padded row with -inf outside n_cols.
5. Subtract the row maximum, exponentiate, reduce the sum, normalize, and store with the original mask.

## Implementation Pattern

~~~python
@triton.jit
def softmax_kernel(out, x, x_stride, out_stride, n_rows, n_cols,
                   BLOCK_SIZE: tl.constexpr):
    first_row = tl.program_id(0)
    row_step = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    for row in tl.range(first_row, n_rows, row_step):
        values = tl.load(x + row * x_stride + cols, mask=mask,
                         other=-float("inf"))
        shifted = values - tl.max(values, axis=0)
        numerator = tl.exp(shifted)
        result = numerator / tl.sum(numerator, axis=0)
        tl.store(out + row * out_stride + cols, result, mask=mask)


def softmax(x):
    assert x.ndim == 2
    n_rows, n_cols = x.shape
    block = triton.next_power_of_2(n_cols)
    out = torch.empty_like(x)
    programs = min(32, n_rows)
    softmax_kernel[(programs,)](
        out, x, x.stride(0), out.stride(0), n_rows, n_cols,
        BLOCK_SIZE=block,
    )
    return out
~~~

## Ascend Guardrails

- Use -inf for padded values so they cannot affect the maximum or denominator.
- Perform the max shift before tl.exp; Triton exponentiation is fast and approximate.
- Reject n_cols == 0 and shapes whose padded row exceeds available on-chip resources.
- Preserve explicit strides or require a contiguous last dimension.
- Keep the program count independent of n_rows when using the row-striding loop; cap it by n_rows.

## Verification

Compare with torch.softmax(x, dim=1) using dtype-appropriate tolerances. Test irregular dimensions such as 1823 by 781, very small rows, and rows near the largest supported BLOCK_SIZE.
