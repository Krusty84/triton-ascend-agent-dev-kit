---
name: optimize-triton-ascend-int-vectors
description: Optimize integer vector kernels for Triton-Ascend by preferring int32 arithmetic when the value and index ranges permit it. Use when an agent ports an int64 GPU elementwise kernel to Ascend NPU, observes scalarized integer arithmetic, or designs vector addition, subtraction, or reduction over integer tensors.
---

# Optimize Triton-Ascend Integer Vectors

## Goal

Keep integer elementwise work on efficient Ascend vector paths without changing numerical semantics.

## Workflow

1. Determine the minimum signed range required by inputs, offsets, intermediates, and outputs.
2. Use torch.int32 tensors and int32-compatible scalar arguments when every possible value fits.
3. Keep pointer arithmetic and tail masking separate from the arithmetic dtype decision.
4. Launch a masked one-dimensional kernel and compare it with an int32 PyTorch reference.
5. Benchmark int32 and int64 only after synchronizing the NPU around timed regions.

## Implementation Pattern

~~~python
@triton.jit
def int_add(x, y, out, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    lhs = tl.load(x + offsets, mask=mask)
    rhs = tl.load(y + offsets, mask=mask)
    tl.store(out + offsets, lhs + rhs, mask=mask)


x = torch.randint(0, 100, (n,), device="npu", dtype=torch.int32)
y = torch.randint(0, 100, (n,), device="npu", dtype=torch.int32)
out = torch.empty_like(x)
int_add[(triton.cdiv(n, block),)](x, y, out, n, BLOCK_SIZE=block)
~~~

## Ascend Guardrails

- Never downcast identifiers, addresses, prefix sums, or accumulated values that can exceed the int32 range.
- Check intermediate overflow, not only the input range.
- Keep tensors, kernel scalar arguments, and the PyTorch reference on consistent dtypes.
- Use the same mask on all tail loads and stores.
- Synchronize before starting and after ending a host-side timing interval; exclude warmup and compilation.

## Verification

Test boundary values near the chosen dtype limits, irregular lengths, and a case that would overflow int32. The overflow case must remain int64 or be rejected explicitly.
