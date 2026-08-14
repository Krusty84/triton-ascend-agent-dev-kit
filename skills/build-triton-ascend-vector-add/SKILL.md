---
name: build-triton-ascend-vector-add
description: Build and debug masked one-dimensional elementwise kernels for Triton-Ascend, including launch wrappers and PyTorch/NPU correctness checks. Use when an agent needs a minimal Triton-Ascend kernel scaffold or must apply program IDs, pointer offsets, tail masks, constexpr block sizes, and launch grids to vector operations.
---

# Build Triton-Ascend Vector Add

## Goal

Implement a standalone NPU kernel that computes elementwise addition for arbitrary tensor lengths. Reuse the same pattern for other one-dimensional elementwise operations.

## Workflow

1. Require both inputs to have the same shape, dtype, device, and supported layout.
2. Allocate the output with torch.empty_like and use output.numel() as the logical length.
3. Assign one contiguous block to each Triton program.
4. Mask every load and store so non-divisible tail elements never access invalid memory.
5. Derive the launch grid from the selected BLOCK_SIZE.

## Implementation Pattern

~~~python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x, y):
    assert x.shape == y.shape and x.dtype == y.dtype and x.device == y.device
    out = torch.empty_like(x)
    n = out.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
    return out
~~~

## Ascend Guardrails

- Import torch_npu before executing on device="npu".
- Treat BLOCK_SIZE as a compile-time meta-parameter and pass it by keyword.
- Use flat addressing only for contiguous tensors. Call contiguous() explicitly or implement stride-aware addressing for non-contiguous inputs.
- Keep the same tail mask on corresponding loads and stores.
- Start with BLOCK_SIZE=1024, then tune only when representative benchmarks justify it.

## Verification

Compare against x + y with torch.testing.assert_close. Include at least one length that is not divisible by BLOCK_SIZE, an empty-or-minimal supported length, and every dtype the operator claims to support.
