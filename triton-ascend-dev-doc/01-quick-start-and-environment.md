# Quick Start and Environment

> Load this section for installation, environment checks, and the first smoke test. For kernel design, continue with [Execution Model and Architecture](./02-execution-model-and-architecture.md).

## Documentation Snapshot

The source quick-start uses the following example environment:

| Component | Documented value |
| --- | --- |
| Operating system | Ubuntu 22.04; Linux `aarch64` or `x86_64` |
| Ascend products | Atlas A2, A3, and 950 series |
| Card memory | 32 GB per card recommended |
| CANN | 9.1.0 |
| Python | 3.11 |
| TorchNPU | 2.7.1.post8 |

These values describe the source snapshot, not a universal compatibility guarantee. Keep CANN, driver/firmware, TorchNPU, PyTorch, Python, and Triton-Ascend versions mutually compatible.

## Install

Install CANN, the matching driver/firmware, PyTorch, and TorchNPU first. Then install Triton-Ascend:

```bash
pip install triton-ascend --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi
```

Load the CANN environment in the shell that will run the program. The documented default root installation is:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## Minimal Runtime Smoke Test

Before debugging a custom kernel, verify that TorchNPU sees the device and that Triton-Ascend can compile a trivial program.

```python
import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x, y):
    out = torch.empty_like(x)
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)
    return out


x = torch.rand(98432, device="npu")
y = torch.rand_like(x)
actual = add(x, y)
expected = x + y
torch.testing.assert_close(actual, expected)
print("Triton-Ascend smoke test passed")
```

This launch is intentionally conventional and easy to validate. It is not automatically the best NPU launch strategy for large grids. Use the physical-core loop pattern in [Common Kernel Development](./03-common-kernel-development.md) when dispatch overhead matters.

## Environment Acceptance Checklist

- `import torch_npu` succeeds.
- `torch.rand(..., device="npu")` succeeds.
- The CANN environment is loaded in the current shell.
- A trivial `@triton.jit` kernel compiles and launches.
- Results match a PyTorch operation executed on the NPU.
- The installed versions are checked as a set, not upgraded independently.

If compilation fails before kernel IR is produced, investigate environment compatibility first. If the compiler reports UB, grid, lowering, or instruction problems, use [Validation and Troubleshooting](./11-validation-and-troubleshooting.md).

