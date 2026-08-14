---
name: max-autotune-triton-ascend
description: Apply Triton-Ascend max_autotune to search base kernel configurations together with Ascend compiler options for vector, cube, or mixed operators. Use when an agent needs multi-parameter performance tuning without manually enumerating every triton.Config combination, must choose kernel_type, or must constrain compiler-option search lists such as num_stages and enable_ubuf_saving.
---

# Max-Autotune Triton-Ascend

## Goal

Expand a small set of kernel meta-parameter configurations into a controlled search over Ascend compiler options.

## Workflow

1. Verify the kernel and PyTorch reference before tuning.
2. Import max_autotune from triton.backends.ascend.runtime.
3. Provide a small base configs list containing only meaningful kernel meta-parameters.
4. Set key to runtime values whose changes may require a new winner.
5. Classify the kernel as vector, cube, or mix.
6. Add short explicit option lists only when the built-in search space is unsuitable.

## Implementation Pattern

~~~python
from triton.backends.ascend.runtime import max_autotune

base_configs = [
    triton.Config({"BLOCK_SIZE": 128}),
    triton.Config({"BLOCK_SIZE": 256}),
]

@max_autotune(
    configs=base_configs,
    key=["numel"],
    kernel_type="vector",
    num_stages=[1, 2],
    enable_ubuf_saving=[True, False],
)
@triton.jit
def kernel(out, x, y, numel, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    ...
~~~

Derive the grid from meta["BLOCK_SIZE"] and do not pass BLOCK_SIZE explicitly at launch.

## Ascend Guardrails

- Choose kernel_type from the actual execution mix: vector for vector instructions, cube for matrix engines, and mix for combined work.
- Keep base configs and option lists small; max_autotune forms a combined search space and tuning cost grows quickly.
- Use only compiler option names supported by the installed Triton-Ascend version.
- Put shape variables in key, not tensor values or unrelated arguments.
- Do not mix up max_autotune with standard triton.autotune: the former expands Ascend-specific compiler choices automatically.

## Verification

Compare the tuned output with the PyTorch reference for every key shape. Record the chosen configuration and benchmark it against the untuned baseline on representative inputs.
