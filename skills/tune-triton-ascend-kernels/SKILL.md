---
name: tune-triton-ascend-kernels
description: Add standard or advanced autotuning to Triton-Ascend kernels, define shape keys and candidate meta-parameters, and structure kernels so automatic split and tiling analysis can recognize their axes. Use when an agent needs to benchmark block-size or multibuffer choices, use configs=[] for generated vector-kernel candidates, or diagnose why Triton-Ascend autotune cannot infer a tunable axis.
---

# Tune Triton-Ascend Kernels

## Goal

Select a correct kernel configuration for each relevant input shape without hardcoding one block size for all workloads.

## Workflow

1. Establish a correct untuned kernel and reference test.
2. Choose standard autotune when candidate configurations are known.
3. Choose advanced autotune when Triton-Ascend should infer vector split or tiling candidates.
4. Put only values that can change the winner in key.
5. Derive the launch grid from meta-parameters and omit tunable meta-parameters at launch.
6. Validate the tuned result, then benchmark representative shapes.

## Implementation Pattern

Use explicit candidates for standard autotune:

~~~python
@triton.autotune(
    configs=[
        triton.Config({"XS": 128, "multibuffer": True}),
        triton.Config({"XS": 8192, "multibuffer": False}),
    ],
    key=["numel"],
)
@triton.jit
def kernel(out_ptr, x_ptr, numel, XS: tl.constexpr):
    offsets = tl.program_id(0) * XS + tl.arange(0, XS)
    mask = offsets < numel
    values = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, values, mask=mask)
~~~

Use generated candidates for advanced vector autotune:

~~~python
@triton.autotune(
    configs=[],
    key={"x": "n_elements"},
    split_params={"x": "BLOCK_SIZE"},
    tiling_params={},
    low_dims=["x"],
    persistent_reduction=False,
    dual_reduction=False,
)
@triton.jit
def kernel(out_ptr, x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, values, mask=mask)
~~~

## Ascend Guardrails

- Standard Triton-Ascend autotune supports block-size and multibuffer choices; do not assume community num_warps or num_stages tuning maps to Ascend hardware.
- Treat configs=[] as generated-candidate mode only when split_params or tiling_params identifies at least one axis.
- Make a split parameter multiply tl.program_id and compare its resulting offsets with the matching key in a mask.
- Make a tiling parameter participate in tl.arange and a loop range, then compare its offsets with the matching key.
- Candidate meta-parameters must remain omitted from the kernel launch; explicitly passed values are excluded from automatic inference.
- Keep advanced automatic tiling to supported vector kernels; do not apply it to cube operators.
- Set TRITON_BENCH_METHOD="npu" only when profiler timing is needed for very fast kernels; expect substantially longer tuning.

## Verification

Compare the selected kernel with the PyTorch reference for every key shape and inspect that retuning happens when the key changes. Benchmark the winner separately; autotune success does not replace correctness tests.
