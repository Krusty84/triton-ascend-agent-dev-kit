# Validation and Troubleshooting

> Load this section for correctness tests, compilation failures, `coreDim`, UB overflow, scalar lowering, or unexplained performance regressions.

## Validation Ladder

Run checks in this order:

1. environment and trivial-kernel smoke test;
2. fixed-configuration correctness;
3. boundary-shape correctness;
4. large-shape compilation and correctness;
5. deterministic repeated execution;
6. performance warmup and measurement;
7. autotune and revalidation of the winning configuration;
8. profiling/IR inspection for remaining bottlenecks.

Do not diagnose performance while correctness or environment compatibility is unresolved.

## Reference Test Template

```python
import pytest
import torch
import torch_npu


@pytest.mark.parametrize("size", [1, 31, 32, 33, 1024, 1025, 98433])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_kernel(size, dtype):
    x = torch.randn(size, device="npu", dtype=dtype)
    y = torch.randn(size, device="npu", dtype=dtype)

    expected = x + y
    actual = triton_add(x, y)

    torch.testing.assert_close(actual, expected)
```

Adapt the shape set to include:

- zero-sized inputs if supported by the wrapper;
- one element/row/tile;
- exact block multiples;
- `block - 1` and `block + 1` tails;
- short alignment-sensitive tails;
- representative production shapes;
- non-contiguous tensors when supported;
- every public dtype;
- extreme values for reductions, exponentials, and integer conversions.

For `tl.dot`/reductions, choose tolerance based on input, accumulator, and output dtypes. Report maximum absolute and relative error when investigating a mismatch.

## Failure Router

| Error or symptom | Likely cause | First action |
| --- | --- | --- |
| import/device failure | incompatible or unloaded CANN/TorchNPU environment | run the quick-start smoke test and verify version set |
| unsupported operation or dtype during lowering | operation/dtype cell is unsupported or an operation-specific constraint is violated | check the [Operator Support Matrix](./12-operator-support-matrix-and-constraints.md) before changing types or algorithms |
| `coreDim=... can't be greater than UINT16_MAX` | launch grid exceeds 65,535 | use Auto-Blockify or increase outer scheduling block; then check UB |
| `ub overflow, requires ... bits while ... available` | live tile/intermediates/buffers exceed UB | reduce inner sub-block or split the phase |
| wrong tail values | missing/incorrect mask or `other` | test `tile±1`, trace every load/store mask |
| wrong reduction | masked lanes use non-neutral values or sub-block reduction is composed incorrectly | set neutral values and verify online reduction math |
| very slow small-tail shapes | 32B/512B auto-padding | expose/reorganize a longer contiguous axis |
| very slow large input with correct output | huge logical grid and repeated dispatch | physical-core loop or applicable Auto-Blockify |
| scalar instructions/loops in IR | unsupported dtype or discrete access | change proven-safe dtype, reshape access, UB gather, or 950 hybrid SIMT |
| low MTE/compute overlap | monolithic pass, dependencies, or no UB for multibuffer | loop tiling, remove unused padding dependency, rebudget UB |
| autotune produces no candidates | DSL cannot be parsed/inferred or constraints filter all candidates | verify decorator/import/tunables; use handwritten `triton.Config` |
| autotune corrupts results | side effects repeated during benchmarking | use reset/pre/post hooks or a side-effect-safe benchmark |

## Diagnose `coreDim`

For a 1D grid:

```text
coreDim = ceil(N / BLOCK_SIZE)
required BLOCK_SIZE >= ceil(N / 65,535)
```

Solutions:

1. enable Auto-Blockify for independent logical blocks;
2. launch physical core count and process multiple tiles inside each program;
3. increase outer `MAIN_BLOCK_SIZE`, but add a smaller `SUB_BLOCK_SIZE` if UB would overflow.

Do not stop when the numeric limit disappears. A grid of 60,000 programs can still be inefficient on a device with only dozens of relevant cores.

## Diagnose UB Overflow

Read the compiler's requested/available bit counts. Then inventory all live arrays and pipeline copies. Apply the smallest structural change:

- reduce `SUB_BLOCK_SIZE`;
- split a large reduction/epilogue;
- avoid materializing a large temporary;
- process accumulator slices;
- reduce or constrain multibuffering;
- change layout only when it reduces actual live storage.

If increasing `BLOCK_SIZE` fixed `coreDim`, keep it as the outer scheduling span and tile it internally. Do not revert to an unsafe grid.

## Diagnose Incorrect Masking

For every memory operation, answer:

1. Can its address be out of range?
2. What mask prevents that?
3. If a lane is invalid, can its loaded value reach a later operation?
4. What is the mathematically neutral `other` value?
5. Is the same validity condition applied on store?

Examples:

- sum: `other=0`;
- max: `other=-float("inf")` conceptually, expressed in a compiler-supported form;
- GEMM input K/M/N tails: `other=0.0`;
- ignored load lanes: `care_padding=False` only when truly unobservable.

## Diagnose Discrete Access and Scalarization

Set:

```bash
export TRITON_DEBUG=1
```

Inspect cached TTIR/adapter/HIVM output with the installed compiler toolchain. Compare source operations with lowering:

- Did a gather become nested scalar loops?
- Did an integer comparison become scalar work?
- Did a flat strided pointer hide a contiguous second axis?
- Did masked padding introduce a Vector initialization before MTE2?
- On Ascend 950, did hybrid mode emit eligible indirect access or fall back?

Candidate fixes must preserve semantics:

- represent the real multidimensional shape and unit stride;
- bulk-load a bounded source region to UB and gather locally;
- select a supported dtype only after range/precision proof;
- compare `"simd"` and `"unstructured_in_simt"` on Ascend 950.

## Performance Measurement Discipline

- Warm up compilation and runtime caches.
- Separate first-tune time from steady-state execution.
- Synchronize using the correct NPU mechanism around host timing.
- Measure the same shapes and dtypes for baseline and candidate.
- Keep correctness checks outside the timed region.
- Record selected autotune configuration.
- Use profiler-based autotune timing (`TRITON_BENCH_METHOD="npu"`) for very short kernels when the extra tuning cost is acceptable.
- Inspect Cube, Vector, and MTE ratios for CV kernels.

An optimization is accepted only when the relevant end-to-end metric improves without violating correctness across the validation matrix.

## Final Review Record

For an AI-generated kernel, retain a compact record:

```text
operator class: Vector | Cube | CV
target product: ...
supported dtypes/layouts: ...
grid strategy: logical | physical-core loop | Auto-Blockify
physical core basis: num_vectorcore | num_aicore
outer/inner tile sizes: ...
autotune strategy and key: ...
correctness shapes/tolerances: ...
known constraints: ...
measured baseline and result: ...
```

This prevents a later agent from reintroducing GPU assumptions or treating a tuned constant as universal.
