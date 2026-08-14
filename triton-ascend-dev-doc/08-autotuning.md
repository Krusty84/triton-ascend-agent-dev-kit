# Autotuning and `max_autotune`

> Load this section after a fixed kernel is correct. It distinguishes automatic tiling, handwritten configurations, and Cartesian expansion of Ascend compiler options.

## Choose One Tuning Strategy

| Situation | Recommended interface |
| --- | --- |
| Let the Ascend backend infer tiling candidates | `@triton.autotune(configs=[], ...)` |
| Automatic generation fails or must be tightly controlled | handwritten `triton.Config` list |
| Tiling candidates are known; expand Ascend compiler options | `max_autotune` |

Do not begin with `max_autotune` merely because it searches more combinations. A small, informed search space is faster to tune and easier to reason about.

## Automatic Tiling with `configs=[]`

### Required setup

```python
import triton
import triton.language as tl
import triton.backends.ascend.runtime


@triton.autotune(configs=[], key=["n_elements"])
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements,
               BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
add_kernel[grid](x, y, out, n_elements)
```

All of these conditions are required:

1. `import triton.backends.ascend.runtime` activates the Ascend autotune extension.
2. `@triton.autotune` directly wraps `@triton.jit`; insert no decorator between them.
3. A tunable tiling parameter is `tl.constexpr` with no default value.
4. Do not pass that parameter explicitly at launch.
5. If it affects grid size, define the grid as `lambda meta: ...`.

On Ascend, `configs=[]` means “generate and benchmark candidates,” not “run with no configurations.” The backend focuses on tiling-related constexpr parameters and filters candidates using alignment, on-chip memory, and core-utilization constraints.

Automatic generation does not automatically infer every parameter. The source explicitly excludes `num_warps`, `num_stages`, and non-tiling kernel/compiler parameters from this generation scope.

For manually enumerated parameters used as `tl.arange` extents, the source examples require power-of-two candidates. Keep all candidates valid for the kernel's compile-time shape operations before benchmarking them.

### Cache key

`key` has community Triton semantics. Include runtime values whose changes can alter the best configuration, commonly `M`, `N`, `K`, sequence length, or hidden size. A broader key improves specialization but increases tuning/cache cost.

### Extra search dimensions with `hints`

```python
@triton.autotune(
    configs=[],
    key=["M", "N", "K"],
    hints={
        "GROUP_SIZE_M": [1, 2, 4, 8],
        "multibuffer": [False, True],
    },
)
@triton.jit
def matmul_kernel(..., BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    ...
```

Tiling parameters remain backend-generated; explicit non-tiling/compiler dimensions come from `hints`.

## Handwritten `triton.Config`

Use handwritten candidates when automatic tiling yields no valid configuration, misses the performance target, or the valid region is already known:

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}),
    ],
    key=["M", "N"],
)
@triton.jit
def kernel(...):
    ...
```

Ascend compiler options can be included in each configuration, for example `{"multibuffer": True}`.

## `max_autotune`

`max_autotune` expands each base configuration with the Cartesian product of supported tuning parameters, then delegates benchmarking, selection, and caching to `@triton.autotune`.

```python
from triton.backends.ascend.runtime import max_autotune


@max_autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_M": 128, "BLOCK_N": 128}),
        triton.Config(kwargs={"BLOCK_M": 64, "BLOCK_N": 256}),
    ],
    key=["M", "N"],
    kernel_type="mix",
    enable_hivm_auto_cv_balance=[True, False],
    tile_mix_vector_loop=[2, 4],
)
@triton.jit
def kernel(..., BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, **META):
    ...
```

### Supported parameter matrix

| Parameter | cube | mix | vector | Documented default candidates | Documented valid candidates |
| --- | :---: | :---: | :---: | --- | --- |
| `num_stages` | yes | yes | yes | `[2]` | `[1, 2]` |
| `unit_flag` | yes | yes | no | `[False]` | booleans |
| `limit_auto_multi_buffer_of_local_buffer` | yes | yes | no | `["no-l0c"]` | `"no-limit"`, `"no-l0c"` |
| `limit_auto_multi_buffer_only_for_local_buffer` | no | yes | no | `[False]` | booleans |
| `set_workspace_multibuffer` | no | yes | no | `[2, 4]` | `2`, `4` |
| `enable_hivm_auto_cv_balance` | no | yes | no | `[True]` | booleans |
| `tile_mix_vector_loop` | no | yes | no | `[2, 4]` | `2`, `4`, `8` |
| `tile_mix_cube_loop` | no | yes | no | `[2, 4]` | `2`, `4`, `8` |
| `enable_ubuf_saving` | no | yes | yes | `[True]` | booleans |

Kernel types:

- `cube`: pure matrix-oriented operators;
- `vector`: pure Vector operators;
- `mix`: CV-fused operators and the default type.

### Value priority

For a supported tuning parameter:

1. list passed to `max_autotune` has highest priority;
2. value already fixed in the base `Config.kwargs` becomes a one-value dimension;
3. otherwise, the internal default candidate list is used.

Every `tuning_params` value must be a non-empty list or tuple. Unsupported parameters produce a warning and are ignored for that `kernel_type`.

### Configuration count

```text
expanded configurations
  = number of base configurations
  × product(length of every participating candidate list)
```

Defaults participate too. In the source `mix` example, two base configs, two two-value user dimensions, and two two-value default dimensions yield `2 × 2 × 2 × 2 × 2 = 32` configurations.

Estimate this count before launching. A 96-configuration search may be justified for a production hot kernel but wasteful during early development.

## Benchmark Behavior and Environment Variables

Autotune executes a kernel multiple times. Kernels with atomics, in-place writes, accumulated outputs, or other side effects must use the standard Triton reset/hook mechanisms.

| Variable | Effect |
| --- | --- |
| `TRITON_PRINT_AUTOTUNING=1` | print selected tuning information |
| `TRITON_AUTOTUNE_PARALLEL_COMPILE=0` | disable the default parallel compilation of candidates |
| `TRITON_BENCH_METHOD="npu"` | use profiler-based on-chip timing; more accurate for short kernels, but slower to tune |

## Tuning Acceptance Checklist

- Fixed configuration is already correct.
- Search key contains every runtime value that materially changes tiling.
- Tunable constexpr values are not fixed by defaults or launch arguments.
- Grid lambda depends on tunable block sizes.
- Side effects are reset between benchmark runs.
- Candidate count is calculated before `max_autotune`.
- Best configuration is revalidated for correctness.
- Performance is measured after warmup and cache population.
