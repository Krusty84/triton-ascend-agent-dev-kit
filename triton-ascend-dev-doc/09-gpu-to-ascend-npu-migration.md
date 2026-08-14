# GPU/CUDA Triton to Ascend NPU Migration

> This is the mandatory guide for a kernel derived from NVIDIA/AMD GPU Triton or CUDA-oriented reasoning. It translates familiar GPU assumptions into Ascend NPU design decisions.

## Migration Goal

Migration is complete only when the operator:

1. uses NPU tensors and runtime interfaces;
2. compiles and matches a PyTorch NPU reference;
3. uses a safe, hardware-appropriate grid;
4. fits UB/L1 across required shapes;
5. avoids accidental scalarization and pathological padding;
6. meets the performance target after NPU-specific profiling.

A source-identical kernel that happens to compile satisfies only part of this goal.

## GPU-to-NPU Translation Table

| GPU/CUDA-trained assumption | Ascend NPU correction | Migration action |
| --- | --- | --- |
| A large Triton grid is cheap because blocks are scheduled over SMs | Launch programs are closely tied to physical core resources; excess tasks are delivered in rounds and add startup overhead | use physical-core launch + inner strided tile loop, or Auto-Blockify for independent logical blocks |
| A Triton program is like one CUDA thread | A Triton program is a block/tile of work | translate at block/tile granularity, never thread-by-thread |
| Grid axes naturally represent CUDA `blockIdx.{x,y,z}` | NPU guidance favors flattened 1D task mapping; 2D adaptations are merged | flatten tasks and reconstruct coordinates explicitly |
| SM count is only an occupancy detail | Vector/Cube physical core count is often the preferred program count | query `num_vectorcore` or `num_aicore` based on operator type |
| Tensor Cores and scalar CUDA cores share the GPU scheduling model | Cube and Vector are distinct resources with different tiling/alignment behavior | classify the kernel as Vector, Cube, or CV before selecting grid and tuning |
| Shared-memory/Register tuning intuition transfers directly | UB/L1 capacities and compiler-created buffers define a different live-set limit | budget every tile/intermediate and add intra-core sub-tiling |
| Large logical block fixes grid-count problems | A larger Triton tile can overflow UB | separate `MAIN_BLOCK_SIZE` from `SUB_BLOCK_SIZE` |
| Masked loads are cheap lane predicates | Padding initialization can create Vector→MTE dependencies | use a neutral `other`; use `care_padding=False` only if masked lanes are provably unused |
| Gather/scatter maps naturally to SIMT lanes | SIMD lowering may create scalar loops for discrete access | bulk-load to UB and gather locally; on 950 benchmark hybrid SIMT |
| CUDA integer dtype performance is representative | A2/A3 Vector add/compare can scalarize for common integer dtypes | choose dtypes by proven range/semantics and inspect/profile |
| 32-element or warp-shaped tails are the main alignment concern | source rules are byte-based: 32 B Vector tail, 512 B CV tail | make the unit-stride axis visible and reorganize short tails |
| CUDA streams/events/synchronization APIs can remain | runtime APIs and synchronization semantics are backend-specific | replace with TorchNPU/NPU counterparts or remove unnecessary host synchronization |
| `num_warps`/`num_stages` are standard tuning axes | automatic Ascend tiling does not generate these; Ascend has separate compiler options | use Ascend autotune/max_autotune deliberately |

## Phase 1: Port the Python Runtime Layer

Perform mechanical runtime changes before changing kernel math:

```diff
 import torch
+import torch_npu
 import triton
 import triton.language as tl

-x = torch.rand(shape, device="cuda")
+x = torch.rand(shape, device="npu")

-y = x.cuda()
+y = x.npu()

-z = y.to("cuda")
+z = y.to("npu")
```

Search and review:

- `device="cuda"` / `device='cuda'`;
- `.cuda()` and `.to("cuda")`;
- `torch.cuda.*` streams, events, synchronization, device queries, and memory APIs;
- CUDA-only custom extensions or inline assembly;
- assertions tied specifically to a GPU driver helper.

Replace GPU APIs with TorchNPU/NPU equivalents when the behavior is required. Remove a synchronization or assertion only when it is genuinely redundant. A generic assertion that all tensors share the same device is still valuable; rewrite it to be backend-neutral rather than deleting it blindly.

At the end of this phase, keep the Triton kernel body unchanged and run a small correctness case on NPU. This isolates runtime failures from optimization failures.

## Phase 2: Classify the Kernel

| Kernel content | Class | Physical concurrency basis | Next guide |
| --- | --- | --- | --- |
| element-wise, reduction, gather/scatter; no `tl.dot` | Vector | `num_vectorcore` | [Vector Kernels](./04-vector-kernels.md) |
| main work is `tl.dot`/GEMM | Cube | `num_aicore` | [Cube Kernels](./05-cube-kernels.md) |
| `tl.dot` plus significant Vector post-processing | CV | `num_aicore`, with Vector cooperation | [Cube-Vector Fusion](./06-cube-vector-fusion.md) |

Query the counts rather than hardcoding `32` or another product-specific value:

```python
import torch_npu
import triton.runtime.driver as driver

device = torch_npu.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
num_vectorcore = properties["num_vectorcore"]
num_aicore = properties["num_aicore"]
```

## Phase 3: Translate the Grid

### GPU-style source

```python
grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
kernel[grid](...)
```

This is a valid correctness baseline on NPU when it stays within `coreDim`, but it may create far more programs than physical cores.

### NPU physical-core form

```python
@triton.jit
def kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_core = tl.num_programs(0)
    num_tiles = tl.cdiv(n_elements, BLOCK_SIZE)

    for tile_id in range(pid, num_tiles, num_core):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x, mask=mask)


kernel[(num_vectorcore,)](x, out, n_elements, BLOCK_SIZE=1024)
```

The inner loop assigns tiles `pid`, `pid + num_core`, `pid + 2*num_core`, and so on. It is correct only when tiles can execute independently and every output has exactly one owner.

### Flatten a multidimensional GPU grid

For a source grid conceptually shaped `[B, H, M_tiles]`:

```python
num_m_tiles = triton.cdiv(M, BLOCK_M)
num_tasks = B * H * num_m_tiles
```

Inside the kernel:

```python
for task_id in range(pid, num_tasks, num_core):
    bh = task_id // num_m_tiles
    tile_m = task_id % num_m_tiles
    batch = bh // H
    head = bh % H
    ...
```

This replaces implicit GPU block-axis scheduling with explicit, reviewable task ownership.

## Large Logical Grids and Auto-Blockify

Without Auto-Blockify:

```text
coreDim = product of launch-grid dimensions
coreDim <= 65,535
```

For independent logical blocks, enable compiler/runtime folding:

```bash
export TRITON_ALL_BLOCKS_PARALLEL=1
```

The source architecture describes a compile-time loop over logical block IDs and a matching runtime cap to physical core count. The per-kernel `enable_auto_blockify` option overrides the environment when set.

Use Auto-Blockify when:

- the existing logical mapping is convenient or autotune-generated;
- logical blocks are independent;
- no code assumes a specific cross-block order;
- workspace can be safely reused across a physical core's iterations.

Prefer an explicit physical-core loop when you need clear multidimensional task reconstruction, custom load balancing, fine control of workspace lifetime, or easy source-level reasoning.

## Phase 4: Check Data Movement

### Alignment

The source documentation specifies:

- Vector tail axis: 32-byte alignment;
- CV tail axis: 512-byte alignment.

These are bytes, not element counts. Compute aligned elements from dtype width. Preserve the unit-stride axis and profile shapes with short tails.

### Discrete access

GPU source:

```python
idx = tl.load(idx_ptr + offsets)
value = tl.load(x_ptr + idx * stride, mask=valid)
```

Candidate NPU SIMD transformation when the source range fits UB:

```python
source_offsets = tl.arange(0, M)
x_ub = tl.load(x_ptr + source_offsets * stride)
value = tl.gather(x_ub, idx, 0)
```

This trades a contiguous bulk load and UB use for fewer discrete global loads. It is not universally better. Benchmark it, and on Ascend 950 compare hybrid `compile_mode="unstructured_in_simt"`.

### Layout visibility

If data is logically `[rows, cols]` and `cols` is contiguous, keep shape `(rows, cols)` and strides `(cols, 1)`. A flattened pointer with stride `cols` can hide the contiguous inner region and lead to poor scalar mapping.

## Phase 5: Check On-Chip Memory

The Atlas A2 source value is 192 KiB UB. Do not treat all of it as payload capacity. Count inputs, outputs, intermediates, masks, indices, reduction buffers, accumulators, and pipeline copies.

If a GPU tile is too large:

- lower the tile;
- introduce an inner sub-block loop;
- split a phase or large epilogue;
- reduce buffering only after evaluating lost overlap.

### Compound `coreDim` and UB Overflow

For a one-dimensional workload:

```text
coreDim = ceil(N / MAIN_BLOCK_SIZE)
MAIN_BLOCK_SIZE >= ceil(N / 65,535)
```

If powers of two are required for the chosen tile scheme:

```python
min_main = triton.next_power_of_2(triton.cdiv(N, 65535))
MAIN_BLOCK_SIZE = max(preferred_main, min_main)
```

Do not allocate `tl.arange(0, MAIN_BLOCK_SIZE)` if that overflows UB. Decouple the live tile:

```python
@triton.jit
def masked_fill_kernel(inp, mask_ptr, value, out, n_elements,
                       MAIN_BLOCK_SIZE: tl.constexpr,
                       SUB_BLOCK_SIZE: tl.constexpr):
    base = tl.program_id(0) * MAIN_BLOCK_SIZE

    for sub in range(0, MAIN_BLOCK_SIZE, SUB_BLOCK_SIZE):
        offsets = base + sub + tl.arange(0, SUB_BLOCK_SIZE)
        valid = offsets < n_elements
        x = tl.load(inp + offsets, mask=valid, other=0.0)
        fill = tl.load(mask_ptr + offsets, mask=valid, other=0).to(tl.int1)
        result = tl.where(fill, value, x)
        tl.store(out + offsets, result, mask=valid)
```

Here `MAIN_BLOCK_SIZE` reduces launch count while `SUB_BLOCK_SIZE` controls UB usage. Tune them separately.

## Phase 6: Translate Compute and Dtypes

Check every migrated operation/dtype pair against the [Operator Support Matrix and Constraints](./12-operator-support-matrix-and-constraints.md). This is especially important for atomics, arbitrary transpose/permute, gather, tensor descriptors, random operations, and GPU compiler hints.

### Vector operations

The source performance guide records A2/A3 scalarization risks:

- Vector add: `int64`;
- Vector compare: `int64` and `int32`.

For offsets and lengths, use `int32` only after proving range safety. For compare, converting to `float32` is safe only while every possible integer is exactly representable and comparison semantics remain unchanged.

### `tl.dot`

Re-evaluate `BLOCK_M/N/K`, K-loop depth, input layout, accumulator dtype, and output cast. GPU Tensor Core tile choices are candidates, not defaults. Use zero for masked input lanes and organize persistent concurrency around `num_aicore`.

### Masked values

GPU-compatible default padding can create an NPU Vector initialization dependency. Use:

- explicit `other=neutral_value` when masked lanes participate;
- `care_padding=False` only when masked lanes are never observed.

## Phase 7: Translate Synchronization and Pipeline Assumptions

Replace host CUDA stream/event/synchronization APIs with required NPU equivalents. Inside kernels, do not invent CUDA barrier semantics. Triton-Ascend exposes block synchronization extensions (`tl.sync_block_set`, `tl.sync_block_wait`, `tl.sync_block_all`) for explicitly designed cross-core pipelines.

Auto-Blockify changes execution into multiple logical iterations per physical core. A kernel that depends on global logical-block order or a CUDA-style cooperative schedule must be redesigned before enabling it.

Multi-buffering is enabled by default according to the source guide, but requires:

- multiple loop iterations/stages;
- enough UB for extra buffers;
- limited data dependencies between movement and compute.

## Phase 8: Autotune on Ascend

Do not reuse only the GPU's winning configuration. First establish a small safe candidate set, then choose:

- `@triton.autotune(configs=[])` for backend-generated tiling;
- handwritten `triton.Config` when the valid space is constrained;
- `max_autotune` when base tilings are known and Ascend compiler options should be expanded.

Required Ascend automatic-tiling details are in [Autotuning](./08-autotuning.md).

## Complete Vector Migration Example

```python
import torch
import torch_npu
import triton
import triton.language as tl
import triton.runtime.driver as driver


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements,
               BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_core = tl.num_programs(0)
    num_tiles = tl.cdiv(n_elements, BLOCK_SIZE)

    for tile_id in range(pid, num_tiles, num_core):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.device == y.device
    assert x.shape == y.shape
    out = torch.empty_like(x)
    properties = driver.active.utils.get_device_properties(
        torch_npu.npu.current_device()
    )
    num_vectorcore = properties["num_vectorcore"]
    add_kernel[(num_vectorcore,)](
        x, y, out, out.numel(), BLOCK_SIZE=1024
    )
    return out


x = torch.rand(98433, device="npu")
y = torch.rand_like(x)
torch.testing.assert_close(add(x, y), x + y)
```

For a tiny input, launching every physical core may be unnecessary; cap the launch to the number of available tiles if the runtime/compiler behavior for the target is verified:

```python
num_tiles = triton.cdiv(x.numel(), 1024)
grid = (min(num_vectorcore, num_tiles),)
```

## Migration Review Checklist

### Runtime

- `torch_npu` is imported.
- Tensor creation, transfer, streams/events, synchronization, and device APIs are NPU-compatible.
- No CUDA-only extension remains on the active path.

### Grid

- Kernel is classified as Vector, Cube, or CV.
- Grid count and physical core basis are explicit.
- `coreDim <= 65,535`, or applicable Auto-Blockify is intentionally enabled.
- Inner-loop tile ownership covers outputs exactly once.
- No cross-block ordering assumption is violated.

### Memory

- Every tail access is masked.
- Masked loads use the correct neutral value.
- The contiguous axis is visible in shape/strides.
- 32-byte Vector or 512-byte CV tail alignment has been considered.
- UB/L1 budget includes intermediates and buffering.
- Large outer scheduling blocks use smaller live sub-blocks if needed.

### Compute

- Integer dtype choices do not accidentally scalarize or overflow.
- `tl.dot` tiling and accumulator dtype are revalidated for Ascend.
- Discrete access has been profiled against UB reorganization and, on 950, hybrid SIMT.

### Verification

- PyTorch NPU reference passes boundary and representative shapes.
- Performance is measured after warmup/tuning.
- Profiling identifies the bottleneck rather than assuming GPU behavior.
