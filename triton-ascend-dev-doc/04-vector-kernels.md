# Vector Kernel Development

> Load this section for element-wise operations, row reductions, conversions, gather/scatter, masked updates, token reordering, and small fused kernels without `tl.dot`.

## Classification

A kernel is Vector-only when its main work is performed by Vector Cores and it has no matrix `tl.dot` stage. Launch concurrency around `num_vectorcore`, not Cube/AI Core count.

Before choosing an operation or dtype, consult the [Operator Support Matrix](./12-operator-support-matrix-and-constraints.md). For example, the documented `gather` support is limited to floating-point dtypes and the last tensor axis, even though other irregular-access strategies may be expressible through different APIs.

## Canonical Vector Pattern

```python
@triton.jit
def vector_kernel(x_ptr, y_ptr, out_ptr, n_elements,
                  BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_core = tl.num_programs(0)
    num_tiles = tl.cdiv(n_elements, BLOCK_SIZE)

    for tile_id in range(pid, num_tiles, num_core):
        offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        out = x + y
        tl.store(out_ptr + offsets, out, mask=mask)
```

Design rules:

- use contiguous `tl.arange` offsets on the innermost axis;
- keep the grid close to physical Vector Core count for large workloads;
- make each tile independent;
- use the largest tile that fits UB and transfers efficiently, not simply the largest legal Triton shape;
- mask the tail and specify `other` if masked lanes enter a reduction.

## Reductions

For row-wise reductions:

1. assign one or more rows/tasks to each physical Vector Core;
2. load a contiguous column tile;
3. reduce in a stable accumulator dtype;
4. loop over the hidden dimension if the whole row does not fit UB;
5. handle invalid columns with a neutral `other` value (`0` for sum, `-inf` for max).

Be careful with online reductions across sub-blocks. A sum can be accumulated directly; softmax needs stable running-max/running-normalizer updates.

## Irregular Gather/Scatter

Many small indirect global-memory loads can lower to scalar loops on the SIMD path. When the source range is reasonably sized, move it to UB contiguously and select locally:

```python
@triton.jit
def pick_kernel(x_ptr, idx_ptr, y_ptr, stride_x, stride_idx, stride_y,
                M: tl.constexpr, N: tl.constexpr):
    rn = tl.arange(0, N)
    idx = tl.load(idx_ptr + rn * stride_idx)
    valid = idx < M

    rm = tl.arange(0, M)
    x_shared = tl.load(x_ptr + rm * stride_x)
    value = tl.gather(x_shared, idx, 0)
    tl.store(y_ptr + rn * stride_y, value, mask=valid)
```

Apply this only when the bulk UB load fits and does not move vastly more data than the discrete alternative. On Ascend 950, also benchmark default hybrid SIMT for truly unstructured access.

## Complex Reordering Pattern

For token/expert/bin gather-scatter:

1. distribute outer tasks across `num_vectorcore`;
2. tile the hidden dimension with `BLOCK_X` according to UB capacity;
3. batch small irregular tasks with `SUB_BLOCK_SIZE`;
4. keep index validity, hidden-column validity, and expert/bin boundaries as separate masks;
5. combine masks only at load/store sites;
6. assemble/scatter UB-local blocks with Ascend slice extensions when ordinary indexing creates discrete access.

The source examples use:

```python
import triton.language.extra.cann.extension as extension

# extension.insert_slice(...)
# extension.extract_slice(...)
```

The compiler reference also documents core forms `tl.insert_slice`, `tl.extract_slice`, and `tl.get_element`. Check the installed Triton-Ascend version and follow the namespace used by its tests/examples.

## UB Budgeting

Budget all live objects, not only inputs. A rough model for one iteration is:

```text
UB use ≈ loaded input tiles
       + output/intermediate tiles
       + mask/index storage
       + reduction temporaries
       + compiler/pipeline buffers
```

For complex gather/scatter, derive `BLOCK_X` and `SUB_BLOCK_SIZE` from the UB budget and element/index byte widths. Alignment-round the contiguous hidden tile. If compilation reports UB overflow, reduce the live sub-block first.

## Alignment and Layout

- The source guide requires the Vector tail axis to be divisible by 32 bytes; otherwise padding may occur.
- Preserve a unit-stride innermost dimension.
- For a logical `[rows, 32]` buffer, represent it as shape `(rows, 32)` with strides `(32, 1)`, rather than flattening it as a strided 1D view.
- For very short tails such as `[2048, 3]`, consider a transpose/reshape that borrows an aligned factor from a long axis, but validate the transformation and profile its own padding cost.

## Dtype Rules for A2/A3

The source performance guide records:

| Vector operation | Dtypes that may scalarize |
| --- | --- |
| add | `int64` |
| compare | `int64`, `int32` |

If semantics permit:

- use `int32` rather than `int64` for offsets, lengths, and arithmetic;
- for a comparison that scalarizes, consider `float32` only when conversion preserves the comparison exactly for the possible value range;
- never trade correctness for vectorization; profile or inspect IR when uncertain.

## Vector Performance Checklist

- Is the launch count close to `num_vectorcore` for a large workload?
- Is the innermost axis contiguous and 32-byte aligned?
- Does any masked load create padding work whose values are never consumed?
- Does a discrete GM access have a bulk-load-to-UB alternative?
- Are index/compare dtypes mapped to Vector instructions?
- Is the live tile small enough for UB with multibuffering enabled?
- Can a single-pass kernel be split into looped sub-tiles to overlap data-in, compute, and data-out?
