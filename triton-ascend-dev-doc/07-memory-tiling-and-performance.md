# Memory, Tiling, and Performance

> Load this section for UB/L1 overflow, padding, irregular access, low instruction overlap, dtype scalarization, or generally poor NPU performance.

## Optimization Order

Optimize in this order because later steps depend on earlier structure:

1. eliminate pathological grid size and startup overhead;
2. fit the live working set in UB/L1;
3. make global-memory movement contiguous and aligned;
4. select Vector-friendly dtypes without changing semantics;
5. create looped tiles that permit data-in/compute/data-out overlap;
6. tune block sizes and compiler options;
7. fuse only when measured traffic justifies it.

## Two Levels of Tiling

Use separate variables for cross-core scheduling and intra-core memory use:

| Parameter | Role |
| --- | --- |
| `ncore` / launch grid | number of concurrent programs |
| `XBLOCK` / `MAIN_BLOCK_SIZE` | logical work assigned to a program; controls grid/coreDim |
| `XBLOCK_SUB` / `SUB_BLOCK_SIZE` | data processed per loop iteration; controls live UB usage |

```python
@triton.jit
def gelu_kernel(in_ptr, out_ptr, n_elements,
                XBLOCK: tl.constexpr,
                XBLOCK_SUB: tl.constexpr):
    base = tl.program_id(0) * XBLOCK

    for sub in range(0, XBLOCK, XBLOCK_SUB):
        offsets = base + sub + tl.arange(0, XBLOCK_SUB)
        mask = offsets < n_elements
        x = tl.load(in_ptr + offsets, mask=mask)
        y = x * 0.5 * (1.0 + tl.erf(x / tl.sqrt(2.0)))
        tl.store(out_ptr + offsets, y, mask=mask)
```

If launch count already equals physical cores, use `range(pid, num_tiles, num_core)` as the outer distribution and an optional sub-loop inside each tile.

## UB/L1 Capacity

The source documentation gives Atlas A2 UB capacity as 192 KiB. The compiler error displays the exact requested and available bit counts, for example:

```text
ub overflow, requires 3072256 bits while 1572864 bits available
```

The largest input tile that fits alone may still fail because live usage also includes:

- multiple inputs and outputs;
- masks and indices;
- reductions and large accumulators;
- layout conversion temporaries;
- double/multi-buffer copies;
- compiler-created local buffers.

When overflow occurs:

1. reduce the intra-core sub-block;
2. shorten the lifetime of intermediates or split phases;
3. reduce buffering only if the lost overlap is acceptable;
4. retune after the structural fix.

## Tail-Axis Alignment and Padding

The source guide documents:

- 32-byte tail-axis alignment for Vector-only work;
- 512-byte tail-axis alignment for CV work.

Short tail axes can be auto-padded. A tensor shaped `[2048, 3]` can therefore perform much worse than its element count suggests.

Preferred remedies:

- transfer a flattened contiguous region when mathematically valid;
- transpose/reorder so a longer aligned dimension becomes innermost during compute;
- preserve shape/stride information that exposes the unit-stride axis;
- store back in the required layout at the end;
- profile the transpose/reorganization cost.

The source shows an axis-borrowing reshape/transpose trick when total bytes are suitably aligned. Treat this as a layout transformation to prove, not a blind text substitution.

## Preserve the Contiguous Axis

Bad conceptual representation for data logically shaped `[1024, 32]`:

```python
tl.make_block_ptr(
    base=input_ptr,
    shape=(1024,),
    strides=(32,),
    offsets=(row_start,),
    block_shape=(BLOCK_ROWS,),
    order=(0,),
)
```

This hides the 32 contiguous elements per row. Prefer:

```python
tl.make_block_ptr(
    base=input_ptr,
    shape=(1024, 32),
    strides=(32, 1),
    offsets=(row_start, 0),
    block_shape=(BLOCK_ROWS, 32),
    order=(1, 0),
)
```

The exact `order` must match the intended layout; the key rule is to expose the stride-1 dimension.

## Convert Discrete GM Access into UB Reuse

Instead of repeatedly loading `x[idx]` from global memory:

1. load a useful contiguous source range into UB;
2. gather/select the required values locally;
3. store results contiguously when possible.

This reduces many small L2→UB transfers, but it increases the bulk bytes moved and UB use. Compare both approaches for the real index distribution. On Ascend 950, the default hybrid SIMT path is another candidate for unstructured access.

## Avoid Unnecessary Padding Dependencies

A masked `tl.load` without an explicit `other` follows GPU-compatible default padding behavior. On NPU, filling masked lanes can require Vector initialization before MTE2 writes valid lanes, creating a dependency and reducing overlap.

If and only if masked lanes cannot affect any later result, the source guide recommends:

```python
data = tl.load(input_ptr + offsets, mask=mask, care_padding=False)
```

Do not use `care_padding=False` when masked lanes enter arithmetic, reductions, stores, address calculations, or control decisions. Use an explicit neutral `other` value when invalid lanes matter.

## Create Instruction Parallelism

Multi-buffering overlaps data-in, compute, and data-out across loop iterations. It may not activate or help when:

- there is only one monolithic pass with no tiling loop;
- data movement depends on the immediately preceding Vector result;
- the extra buffer would exceed UB;
- each iteration is too small or imbalanced.

Split a large single pass into mathematically independent looped tiles. This both lowers UB use and creates multiple stages for overlap. The source states `multibuffer=True` is the default, but default enablement does not guarantee that the compiler can form a useful pipeline.

## Dtype Optimization on A2/A3

The source performance guide records scalar fallbacks for:

- Vector add with `int64`;
- Vector compare with `int64` or `int32`.

Use narrower/supported types only when the value range and semantics allow it. Converting an arbitrary large integer comparison to `float32` can lose exactness. For offsets, prove that the tensor address range fits before changing `int64` to `int32`.

## Performance Symptom Router

| Symptom | Inspect first | Typical action |
| --- | --- | --- |
| High launch latency | logical grid vs physical cores | persistent physical-core loop or Auto-Blockify for independent blocks |
| `coreDim > 65535` | `ceil(numel / BLOCK_SIZE)` | Auto-Blockify or larger outer block plus sub-block |
| UB overflow | live tile/intermediate/buffer set | reduce sub-block, split phase, reduce buffering |
| Low MTE2 ratio | access granularity and dependencies | contiguous bulk movement, loop tiling, avoid unused padding initialization |
| Scalar operations in HIVM IR | dtype, compare, indirect access | supported dtype, layout change, UB gather, 950 hybrid SIMT |
| Short-tail slowdown | 32B/512B padding | reorganize axes or flatten aligned transfers |
| Cube waits for Vector | CV epilogue and balance | split Vector tile, tune CV balance/loop options |
| Vector waits for data | irregular access or no overlap | UB reuse, multibuffer-friendly loop, layout fix |

## Profiling Signals

The source guide uses msProf output (`op_summary_*.csv`) and refers to metrics such as:

- `aiv_mte2_time` and `aiv_mte2_ratio` for Vector-side data movement;
- Cube, Vector, and MTE time ratios for CV balance.

Compare before and after using the same shapes, warmup, synchronization policy, and environment. A higher utilization ratio is useful only when end-to-end latency and correctness also improve.

