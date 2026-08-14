# Common Kernel Development Workflow

> Load this section when implementing a Triton-Ascend kernel from a reference operation. Then load the Vector, Cube, or CV guide that matches the operator.

## Success Contract

A kernel is ready only when it:

1. compiles for the target Ascend product;
2. matches a PyTorch NPU reference across representative and boundary shapes;
3. uses a grid that is safe for `coreDim` and appropriate for physical cores;
4. stays within UB/L1 capacity;
5. has contiguous/aligned data movement where possible;
6. is profiled before performance claims are made.

## Step 1: Specify Semantics

Record before coding:

- input and output shapes;
- dtypes and accumulator dtype;
- strides and layout assumptions;
- broadcasting rules;
- behavior for empty dimensions;
- tail/out-of-bounds behavior;
- acceptable numerical tolerance;
- whether the operation is Vector, Cube, or CV.

Avoid embedding contiguous-layout assumptions unless the wrapper verifies them. Pass strides for general multidimensional tensors.

When a constexpr block size defines `tl.arange(0, BLOCK_SIZE)`, use values accepted by Triton's shape rules; the source autotune examples use powers of two. Do not generate an arbitrary runtime-sized `tl.arange`.

## Step 2: Write a PyTorch NPU Oracle

```python
def reference(x, y):
    return x + y

x = torch.randn(98433, device="npu", dtype=torch.float16)
y = torch.randn_like(x)
expected = reference(x, y)
```

The reference must execute on the NPU so that dtype and backend semantics are comparable.

## Step 3: Implement the Smallest Correct Triton Kernel

Use explicit offsets and tail masks:

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

Use `other=0.0` when invalid loaded lanes participate in reductions or arithmetic. Omitting `other` is safe only if masked lanes cannot affect later results. The Ascend-specific `care_padding=False` optimization is described in [Memory, Tiling, and Performance](./07-memory-tiling-and-performance.md#avoid-unnecessary-padding-dependencies).

## Step 4: Choose a Launch Pattern

### Correctness-first logical grid

```python
grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)
```

### Physical-core loop for a large Vector workload

```python
@triton.jit
def add_persistent_kernel(x_ptr, y_ptr, out_ptr, n_elements,
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


properties = driver.active.utils.get_device_properties(torch_npu.npu.current_device())
num_vectorcore = properties["num_vectorcore"]
add_persistent_kernel[(num_vectorcore,)](x, y, out, n_elements, BLOCK_SIZE=1024)
```

Start with the logical grid for easy verification. Switch to the physical-core loop when the grid is much larger than the hardware and profiling shows launch overhead. Ensure each `tile_id` is independent.

## Step 5: Add Two-Level Tiling if Needed

Separate scheduling granularity from live UB working-set size:

```python
@triton.jit
def tiled_kernel(x_ptr, out_ptr, n_elements,
                 MAIN_BLOCK_SIZE: tl.constexpr,
                 SUB_BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * MAIN_BLOCK_SIZE

    for sub in range(0, MAIN_BLOCK_SIZE, SUB_BLOCK_SIZE):
        offsets = base + sub + tl.arange(0, SUB_BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x, mask=mask)
```

`MAIN_BLOCK_SIZE` controls grid/coreDim. `SUB_BLOCK_SIZE` controls the amount live on chip in one iteration. This pattern solves the common conflict between too many logical blocks and UB overflow.

## Step 6: Handle Multidimensional Work Explicitly

For a logical `[B, H, M_tiles]` task space:

```python
num_m_tiles = tl.cdiv(M, BLOCK_M)
num_tasks = B * H * num_m_tiles

for task_id in range(pid, num_tasks, num_core):
    bh = task_id // num_m_tiles
    tile_m = task_id % num_m_tiles
    batch = bh // H
    head = bh % H
    ...
```

For GEMM, map the output M/N tile and loop over K. For batched GEMM, either include batch in the flattened task ID or use grid axis 2 only after verifying the target mapping.

## Step 7: Validate Before Optimizing

At minimum test:

- one element and one tile;
- exact tile multiple;
- one element past a tile boundary;
- short unaligned tail;
- empty input if the public API permits it;
- representative large input;
- non-contiguous input if strides are part of the contract;
- every supported dtype.

Use `torch.testing.assert_close`. Check maximum absolute/relative error for floating-point reductions and `tl.dot`.

## Step 8: Optimize One Axis at a Time

Recommended order:

1. physical-core utilization and grid count;
2. UB/L1 fit and inner tiling;
3. contiguous and aligned data movement;
4. dtype-induced scalarization;
5. transfer/compute overlap and multibuffering;
6. autotuning of tiling and compiler options;
7. fusion, only when intermediate global-memory traffic is material.

Re-run correctness after each transformation.

## Common Generation Errors

| Error | Corrective rule |
| --- | --- |
| Reusing a huge GPU grid | Bound concurrent programs by the relevant physical core count and iterate tiles, or use Auto-Blockify for independent blocks |
| Increasing `BLOCK_SIZE` until `coreDim` fits | Add a smaller sub-block so UB usage does not grow with scheduling granularity |
| Omitting masks on tails | Mask every potentially out-of-range load/store; define `other` when invalid lanes affect computation |
| Flattening a 2D layout with a non-unit stride | Preserve 2D shape/strides so the contiguous axis stays visible to the compiler |
| Tuning before a correctness baseline | First compare a fixed configuration with PyTorch NPU across boundary shapes |
| Applying CUDA dtype intuition | Check the A2/A3 Vector dtype limitations and generated IR/profile |
| Using an arbitrary block length in `tl.arange` | Keep the range compile-time and use supported sizes, normally power-of-two candidate tiles |
