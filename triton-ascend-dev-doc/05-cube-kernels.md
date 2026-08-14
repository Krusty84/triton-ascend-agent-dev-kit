# Cube Kernel Development

> Load this section for GEMM, batched/grouped matmul, attention QK/PV stages, and other kernels whose main operation is `tl.dot`.

## Classification

Cube kernels use matrix-oriented compute as their main workload. Organize concurrent work around `num_aicore` (Cube/AI Core count). If substantial activation, reduction, softmax, masking, or layout work follows `tl.dot`, also load [Cube-Vector Fusion](./06-cube-vector-fusion.md).

The [Operator Support Matrix](./12-operator-support-matrix-and-constraints.md) documents `dot` for `int8`, `fp16`, `fp32`, and `bf16`, with input forms `A[batch (optional), M, K]` and `B[batch (optional), K, N]`. Check the installed backend before relying on this snapshot-specific support.

## Canonical GEMM Structure

For `C[M, N] = A[M, K] @ B[K, N]`:

1. map one logical task to an output `(M, N)` tile;
2. construct two-dimensional A/B offsets from explicit strides;
3. loop over K using `BLOCK_K`;
4. use zero for masked K/M/N input lanes;
5. accumulate in `tl.float32` when supported by the required accuracy/performance;
6. cast and store with an output boundary mask.

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr,
                  BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(a, b, acc)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)
```

This simple form uses one program per logical output tile. If the output tile count is much larger than physical Cube/AI Core count, flatten `(pid_m, pid_n)` into `tile_id` and process `range(pid, num_output_tiles, num_core)` inside the kernel.

## Physical-Core GEMM Scheduling

```python
pid = tl.program_id(0)
num_core = tl.num_programs(0)
num_pid_m = tl.cdiv(M, BLOCK_M)
num_pid_n = tl.cdiv(N, BLOCK_N)
num_tiles = num_pid_m * num_pid_n

for tile_id in range(pid, num_tiles, num_core):
    pid_m = tile_id // num_pid_n
    pid_n = tile_id % num_pid_n
    # Build tile offsets and run the K loop.
```

Launch with `(num_aicore,)`. Ensure output tiles are independent. For grouped ordering or cache locality, preserve the intended logical ordering when reconstructing `pid_m/pid_n`.

## Batched and Higher-Dimensional GEMM

Flatten batch into the task space when using physical-core scheduling:

```python
tiles_per_batch = num_pid_m * num_pid_n
batch = tile_id // tiles_per_batch
mn_tile = tile_id % tiles_per_batch
pid_m = mn_tile // num_pid_n
pid_n = mn_tile % num_pid_n

a_batch_ptr = a_ptr + batch * stride_ab
b_batch_ptr = b_ptr + batch * stride_bb
c_batch_ptr = c_ptr + batch * stride_cb
```

Explicit batch strides are safer than assuming tightly packed `[B, M, K]`/`[B, K, N]` tensors.

## Tile Selection

Tune `BLOCK_M`, `BLOCK_N`, and `BLOCK_K` together. They determine:

- input tile movement;
- accumulator size;
- Cube utilization;
- tail waste;
- UB/L1 pressure;
- K-loop depth and multibuffer opportunity.

Do not copy CUDA Tensor Core tile values without measurement. A valid GPU configuration can overflow Ascend on-chip storage or map poorly to Cube data movement.

## Complex Cube Kernels

For attention, grouped matmul, discrete KV cache, or irregular shapes:

1. isolate and validate the pure `tl.dot` stage first;
2. keep the contiguous dimension visible in shape/stride expressions;
3. reorganize discrete K/V loads in UB instead of issuing many scalar global loads when feasible;
4. loop over long K or sequence dimensions to cap on-chip usage;
5. establish stable boundaries for max/sum/exp normalization;
6. classify the kernel as CV when Vector work becomes material;
7. tune `BLOCK_M/N/K`, buffering, and CV options only after correctness.

For Ascend 950, benchmark hybrid SIMT for discrete cache indices while keeping the matrix computation on the SIMD/Cube path.

## Cube Correctness Checklist

- M, N, and K tails use the correct masks.
- Masked K lanes load zero.
- Accumulator dtype meets the accuracy target.
- Output cast is intentional.
- Strides describe the actual layout, including batch/group dimensions.
- Logical tile flattening covers every output exactly once.
- The launch uses Cube/AI Core count when persistent scheduling is used.
- The accumulator plus A/B tiles and pipeline buffers fit on chip.
