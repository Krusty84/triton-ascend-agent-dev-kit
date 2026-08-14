# Cube-Vector Fusion Development

> Load this section when a single operator combines `tl.dot`/matrix work with activation, softmax, reduction, masking, layout reorganization, or other Vector work.

## What Makes a CV Kernel

| Layer | Typical work | Main resource |
| --- | --- | --- |
| Matrix layer | GEMM, QK, PV, convolution-like `tl.dot` | Cube/AI Core |
| Vector layer | bias, activation, scale, mask, reduction, softmax, normalization | Vector Core |
| Memory layer | cat/slice, transpose, KV-cache or token reorganization | MTE + UB/Vector |
| Pipeline layer | overlap, buffering, synchronization, workspace | Cube + Vector + MTE |

CV fusion is useful when it removes material global-memory round trips or kernel boundaries. It is not automatically faster: the fused live set can overflow UB/L1, and one resource may wait for another.

## Simple CV Epilogue

Start with a verified Cube kernel. Add only tile-local Vector work before the store:

```python
# acc is the fp32 result of the K-loop tl.dot accumulation.
acc = tl.where(acc >= 0, acc, 0.01 * acc)
c = acc.to(tl.float16)

offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
tl.store(c_ptrs, c, mask=c_mask)
```

Good first fusions are bias, scale, a simple activation, and output conversion. Keep the interface between layers clear: Cube produces a two-dimensional accumulator; Vector consumes that tile locally.

## When to Split Instead

Split the operation into kernels when:

- Vector work needs state shared across many Cube output tiles;
- cross-core ordering/synchronization is complex or unproven;
- the combined accumulator and Vector intermediates exceed on-chip capacity;
- fusion makes tiling constraints incompatible;
- profiling shows Cube or Vector idle time increases more than global-memory traffic decreases.

Workspace or explicit synchronization may be necessary for multi-tile state, but it should be introduced only with a documented ownership and ordering model.

## Complex CV Design by Data Flow

1. **Validate Cube stages.** Verify QK/PV/GEMM shapes, strides, accumulator dtype, and output tiles without the Vector epilogue.
2. **Add tile-local Vector work.** Introduce masks, scaling, activation, or reductions one step at a time.
3. **Control the Vector live set.** For a large accumulator, process slices with ordinary loops and `extract_slice`/`insert_slice`-style operations.
4. **Reorganize discrete data.** Turn irregular KV-cache, MoE, cat/slice, or short-tail access into contiguous UB blocks where possible.
5. **Add pipeline tuning.** Explore buffering and CV-balance options only after the data flow is correct.
6. **Profile balance.** Inspect Cube, Vector, and MTE time rather than only end-to-end latency.

## Attention-Style Fusion

Build complexity in this order:

1. non-causal, short sequence, small head dimension;
2. causal/other masks;
3. K/V loops for long sequences;
4. numerically stable running `m_i`/`l_i` softmax updates;
5. workspace or accumulator slicing for large `HEAD_DIM`;
6. reorganization for discrete KV-cache indices.

For online softmax, invalid elements must not affect the max or sum. Use the correct neutral values and validate rows containing only masked positions according to the public operator contract.

## Core Allocation

Launch CV kernels around `num_aicore`. The source model pairs each Cube Core with two Vector Cores, so a CV launch is Cube-oriented while Vector resources cooperate. Do not reuse a large GPU output-tile grid without either an inner tile loop or applicable Auto-Blockify.

## Relevant Tuning Options

| Option | Purpose |
| --- | --- |
| `multibuffer` | overlap data movement and computation; enabled by default in the source guide |
| `enable_hivm_auto_cv_balance` | automatically balance Cube and Vector execution |
| `set_workspace_multibuffer` | buffer workspace movement, commonly with level 2 or 4 |
| `tile_mix_vector_loop` | split the Vector loop, commonly 2/4/8 |
| `tile_mix_cube_loop` | split the Cube loop, commonly 2/4/8 |
| `enable_auto_bind_sub_block` | enable automatic sub-block binding for CV kernels |
| `sync_solver` | compiler synchronization solving for CV kernels |
| `disable_auto_inject_block_sync` | control automatic block synchronization injection |
| `enable_nd2nz_on_vector` | enable ND-to-NZ transformation on Vector work |

Use [Autotuning](./08-autotuning.md) to search combinations. Do not enable every option simultaneously without a controlled search space.

## Explicit Vector Sub-Block Binding

The source guide mentions `extension.parallel(..., bind_sub_block=True)` as an explicit path for binding Vector sub-blocks. It is hardware/compiler dependent and is not the default pattern. Prefer ordinary loop-based sub-tiling unless the installed version's examples and target confirm support.

## Profiling Interpretation

| Symptom | First hypotheses |
| --- | --- |
| Cube waits for Vector | Vector epilogue too large, poor CV balance, too coarse Vector loop |
| Vector waits for MTE | irregular access, tail padding, insufficient transfer overlap |
| MTE dominates | too many small/global transfers, bad layout, missed UB reuse |
| UB overflow after fusion | accumulator plus Vector intermediates/buffers exceed capacity; slice or split |
| Fused kernel slower than two kernels | resource imbalance or reduced tile quality outweighs saved GM traffic |

## CV Acceptance Checklist

- Cube-only baseline is correct.
- Each Vector transformation is tested independently.
- Launch count is based on Cube/AI Cores.
- The 512-byte CV tail-axis alignment rule has been considered.
- Combined UB/L1/workspace budget includes buffering.
- Cross-core synchronization does not assume Auto-Blockify iteration order.
- Profiling confirms that fusion reduces the actual bottleneck.

