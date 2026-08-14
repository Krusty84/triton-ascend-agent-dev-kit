---
name: tile-triton-ascend-gather
description: Tile two-dimensional torch.gather(dim=1) kernels for Triton-Ascend so the logical grid stays near the physical vector-core count while each program handles multiple batch rows and loops over K. Use when an agent ports a GPU grid of B by ceil(K/BLOCK_K), observes NPU launch or scheduling overhead, or needs stride-aware indexed row gathers.
---

# Tile Triton-Ascend Gather

## Goal

Compute out[b, k] = x[b, index[b, k]] without launching one logical program for every batch-row and K tile.

## Workflow

1. Query the target num_vectorcore value.
2. Choose BLOCK_B so ceil(B / BLOCK_B) is close to, and not far above, the core count.
3. Launch a one-dimensional grid over batch blocks.
4. Loop over K inside each program in BLOCK_K tiles.
5. Load index tiles, validate them against C, gather x values with explicit strides, and store with combined masks.

## Implementation Pattern

~~~python
@triton.jit
def gather_dim1(x, indices, out, sx_b, sx_c, si_b, si_k, so_b, so_k,
                B, C, K, BLOCK_B: tl.constexpr, BLOCK_K: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
    row_mask = rows < B
    for k_start in tl.range(0, K, BLOCK_K):
        ks = k_start + tl.arange(0, BLOCK_K)
        tile_mask = row_mask[:, None] & (ks[None, :] < K)
        index_offsets = rows[:, None] * si_b + ks[None, :] * si_k
        cols = tl.load(indices + index_offsets, mask=tile_mask, other=0)
        valid = tile_mask & (cols >= 0) & (cols < C)
        values = tl.load(x + rows[:, None] * sx_b + cols * sx_c, mask=valid)
        output_offsets = rows[:, None] * so_b + ks[None, :] * so_k
        tl.store(out + output_offsets, values, mask=valid)
~~~

## Ascend Guardrails

- Do not hardcode 40 or 48 cores; query device properties.
- Choose legal tl.arange extents, commonly powers of two, while keeping the actual grid near num_vectorcore.
- Include C and check both lower and upper index bounds.
- Preserve all input, index, and output strides.
- Balance fewer programs against UB use and loop length; one program should not monopolize excessive work.

## Verification

Compare with torch.gather(x, dim=1, index=indices). Cover B smaller and larger than the core count, K tails, non-contiguous strides, repeated indices, and invalid-index rejection.
