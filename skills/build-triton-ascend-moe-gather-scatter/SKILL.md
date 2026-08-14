---
name: build-triton-ascend-moe-gather-scatter
description: Build Ascend-friendly MoE gather, scatter, and router-weight-gradient kernels with vector-core-sized grids, UB-aware row/column tiling, and CANN slice extensions. Use when an agent ports MegaBlocks-style unbinned token routing to Triton-Ascend, reorders top-k token assignments, combines weighted expert outputs, or computes per-route weight gradients.
---

# Build Triton-Ascend MoE Gather and Scatter

## Goal

Implement the unbinned routing triplet over assignment IDs: gather token rows into routing order, scatter weighted rows back to tokens, and compute one router-weight gradient per assignment.

## Workflow

1. Define L = tokens * TOP_K and require indices to be a permutation of assignment IDs in [0, L).
2. Gather routed row i from token row indices[i] // TOP_K.
3. Scatter routed row i into assignment slot indices[i], optionally scale by weights[indices[i]], then reduce the TOP_K slots per token.
4. Compute wgrad[indices[i]] as the float32 dot product of routed row i and grads[indices[i] // TOP_K].
5. Split L approximately evenly across the target vector cores.
6. Process SUB_BLOCK_SIZE assignments and BLOCK_X feature columns at a time.

## Implementation Pattern

~~~python
num_cores = get_npu_properties()["num_vectorcore"]
rows_per_core = triton.cdiv(indices.numel(), num_cores)
block_x = round_up_to_16(min(hidden_size, max_block_x))
sub_block = choose_from_ub_budget(block_x, live_buffers)
kernel[(num_cores,)](
    x, out, indices,
    INDICES_LENGTH=indices.numel(),
    BLOCK_SIZE=rows_per_core,
    SUB_BLOCK_SIZE=sub_block,
    NUM_COLUMNS=hidden_size,
    BLOCK_X=block_x,
    TOP_K=top_k,
    multibuffer=True,
)
~~~

Inside gather, assemble SUB_BLOCK_SIZE rows with extension.insert_slice before a contiguous store. Inside scatter, load a contiguous routed tile and use extension.extract_slice for unique indexed stores.

## Ascend Guardrails

- Import extension from triton.language.extra.cann.extension.
- Align BLOCK_X to 16 elements for 32-byte fp16/bfloat16 alignment.
- Derive SUB_BLOCK_SIZE from all UB-resident indices, input tiles, output tiles, float32 temporaries, and multibuffer copies.
- Set SUB_BLOCK_SIZE=1 when hidden-size column tiling requires accumulation across multiple BLOCK_X tiles.
- Require unique assignment IDs before non-atomic scatter stores.
- Apply optional weights by assignment ID, not routed-row position.
- Accumulate scaling and gradient dot products in float32, then cast once on store.

## Verification

Compare all three operations with a NumPy or PyTorch loop. Cover TOP_K=1 and greater, random routing permutations, weighted and unweighted scatter, small unaligned hidden sizes, large column-tiled hidden sizes, and multiple token/expert counts.
