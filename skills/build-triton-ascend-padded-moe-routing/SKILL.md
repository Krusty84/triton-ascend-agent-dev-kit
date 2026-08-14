---
name: build-triton-ascend-padded-moe-routing
description: Build padded MoE gather, scatter, and router-weight-gradient kernels for Triton-Ascend using real cumulative bins, padded cumulative bins, UB slice assembly, and vector-core-aware tiling. Use when an agent ports MegaBlocks padded routing, aligns each expert's token segment to a fixed multiple, restores weighted top-k outputs, or maps gradients through padded expert storage.
---

# Build Triton-Ascend Padded MoE Routing

## Goal

Store every expert's real assignments in a separately padded segment, preserve zero-filled gaps, and implement inverse scatter and per-assignment weight gradients.

## Workflow

1. Sort assignments by expert to obtain aligned indices and nondecreasing bin_ids.
2. Build bins from real cumulative counts.
3. Round each expert count to the required padding multiple and build padded_bins from cumulative padded counts.
4. Map sorted position i to offset_in_expert = i - bins[e-1] and padded row = padded_bins[e-1] + offset_in_expert.
5. Gather token indices[i] // TOP_K into that padded row.
6. Scatter real rows back through unique assignment IDs and compute wgrad with the same mapping.

## Implementation Pattern

~~~text
real_start = 0 if expert == 0 else bins[expert - 1]
padded_start = 0 if expert == 0 else padded_bins[expert - 1]
offset_in_expert = sorted_position - real_start
padded_row = padded_start + offset_in_expert
token = indices[sorted_position] // TOP_K
~~~

Use num_vectorcore programs over sorted positions, UB-buffered SUB_BLOCK_SIZE rows, and BLOCK_X feature tiles. Flush gather buffers on expert transitions; extract individual rows for scatter and wgrad.

## Ascend Guardrails

- Require bins and padded_bins to be monotonic, with padded segment lengths no smaller than real counts.
- Allocate gather output with padded_bins[-1] rows and initialize it to zero.
- Never process padding rows as real assignments.
- Define weight indexing explicitly; for assignment-ID semantics, use weights[indices[position]], not a flattened feature address.
- Require unique assignment IDs for non-atomic scatter and wgrad stores.
- Align BLOCK_X for 32-byte transfers and budget UB for slices, indices, float32 temporaries, and multibuffering.
- Set SUB_BLOCK_SIZE=1 when the hidden dimension must be accumulated across column tiles.

## Verification

Compare all operations with a reference loop. Cover experts with zero tokens, counts exactly on and just over the padding multiple, multiple experts within one sub-block, TOP_K reduction, weighted scatter, hidden-size tails, and large routed batches.
