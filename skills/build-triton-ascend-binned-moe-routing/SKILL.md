---
name: build-triton-ascend-binned-moe-routing
description: Build capacity-limited binned MoE gather, scatter, and router-weight-gradient kernels for Triton-Ascend using sorted expert bins, UB sub-block buffering, and vector-core-aware tiling. Use when an agent ports MegaBlocks binned routing, packs tokens into fixed expert-capacity tensors, restores top-k token outputs, or must handle sub-blocks that cross expert boundaries.
---

# Build Triton-Ascend Binned MoE Routing

## Goal

Map sorted token assignments into a tensor shaped experts by expert_capacity by hidden_size, truncate overflow per expert, and implement the inverse scatter and router-weight gradient.

## Workflow

1. Require bin_ids to be nondecreasing expert IDs aligned with indices.
2. Interpret bins[e] as the inclusive cumulative assignment count through expert e.
3. For gather, compute each assignment's offset within its expert and keep only offsets below expert_capacity.
4. Buffer rows for the current expert in UB; flush them when the expert ID changes inside a sub-block.
5. For scatter, load valid expert-capacity rows, store to unique assignment slots, optionally scale, and reduce TOP_K slots per token.
6. For wgrad, dot each kept expert row with the gradient of indices[i] // TOP_K.

## Implementation Pattern

~~~text
expert_start = 0 if expert == 0 else bins[expert - 1]
offset_in_expert = sorted_position - expert_start
if offset_in_expert < expert_capacity:
    expert_row = expert * expert_capacity + offset_in_expert
    source_token = indices[sorted_position] // TOP_K
~~~

Split the sorted-position range across num_vectorcore programs. Within each program, iterate over SUB_BLOCK_SIZE assignments and BLOCK_X feature tiles; use extension.insert_slice for gather buffers and extension.extract_slice for scatter rows.

## Ascend Guardrails

- Validate bins as monotonic, bins[-1] == len(indices), and bin_ids consistent with bin boundaries.
- Zero-initialize expert output so unused capacity has defined values.
- Flush a partial UB buffer before switching experts and after the final expert in a sub-block.
- Never read or scatter assignments beyond expert_capacity.
- Align BLOCK_X for the input dtype and include bin vectors, UB row buffers, and multibuffering in the memory budget.
- Require indices to identify unique top-k assignment slots for non-atomic scatter.
- Accumulate weighted values and wgrad reductions in float32.

## Verification

Compare gather, scatter, and wgrad with references across empty experts, expert overflow, exact capacity, sub-block expert transitions, TOP_K values, unaligned hidden sizes, and large token counts.
