---
name: prevent-triton-ascend-ub-overflow
description: Prevent Triton-Ascend Unified Buffer overflow by streaming large logical vectors through bounded compile-time tiles with masked loop iterations. Use when an agent ports a GPU kernel that loads a long sequence or metadata array at once, compilation reports UB pressure, or runtime shapes such as KV-page allocation exceed safe on-chip capacity.
---

# Prevent Triton-Ascend UB Overflow

## Goal

Keep each live tile within Ascend UB while preserving the semantics of a larger logical operation.

## Workflow

1. List all simultaneously live tensors and estimate their bytes, including padding and multibuffer copies.
2. Select a BLOCK_SIZE that fits the UB budget with safety margin.
3. Replace one load over the maximum logical length with a compile-time loop over BLOCK_SIZE tiles.
4. Recompute global offsets from the loop index and apply a runtime validity mask.
5. Store each completed tile before advancing so earlier tiles do not remain live.

## Implementation Pattern

~~~python
block_offsets = tl.arange(0, BLOCK_SIZE)
num_loops = tl.cdiv(MAX_EXTEND_TOKENS, BLOCK_SIZE)

for i in range(num_loops):
    token_offsets = i * BLOCK_SIZE + block_offsets
    mask = token_offsets < valid_tokens
    pages = tl.load(
        free_pages + page_start + token_offsets // page_size,
        mask=mask,
    )
    slots = pages * page_size + token_offsets % page_size
    tl.store(output + output_start + token_offsets, slots, mask=mask)
~~~

## Ascend Guardrails

- Base BLOCK_SIZE on bytes of all live values, not only the primary input.
- Account for 32-byte padding, float32 promotions, temporary results, and multibuffering.
- Keep MAX_EXTEND_TOKENS a compile-time upper bound and valid_tokens the runtime limit.
- Apply the same tail condition to every load and store.
- Avoid retaining a full logical result in a tl.zeros tensor after tiling; stream completed outputs to global memory.
- Re-evaluate the budget whenever dtype, tile rank, or compiler options change.

## Verification

Test valid lengths around BLOCK_SIZE boundaries and the maximum supported length. Compare with the untiled reference and profile memory/compiler behavior for progressively larger inputs.
