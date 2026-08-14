---
name: batch-triton-ascend-token-unreorder
description: Batch inverse MoE token reordering in Triton-Ascend with contiguous tile loads, extension.extract_slice row extraction, and indexed output stores. Use when an agent must scatter sequential routed tokens back to output[indices[j]], reduce excessive one-token logical cores, or exploit contiguous reads while accepting dispersed writes on Ascend NPU.
---

# Batch Triton-Ascend Token Unreorder

## Goal

Read a contiguous block of routed token rows once, extract rows in UB, and scatter each row to its indexed destination.

## Workflow

1. Treat input as S by D contiguous routed rows and indices as destination rows.
2. Assign BLOCK_SIZE consecutive input rows to each program.
3. Load a BLOCK_SIZE by D input tile and BLOCK_SIZE indices with masks.
4. Extract each valid row with extension.extract_slice.
5. Store the row at indices[row] * D with a column mask.

## Implementation Pattern

~~~python
import triton.language.extra.cann.extension as extension

rows = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
cols = tl.arange(0, D)
row_mask = rows < S
tile_offsets = rows[:, None] * D + cols[None, :]
tile = tl.load(x_ptr + tile_offsets, mask=row_mask[:, None], other=0.0)
destinations = tl.load(indices + rows, mask=row_mask, other=0)

for i in range(0, BLOCK_SIZE):
    if pid * BLOCK_SIZE + i < S:
        row = extension.extract_slice(tile, (i, 0), (1, D), (1, 1))
        dst = extension.get_element(destinations, (i,))
        tl.store(out_ptr + dst * D + cols, row.reshape(D))
~~~

## Ascend Guardrails

- Use columns spanning D, not BLOCK_SIZE, when constructing the two-dimensional tile.
- Validate every destination index before launch.
- Require destinations to be unique unless an explicit collision policy and atomics are implemented.
- Fit the input tile and temporaries within UB; reduce BLOCK_SIZE or tile D when necessary.
- Use the extension namespace for extract_slice and get_element.

## Verification

For permutation indices, compare with a reference satisfying output[indices] = input. Test a partial block, nontrivial permutations, duplicate-index rejection, and D values near the UB tiling boundary.
