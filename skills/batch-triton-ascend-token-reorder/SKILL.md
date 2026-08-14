---
name: batch-triton-ascend-token-reorder
description: Batch MoE token reordering in Triton-Ascend with contiguous index loads, random source-row reads, extension.insert_slice assembly in UB, and coalesced output stores. Use when an agent must implement output[j] = input[indices[j]] on Ascend NPU and one-program-per-token GPU tiling creates too many logical cores or scattered writes.
---

# Batch Triton-Ascend Token Reorder

## Goal

Gather several indexed token rows per program, assemble them in an on-chip buffer, and write one contiguous output tile.

## Workflow

1. Treat input and output as S by D and indices as a length-S integer vector.
2. Assign BLOCK_SIZE consecutive output rows to each program.
3. Load their indices as one vector.
4. Loop over valid rows, load each indexed D-element source row, and insert it into a BLOCK_SIZE by D UB tensor.
5. Store the assembled tile with row and column masks.

## Implementation Pattern

~~~python
import triton.language.extra.cann.extension as extension

rows = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
row_mask = rows < S
selected = tl.load(indices + rows, mask=row_mask, other=0)
tile = tl.zeros((BLOCK_SIZE, D), dtype=x_ptr.dtype.element_ty)

for i in range(0, BLOCK_SIZE):
    if pid * BLOCK_SIZE + i < S:
        source_row = extension.get_element(selected, (i,))
        cols = tl.arange(0, D)
        values = tl.load(x_ptr + source_row * D + cols)
        tile = extension.insert_slice(
            tile, values[None, :], (i, 0), (1, D), (1, 1)
        )

cols = tl.arange(0, D)
out_offsets = rows[:, None] * D + cols[None, :]
tl.store(out_ptr + out_offsets, tile, mask=row_mask[:, None])
~~~

## Ascend Guardrails

- Use the extension namespace; current APIs are extension.insert_slice and extension.get_element.
- Validate every selected index as 0 <= index < input_rows before loading.
- Compute the output row offset as row * D; omitting the multiplication overlaps rows.
- Size BLOCK_SIZE by the UB cost of the full BLOCK_SIZE by D tile and temporary row.
- Require D to be a legal compile-time tile or add padded columns and column masks.
- Use this pattern when output writes are contiguous; it does not remove random source reads.

## Verification

Compare with input[indices] for permutations and repeated indices. Include a partial final block and invalid-index tests that must fail before kernel launch.
