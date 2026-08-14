---
name: fuse-triton-ascend-cat-conv1d
description: Refactor fused causal Conv1d state updates for Triton-Ascend by replacing negative-offset cat emulation and tl.where selection with transposed UB assembly through extension.insert_slice. Use when an agent ports causal_conv1d_update-style kernels, sees discrete scalar loads from negative offsets, suffers 32-byte tail-axis padding, or launches far more logical tasks than Ascend vector cores.
---

# Fuse Triton-Ascend Cat and Causal Conv1d

## Goal

Join convolution state and new tokens in UB, update the cached tail, and compute grouped causal Conv1d without materializing a global concatenation.

## Workflow

1. Preserve the reference contract for x, conv_state, depthwise weights, bias, activation, and state-row indices.
2. Load state and input from nonnegative contiguous addresses.
3. Transpose or reshape them so concatenation occurs along a contiguous flattened UB axis.
4. Insert state followed by new tokens with extension.insert_slice.
5. Derive the new state tail and convolution windows from the assembled tensor.
6. Map work to approximately the available vector-core count and loop over remaining feature tiles inside each program.

## Implementation Pattern

~~~python
import triton.language.extra.cann.extension as extension

state_t = load_state_as_contiguous_transposed_tile()
x_t = load_input_as_contiguous_transposed_tile()
joined = tl.zeros((CAT_LEN * DIM_BLOCK,), dtype=x_ptr.dtype.element_ty)
joined = extension.insert_slice(
    joined, state_t, (0,), (STATE_LEN * DIM_BLOCK,), (1,)
)
joined = extension.insert_slice(
    joined, x_t, (STATE_LEN * DIM_BLOCK,), (SEQ_LEN * DIM_BLOCK,), (1,)
)
~~~

## Ascend Guardrails

- Do not emulate cat with negative pointer offsets plus tl.where; Triton-Ascend may classify it as discrete scalar access.
- Use extension.insert_slice, not deprecated tl.insert_slice.
- Account for Ascend UB padding of the final axis to 32 bytes.
- For short final axes such as 1 or 3, consider a proven reshape-transpose “borrowed axis” layout only when total storage alignment permits it.
- Include transpose cost and padding in the UB budget.
- Query the target vector-core count instead of hardcoding a 40/48-core assumption.
- Verify state updates when state indices repeat; concurrent writes need an explicit ordering policy.

## Verification

Compare output and updated conv_state with the PyTorch reference for activation=None and SiLU, several state/sequence lengths, repeated and unique state indices, short widths, and feature dimensions that trigger alignment padding.
