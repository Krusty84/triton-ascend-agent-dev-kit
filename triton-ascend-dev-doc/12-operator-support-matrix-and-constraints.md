# Triton Operator Support Matrix and Constraints

> Load this section before selecting a Triton operation or dtype for an Ascend kernel, and when compilation succeeds but an operation behaves unexpectedly. This is an English translation and agent-oriented reorganization of [`../outline.md`](../outline.md).

## How to Read the Matrix

| Marker | Meaning |
| --- | --- |
| `✓` | Supported for the listed dtype, subject to the constraints below |
| `×` | Not supported for the listed dtype in the source documentation snapshot |
| `✓*` | Supported by converting `bool` to `int8` internally and executing the operation |

This matrix describes functional backend support, not performance. A supported operation may still lower to scalar work or have generation-specific performance limitations. In particular, consult [Memory, Tiling, and Performance](./07-memory-tiling-and-performance.md) for A2/A3 integer Vector limitations.

Treat the matrix as a documentation-snapshot contract. Confirm it against the installed Triton-Ascend version when backend version or target hardware is material.

## Triton Operator Support by Dtype

| Category | Triton Op | int8 | int16 | int32 | uint32 | int64 | fp16 | fp32 | bf16 | bool |
|:---:|:---:|---|---|---|---|---|---|---|---|---|
| Creation Ops | arange | × | × | ✓ | × | × | × | × | × | × |
| | cat | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓ |
| | full | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓ |
| | zeros | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓ |
| | zeros_like | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓ |
| | cast | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓ |
| Shape Manipulation Ops | broadcast | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓ |
| | broadcast_to | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓ |
| | expand_dims | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓ |
| | interleave | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓ |
| | join | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓ |
| | permute | ✓ | ✓ | ✓ | × | × | ✓ | ✓ | ✓ | ✓ |
| | ravel | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓ |
| | reshape | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓ |
| | split | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓ |
| | trans | ✓ | ✓ | ✓ | × | × | ✓ | ✓ | ✓ | ✓ |
| | view | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓ |
| Linear Algebra Ops | dot | ✓ | × | × | × | × | ✓ | ✓ | ✓ | × |
| | dot_scaled | × | × | × | × | × | × | × | × | × |
| Memory/Pointer Ops | load | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓ |
| | store | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓ |
| | make_block_ptr | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | × |
| | make_tensor_descriptor | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | × |
| | load_tensor_descriptor | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | × |
| | store_tensor_descriptor | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | × |
| | advance | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | × |
| Indexing Ops | flip | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓ |
| | where | ✓ | ✓ | ✓ | × | × | ✓ | ✓ | ✓ | ✓* |
| | swizzle2d | ✓ | ✓ | ✓ | × | ✓ | × | × | × | × |
| Math Ops | add | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓* |
| | sub | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓* |
| | mul | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓* |
| | div | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓* |
| | floordiv(//) | ✓ | ✓ | ✓ | × | ✓ | × | × | × | ✓* |
| | mod | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓* |
| | neg | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | × |
| | invert(~) | ✓ | ✓ | ✓ | × | ✓ | × | × | × | ✓ |
| | and(&) | ✓ | ✓ | ✓ | × | ✓ | × | × | × | ✓ |
| | or(\|) | ✓ | ✓ | ✓ | × | ✓ | × | × | × | ✓ |
| | xor(^) | ✓ | ✓ | ✓ | × | ✓ | × | × | × | ✓ |
| | not(!) | ✓ | ✓ | ✓ | × | ✓ | × | × | × | ✓ |
| | lshift(<<) | ✓ | ✓ | ✓ | × | ✓ | × | × | × | × |
| | rshift(>>) | ✓ | ✓ | ✓ | × | ✓ | × | × | × | × |
| | gt | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓* |
| | ge | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓* |
| | lt | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓* |
| | le | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓* |
| | eq | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓* |
| | ne | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓* |
| | logical and | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| | logical or | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| | abs | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓* |
| | cdiv | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | × |
| | ceil | × | × | × | × | × | ✓ | ✓ | ✓ | × |
| | clamp | × | × | × | × | × | ✓ | ✓ | ✓ | × |
| | cos | × | × | × | × | × | ✓ | ✓ | ✓ | × |
| | div_rn | × | × | × | × | × | ✓ | ✓ | ✓ | × |
| | erf | × | × | × | × | × | ✓ | ✓ | ✓ | × |
| | exp | × | × | × | × | × | ✓ | ✓ | ✓ | × |
| | exp2 | × | × | × | × | × | ✓ | ✓ | ✓ | × |
| | fdiv | × | × | × | × | × | ✓ | ✓ | ✓ | × |
| | floor | × | × | × | × | × | ✓ | ✓ | ✓ | × |
| | fma | × | × | × | × | × | ✓ | ✓ | ✓ | × |
| | log | × | × | × | × | × | ✓ | ✓ | ✓ | × |
| | log2 | × | × | × | × | × | ✓ | ✓ | ✓ | × |
| | maximum | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓* |
| | minimum | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓* |
| | round | × | × | × | × | × | × | ✓ | × | × |
| | rsqrt | × | × | × | × | × | ✓ | ✓ | ✓ | × |
| | sigmoid | × | × | × | × | × | ✓ | ✓ | ✓ | × |
| | sin | × | × | × | × | × | ✓ | ✓ | ✓ | × |
| | softmax | × | × | × | × | × | ✓ | ✓ | ✓ | × |
| | sqrt | × | × | × | × | × | ✓ | ✓ | ✓ | × |
| | sqrt_rn | × | × | × | × | × | ✓ | ✓ | ✓ | × |
| | umulhi | × | × | ✓ | × | × | × | × | × | × |
| Reduction Ops | argmax | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | × |
| | argmin | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | × |
| | max | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓* |
| | min | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓* |
| | reduce | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓* |
| | sum | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓* |
| | xor_sum | ✓ | ✓ | ✓ | × | ✓ | × | × | × | ✓* |
| Scan/Sort Ops | associative_scan | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | ✓ |
| | cumprod | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓ |
| | cumsum | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓ |
| | histogram | × | × | ✓ | ✓ | ✓ | × | × | × | × |
| | sort | × | × | × | × | × | × | × | × | × |
| | gather | × | × | × | × | × | ✓ | ✓ | ✓ | × |
| Atomic Ops | atomic_add | ✓ | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | × |
| | atomic_and | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | × |
| | atomic_cas | × | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | × |
| | atomic_max | ✓ | ✓ | ✓ | × | × | ✓ | ✓ | ✓ | × |
| | atomic_min | ✓ | ✓ | ✓ | × | × | ✓ | ✓ | ✓ | × |
| | atomic_or | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | × |
| | atomic_xchg | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | × |
| | atomic_xor | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | × |
| Random Number Generation | randint4x | ✓ | ✓ | ✓ | ✓ | × | × | × | × | ✓ |
| | randint | ✓ | ✓ | ✓ | ✓ | × | × | × | × | ✓ |
| | rand | × | × | × | × | × | ✓ | ✓ | ✓ | ✓ |
| | randn | × | × | × | × | × | ✓ | ✓ | ✓ | ✓ |
| Iterators | range | ✓ | ✓ | ✓ | × | ✓ | × | × | × | × |
| | static_range | ✓ | ✓ | ✓ | × | ✓ | × | × | × | × |
| Inline Assembly | inline_asm_elementwise | × | × | × | × | × | × | × | × | × |
| Compiler Hint Ops | assume | × | × | × | × | × | × | × | × | ✓ |
| | debug_barrier | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓ |
| | max_constancy | × | × | × | × | × | × | × | × | × |
| | max_contiguous | × | × | × | × | × | × | × | × | × |
| | multiple_of | × | × | × | × | × | × | × | × | × |
| Debug Ops | static_print | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓ |
| | static_assert | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓ |
| | device_print | ✓ | ✓ | ✓ | × | ✓ | ✓ | ✓ | × | ✓ |
| | device_assert | × | × | × | × | × | × | × | × | ✓ |

## Operation-Specific Constraints

### Linear Algebra and Indexing

- `dot`: inputs must have the forms `A[batch (optional), M, K]` and `B[batch (optional), K, N]`.
- `gather`: for `triton.gather(x, index, axis)`, if `x` has rank `n`, only `axis = n - 1` is currently supported.
- `permute`: `triton.permute(x, dims)` does not support `dims=[2, 1, 0]`. More generally, non-adjacent-axis transpositions such as `(0, 1, 2) -> (2, 1, 0)` are unsupported.
- `trans`: `triton.trans(x, dims)` does not support `dims=[2, 1, 0]`. More generally, non-adjacent-axis transpositions such as `(0, 1, 2) -> (2, 1, 0)` are unsupported.

### Debug Operations

- `device_print` requires `TRITON_DEVICE_PRINT=1`.
- `device_assert` requires both `TRITON_DEBUG=1` and `TRITON_DEVICE_PRINT=1` to take effect.

### Atomic Operations

- `atomic_add`: Ascend does not support using `atomic_add` to implement a multicore add while saving intermediate results. Use ordinary addition when saving the intermediate result.
- Atomic `sem`: the Ascend backend supports only the default `"acq_rel"` mode. Other values are treated as the default.
- Atomic `scope`: the Ascend backend supports only the default `"gpu"` value. Other values are treated as the default.
- `atomic_or`, `atomic_xor`, `atomic_and`, `atomic_xchg`, and `atomic_cas` are not currently supported inside loops on Ascend.

### Arithmetic and Random Operations

- `umulhi` does not support negative inputs.
- `mod` with `int64` supports only values in the range `-2^24` through `2^24`.
- For the `rand` family, the listed supported dtype applies only to the operator output.

### Tensor Descriptors

Tensor-descriptor operations are currently supported only as a bound set: `make_tensor_descriptor`, `load_tensor_descriptor`, and `store_tensor_descriptor` must be used together.

## Kernel-Wide Constraints

- `int8` receives special handling and occupies more on-chip space. It can make UB overflow more likely during compilation; adjusting tiling usually resolves the problem.
- The sum of all tensors simultaneously present in a Triton kernel must not exceed 96 KiB. With double buffering disabled, the documented limit is 192 KiB.
- No tensor shape dimension may have a size smaller than 1.
- `✓*` means Triton internally converts `bool` to `int8`, executes the operation, and returns a result.
- Scalar-tensor computation using shape representation `"[[]]"` is unsupported.

The 96/192 KiB limits above are stated globally in the source document. Because available on-chip storage and compiler buffering can depend on the product and backend version, treat them as conservative snapshot constraints and use the compiler's actual UB diagnostic as the final resource check.

## Agent Decision Rules

1. Check the exact operation/dtype cell before generating code.
2. Apply every relevant operation-specific and kernel-wide constraint even when the matrix cell is `✓`.
3. Do not infer performance from functional support.
4. For unsupported dtype combinations, change the algorithm or dtype only after proving numerical and range safety; do not insert a silent cast.
5. Recheck this matrix when migrating a GPU kernel that uses atomics, gather, arbitrary transpose, tensor descriptors, debug operations, or random-number generation.
6. Validate against the installed Triton-Ascend version and target hardware when a generated kernel depends on a snapshot-specific limitation.

