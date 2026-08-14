# Ascend Language and Compiler Reference

> Load this section when an implementation needs Ascend-only language features, compiler options, IR-path reasoning, or Ascend 950 compilation modes.

## Language Extensions

The architecture document lists:

| API | Purpose |
| --- | --- |
| `tl.insert_slice(full, src, offsets, sizes, strides)` | insert a source tensor slice into a target tensor |
| `tl.extract_slice(full, offsets, sizes, strides)` | extract a tensor slice |
| `tl.get_element(source, offset)` | read one element from a tensor at a multidimensional offset |
| `tl.custom_op` | Ascend-specific operations such as index select/put, UB gather/scatter, indirect load/store |
| `tl.compile_hint` | pass hardware-specific optimization/resource hints to the backend |
| `tl.sync_block_set(sender, receiver, event_id)` | signal a cross-block event |
| `tl.sync_block_wait(sender, receiver, event_id)` | wait for a cross-block event |
| `tl.sync_block_all(mode, event_id)` | global/collective block synchronization |

Complex operator examples also import:

```python
import triton.language.extra.cann.extension as extension
```

and use `extension.insert_slice` / `extension.extract_slice`. Namespace availability can vary with the installed version; follow its tests and exported API rather than assuming both spellings are aliases.

Use synchronization extensions only with a defined producer/consumer mapping and event lifecycle. They are not drop-in equivalents of CUDA `__syncthreads()`.

For functional support of standard Triton operations by dtype—and backend constraints for `dot`, `gather`, atomics, tensor descriptors, debug operations, and kernel-wide on-chip storage—use the [Operator Support Matrix and Constraints](./12-operator-support-matrix-and-constraints.md).

## Compiler Options Relevant to Kernel Authors

| Option | Scope | Meaning |
| --- | --- | --- |
| `multibuffer` | general | enable/disable ping-pong pipeline; source default is enabled |
| `enable_auto_bind_sub_block` | CV | automatic sub-block binding |
| `enable_hivm_auto_cv_balance` | CV | automatic Cube/Vector balance |
| `sync_solver` | CV | synchronization solver |
| `unit_flag` | Cube/CV | synchronization/cube-out optimization |
| `inject_barrier_all` | general | automatically inject barriers for operations |
| `inject_block_all` | general | automatically inject blocks for operations |
| `limit_auto_multi_buffer_only_for_local_buffer` | CV | restrict auto multibuffering to local buffers |
| `limit_auto_multi_buffer_of_local_buffer` | Cube/CV | scope of local-buffer multibuffering, such as `"no-limit"` or `"no-l0c"` |
| `set_workspace_multibuffer` | CV | workspace multibuffer level, commonly 2 or 4 |
| `tile_mix_vector_loop` | CV | number of Vector-loop splits, commonly 2/4/8 |
| `tile_mix_cube_loop` | CV | number of Cube-loop splits, commonly 2/4/8 |
| `disable_auto_inject_block_sync` | CV | control automatic block-sync injection |
| `stream` | runtime/compiler | inform compiler of the NPU stream |
| `enable_linearize` | general | enable/disable linearization pass |
| `enable_nd2nz_on_vector` | CV | ND-to-NZ transformation on Vector path |
| `auto_blockify_size` | Auto-Blockify | leftmost expansion size; relevant when all-blocks-parallel mode is enabled |
| `enable_auto_blockify` | Auto-Blockify | per-kernel override: option > environment > disabled |
| `enable_ubuf_saving` | Vector/CV tuning | enable UB-saving optimization |
| `compile_mode` | Ascend 950 | `"unstructured_in_simt"`, `"simd"`, or `"simt_only"` |

Not every option is supported by every `max_autotune` kernel type. Use the exact support matrix in [Autotuning](./08-autotuning.md#supported-parameter-matrix).

## SIMD Lowering and Coding Implications

The SIMD compiler contains these conceptual stages:

1. structured pointer/mask rewriting;
2. discrete-mask analysis;
3. unstructured-access lowering;
4. TTIR-to-Linalg conversion;
5. Ascend-specific HIVM/HFusion/LLVM lowering.

### Structured pointer limitations

Rewrites can eliminate integer division/modulo from pointer and mask expressions, but the source documents limitations:

- the original iteration axis must divide cleanly by the split divisor;
- outer block size must be an integer multiple or divisor of that split;
- complex non-standard mask expressions may not be reconstructed;
- pointer paths with conditional branches inside some loops are unsupported.

Coding implication: express multidimensional offsets with explicit axes/strides and simple per-axis masks. Avoid unnecessarily entangling `//`, `%`, branches, and pointer arithmetic.

### Discrete access

On pure SIMD, non-contiguous masked access may be rewritten into load/select/store sequences and unstructured accesses may expand to scalar loops. Bubble-up passes can eliminate some unnecessary loops, but no optimization is guaranteed.

Coding implication: expose contiguous regions, use UB-local gather/scatter when beneficial, and inspect IR for performance-critical irregular kernels.

### Key lowerings

The architecture document maps common operations approximately as follows:

| Triton operation | Lowering direction |
| --- | --- |
| load/store | memref copy plus tensor/bufferization operations |
| add pointer / tensor pointer | memref reinterpret casts |
| `tl.program_id` / number of programs | function parameters |
| atomics | Linalg generic or custom atomic handling |
| `tl.arange` / splat / reshape / broadcast / transpose | Linalg/arith/tensor transformations |
| `tl.dot` / scaled dot | Linalg matmul path |
| block synchronization | HIVM synchronization operations |
| inline assembly | LLVM inline assembly / CCE intrinsic path |

## Ascend 950 SIMD/SIMT Modes

### `unstructured_in_simt` (default)

- structured kernel regions remain on the SIMD/Linalg path;
- eligible non-contiguous/discrete accesses of rank at most 5 are marked for SIMT indirect load/store;
- unsupported cases fall back to scalar loops;
- atomics can use an indirect atomic custom path.

### `simd`

- all work follows the SIMD lowering path;
- discrete access is analyzed and may expand to scalar loops.

### `simt_only`

- Triton IR is sent directly toward pure SIMT compilation;
- the standard Linalg path is skipped;
- pass `num_warps` deliberately and validate the complete kernel.

```python
kernel[grid](..., compile_mode="unstructured_in_simt")
kernel[grid](..., compile_mode="simd")
kernel[grid](..., compile_mode="simt_only", num_warps=32)
```

## Auto-Blockify Semantics

The compiler and runtime share the resolved enable flag:

```text
per-kernel enable_auto_blockify
    > TRITON_ALL_BLOCKS_PARALLEL
    > disabled
```

When enabled, the compiler creates an internal loop over chunks of logical blocks and the runtime caps the physical launch. This pairing prevents compiling for one mapping and launching with another.

Constraints:

- logical block iterations must be order-independent;
- workspace lifetime changes from logical-block to physical-core reuse;
- cross-block synchronization must not depend on a strict logical order.

## IR Inspection

For a performance-critical lowering problem, the source suggests:

```bash
export TRITON_DEBUG=1
```

Then inspect cached `.ttadapter` output and compile/print relevant BiSheng/HIVM IR using the tools and target flags shipped with the installed CANN/Triton-Ascend version. Look for:

- pure scalar transfer or computation where SIMD was expected;
- unstructured loads/stores expanded into loops;
- unexpected padding or layout conversions;
- synchronization that serializes MTE and Vector work;
- missing multibuffer stages.

Do not copy a hardcoded target such as `Ascend910B3` into automation without matching the actual device.
