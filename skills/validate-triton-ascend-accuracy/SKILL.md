---
name: validate-triton-ascend-accuracy
description: Validate Triton-Ascend kernel outputs against PyTorch references with dtype-aware tolerances, exact integer checks, bfloat16 promotion, and boolean handling. Use when an agent writes or reviews NPU kernel tests, needs a reusable accuracy-comparison helper, or must choose an initial comparison policy for float16, bfloat16, float32, integer, or boolean outputs.
---

# Validate Triton-Ascend Accuracy

## Goal

Make correctness checks explicit and consistent across Triton-Ascend kernel tests.

## Workflow

1. Compute the PyTorch reference from the same inputs before interpreting performance.
2. Assert identical output shape and dtype.
3. Compare floating-point tensors with dtype-aware tolerances.
4. Compare integer and boolean tensors exactly.
5. Add edge shapes, tail masks, zeros, extreme values, and NaNs when the operator semantics allow them.

## Implementation Pattern

~~~python
def assert_ascend_close(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    dtype = actual.dtype

    if dtype == torch.float16:
        torch.testing.assert_close(
            actual, expected, rtol=1e-3, atol=1e-3, equal_nan=True
        )
    elif dtype == torch.bfloat16:
        torch.testing.assert_close(
            actual.float(), expected.float(),
            rtol=1e-3, atol=1e-3, equal_nan=True,
        )
    elif dtype == torch.float32:
        torch.testing.assert_close(
            actual, expected, rtol=1e-4, atol=1e-4, equal_nan=True
        )
    elif dtype in {torch.int8, torch.int16, torch.int32, torch.int64}:
        assert torch.equal(actual, expected)
    elif dtype == torch.bool:
        assert torch.equal(actual.cpu(), expected.cpu())
    else:
        raise ValueError(f"unsupported dtype: {dtype}")
~~~

## Ascend Guardrails

- Create inputs and run both implementations on the intended NPU device; avoid unnecessary transfers during comparison.
- Promote bfloat16 only for the comparison, not for the kernel result.
- Treat the listed tolerances as starting points, not universal guarantees. Tighten or relax them only from the operator's numerical analysis.
- Use equal_nan=True only when matching NaN positions is acceptable for the operator.
- Never let a permissive tolerance hide mismatched shapes, dtypes, infinities, or systematic bias.

## Verification

Include a test that intentionally perturbs the result enough to fail, proving the chosen tolerance detects meaningful errors. For masked kernels, always include a shape that exercises the tail.
