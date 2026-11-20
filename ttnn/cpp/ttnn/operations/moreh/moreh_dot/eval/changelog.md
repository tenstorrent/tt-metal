# moreh_dot changelog

## Adopted

2026-11-20, by `/adopt-op` for extend and reiterate runs.

- **Origin:** the existing C++ ttnn op `ttnn.moreh_dot` (`ttnn/cpp/ttnn/operations/moreh/moreh_dot/`): one program factory, single core, three kernels. Deprecated in favor of `ttnn.matmul` (#62325) and scheduled for removal after 2026-11-26; adopted as a test vehicle for the C++ path.
- **Supplied by the user:** the reference (`torch.dot` of the flattened operands), the perf case (`w2048`: `[1, 1, 1, 2048]` bf16 TILE, DRAM interleaved, `torch.rand` inputs, seed 0), the tolerance (relative error 5%) and the guarded shapes (the unit test's W = 10, 32, 323, 352, plus `[1, 323]` and the perf shape).
- **Derived:**
  - Registry (`eval/golden_tests/moreh_dot/registry.py`), from the op's `TT_FATAL` checks and a device probe: input dtype bf16; output dtype bf16, float32 or int32 (the last two through `ttnn.typecast`); TILE layout; rank 4.
  - Not supported: float32 and bf8b inputs (rejected by `TT_FATAL`); ROW_MAJOR input (not checked by the op; hangs the device); rank 2 (admitted by the op's check, then the factory raises `RuntimeError` on the missing dims; the golden suite expects that).
  - Accuracy: the op accumulates in bf16. Relative error at W = 2048 over 20 seeds: 2.3% median, 3.8% max (0.45% / 1.2% at W = 512).
  - Guard test: `tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_dot.py`, about 7 s per phase.
- **Baseline:** `w2048` (`78c6412933a7`) 39,264 ns, median of 5, Blackhole p100a.
