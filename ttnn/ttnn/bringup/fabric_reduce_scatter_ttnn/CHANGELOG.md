# fabric_reduce_scatter_ttnn (ttnn.bringup.fabric_reduce_scatter, Python op)

- Source: `ttnn/ttnn/operations/examples/fabric_reduce_scatter` on `mstaletovic/mimo-v2-dp` @ `b45a923d73c`. New op: no
  TTNN source on this tree (no upstream tests to carry).
- Python: `ttnn.bringup.fabric_reduce_scatter` (PYTHON_OPS in `ttnn/ttnn/bringup/__init__.py`), planned with
  fabric_all_gather_ttnn's `plan()`.
- Used by: glm53_flash_d_p (tt/common.py:scatter_rows, tt/kda_attention.py, tt/experts_ag.py); deepseek_v41_flash_d_p (tt/ccl.py, CCL_IMPL=fabric, P.5)
- Tests: `tests/unit/` (the op's own suite); model cases `tests/test_fabric_reduce_scatter_ttnn.py`.

## Changes

<!-- One entry per change, newest last:
### <short title>
- What: the change, and the switch or argument that turns it on (default = source behaviour).
- Why: the symptom it fixes or the feature it adds.
- Needed by: <model> <task>
- Files: <paths inside this folder>
-->

### Tests: fork test suite with glm53_flash_d_p cases (O.1, 2026-10-13)
- Model-case suite `tests/cases.py`, `tests/reference.py`, `tests/test_fabric_reduce_scatter_ttnn.py`: GLM's two calls (full-mesh snake ring, and axis 1), vs a float64 sum, plus determinism. No op change.

### Exact fp32 adds, column blocks, a load-time prepare and a descriptor cache (deepseek_v41_flash_d_p P.5, 2026-10-18)
- What: four opt-in additions, every default = the previous program and behaviour (the off path builds the same kernel
  sources, compile-time / runtime arguments, CBs and compute config as before):
  - `exact_fp32_add=False`: True = FLOAT32 partials only; the port and final cores' compute kernel
    (`_EXACT_ADD_COMPUTE`) unpacks every operand straight into an fp32 DEST (UnpackToDestFp32 on c_0 / c_1 / c_2,
    fp32_dest_acc_en) and adds on the SFPU (`add_binary_tile`), so each add is one IEEE fp32 rounding; the ring
    pass-through copy is exact too. Same chunk schedule and runtime arguments as `_ADD_COMPUTE`.
  - `dim=2`: 3 = blocks of columns (input [1, 1, S, G Wb], chip p gets columns [p Wb, (p+1) Wb) of the sum, as
    ttnn.reduce_scatter(dim=3)); only the readers' input reads are remapped (`_col_reader`: block-local page
    q = r Wb + c -> input page r Wt + j Wb + c, read page by page); scratch, output and counters stay block-local.
  - `prepare(mesh, cluster_axis=, topology=, num_links=, scheme=, placement=)`: builds (and caches) the plan and the
    global semaphores, incl. the one-time `synchronize_device`, so a model can do it at load time instead of in its
    first forward pass (fabric_reduce_scatter calls it; same caches as before).
  - `cache_descriptors=False`: True = an LRU (512) of mesh program descriptors keyed by the group configuration, dim,
    exact_fp32_add, the counters' addresses, dtype, page size, padded shapes and the input / scratch / output buffer
    addresses: the Python descriptor build (0.655 ms of 0.85 ms host per call, 2x2, [1024, 5120] fp32) is skipped
    when a steady-state loop's allocations repeat; generic_op ~0.1 ms.
- Why: the default adds use the FPU (`add_tiles`) at the default compute config (fp32 DEST off): on fp32 partials the
  sum is rel 3.3e-3 off the float64 sum (randn x 2^U[-12, 12], 2x2, tests/unit/test_fabric_rs_exact_fp32.py),
  bf16-level, not even TF32 (K-30). DeepSeek-V4.1's TP partial sums must be exact fp32 (today: all_gather + SFPU
  ttnn.add, then mesh_partition to the chip's hidden columns); with both options one call replaces that chain:
  [1024, 5120] fp32 over axis 1 of 2x2, device 0.150 ms at 4 links (0.235 at 2) vs 0.668 ms for the stock chain,
  bit-identical to it.
- Tests: `tests/unit/test_fabric_rs_exact_fp32.py` (7 cases, 2x2): pairs on axis 0 / 1 x rows / columns bit-identical
  to torch's fp32 a + b and to all_gather + ttnn.add (rel vs float64 2.5e-8), the default FPU path on the same inputs
  not bit-exact (rel 3.3e-3; the negative control of the checker); the whole mesh (snake line of 4, three adds) within
  3 fp32 roundings of the float64 sum (worst 0.95 / 0.93 of the bound; FPU path 8.2e4 x the bound); refusals (bf16
  with exact_fp32_add; 32 columns over 2 chips with dim=3).
- Needed by: deepseek_v41_flash_d_p P.5 (tt/ccl.py, CCL_IMPL=fabric: sum_cols = dim 3 + exact; tp_sum = dim 2 +
  exact + fabric_all_gather; prepare at load; cache_descriptors=True)
- Files: fabric_reduce_scatter.py, __init__.py (exports prepare), tests/unit/test_fabric_rs_exact_fp32.py
