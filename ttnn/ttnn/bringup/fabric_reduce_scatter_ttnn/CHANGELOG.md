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

### program_hash: a custom program hash on the cached descriptors (deepseek_v41_flash_d_p P.19, 2026-10-22)
- What: opt-in `program_hash=False` (needs `cache_descriptors=True`). True: `_fast()` = the same checks, scratch,
  descriptor (create_mesh_program_descriptor, new keyword `hash_key=None`) and generic_op call, but each chip's
  ProgramDescriptor carries `custom_program_hash = hash(salt, key, chip)`, key = everything the program depends on
  except the buffer addresses (group configuration, dim, exact_fp32_add, counters, dtype, page size, padded shapes;
  addresses are runtime arguments, which generic_op patches on a program-cache hit either way), so generic_op skips
  hashing the inline kernel sources, CBs and semaphores of 4 programs on every call; the plan lookup is one dict hit
  (`_FAST_PREP`), and its descriptors are cache entries of their own (dkey + "hashed"). Default False: unchanged.
- Why: host cost per call. 2x2 p150, [1024, 10240] bf16 dim 3 (300 calls, host only, steady state): 0.179 -> 0.134
  ms; generic_op alone on the cached descriptor 0.106 -> 0.083 ms. In the DeepSeek-V4.1 model (33 calls a chunk under
  GLUE_FOLD): 0.305 -> 0.271 ms a call (the model's own host profile, generated/ds41_p19/test_hostsec.py).
- Tests: `tests/unit/test_fabric_rs_program_hash.py` (2x2: both axes x rows / columns x bf16 / exact fp32:
  bit-identical to the default and cache_descriptors paths, deterministic, every hashed program has a custom hash, an
  input in fresh buffers gives 2x the output exactly on the reused program; program_hash without cache_descriptors
  refused). Regression, option off (2026-10-22, 2x2): tests/unit 19 passed / 13 skipped (the 2x4-only files; 9 of the
  passes new), model cases 2 passed (no earlier full-suite count recorded).
- Needed by: deepseek_v41_flash_d_p P.19 (tt/ccl.py, setting GLUE_FOLD).
- Files: fabric_reduce_scatter.py, tests/unit/test_fabric_rs_program_hash.py

### Tests: deepseek_v41_flash_d_p cases (O.1, 2026-10-23)
- What: 6 random-input cases for the calls deepseek_v41_flash_d_p makes (tt/ccl.py sum_cols / tp_sum, 2x2 mesh-row
  groups, num_links 4, cache_descriptors + program_hash, dim 2 and dim 3, bf16 FPU adds and one exact_fp32_add fp32
  call); the test passes dim / exact_fp32_add / cache_descriptors / program_hash, a `noise` case checks rel L2 against
  fork_cases.noise_limit (F139; measured 1.80e-3 vs limit 3.6e-3: one bf16 rounding), the exact case bit for bit vs
  torch's fp32 a + b (0 differences). reference.py: `dim`, `blocks`. No op change.
- Why: task O.1, every call a model makes to a fork gets a case.
- Needed by: deepseek_v41_flash_d_p O.1
- Files: tests/cases.py, tests/reference.py, tests/test_fabric_reduce_scatter_ttnn.py
