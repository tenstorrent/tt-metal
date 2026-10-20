# ttnn.bringup.fabric_reduce_scatter  (Python op, PYTHON_OPS; plus `ttnn.bringup.fabric_reduce_scatter_ttnn.prepare`)

## What it does
New op (no TTNN source on this tree; from `ttnn/ttnn/operations/examples/fabric_reduce_scatter` on
`mstaletovic/mimo-v2-dp` @ `b45a923d73c`): a reduce-scatter over the fabric, DRAM to DRAM, as one `ttnn.generic_op`
(Python ProgramDescriptor, no C++ build), planned with fabric_all_gather's `plan()`
(`fabric_all_gather_ttnn/fabric_all_gather_py.py`).
- Input per chip: a partial `[1, 1, G * Sb, W]` TILE, DRAM interleaved. Output per chip: its block of the sum over the
  group, `[1, 1, Sb, W]` (as `ttnn.reduce_scatter(dim=2)`); with `dim=3` column blocks (as `ttnn.reduce_scatter(dim=3)`).
  Chip (r, c) owns the block of its row-major rank in the group, whatever order the group travels in.
- Groups (fabric_all_gather's): a mesh axis (independent groups on the other axis), or the whole mesh as a snake;
  line or ring; or two edge-disjoint Hamiltonian cycles on a torus with both sides >= 3 (each reduces half of every
  block).
- Program: per chip, per direction, per link one port core (reader / compute / sender): relays read what upstream sent
  into this chip's scratch, add the own partial, send the sum one hop on. Per chip and link two final cores (each half
  of the link's DRAM banks) sum own partial + both arrivals into the output. Arrival counters every 8th chunk; the
  ready fence between calls is fabric_all_gather's. Scratch `[1, 1, (G + 1) Sb, W]` per chip, cached per shape/dtype.

## Consider it instead of
- `ttnn.reduce_scatter` / `ttnn.experimental.reduce_scatter_minimal_async` on DRAM TILE partials: faster at the
  measured model shapes (Measured), and a whole-mesh reduce in one call (`cluster_axis=None`) instead of axis 0 then 1.
- `ttnn.all_reduce` when every chip only needs its slice; for a replicated sum follow with `ttnn.bringup.fabric_all_gather`.
- Precision only (rule 7: cross-chip sums in bf16): `ttnn.all_gather` + `ttnn.add` (+ `ttnn.mesh_partition`) for exact
  fp32 partial sums: `exact_fp32_add=True` with `dim=3` is bit-identical to that chain at TP 2 and one call; for
  per-step checks or a measured end-to-end gap.

## Options
`fabric_reduce_scatter(input_tensor, *, cluster_axis=0, topology=ttnn.Topology.Linear, num_links=1, scheme="ring",
output=None, placement="auto", dim=2, exact_fp32_add=False, cache_descriptors=False)`

Realistic (use by default):
- `cluster_axis=0`: 0 / 1 a mesh axis; `None` the whole mesh (snake; ring where it closes, e.g. 2 x 4).
- `topology=Linear`: or Ring (a group of 2 is a line).
- `num_links=1`: links per hop.
- `scheme="ring"`: or "dual_cycles" (cluster_axis=None, Ring, torus sides >= 3).
- `output=None`: preallocated `[1, 1, Sb, W]` (same dtype); else allocated.
- `placement="auto"`: port cores by Ethernet-core probe; "simple" (forced under the emulator).
- `dim=2` (opt-in fork addition, 3): column blocks; Wb a multiple of 32.
- `cache_descriptors=False` (opt-in fork addition): LRU (512) of mesh program descriptors keyed by group config, dim,
  exact_fp32_add, counters, dtype, page size, padded shapes and buffer addresses.
- `prepare(mesh_device, *, cluster_axis=0, topology=Linear, num_links=1, scheme="ring", placement="auto")` (opt-in fork
  addition): builds and caches the plan and global semaphores (incl. the one-time `synchronize_device`); call at load
  time.

Precision only: per-step checks or a measured end-to-end gap (rule 7):
- `exact_fp32_add=False` (opt-in fork addition): FLOAT32 partials only; UnpackToDestFp32 into fp32 DEST, SFPU add,
  one IEEE fp32 rounding per add. Default adds on the FPU at fp32 DEST off.

## Measured
- DeepSeek-V4.1, 2x2 BH, [1024, 5120] fp32 over axis 1, `dim=3` + `exact_fp32_add`: device 0.150 ms at 4 links (0.235
  at 2) vs 0.668 ms for the stock all_gather + add + mesh_partition chain, bit-identical (`CHANGELOG.md`). Default FPU
  add on fp32 partials: rel 3.3e-3 vs float64; exact: rel 2.5e-8 (`CHANGELOG.md`,
  `tests/unit/test_fabric_rs_exact_fp32.py`). Host: descriptor build 0.655 of 0.85 ms per call, generic_op ~0.1 ms
  with `cache_descriptors` (`CHANGELOG.md`); 0.17-0.23 ms host (`models/demos/deepseek_v41_flash_d_p/bringup/
  known_issues.md`, DS41-66). bf16 input: RS 0.080 ms at [1024, 5120] (DS41-67). Per call stock -> fabric (BREADCRUMBS.md
  P.5 / P.6): q_a tp_sum [1024, 1792] 0.321 -> 0.106 ms, compressor [1024, 1024] 0.165 -> 0.065.
- GLM-5.3, LoudBox 2x4 p150b, per chip [5120, 4096] bf16, 2 links, 8192 B payload: one full-mesh ring 427.7 us vs axis 0
  then 1 548.6 us (`models/demos/glm53_flash_d_p_lb/bringup/BREADCRUMBS.md`, "Full-mesh reduce-scatter as one ring";
  `tests/unit/test_fabric_rs_model_shapes.py`). ag MoE block's RS 0.35 ms (same file, "Whole model on the LoudBox").
  scatter_rows on the fork (bf16): warm 56k prefill 9.34 -> 8.95 s; KDA output RS: 8.96 -> 8.47 s (same file).
- Model-case accuracy (2x2, seed 0, vs float64): full-mesh ring PCC 0.9999970, rel 3.03e-3; axis 1 PCC 0.9999987, rel
  1.80e-3 (`tests/cases.py`).

## Limits
- TILE, `[1, 1, R, W]`, DRAM interleaved only. dim 2: R % (32 G) == 0; dim 3: R % 32 == 0 and W % (32 G) == 0.
- `exact_fp32_add` refuses non-FLOAT32 input. The default FPU add on fp32 is bf16-level (DS41-65).
- Needs room for two final cores per link lane ("too few cores for two final cores per lane"); DeepSeek caps links at
  DRAM banks / 2 (`models/demos/deepseek_v41_flash_d_p/tt/ccl.py:rs_links`).
- Every hop must be a direct fabric link (`plan()` ValueError "not a direct link"; e.g. no ring along a LoudBox row).
- Without `cache_descriptors` the Python descriptor build costs ~0.65 ms host per call.
- A persistent scratch `(G + 1) Sb x W` per (shape, dtype) stays allocated (module cache).
- `dual_cycles` only exercised via fabric_all_gather's emulated Galaxy; no hardware number.
- Unit files other than `test_fabric_reduce_scatter.py` / `test_fabric_rs_exact_fp32.py` need a 2x4 mesh. The 2x2
  regression (`tests/unit/test_fabric_reduce_scatter.py` + `tests/test_fabric_reduce_scatter_ttnn.py`): 5 passed, with
  errors identical to the unchanged file (`models/demos/deepseek_v41_flash_d_p/bringup/BREADCRUMBS.md`, P.5).
- `import ttnn.bringup.fabric_reduce_scatter_ttnn.fabric_reduce_scatter as m` yields the function, not the module (use
  `sys.modules`) (same file, P.5).
- No DeepSeek-V4.1 model case in `tests/cases.py` yet.

## How to call
```python
from ttnn.bringup.fabric_reduce_scatter_ttnn import prepare
prepare(mesh, cluster_axis=1, num_links=links)  # load time
y = ttnn.bringup.fabric_reduce_scatter(t, cluster_axis=1, num_links=links, dim=3,
                                       exact_fp32_add=True, cache_descriptors=True)
```
(`models/demos/deepseek_v41_flash_d_p/tt/ccl.py:sum_cols`). Tests: `tests/test_fabric_reduce_scatter_ttnn.py` +
`tests/cases.py` (model cases); `tests/unit/test_fabric_reduce_scatter.py` (vs ttnn.reduce_scatter),
`test_fabric_reduce_scatter_topologies.py`, `test_fabric_rs_exact_fp32.py`, `test_fabric_rs_model_shapes.py`,
`test_fabric_rs_vs_ag.py`.

## Used by
- glm53_flash_d_p: `tt/common.py:scatter_rows` (GLM_SCATTER_OP=fabric_ring default: one full-mesh Ring call;
  fabric_bf16: axis 0 then 1), `tt/kda_attention.py` (o_proj partial over the TP axis, GLM_KDA_OUT_RS=fabric),
  `tt/experts_ag.py` (GLM_EXPERTS_MODE=ag, GLM_AG_RS_OP=fabric); bf16 inputs; task O.1 model cases. Whole model on the
  LoudBox via glm53_flash_d_p_lb.
- deepseek_v41_flash_d_p: `tt/ccl.py` (CCL_IMPL=fabric, P.5: sum_cols dim 3 + exact, tp_sum dim 2 + exact +
  fabric_all_gather, cache_descriptors, `prepare` at load; CCL_IMPL=fabric_bf16, P.6: default FPU add on bf16).

## History
CHANGELOG.md
