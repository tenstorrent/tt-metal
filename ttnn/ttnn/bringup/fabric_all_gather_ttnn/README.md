# ttnn.bringup.fabric_all_gather  (C++; plus the Python prototype `ttnn.bringup.fabric_all_gather_ttnn.fabric_all_gather` and its `plan()`)

## What it does
New op (no TTNN source on this tree; from `mstaletovic/mimo-v2-dp` @ `b45a923d73c`): a drop-in for
`ttnn.experimental.high_bw_all_gather` with its own, link-rate program. Same Python signature, validation, program
hash, output spec and output bytes (incl. the bytes a partial gather leaves untouched): the device operation derives
from high_bw_all_gather's (`device/fabric_all_gather_device_operation.hpp`); only the program differs.
- Contract: input DRAM (interleaved or ND-sharded), TILE or ROW_MAJOR, any dtype; output preallocated interleaved
  DRAM at the worst-case gathered shape, always in row-major chip order. `cluster_axis` 0 / 1 gathers along one mesh
  axis (the other axis runs independent gathers); `None` over the whole 2D mesh.
- Program: per chip, per ring direction, per link one fabric link worker core next to its link's Ethernet core
  (placed by NoC1 hops): reader (NCRISC, NoC0) reads the own shard from the input, then the shards it forwards from
  its own output; sender (BRISC, NoC1) sends each fabric chunk (pages contiguous in one DRAM bank, one packet each)
  one hop into the same pages of the neighbour's output. Local copy cores write the own shard (two per link for an
  ND-sharded input, which they convert). Arrival counters gate the relays (shard k forwarded once upstream's last
  packet of it incremented the counter); a ready fence stops a chip writing a neighbour's output before the neighbour
  started the same call. Even rings are balanced (the opposite shard goes half each way). `cluster_axis=None` on a
  torus with both sides >= 3: two edge-disjoint Hamiltonian cycles (all four neighbours); else a snake ring.
  Terms and the chunk walk: `device/kernels/fabric_all_gather_chunk_walk.hpp`.
- Python prototype (`fabric_all_gather_py.py`, importable as `ttnn.bringup.fabric_all_gather_ttnn`, not a registered
  op): a ProgramDescriptor all-gather (DRAM interleaved input only) with explicit `topology` / `scheme` / `balance`;
  its `plan()` (groups, schedules, probe-based port placement) is the planner `ttnn.bringup.fabric_reduce_scatter`
  uses. CLI: `python -m ttnn.bringup.fabric_all_gather_ttnn`.

## Consider it instead of
- `ttnn.experimental.high_bw_all_gather`: same contract, faster program; switch by name.
- `ttnn.all_gather` for DRAM gathers whose shape fits the contract (tile-aligned TILE dims, no ROW_MAJOR innermost
  gather): faster at every GLM-5.3 row-gather shape and at DeepSeek-V4.1's gathers when every link is used (Measured).
- Trace-safe KV-cache prefix gathers (`gathered_prefix_tensor`, `input_batch_index_tensor`) and sub-device overlap
  (caller-owned semaphores: no allocation, no sync).

## Options
Realistic (use by default): every option below.

C++ `ttnn.bringup.fabric_all_gather(input_tensor, dim, output_tensor, *, cluster_axis, ...)` (all as
high_bw_all_gather; no fork additions):
- `input_tensor`, `dim`, `output_tensor`: required; output preallocated at the worst-case gathered shape.
- `cluster_axis`: required keyword; 0 / 1 or `None` (whole mesh).
- `num_links=None`: every usable link on the axis (full mesh: the minimum over both axes); explicit > 0 and <= that.
- `subdevice_id=None`, `sub_core_grids=None`: confine the op's cores.
- `input_batch_index=None`: one batch slot of a persistent input cache ([B, 1, ..] -> batch 1).
- `gathered_dim_size=None`: valid global gathered length; each rank writes its prefix into its fixed slot, the rest
  is left unchanged.
- `input_batch_index_tensor=None`: trace-safe `input_batch_index` (1-element uint32 RM DRAM user id), with
  `batch_slot_num_layers=1`, `batch_slot_layer_idx=0` (runtime, not hashed). Exclusive with `input_batch_index`.
- `gathered_prefix_tensor=None`: trace-safe `gathered_dim_size` (chunk start read on device), with
  `gathered_slab_global=0` (block-cyclic slab width, required with it, hashed). Exclusive with `gathered_dim_size`.
- `ready_semaphore=None`, `data_valid_semaphore=None`: caller-owned, zero-initialised global semaphores (both or
  neither); allocation-free, no internal sync.

Python prototype `fabric_all_gather(input_tensor, *, cluster_axis=0, topology=ttnn.Topology.Linear, num_links=1,
dim=0, placement="auto", scheme="ring", balance=False, desync=False, output=None, subdevice_id=None,
sub_core_grid=None, ready_semaphore=None, data_valid_semaphore=None)`:
- `topology`: Linear or Ring (a 2-chip ring is a line). `scheme`: "ring" or "dual_cycles" (cluster_axis=None, Ring,
  torus sides >= 3). `placement`: "auto" (probe Ethernet cores, NoC1-nearest port) or "simple" (forced under the
  emulator). `balance`: split the opposite shard between both directions of an even ring. `desync`: start the
  backward walk halfway round its bank set. Others as the C++ op.

Precision only: per-step checks or a measured end-to-end gap (rule 7): none.

## Measured
- QuietBox 4x p150a, 16 MiB bf16 per chip, 2 links, Python prototype: FABRIC_1D 4-chip ring 125.7 GB/s per chip, line
  87.8, 2-chip line 88.3; FABRIC_2D ring 104.7, line 72.9; FABRIC_1D 4 links ring 143.3
  (`report_python.md`). `balance=True` ring 153-156 GB/s at 2 links, 166 at 4 (`README_python.md`). Increment on
  every packet 24.5 vs every 8 chunks 46.4 GB/s per chip (`report_python.md`). All 112 fabric x topology x link
  combinations bit-exact (`README_python.md`).
- C++ op at ~84% of a LoudBox link: claimed in `CHANGELOG.md`, unsourced (its 2026-10-19 note; no measurement in this tree).
- GLM-5.3, LoudBox 2x4 p150b, 2 links, device time vs `ttnn.all_gather`: 640x4096 bf16 axis 1 281 -> 208 us, 640x1536
  132 -> 88, 2560x4096 axis 0 340 -> 289, 640x512 / 2560x512 56 / 65 -> 42 / 53; ~7 ms per chunk
  (`models/demos/glm53_flash_d_p_lb/bringup/BREADCRUMBS.md`, "Row gathers on fabric_all_gather"). Striped MLA latent
  prefix gather (full mesh, gathered_dim_size): 0.58 ms (high_bw_all_gather) -> ~0.31 ms at the longest context
  (same file, "Router as one fused gate; striped MLA latent cache").
- DeepSeek-V4.1, 2x2 BH: num_links=2 slower than `ttnn.all_gather` (fp32 [1024, 2560] dim 3 0.235 vs 0.139 ms; bf16
  [6912, 512] 0.159 vs 0.098); num_links=None 0.126 / 0.090 (`models/demos/deepseek_v41_flash_d_p/bringup/
  BREADCRUMBS.md`, P.5; `known_issues.md` DS41-66). 25 calls 0.86 ms per profiled layer pair (BREADCRUMBS.md, P.5).
- Two Hamiltonian cycles: bit-exact on an emulated 32-chip Galaxy (4 x 8 torus); speed not measured (no real Galaxy)
  (`report_python.md`).

## Limits
- Not supported: a ROW_MAJOR gather along the innermost dim; a partial length along a TILE gather dim must be tile
  aligned; rows that are not whole tiles fail (GLM's KDA 3x6144 conv state keeps `ttnn.all_gather`,
  `models/demos/glm53_flash_d_p_lb/bringup/BREADCRUMBS.md`).
- The input must carry a 2D declared distribution; split-layout tensors with a 1D topology are rejected (re-declare
  it, as glm53_flash_d_p `tt/common.py`).
- Caller-owned semaphores must be `L1_SMALL` (device_params must reserve `l1_small_size`)
  (`models/demos/glm53_flash_d_p_lb/bringup/BREADCRUMBS.md`).
- Use `num_links=None` on a 2x2 BH (4 links per neighbour pair): 2 links is slower than stock (DS41-66).
- Python prototype: DRAM interleaved only; its `plan()` rejects a hop between chips that are not direct fabric
  neighbours (ValueError "not a direct link", CHANGELOG.md "Direct hops only").
- Emulator (tt-emule): FABRIC_1D rejected by the control plane on a 2D mesh, FABRIC_1D_RING aborts in the emulator
  (`README_python.md`).
- Model cases cover glm53_flash_d_p's 2x2 calls only (`tests/cases.py`); no DeepSeek-V4.1 case yet.

## How to call
```python
out = ttnn.allocate_tensor_on_device(ttnn.Shape(gathered_shape), t.dtype, t.layout, mesh, ttnn.DRAM_MEMORY_CONFIG)
ttnn.bringup.fabric_all_gather(t, 2, out, cluster_axis=0, num_links=None,
                               ready_semaphore=s0, data_valid_semaphore=s1)  # s0/s1: L1_SMALL global semaphores
```
(`models/demos/deepseek_v41_flash_d_p/tt/ccl.py:all_gather`). Tests: `tests/test_fabric_all_gather_ttnn.py` +
`tests/cases.py` (model cases, bit-exact incl. untouched fill); `tests/unit/test_fabric_all_gather_op.py` (vs
high_bw_all_gather, trace-safe metadata, sub-device, large pages, host dispatch cost);
`tests/unit/test_fabric_all_gather_device_time.py` (device time vs high_bw_all_gather);
`tests/unit/test_fabric_all_gather_py.py` (Python prototype sweep). `tests/source.yaml` is empty.

## Used by
- glm53_flash_d_p: `tt/common.py` gather_axis (GLM_FABRIC_GATHER=1, default), `tt/mla_attention.py` (striped latent
  prefix gather, cluster_axis=None, gathered_dim_size), `tt/experts_ag.py` (GLM_EXPERTS_MODE=ag, caller-owned
  semaphores; GLM_AG_GATHER_OP=high_bw keeps the stock op); task O.1 model cases. The whole model runs on the LoudBox
  via glm53_flash_d_p_lb.
- deepseek_v41_flash_d_p: `tt/ccl.py` (CCL_IMPL=fabric / fabric_bf16, P.5 / P.6): every all_gather whose contract
  fits, num_links=None, caller-owned L1_SMALL semaphores.
- `ttnn.bringup.fabric_reduce_scatter` plans with the Python prototype's `plan()`.

## History
CHANGELOG.md; worklogs README_python.md (Python prototype: design, CLI, measured results) and report_python.md
(device report).
