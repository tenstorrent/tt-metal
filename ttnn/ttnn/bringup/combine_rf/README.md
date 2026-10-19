# ttnn.bringup.combine_rf  (Python package, NOT registered as ttnn.bringup.<name>: not in PYTHON_OPS; import the modules)

Public entry points (`from ttnn.bringup.combine_rf.<module> import ...`):
`sparse_combine_rf.sparse_combine_rf_prepare`, `sparse_combine_l1.sparse_combine_l1_prepare`,
`sparse_all_gather_2d.sparse_all_gather_2d` (+ `mesh_plan`, `place_ports`, `ensure_semaphores`, `manifest_spec`,
`port_cores_all`, `route_words`), `dispatch_prep.dispatch_prep_prepare` / `own_rows_copy_prepare` /
`all_chips_route_words`, `combine_plan.combine_planner_prepare` (+ `port_luts`, `plan_reference`, `link_of`,
`first_hop_idx`, `det_page_bytes`), `fabric_all_gather_py.fabric_all_gather` (+ `plan`, `link_load`, `worker_cores`),
`stream_linear.stream_linear_prepare`, `lin2d.lin2d_prepare` (+ `glu_weight`, `upload_stacked`),
`router_topk.router_topk_prepare`, `sum_ops.sum_parts_prepare`, `layout_ops.layout_op_prepare`,
`l1_zero.l1_zero_program`.

## What it does
New ops (Marko Staletovic's; no TTNN source). Each is a host builder that emits one `ttnn.generic_op`
(Mesh)ProgramDescriptor per call; the kernels are inline C++ strings in the module (SourceType.SOURCE_CODE). Most
`*_prepare(...)` return `run(...)`, which caches the descriptor per buffer-address set. They are the pieces of the
"token-split" MoE block (Kimi K3 / DeepSeek-V4.1 / GLM-5.3 on a 2D fabric mesh) around the flat routed expert.
- `sparse_combine_rf` (the main op): MoE combine level 1 with the fabric routers relaying. Every expert output row goes
  from its expert's chip to its token's home chip as one multi-hop 2D unicast (Y then X, fixed link per (accumulator,
  source, home)) straight into a home accumulator ring in L1. Accumulators multiply each row by its gate weight
  (HiFi4, fp32 DEST) and add it onto the token's fp32 sum with packer L1 accumulation; output bf16 rows per home token.
  Cores: one port core per (direction, link) + A accumulator cores. Standalone (`run(y, out)`, y in DRAM) or appended
  to the flat expert via `ttnn.bringup.flat_routed_expert_with_program` (`run.build` / `build_l1`: rings and state in
  the flat expert's arena, rows handed over from the flat y writers in L1, per-call counts from `combine_plan`).
  Options for a deterministic sum order, one port per direction, accumulators on port cells, and (P.8) a combine
  along each mesh column only (`axis=0`: tokens replicated over the mesh row), fp32 output, an output row offset (a
  pass of a larger output), an exact fp32 final copy, zeros for home tokens with no row.
- `sparse_combine_l1`: the earlier prototype (relays hold rows in L1 rings hop by hop, unweighted sum, rows read from
  a DRAM copy of y). Kept: sparse_combine_rf reuses its port reader / accumulator kernels.
- `sparse_all_gather_2d`: sparse all-gather on a 2D mesh (line or ring per axis): each own row goes only to the chips
  in its 32-bit route word (destination mask, X- or Y-first, link), over dimension-order trees, one hop per port.
  Output [1, 1, N*S, W] RM DRAM, row t of chip idx at slot idx*S + t. Used as the sparse MoE dispatch.
- `dispatch_prep`: per chip, one generic_op making sparse_all_gather_2d's inputs from the replicated joint routing
  rows: route words of the own tokens (route cores) and the own rows copied into their dispatch-buffer slots (copy
  cores).
- `combine_plan`: the per-call tables built on device from the replicated routing (top-k ids + bf16 gate weights):
  the flat expert's counts / regions / token_index, the y writers' L1 hand-off table, and the combine's per-port /
  per-accumulator counts (plus an order table for the deterministic combine). `plan_reference` is its host model.
- `fabric_all_gather_py`: line / ring all-gather over the fabric, DRAM to DRAM (per-mesh-axis or whole-mesh snake,
  `scheme="dual_cycles"` on a torus); a copy of `../fabric_all_gather_ttnn/fabric_all_gather_py.py` plus memoized
  plans / descriptors and an `extra` hook (another op's CBs / kernels appended).
- `stream_linear`: y = x @ W for M = 32 x grid rows (bf16 x, bf8 W), 2D matmul with x resident and a W ring per
  column. A dead end kept as a probe (FLAT_COMBINE_OVERLAP_WORKLOG.md, "custom streaming linear").
- `lin2d`: y = x @ W for M = 32 x grid rows (K3: 320), 2D multicast matmul laid out for the Blackhole NoC / DRAM (x
  multicast first along rows, then W down columns from both ends on NOC0 / NOC1, DRAM reads on NOC0 only). Modes:
  `glu` (gate / up tile pairs, SiTU-GLU on the pack thread, y = activation) and `residual` (y = r + x @ W, one rounding).
- `router_topk`: Kimi K3's router selection: top-K by the biased score, weights s / sum(s) * route_scale, written as
  the joint routing rows [S, 2K] uint16 (ids, then bf16 weight bits) that combine_plan and the all-gather read.
- `sum_ops`: y = a + b[0] + ... + b[P-1] for bf16 TILE DRAM tensors in one pass and one rounding.
- `layout_ops`: bf16 tilize / untilize of a [rows, W] DRAM tensor, parallel over (row tile, CW-column chunk) units on
  the whole grid.
- `l1_zero`: zero runs of L1 on chosen cores on device (per-call scratch: counters, ring trailers).

## Consider it instead of
- sparse_combine_rf: `ttnn.experimental.deepseek_prefill.combine` / `combine_fabric2d` + `post_combine_reduce`, or
  `ttnn.bringup.combine` + a weighted sum; when the flat expert runs and the combine should overlap it.
- sparse_all_gather_2d + dispatch_prep: `ttnn.all_gather` (dense) / `dispatch_fabric2d` / `ttnn.bringup.dispatch`.
- fabric_all_gather_py: `ttnn.all_gather`, `ttnn.experimental.high_bw_all_gather`; prefer the C++
  `ttnn.bringup.fabric_all_gather` (../fabric_all_gather_ttnn) unless you need `extra`.
- lin2d / stream_linear: `ttnn.linear` / `ttnn.matmul` (2D mcast) at small M (~320 rows) with large K, N.
- router_topk: `ttnn.sigmoid` + `ttnn.add` (bias) + `ttnn.topk` + normalize + layout / concat ops.
- sum_ops: `ttnn.sum` over parts + `ttnn.add` (two passes, two roundings).
- layout_ops: `ttnn.tilize` / `ttnn.untilize` on short-and-wide tensors (they split only over row tiles).
- l1_zero: re-allocating or `ttnn.zeros` for kernel scratch.

## Options
- `sparse_combine_rf_prepare(mesh_device, S, W, rows_by_chip, *, num_links=2, torus=(False, False), n_acc=16, depth=8,
  cells=None, arena=None, writers=0, counts_off=None, l1=None, devplan=None, deterministic=False,
  ports_per_direction=False, acc_on_ports=False, piece_max=None, axis=None, out_dtype=None, out_row0=None,
  acc_unpack_fp32=False, zero_unseen=False)`: S home tokens per chip, W row width; rows_by_chip
  places the cores (and, without devplan, gives the rows); n_acc accumulators (<= 16); depth ring slots cap; cells =
  allowed logical cells; arena = caller-owned L1 tensor; writers > 0 = overlapped with the flat expert; l1 = L1
  hand-off from the flat y writers; devplan = combine_plan's device tables; deterministic = fixed sum order (needs
  l1 + devplan with "dord"); ports_per_direction = one port core per direction; acc_on_ports = ports on accumulator
  cells; piece_max = row bytes per fabric packet (env SPC_PIECE, default 7168). axis=0 (default None: the whole
  mesh) = the combine within each mesh column (chip r of an R-chip group, Y ports, R - 1 rings; with
  combine_planner_prepare(axis=0)); out_dtype ttnn.float32 (default bf16) = the fp32 sums written; out_row0 (default
  None) = home token t at output row out_row0 + t; acc_unpack_fp32 (default False) = the sums' final copy through
  UnpackToDestFp32 (else srcA: TF32); zero_unseen (default False, with l1) = a home token with no row writes zeros.
  `accumulator_bytes(N, A, W, S, depth=1, out_dtype=None, local_slots=2)`: an accumulator's L1 for S home tokens (a
  caller sizes passes / the arena with it).
- `sparse_combine_l1_prepare(mesh_device, S, W, rows_by_chip, *, num_links=2, torus=(False, False), n_acc=16, qc=64,
  qt=32, qa=16, acc_cores=None)`: qc / qt / qa ring depths (continue / turn / accumulate).
- `sparse_all_gather_2d(output, plan_tensor, *, manifest=None, num_links=2, torus=(False, False), group=8,
  manifest_batch=16, pack=None, debug=None, spin_limit=0, extra=None)`: manifest = per-call ring tensor
  (`manifest_spec`) instead of the persistent one; extra = another op's CBs / kernels appended.
- `dispatch_prep_prepare(mesh_device, *, S, K, EPC, n_route=8, n_copy=16, batch=4, torus=(False, False), rows=(8, 9),
  tokens_per_page=1, copy=True)`: route / copy core counts and their grid rows; copy=False = route words only.
- `combine_planner_prepare(mesh_device, *, S, K, EPC, A, L, port_luts, n_ports, rows_cap, core=(0, 0), cores=None,
  row_pages=False, joint=False, tokens_per_page=1, torus=(False, False), det=None, token_major=True, axis=None,
  id_offset=None, tokens=None, token_base=0, ids_uint32=False, tile_in=False)`: cores = the
  multi-core planner; joint = one [N S, 2K] ids+weights tensor; det = order table for the deterministic combine;
  token_major = an expert's rows in (token, home, k) order (default). The one-core planner's column / pass options
  (defaults: the whole-mesh plan): axis=0 = plan as chip r of its column (counts / regions [1, R EPC] of the
  column's experts); id_offset(r, c) = subtracted from every id (others' experts skipped); tokens / token_base = a
  pass of S tokens of `tokens` per home (token index home x tokens + token); ids_uint32 = uint32 ids; tile_in = ids
  / weights as TILE [tokens, K] (K <= 16), only the pass's tiles read; det with these options = the order table from
  every column chip's flat schedule. `run.programs(ids, wts)` / `run.cores` / `run.tensors` (several planners in one
  program). `port_luts(..., axis=0)`, `axis_ports(chips, axis)`, `plan_reference(..., id_offset=, token_base=)`.
- `fabric_all_gather(input_tensor, *, cluster_axis=0, topology=ttnn.Topology.Linear, num_links=1, dim=0,
  placement="auto", scheme="ring", balance=False, desync=False, output=None, subdevice_id=None, sub_core_grid=None,
  ready_semaphore=None, data_valid_semaphore=None, extra=None)`.
- `stream_linear_prepare(mesh_device, *, M, K, N, kb=16, nb=7, np_=None, fidelity=None, w_dtype=ttnn.bfloat8_b)`.
- `lin2d_prepare(mesh_device, *, M, K, N, kb=8, nb=4, nst=3, nstw=None, xs=None, ws=None, fidelity=None,
  w_dtype=ttnn.bfloat8_b, out_dtype=ttnn.bfloat16, glu=False, residual=False)`: fidelity default HiFi2 (env L2D_FID);
  kb K-block tiles, nb W ring slots, nst / nstw staging slots.
- `router_topk_prepare(mesh_device, *, S, E, K, route_scale=1.0, split=8, bias_mode=False)`: split cores per row
  tile; bias_mode = read only the biased scores + the bias row.
- `sum_parts_prepare(mesh_device, *, P, bt=8, dist="rr", split=None)`: bt tiles per unit, dist "rr" / "contig".
- `layout_op_prepare(mesh_device, kind, *, cw=8, pages_per_row=1, out_memory_config=None, row0_of=None, exclude=())`:
  kind "tilize" / "untilize"; cw tile columns per unit.
- `l1_zero_program(mesh_device, runs_by_core)`: runs (address, bytes, stride, count) per logical core.
Many env knobs (SPC_*, FC_*, L2D_*, SUM_SPLIT) change behaviour; read the module before setting them.

## Measured
All from ../flat_routed_expert_ttnn/FLAT_COMBINE_OVERLAP_WORKLOG.md (LoudBox 2 x 4, bh-lb-09, 12 x 10 grid):
- sparse_combine_rf alone vs the busiest-link floor, us (uniform / spread / packed): K3 1023 / 1061 / 2332 vs 739 / 798
  / 2074; V4.1 629 / 662 / 1278 vs 400 / 466 / 1108 ("Combine alone, profiled").
- Overlapped with the flat expert (flat_routed_expert_with_program), V4.1 dims uniform: flat alone 1089, combine
  alone 646, serial 1735, fused 1238 us; worst per-token rel err 2.4e-3 ("DeepSeek V4.1-Flash: flat + combine overlap").
  K3 uniform: flat 2462, combine 1013, fused 2674 us ("Flat expert + router-forwarded level-1 combine").
- Deterministic combine: K3 block 0 of 99 replays differ (was 26 of 99 in arrival order) ("Deterministic combine").
- sparse_all_gather_2d as K3 dispatch: 229 us vs 393 us for two dense ttnn.all_gather; dispatch_prep 28 us ("K3
  block: sparse 2D dispatch").
- lin2d, K3 shared expert: gate 158 us vs mm2d 196, down 156 vs 195 ("K3 block: the shared expert as lin2d").
- stream_linear for K3 latent projections: 133-230 us vs ttnn mm2d 117 / 123 (slower; "K3 block to 3700 us traced").
- sum_ops fused SE part sum + add: 74 -> 55 us, then 48 us with reads on both NoCs; layout_ops [320, 3584] tilize
  18 -> 15 us ("K3 block to 3700 us traced").
- l1_zero arena zeroing 5.6 us ("Flat expert + router-forwarded level-1 combine" bring-up notes).
- router_topk: not measured standalone (traced K3 router programs 41.4 + 7.1 + 38.4 us, same section as sum_ops).
- Column combine (axis=0) with the flat expert, DeepSeek-V4.1 P.8, 2x2 QuietBox p300c (11 x 10 grid), layer 0, chunk
  2048 (1024 home tokens per chip in 2 passes of 512; 1 port cell + 9 accumulators), device µs per layer, profiler op
  mode (deepseek_v41_flash_d_p/tt/perf_experts.py::test_weighted_vs_separate, EXPERTS_PRECISION=glm): the two fused
  calls 4104-4121, both passes' one-core planners in one program 616 (arrival order) / 807 (det, full precision);
  the whole experts call (q(x) gather .. reduce-scatter) 5.27-5.4 ms arrival order, 6.05 ms det at full precision vs
  8.0 / 8.5 ms for the separate dispatch / flat / combine / K-slot sum chain. vs float64 (128 tokens, full precision,
  fp32 out): rel 0.0029, bias +2e-5; det: bit-identical calls (arrival order: 149 of 10.5M values differ at 1e-8).
- One-core planner, column options, tile input: 2 passes x (2048 tokens x 6) in 616 µs (one program, two cores); the
  same with the one-page input and a ttnn.reshape to one page: 758 + 2040 µs (the two reshapes).

## Limits
- Tests in this tree: tests/unit/test_column_planner.py only (the one-core planner's column / pass options vs
  plan_reference, det order lists); the harnesses named in the docstrings / worklog (scratch_flat_combine_rf.py,
  scratch_k3_moe_layer.py, zz_scratch_*.py) are on the source branch only.
- axis=0: the one-core planner only (no multi-core column planner); the column's chips must fit the accumulators'
  rings (R - 1 per accumulator); the fp32 accumulators hold every home token in L1 (W x 4 B a token): a caller with
  more home tokens than the free cells hold runs passes (tokens / token_base, out_row0).
- With the L1 hand-off a combine needs a fabric payload of row + 32 B (V4.1: 10880 B) and nothing in the arena's L1
  between the zeroing and the call but a cross-chip barrier whose CBs fit under the arena (deepseek_v41 uses a
  CB-free fabric semaphore barrier, tt/weighted_combine.py `_column_barrier`; a CCL sized for 10880 B packets clashed).
- Blackhole fabric meshes only; worklog runs are LoudBox 2 x 4 / 8 x 1 (and Galaxy prep notes).
- sparse_combine_rf: every home token's fp32 sum stays in its accumulator's L1 for the whole call (W x 4 B a token,
  ceil(S / A) tokens an accumulator; accumulator_bytes). With an arena that cannot hold them the op raises ValueError
  with the largest S and the smallest A that fit. V4.1 (W 5120, 9 accumulators, <= 1356 KB) fits <= 558 home tokens a
  call; 1024 need >= 17 accumulators. There is no 8-bit tile limit: the earlier "255-tile hang" was the local ring
  grant deadlock fixed in P.8 (CHANGELOG).
- sparse_combine_rf: N <= 32 chips, A <= 16 accumulators (assert); with the L1 hand-off a row must fit one piece:
  SPC_PIECE >= row bytes and fabric payload >= row + 32 B, else a hang (assert in sparse_combine_rf_prepare);
  acc_on_ports / deterministic / ports_per_direction need the L1 hand-off; the arena must not move between calls (the
  kernels bake its address).
- lin2d: M % 32 == 0, 2 <= M/32 <= grid rows, K/32 % kb == 0, nb even and <= 8 (asserts).
- layout_ops: rows % 32 == 0, W % (32 * cw) == 0, bf16, DRAM interleaved.
- sparse_all_gather_2d: output RM DRAM interleaved, own rows already in place.

## How to call
Wiring examples (flat expert + this combine in one program; Kimi K3's MoE block): `examples/` (reference only).
```python
from ttnn.bringup.combine_rf.sparse_combine_rf import sparse_combine_rf_prepare
run = sparse_combine_rf_prepare(mesh, S, W, rows_by_chip, num_links=2, n_acc=16)
run(y, out)            # standalone: y rows in DRAM -> out, home tokens' weighted sums (bf16)
# overlapped: desc, l1_args = run.build_l1(y, out, arena) then
# ttnn.bringup.flat_routed_expert_with_program(..., arena=arena, y=y, extra=desc, l1_args=l1_args, ...)
```
Tests: tests/unit/test_column_planner.py (column planner); the model's tt/perf_experts.py::test_weighted_vs_separate /
test_weighted_combine_routes (deepseek_v41_flash_d_p).

## Used by
- `ttnn.bringup.flat_routed_expert_with_program` (../flat_routed_expert_ttnn) is the C++ side built for
  sparse_combine_rf; lin2d imports `flat_expert._bank_sharded`.
- deepseek_v41_flash_d_p P.8: tt/weighted_combine.py (EXPERTS_COMBINE=weighted): sparse_combine_rf (axis=0, L1 hand-off,
  device plan, deterministic, ports_per_direction, out_dtype fp32 / bf16, out_row0, acc_unpack_fp32, zero_unseen),
  combine_planner_prepare (axis=0, id_offset, tokens, token_base, ids_uint32, tile_in, det), l1_zero.
  Upstream it served the K3 token-split MoE block and the GLM-5.3 "ts" mode (FLAT_COMBINE_OVERLAP_WORKLOG.md).

## History
CHANGELOG.md; design and measurements in ../flat_routed_expert_ttnn/FLAT_COMBINE_OVERLAP_WORKLOG.md.
