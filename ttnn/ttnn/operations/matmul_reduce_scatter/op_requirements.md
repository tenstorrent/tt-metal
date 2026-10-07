# Operation Requirements: matmul_reduce_scatter

## Definition
- **Formula**: `P = Σ_{g ∈ group(cluster_axis)} A[g] @ W[g]`; the device at group position `p` keeps block `p` of `P`
  along `scatter_dim` (`-2`: rows `p·M/G ..`, `-1`: columns `p·N/G ..`). Output is bfloat16.
- **PyTorch Reference**:
  ```python
  def matmul_reduce_scatter_ref(a_stacked, w_stacked, cluster_axis, scatter_dim):
      # a_stacked: (R, C, ..., M, K), w_stacked: (R, C, K, N); stacked[r, c] is what device (r, c) holds
      partial = a_stacked.float() @ w_stacked.float()
      total = partial.sum(dim=cluster_axis, keepdim=True).expand_as(partial)
      g = a_stacked.shape[cluster_axis]
      return torch.stack([torch.stack([torch.chunk(total[r, c], g, dim=scatter_dim)[(r, c)[cluster_axis]]
                                       for c in range(a_stacked.shape[1])]) for r in range(a_stacked.shape[0])])
  ```
- **Import Path**: `from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter`
- **Function Signature**:
  `matmul_reduce_scatter(input_tensor, weight, *, cluster_axis: int, scatter_dim: int = -2,
  topology: Topology = Topology.Linear, num_links: int | None = None,
  compute_kernel_config: ttnn.ComputeConfigDescriptor | None = None, memory_config: ttnn.MemoryConfig | None = None)
  -> ttnn.Tensor` (one `ttnn.generic_op` dispatch per call)

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype** (activation): [bfloat16]
- **SUPPORTED weight_dtype**: [bfloat16, bfloat8_b]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED shape-derived axes**: alignment = tile_aligned
- **SUPPORTED op-specific axes**: cluster_axis ∈ {0, 1}, scatter_dim ∈ {-1, -2}, topology ∈ {Linear}, num_links ∈ {1, 2}
- **Cores**: per chip, a compute rectangle (all grid rows but the transport row; up to `m_lines × n_lines` ≈ 80–99
  cores on the shapes of interest) running a 2D-multicast matmul, plus `4·num_links` transport cores (ports + finals)
  running the line reduce-scatter. Regimes R1 (block-invariant operand resident) and R2 (both operands streamed).
- **Compute config**: user `compute_kernel_config` (default HiFi2 + `fp32_dest_acc_en=True`) on the matmul;
  transport adds always fp32 DEST.
- **Golden baseline**: 375 supported_pass / 1041 (+ 6 registry-free regression tests passing), 660 xfail_expected,
  0 supported_fail / xpass_drift / xfail_wrong_mode. The `xfail_expected` bucket is exactly the TARGET − SUPPORTED
  gap below (`dtype=bfloat8_b` 312, `weight_dtype=bfloat4_b` 156, both 156, `topology=Ring` 36).
- **Perf baseline** (device kernel duration, steady state, max over 8 chips): FOCUS 209.5 us, GLM 241.9 us,
  MiMo 439.3 us (see `verification_report.md` → Performance baseline).

**TARGET − SUPPORTED** (every value is covered below; no INVALID entries apply):

| Axis | Missing value | Refinement |
|------|---------------|------------|
| `topology` | `Ring` | Refinement 1 |
| `dtype` (activation) | `bfloat8_b` | Refinement 2 |
| `weight_dtype` | `bfloat4_b` | Refinement 2 |

**Perf-1 anchor**: the PERF FOCUS loose case (`640×2048×7168`, A bf16, W bfloat8_b, TILE, `cluster_axis=1`,
`scatter_dim=-1`, Linear, `num_links=2`, HiFi2, `fp32_dest_acc_en=False`) is **already fully inside SUPPORTED** at
Phase 0 — no generality refinement is needed to unlock it, so Refinements 1–2 are ordered purely by difficulty
(hardest first).

### [x] Refinement 1 — Ring topology (both ring directions)

**Goal**: add `Topology.Ring` to `SUPPORTED["topology"]`. The cells that move from `xfail_expected` to passing on
this board are the Ring cells of `eval/golden_tests/matmul_reduce_scatter/test_ring_mock.py` (the 8-device snake
ring opened as `1×8` under `FABRIC_2D_TORUS_X` / `FABRIC_2D_TORUS_XY` / `FABRIC_1D_RING` along axis 1, and as `8×1`
under `FABRIC_2D_TORUS_Y` / `FABRIC_2D_TORUS_XY` / `FABRIC_1D_RING` along axis 0; FOCUS shape `scatter_dim=-1`,
MiMo shape `scatter_dim ∈ {-1, -2}`; `num_links ∈ {1, 2}`). Per the prompt rule, Ring MUST use both ring directions.

**Blocking-model class**: **scheme-change** (design regime R3) — a new transport topology: each block's reduction
chain becomes a line centred on its owner (`fwd` list `p+hf … p+1`, `bwd` list `p−hb … p−1`, mod G), plus the wrap
hop. It stands alone.

**Verifier notes**:
- Hardest generality item, so it goes first (difficulty tier: cross-core restructure). No dependency on
  Refinement 2 (Ring touches only the host schedule + transport kernels; Refinement 2 touches only the compute-core
  operand CBs), so the order is purely difficulty.
- The design's R3 row says "no new structure", but one kernel-level change is real: in a Ring every port's list
  starts with an entry that has **no upstream** (the chip's own partial for the farthest block of that direction)
  followed by relay entries. Phase 0 compiles the port role per core (`relay` → reader feeds `cb_xport_partial` and
  the add kernel produces `cb_xport_sum`; `line end` → reader feeds `cb_xport_sum` directly), and `cb_xport_sum`
  must keep exactly one producer per core. A mixed port therefore needs a per-entry `has_upstream` flag carried to
  the reader **and** the add kernel (e.g. the add kernel copies the partial through for a no-upstream entry); do not
  zero-fill an arrival CB to fake an add.
- Host work: `_groups` must emit ring neighbours (wrap prev/next), `_links` must include the wrap hop,
  `_schedule_mmrs(p, G, ring=True)` per the design's schedule table, landing slots and arrival-counter expectations
  per entry, and the design's validation row #5 (fabric config wraps `cluster_axis` and the wrap hop has links,
  else `ValueError`). `compute_order` must still interleave fwd/bwd one-by-one and end with `p`.
- Keep the Linear cells under every fabric config green (`test_fabric_configs.py`, and the Linear cells of
  `test_ring_mock.py` that already pass at G = 8).
- The real per-axis Ring cells of `test_golden.py` are pruned on this 2×4 LoudBox (no wrap links); they only exist
  on a Galaxy — that is a report finding, not part of this refinement's Done-when.
- Optional (only if it stays within the pass): the `fabric_all_gather` "balanced" far-block split — it halves the
  busiest direction on odd `G−1`. Not required for Done.

**Done when**: every Ring cell of `test_ring_mock.py` passes (functional; deterministic across repeated calls),
`Topology.Ring` is in `SUPPORTED["topology"]`, the exact-once regression tests still pass, and the full golden suite
is green with zero loud categories.

### [x] Refinement 2 — Numerical configurability: bfloat8_b activations + bfloat4_b weights

**Goal**: add `ttnn.bfloat8_b` to `SUPPORTED["dtype"]` (activation `A`) and `ttnn.bfloat4_b` to
`SUPPORTED["weight_dtype"]`. `compute_kernel_config` is already exposed and honoured (HiFi2 / `fp32_dest_acc_en`
floor); keep it. The output stays bfloat16 for every input dtype; the transport stays bf16 on the wire with fp32
DEST adds. Operand CB page formats and tile sizes must follow the input dtypes (`cb_act_operand` Bfp8_b,
`cb_weight_operand` Bfp4_b), and the R1/R2 residency predicate must use the real tile sizes (it already takes
`_tile_bytes(dtype)` — confirm the bigger resident K for the smaller formats is actually exploited).
Cells that fail out of the box for a *structural* reason go to `EXCLUSIONS` (none expected — every TARGET shape is
tile-aligned); `numerical-precision` failures are not silenced.

**Implementation skill**: /numeric-formats-metal

**Blocking-model class**: **knob-turn** (operand formats + tile sizes in the L1 predicate; no new topology).

**Verifier notes**:
- Ordered after Refinement 1 only by difficulty (tier 5 vs tier 2); no dependency either way.
- Expected to be mostly host-side: the injectors already read with `a_tile_bytes` / `w_tile_bytes` CT args and the
  CBs already take `a.dtype` / `w.dtype`. Relax `validate()` first and see which cells XPASS.
- Precision bar for bfloat4_b weights is the golden `TOLERANCES[bfloat4_b]` (PCC 0.99, rel-RMS 0.08); for bfloat8_b
  activations the golden quantizes A like the device, so the error budget is the weight's.
- Run the precision baseline (`test_matmul_reduce_scatter_precision_baseline.py`) with the new dtypes added and
  record the numbers in the changelog.

**Done when**: every `dtype=bfloat8_b` and `weight_dtype=bfloat4_b` cell of `test_golden.py` passes (no new
`supported_fail`), both values are in `SUPPORTED`, and the golden suite is green with zero loud categories.

### [x] Refinement 3 — Speed up the PERF FOCUS case: transport send rate

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES` flags the Kimi K2.7 o_proj as the mandatory perf target:
`A (1,1,640,2048) bf16`, `W (2048,7168) bfloat8_b`, TILE, `cluster_axis=1`, `scatter_dim=-1`, Linear,
`num_links=2`, HiFi2, `fp32_dest_acc_en=False` — perf goal `target_us = 118` (roofline 88.3 us, unfused baseline
394 us), soft precision gate `rel_rms_threshold = 0.055`. Measured at Phase 0 (verifier, `--profile`, 2×4 LoudBox,
FABRIC_2D, steady state, max over 8 chips): **209.5 us** (median chip 199.7 us); Phase-0 rel-RMS 0.024. The **fabric send path** is the binding stage:
the implementer's ablations (matmul stubbed → no change; transport reads stubbed → no change) leave the ports'
packet issue as the critical path (~840 × 4 KiB packets per link per call, live max payload 4352 B → `seg_tiles = 2`).
Speed it up using the relevant patterns in `ttnn/ttnn/operations/examples/master.md` and the design's perf lamps:
- **L5 transport placement** (⭐/⭐⭐): place each port core next to the Ethernet core of its (link, direction) —
  the Ethernet channel is already on the host as the first `setup_fabric_connection` RT arg, so no probe dispatch is
  needed; the design measured 48.5 → 40.6/30.5 GB/s for far placements. This makes transport placement per chip.
- `noc_placement` (⭐⭐): the port's gather + scratch reads and the sender's EDM writes share NoCs; check
  reads-on-NoC0 / writes-on-NoC1 holds for the transport row.
- `split_reader` (⭐⭐, lamp L4): only if a port RISC is shown issue-bound after placement.
No SUPPORTED change.

**Verifier notes**: the 118 us target assumes 14336 B packets at 48.5 GB/s; under the harness's FABRIC_2D router
config the payload is 4352 B: the busiest direction carries `(G−1)·B` = 6.88 MB, i.e. 3.44 MB per link, and the
implementer measured the bare FABRIC_2D 4352 B one-hop stream at ~24.8 GB/s, so the link floor under this config is
≈ 139 us (the 118 us target is not reachable at this payload; ~140 us is the realistic ceiling). Report achieved vs. that floor (achieved / roofline per
resource), not only vs. the 394 us baseline. Measure the exact flagged config — never an `fp32_dest_acc_en=True`
stand-in.

**Done when**: measured device-ns (steady-state call, `test_matmul_reduce_scatter_perf.py` under
`run_safe_pytest.sh --profile`) improves on the FOCUS case, its rel-RMS gate (0.055) still holds, the golden suite is
green, and there is no regression across the config-spanning guard set (one representative per distinct path:
R1 `scatter_dim=-1` (FOCUS), R1 `scatter_dim=-2` (MiMo `2048×2048×4096`), the R2 streamed regime, a link-bound
small-K shape (`640×512×7168`), `num_links=1`, and the Ring path once Refinement 1 has landed). MiMo is itself
link-bound and currently *slower* than its unfused baseline (439 vs 385 us); report its number too — the same
transport levers should move it.

**Outcome**: L5 placement landed. Each port is placed per chip on the transport-row core nearest (in NoC1 hops) its
(direction, link) Ethernet core. Placement is host-only, from the `setup_fabric_connection` channel. Measured device
kernel time (steady state, max over 8 chips, `--profile`, FABRIC_2D at the **production 14 KiB payload**, as
feature_spec's operator note requires; the perf harness now opens the mesh that way):
- FOCUS: 208.5 → 166.6 us (median chip 162 → 143); rel-RMS gate holds (golden loose case passes).
- GLM: 278 → 219 us.
- MiMo: 394 → 357 us (unfused baseline 385).
- Small-K 640×512×7168: 151 → 133 us.
- R2 2048×4096×4096: 444 → ~440 us (compute-bound, unchanged).
- num_links=1: FOCUS 252 → 240 us, small-K 223 → 215 us.
- Under the old 4352 B payload: FOCUS 210 → 208 us max, 200 → 191 us median.

Bottleneck now: sender zones show end-chip steady-state blocks at ≈36 GB/s per link (~15 us/block in
fabric-slot waits, ~3 us CB wait), which is the LoudBox TP-axis link ceiling, so the send *rate* is at roofline. The
first block is ~36 us of pure data starvation, because the matmul runs ~28–34 us/block on the ~70 cores the 20×56-tile
block factorizes onto. The interior chips' extra 15–30 us is relay-chain latency plus cross-chip launch skew absorbed at
the ready fence. noc_placement already holds on the transport row (port reader NCRISC/NoC0 reads, sender and final
writer BRISC/NoC1 writes). split_reader (L4) was not built because no port RISC is issue-bound. Next: pipeline fill
(Refinement 4, sub-block sends) and matmul grid utilization of the scatter block.

### [ ] Refinement 4 — Speed up the PERF FOCUS case: pipeline fill (sub-block sends)

**Type**: perf

**Goal**: same flagged target as Refinement 3 (FOCUS, exact config above; `target_us = 118`). Once the send rate is
fixed, the next term of the FOCUS roofline is the **pipeline fill**: transport cannot start until one whole scatter
block (`640 × 1792` per chip, ~17 us of matmul) is finished. Shrink it by sending at sub-block granularity — design
regime **R4** (`waves_per_block > 1`: a scatter block split into row-waves, each its own K pass on the same grid,
the transport draining wave `w` while wave `w+1` computes), the prompt's "send at sub-block granularity" design
consequence, and lamp L3 (`blocks_in_flight`) as the cheaper neighbour to measure against. Relevant catalog entries
in `ttnn/ttnn/operations/examples/master.md`: overlap scheduling / `double_buffer` (hand-off depth), `mcast_topology`.
No SUPPORTED change.

**Blocking-model class**: a T3 overlap restructure on the `scatter_block` axis (the block walk already carries a
wave index with trip count 1 and segments are row-ordered, so a wave is a contiguous segment range of every scratch
slot); one lever per phase.

**Verifier notes**: gate on headroom — measure the fill first (time from kernel start to the first fabric packet on
a port, e.g. with a `DeviceZoneScopedN` on the port sender) and file the result in the changelog; if, after
Refinement 3, FOCUS is within noise of its link floor, record that the fill is hidden and close this as measured-no-op
rather than forcing the restructure. The streamed operand (W for `-1`) is re-streamed per wave unless the
`Kt·core_n_tiles` W slice is also resident — account for it in the ledger's traffic budget.

**Done when**: measured device-ns improves on the FOCUS case (steady state) with the 0.055 rel-RMS gate holding, the
golden suite green, and no regression across the same guard set as Refinement 3.
