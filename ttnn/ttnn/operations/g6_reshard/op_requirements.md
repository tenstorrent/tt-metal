# Operation Requirements: g6_reshard

## Definition
- **Formula**: `out[i] = in0[i]` for every logical index `i`. `out` is held BLOCK_SHARDED in L1 on the placement `memory_config` names, and `in0` is unchanged. For bfloat8_b every tile moves as stored, shared exponents included.
- **PyTorch Reference**:
  ```python
  def pytorch_g6_reshard(input_tensor, memory_config=None):
      return input_tensor.clone(memory_format=torch.contiguous_format)
  ```
- **Import Path**: `from ttnn.operations.g6_reshard import g6_reshard`
- **Function Signature**: `g6_reshard(input_tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor`. `input_tensor` is rank 4, bfloat8_b, TILE, BLOCK_SHARDED L1. `memory_config` is BLOCK_SHARDED L1 with a shard spec. The output has the input's shape, dtype and layout, in `memory_config`.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

> **Queue shape**: `TARGET` (`feature_spec.py`) equals Phase 0 SUPPORTED (`dtype: [bfloat8_b]`, `layout: [TILE_LAYOUT]`), so `TARGET − SUPPORTED = ∅`. No generality refinement exists to file. Placements are case arguments, not axes, and every legal placement pair is already served. Following the 2:1 cadence rule with the generality candidates exhausted, every phase below is a measured perf refinement. All target the perf-flagged loose case P (`LOOSE_CASES[0]`), whose full contract (bfloat8_b, TILE, BLOCK_SHARDED L1 on both sides, no compute config) is already SUPPORTED.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat8_b]
- **SUPPORTED layout**: [TILE_LAYOUT]
- **SUPPORTED shape-derived axes**: none (`INPUT_TAGGERS = {}`). Any rank-4 shape with H, W multiples of 32, and every legal block placement on either side.
- **SUPPORTED op-specific axes**: none
- **Cores**: out's block-holder rectangle (P: 64 cores, O1: 56 cores). Two issuers per core (RISCV_1, RISCV_0). The NoC per issuer comes from the `balanced` route-model policy.
- **Compute config**: none (no compute kernel)
- **Golden baseline**: 9 / 9 registry cells passing (12 / 12 tests incl. 3 regression); verifier CLI: 0 supported_fail, 0 xpass_drift, 0 xfail_wrong_mode
- **Perf baseline (P)**: 1637 ns Tracy `DEVICE KERNEL DURATION` (warm median); 1980–2041 ns golden `device_kernel_ns`; stock 1990 ns

### [ ] Refinement 1 — Cut P's busiest NoC link

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES[0]` flags P as the mandatory perf target: `(1,1,2048,1024)` bfloat8_b TILE, in0 on (0,0)-(10,9) shard (224,96) ROW_MAJOR, out on (0,0)-(7,7) shard (256,128) ROW_MAJOR. Its perf goal: faster than stock (1.99 µs) and within 1.35× of the lead's hardware bound; exact gate. The route model in `g6_reshard_program_descriptor.py` (dimension-order, BH virtual coordinates; `probes/probe_024.py`) says P is link-bound:
- P's sources lie east/south of their targets, so the shipped `balanced` policy puts 121 of 128 issuers on NoC1.
- The busiest NoC1 link carries 64 tiles, about 1088 cycles at 64 B/cycle. That is the largest single term of P's ~2200-cycle span.
- Per-issuer NoC choice cannot go below 64.
- Per-piece choice reaches 56 in a host-only greedy (`probes/probe_025.py`). Finer splits that use both NoCs from one issuer (per run or per row group, wrapped NoC0 routes included) may go lower.

Lower P's busiest-link load and measure it. The `noc_placement` entry of `ttnn/ttnn/operations/examples/master.md` describes the NoC-selection lever (NoC0 east/south, NoC1 west/north). Before building, check the model against the device: the existing `fixed` / `wrap_free` / `balanced` A/B in `test_g6_reshard_noc_policy_perf.py` should rank P's device-ns the way the modeled busiest link ranks them. If it does not, find the real bound before spending this lever. The kernel's per-run NoC must come from an O(1) per-core encoding (e.g. a host-chosen threshold per issuer). A per-run list in RT args breaks BB-04, because its size grows with the placement. Same-NoC-only issuers keep dedicated mode. Anything else stays in `DM_DYNAMIC_NOC` and keeps one barrier per NoC it used. No SUPPORTED change.

**Verifier notes**:
- First perf phase because the link term is the largest modeled term and this change reshapes the issue loop. Refinement 2's ordering and split knobs are tuned on top of that loop.
- I3 (busiest link 112 tiles) and O1 (128) are the slowest cells in the guard set. They will likely move with this lever and must not regress.
- The cache key must gain any new policy knob (BB-07).
- Do not touch the launch skew: it is dispatch-side and outside the op.

**Done when**: Tracy `DEVICE KERNEL DURATION` on P improves beyond noise (> 3%) on a fresh kernel cache, and golden `device_kernel_ns` for P moves the same way. The golden suite stays green (exact gate) and `test_g6_reshard.py` passes in plain and `--dev` mode. No regression on the guard set: one representative per distinct kernel path × placement, which here is I0 (all-local, one run per issuer), I3 (COL_MAJOR input, transpose links), I5 (every tile crosses, grid off origin) and O1 (COL_MAJOR output off origin, 5-over-3 shards, 56 cores).

### [ ] Refinement 2 — Trim P's per-issuer fixed cost and co-tune issue order

**Type**: perf

**Goal**: P's per-issuer timeline (Tracy device log, cycles) has three costs outside the link-transfer term:
- **Fixed cost before the first read**: about 300 cycles per issuer. I0, with one 16-tile run per issuer, still spends 310–330 cycles in `RESHARD-ISSUE`: arg loads, sharded `TensorAccessor` construction, first `get_noc_addr`, zone markers.
- **Late BRISC start**: BRISC kernels start ~90–140 cycles after NCRISC on the same core, yet both issuers get equal row halves.
- **Egress collisions**: every issuer starts at its top-left piece, so in P up to 4 out cores pull from the same in core's egress at the same moment.

Speed P up using the relevant ⭐/⭐⭐ levers in `ttnn/ttnn/operations/examples/master.md`:
- `bank_stagger`: rotate each core's piece walk by its block index. This is the design's start-piece lamp.
- `split_reader`: issuer split shape and ratio — even/odd rows, or an uneven split that gives the later-starting RISC fewer rows. This is the design's split lamp.
- Get the first run in flight before the rest of the setup.
- Per-piece `noc_async_read_one_packet_set_state` + `_with_state`, clamped to ≤ 15 tiles per transfer (the design's mechanism cap).

Each lever is a compile-time or host switch, so several can be measured in one phase. Co-tune them, and keep only measured wins. No SUPPORTED change.

**Verifier notes**: these levers are cheap and can be measured independently of each other. They run after Refinement 1 because rotation and split shape change which links are busy at which moment, so their best setting depends on the NoC assignment Refinement 1 lands. This is a soft order, not a hard dependency: if Refinement 1 ends in a measured null result, tune these on the current loop. The ~500-cycle cross-core launch skew is dispatch-side and is not a target here.

**Done when**: Tracy `DEVICE KERNEL DURATION` on P improves beyond noise (> 3%) over the Refinement 1 result on a fresh kernel cache, and golden `device_kernel_ns` for P moves the same way. The golden suite stays green and `test_g6_reshard.py` passes in plain and `--dev` mode. No regression on the guard set I0 / I3 / I5 / O1, as in Refinement 1.
