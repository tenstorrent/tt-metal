# Operation Requirements: distributed_rms_norm

## Definition
- **Formula**: for every device `p` of a group of `G` devices along `cluster_axis`, holding `x[p]` of per-device shape
  `(..., M, W_local)`: `out[p] = x[p] · rsqrt(Σ_g Σ_w x[g]² / (G·W_local) + ε) · γ[p]` (γ ≡ 1 when absent).
- **PyTorch Reference**:
  ```python
  def distributed_rms_norm_ref(xs, gammas, eps=1e-6):          # xs / gammas: the G per-device slices, in group order
      full = torch.cat(xs, dim=-1)
      w = torch.cat(gammas) if gammas is not None else None
      y = torch.nn.functional.rms_norm(full, (full.shape[-1],), weight=w, eps=eps)
      return list(torch.split(y, xs[0].shape[-1], dim=-1))  # device p gets slice p
  ```
- **Import Path**: `from ttnn.operations.distributed_rms_norm import distributed_rms_norm`
- **Function Signature**:
  `distributed_rms_norm(input_tensor: ttnn.Tensor, *, cluster_axis: int, gamma: Optional[ttnn.Tensor] = None,
  epsilon: float = 1e-6, topology=Topology.Linear, num_links: Optional[int] = None,
  compute_kernel_config: Optional[ttnn.ComputeConfigDescriptor] = None,
  memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor` — exactly one `ttnn.generic_op` dispatch.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

> **Queue shape (verifier).** `SUPPORTED == TARGET` on every axis after Phase 0 (plus `num_links=None`), so there are no
> axis values left to add. The generality refinements below move named **failing** cells to passing; the perf
> refinements all target the one perf-flagged case. Perf-1's full contract (bf16, `fp32_dest_acc_en=True`, bf16 TILE γ,
> interleaved, `cluster_axis=1`, Ring, `num_links=2`, 1×4 ring under `FABRIC_2D_TORUS_XY`) is **already in SUPPORTED and
> passing**, so Refinements 1–2 are not needed to unlock it.
>
> **Measuring perf.** Always with the eval harness, never a hand-rolled loop:
> `CCL_PERF_EXPERIMENTS=5 CCL_PERF_OUT=<dir> scripts/run_safe_pytest.sh --no-precompile eval/golden_tests/distributed_rms_norm/perf_measure.py -s`
> (add `CCL_PERF_CASES=all` for the guard cases). Keep or revert with `python -m eval.ccl_perf compare <base.json> <cand.json>`
> (fastest chip lower beyond noise AND slowest chip not higher). Exposed transport = op time − op time under
> `DRMS_ABLATE_EXCHANGE=1` (`tests/ttnn/unit_tests/operations/distributed_rms_norm/perf_ablate.py`, same env vars).
>
> **Measured per-chip breakdown at verification** (Blackhole QuietBox 2, 1×4 ring, `FABRIC_2D_TORUS_XY`, 2 links,
> traced, 5 experiments; µs, fastest chip / slowest chip):
>
> | Case | op | exchange ablated | exposed transport | roofline / target |
> |------|----|------------------|-------------------|-------------------|
> | **640×1792 (PERF FOCUS)** | **23.93 / 25.77** | 20.99 / 21.52 | **2.94** | 9.0 / 11.9 |
> | 32×1792 (decode, guard) | 12.03 / 13.61 | 10.44 / 10.56 | 1.60 | 0.45 / 0.6 |
> | 2048×1280 (prefill, guard) | 44.10 / 47.41 | 34.10 / 34.81 | 10.00 | 20.5 / 27.3 |
>
> At the focus case the local half plus the in-device exchange chain is 21.0 µs against a 9.0 µs DRAM roofline. The base
> op ran its 640-row interleaved shapes at ~363 GB/s effective (640×6144: 43.3 µs on 120 cores), i.e. ~12.6 µs for this
> shape's 4.59 MB. The gap is not the base op's per-core pipeline: in the fused op every core owns one block, so the
> whole grid reads, then the exchange chain (publish → relay → collect → finalize → distribute) runs, then the whole grid
> writes. Overlapping those phases *is* the fusion (OVERLAP). That is why the OVERLAP refinement leads the perf queue.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [float32, bfloat16, bfloat8_b]
- **SUPPORTED fp32_dest_acc_en**: [True, False] (EXCLUSION `{float32, False}`: natively rejected precision combination)
- **SUPPORTED layout**: [TILE]
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned}; rank ∈ {2, 3, 4}
- **SUPPORTED op-specific axes**: gamma_mode ∈ {gamma, no_gamma}; gamma_dtype ∈ {float32, bfloat16, none};
  gamma_layout ∈ {TILE, ROW_MAJOR sticks, none}; memory_layout ∈ {INTERLEAVED, HEIGHT_SHARDED, WIDTH_SHARDED};
  cluster_axis ∈ {0, 1}; topology ∈ {Linear, Ring}; num_links ∈ {1, 2, None (= every usable link)}
- **Cores**: the grid minus 2L port cores (106 / 108 of 110 on Blackhole); regimes R1 / R3-i / R1-HS / R3-s
- **Compute config**: `compute_kernel_config` exposed (default HiFi4 + fp32 DEST); intermediate CBs Float32 under fp32
  DEST, Float16_b under 16-bit DEST; `UnpackToDestFp32` on `cb_stats_gather`
- **Golden baseline** (final verifier run): 10,908 supported_pass / 133 supported_fail (tracked by Refinements 1, 2, 4) / 0 xpass_drift / 1 xfail_wrong_mode (harness) / 2,160 xfail_expected (the native `{float32, 16-bit DEST}` EXCLUSION) / 33,696 invalid_skipped / 40 infeasible_skipped

### [x] Refinement 1 — Port the base op's streaming-γ fallback (L1 fit for wide-row sharded shapes)

**Goal**: move the remaining **120** L1-fit `supported_fail` cells to passing. They are refused today with
`NotImplementedError: the shard does not fit one resident tile-row in L1`. All are `HEIGHT_SHARDED` (one core holds
whole rows) with a γ:
- `(1,1,32,8192)` / `(128,8192)` × {bf16 input with bf16 or fp32 γ, bf8b input with fp32 γ} (16 cells each);
- `(1,1,32,4096)` / `(1,32,4096)` / `(1,1,128,4096)` fp32 input with fp32 γ (8 each).

Arithmetic at `(1,1,32,8192)` bf16: a 512 KiB input shard + a 512 KiB output shard on one core, plus the
**always-resident γ window** (`cb_gamma_tiles` = `Wp` = 256 pages = 512 KiB bf16 / 1 MiB fp32), plus `cb_x_gamma`
(`R·C·T_i` = 128 KiB). The no-γ variants of the same shapes pass. The exact list is in `verifier_report.json` →
`supported_fail` with that message.
This is a **from-base port** (a knob-turn on the γ supply's buffer depth). Phase 0 ported only the base op's resident-γ
path, and the base op's `op_design.md` / `changelog.md` define the fallback:
- `cb_gamma_tiles` resident **when it fits**, else streaming at `stream_depth · C` pages: the writer pushes the γ
  window chunk by chunk per block, and compute pops per chunk instead of holding it for the kernel.
- `choose_blocking` adds the streaming footprint as the second candidate (resident first, then streaming), and so do
  the `cb_inventory` / `footprint` single source.

Also re-measure every L1-fit cell the final golden run still lists, and fold any sharded `fp32` residue into this
refinement (that residue would be the base's `R3-s` multi-block path).

**Implementation skill**: /memory-budget-metal

**Verifier notes**: from-base port, so it goes first (prompt rule). Any variant of the γ supply must keep the
`cb_inventory()` single source: the CB descriptors and the L1 solve read the same list. If a cell still cannot fit
because the input shard and the output shard together leave less than one tile-row of CBs, that is a structural
capacity limit; record it in the changelog with its byte arithmetic rather than adding an EXCLUSION.

**Done when**: every golden `supported_fail` cell carrying the "does not fit one resident tile-row" message passes (or
is documented with its byte arithmetic as a structural L1 limit), and the acceptance suite plus Phase 0 golden cells
stay green.

**Outcome**: all 120 target cells pass. Streaming γ (`STREAM_DEPTH·C` ring) is chosen only when the resident window
does not fit, so Phase 0 plans are unchanged. It also fixed an order-dependent CB clash: `l1_budget` is now bounded
by the lowest known L1 buffer, including the persistent semaphores. No structural L1 limits remain in this set.

### [ ] Refinement 2 — Output placement independent of the input placement

**Goal**: move the translated model calls that ask for a **different output placement** than the input's from
`NotImplementedError` (regime refusal (b)) to passing. Concretely, `test_translated.py::test_simulated_distributed_layernorm`:
a WIDTH-sharded input with a WIDTH-sharded output on a different core grid / offset (12 cells). Build in the kernels:
1. **sharded in → interleaved out**: keep the shard-backed `cb_x_tiles` and switch `cb_output` to the interleaved
   writer path (both already exist in the op).
2. **interleaved in → sharded out**: the interleaved reader with the writer writing into the output shard's cores
   through a `TensorAccessor` of the sharded output.
3. **sharded in → sharded out on a different grid**: the writer addresses the output's shard grid through the output
   `TensorAccessor`. That is a genuinely non-local write, so `TensorAccessor` is the right tool.

The `custom_program_hash` must cover both placements.

**Verifier notes**: placement only, no new math. Classified "complicated sharding" (sharded↔interleaved reshards), so
it is ordered ahead of Refinement 4. Do **not** implement it as a `ttnn.to_memory_config` after the op: that is a second
dispatch and violates the one-dispatch rule. There is no implementation skill for `memory_layout` placement yet, and
`/memory-layouts` covers RM/TILE, not placement. The design's deferred regime row "Sharded in / interleaved out" is
the starting point.

**Done when**: the 12 `test_simulated_distributed_layernorm` placement-mismatch cells pass, `validate()` no longer
raises regime refusal (b) for any placement pair the kernels now handle, and the golden suite stays green.

### [ ] Refinement 3 — OVERLAP: pipeline read → exchange → write on the perf-flagged case

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES[0]` is the mandatory perf target (`attention: PERF FOCUS`): `(1,1,640,1792)` bf16,
`fp32_dest_acc_en=True`, bf16 TILE γ, interleaved, `cluster_axis=1`, Ring, `num_links=2`, on the 1×4 ring
(`roofline_us` 9.0, `target_us` 11.9). Measured: 23.93 µs fastest chip, 2.94 µs of it exposed transport, 20.99 µs
with the exchange ablated. **Label: OVERLAP** (measured only in the op, as the exposed transport and the ablated
time on the fastest chip). Today every one of the 100 cores holds one block (R3-i, `(S, G_w, G_h) = (6, 10, 10)`, 2
tile-rows per row-group, 2 waves of 1 row). So the grid's reads, the exchange chain and the grid's writes run as
three serial phases, and only wave 0's writes overlap wave 1's exchange. Levers, all on knobs the planner exposed
(`op_design.md` → Perf lamps):
- **L1 + L2 co-tune.** Re-shape the row-groups so each root walks more waves: fewer, wider row-groups (`G_w`) with
  more rows per group (`R3_ROWS_PER_GROUP`), and `NUM_WAVES` up to the group's rows. The first wave's exchange then
  starts after one tile-row's reads, and later waves' reads and writes run under it.
- **L4: notify quantum.** Notify per landed entry, so the root reads slot q while q+1 relays.
- **L5 / L6: owners per group, inv_rms distribution.** If zones show the root's collect → finalize → distribute
  serial chain on the critical path.
- **Shorten the exposed tail.** collect → finalize → distribute → scale → store of the last wave: hoist more of the
  `scale_by_gamma` chunks ahead of the inv_rms wait.

Per-stage device zones on the root's reader and compute (`rt_publish`, `rt_landed_wait`, `rt_collect`,
`rt_distribute` are named in the design and not yet instrumented) give the breakdown. Only the harness gives the
headline.
Catalog: `ttnn/ttnn/operations/examples/master.md` → `fabric_ring_exchange` (the fence and relay floor this overlap has
to hide), `compute_block_size`, `double_buffer`. No SUPPORTED change.

**Done when**: `perf_measure.py` on the focus case shows the fastest chip lower beyond the noise and the slowest chip
not higher (`python -m eval.ccl_perf compare`), and the exposed transport is re-measured with `DRMS_ABLATE_EXCHANGE`.
Golden suite green. No regression across the config-spanning guard set: one representative per distinct kernel path ×
layout × placement, which includes the other two `LOOSE_CASES` (32×1792 R3-i decode, 2048×1280 R1 prefill), a
`HEIGHT_SHARDED` and a `WIDTH_SHARDED` cell, and a Linear `cluster_axis=0` cell.

### [ ] Refinement 4 — Bounded exchange landing for long sequences (TRANSPORT scheme change)

**Goal**: move `test_translated.py::test_ds_distributed_norm[random_weights-prefill-131072-0.98]` (and any longer
sequence) to passing. Today it raises a clean `NotImplementedError: the exchange landing needs 1048576 B per port core`.
Before the verifier fix this call hit an allocator `Out of Memory`, and prefill 32768 hit an `assert`. 32768 now passes
because the plan is re-solved against the landing's real bytes. The landing holds every (wave, slot) of the call
(`num_waves · G · seg_stride` bytes, per call, over the whole grid) with no reuse, so it grows with T.
**Label: TRANSPORT.** Make the landing a bounded ring of `K` wave slots reused across waves:
- a receiver's root returns a credit once it has collected a wave;
- a sender does not overwrite slot `w mod K` of wave w until the destination's credit for wave `w − K` has arrived.

This is the credit protocol `blackhole-ccl-kernels.md` §7 trades away for small calls. Keep the current credit-free
layout as the fast path whenever the whole call fits (the design's "reuse-free layout removes the credit round trip"
decision). Gate on a predicate; don't replace it.

**Verifier notes**: a scheme change (a new flow-control topology on the landing), so it stands alone. It must keep
every global semaphore re-armed to 0 at the end of each call (trace safety) and keep the cross-call ready fence
semantics. Refinement 5 reworks the same landing protocol, so do this one first.

**Done when**: the prefill-131072 cell passes (or is documented as exceeding another, named limit), the credit-free
fast path is unchanged on every INPUTS shape (no perf regression on the focus case, measured with the harness), and
the golden suite stays green.

### [ ] Refinement 5 — TRANSPORT: the per-call ready fence and the link count on the focus case

**Type**: perf

**Goal**: the perf-flagged case (`(1,1,640,1792)`, same contract as Refinement 3), **TRANSPORT** side. The
`fabric_ring_exchange` example measured the minimal correct 1×4 ring exchange at ~5.0 µs, almost all of it the
cross-call **ready fence** (each port waits for its peer to launch; 5.09 µs vs 5.38 µs with the 128 B relay). It also
found that a second link does not help payloads this small, because each link pays its own fence. The focus case
sends 2.5 KiB per chip per wave. Levers:
1. **Fence-free steady state.** Double-buffer the landing by call parity (epoch bit in the RT args or a per-call
   epoch counter), so a sender may write call k+1 into the half the receiver is not reading, and the fence leaves the
   per-call critical path. **Trade-off to measure:** a fence-free write must land in memory that exists before the
   receiver's program starts, so the landing has to become persistent again. The verifier made it per call because a
   persistent landing, created mid-call, pinned the middle of every L1 bank and shrank later calls' CB space. If this
   lever wins, allocate the persistent landing where it cannot fragment (e.g. at mesh-resource creation, before any
   caller tensor, or in a reserved region) and re-run the long-session golden suite to prove no L1 ratchet. Gate it on
   the same correctness argument as the current fence (`blackhole-ccl-kernels.md` §6.3).
2. **Links.** The case pins `num_links = 2`. Measure whether carrying both links' records but sharing **one** fence
   per port pair (the second link's ports fence-free) recovers the 1-link latency inside the 2-link contract.
3. **Port placement (lamp L8)**, measured as TRANSPORT only.

Measure TRANSPORT in isolation (exchange time against the bytes sent, `fabric_ring_exchange`'s harness) and cite the
example's numbers, instead of re-deriving the fence cost. The headline still comes from `perf_measure.py`.

**Done when**: the focus case's fastest chip is lower beyond the noise, the slowest chip is not higher, the spread
(launch skew) is reported, the golden suite stays green (including `test_distributed_rms_norm_debug.py::test_repeat_rows`
bit-identical repeats), and there is no regression across the guard set.

### [ ] Refinement 6 — Local-half headroom on the perf-flagged case (only if Refinements 3 and 5 leave it)

**Type**: perf

**Goal**: after Refinements 3 and 5, re-measure the focus case. If the exchange-ablated time is still well above
the base op's measured efficiency for this shape (~12.6 µs at the base's 363 GB/s; roofline 9.0 µs), speed up the
local half on the focus case. Candidate levers from the base op's own perf rounds and the catalog:
- **G-MC (γ read once + multicast per window group)**, the base op's Perf 1 win (8192×5120 interleaved 1.11×). Its
  groups must exclude the port hole (design regime row G-MC, deferred).
- `noc_placement` / `double_buffer` on the read and write phases.

If the re-measured local half is within noise of the base op's efficiency, close this refinement as measured-null
in the changelog.

**Done when**: a measured device-ns win on the focus case (fastest chip lower beyond noise, slowest not higher) with
the golden suite green and no regression across the guard set, or a documented measured null.
