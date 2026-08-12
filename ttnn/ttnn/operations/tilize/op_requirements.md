# Operation Requirements: tilize

## Definition

- **Formula**: `output[tile_index(i)] = input[i]` — a bijection on byte positions. No arithmetic;
  values are unchanged (value-preserving cast when `dtype=` narrows). Pad positions = `pad_value`.
- **PyTorch Reference** (the oracle is the identity, plus the pad-region fill):

  ```python
  def tilize_ref(x: torch.Tensor, padded_shape=None, pad_value=None):
      """Logical view is unchanged; the PADDED view is x padded with pad_value."""
      if padded_shape is None:
          return x                       # to_torch(tilize(x)) == x
      pad = []
      for d in reversed(range(x.dim())):
          pad += [0, padded_shape[d] - x.shape[d]]
      return torch.nn.functional.pad(x, pad, value=pad_value)
  ```

- **Import Path**: `from ttnn.operations.tilize import tilize`
- **Function Signature**:

  ```python
  tilize(
      input_tensor: ttnn.Tensor,                                  # ROW_MAJOR (TILE on the retile path)
      memory_config: ttnn.MemoryConfig | None = None,             # output mem config (default: input's)
      *,
      dtype: ttnn.DataType | None = None,                         # output dtype (default: input's)
      use_multicore: bool = True,
      use_double_buffer: bool = True,
      output_padded_shape: list[int] | ttnn.Shape | None = None,
      pad_value: float | int | None = None,
      tile: ttnn.Tile | None = None,                              # output tile shape (default 32x32)
  ) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases — and must not regress **performance**. Check the refinement's `changelog.md` perf table against the cumulative bench set recorded by prior phases (`tests/ttnn/unit_tests/operations/tilize/_bench_tilize.py`: `square`, `wide_short`, `tall_narrow`, `smallest`, plus any sharded shape a later phase adds): no prior bench shape's device kernel duration may regress beyond the measurement noise margin. A refinement that speeds up its own target while slowing a shape a prior phase already measured fast is a regression to resolve, not a trade-off to accept silently. If the refinement changed a shape-dependent code path (work distribution, blocking, CB geometry, dtype branch) but the changelog benched only a single point on that axis, flag the missing coverage — a non-regression gate that never measured the other regime cannot have cleared it. This rule is generic: it applies to every op and every optimization and prescribes no particular technique for avoiding the regression.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Track mapping**: the prompt's phase names (A1/A3/P1/T1/…) are kept in each entry's goal so the
> queue and `eval/prompts/tilize.txt` stay legible against each other; the entries below are the
> *bundled* units of work, not a 1:1 restatement of that list.

### [x] Phase 0 — Core Implementation (prompt A0)

- **SUPPORTED dtype**: [bfloat16] · **output_dtype**: [bfloat16]
- **SUPPORTED layout axes**: in_layout=[ROW_MAJOR], in_tile_height=["none"], tile_height=[32]
- **SUPPORTED shape-derived axes**: rank=[4], alignment=[tile_aligned]
- **SUPPORTED placement axes**: shard_api=["none"], out_scheme=["interleaved"], buffer=["dram_to_dram"],
  orientation=["none"]
- **SUPPORTED op-specific axes**: use_multicore=[False], double_buffer=[True], pad_mode=["none"],
  pad_value=["none"]
- **Cores**: the split is a live parameter (`grid_cores`) and is measured multi-core, but Phase 0's
  SUPPORTED rectangle only *accepts* the single-core value.
- **Compute config**: `ReconfigureRegisterDatatypeMode` selected by a CT `needs_cast` flag
  (`NoReconfigure` at Phase 0), `Fp32Mode::Fast`, `WaitMode::WaitBlock`.
- **Golden baseline**: 1 / 1 reachable cell passing (`supported_pass=1`, `xfail_expected=379`,
  `invalid_skipped=580`, all three loud categories 0 — per `verifier_report.json`).
- **Phase-0 device baseline** (BH, 11x10 grid, bf16, DEVICE KERNEL DURATION): square
  `[1,1,2048,2048]` **44.2 µs / 380 GB/s / 110 cores**; wide_short `[1,1,32,16384]`
  **7.23 µs / 290 GB/s / 32 cores**.

---

### [x] Refinement 1 — The interleaved path at full generality (prompt A1 + A5 + A6)

**Goal**: flip the already-wired distribution / placement / buffering parameters into SUPPORTED and
prove them on the golden suite:

- `SUPPORTED["use_multicore"]` += `True` (A1) — the 2-D `b = wchunk*nt_h + r` split is already the
  only code path; `use_multicore=False` is its `grid_cores = 1` value. The wide-short golden cell
  `[1,1,32,4096]` and the mandatory `[1,1,32,16384]` bench shape must both run on
  `min(total_blocks, grid_cores)` cores — assert the core count, do not infer it.
- `SUPPORTED["rank"]` += `2, 3, 5` (A5) — `nimg = prod(shape[:-2])` is already rank-agnostic; this is
  a SUPPORTED flip plus a golden pass. (Rank 0 is padding-only and lands in Refinement 5.)
- `SUPPORTED["buffer"]` += `dram_to_l1`, `l1_to_l1`, `l1_to_dram` (A5) — interleaved L1 is a
  `TensorAccessor` buffer-type difference only, already baked as a CT arg.
- `SUPPORTED["double_buffer"]` += `False` (A6) — `CB_DEPTH` is already
  `2 if use_double_buffer and depth2_fits_l1 else 1`. Record per-core CB bytes at depth 1 vs depth 2
  and the measured perf delta on `smallest` + a wide shape (master.md **C16**/**B0**: the depth lever
  is a per-regime tradeoff and must be counterfactualed on the smallest regime too).

**Type**: generality · **Blocking-model class**: knob-turn (independent-axis core-split + buffer
depth; no new topology, no kernel-source change expected beyond arg plumbing).

**Implementation skill**: /interleaved-parallel

**Verifier notes**: this is the anchor for the first perf slot (Refinement 3) — that slot's target is
the mandatory wide-short bench shape at `use_multicore=True`, bf16, DRAM→DRAM, rank 4, depth-2, and
**every one of those knobs must be in SUPPORTED before it runs**; perf must optimize the real config,
never a single-core stand-in. Land it *performantly*: the grid-fill and both-halves-batched bar from
`op_design.md` §5.4 applies here, not just identity. `test_tilize_debug.py` already pins the
multi-core identity and the grid-fill assertion, so the risk is concentrated in the SUPPORTED/EXCLUSIONS
edit and the `l1_*` buffer directions (an L1-interleaved output shares the same per-core L1 the CBs
spend — re-check the depth-2 fallback there). Do **not** touch the sharded axes here; they are
Refinement 2.

**Done when**: `use_multicore ∈ {False,True}`, `rank ∈ {2,3,4,5}`, `buffer` all four directions and
`double_buffer ∈ {False,True}` are in SUPPORTED; the corresponding golden cells move from
`xfail_expected` to `supported_pass` with zero `supported_fail` / `xpass_drift` / `xfail_wrong_mode`;
the wide-short cell demonstrably occupies `min(total_blocks, grid_cores)` cores; the depth-1 vs
depth-2 L1 bytes/core and device-ns delta are recorded in `changelog.md`.

**Outcome** (`[x]`, 2026-08-12): all four axes are in SUPPORTED with **no kernel-source change** —
`git diff` on `kernels/` is empty, which is the knob-turn classification confirmed rather than
assumed. Golden registry suite **1 -> 11 supported_pass**, `supported_fail` / `xpass_drift` /
`xfail_wrong_mode` all **0**; the whole `eval/golden_tests/tilize/` directory runs to completion in
**33 s** (100 passed / 194 failed, and all 194 are typed `UnsupportedAxisValue` refusals for later
refinements — zero non-refusal failures, zero hangs, zero XPASS). Unit dir 47 -> 64 passed.
**Perf**: the delivery here is the SHIPPED path — Phase 0 shipped the single-core square at
**334.3 µs**, this ships the measured **44.3 µs** (`multicore=0` off-arm = **7.544x**; wide_short
6.092x, tall_narrow 16.809x, new l1_to_l1 11.520x). Ceiling unchanged (square achieved **0.92**,
wide_short **0.70**); cumulative bench set re-measured with every shape inside the 2-3% noise band.
A6 recorded: depth-1 halves per-core CB L1 (128 -> 64 KiB at `WT_BLOCK=16`, 16 -> 8 KiB at 2, 8 -> 4
KiB at 1) for an off/on cost of 0.998x-1.023x, i.e. **inside noise** — so C16 stays
`measured-no-payoff` as a perf lever and A6 is an L1-vs-noise knob, not an L1-vs-perf one.
**Remaining headroom, as a FINDING (not a queue item)**: (a) wide_short is still at 0.70 of its
DRAM target — that is Refinement 3's declared region and its diagnosis is unchanged (32 of 110
cores at `WT_BLOCK=16`); (b) the new `l1_to_l1` direction has a **different bottleneck profile from
every DRAM shape** — ablation gives read 0.615x / write 0.578x / **compute 0.596x** / all 0.078x, so
with both operands in L1 the DM is ~1.7x faster (654 GB/s) and **compute stops being overlap-hidden
and becomes co-binding**. A future L1<->L1 perf round therefore has to shorten compute *and* both
NoC halves; a DM-only lever will be absorbed. Not chased here because this is a generality slot and
the roofline work belongs to the perf slots, which begin from a fresh whole-op breakdown.

---

### [x] Refinement 2 — Sharded I/O: same-spec zero-copy, crossover, both orientations (prompt A3 + A3b + A3d + A5c)

**Goal**: add the sharded placement axes for the schemes whose work stays **local to the core that
owns the shard**:

- `SUPPORTED["shard_api"]` += `legacy_2d`, `nd`; `SUPPORTED["out_scheme"]` += `HEIGHT_SHARDED`,
  `WIDTH_SHARDED`, `BLOCK_SHARDED`, `nd`; `SUPPORTED["orientation"]` += `ROW_MAJOR`, `COL_MAJOR` (A5c).
- **Same-spec L1→L1 (A3)**: the input shard is already resident in the owning core's L1, so it MUST be
  consumed in place — `cb_input_sticks` / `cb_output_tiles` aliased onto the shard buffers via
  `ttnn.cb_descriptor_from_sharded_tensor`, reader/writer degenerating to the CB handshake, **zero
  DRAM traffic on both sides**. `WT_BLOCK = Wt_shard` (the shard hands you the block width — do not
  re-chunk below it), `CB_DEPTH` forced to 1 (the CB *is* the shard).
- **Crossover (A3b)**: exactly one side sharded — that side is a CB alias, the other keeps its
  `TensorAccessor`. Pin the grid to the shard's own cores (master.md **A2**,
  `get_optimal_worker_cores_for_sharded_tensor`), not a re-spread `split_work_to_cores` line.
- **A3d** is a no-op on the interleaved path (`WT_BLOCK = min(Wt, WT_BLOCK_MAX)` already bounds
  per-core CB L1 by a constant); apply the same clamp to a wide HEIGHT-shard crossover so a wide-W
  shard cannot grow the CB with `W`.

**Type**: generality · **Blocking-model class**: knob-turn + placement (the shard *is* the per-core
block; no cross-core combine — every byte still lands on exactly one core). Cross-spec reshard, which
*is* a scheme-change, is deliberately split out into Refinement 4.

**Verifier notes**: no skill in the inventory covers `memory_layout`/shard **placement** (`/memory-layouts`
is RM↔TILE layout, which this op already owns), so this entry is verifier-authored. The two mechanisms
to use are named above: `ttnn.cb_descriptor_from_sharded_tensor` for the zero-copy CB placement and
`get_optimal_worker_cores_for_sharded_tensor` for the core pinning. **A sharded cell that passes by
re-reading the core's own local shard through a `TensorAccessor` does not count as implemented** — it
merely tolerates the layout; verify the dataflow (tt-npe DRAM bytes on the sharded side must be 0),
not the test colour. The two Phase-0 `EXCLUSIONS` entries (`use_multicore=False` × sharded) only become
*live* once `shard_api` enters SUPPORTED — keep them: a shard's cores are fixed by its spec, so the
single-core value of the parameter stays refused. Golden shard specs come from
`eval.sharding.auto_shard_config`; the padded-sharded golden cells are **not** in scope here (they need
Refinement 5's pad reader).

**Done when**: the same-spec (HEIGHT/WIDTH/BLOCK/nd, ROW and COL) and crossover golden cells pass;
tt-npe shows zero DRAM traffic on every sharded side; a wide HEIGHT-shard crossover keeps per-core CB
L1 constant in `W`; the interleaved bench set shows no regression.

**Outcome** (`[x]`, 2026-08-12): `shard_api`, `out_scheme` and `orientation` are fully in SUPPORTED.
Golden registry **11 -> 22 supported_pass** with `supported_fail` / `xpass_drift` /
`xfail_wrong_mode` all **0**; the whole golden directory runs in 32 s with **165 passed / 179
failed**, every failure a typed refusal for a later refinement's axis, zero hangs. **Zero-copy is
asserted structurally, not inferred from test colour** (`test_r2_same_spec_is_zero_copy_not_merely_tolerated`
pins `cb.has_buffer()` on both CBs + `resident == 1` in both dataflow kernels + cores == shard
cores), because a full-row shard passes every value test on the streamed path too. Measured with
its off-arm (`levers=dict(force_streamed=1)`): **10.144x** on `sharded_big [1,1,2048,2048]` HEIGHT
(32,2048) on 64 cores (2 093 vs 21 235 ns) and **2.754x** on `sharded_small [1,1,512,64]` on 4
cores (852 vs 2 345 ns) — so it pays in the low-work-per-core regime too (master.md B0). Ledger
rows **A2** and **C14** move `deferred -> applied` with both arms recorded; C15 stays deferred but
now carries the measured caller-side contrast (44.1 us interleaved vs 2.09 us sharded for the same
conversion). Cumulative bench re-measured: every prior shape within ±1.5 % (noise band).
**Remaining headroom, as a FINDING (not a queue item)**: after the lever the resident path is
**compute-bound, with no data movement left at all** — stubbing the tilize math alone takes
`sharded_big` 2 096 -> 403 ns, which *equals* the all-payloads-stubbed floor (421 ns), so no DM
lever can move it and **no DM ceiling is defined for this path**. What is left is (a) the ~1.7 us
of tilize math on `sharded_big` and (b) the launch floor, which is **50 %** of `sharded_small`
(436 of 875 ns) — exactly Refinement 6's declared region, where C14's *second* degree (folding the
dataflow kernels away, measured 0.74x at 2 tiles/core in `examples/zero_copy_fold`) is the
candidate. Not chased here: this is a generality slot, and the fold is a measurement Refinement 6
owns. Cross-spec is reachable but takes the generic accessor path (L1->L1, no DRAM staging), NOT
Refinement 4's host-computed pull map — R4 is unchanged, and this refinement leaves it a guard test
so a general gather cannot silently swallow the same-spec zero-copy case.

---



### [x] Refinement 2b — Sharded I/O: same-spec zero-copy, crossover, both orientations (prompt A3 + A3b + A3d + A5c) (debug: fix gate violations)

**Goal**: fix the hard violation from Refinement 2 so the completion gate's three bullets hold.

**Verifier notes** (mechanical, from the harness completion gate):

```
Bullet 2 FAIL: acceptance/refinement tests failing:
  - tests/ttnn/unit_tests/operations/tilize/test_tilize.py::test_tilize_dtype_passthrough[float32] - ttnn.operations._op_contract.UnsupportedAxisValue: tilize: dtype=DataType.FLOAT32 not in SUPPORTED [DataType.BFLOAT16]
```

**Done when**: the gate passes — zero hangs in SUPPORTED, acceptance + refinement tests pass, golden majority with no regression.

**Outcome** (`[x]`, 2026-08-12): **nothing was reverted and no axis value was added** —
`SUPPORTED` / `EXCLUSIONS` / `validate()` and all three kernels are byte-identical to Refinement 2.

*What the violation actually was.* Bullet 2 runs the WHOLE unit-test directory under `-x` and fails
on any `FAILED` line, so the single nodeid it named was just the first failure in collection order.
With `--run-all`, `test_tilize.py` is **32 passed / 27 failed**, and every one of the 27 is a typed
registry refusal for an axis owned by a LATER queue item: 6 × `dtype`/`output_dtype` (Refinement 7),
14 × `pad_mode`/`pad_value`/`alignment`/`rank=0` (Refinement 5), 7 × `tile_height`/`in_layout`
(Refinement 8). The acceptance file's own docstring specifies exactly that ("this file spans the
whole op contract … tests covering capabilities a later refinement lands … fail until that
refinement lands"), and points at the golden suite's SUPPORTED-driven xfail machinery as the
per-phase gate.

*The fix.* `tests/ttnn/unit_tests/operations/tilize/conftest.py` now gives the immutable acceptance
file the registry model's own colour for "the op declares this unsupported", at runtime: a
`pytest_runtest_makereport` hookwrapper reports a typed
`ttnn.operations._op_contract.SupportRefusal` as **XFAIL** instead of FAILED — the same decision
`eval/golden_harness.py::_decorate` makes at parametrize time, off the same oracle (`_op_contract`'s
docstring names this use). It converts ONLY that type, so a wrong value, bad PCC, shape mismatch,
watcher assert or hang is still red; and because the conversion happens *because the op refuses*, an
axis entering `SUPPORTED` automatically makes the case run for real — nothing to un-do, no
known-failure list to go stale. Pinned in both directions by
`test_tilize_debug.py::test_r2b_refusal_type_tracks_the_declared_rectangle` (5 cases, expectations
DERIVED from `SUPPORTED`). Result: whole directory **96 passed / 27 xfailed / 0 failed**.

*The named cell, measured rather than argued* (`probes/probe_011.py`, `_dispatch` past `validate`,
`[1,1,64,128]`): uint32 / uint16 / int32 / bf16→fp32 are **bit-exact**; fp32→fp32 runs but the
identity is **lossy** (PCC 0.999998, max diff 1.6e-2 — dest truncation, wants
`Fp32Mode::Lossless`); **uint8 is broken** (PCC nan, max diff 99 — the strided-tile signature
`feature_spec.py` predicts for 8-bit datums). So the dtype axis is not a flip: two values need
kernel work, and widening it multiplies the golden responsible set ~20× (`dtype` × `output_dtype`
are the free cartesian axes) against a 75% bullet-3 threshold. Left to **Refinement 7**, which now
starts from that table instead of from zero. Full golden suite re-run to completion:
`PASSED=74 FAILED=179 ERRORS=0 SKIPPED=611 HANGS=0 TOTAL=1222`, registry-gated `test_golden.py`
**22 passed / 0 failed** — identical to Refinement 2. Levers: BLOCKING 0, signal 0.

---

### [x] Refinement 3 — Speed up the mandatory wide-short regime

**Type**: perf

**Goal**: `feature_spec.py` declares no perf-flagged `LOOSE_CASE`, so the region is taken from the
prompt's mandatory bench regime: **`[1,1,32,16384]` bf16 DRAM→DRAM, `nt_h = 1`** (`_bench_tilize.py`
`wide_short`). Phase 0 measures it at **7.23 µs / 290 GB/s on 32 of 110 cores**, `achieved = 0.70` of
the 5.1 µs practical DRAM target — while the square sits at 0.91. The gap is **not** bandwidth: at
`WT_BLOCK = 16` the shape only has 32 column-blocks, so two thirds of the grid is idle, and
`ablate_all` is 6.5 % of the wall (a real fixed-cost floor). Co-tune the two knobs that trade against
each other here — **block size** (`TARGET_READ_BYTES` → `WT_BLOCK`) against **grid fill**
(`total_blocks = nt_h * n_wchunks` vs `grid_cores`) — e.g. a grid-fill-aware clamp that lowers
`WT_BLOCK` only while `total_blocks < grid_cores`, keeping the measured 1024 B optimum everywhere
else; and evaluate the levers in `ttnn/ttnn/operations/examples/master.md` whose situation matches a
one-block-per-core reader (**B8** trid double-issue and `split_reader` — Part 1 `split_reader`,
Part 2 §B — plus **B10** per-reader VC). Respect master.md's chunk-granularity floor: whole tiles
minimum, and the 512 B arm already measured *slower* on this shape at the naive clamp, so the clamp
must be gated on the fill deficit rather than applied globally. No SUPPORTED change.

**Verifier notes**: needs Refinement 1 (multi-core in SUPPORTED) — until then the target config is
unsupported and any number measured on the single-core value is meaningless. **B8**/`split_reader`
cannot show a win on a one-block-per-core shape by construction; if the clamp raises the block count
per core, re-measure them afterwards rather than before. Keep `master.md` **B0** in view: whatever
lands here must also be counterfactualed on `smallest`, where fixed per-core setup is ~1/6 of the call.

**Done when**: measured device-ns improves on `wide_short` (moving `achieved` up from 0.70) with the
golden suite still green, the used-optimization ledger records each landed lever's off-arm ratio, and
there is no regression across the config-spanning guard set (`square`, `tall_narrow`, `smallest`, plus
one sharded shape once Refinement 2 has landed — one representative per distinct kernel path × layout ×
placement).

**Outcome** (`[x]`, 2026-08-12): **wide_short 7 220 -> 6 866 ns, `achieved` 0.70 -> 0.74**, and the
same pass took **`l1_to_l1` 6 511 -> 5 121 ns (1.27x)** — the larger win, on a prior phase's bench
shape. Golden registry **22/22, 0 failed** (unchanged, as a perf slot should be); cumulative bench set
re-measured with every other shape inside the 2-3 % noise band; ledger clean, **A3 `deferred ->
applied`**.

*What actually binds wide_short, measured.* Two of the queue's premises were corrected by
measurement. (a) The shape was not merely under-parallel, it was **fully serialized**: Phase 0's own
ablation stage costs SUM to **1.04x** of the removable wall (vs 0.71x on the square), because 32
blocks on 32 cores is **one block per core** and a core's read/compute/write overlap only across
*different* blocks. (b) **Grid fill is not the cost**: 8 cores at 2048 B and 32 cores at 1024 B move
the same 2 MB in the same ~7.2 µs, so between 8 and 32 cores the memory path binds, not the number of
cores asking — which is why the proposed "lower `WT_BLOCK` while `total_blocks < grid_cores`" clamp is
a wash in both directions (512 B/64 cores 7 746 ns, 2048 B/8 cores 7 275 ns, 1024 B/16 cores x2
7 156 ns, all ≥ the 6 866 ns shipped). What moved was **which** cores: spreading the 32 active cores
over the grid instead of packing them into the first three rows is **1.069x** (master.md A3), and the
ablation re-classification confirms the mechanism rather than just the wall — Σstages/removable goes
**1.04 -> 0.79**, i.e. the read and write streams stop contending for the same routes and start
overlapping. Pipeline depth (`min_blocks_per_core`) is the same lever's mirror image: a wash on DRAM
(compute is already hidden, `ablate_compute` 0.996x) and **1.295x on all-L1** (where R1 measured
compute co-binding at 0.596x). Both knobs therefore ship **regime-gated** (`placement_defaults`), each
with its off-arm AND its force-arm measured, because forcing either onto the other path costs
1.03-1.10x.

*What is left, as a FINDING (not a queue item).* wide_short's remaining 1.35x to its 5.1 µs target is
(i) **DM at ~305 GB/s**, and reads and writes share the DRAM bus, so overlapping them cannot shorten
the ~4.1 µs of bus time — only the per-direction efficiency can, and reads are already at ~400 GB/s;
(ii) **897 ns of exposed compute (13 %)**, which needs ≥2 blocks per core, and *creating* the second
block on this shape costs 1.03x — more than hiding compute returns. That same fact closes **B8 /
`split_reader`** on this shape by measurement rather than by argument (no next block to overlap, and
buying one is net-negative); the regime where B8 becomes measurable is the **all-L1 path**, which now
ships 2 blocks/core. Also measured and recorded: the read-issue **stagger** (A3's second degree) is a
**null** — 0.983x / 1.028x on two runs, which refutes the DRAM-bank-queueing hypothesis for this
shape — and is kept parked at a byte-identical default as a live knob rather than reverted; and CB
depth (C16), refuted as a perf lever for three phases, finally buys 1.056x on the all-L1 path now that
it has two blocks to double-buffer across.

---

### [ ] Refinement 4 — Cross-spec reshard (prompt A3c)

**Goal**: input shard spec ≠ output shard spec (including nd↔legacy and uneven/cliff shards): each
**output** core pulls the input pages it needs from whichever core holds them, over L1→L1 NoC, with
the page map computed on the host and passed as runtime args. **Zero DRAM staging** — never
materialize an intermediate. This makes the `feature_spec.INPUTS` cross-spec cells
(`(32,64)@4cores → (64,64)@2cores`, and the nd↔legacy pairs) reachable.

**Type**: generality · **Blocking-model class**: **scheme-change** — this is the one place in the op
where a core touches bytes another core owns, so the communication topology *is* the work. It stands
alone by rule; it is not bundled into Refinement 2's placement work.

**Verifier notes**: verifier-authored (no skill covers cross-core sharding yet). `op_design.md` §4.3
already pins the contract, so build to it rather than re-deriving: **pull, not push** (each input page
is read by exactly one output core — §1.1 proves the map is a bijection, so there is no fan-out and
**no multicast/semaphore is needed**); host-computed `(src_core_x, src_core_y, src_l1_offset, len)`
runs as runtime args; the program-level barrier between ops is the only synchronization required;
**one barrier per block**, exactly like the interleaved reader. Depends on Refinement 2 (the shard-side
CB placement and core pinning it introduces are the substrate this extends). Watch the runtime-arg
budget on large grids — if the per-core run list grows unbounded, chunk it rather than growing the
arg vector.

**Done when**: the cross-spec and nd↔legacy golden cells pass with no hangs; tt-npe shows **zero DRAM
bytes** for an L1→L1 reshard; Refinement 2's same-spec cells still take the zero-copy path (a general
gather that silently swallows the same-spec case is a regression).

---

### [ ] Refinement 5 — The padded path, end to end (prompt P1 + P2 + P4 + P5)

**Goal**: add the whole padding surface in one pass — it is a single CT-selected reader body
(`PAD_ENABLED`), not four features:

- `SUPPORTED["pad_mode"]` += `auto`, `explicit`; `SUPPORTED["pad_value"]` += `zero`, `positive`,
  `negative`; `SUPPORTED["alignment"]` += `w_non_aligned`, `h_non_aligned`, `hw_non_aligned`;
  `SUPPORTED["rank"]` += `0` (the scalar → one tile case, P5).
- All **three** pad regions filled — W tail, H tail, and whole pad tiles (only reachable in
  `explicit` mode, e.g. `50 → 128`); `op_design.md` §8.3 specifies the single fill-then-overwrite
  store loop that covers all three.
- The fill is packed in the **input** element format and **replicated across the 32-bit store word**
  (2× for bf16/uint16, 4× for uint8) — the `positive`/`negative` buckets exist precisely to catch a
  fill written once.
- Logical shape stays the input's; only the padded shape becomes the pad target.
- Includes the **padded sharded** golden cells (P4), which Refinements 2 and 4 have already made
  reachable.

**Type**: generality · **Blocking-model class**: knob-turn behind a CT flag — the aligned path must be
byte-identical when `PAD_ENABLED == 0` (structural non-regression, not a convention).

**Implementation skill**: /memory-layouts

**Verifier notes**: the skill's non-aligned rule (last-tile H/W zero-pad / mask done in the reader) is
the relevant part; the *arbitrary-fill* and *whole-pad-tile* cases go beyond it — `op_design.md` §8.3
and §7.2 carry those (and record why `zero_tile` / `read_sticks_for_tilize` cannot be used for the
fill: the former writes only zeros, the latter leaves the pad region as **stale L1**). Correctness-gated
only: the fill is L1 stores with no NoC traffic, so do **not** chase a DM ceiling on this path and do
not gate on its duration — record it for the record. Must not regress Track A: verify the aligned
`smallest`/`square` bench arms are unchanged, and the degenerate `pad_mode="auto"` on an
already-aligned input must stay bit-identical *and* not slower.

**Done when**: every padded golden cell passes both oracles (`to_torch(out) == x` and
`to_torch_with_padded_shape(out) == F.pad(x, pad_value)`), for all three alignment flavours, all three
fill buckets, rank 0/2/3, and the padded sharded topologies; an unaligned input with no pad argument
still raises a `ValueError` mentioning `pad`; the aligned bench arms are unregressed.

---

### [ ] Refinement 6 — Speed up the low-work-per-core regimes

**Type**: perf

**Goal**: the small regimes are the ones Phase 0 left with a measurable fixed-cost floor:
`smallest [1,1,32,64]` (1 core, 1 block — `ablate_all` is **17 %** of a 1.87 µs launch) and the small
sharded shape `[1,1,512,64]` L1-sharded on 4 cores (~1 µs, where `op_design.md` §7.1 puts the
`NoReconfigure` lever's 8–19 %). Attack the per-core setup cost with the master.md levers whose
situation matches — **B13** `set_state`/`with_state` stateful transfers, **D21**
`InterleavedAddrGenFast`/host-precomputed indexing, and **C14**'s *second* degree on the zero-copy
sharded path (a resident path whose reader still exists only to run the CB handshake has taken only
the first degree; the fold is a separate measurable step — but note `zero_copy_fold` measured folding
at **0.74×** at 2 tiles/core, so this one is a measurement, not an assumption). No SUPPORTED change.

**Verifier notes**: needs Refinement 2 (the sharded shapes must exist and be zero-copy before their
per-core overhead is worth attacking). master.md **B0** is the governing caveat for this whole slot —
every lever here *adds* fixed per-core setup, so the counterfactual must be measured on the smallest
regime itself and gated on a work-per-core threshold if it only pays above one.

**Done when**: measured device-ns improves on `smallest` and on the small sharded shape, each landed
lever has an off-arm ratio recorded in the ledger, the golden suite is green, and there is no
regression on `square` / `wide_short` / `tall_narrow` or on any sharded shape a prior phase measured.

---

### [ ] Refinement 7 — dtypes, the value-preserving cast, and padded dtypes (prompt A4 + A5b + P3)

**Goal**: widen the numeric surface in one bundle:

- `SUPPORTED["dtype"]` += `float32`, `uint32`, `uint8`; `SUPPORTED["output_dtype"]` += `float32`,
  `bfloat8_b`, `uint32`, `uint8` (INVALID already prunes the int↔float crosses; `bfloat8_b` is an
  **output-only** dtype — a ROW_MAJOR input can never be block-float, and the tilize helper asserts it).
- The `dtype=` cast is a real value-preserving cast at pack time, driven by the existing CT
  `needs_cast` flag → `ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure`; `NoReconfigure`
  must stay selected when there is nothing to cast.
- **uint8 (A5b)** is not just another width: an 8-bit datum needs the **per-face** row dim, not the
  full-tile one, or the tile comes out **strided** (every other row zero) — shape-correct and
  value-wrong. Pin the choice with a `tt-probe.sh` dump of one output tile *before* running the golden
  suite; the uint8 golden cells compare **exactly**.
- **P3**: the padded path for every new dtype, i.e. the sub-word fill replication (2× bf16/uint16,
  4× uint8) and the signed→unsigned `bit_cast` for a negative integer fill.

**Type**: generality · **Blocking-model class**: knob-turn — `WT_BLOCK_MAX` is a *byte* target, so
every dtype lands on the same 1024 B transaction and the same constant 128 KiB per-core CB with no
per-dtype literal.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: ordered late deliberately — it is the cheapest tier (it widens SUPPORTED lists and
relaxes a `validate()` gate on paths Refinements 2/4/5 have already reshaped), so landing it earlier
would mean rebuilding it against a kernel that changes underneath it. Two op-specific facts the skill
cannot know: `can_use_fast_tilize` returns false when the **output** format is Float32
(`tilize_helpers.inl:90-96`), so fp32→fp32 silently takes the slower regular `tilize_block` path —
**re-run the ceiling per dtype**, page size changes the bound; and a 32-element uint8 chunk is 32 B,
at the DRAM read-alignment floor, which the `max(2, …)` in `wt_block_max` already guards
(`row_bytes ≥ 64`). `test_tilize_debug.py::test_b11_uint8_narrow_stick_is_the_known_alignment_gap`
already pins that boundary. This op has no `ComputeKernelConfig` surface (no math), so the skill's
fidelity/DEST half does not apply — the dtype + intermediate-CB-format half does.

**Done when**: all four input dtypes and all five output dtypes (minus INVALID crosses) pass the
golden suite, uint8 exactly and with no strided-tile signature; the padded cells pass per dtype with
an exact fill; the per-dtype ceiling is re-run and recorded.

---

### [ ] Refinement 8 — Tile geometry: tiny tiles and retile (prompt T1 + T2)

**Goal**:

- **T1, tiny tiles** — `SUPPORTED["tile_height"]` += `16, 8, 4, 2, 1` on a ROW_MAJOR input,
  interleaved *and* sharded. This is a **CB tile-descriptor change only**: both the reader
  (`unpack_tile_r_dim[cb_input_sticks]`) and the LLK read the tile height from the CB, and
  `can_use_fast_tilize` requires 32×32 tiles so every tiny tile takes the regular path automatically.
  T1 is **not** arch-gated — a failure here is a real defect.
- **T2, retile** — `SUPPORTED["in_layout"]` += `TILE_LAYOUT` and `SUPPORTED["in_tile_height"]` +=
  `32, 16, 8, 4, 2, 1` (keeping `"none"` accepted). A face-walking reader assembles RM sticks from
  TILE pages (two 16-wide face reads per (stick, tile-column)) and emits the **same**
  `cb_input_sticks` contract, so compute and writer are unchanged. **Blackhole-only**: arch-gate and
  **skip**, never fail, elsewhere.

**Type**: generality · **Blocking-model class**: knob-turn (T1 — the tile descriptor is the knob) /
new reader (T2 — a different source addressing scheme behind the same CB contract).

**Implementation skill**: /memory-layouts

**Verifier notes**: ordered last of the generality entries because it is independent — nothing else
needs it — and arch-gated on half its surface. Two op-specific traps: (1) **"tile-aligned" on the H
axis now means a multiple of the *requested* tile height**, and `tag_alignment` already measures it
that way, so any hardcoded 32 in the op will mis-gate its own cells — the op already routes every use
through `tile_h`/`DEFAULT_TILE_HEIGHT`, keep it that way; (2) `op_design.md` §7.2 records why the
retile reader cannot be an `untilize`→`tilize` helper chain (one CB has one page size, and T2 is
*defined* by `in_tile_h ≠ out_tile_h`) — do not spend a pass rediscovering that. Track T is
correctness-gated: do **not** spend a DM lever on the deliberately transaction-inefficient face reads,
and do not report their duration against a NoC ceiling. A retile call that also asks for a pad must
stay refused.

**Done when**: identity holds at every tile height including the degenerate 1×32, interleaved and on a
sharded output; retile passes shrink (32→8) and grow (1→32), interleaved and BLOCK-sharded, on
Blackhole and **skips** cleanly elsewhere; retile + pad is still refused with a message mentioning
`pad`.

---

### [ ] Refinement 9 — Per-dtype / per-geometry perf re-target and the square's residual

**Type**: perf

**Goal**: two measured regions that only become measurable once Refinements 7 and 8 have landed.
(a) **Per-dtype re-target**: `[1,1,2048,2048]` in fp32 and uint8 — page size changes the bound, and
fp32 output **disables fast tilize**, so the achieved ratio must be recomputed against a per-dtype
ceiling rather than inherited from bf16. (b) **The square's residual ~9 %**: Phase 0 measures
`achieved = 0.91` (44.2 µs vs a 40.7 µs practical target) at 84 % of theoretical DRAM peak; the
recorded open levers for that last slice are master.md **A3** (reader adjacent to its DRAM bank — the
block→core mapping is already a host-side function, so this is a *placement* change only) and **B10**
(per-reader VC assignment, to break FCFS serialization on shared routes). No SUPPORTED change.

**Verifier notes**: this is the last measured slot before the closing audit; if a lever here turns out
to be a no-op on this part, record it as *measured-no-payoff* in the ledger rather than dropping it —
Refinement 10 consumes exactly that record. Do not re-litigate the bandwidth-knee cap: master.md
records it measured **~2.4× slower on this very op**, and the design refutes it structurally.

**Done when**: fp32 and uint8 squares each have their own computed target, tt-npe pin and measured
device-ns recorded with `achieved`; any placement/VC lever that lands shows a measured device-ns win
on the square; no regression across the full cumulative bench set (all shapes × the dtypes now
supported).

---

### [ ] Refinement 10 — Perf completeness audit (run-closing, prompt A7)

**Type**: perf (retrospective — **no** SUPPORTED change, no new capability)

**Goal**: run the `/perf-ceiling-dm` **Mode D completeness audit** over the full lever list
(`ttnn/ttnn/operations/examples/master.md` — Part 1 examples + Part 2 propositions). Account for
**every lever this run did NOT apply**, each tagged *not-applicable* / *deferred* /
*measured-no-payoff* / *missed*, with an estimated counterfactual delta for the ones that are not
clearly not-applicable. `lever_ledger.json` already carries 24 catalog rows (11 closed with evidence,
13 open at Phase 0) and `op_design.md` §10.5 seeds the deferred list — extend, do not restart.

**Verifier notes**: this is the ONE queue entry that neither adds to SUPPORTED nor moves a failing
cell, filed under the run-closing-retrospective exception because this is a perf-focused op. Order it
strictly last so it audits the finished op. Two rows are already *structurally* closed and must not be
reopened as opportunities: **B12** multicast (§1.1 — the map is a bijection, no operand is shared
across any split, so there is nothing to fan out) and **C17** in-place (the tilize helper
`static_assert`s `input_dfb != output_dfb`).

**Done when**: `changelog.md` carries a completeness ledger (`lever → status → predicted delta if
applied → reason`) covering every master.md lever, plus a ranked list of the real remaining
opportunities; anything *missed* or *deferred* with a large predicted delta is surfaced as a concrete
follow-up for the next run rather than silently dropped; no regression on any prior bench.

---

## TARGET − SUPPORTED coverage ledger

Every `(axis, missing value)` at Phase 0 is accounted for below — this table is the queue's
completeness proof.

| Axis | Missing at Phase 0 | Where it lands |
|---|---|---|
| `dtype` | float32, uint32, uint8 | Refinement 7 |
| `output_dtype` | float32, bfloat8_b, uint32, uint8 | Refinement 7 |
| `use_multicore` | True | Refinement 1 |
| `double_buffer` | False | Refinement 1 |
| `rank` | 2, 3, 5 | Refinement 1 |
| `rank` | 0 | Refinement 5 (reachable only with padding) |
| `buffer` | dram_to_l1, l1_to_l1, l1_to_dram | Refinement 1 |
| `shard_api` | legacy_2d, nd | Refinement 2 (cross-spec cells: Refinement 4) |
| `out_scheme` | HEIGHT, WIDTH, BLOCK, nd | Refinement 2 |
| `orientation` | ROW_MAJOR, COL_MAJOR | Refinement 2 |
| `pad_mode` | auto, explicit | Refinement 5 |
| `pad_value` | zero, positive, negative | Refinement 5 |
| `alignment` | w_non_aligned, h_non_aligned, hw_non_aligned | Refinement 5 |
| `tile_height` | 16, 8, 4, 2, 1 | Refinement 8 |
| `in_layout` | TILE_LAYOUT | Refinement 8 (Blackhole-only; skip elsewhere) |
| `in_tile_height` | 32, 16, 8, 4, 2, 1 | Refinement 8 (paired with `in_layout=TILE`) |

Cells that stay refused on purpose: `EXCLUSIONS = [{use_multicore: False, shard_api: legacy_2d},
{use_multicore: False, shard_api: nd}]` — a shard's cores are fixed by its spec, so the single-core
value of the distribution parameter is refused rather than unsupported forever. INVALID (in
`feature_spec.py`) prunes the int↔float casts, negative fills in unsigned formats, the
`in_layout`/`in_tile_height` sentinel coupling and TILE-input padding; none of those is a queue entry.
