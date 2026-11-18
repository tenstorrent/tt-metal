# Self-Reflection: wq_b_linear

## Summary
- Final blind: **3/3 green** (2 registry cells + 1 regression bit-identity; there are no translated tests). `verifier_report.json`: supported_pass 2, supported_fail 0, xpass_drift 0. TARGET is one cell, SUPPORTED == TARGET since Phase 0, and every refinement is ticked `[x]`.
- Perf trajectory on the one cell (device kernel ns, golden runs): 26.8 µs (Phase 0) → 23.6 (R1) → 22.4 (R2) → 22.5 (R3, R4 parked) → 22.3 (Perf 1) → **20.6 µs (Perf 2 / blind)**, 1.29× the 16.0 µs DRAM SOL. PCC 0.99997 throughout.
- **Most important finding (framework-level):** three phases (R2, R3, Perf 1) attributed the W-stream cap to "NoC0 link load next to DRAM endpoints". Perf 2 found the real cause: one NoC2AXI port per bank caps at ≈50 GB/s, and `TensorAccessor` can only address the preferred port. No reference doc states the per-port ceiling, and no API reaches the bank's third port. The fix shipped as hard-coded Blackhole coordinates in an op kernel.
- The run produced no correctness problems. Every finding is about framework docs, the helper API, or the process.

## 1. Golden coverage (`eval/golden_tests/wq_b_linear/feature_spec.py`)

No evidence-backed gap. `cartesian(TARGET × INPUTS)` is one cell, `INPUT_TAGGERS = {}`, and the blind run had no failures to cluster.

**Note (outside golden's reach):** the input is fixed, but which W-read path runs depends on the board. The graduated port spread (`kernels/wq_b_linear_w_read.hpp:30-53`) switches to preferred-endpoint-only reads on non-Blackhole builds, when programmable DRAM cores are enabled, or when the coordinates don't match. Changelog:143 says *"The prefetcher fallback was not run on device."* No golden input can select that path, so nothing changes in golden. **Recommendation:** add an op-level unit test that forces the fallback (for example a CT define that sets `dram_port_spread_state = 2`) and checks correctness. **Confidence:** med.

## 2. SUPPORTED honesty (`wq_b_linear.py` `SUPPORTED` / `EXCLUSIONS`)

No finding. 0 `supported_fail` and 0 `xpass_drift` in the blind run and in every phase (`golden_phase0` … `golden_perf_2`). `validate()` rejects any other shape or placement with `ValueError` (`wq_b_linear.py:63-70`), which matches `INVALID = []` and a single-cell TARGET.

## 3. Helper / reference docs

### 3.1 Per-port DRAM ceiling missing from `speed-of-light.md` — **absence**
- **What:** the SOL reference treats DRAM peak as reachable (`speed-of-light.md:20`, *"DRAM: (bytes read + bytes written) / DRAM peak"*). It never mentions that each bank is reached through one NoC2AXI port per request address, and one port caps at ≈50 GB/s on Blackhole (7 × 50 ≈ 350 GB/s, 77% of 448 GB/s). Without this, the agents spent two refinements and one perf round tuning splits, depths, placement and NoC1 shares against a ceiling that none of those levers could move.
- **Evidence:** changelog:47 (R2) *"The cap is NoC0 link load next to the DRAM endpoints, not the DRAM banks"*. Repeated at changelog:66 (R3) and in the Perf 2 hypothesis (perf-coordinator breadcrumb 7, *"NoC0 egress link … carries west bank + east bank"*). Corrected at changelog:139: *"the cap is per DRAM port (≈50 GB/s per bank on one NoC2AXI port)"*. W stream alone dropped from 19.1–19.6 to 16.2–16.3 µs once two ports were used.
- **Recommendation:** add to `speed-of-light.md` § DRAM: "Blackhole: reads through one NoC2AXI port per bank (TensorAccessor's preferred endpoint) top out near 50 GB/s/bank. A DRAM-bound stream stuck at ~1.3× SOL across every split/depth/placement is at that ceiling. Spreading reads over the bank's ports is the lever (see `wq_b_linear_w_read.hpp`)." Also add a `master.md` catalog entry with these measured numbers.
- **Confidence:** med. The mechanism comes from one board and one op. The fact that `rot2` moved the number that every other lever left flat is strong evidence.

#### Helper gaps (perf)

| helper | claimed | verdict | evidence | proposed fix |
|---|---|---|---|---|
| `TensorAccessor` page read (`get_noc_addr` / `noc.async_read(accessor, …)`) | capability: cannot target a bank's non-preferred NoC2AXI port | **missing** (partly): `get_noc_addr(page_id, offset, uint8_t noc)` (`tt_metal/hw/inc/api/tensor/tensor_accessor.h:165`) can address the bank's *other NoC's* endpoint (`dram_bank_to_noc_xy[noc]`), which is 2 of the 3 ports. The third, free port is not addressable. That free port is the one the graduated `rot2` needs: the NoC1 endpoint port regressed the whole op (`rot` 23,330 ns, x prefetch queued behind W). | Bypass hard-codes `kDramTranslatedX0 = 17`, `kDramTranslatedY0 = 12` and a 3-ports-per-block layout (`wq_b_linear_w_read.hpp:19-21,47`), and probes `tensor_prefetcher_signal_noc_xy` for correctness (`:33`). Measured 19,123–19,550 → 16,219–16,336 ns (W alone). | Expose the port table from the HAL/firmware (like `dram_bank_to_noc_xy`) plus an accessor overload that takes a port index and returns the preferred endpoint whenever the port is unusable (stream mode, other arch). That removes the arch constants and the stream-mode correctness hazard from op kernels. |
| `compute_kernel_lib::matmul_block` | capability: (1) no per-column push on the last K-block under `caller_owns_pack_target`; (2) L1-acc keyed on block index within a call; (3) no column-outer subblock order | **missing**, all three confirmed in the `.inl`. (2) `llk_pack_reconfig_l1_acc(block == 0 ? 0 : 1)` (`matmul_block_helpers.inl:414,467`), with no way to make a second call keep accumulating. (1)/(3) Loops are `in0_subblock` outer, `in1_subblock` inner (`:294,306`), and pushes happen only per M-row-group (`:490-492`). The chosen subblock is (5,1), so `in0_num_subblocks = 1` and the whole block is one row group. Under `caller_owns_pack_target` nothing is pushed at all. (2) is the root gap: with it, a caller could split off the last K-block into a second call with a different subblock shape and get row release from the existing per-row-group push. | Raw last block at `kernels/wq_b_linear_compute.cpp:30-76`. Measured ≈21,037 → 20,476 ns whole op. The optimizer's raw *row* release variant (1×N subblocks on the last block) tied column release (22,019 vs 22,040, breadcrumb part-optimizer 17). | Add a `first_block_accumulates` (continue-L1-acc) option, then a per-column-group release on the last K-block. See the sketch below. |

No `ergonomics` rows were recorded, so no equal-ns cases exist. **Unrecorded bypasses:** none that are perf-driven raw LLK. One gap in the record: Perf 1 wrote *"Helper bypasses — none … plain local semaphores"* (changelog:110) and graduated a `y_done` handshake built from legacy `noc_semaphore_set/wait(reinterpret_cast<…>(get_semaphore(id)))` (`wq_b_linear_reader.cpp:147`, `wq_b_linear_writer.cpp:343-345`). The `Semaphore` class (`tt_metal/hw/inc/api/semaphore.h:46`, DM ctor from a bare id at `:58`) covers this. It is not raw LLK, but it is an API bypass the "none" claim skips. Confidence: low.

```cpp
// confidence: low (one call site)
// (2) continue packer L1 accumulation from block 0 of this call
matmul_block<false, true, LastBlockTarget::Interm, OutputCBLayout::TileRowMajor, /*...*/,
             /*caller_owns_pack_target=*/true, /*...*/, /*accumulate_into_existing=*/true>(...);

// (1)+(3) last K-block released column-group by column-group to a DM consumer
matmul_block<..., LastBlockRelease::PerColumnGroup>(x, w, y, y,
    MatmulBlockShape::of(1, core_n / sb_w, block_m, sb_w, k_block, num_k_blocks));
// helper walks in1 subblocks outermost on the last block and pushes each packed column group;
// the call site no longer owns matmul_block_init, llk_pack_reconfig_l1_acc, or the
// "subtract tiles already pushed" write-pointer arithmetic (wq_b_linear_compute.cpp:63).
```

### 3.2 `matmul_block` leaves packer L1 accumulation on — undocumented post-condition
- **What:** the helper returns with L1 accumulation enabled when `num_k_blocks > 1`, and `PostKBlockFn` also runs with it enabled. The `.hpp` states neither. An `eltwise_chain` copy run after or inside the helper therefore accumulated onto stale L1. The `PostKBlockFn` doc also does not refer to the matmul-state-restore contract, which covers only `PreKBlockFn` (`matmul_block_helpers.hpp:182-196`). So the caller hand-wrote `restore_main` (`wq_b_linear_compute.cpp:145-149`).
- **Evidence:** changelog:69 *"`matmul_block` returns with packer L1 accumulation enabled … Inf / p0+p1 on the leftover columns"*. Implementer breadcrumb 6. No `l1_acc` exit-state text at `matmul_block_helpers.hpp:198-204` or `:232-338`.
- **Recommendation:** add to the `NoPostKBlock` / `matmul_block` docs: "With `packer_l1_acc`, packer L1 accumulation is left ON on return and inside `PostKBlockFn` (block > 0). Call `pack_reconfig_l1_acc(0)` before any non-accumulating pack. A `PostKBlockFn` that runs another op must restore matmul state as in the PreKBlockFn contract (B)." Alternatively, have the helper reset the packer on exit.
- **Confidence:** high.

### 3.3 `McastConfig::data_ready` hides a ~2× bandwidth consequence
- **What:** `DataReadySignal::Counter` silently selects unlinked multicast plus a write barrier per send. `Flag` selects linked. The planner picked Counter as the "fast" option, and it was 1.4× slower for the broadcast. The semantics appear only in the `.inl` comment.
- **Evidence:** `mcast/host/mcast.hpp:105` (no doc on the field). `mcast/kernel/pipes/mcast_pipe.inl:243` `constexpr bool linked = DATA_READY_SIGNAL == DataReadySignal::Flag;` and `:251` barrier. Implementer breadcrumb 3: *"Counter … caps at ~18-25 B/cycle vs ~41 B/cycle linked … 26.3 vs 19.1 us"*.
- **Recommendation:** document on `McastConfig::data_ready` and in `kernel_lib/mcast/README.md`: "Flag ⇒ linked multicast (≈41 B/cycle to ~110 dests on BH). Counter ⇒ unlinked + write barrier per send (≈18-25 B/cycle). Use Flag for bandwidth-bound broadcasts."
- **Confidence:** high.

### 3.4 Data-movement ablation has no built-in switch, so text-needle benches break
- **What:** `perf-measure` asks for the helper family's built-in skip switch and has one only for compute (`SKIP_COMPUTE`). For reads and writes the op bench used string replacement on kernel source. Each perf graduation broke it.
- **Evidence:** `_op_bench.py:42-47` (`assert needle in src`). changelog:116 *"`no_y_write` needle was stale after Perf 1 (assert failed)"*. Part-optimizer breadcrumb 18 *"_op_bench no_y_write needle breaks after patch"*. `.claude/skills/perf-measure/SKILL.md:120-125`.
- **Recommendation:** consider `SKIP_NOC_READ` / `SKIP_NOC_WRITE` defines in the `Noc` API, parallel to `SKIP_COMPUTE` (payload elided, transaction counts and barriers kept). Then the skill line becomes "use `SKIP_NOC_*` for DM stages".
- **Confidence:** med.

## 4. Agent prompts

### 4.1 Implementer "park, never delete" rule keeps losing scheme-changes in the shipped kernels
- **What:** `ttnn-implementer.md:92` (*"Park the lever at a trivial, byte-identical default as a live knob … never a deleted knob"*) is written for block knobs. It was applied to whole schemes that lost their A/B: R4 K-remainder combine (`K_REMAINDER_SPLIT`), `X_ROUNDS_CONCURRENT`, and `W_NOC1_K_BLOCKS`. The parked R4 path then forced a carve-out in a perf graduation.
- **Evidence:** `wq_b_linear_compute.cpp:144-218` (R4 path, default off). changelog:144 *"Carve-out: the parked R4 path (`K_REMAINDER_SPLIT`) keeps the single helper call"*. Perf 2 also had to re-measure R4 for every domain sweep.
- **Recommendation:** narrow `ttnn-implementer.md:92` to block-size/depth/split knobs. A scheme-change that loses its A/B ships as an artifact under `perf_experiments/` (with its patch) rather than as a parked path in the op's kernels.
- **Confidence:** med.

### 4.2 Verifier gated a scheme-change on a metric the scheme does not address
- **What:** R3 (R4 combine) was built to remove per-core W imbalance. Its skip gate was a conjunction: "max/p50 ≤ 1.05 **and** full op within ≈5% of `w_only`". R2 recorded 1.045, so there was no imbalance, but the second clause (overlap loss, 13%) failed. A full scheme-change was built and parked for +0.3 µs. The verifier's original imbalance number was also stale (1.25 "taken at kb=6").
- **Evidence:** op_requirements.md R3 Verifier notes. changelog:46 *"the verifier's 1.25 was taken at kb=6"*. changelog:66 *"per-core imbalance does not [bind] (… max/p50 is already 1.045)"*. Commits `f71e390c823`, `147656d7217`.
- **Recommendation:** add to `incremental-verifier.md` § Perf refinements: "Gate a scheme-change only on the metric its mechanism changes. Re-measure the gating metric at the current knob values before filing."
- **Confidence:** med.

### 4.3 Verifier prompt has no single-cell / all-perf clause
- **What:** the verifier prompt assumes a generality gap (2:1 cadence, R1/R2 anchor, a 3–4-shape precision baseline). For a one-cell op none of these apply, and the verifier had to improvise the queue order and the precision baseline (distribution and seed instead of shape).
- **Evidence:** incremental-verifier breadcrumb 1 (`friction`, surface `prompt`). `incremental-verifier.md:20,341,369`.
- **Recommendation:** add the clause the breadcrumb proposes. When TARGET − SUPPORTED = ∅, the queue is perf-only, ordered by the currently binding ablation term. The precision baseline varies distribution and seed. A perf entry whose headroom depends on a predecessor may close on a recorded no-headroom measurement.
- **Confidence:** high.
