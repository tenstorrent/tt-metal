# Self-Reflection: wq_b_linear

## Summary
- Final blind pass: **3/3 passed** (2 golden registry cells + 1 regression; no translated cells in this suite). Every phase (phase0, R1, R2, perf_1, perf_2) passed 3/3. `verifier_report.json`: supported_fail 0, xpass_drift 0.
- The universe is one cell (`TARGET` = one value per axis, one `INPUTS` shape, `INPUT_TAGGERS = {}`), so coverage and SUPPORTED honesty have nothing to find. This was a perf run: 26.6 → 16.75 µs on p150 (1.20× the 14.0 µs SOL).
- **Most important finding (framework, helper docs):** `matmul_block`'s documented K-blocked L1-acc example (`LastBlockTarget::Out`) serializes unpack behind pack on every K-step. That cost 3.1 µs of 15.3 µs of compute. The fast configuration (`caller_owns_pack_target` + `Interm`) already exists but is never described as the fast path.
- Second: both perf rounds wrote `### Helper bypasses — none` even though a graduated kernel calls raw multicast instead of `SenderPipe`. The prompt only asks for rows on "raw-LLK" paths, so this dataflow-helper bypass ended up in prose instead of the table.
- Every problem found is framework-level (helper docs, perf prompt scope, perf-measure skill, golden SOL metadata). None is an op bug.

## 1. Golden coverage (`eval/golden_tests/wq_b_linear/feature_spec.py`)

No failure clusters and no in-TARGET gaps: `cartesian(TARGET × INPUTS)` is one cell, and it plus its `LOOSE_CASES` copy passed in every phase.

**F1.1 — The SOL in `LOOSE_CASES` is for a different board than the one the run used.**
- **What:** `sol_ns_blackhole` = 16,018 ns is priced for a 7-bank p100a. The run board was a p150 with 8 banks (SOL 14.0 µs). Every agent had to notice this and recompute it. If one had not, the gap to SOL would have been understated by ~14%.
- **Evidence:** `feature_spec.py:41` "7 DRAM channels"; `feature_spec.py:71` `"sol_ns_blackhole": 16018`. Verifier friction breadcrumb (`incremental-verifier_breadcrumbs.jsonl`): "sol_ns_blackhole … assume a 7-bank p100a, but the run board is a p150 with 8 DRAM banks". `op_requirements.md` and every changelog entry restate "14.0 µs on this board".
- **Recommendation:** Store the SOL inputs (bytes moved, per-bank bandwidth, bank count) next to the ns figure, or key `sol_ns` by board (`{"p100a": 16018, "p150": 14016}`). Then agents recompute from `device.dram_grid_size()` instead of trusting one number.
- **Confidence:** high.

## 2. SUPPORTED honesty (`wq_b_linear.py` SUPPORTED / EXCLUSIONS)

No findings. `SUPPORTED` equals `TARGET` (one cell). Blind: supported_pass 2, supported_fail 0, xpass_drift 0, xfail_wrong_mode 0 (`golden_blind_final/verifier_report.json`).

## 3. Helper / reference docs

#### Helper gaps (perf)

| helper | claimed | verdict | evidence | proposed fix |
|---|---|---|---|---|
| `compute_kernel_lib::matmul_block` (L1-acc K-blocking) | feedback only ("none" bypass; header doesn't say the default L1-acc path serializes unpack behind pack) | **undocumented** | `.inl:518-525`: with `Out`, each non-last K-block does `interm_buf.wait_front/pop_front` (unpack waits for pack), then reloads into DEST on the last block. `.inl:505-516`: with `caller_owns` + `Interm` there is no drain. `.hpp:333-334`: the canonical K-blocked example uses `LastBlockTarget::Out`, i.e. the slow path. Measured: compute 15.34 → 12.17 µs (changelog.md:135, :150) | Add to the `.hpp` doc (`:27-29`/`:278`): "With packer_l1_acc, `Out`/`OutWithRelu` drain + reload interm on the unpack thread every K-block (a pack→unpack sync per K-step). For a pure-accumulate output use `Interm` + `caller_owns_pack_target` with pack target = out_buf." Switch the `:333` example to that configuration. |
| `compute_kernel_lib::matmul_block` (exit state) | not recorded | **undocumented** (unrecorded raw-LLK call) | `wq_b_linear_compute.cpp:67` `PACK((llk_pack_reconfig_l1_acc(0)));` after the helper. `.inl:412-414` (`Interm`) sets `l1_acc(block==0?0:1)` and never resets it, so the helper exits with L1_ACC on. Nothing in the `.hpp` says so | Either reset L1_ACC at helper exit under `Interm` + `packer_l1_acc`, or add one `.hpp` line: "Interm + packer_l1_acc leaves packer L1_ACC enabled; the caller resets it before any later pack." |
| `compute_kernel_lib::matmul_block` (per-tile tail) | capability (non-graduated `tail_store2`, changelog.md:210) | **missing** | `perf_experiments/tail_store2/kernels/*compute*.cpp:27-31`: with `caller_owns` the helper never pushes. Without it, TileRowMajor pushes only through the interm-drain path. A second helper call "cannot L1-accumulate (block==0 packs with L1-acc off)". Gain 16,753 → 16,433 ns (−1.9%, noise), so not graduated | Low priority (noise-level gain here). Record it as a candidate feature: per-row-group/per-tile push in caller-owns mode, or a "continue accumulating" flag so `block==0` packs with L1-acc on. |
| `dataflow_kernel_lib::SenderPipe` (Counter mode) | capability, prose only ("none" table, changelog.md:207-209) | **missing** | `mcast_pipe.inl:243` `constexpr bool linked = DATA_READY_SIGNAL == DataReadySignal::Flag;` so Counter payload is unlinked. `:250-251` write barrier after each send; `:276` `inc_multicast` + `:327` atomic barrier on every round. No API exists for a resident, handshake-free stream with an ordered, linked count. Raw path: `wq_b_linear_sender.cpp:99-155`. Measured: multicast alone 14,910 → 10,310 ns, whole op 18,729 → 16,708 ns (changelog.md:189); R1's ~1.1 µs/round per-round barrier+atomic (changelog.md:28) | Add a streaming mode to `SenderPipe` (sketch below). Also record this bypass as a table row (see §4, F4.1). |

```cpp
// confidence: low — derived from one call site (wq_b_linear_sender.cpp:99-155)
// Resident, handshake-free Counter stream: every round's payload and its count write go out as ONE linked
// sequence (the count is written from a per-round L1 word, so it lands after its payload on the same path/VC).
// The path is reserved once, and only finish() unlinks and barriers.
template <uint32_t MAX_ROUNDS>
class SenderStream {  // built from the same SenderRuntimeArguments as SenderPipe
    void begin(uint32_t count_src_l1 /* MAX_ROUNDS x 16 B, sender-only */);
    void send_round(uint32_t src_l1, uint32_t dst_l1, uint32_t size);  // payload + linked count = round+1
    void finish();                                                     // last count unlinked + write barrier
};
// call site:
stream.begin(get_write_ptr(cb_count_src));
for (uint32_t r = 0; r < num_k_blocks; ++r) { wait_for_reader(r); stream.send_round(src(r), dst(r), round_bytes); }
stream.finish();
// receivers unchanged: ReceiverPipe::receive(r)  (counter >= r + 1)
```

**F3.1 — The perf-measure skill reads the device duration by column index, and two agents got a corrupted span from it.**
- **What:** The skill says to read column 19 of the ops CSV. On this p150 that gave ~22 s / ~9.9e12 ns spans, because one core had unsynced profiler timestamps (and the column index differed). The implementer and the verifier each rebuilt a span reader on their own (`verify/span.py`).
- **Evidence:** `.claude/skills/perf-measure/SKILL.md:197` "**19 = DEVICE KERNEL DURATION [ns]**". Two friction breadcrumbs (`ttnn-implementer_breadcrumbs.jsonl` #3, `incremental-verifier_breadcrumbs.jsonl`), both cost "med".
- **Recommendation:** Read the column by header name. Add a sanity check (duration > 1 s means suspect). Add a fallback recipe: the span from `profile_log_device.csv` `*-KERNEL` zones, dropping cores with out-of-domain timestamps. Consider shipping `verify/span.py` as a shared tool.
- **Confidence:** high.

**F3.2 — The perf prompts reference a framework header that did not exist in the tree.**
- **What:** The perf prompts tell agents to include `ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp`. It was missing, so the Perf 1 coordinator wrote a minimal version and committed it inside the shared kernel_lib.
- **Evidence:** `perf_tournament_prompt.txt:9` and `perf-coordinator.md:50` reference it. changelog.md:107 says "did not exist in this tree; added a minimal one". The file's only commit is `00a53c45d54` (Perf 1).
- **Recommendation:** Ship the header in the framework (or have the perf phase assert it exists), so ops don't each write their own copy into shared code.
- **Confidence:** high.

**F3.3 — The DM estimator cannot price linked multicast, and linked multicast was the winning lever.**
- **What:** `estimate.py` has no linked-multicast option. It priced the current schedule at 25.5 µs against 14.9 µs measured. It could rank the direction (fewer reservations) but not the winner, so both rounds had to measure every candidate on device.
- **Evidence:** changelog.md:24 "has no linked option"; :181 "estimates the current schedule at 25.5 µs … against 14.9 µs measured"; :202 "41 evaluated, 0 built. The tool cannot express a linked stream".
- **Recommendation:** Add a `linked` / one-reservation-per-stream mode to `.claude/tools/perf_calculator` multicast estimates. Until then, add a caveat in its README that Counter-mode per-round estimates assume an unlinked burst per round.
- **Confidence:** med.

## 4. Agent prompts

**F4.1 — The bypass table only covers raw-LLK paths, so a graduated dataflow-helper bypass was recorded as "none".**
- **What:** The prompt asks for "one row per graduated raw-LLK path". The graduated sender replaces `SenderPipe::send()` with raw `noc_async_write_multicast`, which is not LLK, so both rounds wrote `— none` and put the real gap in prose without the required helper-ns / raw-ns / site columns. That loses exactly the cross-run feedback the table exists for.
- **Evidence:** `perf_tournament_prompt.txt:31` "one row per graduated raw-LLK path"; `perf-coordinator.md:271` "Every graduated raw-LLK path". changelog.md:207-209: "### Helper bypasses — none … The sender's direct `noc_async_write_multicast` calls, used instead of `SenderPipe::send()`".
- **Recommendation:** In both prompts, change the scope to "every graduated path that replaces a kernel_lib helper (compute or dataflow) with raw LLK or raw NoC/semaphore calls". Also require a row for any raw call that remains after a helper call (e.g. `compute.cpp:67`).
- **Confidence:** high.

**F4.2 — No other prompt findings.** The process went cleanly. Planner/verifier constraints were stated and respected (`block_k_tiles ≤ 16`, CB address parity, tie bands), NULL ideas were recorded with numbers, and noise-level raw-LLK wins were correctly not graduated (changelog.md:140, :191).
