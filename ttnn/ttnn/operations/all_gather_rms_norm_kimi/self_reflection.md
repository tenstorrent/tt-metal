# Self-Reflection: all_gather_rms_norm_kimi

## Summary
- **Correctness was clean throughout.** The final blind pass was 7/7: 1 golden `supported_pass` plus 6 regression checks. Every phase (phase0, refinement 1–3, perf 1–2) also scored 7/7. There were 0 `supported_fail` and 0 `xpass_drift`. This is a single-cell suite (TARGET has one value per axis, `INPUTS = []`), so failure clustering has nothing to work on.
- **Perf** (ccl_perf, 100 experiments): fastest chip 18.72 → **18.09 µs** over the two perf rounds. The roofline is 7.6 µs and the target 10.1 µs, so the op is still ~1.8× off target. The bulk ring transport is the binding term.
- **Most important finding (framework-level):** the perf prompts require "PERMANENT" zones from `kernel_lib/perf_instrumentation.hpp`. That header does not exist. Also, the CCL perf harness runs with the device profiler on, so permanent zones overflow its buffer and drop programs. The op had to invent its own define-gated alias header.
- **Second:** the record of helper bypasses is incomplete.
  - The one graduated raw-LLK path (column-0 SFPU combine, Refinement 3) appears in no bypass table, and both perf entries say "none".
  - Perf 2's claim that "no kernel_lib helper covers" trid reads is half wrong: the tag and barrier calls exist, but under a "local copy" header.
- Problems are framework / doc-level, not op-level.

## 1. Golden coverage (`eval/golden_tests/all_gather_rms_norm_kimi/`)

There are no blind failures and the universe is one cell, so there is no in-TARGET axis gap. One output-property gap:

**F1.1 — Bit-identity of the replicated row across a group's members is not tested by golden.**
- **What:** every member of a group receives the same normalized row. During the run, one lever made the replicas differ in rounding: the golden tolerance check would pass them, and only the implementer's own acceptance test caught it.
  - `test_deterministic` compares repeated calls *on one device*.
  - `check_output` is PCC / relative-RMS per device.
  - So nothing in golden asserts that the members' outputs match each other bit for bit.
- **Evidence:**
  - `agent_logs/ttnn-implementer_breadcrumbs.jsonl` (2026-11-17T20:52:17): `"replication mismatch: own slice via cb_xg fp32 round trip vs remote via DEST reuse are not bit-identical"`.
  - `test_regression.py:106-115` compares only `first[rc]` against the same `rc`.
  - `op_design.md:347` claims "Determinism across calls and devices — the regression suite checks bit-identical repeats", which overstates what the suite checks.
- **Recommendation:** add a regression check (not a `LOOSE_CASES` entry; this is not an input facet): within each group, `torch.equal` every member's output against member 0's, at the suite's one shape. If replica bit-identity is *not* part of the op contract, state that explicitly in `helpers.py`'s oracle docstring instead.
- **Confidence:** med. Whether bit-identity is required is a contract choice, but the run's own design treated it as one (`op_design.md:376`).

## 2. SUPPORTED honesty (op file `SUPPORTED` / `EXCLUSIONS`)

No finding.
- `verifier_report.json` (blind): `supported_pass 1`, `supported_fail 0`, `xpass_drift 0`, `xfail_wrong_mode 0`.
- The 6 `no_axes_found` entries are the axis-free `test_regression.py` checks, which is expected.
- `SUPPORTED` equals TARGET and `EXCLUSIONS = []`, which is honest for a one-cell universe.

## 3. Helper / reference docs

#### Helper gaps (perf)

| helper | claimed | verdict | evidence | proposed fix |
|---|---|---|---|---|
| `eltwise_chain` (`kernel_lib/eltwise`): one operand shared across DEST lanes | Perf 1, rejected `late_chunk_one_pass` impl3: "No helper keeps one operand in a single DEST slot shared across a window" (changelog:257) | **missing** | `core/chain.inl:1389-1392`: "Each element writes to DEST[dst_slot + j * chain_lane_width]", so every lane gets its own copy of the shared `rstd_full`. Same arithmetic, bit-identical: helper chain impl1 **21.13 µs** vs raw impl3 **19.78 µs** (changelog:227). The helper alone cost ~1.35 µs. | Add a lane-invariant operand: load once into a fixed slot outside the lane stride, and let lane elements read it. |
| `dataflow_kernel_lib` trid reads (`local_copy_helpers_dataflow.hpp`) | Perf 2, graduated `port_supply_pipeline`: "No kernel_lib helper covers this; it is the base dataflow layer, not a bypass" (changelog:363) | **undocumented** (tag + barrier) + **missing** (non-blocking poll, read ring) | `set_read_trid` and `async_read_barrier_with_trid` exist (`local_copy_helpers_dataflow.hpp:147,158`), with a rolling-pipeline example at `.inl:98-108`. But the file is titled "Local L1 -> L1 copy helpers" (`.hpp:7-10`), so an author doing remote reads never looks there. Genuinely absent: a non-blocking completion poll (raw `ncrisc_noc_read_with_transaction_id_flushed`, `port_reader.cpp:66`) and the "NIU has counted the read" guard (`noc_cmd_buf_ready` spin, `port_reader.cpp:110-113`). | (a) Move or re-export the trid pair to a general dataflow-read header, and say in the header that it is not local-copy specific. (b) Add `read_trid_done(noc, trid)` and a rolling-ring helper (sketch below). |
| `eltwise_chain` SFPU elements: SFPU op scoped to column 0 / faces (`VectorMode::C`) and a G-way DEST sum | **Unrecorded.** It graduated in Refinement 3, before perf, and has no row anywhere. Both perf entries say "Helper bypasses — none" (changelog:255, 361). | **missing** | `kernels/..._compute.cpp:178-203`: a raw-sfpi `combine_col0_body` called through `_llk_math_eltwise_unary_sfpu_params_(…, VectorMode::C, …)`. The kernel comment says "no helper scopes an SFPU op to a column, and none fuses a G-way DEST sum with rsqrt". Confirmed: no `kernel_lib/eltwise` element takes a vector mode (the only `VectorMode::C` mapping is `sfpu_activation_helpers.hpp:139`, which is not in the chain). Measured: `nobulk` 17.02 → 15.80 µs (changelog:140). | Add a face/column-scope template parameter to the SFPU `UnaryOp` elements, plus a DEST N-way fixed-order sum element. Record this row in the changelog. |

None of the rows is an `ergonomics` bypass with equal ns: every one is a measured capability gap.

What went right: the custom element kept the chain's lifecycle by deriving from the `UnaryOp` CRTP (`compute.cpp:195-203`). But that extension point is documented only in `core/chain.inl:562-577`, while `api/chain.hpp:79-83` calls the tag hierarchy "internal pipeline scaffolding". Propose one paragraph in `chain.hpp` that names `UnaryOp<Derived, Slot>` + `lane_width` override + `exec_impl` as the sanctioned way to add a custom element.

API sketches (confidence: low; each is derived from a single call site):
```cpp
// 1. lane-invariant operand: loaded once into a fixed slot, read by every lane
eltwise_chain(IterationShape::tiles(n).block_size(4),
    SharedLoad<input(cb_rstd_full), Dst::D7>{},            // once per window, outside the lane stride
    CopyTile<input(cb_xg)>{},
    MulBinary<Dst::D0, Shared<Dst::D7>, Dst::D0>{},
    PackTile<output(cb_out)>{});

// 2. rolling trid read ring: owns trid rotation, in-order push, the NIU-counted guard,
//    and "push the oldest when the CB is not reservable" (the batch-pop deadlock, breadcrumb 00:58:18)
dataflow_kernel_lib::ReadRing<cb, /*in_flight=*/4> ring(noc, chunk_tiles);
ring.read(src_noc_addr, chunk_bytes);   // non-blocking; pushes any completed reads first
ring.drain();

// 3. face-scoped SFPU element and an N-way DEST sum
CombineSum<Dst::D0, /*N=*/G>{}, MulUnary<>{inv_w}, AddUnary<>{eps}, Rsqrt<Approx::Exact, VecScope::Col0>{}
```

**F3.1 — The NoC alignment reference gives the wrong rule for semaphore multicast.**
- **What:** the reference says semaphore and inline APIs need only 4-byte word alignment. The run hung because `noc_semaphore_set_multicast` source words sat at a 4 B stride while the destination semaphore is 16 B aligned.
- **Evidence:**
  - `.claude/references/noc-alignment-theory.md:137`: "They have their own alignment rules (usually 4-byte word-aligned)".
  - Breadcrumb 2026-11-17T20:36:49: `"semaphore_set_multicast source mask word at 4 B stride not 16 B aligned like the destination semaphore -> NOC alignment error -> hang"`.
  - changelog:96.
- **Recommendation:** add one line: "`noc_semaphore_set_multicast` / `_loopback_src` is a NoC write from a local L1 word. Source and destination must share the same offset mod the L1 alignment (16 B), so per-slot source words need a 16 B stride."
- **Confidence:** med-high.

**F3.2 — A referenced helper header does not exist.**
- **What:** the perf docs point at `ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp` and its "durability contract". The file is absent, so the op wrote a local alias header that uses `__has_include`.
- **Evidence:**
  - `.claude/agents/perf-coordinator.md:50`, `.claude/eval/prompts/perf_tournament_prompt.txt:9`, `.claude/references/device-zone-scope-attribution.md:3`.
  - changelog:173: "`kernel_lib/perf_instrumentation.hpp` does not exist in this tree".
  - `kernels/all_gather_rms_norm_kimi_perf.hpp:13-24`.
- **Recommendation:** land the header, with the gating from F4.1, or remove the references.
- **Confidence:** high.

## 4. Agent prompts

**F4.1 — "PERMANENT zones" conflicts with the CCL perf harness.** (`perf-coordinator.md`, `perf_tournament_prompt.txt`)
- **What:** the prompt says to keep `MaybeDeviceZoneScope` zones permanently and calls the macro "free when the profiler is off". But `eval/ccl_perf.py` always runs with the device profiler on, and 500 traced calls with zones compiled in overflow the profiler buffer, so the harness miscounts programs. The coordinator had to invent an op-specific define (`AGRN_PERF_ZONES`), keep it out of harness builds, and add it to the program hash.
- **Evidence:**
  - `perf-coordinator.md:49-53`.
  - `eval/ccl_perf.py:56-57` (`"TT_METAL_DEVICE_PROFILER": "1"`).
  - changelog:182: `programs per chip {3: 579, ...}, expected 600`.
  - `all_gather_rms_norm_kimi_program_descriptor.py:45-47,887`.
- **Recommendation:** propose that the prompt (and the missing header) say: "for ops measured by `ccl_perf`, the zones compile in only under a host-set define that is part of the program hash and is off in harness builds; turn it on for breakdown runs". Better still, make this the header's default so every op does not reinvent it.
- **Confidence:** high.

**F4.2 — The bypass record only covers perf graduations, so a refinement's raw-LLK path disappears.** (`ttnn-implementer.md`, `perf-coordinator.md`)
- **What:** the implementer prompt says to "Record the gap exactly as a raw-LLK bypass is recorded", but the only schema is the perf coordinator's `### Helper bypasses` table, and that applies to perf graduations only. As a result:
  - the Refinement 3 raw-sfpi combine has no row;
  - the perf entries' "— none" reads as "the op has no raw LLK", although the kernel carries one.
- **Evidence:**
  - `ttnn-implementer.md:563`.
  - `perf-coordinator.md:316-352`.
  - Commits `8ed6eab24f9` and `db600ae0211` (R3 `COMBINE_COL0`).
  - changelog:136-165 (no bypass section) vs changelog:255 and 361 ("none").
- **Recommendation:** propose that the refinement changelog entry carry the same `### Helper bypasses` table, and that the perf coordinator scope its claim ("none *this round*") or list the inherited bypasses it left in place.
- **Confidence:** med.

**F4.3 — Perf 2 declared "not a bypass" without checking `kernel_lib`.** (`perf-coordinator.md`)
- **What:** Perf 2 ruled the raw trid calls "the base dataflow layer, not a bypass" without looking in `kernel_lib`, where matching helpers exist (§3 row 2).
- **Evidence:** changelog:363 vs `local_copy_helpers_dataflow.hpp:147,158`.
- **Recommendation:** add to the bypass-table instructions: "before writing 'no helper covers this', grep `kernel_lib/` for the raw API names you used".
- **Confidence:** med.
