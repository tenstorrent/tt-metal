# Self-Reflection: q_stem

## Summary
- **Final blind pass:** golden **1092/1092 pass** (every TARGET value is in SUPPORTED: `supported_fail` 0, `xpass_drift` 0). Regression **10/11**. The one failure, `test_regression.py::test_small_packet_size`, is a **harness error and not an op failure**. It failed the same way in all 11 graded phases (phase0 → perf_2).
- All 7 refinements are `[x]`. The queue drained. Two perf rounds ran: the fastest chip went 148.10 → 144.44 → 140.46 µs on the PERF FOCUS.
- **Most important finding (framework):** the Mcast helper has no slot-credit / credit-ahead flow-control mode. So the op wrote its own cumulative-credit protocol in the graduated path, which is recorded in no bypass table. That protocol then produced the same class of race twice: a shared threshold counter where a fast producer stands in for a slow one. No reference names that class of race.
- The problems are **framework-level** (harness, helper, references, perf-coordinator record format). Nothing points to an op-level correctness gap in the measured universe.

## 1. Golden coverage (`eval/golden_tests/q_stem/feature_spec.py`)

There are no blind golden failures, so there is no failure cluster to project through `INPUT_TAGGERS`. TARGET == SUPPORTED, so no in-TARGET region is untested by the axes. Two findings remain.

**1.1 The small-payload regression is never graded (harness bug)**
- **What:** `test_small_packet_size` is the only check that the op sizes its fabric packets from the active config, not a hard-coded payload. It errors in every graded run because it takes `_fabric_mesh_cache` instead of a `device`/`mesh_device` fixture. The L1-capture plugin raises an error when it cannot find a device. So this MUST-level contract is unmeasured in every phase's results.
- **Evidence:** `eval/l1_profiling.py:31-35`, which raises `"…but the test has no device or mesh_device fixture"`. The test is at `eval/golden_tests/q_stem/test_regression.py:231`, `def test_small_packet_size(_fabric_mesh_cache)`. The verifier flagged it at `verification_report.md:147-151` ("Please fix it in the golden suite"). It failed identically in `golden_phase0` … `golden_blind_final`.
- **Recommendation:** propose that `_device_from_item` also look for a mesh returned by a known fabric-mesh fixture. Alternatively, skip L1 capture (record "n/a") instead of raising when no device fixture is found. A narrower option is to have the test request `mesh_device` and fetch the custom-payload mesh inside the test. Fix the harness, not the op. Any other fused-CCL suite that uses `_fabric_mesh_cache` probably has the same latent failure.
- **Confidence:** high.

**1.2 Call-sequence dependence is axis-blind (hazard, not reproduced in blind)**
- **What:** the landing scratch is grow-only per mesh, and `cb_l1_budget` is derived from the lowest allocated L1 address. A large-S call followed by a deep-Q call on the same mesh can therefore shrink the second call's CB budget. Refinement 6 saw exactly that failure. No tagger or golden cell expresses "what ran before on this mesh".
- **Evidence:** `changelog.md:288-290`: "pre-existing order dependence (Q=4096 bf16 after the S=2048 cells in one session: the grow-only landing scratch lowers the CB budget; fails identically at HEAD, passes alone)". The grow-only logic is at `q_stem_program_descriptor.py:523-526`. The single-session blind run passed it, so whether the bug still exists is unconfirmed.
- **Recommendation:** consider a `test_regression.py` sequence case: call (1,1,2048,1536) and then (1,1,128,4096) on one mesh, with both checked. Do not add an axis (this is history, not an input facet).
- **Confidence:** low.

## 2. SUPPORTED honesty (op file `SUPPORTED` / `EXCLUSIONS`)

The registry is honest. `verifier_report.json` (blind): `supported_pass` 1092, `supported_fail` 0, `xpass_drift` 0, `xfail_*` 0, `no_axes_found` 11 (regression tests; see 1.1). I confirmed this from `golden_blind_final/verifier_report.json`. No promote or demote is needed.

One note for completeness: the parked knobs `X_INJECTORS=2` and `S1_KSPLIT>1` (`q_stem_program_descriptor.py:66,91`) are not SUPPORTED axes. Golden runs only the defaults. The perf phase found a credit-aliasing race on the multi-injector path when credits run more than one round ahead (`changelog.md:494`; breadcrumb `root_cause` 03:17). The committed `ahead = min(NI, depth)` bound is reported clean at lead 1, but that claim rests only on the subagent's runs (confidence low). If these knobs are ever promoted to user-visible options, treat that path as needing per-slot credits first.

## 3. Helper / reference docs

**3.1 `ReceiverPipe` constructor silently clears the Flag data-ready semaphore (undocumented, caused a hang)**
- **What:** building a receiver pipe after a round had already landed lost that round's flag, so the receiver hung. The contract exists only in the `.inl`.
- **Evidence:** `ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_pipe.inl:381-384` does `if Flag: data_ready_.set(INVALID)` in the constructor. The public declaration at `mcast_pipe.hpp:134-135` says nothing about it. Breadcrumb (perf-part-optimizer, 23:56): "xpre v1 hung: ReceiverPipe ctor clears flag -> built after round landed lost it". The op had to add its own warning comment at `q_stem_reader.cpp:115-116`.
- **Recommendation:** add one line at `mcast_pipe.hpp:134`: "Flag mode: the constructor resets the data-ready semaphore to INVALID. Construct the pipe before the first round can land (before crediting/acking the sender), and construct it once per kernel."
- **Confidence:** high.

**3.2 No reference names the "shared threshold counter, several producers" race outside the CCL arrival context**
- **What:** this run had two independent corruptions of the same class. In both, several producers increment one counter, and the consumer waits for `N × (k+1)`. A fast producer's next report then stands in for a slow producer's missing one. Both were silent first-call-only or 1-in-8 corruptions, not hangs. The rule exists only in `blackhole-ccl-kernels.md:272-274` ("One ordered sender per counter…") for fabric arrival counters. The generic `static-analysis-checklist.md` §6.3-6.4 (`:314-337`) checks only signal-count *balance*, which these bugs satisfy.
- **Evidence:** the port block-ready race (`changelog.md:324-334`) was "Found by ttnn-static-analyzer after two fix attempts … did not help" (also commit `f003075111b` → `c3b9b861754`). The x-credit race is breadcrumb `root_cause` (03:17): "aggregate x credit counter … a fast receiver over-credits and stands in for a slow one". The host bound `ring >= depth + 4` was itself called "not sound" (`changelog.md:427`).
- **Recommendation:** add a §6.5 to `static-analysis-checklist.md`: "Aggregate threshold waits: if M producers increment one semaphore and the consumer waits `>= M·(k+1)`, it is exact only while producers stay within one round of each other. Otherwise use per-producer words or a ring indexed by round, and bound the drift."
- **Confidence:** high.

#### Helper gaps (perf)

Both perf entries say "`### Helper bypasses` — none". That is true for the *graduated* code (Perf 2 graduated host-only changes). But the prose under each heading records bypasses in non-graduated experiments, and the graduated path carries one bypass from Refinement 6 that is in no table.

| helper | claimed | verdict | evidence | proposed fix |
|---|---|---|---|---|
| Mcast `SenderPipe`/`ReceiverPipe` (data-ready publish) | capability: no linked, never-cleared round counter; Counter pays write + atomic barrier per round (helper 2.70 vs raw 1.39 µs/round) | **missing** (narrower than claimed) | The receiver side already exists: `ReceiverPipe::receive` Counter does `wait_min(round+1)` with no clear (`mcast_pipe.inl:411-412`). What is absent is the sender side: Counter forces `linked=false` + `async_write_barrier` (`mcast_pipe.inl:238,245-247`) and a non-posted `inc_multicast` + `async_atomic_barrier` (`:271`, `:319-323`). Flag is cleared per receive (`:414-415`), so only one round can be in flight. Refinement 6 measured the cost: Counter, 1 injector, 147.7 → 172.6 µs (`changelog.md:261`). | Add a Counter-linked sender mode that publishes `round+1` as a linked posted write behind the payload. Receivers already wait without clearing. |
| Mcast `PRE_HANDSHAKE` flow control (graduated path, **unrecorded**) | none in any table | **missing** | The handshake does `consumer_ready_.wait(ack_count); set(0)` per send (`mcast_pipe.inl:75-77`), and the receiver acks only inside `receive()` (`:406-409`), so rounds run lock-step. The op disables it (`cfg.handshake = False`, `q_stem_program_descriptor.py:1139`) and runs raw cumulative slot credits: `noc_semaphore_inc` (`q_stem_reader.cpp:77,126,133`) and `noc_semaphore_wait_min` (`q_stem_injector.cpp:53`). That raw protocol later produced the aggregation race in 3.2. | Add a credit-ahead mode with **per-slot or per-receiver** credits (depth D). Record this row in the changelog. |
| CB / `cb_landing` consumption (`s3_tail`, not graduated) | capability: no CB helper consumes a static buffer in data-dependent order | **missing** (low confidence: no kernel_lib helper claims this; the CB API is FIFO by design) | `changelog.md:476` "raw-LLK `cb_landing` read-pointer override", fastest −0.7% (noise), LC1 +2.6% | Only worth a helper if a second op needs it. Track it, don't build it. |
| `matmul_block` (`s1_pace`) | not a bypass: helper vs raw compared | **no gap** | helper 2556 ns vs raw 2515 ns (−1.6%, `changelog.md:477`). The helper is at parity, which is a positive data point for the library. | none |

Sketch of the wanted Mcast mode, derived from the raw sequence at `q_stem_injector.cpp:48-56` and `q_stem_reader.cpp:65-79` (confidence: low, single op):
```cpp
// host: McastConfig cfg; cfg.flow = McastFlow::SlotCredits; cfg.depth = D;   // per-receiver credit words, D slots
// sender (injector):
pipe.send_slot(src, round);            // waits all receivers' credit for slot round % D, linked mcast, posted round+1
// receiver:
pipe.credit_slot(round);               // "slot round % D is free"  (caller calls when its CB slot frees)
pipe.receive(round);                   // wait_min(round+1), no clear
```

## 4. Agent prompts (`.claude/agents/*.md`)

**4.1 perf-coordinator: the bypass table scope drops non-graduated and pre-perf bypasses into prose**
- **What:** the table is scoped to "every graduated raw-LLK path" (`perf-coordinator.md:346`). So the run's real helper feedback went into a parenthetical under a "— none" heading, in a non-machine-readable form: two Mcast rows with measured ns, plus one CB row. The Refinement 6 Mcast flow-control bypass in shipped code appears in no table at all, because refinement entries have no bypass table.
- **Evidence:** `changelog.md:432-437` and `:511-517` both say "— none" and then "Not graduated, recorded for the helper library: … (`capability`) … helper 2.70 µs vs raw 1.39 µs".
- **Recommendation:** (a) In `perf-coordinator.md` §"The `### Helper bypasses` table", add a `graduated` column (yes/no). Non-graduated bypasses that were measured get rows too, and "— none" is reserved for "no bypass attempted". (b) Have the coordinator's Step-1 code read grep the kernels for raw semaphore or helper-disabling configs (e.g. `handshake=False`) already in the tree, and table them as `graduated=pre-existing`.
- **Confidence:** med.

**4.2 Verifier: a known harness failure persisted for 11 phases with no path to a fix**
- **What:** the verifier diagnosed `test_small_packet_size` correctly in Phase 0 and asked a human to fix it. Agents cannot edit the read-only golden suite, so it stayed red through the blind pass and counted against regression every phase.
- **Evidence:** `verification_report.md:147-151`; `op_requirements.md:50`.
- **Recommendation:** consider having `incremental-verifier.md` emit harness or test-file defects as a structured item (e.g. a `harness_issues` field in `verifier_report.json`) that the runner surfaces on the dashboard. A human then sees it before the blind pass, instead of it living only in prose.
- **Confidence:** med.

The planner, implementer and refinement prompts showed no other process confusion. Every refinement landed `[x]` with honest SUPPORTED updates and clean categories.
