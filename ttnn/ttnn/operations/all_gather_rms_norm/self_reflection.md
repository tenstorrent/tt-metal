# Self-Reflection: all_gather_rms_norm

## Summary

- **Final blind result:** 7429 passed, 3860 failed, 2280 xfailed, 0 hangs (`golden_blind_final/golden_results.txt`). `verify_supported` reports `supported_pass` 7424, `supported_fail` 3799, `xpass_drift` 0.
- **All 3860 failures have one cause, and it is in the harness, not the op.** Every failure message is `L1ProfilingError: L1 profiling returned peak_total=0`. The `L1ProfilingError` is raised only after the test body passes, so these cells most likely passed numerically.
- **What separates failures from passes:** `memory_layout=INTERLEAVED`. It fails 3799 times and passes 4 times. HEIGHT_SHARDED and WIDTH_SHARDED never fail. Graph capture records no CB allocations for this op's mesh `generic_op`.
- **The verifier diagnosed this on 2026-11-07 and worked around it with `EVAL_CAPTURE_L1=0`.** Nobody fixed it. The pipeline's own golden passes (phase0, perf_1, perf_2, blind) all show the same 3799 failures. Note that the per-phase directories are re-measurements of the final op (registry snapshots are byte-identical), not a real trajectory.
- **Second finding (framework, coverage):** golden `test_op` runs only on the 2×2 mesh, so the group size G is always 2. The relay path (G>2) and `Ring` are exercised only by 3 bf16 LOOSE_CASES plus `test_ring_mock` (bf16, interleaved).
- The perf phase ran 2 rounds (focus 23.81 → 20.86 µs). It recorded one helper bypass; that row is partly a documentation gap, not a missing capability.
- All refinements are `[x]`; the queue drained.

## 1. Golden coverage → `eval/golden_tests/all_gather_rms_norm/feature_spec.py`

No blind failure cluster points to an op coverage gap: all failures are the instrumentation artifact (§2). The gap below comes from what golden *does not run*, not from a failure.

**1a. Group size G>2 (the relay path) is invisible to golden `test_op`. This is an axis-blind gap.**
- **What:** `TARGET.topology` contains `Ring`, but every `test_op` cell runs on the 2×2 system mesh, where G=2 on both cluster axes. Ring cells are pruned at collection time: 0 `test_op` Ring cells in the blind run, only 3 `test_op_loose`. With G=2 the op's multi-entry relay (`E_out>1`, `relay_lag`, relayed items in `port_sender.cpp:191-207`) never runs. That relay path is where Refinements 3/4 and Perf 1/2 made most of their changes. The only G=4 coverage is bf16 interleaved with gamma: `test_ring_mock` (3 shapes) and `LOOSE_CASES[0..2]`. No tagger captures G (`INPUT_TAGGERS` = `alignment`, `rank`), so the axes tuple looks "covered" by the G=2 cells.
- **Evidence:** blind `test_results.json` has topology=Ring on 3 cells, all `test_op_loose`. `verification_report.md:3`: "Golden `test_op` cells run on the 2×2 system mesh (Linear; Ring needs wrap links and is pruned there)". The Perf 2 graduation says "every cell … interleaved and native sharded", but its sharded/1-link G=4 numbers come from the subagent's copy (`changelog.md` Perf 2 table), not from golden.
- **Recommendation:** propose G=4 LOOSE_CASES on the existing `_PLACEMENT` (1×4, FABRIC_2D_TORUS_XY) for the riskiest relay conjunctions:
  - `((1,1,64,1000),)` bf8b, `w_non_aligned`, Ring, L1, `no_gamma`, fp32_dest_acc_en False
  - `((1,1,640,384),)` HEIGHT_SHARDED, Ring, L2, gamma fp32 TILE
  - `((17,64),)` fp32, rank 2, `h_non_aligned`, Linear (G=4 line ends), L1

  Separately, consider a harness-derived `group_size` facet in `test_axes.json`, so verify_supported can report G>2 coverage explicitly.
- **Confidence:** high that the gap exists; med on which conjunctions are riskiest.

## 2. SUPPORTED honesty → op file `SUPPORTED` / `EXCLUSIONS`

**2a. `supported_fail` = 3799 is one cluster, `memory_layout=INTERLEAVED`. Recommend neither *fix* nor *demote*: the op is honest and the harness is wrong.**
- **What:** Of the INTERLEAVED cells, 3799 fail and 4 pass; of the sharded cells, 0 fail and 7420 pass. No other axis separates fail from pass (every other axis value fails at ≈⅓). The 4 interleaved passes and the 5 ring_mock passes all record exactly `device_l1_peak_bytes=16`, which is the op's module-cached GlobalSemaphore (`all_gather_rms_norm_program_descriptor.py:764-778`, `_READY_SEM_CACHE`), allocated only on the first call per mesh. Every later interleaved call allocates nothing that graph capture sees, so the peak is 0. Sharded cells pass only because their L1 I/O buffers are counted, so their recorded L1 peak is also an under-count (it misses all CBs).
- **Evidence:** `eval/l1_profiling.py:85` (`if candidate <= 0: raise L1ProfilingError`). Every failed record has `pcc=None` and `failure_category="other"`. `verification_report.md:158-163`: "`circular_buffer_allocate` nodes are never emitted for this mesh `generic_op` … needs a fix in tt-metal's graph tracking of mesh workloads or in `eval/l1_profiling.py`". In NORMAL mode, `graph_processor.cpp:452-456` defers CB tracking to `Program::allocate_circular_buffers` (`program.cpp:2254-2280`).
- **Recommendation:** keep SUPPORTED as is. Propose a framework fix, in either order:
  1. Make the CB-allocate graph events fire for MeshWorkload / `generic_op(MeshProgramDescriptor)` programs.
  2. Until then, have `l1_profiling.capture_for_item` record `NULL` plus a distinct `failure_category` (e.g. `l1_capture_unsupported`) instead of failing a passing cell.

  Re-score this run with `EVAL_CAPTURE_L1=0`. The verifier's run that way gave `supported_fail` 0 (`verifier_report.json`), and the 720 bf8b×w_non_aligned cells were unlocked later by Refinement 2.
- **Confidence:** high on the effect and the classification; med on the exact tt-metal root cause (mesh CB tracking).

**2b. `xpass_drift` = 0.** Nothing to promote. The only EXCLUSION (`float32 × fp32_dest_acc_en=False`, 2280 cells) is prompt-mandated and xfails as expected.

## 3. Helper / reference docs

**3a. The L1-capture limitation for fused mesh ops is not written down anywhere an agent reads (absence).**
- **What:** The verifier spent a probe to rediscover that graph capture misses mesh-`generic_op` CBs, then switched measurement mode. No `.claude/references/` doc and no `eval/l1_profiling.py` docstring mentions it, so every fused-CCL op on this path will hit it again.
- **Evidence:** `verification_report.md:158`. A grep of `.claude/references/` for `L1Profiling|EVAL_CAPTURE_L1` finds nothing.
- **Recommendation:** add one line to the `eval/l1_profiling.py` module docstring and to the fused-CCL reference (`blackhole-ccl-kernels.md`): "Graph capture does not record CBs of `generic_op(MeshProgramDescriptor)` programs; `device_l1_peak_bytes` is unreliable for fused CCL ops until fixed."
- **Confidence:** high.

**3b. `ccl::FabricStream`'s staged `open → arm_inc → issue` path (a ready sent on a still-open stream) was misread as impossible.** See the table.

#### Helper gaps (perf)

| helper | claimed | verdict | evidence | proposed fix |
|---|---|---|---|---|
| `ccl::FabricStream` (ready ordering) | capability: "no way to send a fence ready on the open connection after routing one header … `signal()` closes the connection" | **undocumented** | Staged path: `open(route)` → `arm_inc()` → `inc()` keeps the stream open (`ccl_helpers_dataflow.hpp:86-91`: "Use the staged open()->arm_*->issue path"). Each `arm_*` draws its own pooled header with no ordering constraint (`hpp:121-123`, `.inl:220-233,299-311`), so the ready can go out after routing a single header and the data channels can be armed afterwards. The same misreading is already in `op_design.md:441`. | Doc line in the `signal()` paragraph: "`signal()` is terminal; for a ready followed by data on the same connection, `open()` → `arm_inc().inc()` → `arm_*` the data channels." |
| `ccl::FabricStream` (variable payload) | capability: arms one fixed payload size; chunks here vary in size | **missing** | `UnicastWriteChannel::write(dst, src)` has no size argument; `arm_unicast_write` sets `PayloadSize` once (`.inl:227-231,237-241`). Same for `FusedWriteIncChannel::write_fused` (`.inl:364-372`). The op's raw path sets the size per packet (`port_sender.cpp:167-182`). | Add a per-issue-size overload that updates `PayloadSize` with `with_state`, on the write and fused channels (sketch below). |
| `ccl::FabricStream` (2D route by node) | `unicast_route(num_hops)` is 1-D only (`op_design.md:441`) | **undocumented** | `fabric_set_line_unicast_route` routes `HybridMeshPacketHeader` by `dst_chip_id/dst_mesh_id` (`worker_routing_utils.hpp:61-66`); only the `unicast_route()` convenience fills hops alone. | Doc line at `hpp:298-307`: "On 2-D fabrics fill `line_unicast_route_info_t{dst_chip_id, dst_mesh_id}` yourself." Or add `unicast_route_to(chip, mesh)`. |

The `FabricStream` row's "helper ns" is "not measured" (raw 22830 ns), so the ns-parity test does not apply. No raw-LLK or raw-NoC bypass was added by the perf phase without a row: Perf 2 used `noc_semaphore_inc` only, and the compute kernels are unchanged since Refinement 5. The raw compute pieces (`compute.cpp:89,249-264`) are recorded in `op_design.md:437-446`.

Sketch (confidence: low, from one call site; derived from `port_sender.cpp:164-190`):
```cpp
// wanted: one armed header ring (N headers in flight), per-issue payload size, write or fused write+inc
auto ring = stream.arm_write_ring<N>();                               // route bound once, N pooled headers
ring.write(dst_noc, src_l1, bytes);                                   // non-blocking, rotates headers
ring.write_fused(dst_noc, src_l1, bytes, arrival_sem_noc);            // last chunk of an item
```

**3c. A planner design note gave a wrong build fact (`-Os`).** The planner wrote "Kernels build at -Os" (`op_design.md:527`); the perf subagent measured "descriptor kernels already build at O2; op_design's '-Os' note was wrong" (`changelog.md` Perf 2, port-stream-rate menu). No reference states the descriptor kernels' optimization level. Recommend adding it to the perf-pattern reference. Confidence: med.

## 4. Agent prompts → `.claude/agents/*.md`

**4a. `perf-coordinator.md` requires a header that does not exist in this tree.**
- **What:** Step 1.2 requires `MaybeDeviceZoneScope` from `ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp`, and so does `perf_tournament_prompt.txt:9`. The file is absent, so the coordinator fell back to an op-local gated macro in both rounds. `references/device-zone-scope-attribution.md:3-5` itself says the header's distribution "is still open".
- **Evidence:** `perf-coordinator.md:49-50`; `changelog.md` Perf 1: "`perf_instrumentation.hpp` (`MaybeDeviceZoneScope`) does not exist in this tree, so the op's own gated macro is used". Perf 2 repeats it.
- **Recommendation:** land the header, or change the prompt line to "use `DeviceZoneScopedN` behind an op-local compile-time gate (see `device-zone-scope-attribution.md`)".
- **Confidence:** high.

**4b. `incremental-verifier.md` let a harness defect be routed around rather than escalated.**
- **What:** The verifier correctly classified the L1 failure as "not op defects; nothing for the queue", reran with `EVAL_CAPTURE_L1=0`, and published `supported_fail: 0`. The pipeline kept measuring with capture on, so every later phase and the final blind run report 3799 failures that nobody owned. The verifier's numbers and the pipeline's numbers come from different measurement modes, and nothing flags the difference.
- **Evidence:** `verifier_report.json` `run.how`: "EVAL_CAPTURE_L1=0"; `golden_phase0/verifier_report.json` and later: `supported_fail` 3799. The prompt has no guidance for instrumentation failures (`incremental-verifier.md:204-205` only gives the runner command).
- **Recommendation:** add to the verifier prompt: "If a harness or instrumentation failure (not an op failure) affects the cells, do not change the measurement mode silently. Record it as a framework blocker in `verifier_report.json` (e.g. `harness_blockers: [...]`) so the pipeline/dashboard re-scores or surfaces it."
- **Confidence:** med.

**4c. The perf coordinator's golden guards also ran in a different mode from the pipeline gate.** Its guard slices report "0 failed" (Perf 1: 183 passed; Perf 2: 1456 cells), while `golden_perf_1`/`golden_perf_2` show 3799 `supported_fail`, so the coordinator evidently ran with capture off. Same fix as 4b. Also consider making `perf-coordinator.md` compare against the pipeline's `golden_perf_N/verifier_report.json` delta rather than its own slice. Confidence: med.
