# Self-Reflection: distributed_rms_norm

## Summary
- **Blind (post-perf):** golden 10,763 pass / 0 fail (2,160 xfail_expected, 40 infeasible_skipped); translated 278 pass / **1 fail** / 51 skip; regression 12/12; ring-mock 24/24. verify_supported: `supported_fail 0`, `xpass_drift 0`, `xfail_wrong_mode 1`.
- **The one blind failure is caused by the harness, not the op.** The eval's L1-profiling plugin fails a translated *refusal* test (`pytest.raises`): the op rejects the call correctly, before it allocates anything, so the measured L1 peak is 0. The verifier flagged this in Phase 0, and it was still there at the blind pass.
- **Most important finding:** golden could not see a **silent-wrong-output** kernel bug: a ragged per-core block × more than one W chunk × γ gave PCC ≈ 0. Only a translated prefill test caught it. A pinned `LOOSE_CASE` would close that gap cheaply.
- **Perf phase ran** (2 rounds, 3 graduations; focus case 20.17 → 17.75 µs). All 3 `Helper bypasses` rows hold up as `missing`. Two raw paths have no row, or a rejection reason that no longer holds (mcast, 2-D fabric routing).
- **Overall:** the op-level problems were fixed during the run. What remains is in the framework: the eval harness (L1 profiling, OOM attribution, defaulted-axis classification) and a perf prompt that names a header missing from the tree.

## 1. Golden coverage (`eval/golden_tests/distributed_rms_norm/feature_spec.py`) and test harness

**1a. Golden cannot see the silent-wrong "ragged block × multi-chunk × γ" regime** (in TARGET, but no tagger captures it)
- **What:** `cb_x_gamma` is an `R·C` ring. On cores with fewer tile-rows than `R` (an uneven split), each W chunk advanced the ring by only `R_b·C`. The next chunk then read past the ring end, giving garbage output with no error. Every golden cell passed numerically while this bug was live. The discriminating condition is (tile-rows ÷ active cores not an integer) ∧ (Wt > one chunk) ∧ γ present. That depends on the grid split, so `alignment`/`rank` cannot capture it.
- **Evidence:** `verification_report.md:12-21`: "Without `--dev`, **silently wrong output** (PCC ≈ 0). Found on the translated DeepSeek `prefill-8192`". The verifier also confirmed the first golden run's failures "pass numerically" (`verification_report.md:212`). Regression test: `tests/ttnn/unit_tests/operations/distributed_rms_norm/test_distributed_rms_norm_debug.py:70` uses `(1,1,7040,1152)`, i.e. 220 tile-rows on 106 cores (3/2 rows), Wt=36 in 2 chunks.
- **Recommendation:** add a `LOOSE_CASE` with `inputs=((1,1,7040,1152),)`, bf16, `fp32_dest_acc_en=True`, γ bf16 **TILE**, INTERLEAVED, Linear, `num_links=1` on the default mesh. Add a second entry with γ `ROW_MAJOR_LAYOUT`. Do **not** promote raggedness to an `INPUT_TAGGERS` axis: it is a property of the op's work split, not of the input. A pinned shape is the right tool here.
- **Confidence:** med. The shape triggers the bug on the 2×2 / 106-core grid the regression test uses; another grid needs a recomputed M.

**1b. Harness: L1 profiling fails refusal tests**. This is the only blind failure.
- **What:** `eval/l1_profiling.py:84-89` raises `L1ProfilingError` whenever `peak_total <= 0`. A test that *expects* the op to refuse (here `{float32, fp32_dest_acc_en=False}` → `ExcludedCell`) allocates no L1, because its input is in DRAM. So a correct refusal becomes a failure. `verify_supported` then files it as `xfail_wrong_mode` because its axes match the EXCLUSION.
- **Evidence:** `test_translated.py:655-676` (`with pytest.raises(...)`). Blind `test_results.json`: "L1 profiling returned peak_total=0; expected a positive allocation peak". The verifier diagnosed it (`verification_report.md:202-206`), but nothing in the pipeline routed the fix, so it reached blind.
- **Recommendation:** in `capture_for_item`, record no metric instead of raising when the test body passed and recorded nothing. Alternatively, exempt tests marked as refusal tests, e.g. a `@pytest.mark.no_l1_capture` on every `pytest.raises`-only translated test.
- **Confidence:** high.

**1c. Axis model: the defaulted `num_links=None` is recorded literally, not resolved**
- **What:** `axes.py` resolves the default `compute_kernel_config` from the op's factory, but records `num_links` exactly as passed. Every default-links call therefore classified outside TARGET (`num_links: [1, 2]`): 35 `xpass_drift` and 226 `xfail_wrong_mode` on the verifier's first run. The op patched around it by adding `None` to `SUPPORTED`, which TARGET does not contain.
- **Evidence:** `eval/golden_tests/distributed_rms_norm/axes.py:52` `"num_links": num_links,` versus `:42` "never hardcode the None default". `verification_report.md:54-56`. `distributed_rms_norm.py:95-98`.
- **Recommendation:** canonicalize `None` in `classify_call`, either to the op's resolved link count or to a declared `"all"` sentinel added to TARGET, and then drop `None` from the op's `SUPPORTED`. Treat this as the general rule for any defaulted axis argument.
- **Confidence:** med.

**1d. `oom_guard("op")` cannot tell output-shard OOM from an op-allocated L1 intermediate**
- **What:** the 40 `infeasible_skipped` cells (fp32 `HEIGHT_SHARDED` `(1,1,32,8192)` / `(128,8192)`) are skipped by the guard around the **op call**. Geometrically they are honest: a 1 MiB input shard plus a 1 MiB output shard on one core. But this op also allocates its own per-call L1 landing (`distributed_rms_norm_program_descriptor.py:902`, `allocate_tensor_on_device`). A buffer OOM there would also be skipped as "not the op's fault". The skip text keeps only the first line (`bank_manager.cpp:495: false`), so the failing site cannot be audited.
- **Evidence:** `eval/oom.py:64-84` ("One guard covers both setup and the op call"). Blind skip message: `INFEASIBLE_L1: op: TT_FATAL @ …bank_manager.cpp:495: false`.
- **Recommendation:** keep the full allocator message in the skip reason, or the requested size and buffer type. Consider charging op-call buffer OOMs to the op unless the request matches the output shard size.
- **Confidence:** low-med. No misattribution is shown in this run; the problem is that it cannot be checked.

**1e. (minor) The perf LOOSE_CASES are all INTERLEAVED.** The perf phase noted that WIDTH_SHARDED 640×1792 runs about 78 µs, 4× the interleaved case (`changelog.md` Perf 2, "Remaining headroom"), and no perf case pins it. If a WIDTH_SHARDED decode or prefill row exists in a model, consider pinning it. **Confidence:** low.

## 2. SUPPORTED honesty (op file `SUPPORTED` / `EXCLUSIONS`)
- **Clean:** `supported_fail 0` and `xpass_drift 0` at blind and at every phase from Refinement 1 on. Phase 0's 120 `supported_fail` (wide γ on `HEIGHT_SHARDED`) were fixed by Refinement 1, not demoted. `SUPPORTED` equals TARGET on every axis. The only EXCLUSION, `{float32, fp32_dest_acc_en=False}`, is the documented native rejection (2,160 `xfail_expected`).
- The one `xfail_wrong_mode` comes from the harness (§1b), and the `num_links: None` entry works around the axis model (§1c). Neither is an op over-claim or under-claim. No SUPPORTED edit is recommended beyond the cleanup in §1c.

## 3. Helper / reference docs

#### Helper gaps (perf)
| helper | claimed | verdict | evidence | proposed fix |
|---|---|---|---|---|
| kernel_lib eltwise chain SFPU elements (`AddBinary`/`MulUnary`/`AddUnary`/`Rsqrt`) | capability: no vector-subset mode; no multi-op fused pass | **missing**. There is no `VectorMode` anywhere under `kernel_lib/eltwise/` (grep: 0 hits); every element runs RC. Raw 300 ns vs 1161 ns, so this was a real win. The custom pass correctly used the documented `UnaryOp` CRTP extension point (`eltwise/core/chain.inl:563-577`) | `changelog.md` Perf 1 `### Helper bypasses`, row 1; `distributed_rms_norm_compute.cpp:121,206`; `op_design.md:532` "The helpers hardcode `VectorMode::RC`" | Add a `VectorMode` template knob (default RC) to the SFPU unary and binary elements. This alone would have removed the Refinement 3 `*Col` raw copies (`compute.cpp:79-110`). The fused multi-stage pass stays custom. |
| dataflow record ↔ tile-face layout moves | capability | **missing**. `local_copy_helpers_dataflow.hpp` covers L1→L1 reads, stateful reads and trid only; `tilize_helpers_dataflow` moves row-major sticks. Nothing scatters into face rows or column 0 | Perf 1 row 2; `reader.cpp:220,248,611` | Consider `tile_face_row_offset(row, col)` / `scatter_to_tile_col0()` next to `local_copy`. Low priority: one op so far. |
| timed read start (wall-clock gate) | capability | **missing**. No helper covers it, and arguably out of kernel_lib scope | Perf 2 row; `reader.cpp:89,477` | No helper change. Document the `RISCV_DEBUG_REG_WALL_CLOCK_L` idiom in the perf-pattern catalog with its calibration caveat (`ESW_READ_GBPS`, specific to one box). |
| `kernel_lib/mcast` `McastArgs` / `SenderPipe` (inv_rms distribution) | **unrecorded**. Refinement 3 added raw `noc_async_write_multicast` + `noc_semaphore_set_multicast`; the design rejected the helper because it "needs a rectangle of receivers" | **rejection reason incorrect.** `mcast/kernel/mcast_args.hpp:24-27` documents irregular receiver sets (several hardware multicasts or chain links), and `SenderPipe` takes up to 3 rectangles (`mcast_common.hpp:11`). Whether it can express the faces-0/2 linked payload with a cumulative counter is unverified | `op_design.md:450` vs `:529` (the implementation switched to per-row-run *rectangles* and never revisited the helper); `reader.cpp:739-764` | State the irregular-set / multi-rectangle capability in `mcast_pipe.hpp`'s header, which is where the planner looked, not only in `mcast_args.hpp`. Have the perf coordinator's bypass grep also cover raw paths added by refinements. |
| `kernel_lib/ccl` `FabricStream*` (port sends) | "No new helper is bypassed" (Perf 2) | **missing, unrecorded.** Perf 2's direct exchange routes multi-hop unicasts through the 2-D fabric table (`fabric_set_unicast_route(hdr, chip_id, mesh_id)`). The helper is 1-D hop-distance only (`ccl_helpers_dataflow.hpp:301` "routing is intra-mesh and hop-distance based"). The design's original rejection ("no relay-with-arrival-counter item model", `op_design.md:448`) no longer describes the default path, which is now direct, not relay | `port.cpp:65,69,260`; `changelog.md` Perf 2 bypass footnote | Record a bypass row with the new reason. In the CCL helper, consider a 2-D/table route constructor so `FabricStream` could carry the direct exchange. |

No row is an ergonomics bypass with two roughly equal ns values. Every recorded bypass bought speed or covers a capability that does not exist.

```cpp
// sketch (confidence: med). Derived from compute.cpp:79-110, where the caller re-wrote 3 elements just to pass VectorMode::C
template <Approx A = Approx::Exact, Dst Slot = Dst::D0, VectorMode VM = VectorMode::RC> struct Rsqrt;
template <Dst Slot = Dst::D0, VectorMode VM = VectorMode::RC>                         struct MulUnary;
// call site:
ckl::eltwise_chain(shape, CopyTile<in, D0>{}, MulUnary<D0, VectorMode::C>{inv_w}, AddUnary<D0, VectorMode::C>{eps},
                   Rsqrt<Approx::Exact, D0, VectorMode::C>{}, PackTile<out>{});
```
```cpp
// sketch (confidence: low, single call site). Derived from port.cpp:65,260: route by destination chip, not hop count
auto route = ccl::table_unicast_route(dst_chip_id, dst_mesh_id);
auto ch = stream_sender.open(route).arm_fused_write_inc(page_size, 1);
ch.write_fused(dst_noc_addr, arrival_sem_noc_addr);
```

**Reference-doc findings**
- **No reference states the ring-CB invariant behind §1a** ("every chunk must push the nominal `R·C` pages, or pad with fillers, so indexed reads never wrap"). The design implied it ("the design's own rule, a nominal push count", `verification_report.md:19`), but the implementer missed it. Consider adding one line to `.claude/references/blocking-model.md` or `ttnn-cb-memory-fundamentals.md`: *"An indexed/strided reader on a ring CB has no wrap; every producer iteration must advance the ring by the nominal block size, even on ragged tail blocks."* **Confidence:** med.

## 4. Agent prompts

**4a. The perf prompts require a header that does not exist in the tree**
- **What:** both perf rounds were told to instrument with `MaybeDeviceZoneScope` from `ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp`, a file that is missing from this tree. Both rounds had to notice that and fall back to the op's own `DRMS_ZONE` macro.
- **Evidence:** `.claude/agents/perf-coordinator.md:49-54`; `.claude/eval/prompts/perf_tournament_prompt.txt:9`. `changelog.md` Perf 1 and Perf 2: "`perf_instrumentation.hpp` (`MaybeDeviceZoneScope`) does not exist in this tree."
- **Recommendation:** ship the header (`.claude/references/device-zone-scope-attribution.md:11` already calls it a direct alias of `DeviceZoneScopedN`), or change both prompts to `DeviceZoneScopedN` from `tools/profiler/kernel_profiler.hpp`.
- **Confidence:** high.

**4b. Planner helper-rejection rows are not checked against the helper's own scope statement**
- **What:** `op_design.md:450` rejected `kernel_lib/mcast` for a limitation the helper says it does not have (§3). The verifier then listed the raw dataflow as "reviewed, conformant" (`verification_report.md:62-65`).
- **Recommendation:** in `incremental-planner.md` ("Helpers considered and rejected"), require each mismatch to quote the helper header line that shows the limitation. In `incremental-verifier.md`, re-check rejected-helper reasons whenever a refinement changes the shape of the raw path (here, unicast became rectangle multicast).
- **Confidence:** med.

**4c. Verifier "harness findings" have no owner.** The L1-profiling refusal failure (§1b) and the CB under-count in `device_l1_peak_bytes` (`verification_report.md:212-216`, not re-verified here) were reported in Phase 0 and never acted on. Consider having `incremental-verifier.md` write harness findings to a machine-readable list the pipeline surfaces next to the blind result, so they do not reach blind as op failures. **Confidence:** low-med.
