# Self-reflection: matmul_reduce_scatter (run 1)

## Summary
- **Final blind result: 1041/1041 passed.** That is 939 golden, 72 ring-mock, 24 fabric-config and 6 regression cells. verify_supported shows 1035 `supported_pass`, 0 `supported_fail`, 0 `xpass_drift`, and 6 `no_axes_found` (the regression cells have no axes by design). There are no failure clusters to work from.
- All 4 refinements are ticked `[x]`. Two perf rounds ran: focus case 126.0 → 116.7 µs against a 124.8 µs target.
- **Most important finding:** the perf phase produced 4 helper-bypass rows, and all 4 hold up against the `.hpp`/`.inl` as genuinely **missing** capabilities. Two of the bypasses also reach into library `detail::` symbols or raw CB registers.
- **Second:** Perf 2 graduated a branch that is decided by run-time timing (fused vs late finish). The permanent tests have no hook that forces the late side.
- The problems are **framework-level, not op-level**: helper coverage, reference-doc silences, the L1-profiling harness, and the phase-0 artifact being overwritten.

## 1. Golden coverage (`eval/golden_tests/matmul_reduce_scatter/feature_spec.py`)
No failure cluster exists, so no `LOOSE_CASES` or `INPUTS` change is proposed.

**F1.1: The only coverage gap is a timing-dependent kernel branch, which no input can pin.**
- **What:** Fusable arrival entries pick fused vs late at run time. Which way they go depends on when the arrival lands, not on the input shape, so a loose case cannot force the late side. The late side is likely rare on the golden shapes.
- **Evidence:**
  - `kernels/matmul_reduce_scatter_compute.cpp:123`: a non-blocking received-count read decides `fused`/`late`.
  - `changelog.md` Perf 2: "`FIN` fires only for late, lagged and post-lag entries; on focus it no longer fires at all".
  - The forcing defines `FF_FORCE_LATE`/`FF_FORCE_FUSED` exist only as kernel `#ifdef`s. Nothing in the op's `.py` files or the tests references them.
- **Recommendation:** Track this under §4 (forcing hook plus a permanent test), not under golden.
- **Confidence:** med.

**Checked and found adequate:** the perf-added `super_block` S=1 fallback, which applies when the K-block count is odd.
- INPUTS with K ∈ {512, 1536, 4608} give Kt/kb ∈ {1, 3, 9} at `KB_MAX=16`, so S=1 is exercised alongside S=2.
- Confidence med: kb also depends on L1 fit (`matmul_reduce_scatter_program_descriptor.py:459`).

## 2. SUPPORTED honesty (op file `SUPPORTED` / `EXCLUSIONS`)
`SUPPORTED` is honest: 0 `supported_fail` and 0 `xpass_drift` in `golden_blind_final/verifier_report.json`, and the same in every phase directory.
- Refinement 1 promoted bf8_b activations and bf4_b weights correctly. Before it, those cells were 468 + 156 expected xfails (`golden_phase0_capture_on/test_results.json`: "dtype=DataType.BFLOAT8_B not in SUPPORTED").
- No edit is proposed.

## 3. Helper / reference docs

#### Helper gaps (perf)
| helper | claimed | verdict | evidence | proposed fix |
|---|---|---|---|---|
| `compute_kernel_lib::matmul_block` | capability (Perf 1: no DEST residency across K-blocks; no wrapped-window addressing) | **missing** | `matmul_block_helpers.inl:214` loops over K-blocks on the outside and over subblocks inside (`:294`/`:306`), with `tile_regs_acquire` per (K-block, subblock) at `:307`. Folding S windows into one K-block (`in0_block_k = S·kb`) does not work around it: A windows are pushed as `rows×kb` each, so the in0 row stride is wrong, and the helper waits for the whole block up front. That loses the progressive per-window wait/pop the raw path relies on (`compute.cpp:244`, `:272`). The raw path also writes the UNPACK `fifo_rd_ptr` itself (`:258`). Measured: compute floor 112.8 → 109.1 µs. | Add a DEST-resident K mode, sketched below. |
| `dataflow_kernel_lib::ReceiverPipe::receive()` | capability (Perf 1: fused ack + wait, no poll hook) | **missing** | `mcast/kernel/mcast_pipe.inl:407-416`: `consumer_ready_.up(...)` and then `data_ready_.wait(VALID)` happen in one call. `mcast_pipe.hpp:137` documents them as fused. The bypass at `reader.cpp:47,53` calls **`detail::mcast_semaphore_id`**, so the op now depends on a library-internal symbol. Measured: focus 126.1-126.6 → 123.1-124.0 µs. | Split into `ack(round)` and `wait(round, poll)`, keeping `receive()` = `ack` + `wait`. Sketch below. |
| `compute_kernel_lib::copy` / `add` / `eltwise_chain` | capability (Perf 2: no op on caller-owned, pre-filled DEST; no `acc_to_dest`) | **missing** (one sub-claim is imprecise) | The chain always owns acquire/commit/release (`eltwise/core/chain.inl:2585-2590`, `:3123`). The claim that "acc_to_dest is missing" is inaccurate as worded: `DestAccumulation` exists (`eltwise/api/chain.hpp:277-287`). Its semantics are different, though: it reduces a walk into one helper-acquired tile, not a per-tile `DEST += x` onto an existing DEST. The raw path also rewinds the PACK `fifo_wr_ptr` for the in-place late pack (`compute.cpp:307`). Measured: focus 119.0 → 116.6-117.2 µs. | (a) Add a caller-owned-DEST element mode, sketched below. (b) Add a doc line to `chain.hpp:277`: "DestAccumulation reduces into one helper-acquired tile. It does not add onto a DEST the caller already filled." |
| No helper: compute-side CB "pages landed?" check plus UNPACK→MATH/PACK broadcast | capability | **missing** | `cb_pages_available_at_front` exists only in the non-TRISC branch (`tt_metal/hw/inc/api/dataflow/circular_buffer.h:120`, inside `#else` of `COMPILE_FOR_TRISC`). The mailbox broadcast exists only in special-purpose forms (`api/compute/cb_api.h:172-215`, `get_tile_address`/`read_tile_value`). The raw code reads `get_cb_tiles_received_ptr` and `tiles_acked` directly (`compute.cpp:123`) and broadcasts by mailbox (`:138`). | Add a compute `cb_pages_available_at_front(cb, n)` that returns the same bool on all 3 TRISCs. |

- **No ergonomics or too-hard row has ~equal ns.** Every row is a capability gap with a measured win of 2.4-3.7 µs.
- **Record vs code:** I found no unrecorded perf-phase bypass. The other raw sites predate the perf phase and are listed under `op_design.md:367` "Helpers considered and rejected":
  - the port sender's raw fabric API calls;
  - the arrival-counter `noc_semaphore_wait_min/inc` at `reader.cpp:461-466`, from commit `1bb68ceecfb`.
  - The fabric rejection reason is "the CCL helper library is unmeasured" (`port_sender.cpp:14`). The fix there is to measure `ccl_helpers_dataflow.hpp` against `fabric_reduce_scatter`, not to change the op.

```cpp
// matmul_block — DEST-resident K (confidence: med; derived from compute.cpp:233-275)
matmul_block<..., KResidency::Windows>(in0_buf, in1_buf, out_buf, interm_buf, shape,
    /*windows_per_pack=*/S, /*in0_window_tiles=*/rows*kb, /*in1_window_tiles=*/win_n*kb);
// The helper waits progressively per window on the first subblock, pops per window on the last,
// and handles ring wrap itself.

// ReceiverPipe — split phase (confidence: high; reader.cpp:44-58)
rx.ack(round);                               // right after reserving the slot
rx.wait(round, [&] { try_land(); });         // poll hook runs while data_ready != VALID

// eltwise on caller-owned DEST (confidence: low; single call site compute.cpp:149-190)
tile_regs_acquire(); /* matmul subblock into DEST[0..h*w) */
eltwise_in_dest(SubblockView{h, w, col_base, cols},
    BinaryFpu<Add, input(cb_hold), input(cb_zero), DestAccumulate::Tile>{},
    DestReuseBinary<Add, input(cb_arr), DEST_TO_SRCA>{});   // no acquire / pack / release
```

**F3.1: `blackhole-ccl-kernels.md` §3.4 does not say a fused increment only orders its own write.**
- **What:** The `inc_every` streaming-sender pattern shows one fused increment covering earlier chunks. That holds only because every chunk goes to a single peer counter or core. The doc never states that limit. The implementer generalised the pattern to groups of destination cores, which is incorrect.
- **Evidence:**
  - `.claude/references/blackhole-ccl-kernels.md:167,184`.
  - Commit `b4d22e88045`: "remove INC_GROUP lever: incorrect for groups > 1".
  - Implementer breadcrumb: "ring G=4 PCC 0.84-0.93".
- **Recommendation:** Add this line: "A fused write+inc orders only its own packet's write. It covers earlier chunks only when they went to the same destination core over the same connection."
- **Confidence:** high.

**F3.2: L1 profiling cannot measure per-chip MeshWorkload ops, so this run has no L1 metric.**
- **What:** `EVAL_CAPTURE_L1=1` returned `peak_total=0` and failed 386 cells of the original phase-0 golden run. Since then, every run has used `EVAL_CAPTURE_L1=0`, so `device_l1_peak_bytes` is empty for all 1041 blind cells.
- **Evidence:**
  - `golden_phase0_capture_on/test_results.json`: "L1 profiling returned peak_total=0".
  - Verifier breadcrumb: "MeshWorkloadImpl::compile allocates CBs on thread-pool workers … GraphTracker processors are thread_local".
  - `eval/eval_test_runner.sh:311-314` now defaults capture to off.
- **Recommendation:** Fix the root cause by propagating graph processors to the compile workers. Until then, have the metrics plugin record "L1 not captured (multi-device)" explicitly instead of leaving the field blank.
- **Confidence:** high.

**F3.3: No documented Python source for the L1 budget.**
- **What:** The implementer found no documented way to get the allocatable L1 region. The obvious candidate is wrong.
- **Evidence:** Implementer breadcrumb on `.claude/references/ttnn-python-utility-bindings.md`: "get_max_worker_l1_unreserved_size … ~70 KiB larger than l1_bank_size … causing CB/L1-buffer clashes".
- **Recommendation:** Document `get_device_info(mesh).l1_bank_size` as the L1 budget source.
- **Confidence:** med.

**F3.4: `blackhole-fabric.md` rule 6 assumes a probe kernel, which a one-dispatch op cannot run.**
- **What:** Rule 6 tells the author to find Ethernet cores with a probe kernel. A probe-free mapping works and is not documented.
- **Evidence:** `.claude/references/blackhole-fabric.md:131-138`. Implementer breadcrumb: the eth channel "indexes the chip non-harvested eth list".
- **Recommendation:** Add the probe-free channel → core mapping to rule 6.
- **Confidence:** med.

**F3.5: No documented way to reset the device after a passing `--dev` fabric run.**
- **What:** After a passing `--dev` fabric run, the next mesh open fails (Ethernet core heartbeat). The wrapper only resets after hangs, and agents may not run `tt-smi -r`.
- **Evidence:** `blackhole-fabric.md:169` says "reset the device if the next open fails" but not how. Implementer breadcrumb: "The only workaround was touching /tmp/tt-device.dirty".
- **Recommendation:** Document the dirty-flag path, or make `run_safe_pytest.sh --dev` reset after fabric runs.
- **Confidence:** med.

## 4. Agent prompts

**F4.1: `perf-coordinator.md` graduation steps do not cover branches decided at run time.**
- **What:** `fused_finish` graduated with determinism checked only through experiment-only `#ifdef`s (`FF_FORCE_LATE`/`FF_FORCE_FUSED`). The shipped op cannot force the late branch, so its bits are covered only when timing happens to produce it.
- **Evidence:**
  - `compute.cpp` `#ifdef FF_FORCE_LATE`. Nothing in the op's `.py` files or the tests references it.
  - `.claude/agents/perf-coordinator.md:192-203`: steps 1-3 ask for golden green and a guard set, nothing about forcing.
- **Recommendation:** Add a graduation step: "A branch chosen by run-time timing or data must ship a host test hook (like `_FORCE_XBLOCK_RESIDENCY`) and a permanent test that forces each side."
- **Confidence:** med.

**F4.2: Pipeline issue: the `golden_phase0` artifact is not phase 0.**
- **What:** `golden_phase0/` was re-run at 12:03-12:17 on 2026-10-16, against HEAD `09d7f1d75b9` (Refinement 2). Its snapshot lists bf8_b activations as SUPPORTED, but the true phase-0 snapshot (`golden_phase0_capture_on/registry_snapshot.json`) has `dtype` = `['BFLOAT16']`. The true phase-0 run was itself voided by F3.2. So the dashboard's refinement trajectory starts from a post-R2 measurement, and no valid phase-0 golden result exists.
- **Evidence:** directory mtimes. `golden_phase0/pytest_stdout.log` starts at "2026-10-16 12:03:47".
- **Recommendation:** When the orchestrator resumes, it should not overwrite `golden_phase0`. Write `golden_phase0_rerun_<sha>` and record the HEAD sha in `registry_snapshot.json`.
- **Confidence:** high.

**F4.3: The verifier's golden run cannot finish in the foreground.**
- **What:** `incremental-verifier.md:204` runs `eval/eval_test_runner.sh` in the foreground. The multi-device suite takes about 8 minutes to precompile plus 13 minutes of tests, which exceeds the 10-minute Bash limit. The killed run wasted a cycle and left the device dirty.
- **Evidence:** Verifier friction breadcrumb (`incremental-verifier_breadcrumbs.jsonl:2`).
- **Recommendation:** "For suites longer than about 8 minutes, launch detached (`nohup setsid … &`), poll `golden_results.txt`, and set the dirty flag before starting."
- **Confidence:** med.

**F4.4: The implementer's acceptance test cannot run as one session, and the problem persisted.**
- **What:** `line_mesh` and `ring_mesh` are separate module fixtures, so a single invocation fails with `SetFabricConfig ... while devices are still open`. Every later phase worked around this by running per placement group instead of fixing it.
- **Evidence:** Implementer breadcrumb on `test_matmul_reduce_scatter.py:82`. Perf 1 and Perf 2 changelog: "12 fixture errors … come from the harness".
- **Recommendation:** In `ttnn-implementer.md`, require multi-placement CCL tests to use one parametrized mesh fixture that closes the previous mesh first.
- **Confidence:** low-med.
