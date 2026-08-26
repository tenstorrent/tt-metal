# WH Galaxy Device Evidence — Gap 2: `Prefetcher2D` / Galaxy CCL hardware qualification

Commit `de4c8f4e659cb7a0dfd255f0b34d33f42f43a0bd` ("add reusable WH Galaxy 2D modules"),
branch `gongyu/tttv2_wh_glx_2d_modules`, run on real hardware across two reserved 6U WH Galaxy hosts
(32 devices each): `wh-glx6u-05-special-ctr-apbernal-for-reservation-116970` on 2026-08-25 and
`wh-glx6u-05-special-ctr-apbernal-for-reservation-117439` on 2026-08-26, the reservation having rolled
over between sessions (§2).

Brief: [`../tttv2_milestone_a_gap_briefs/gap2_prefetcher2d_galaxy_ccl_hardware.md`](../tttv2_milestone_a_gap_briefs/gap2_prefetcher2d_galaxy_ccl_hardware.md).
Completion handoff: [`../tttv2_milestone_a_gap_briefs/gap2_completion_handoff.md`](../tttv2_milestone_a_gap_briefs/gap2_completion_handoff.md).
Session-3 working notes: [`SESSION3_NOTES.md`](SESSION3_NOTES.md).

## 1. Summary

**The gap is closed for the seven lifecycle cases and closed as a documented incompatibility for the
eighth.** `Prefetcher2D` and the Galaxy resource owner now have their own hardware qualification: the
full prefill↔decode transition matrix, the failed-transition rollback, cleanup from either active
mode, the second-owner leak detector, the sealed-address readback off the 12 sender cores, and the
registration/sealing rejections — none of which had ever executed on silicon — all run green, three
times over in fresh processes.

| | |
| --- | --- |
| Cases attempted | 8 of 8 |
| Cases passing on hardware | **7** (three fresh whole-file processes each) |
| Cases terminal-but-not-passing | **1** — case 8, `attention_decode_with_active_prefetch`: structurally incompatible, diagnosed, recommended for Milestone B (§6) |
| Cases blocked on infrastructure | 0 |
| Findings | **F1** — `cleanup()` cannot release the global circular buffer (§5). **F2** — an undersized `global_cb_size` is silently accepted at seal time (§7). |
| Wall clock, session 3 | 08:23–08:52 UTC, ~29 m of a 12 h budget: 12 m 32 s of green device testing (three repeats plus the attention suite), 7 m of case-8 abort-and-kill, ~7 m in two `tt-smi -glx_reset` runs |
| Wall clock, all three sessions | 2 h 52 m (session 1) + 16 m (session 2) + 29 m (session 3) |
| Device left | clean — `tt-smi -ls` lists all 32 boards, ids 0–31, no `python`/`pytest` processes ([`logs/host07_tt_smi_final.log`](logs/host07_tt_smi_final.log)) |

**The run spanned three sessions, and neither of the first two ended on a technical failure.**
Session 1 (2026-08-25) wrote the suite, performed the shared-helper refactor, and was killed by the
org's monthly spend limit (HTTP 429) mid-way through the repeat runs. Session 2 (2026-08-26) ran case
8 for the first time and diagnosed its teardown stall, then killed itself 16 minutes in: it ran
`pgrep -af pytest`, matched its own `timeout … claude -p "<prompt>"` wrapper because the driver passed
the prompt as an argv element and the prompt contained the word "pytest", and signalled it. The
driver now pipes the prompt on stdin. Both sessions' findings were written to disk before they died
and are re-verified against the logs here; §4 flags one place where the handoff's own account of
them was wrong.

The headline, plainly: **the ownership defect this gap existed to find is real, and it is F1.** A
`Prefetcher2D` cannot free the global circular buffer it created, because ttnn exposes no free for
one; every consumer module handed a `Prefetcher2DContext` holds a live handle, and a consumer that
outlives `cleanup()` keeps ~55 MB of L1 pinned. The next owner's `seal()` then dies with an L1
out-of-memory error that names no prefetcher. It is invisible to the mock suite, and a Milestone C
model-owned executor doing repeated startup/serving/cleanup cycles will hit it.

## 2. Environment

| Item | Value |
| --- | --- |
| Commit | `de4c8f4e659cb7a0dfd255f0b34d33f42f43a0bd` (`gongyu/tttv2_wh_glx_2d_modules`) |
| Host | `wh-glx6u-05-special-ctr-apbernal-for-reservation-117439` |
| Devices | 32 `/dev/tenstorrent` nodes; all 32 Wormhole `tt-galaxy-…L` boards |
| Mesh | `(8, 4)`, `FabricConfig.FABRIC_1D_RING`, `DispatchCoreAxis.COL` |
| Compute grid | `7 × 10` = 70 Tensix cores per device (derived device-free in [`logs/host06_grid_derivation.log`](logs/host06_grid_derivation.log)) |
| Arch | `wormhole_b0` |
| Build | `Release` (`build_Release`) |
| Python | 3.10.21 in `python_env/`, pytest 9.0.3 |

**The reservation rolled over between sessions.** The 2026-08-25 logs (`host01`–`dev05`, `probe01`,
`probe02`) were taken on `…116970`; everything from `dev06` onward is on `…117439`. The hostname in
each log's `###` header is the authoritative one, and `host01`–`dev05`/`probe01`/`probe02` predate that
convention — their host is attributed from the completion handoff, which records the rollover
explicitly, and from the same-reservation gap 1 report. Worth keeping in mind when comparing L1
addresses or timings across the two days.

**The working tree is dirty by design.** The new suite and the shared-helper extraction *are* the
deliverable of this job and are deliberately left uncommitted for review. No `git commit`, `push`,
`checkout`, `stash`, or `reset` was run at any point. Nothing under `models/common/modules/` was
touched by this gap job at all — in particular `prefetcher_2d.py` is unmodified, as the handoff
requires.

Files this gap job created or changed:

| File | State |
| --- | --- |
| `models/common/tests/modules/prefetcher/test_prefetcher_2d_wh_galaxy.py` | **New**, untracked, 863 lines. The qualification suite: 7 tests, 8 cases |
| `models/common/tests/modules/_mlp_2d_galaxy.py` | **New**, untracked, 468 lines. The qualified MLP2D geometry, factored out of the MLP suite so the prefetcher suite drives the identical payload |
| `models/common/tests/modules/_wh_galaxy_hardware.py` | Tracked, modified. `+71 −23`. `_create_hardware_prefetcher` split so `GALAXY_PREFETCH_SENDER_COORDS`, `galaxy_prefetcher_sender_cores()` and `galaxy_prefetcher_config(…, global_cb_size=…)` are public. Behaviour-preserving |
| `models/common/tests/modules/mlp/test_mlp_2d_wh_galaxy.py` | Tracked, modified, `+22 −421`. Gutted into `_mlp_2d_galaxy.py`; now imports what it used to define |
| `tttv2_2d_modules_work_log.md` | Tracked, modified. The hardware checkpoint appended by this job |
| `tttv2_milestone_a_gap2_evidence/` | **New**, untracked: this report, `SESSION3_NOTES.md`, and `logs/` |

Note that `.gitignore:7` ignores `*.log`, so the raw logs under `logs/` are **not** stageable and
live only on this host. That is pre-existing repo policy, not a choice made here, but it means the
report has to stand on its own for anyone reading the diff.

The MLP-suite gutting is the one genuinely risky edit in the set: it is a large change to a file that
is *recorded Milestone A evidence*. Its validation is the device re-run in
[`logs/dev01_mlp_regression.log`](logs/dev01_mlp_regression.log) (`4 passed in 127.80s`) plus the
attention re-run in §8.

## 3. What this suite closes

Before this suite, `Prefetcher2D` and the Galaxy resource owner had 446 lines of host coverage built
entirely on `FakeMesh`/`FakeTensor`/`MagicMock`, and were exercised on silicon only *incidentally*, as
a side effect of MLP2D and one RMSNorm2D test. Every one of those device tests pins a single mode for
its entire lifetime — `_invoke` in the MLP suite takes `mode` as a constant — so on real hardware the
following had **never executed at all**:

| Never run on silicon before this suite | Now covered by |
| --- | --- |
| `decode → prefill` transition | case 2, steps 01/03/06/08/10 |
| `prefill → decode` transition | case 2, steps 02/04/07/09 |
| repeated decode (`decode → decode`) | case 2, step 05 — the seam between the two cycles |
| repeated prefill (`prefill → prefill`) | case 2, step 11 |
| failure *during* a transition, and the rollback | case 3 |
| cleanup from an active decode mode | case 4 |
| cleanup from an active prefill mode | case 5 |
| a second `Prefetcher2D` on the same mesh in the same process | cases 4 and 5 (the leak detector) |
| the sealed weight addresses being the real device buffer addresses | case 1 |
| the packed address tensor actually residing on the 12 sender cores | case 1, read back with `ConcatMeshToTensor` |
| the `__enter__`/`__exit__` path removing both managers when the body raises | case 6 |
| registration/sealing rejections against real device tensors | case 7 |

Two design choices in the suite are worth stating because they are what make the evidence mean
something:

- **The payload is the qualified MLP2D geometry, imported wholesale** from `_mlp_2d_galaxy.py`. The
  PCC numbers here are therefore directly comparable with the recorded MLP2D evidence, and a wrong
  core grid cannot creep in through transcription. That is why the extraction was done at all.
- **PCC is asserted at every transition step.** A mode that merely switched but left the ring, the
  global CB or the stall group inconsistent still produces a tensor. Only correlating that tensor
  against the HuggingFace reference proves the context is *usable* after the transition rather than
  merely *selected*. ttnn exposes no getter for the loaded subdevice manager or the active stall
  group, so `_SubdeviceRecorder` shadows the four `ttnn.MeshDevice` lifecycle methods with forwarding
  wrappers; the device sees an unchanged call sequence and the test additionally sees which calls
  were made.

## 4. Results

Node IDs are abbreviated: every one carries the
`[wormhole_b0-device_params0-mesh_device0]` suffix, and cases 4/5 additionally `-decode`/`-prefill`.

### 4.1 Final state, one row per case

| # | Case | Final state | Evidence |
| --- | --- | --- | --- |
| 1 | `sealed_resources_are_real_on_device` | **PASSED** ×3 | [dev07](logs/dev07_prefetcher_cases1to7_run01.log), [dev08](logs/dev08_prefetcher_cases1to7_run02.log), [dev09](logs/dev09_prefetcher_cases1to7_run03.log) |
| 2 | `mode_transition_matrix` | **PASSED** ×3 | dev07, dev08, dev09 |
| 3 | `failed_transition_rolls_back_on_device` | **PASSED** ×3 | dev07, dev08, dev09 |
| 4 | `cleanup_from_active_mode_frees_the_mesh[decode]` | **PASSED** ×3 | dev07, dev08, dev09 |
| 5 | `cleanup_from_active_mode_frees_the_mesh[prefill]` | **PASSED** ×3 | dev07, dev08, dev09 |
| 6 | `context_manager_cleanup_leaves_mesh_reusable` | **PASSED** ×3 | dev07, dev08, dev09 |
| 7 | `registration_and_sealing_rejections` | **PASSED** ×3 | dev07, dev08, dev09 |
| 8 | `attention_decode_with_active_prefetch` | **FAILED, terminal — incompatible by construction** (§6) | [dev06](logs/dev06_attention_with_prefetch_isolated.log), [dev06b](logs/dev06b_gdb_teardown_backtrace.log), [dev11](logs/dev11_attention_with_prefetch_traceback.log), [host05](logs/host05_case8_subdevice_overlap.log) |

### 4.2 The whole path, honestly — including the two failing runs

A reader should be able to see how the file got from "written" to "green three times", because two of
the intermediate runs failed and they failed for **two different reasons**, only one of which the
handoff recorded correctly.

| Log | Selection | Result | What it means |
| --- | --- | --- | --- |
| [`host01_regression.log`](logs/host01_regression.log) | prefetcher/galaxy/MLP **host** suites, post-refactor | `78 passed in 12.12s` | The shared-helper extraction did not disturb the mock suites |
| [`host02_precommit.log`](logs/host02_precommit.log) | pre-commit on the touched files | `exit=1` — `isort` reformatted `test_mlp_2d_wh_galaxy.py` | Auto-fix, not a violation |
| [`host03_precommit_rerun.log`](logs/host03_precommit_rerun.log) | same | `exit=0`, all hooks Passed/Skipped | Clean |
| [`dev01_mlp_regression.log`](logs/dev01_mlp_regression.log) | `test_mlp_2d_wh_galaxy.py` | **`4 passed in 127.80s`** | The MLP suite survives being gutted into `_mlp_2d_galaxy.py` |
| [`dev02_rmsnorm_regression.log`](logs/dev02_rmsnorm_regression.log) | `test_rmsnorm_2d_wh_galaxy.py` | **`8 passed in 33.55s`** | The other shared-helper consumer is unaffected |
| [`probe01_seams.log`](logs/probe01_seams.log) | `tttv2_gap2_scratch/test_probe.py` | test `PASSED`, then the **process aborted**, `exit=134` | Scratch probe of the rejection seams. See §7 and §9 |
| [`dev03_prefetcher_run01.log`](logs/dev03_prefetcher_run01.log) | whole file, case 8 deselected | `2 failed, 5 passed, 1 deselected in 212.64s` | **The two failures are finding F1**, not a test bug — see below |
| [`probe02_global_cb_lifetime.log`](logs/probe02_global_cb_lifetime.log) | `tttv2_gap2_scratch/test_probe2.py` | `1 passed in 28.39s` | The 10-step experiment that isolated F1 to the global-CB handle specifically (§5) |
| [`dev04_prefetcher_run02.log`](logs/dev04_prefetcher_run02.log) | whole file, case 8 deselected | `2 failed, 5 passed, 1 deselected` | The two failures here **are** the agent's own test bug: `UnboundLocalError: local variable 'prefetcher' referenced before assignment`, introduced by the F1 workaround referencing `prefetcher` after `del prefetcher` |
| [`dev05_cleanup_nodeids.log`](logs/dev05_cleanup_nodeids.log) | cases 4 and 5 by node ID | **`2 passed, 6 deselected in 92.25s`** | Both repaired cases pass isolated |
| [`dev06_attention_with_prefetch_isolated.log`](logs/dev06_attention_with_prefetch_isolated.log) | case 8 by node ID | **`FAILED`** — `TT_FATAL: Programs must be executed on a single sub-device` | Case 8's first ever execution (§6) |
| [`dev06b_gdb_teardown_backtrace.log`](logs/dev06b_gdb_teardown_backtrace.log) | — | gdb backtrace | The post-abort teardown stall (§6.3) |
| [`reset01_before_session3.log`](logs/reset01_before_session3.log) | `tt-smi -glx_reset` | `Re-initialized 32 boards`, `exit=0` | Session 2 hard-killed a process wedged in `MeshDevice::close()` and never reset; session 3 reset before touching the device |
| [`host04_collect_only.log`](logs/host04_collect_only.log) | `--collect-only -q` | **`8 tests collected`**, `exit=0` | 8 cases, no collect-time deselection. The `1 deselected` every run reports is the explicit `--deselect` of case 8 |
| [`dev10_attention_regression.log`](logs/dev10_attention_regression.log) | `test_attention_2d_wh_galaxy.py` | **`2 passed in 75.36s`** | The last outstanding regression gate (§8) |
| [`dev11_attention_with_prefetch_traceback.log`](logs/dev11_attention_with_prefetch_traceback.log) | case 8 by node ID | `TT_FATAL` reproduced; `exit=124` | The traceback that pins the failing call (§6.1) |
| [`reset02_after_case8.log`](logs/reset02_after_case8.log) | `tt-smi -glx_reset` | `Re-initialized 32 boards`, `exit=0` | Budgeted after case 8, per §6.3 |
| [`host07_tt_smi_final.log`](logs/host07_tt_smi_final.log) | `tt-smi -ls` | 32 boards, ids 0–31 | Device left clean |

**Correcting the handoff.** The handoff states that the two `dev03`/`dev04` failures were both "the
agent's own test bug … `UnboundLocalError`". That is true of `dev04` only. `dev03`'s two failures are
finding F1 manifesting in the real suite: the *second* owner's
`Prefetcher2D.seal()` → `create_global_circular_buffer` aborts with

```
TT_FATAL @ tt_metal/impl/allocator/bank_manager.cpp:462: false
Out of Memory: Not enough space to allocate 55444480 B L1 buffer across 70 banks,
where each bank needs to store 792064 B, but bank size is 1393472 B
(allocated: 794816 B, free: 598656 B, largest free block: 574688 B)
```

with the Python frames landing in `_wh_galaxy_hardware.py:297 _create_hardware_prefetcher →
prefetcher.seal()` (`dev03_prefetcher_run01.log:880-935`). This matters: **F1 was first observed as a
genuine failure of the real qualification suite**, not only in a scratch probe. The `UnboundLocalError`
in `dev04` was introduced *by the fix for it*.

### 4.3 The three fresh whole-file processes

The brief's anti-aliasing requirement, and the one the two 2026-08-25 root causes justify: a case that
flips between processes is reading aliased or uninitialised L1.

| Run | Log | Result |
| --- | --- | --- |
| 01 | [`dev07_prefetcher_cases1to7_run01.log`](logs/dev07_prefetcher_cases1to7_run01.log) | **`7 passed, 1 deselected in 224.28s`**, `exit=0` |
| 02 | [`dev08_prefetcher_cases1to7_run02.log`](logs/dev08_prefetcher_cases1to7_run02.log) | **`7 passed, 1 deselected in 225.96s`**, `exit=0` |
| 03 | [`dev09_prefetcher_cases1to7_run03.log`](logs/dev09_prefetcher_cases1to7_run03.log) | **`7 passed, 1 deselected in 226.15s`**, `exit=0` |

Three separate `python -m pytest` processes, each opening and closing the 32-device mesh seven times.
Case 8 is deselected by node ID so that its abort cannot wedge the mesh mid-file and poison the other
seven; it is run separately in §6.

**The `[gap2]` output of all three runs is byte-identical** — every PCC value to seven decimal places
*and* the sealed weight addresses (`{'mlp.w1': 2968640, 'mlp.w3': 3664960, 'mlp.w2': 4361280}`).
That is the strongest statement available here: nothing in these seven cases depends on residual L1
or on allocator history, which is exactly the failure signature both 2026-08-25 root causes had.

No `tt-smi -glx_reset` was needed between any of the three runs, and all 32 devices closed normally
each time.

### 4.4 The transition matrix actually executed, with PCC at every step

`_TRANSITION_MATRIX` is `("decode","prefill","decode","prefill","decode") * 2 + ("prefill","prefill")`
— the plan's five-step cycle run twice so the seam between the cycles is itself a `decode → decode`,
plus a tail that supplies the `prefill → prefill` the plan lists separately. Twelve steps, twelve real
MLP2D invocations, twelve PCC assertions, identical in all three runs:

| Step | Transition | PCC |
| --- | --- | --- |
| 00 | `cold → decode` | 0.9982190 |
| 01 | `decode → prefill` | 0.9993101 |
| 02 | `prefill → decode` | 0.9982190 |
| 03 | `decode → prefill` | 0.9993101 |
| 04 | `prefill → decode` | 0.9982190 |
| 05 | **`decode → decode`** | 0.9982190 |
| 06 | `decode → prefill` | 0.9993101 |
| 07 | `prefill → decode` | 0.9982190 |
| 08 | `decode → prefill` | 0.9993101 |
| 09 | `prefill → decode` | 0.9982190 |
| 10 | `decode → prefill` | 0.9993101 |
| 11 | **`prefill → prefill`** | 0.9993101 |

Threshold is 0.99 (`MLP_PCC_THRESHOLD`, `_mlp_2d_galaxy.py:37`) and was not touched. Every decode step
returns exactly 0.9982190 and every prefill step exactly 0.9993101 — the transition is numerically
inert, which is the result you want: activating a mode restores that mode's context bit-for-bit rather
than leaving it perturbed by the mode it came from.

At each step the test additionally asserts, through `_SubdeviceRecorder`, that the device really
loaded that mode's `sub_device_manager_id`, really set that mode's `stall_group`, that
`prefetcher.active_mode` matches, and that the DRAM prefetch producer exists **iff** the mode is
decode (`prefill` must have stopped it).

### 4.5 The other cases' device numbers

| Case | Invocations and PCC |
| --- | --- |
| 3, rollback | `pre-failure prefill` 0.9993101 → injected `RuntimeError` at `_dram_prefetch_start` during `activate("decode")` → rolled back to prefill → `post-failure prefill` 0.9993101 → `post-failure decode` 0.9982190 |
| 4, cleanup from decode | `first prefetcher decode` 0.9982190; then cleanup ×2; then `second prefetcher decode` 0.9982190 and `second prefetcher decode -> other mode` 0.9993101 |
| 5, cleanup from prefill | `first prefetcher prefill` 0.9993101; then cleanup ×2; then `second prefetcher prefill` 0.9993101 and `second prefetcher prefill -> other mode` 0.9982190 |

The rollback case is the one worth dwelling on: the proof that the rollback worked is not that
`active_mode` reads `"prefill"` afterwards — it is that the **next prefill invocation still hits
0.9993101**, the same value it hit before the injected failure. A rollback that re-loaded the manager
but left the ring or the stall group inconsistent would produce a tensor and fail that assertion.

Case 1 additionally reads the packed address tensor back off the senders with
`ConcatMeshToTensor(dim=0)` and asserts all `32 × 12 = 384` rows equal
`[2968640, 3664960, 4361280]` — i.e. every sender core on every one of the 32 devices holds the same
address vector the seal published, and each entry is the registered tensor's actual
`buffer_address()`. Nothing before this proved that on silicon.

## 5. Finding F1 — `cleanup()` cannot release the global circular buffer

### 5.1 The mechanism

`Prefetcher2D.seal()` creates the global circular buffer
([`prefetcher_2d.py:291`](../models/common/modules/prefetcher/prefetcher_2d.py#L291)) and publishes
the **live handle** inside the immutable decode context
([`:329`](../models/common/modules/prefetcher/prefetcher_2d.py#L329)). `Prefetcher2DContext` carries it
as a plain field (`global_cb`, [`:113`](../models/common/modules/prefetcher/prefetcher_2d.py#L113)), and
every consumer keeps the whole context for its lifetime — `MLP2D` as `decode_prefetch_context`,
`Attention2D` likewise, both reading `context.global_cb` at each `ttnn.linear`.

`cleanup()` does everything it can: it stops the DRAM prefetch producer, resets the stall group,
clears the loaded manager, deallocates the packed weight-address tensor, removes both subdevice
managers, drops its own `_global_cb` reference
([`:440`](../models/common/modules/prefetcher/prefetcher_2d.py#L440)) and clears `_contexts`
([`:447`](../models/common/modules/prefetcher/prefetcher_2d.py#L447)). What it **cannot** do is free the
CB: ttnn exposes no free for a `global_circular_buffer`, so its ~55 MB of L1 is reclaimed by RAII when
the last Python handle dies. `ttnn.deallocate` takes a `Tensor`, so the CB is deliberately not in
`cleanup()`'s deallocation loop — that omission is correct, not an oversight.

The consequence is that after `cleanup()` the owner **truthfully reports that it owns nothing**
(`prefetcher.owned_resources == ()`, asserted by cases 4 and 5) while up to 55 MB of L1 is still
resident, pinned by a consumer. The owner's self-report is accurate about *ownership* and silent about
*residency*. That gap is the finding.

### 5.2 The evidence, in order

1. **It fails the real suite.** `dev03` — the second owner's `seal()` →
   `create_global_circular_buffer` dies with
   `Out of Memory: Not enough space to allocate 55444480 B L1 buffer across 70 banks` (§4.2).
2. **`probe02` isolates it to the CB handle specifically**, with no MLP2D, no Galaxy CCL and no
   kernels in play — ten steps, each printing its own verdict
   ([`logs/probe02_global_cb_lifetime.log:136-155`](logs/probe02_global_cb_lifetime.log)):

   | Step | What is held across `cleanup()` | Second `seal()` |
   | --- | --- | --- |
   | A | the whole decode context | **FAILS** — L1 OOM |
   | B | nothing (context dropped, `gc.collect()`) | ok |
   | C | nothing — no context handle was ever taken | ok |
   | D | **only** `weight_address_metadata` | ok (and `metadata.is_allocated()` is already `False`) |
   | E | **only** `context.global_cb` | **FAILS** — same L1 OOM |
   | E′ | the same, after dropping the CB | ok |

   Step C is the important one: with no consumer handle in existence, the owner's own bookkeeping is
   *sufficient*. Step D rules out the address metadata. Step E convicts the CB handle alone.
3. **The test-side workaround.** The leak detector drops its `MLP2D` reference and `gc.collect()`s
   before constructing the second owner, with the reasoning recorded in a comment at
   [`test_prefetcher_2d_wh_galaxy.py:559-569`](../models/common/tests/modules/prefetcher/test_prefetcher_2d_wh_galaxy.py#L559-L569).
   That is what turns `dev03`'s failure into `dev07`/`dev08`/`dev09`'s pass.

Note the size arithmetic is consistent throughout: `GALAXY_PREFETCH_GLOBAL_CB_SIZE = 728 * 1088 =
792,064 B` per bank × 70 banks = `55,444,480 B`, and the failure message reports exactly
`792064 B` per bank against a `1393472 B` bank size with `~792 KB` already allocated. One live CB
leaves no room for a second.

### 5.3 The ruling — this is a real contract, and the contract is currently misdocumented

**Is "cleanup does not release the global CB; consumers must drop their contexts first" an intended
part of the ownership contract?** Partly, and that is the problem.

It is *unavoidable* today: given that ttnn offers no free, an owner that publishes the handle cannot
also guarantee its release. So the ordering requirement is real and will not go away by wishing.

But it is *not* what the repo currently claims. [`modules/README.md:212`](../models/common/modules/README.md#L212)
says `Prefetcher2D` is "the model-owned resource root for subdevice managers, **the global circular
buffer**, and sealed weight-address registration", and that "the executor … owns deterministic
cleanup". On silicon, cleanup of the global CB is *not* deterministic from the owner's side — it is
contingent on consumer lifetimes the owner cannot see. That sentence overstates what `cleanup()`
delivers, and it is the sentence a Milestone C executor author would rely on.

**Recommendation, in order of preference:**

- **(a) Adopt and document the ordering contract now. This is what I recommend for Milestone A**, and
  it needs no code change. Add to the `Prefetcher2D` class docstring and to the README's
  prefetcher-ownership paragraph, in substance:

  > `cleanup()` releases everything `Prefetcher2D` can release: it stops the DRAM prefetch producer,
  > resets the stall group, clears and removes both subdevice managers, and deallocates the packed
  > weight-address tensor. It **cannot** release the global circular buffer — ttnn exposes no free for
  > one, so the CB's L1 is reclaimed by RAII when the last `global_circular_buffer` handle dies, and
  > every consumer handed a `Prefetcher2DContext` holds one. A consumer that outlives `cleanup()`
  > therefore keeps the CB's L1 (55 MB on the qualified WH Galaxy configuration) pinned, and the next
  > owner's `seal()` fails with an L1 out-of-memory error that names no prefetcher. **Tear consumers
  > down before, or together with, the owner.**

- **(c) Fix the design in Milestone B/C: stop publishing the raw handle.** If `Prefetcher2DContext`
  exposed `global_cb` as a property that fetches from the owner (and raises after cleanup, the way
  `context()` already does via `_ensure_open`), then the owner dropping `_global_cb` would be
  sufficient regardless of who holds a context, because nobody else would hold the handle. This makes
  the contract *enforceable* instead of documented. It is source-compatible with both current
  consumers, which read `context.global_cb` per call through `_prefetch_kwargs`; the residual hazard
  is a future consumer that caches the value into a long-lived field, which is exactly what a lint or
  a host contract test should catch. This is a design change with two module consumers to re-qualify,
  so it does not belong in a gap-closing job.

- **(b) Rejected: make `cleanup()` fail loudly by introspecting references.** Checking
  `sys.getrefcount`/`gc.get_referrers` on the CB and raising is fragile (temporaries, exception
  tracebacks and the `attempt()` closure all perturb the count), it makes the one method that must
  always succeed throw, and it inverts the intended teardown order — cleanup would have to run *after*
  every consumer, which is the opposite of a resource owner's job. It converts a latent failure into
  a flaky one.

**Should the host suite pin the chosen behaviour?** Yes, and here is the test I would add — I did
**not** add it, because the handoff scopes F1 as analysis-and-recommend and explicitly says to leave
the module alone unless the analysis is conclusive on the fix, which it is not:

> `test_cleanup_drops_the_owners_global_cb_reference_but_not_a_consumers` — seal, take
> `context("decode")`, call `cleanup()`, then assert both halves of the fact: `owned_resources == ()`
> **and** the consumer's context still holds a truthy `global_cb`. On the `FakeMesh` the CB is a plain
> tuple, so this is expressible at host speed. It costs nothing and it means the next person meets
> shared CB ownership in a unit test rather than as a 55 MB L1 OOM on silicon.

Per the handoff's prohibition, `prefetcher_2d.py` was **not** modified. No module file was.

## 6. Test 5 outcome — Attention2D decode with an active prefetch producer

**Result: terminal FAILED, and it is an incompatibility by construction rather than a defect. It
belongs in Milestone B.** The case has now been executed twice, on two separate days, with the same
deterministic result: it aborts in ~40 s with

```
TT_FATAL @ tt_metal/distributed/fd_mesh_command_queue.cpp:388: sub_device_ids.size() == 1
Programs must be executed on a single sub-device
```

Not a hang, not a numerical failure, not intermittent.

| Log | Date | What it adds |
| --- | --- | --- |
| [`dev06_attention_with_prefetch_isolated.log`](logs/dev06_attention_with_prefetch_isolated.log) | 08-26, session 2 | First execution. The `TT_FATAL` at `:135`. The pytest traceback was lost with the process, because the teardown stall meant the terminal summary never printed |
| [`dev06b_gdb_teardown_backtrace.log`](logs/dev06b_gdb_teardown_backtrace.log) | 08-26, session 2 | External gdb backtrace of the teardown stall |
| [`dev11_attention_with_prefetch_traceback.log`](logs/dev11_attention_with_prefetch_traceback.log) | 08-26, session 3 | Re-run with a diagnostic plugin that flushes the failure repr at the end of the *call* phase, so the traceback survives the teardown stall. **This is what pins the failing call.** `exit=124`; the log contains no pytest terminal summary, because the session never finished |
| [`host05_case8_subdevice_overlap.log`](logs/host05_case8_subdevice_overlap.log) | 08-26, session 3 | Device-free proof of the core overlap |
| [`host06_grid_derivation.log`](logs/host06_grid_derivation.log) | 08-26, session 3 | Device-free proof that the compute grid is `7 × 10` and that the CB's receiver sets tile exactly the worker subdevice |

### 6.1 What conflicts with what, exactly

`dev11` pins the raising call: `attention_2d.py:851`, the **decode QKV projection**
`ttnn.linear(x, self.wqkv, …, program_config=cfg.decode_program_config, …)`, on
`[1,1,32,2048] × [2048,1280]` with a DRAM-interleaved output. It is the first program
`decode_forward` enqueues. Everything before it worked: `resources.activate("decode")` loaded the
partition and started the DRAM prefetch producer — itself a program, on the sender subdevice — and the
test's `assert resources.prefetcher.prefetch_result is not None` had already passed. So the producer
side of the partition is fine; it is the consumer's grid that is wrong.

The two geometries:

- **The prefetch decode partition** (`galaxy_prefetch_decode_mode_plan`,
  [`_wh_galaxy_hardware.py:344`](../models/common/tests/modules/_wh_galaxy_hardware.py#L344)) splits the
  `7 × 10` grid into `SubDeviceId(0)` = the 12 sender cores on `x ∈ {0,4}`, and `SubDeviceId(1)` =
  workers `CoreRange((1,0),(3,9)) ∪ CoreRange((5,0),(6,9))`, i.e. `x ∈ {1,2,3} ∪ {5,6}`, 50 cores.
  Eight cores — the dummy senders at `(0,1..3)`, `(0,6..8)`, `(4,3)`, `(4,8)` — belong to **neither**
  subdevice; they exist only so the global CB covers every worker core.
- **Attention2D's qualified decode QKV/WO projections** use
  `_matmul_program(_BATCH_SIZE=32, …)` ([`test_attention_2d_wh_galaxy.py:124`](../models/common/tests/modules/attention/test_attention_2d_wh_galaxy.py#L124)),
  where `grid_x` is hardcoded `7` and `grid_y = min(4, ceil(32/32)) = 1`. So
  `compute_with_storage_grid_size=(7,1)`, which ttnn normalizes
  ([`matmul_program_config_types.hpp:106`](../ttnn/cpp/ttnn/operations/matmul/device/config/matmul_program_config_types.hpp#L106))
  to `allowed_worker_cores = CoreRange((0,0),(6,0))`.

Those seven cores decompose as **2 sender cores** `(0,0)` and `(4,0)`, and **5 worker cores** `(1,0)`,
`(2,0)`, `(3,0)`, `(5,0)`, `(6,0)`. Zero cores outside both. `host05` computes exactly that.

The rejection is mechanical.
[`program.cpp:2166 determine_sub_device_ids`](../tt_metal/impl/program/program.cpp#L2166) intersects
each kernel group's core ranges with every subdevice's worker cores and collects **every** subdevice
that intersects; `enqueue_mesh_workload` then `TT_FATAL`s unless that set has exactly one element.
The `(7,1)` kernel group intersects both, so the set has two, and the program is refused before it
runs.

### 6.2 Can it be made to work? No — not without choosing a new attention decode geometry

Four things close this off, in increasing order of decisiveness.

1. **You cannot pass your way out of it.** `ttnn.linear` does accept an optional `sub_device_id`, and
   `Attention2D` already forwards one when a prefetch context is supplied
   (`_prefetch_kwargs`, [`attention_2d.py:337`](../models/common/modules/attention/attention_2d.py#L337)).
   But `determine_sub_device_ids` derives the set from **kernel placement**, never from that argument.
   The subdevice set is a property of where the kernels land, full stop.
2. **The worker subdevice is not expressible as a matmul grid.**
   `MatmulMultiCoreReuseMultiCastProgramConfig` does expose `allowed_worker_cores`, which "overrides
   `compute_with_storage_grid_size` for determining the active compute grid" — but
   [`matmul_program_config.cpp:1075-1077`](../ttnn/cpp/ttnn/operations/matmul/device/config/matmul_program_config.cpp#L1075-L1077)
   `TT_FATAL`s unless it is a **dense rectangle**: `awc.num_cores() == bounding_box().size()`. The
   worker subdevice is 50 cores in a `6 × 10 = 60` bounding box. It is not a dense rectangle, so it
   cannot be handed to a multicast matmul as a whole.
3. **Every dense rectangle anchored at the origin includes a sender column.** `x = 0` is a sender
   column, so no value of `compute_with_storage_grid_size` — `(7,1)`, `(3,1)`, anything — avoids the
   senders. Offset rectangles wholly inside the worker subdevice *do* exist:
   `CoreRange((1,0),(3,0))` = 3 cores, `((1,0),(3,2))` = 9, `((5,0),(6,3))` = 8 (all verified in
   `host05`). But none of them has 7 cores, so each changes `per_core_N = ceil(n_tiles / grid_x)`,
   and the matmul's **output shard grid moves with them** — the factory derives `start_core` from
   `allowed_worker_cores.bounding_box().start_coord`
   ([`matmul_device_operation.cpp:2541`](../ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp#L2541))
   rather than trusting a user-supplied output grid. Downstream, the fused QKV collective is qualified
   against `decode_all_reduce["qkv_input_memcfg"]`. So this is not a grid swap; it is a re-tiling of
   the qualified decode projections plus a re-derivation of the grids the next collective depends on,
   needing its own PCC baseline. I could not argue such a value is "the correct one", which the brief
   requires before changing a core grid.
4. **And it is the wrong target anyway.** In production, decode attention *with* a prefetcher does not
   use the DRAM-sharded multicast matmul at all — it uses the ring/`gather_in0` matmul that reads
   weights out of the global CB, exactly as MLP2D does. That form already exists in the attention
   suite: `_decode_ring_config` builds `qkv_program` and `wo_program` as
   `MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=(8,3), gather_in0=True,
   hop_cores=((3,6),), num_global_cb_receivers=2)` over 24 `ring_cores`. `host05` confirms **all 24
   ring cores and the hop core lie inside the worker subdevice** — the ring form is
   prefetch-partition-compatible by construction. But those two program configs are built and never
   passed to `Attention2D.from_config`; `_make_module` wires `decode_program_config=_matmul_program(…)`
   and `decode_prefetch_context=None`.

There is a further scope point worth stating plainly: **even if the grid conflict vanished, case 8 as
written would not qualify what production needs.** Because `_make_module` hardcodes
`decode_prefetch_context=None`, the attention matmuls never receive `global_cb`, so the producer
streams `attn.wqkv`/`attn.wo` into the CB and nothing reads them. The case tests *coexistence* —
attention's qualified geometry running while a producer occupies the sender subdevice — not
*consumption*. Consumption is entirely unqualified and is not reachable from the current attention
decode configuration.

**Recommendation: Milestone B.** Wiring attention decode onto the ring/global-CB matmul *is* choosing
a production grid, which is what Milestone B does. Doing it inside a Milestone A gap-closing job would
mean qualifying a brand-new attention decode geometry — new program configs, new output shard grids, a
new PCC baseline for both model specs — under the cover of a prefetcher test. The honest Milestone A
statement is: `Prefetcher2D`'s subdevice partition is qualified with the MLP2D consumer, and
Attention2D's *current* qualified decode geometry is structurally incompatible with it. That is a
grid-choice question, not a `Prefetcher2D` defect — nothing in the finding implicates the prefetcher.

Two prohibitions were live here and both were respected: `semaphore_cores` was **not** narrowed (the
partition already keeps `semaphore_cores` equal to the worker subdevice, which is the invariant the
2026-08-25 attention fix established), and no core grid was changed to make the failure disappear.

### 6.3 The teardown stall — worth its own line for whoever sequences Milestone B

**A `TT_FATAL` abort inside a multi-subdevice program leaves the mesh un-drainable.** After the abort
the process sits forever in `mesh_device` fixture teardown. `dev06b`'s gdb backtrace puts the main
thread at

```
pthread_cond_wait
FDMeshCommandQueue::wait_for_outstanding_reads
FDMeshCommandQueue::clear_expected_num_workers_completed
FDMeshCommandQueue::~FDMeshCommandQueue
MeshDeviceImpl::close_impl → MeshDevice::close()
```

so the refused workload leaves reads outstanding that the command-queue destructor waits on
unconditionally. Session 3 reproduced the same stall and let `timeout` do the killing rather than
signalling any PID by hand: `dev11` ran under `timeout --signal=TERM --kill-after=90 420` and ended
`exit=124` having produced **no pytest terminal summary at all** — the session never finished, which is
exactly why `dev06`'s traceback was lost and why an eager-longrepr plugin was needed to recover it. The
`pytest-timeout` setting in `pytest.ini` (`timeout: 300.0s`, `method: signal`) did not help either: a
SIGALRM cannot be serviced while the main thread is blocked in a C++ destructor with the GIL released.

A second operational note for whoever debugs the next teardown stall: **the in-process
`faulthandler.dump_traceback_later` trick does not survive a test failure.** pytest's built-in
faulthandler plugin cancels every pending dump in `pytest_exception_interact`
([`_pytest/faulthandler.py:114`](../python_env/lib/python3.10/site-packages/_pytest/faulthandler.py#L114),
`tryfirst`), so the very failure you want to trace past disarms the dumper — `dev11` was launched with
a repeating 150 s dumper and produced no dumps. That is not a contradiction of the 2026-08-25 attention
debugging, where the stall was *in the call phase* and no exception had been raised yet. For a stall
**after** a failure, attach gdb from outside, as `dev06b` did.

**Operational consequence: budget a kill plus a `tt-smi -glx_reset` after any run that aborts a
multi-subdevice program on this stack.** This is not specific to attention or to the prefetcher — any
`TT_FATAL` raised from `enqueue_mesh_workload` will do it. `session 2` did not reset afterwards, which
is why `session 3` began with [`reset01_before_session3.log`](logs/reset01_before_session3.log) before
touching the device at all.

## 7. Finding F2 — an undersized `global_cb_size` is silently accepted

Smaller than F1, but it changes what the suite can honestly claim. The original brief's Test 4 asks
for "`seal()` with an undersized `global_cb_size` is rejected (host: `:262`)". **There is no such
rejection, on host or on device.**

- The host test named there,
  `test_seal_derives_cb_size_and_rejects_undersized_configuration`
  ([`test_prefetcher_2d.py:262`](../models/common/tests/modules/prefetcher/test_prefetcher_2d.py#L262)),
  asserts *derivation* — that `global_cb_size=None` resolves to `2 × max(weight buffer size)` and that
  an explicit size is honoured verbatim. It asserts no rejection. The name overstates it.
- `Prefetcher2D.seal()` validates only `resolved_cb_size > 0`
  ([`prefetcher_2d.py:279`](../models/common/modules/prefetcher/prefetcher_2d.py#L279)). There is no
  comparison against the registered weights' sizes when a size is configured explicitly.
- On device, `probe01` drove exactly this and it was accepted:
  [`logs/probe01_seams.log:154-155`](logs/probe01_seams.log) —
  `PROBE5 undersized global_cb_size=1024 ACCEPTED (resolved=1024)` and the same at `4096`. The probe's
  registered weights are 4-tile `bfloat8_b` tensors (`1×1×32×128`), so `2 × max(buffer_size())` — the
  rule the module applies to itself when `global_cb_size is None` — is several times 4096. Both
  explicit sizes are below what the module would have derived for itself, and
  `create_global_circular_buffer` allocated the undersized CB without complaint. `global_cb_size` is a
  per-bank figure: the qualified `728 × 1088` shows up in the F1 failure message as "each bank needs to
  store 792064 B", so `1024` really is a 1 KB-per-bank CB.

That is why the committed suite does **not** contain an undersized-CB case, and why
`galaxy_prefetcher_config(…, global_cb_size=…)` — exposed by this job's refactor specifically to allow
one — currently has no caller outside `tttv2_gap2_scratch/test_probe.py`. Writing a test that asserts a
rejection that does not exist would be writing a failing test; writing one that asserts acceptance
would be pinning behaviour nobody has ruled on.

**Recommendation.** Decide whether an undersized global CB should be rejected at `seal()`. It is
cheap to check — the deriver already computes `2 × max(buffer_size())` — and the failure mode it
prevents is a decode ring silently streaming through a CB too small for its weights, which is the kind
of thing that shows up as corrupted activations rather than an error. If the answer is "reject", it
belongs with the F1 documentation change and needs one host test plus the device case this refactor
already made expressible. If the answer is "the configured size is authoritative by design", then the
host test at `:262` should be renamed, because its current name is the reason the brief asked for a
contract that is not there.

Two other observations from `probe01`, recorded so they are not rediscovered:

- **The foreign-mesh rejection does work on device.** `PROBE4` created a `(1,1)` submesh, put a tensor
  on it, and `register_weight` rejected it with
  `ValueError: registered weight belongs to a different mesh`. But the probe process then **aborted at
  interpreter shutdown** — `TT_THROW: MeshDevice cq ID 0 is in use by parent mesh ID 0 during close of
  mesh ID 1`, `std::runtime_error`, `exit=134`, core dumped — after the test itself had already
  reported `PASSED`. That is precisely why case 7 documents the foreign-mesh rejection as deliberately
  out of scope: on a reserved 32-device Galaxy the only available second mesh is a submesh of this one,
  and leaving one alive aborts teardown. The rejection is mesh-agnostic and is covered on host.
- 128 `matmul_multi_core_reuse_mcast_1d_optimized_helper: program_config.allowed_worker_cores not
  populated; auto-populating from compute_with_storage_grid_size … This will become a hard error in a
  future release` warnings appear in each whole-file run. These come from the MLP2D ring matmul, which
  reaches the factory through a path that bypasses `ttnn::prim::matmul()` and therefore never calls
  `normalize_program_config`. **This is pre-existing, not introduced here** — `dev01_mlp_regression.log`
  has 64 of them, and this suite simply runs twice as many invocations. It is benign today because
  `gather_in0` takes its cores from the input shard grid, but it is on a deprecation path and the
  prefetch-fed ring matmul is the only prefetch consumer qualified anywhere, so it is worth someone's
  attention before it becomes an error.

## 8. Regression gates

| Gate | Expected | Observed | Log |
| --- | --- | --- | --- |
| Prefetcher/Galaxy/MLP host suites | — | **`78 passed in 12.12s`** | [`host01_regression.log`](logs/host01_regression.log) |
| `pre-commit` on every touched file | clean | clean on re-run (first pass auto-fixed `isort`) | [`host02`](logs/host02_precommit.log), [`host03`](logs/host03_precommit_rerun.log) |
| MLP2D device suite | `4 passed` | **`4 passed in 127.80s`** | [`dev01_mlp_regression.log`](logs/dev01_mlp_regression.log) |
| RMSNorm2D device suite | `8 passed` | **`8 passed in 33.55s`** | [`dev02_rmsnorm_regression.log`](logs/dev02_rmsnorm_regression.log) |
| Attention2D device suite | `2 passed` | **`2 passed in 75.36s`** | [`dev10_attention_regression.log`](logs/dev10_attention_regression.log) |
| `pre-commit` on this session's files (work log, report, notes) | clean | `exit=0`, all hooks Passed/Skipped | [`host08`](logs/host08_precommit_worklog.log), [`host09`](logs/host09_precommit_final.log) |

The attention re-run was the one outstanding gate: `_wh_galaxy_hardware.py` was refactored and
`test_attention_2d_wh_galaxy.py` imports it. Both `llama-70b` and `qwen3-32b` pass, at 75.36 s against
the 74.90–76.15 s recorded in the 2026-08-25 checkpoint, so the shared-helper split is confirmed
behaviour-preserving for its third consumer as well as its first two.

## 9. Caveats, gaps and the scratch directory

### 9.1 What this evidence does not establish

- **The payload is MLP2D geometry only.** Every one of the seven passing cases drives an `MLP2D`
  through the ring. So the contexts are qualified *for that consumer shape*: 3 registered weights,
  the 24-core decode ring, `global_cb_size = 728 × 1088`. A different consumer with a different weight
  count, ring or CB size is not covered, and case 8 is the concrete demonstration that "a different
  consumer" is not a formality.
- **Attention2D consuming prefetched weights is unqualified**, and is not reachable from the current
  attention decode configuration at all (§6.2).
- **No trace/capture coverage.** The plan's transition requirement names "compilation, capture, replay,
  and cleanup". This suite covers compilation and cleanup on silicon; capture and replay of a
  transition are untested, and `Prefetcher2D` has no trace mode (`context("trace")` is rejected, which
  case 7 asserts).
- **Failure injection covers the seam the host suite models, not arbitrary faults.** Case 3 injects at
  `_dram_prefetch_start`, which is the seam `Prefetcher2D.__init__` exposes for exactly this. A failure
  inside `load_sub_device_manager` or part-way through `create_global_circular_buffer` is not
  reachable from Python and is not covered.
- **`(8,4)` only, one mesh shape, one architecture.** No reduced mesh was run, per the brief.
- **PCC is asserted, and now also quoted.** The threshold is 0.99 and was not touched; the observed
  values are 0.9982190 (decode) and 0.9993101 (prefill). Nothing here relaxed a tolerance.
- **F1's ordering requirement is worked around in the test, not fixed in the module.** The leak
  detector passes because it drops consumer handles first. A caller who does not know to do that still
  hits the 55 MB L1 OOM.
- **Two runs of case 8, not three.** It is terminal-failing, so the three-fresh-process rule does not
  apply; the two runs on two different days with identical `TT_FATAL` are what establish determinism.

### 9.2 `tttv2_gap2_scratch/` — diagnostic scratch, not a deliverable

Two probe files plus a `__pycache__`:

- `test_probe.py` — the seam probe: subdevice-call recording, sealed-address readback, `borrow_context`
  mismatch, duplicate and foreign-mesh registration, short seal, undersized CB, and cleanup's effect
  on weights and metadata. It produced **F2** and the foreign-mesh/teardown-abort result in §7.
- `test_probe2.py` — the ten-step global-CB lifetime experiment that isolated **F1** (§5.2).

**Recommendation: keep the directory as-is and treat it as diagnostic scratch.** It is not a
deliverable and should not be committed — it is outside `models/`, it has no assertions worth
regressing (both probes `print` and pass unconditionally), and `test_probe.py` aborts its own process
at interpreter shutdown — not by design, but as an unavoidable consequence of the submesh it creates to
exercise the foreign-mesh rejection (§7). Everything either probe established that matters is now in this report with the log line
that established it: F1 in §5.2, F2 and the foreign-mesh result in §7. Nothing was folded into the
suite, because the two things worth pinning are a host contract test for F1 (§5.3) and an
undersized-CB decision for F2 (§7), and both are recommendations rather than settled behaviour.

The other scratch artefacts of this job also stay out of the repo: `/tmp/gap2_run.sh`,
`/tmp/gap2_run_short.sh`, `/tmp/gap2_chain.sh` (the run wrappers) and `/tmp/gap2_diag/` (the two
diagnostic-only pytest plugins — `gap2_faultdump`, a repeating `faulthandler.dump_traceback_later`, and
`gap2_eagerreport`, which flushes a failure repr at the end of the call phase so it survives a hanging
teardown). They live outside the tree on purpose and are described here so the runs are reproducible.

## 10. Proposed status-page replacements — drafted, **not applied**

`MILESTONE_A_STATUS.md`, `modules/README.md` and `tttv2_2d_modules_plan.md` were **not edited**; they
are being rewritten wholesale as a separate task. The text below is for that task to take or leave.

### 10.1 `MILESTONE_A_STATUS.md:31` — `Galaxy CCL/resources`

Current:

> | Galaxy CCL/resources | Concrete CCL/resource/composition host contracts included in final gate | Repeated MLP/RMS paths and fused Attention axis-1 decode pass with clean teardown | Qualified for Milestone A; non-fused Attention decode is not required or qualified |

Proposed:

> | Galaxy CCL/resources | Concrete CCL/resource/composition host contracts included in final gate; prefetcher/galaxy/MLP host suites `78 passed` after the shared-helper split | Repeated MLP/RMS paths and fused Attention axis-1 decode pass with clean teardown; the resource owner's own `activate`/`synchronize`/`cleanup` lifecycle now qualified directly across a 12-step prefill↔decode matrix, three fresh processes (`7 passed` each) | Qualified for Milestone A. Non-fused Attention decode is not required or qualified. Attention decode on the *prefetch* subdevice partition is qualified as **incompatible**: its `(7,1)` QKV/WO matmul grid straddles the sender and worker subdevices, deferred to Milestone B grid selection |

### 10.2 `MILESTONE_A_STATUS.md:32` — `Prefetcher2D`

Current:

> | Prefetcher2D | Concrete composition regression: 29 passed | Repeated Llama/Qwen MLP decode consumes production-prefetched weights and tears down cleanly | Qualified for recorded Milestone A integration |

Proposed:

> | Prefetcher2D | Concrete composition regression: 29 passed | Own hardware suite, 7 cases, `7 passed` in each of three fresh processes with byte-identical output: sealed weight addresses read back off all 12 sender cores on all 32 devices; the full transition matrix (`decode→prefill`, `prefill→decode`, `decode→decode`, `prefill→prefill`) with PCC at all 12 steps; failed-transition rollback; cleanup from either active mode proven by a second owner sealing and computing on the same mesh in the same process | Lifecycle qualified for the MLP2D consumer shape. **One documented ownership limit:** `cleanup()` cannot free the global circular buffer (ttnn exposes no free); a consumer that outlives the owner keeps 55 MB of L1 pinned and the next owner's `seal()` fails with an L1 OOM. Consumers must be dropped before or with the owner |

### 10.3 `modules/README.md:212`

**The hardware-qualification caveat can be dropped, but the sentence before it must be weakened at
the same time** — the ownership claim it makes is the part this gap disproved. Current text:

> Galaxy collectives are injected from `models/common/models/galaxy`; 2D modules do not extend or
> specialize the 1D `TT_CCL` owner. The target ownership contract makes `Prefetcher2D` the model-owned
> resource root for subdevice managers, the global circular buffer, and sealed weight-address
> registration. Modules borrow immutable prefill/decode contexts, and the executor activates a context
> at operation boundaries and owns deterministic cleanup. **Integrated Prefetcher2D/Galaxy-resource
> ownership has host coverage but is not yet qualified on hardware.**

Proposed:

> Galaxy collectives are injected from `models/common/models/galaxy`; 2D modules do not extend or
> specialize the 1D `TT_CCL` owner. The target ownership contract makes `Prefetcher2D` the model-owned
> resource root for subdevice managers, the global circular buffer, and sealed weight-address
> registration. Modules borrow immutable prefill/decode contexts, and the executor activates a context
> at operation boundaries and owns cleanup. Integrated Prefetcher2D/Galaxy-resource ownership is
> qualified on hardware for the MLP2D consumer shape: the prefill↔decode transition matrix, the
> failed-transition rollback, and cleanup from either active mode all run on a `(8, 4)` 6U Galaxy with
> PCC asserted at every step. **Cleanup of the global circular buffer is not deterministic from the
> owner's side:** ttnn exposes no free for one, so its L1 is reclaimed when the last handle dies, and
> every module holding a `Prefetcher2DContext` holds one — tear consumers down before, or together
> with, the owner. Attention2D's current decode projection grid is incompatible with the prefetch
> subdevice partition; choosing a compatible one is Milestone B work.

## 11. Reproducing this

```sh
export TT_METAL_HOME=/proj_sw/user_dev/ctr-apbernal/tt-metal
export PYTHONPATH=$TT_METAL_HOME
source $TT_METAL_HOME/python_env/bin/activate
cd $TT_METAL_HOME

# the seven lifecycle cases; repeat in three fresh processes
timeout --signal=TERM --kill-after=180 2700 python -m pytest -v -rA --color=no -p no:cacheprovider \
  --deselect models/common/tests/modules/prefetcher/test_prefetcher_2d_wh_galaxy.py::test_prefetcher_2d_wh_galaxy_attention_decode_with_active_prefetch \
  models/common/tests/modules/prefetcher/test_prefetcher_2d_wh_galaxy.py > LOG 2>&1

# case 8: aborts in ~40 s, then wedges the mesh. Budget a kill and a tt-smi -glx_reset.
timeout --signal=TERM --kill-after=90 420 python -m pytest -v -rA --color=no -p no:cacheprovider \
  'models/common/tests/modules/prefetcher/test_prefetcher_2d_wh_galaxy.py::test_prefetcher_2d_wh_galaxy_attention_decode_with_active_prefetch' > LOG 2>&1
tt-smi -glx_reset
```

One pytest process on the device at a time; never pipe pytest, redirect with `> LOG 2>&1`.

## 12. Finish-condition checklist

| Requirement | State |
| --- | --- |
| All 8 cases have a terminal state with at least one log | yes — §4.1 |
| Case 8 passing, or a written diagnosis and a Milestone A-vs-B recommendation | diagnosis and recommendation, §6; recommended for **Milestone B** |
| The subdevice conflict documented | §6.1, plus the device-free proof in [`logs/host05_case8_subdevice_overlap.log`](logs/host05_case8_subdevice_overlap.log) |
| The post-abort teardown stall documented | §6.3 |
| Cases 1–7 green in three fresh processes | yes — 224.28 s / 225.96 s / 226.15 s, byte-identical output, §4.3 |
| Case 8 run separately so it does not poison the file runs | yes — deselected by node ID in all three file runs |
| Attention device suite still `2 passed` | yes — `2 passed in 75.36s`, §8 |
| F1 carries a written recommendation | yes — §5.3 |
| `REPORT.md` written | this file |
| Work-log checkpoint appended | yes — `## Hardware checkpoint: Prefetcher2D and Galaxy resource hardware qualification 2026-08-26` in `tttv2_2d_modules_work_log.md` |
| Status-page replacements drafted, not applied | yes — §10; `MILESTONE_A_STATUS.md`, `modules/README.md` and `tttv2_2d_modules_plan.md` are untouched |
| Scratch directory's status stated | yes — §9.2, keep as diagnostic scratch |
| Device left clean, `tt-smi -ls` showing 32 boards | yes — [`logs/host07_tt_smi_final.log`](logs/host07_tt_smi_final.log), ids 0–31 |

Prohibitions observed: no `models/common/modules/**/*_1d.py` touched; `prefetcher_2d.py` unmodified
(nothing under `models/common/modules/` was modified at all); `semaphore_cores` not narrowed; no PCC
threshold or tolerance relaxed; `MILESTONE_A_STATUS.md`, `modules/README.md` and
`tttv2_2d_modules_plan.md` not edited; no `git commit`/`push`/`checkout`/`stash`/`reset`; tt-metal not
rebuilt and the venv not recreated; neither the full `models/common/tests` suite nor any 1D hardware
matrix was run; and no result is claimed here that was not observed in a log in `logs/`.
