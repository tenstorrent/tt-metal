# Session 3 running notes

Working notes kept during the run so a mid-flight kill would not lose the trail. The deliverable is
[`REPORT.md`](REPORT.md); read that. This file only records the session's own mechanics and the two
places where it had to correct its inherited handoff.

Host `wh-glx6u-05-special-ctr-apbernal-for-reservation-117439`, 2026-08-26, 08:23–08:53 UTC.
Session 3 completed the job; nothing is left open that it could have closed.

## What session 3 ran, in order

| Step | Log | Result |
| --- | --- | --- |
| Reset first — session 2 hard-killed a process wedged in `MeshDevice::close()` and never reset | `reset01_before_session3.log` | `Re-initialized 32 boards`, `exit=0` |
| Confirm what collects | `host04_collect_only.log` | `8 tests collected`, `exit=0` |
| Cases 1–7, repeat 01 | `dev07_prefetcher_cases1to7_run01.log` | `7 passed, 1 deselected in 224.28s` |
| Cases 1–7, repeat 02 | `dev08_prefetcher_cases1to7_run02.log` | `7 passed, 1 deselected in 225.96s` |
| Device-free proof of the case-8 core overlap | `host05_case8_subdevice_overlap.log` | the `(7,1)` grid holds 2 sender + 5 worker cores |
| Device-free derivation of the `7×10` grid | `host06_grid_derivation.log` | CB receiver sets tile the worker subdevice exactly |
| Cases 1–7, repeat 03 | `dev09_prefetcher_cases1to7_run03.log` | `7 passed, 1 deselected in 226.15s` |
| Attention device suite (the outstanding regression gate) | `dev10_attention_regression.log` | `2 passed in 75.36s` |
| Case 8 isolated, with an eager-longrepr plugin | `dev11_attention_with_prefetch_traceback.log` | `TT_FATAL` reproduced; traceback captured; `exit=124` |
| Reset after the case-8 abort | `reset02_after_case8.log` | `Re-initialized 32 boards`, `exit=0` |
| Final device state | `host07_tt_smi_final.log` | 32 boards, ids 0–31 |
| `pre-commit` on the files this session changed | `host08_precommit_worklog.log`, `host09_precommit_final.log` | `exit=0` both times |

All device runs were strictly sequential — each waited for the previous log's `exit=` line and for
every `python` process to be gone. No PID was ever signalled by hand: `timeout` did the one kill that
was needed, on case 8.

## Two corrections to the inherited handoff

1. **The `dev03` and `dev04` failures were not the same failure.** The handoff attributes both to the
   agent's own `UnboundLocalError`. That is true of `dev04` only. `dev03`'s two failures are **finding
   F1 itself** — the second owner's `seal()` → `create_global_circular_buffer` dying with
   `Out of Memory: Not enough space to allocate 55444480 B L1 buffer across 70 banks`, frames landing
   in `_wh_galaxy_hardware.py:297 → prefetcher.seal()` (`dev03_prefetcher_run01.log:880-935`). So F1
   was first seen as a genuine failure of the real suite, and `dev04`'s bug was introduced *by the fix
   for it*. `REPORT.md` §4.2 and §5.2.
2. **The in-process `faulthandler.dump_traceback_later` plugin cannot trace past a test failure.**
   pytest's built-in faulthandler plugin cancels every pending dump in `pytest_exception_interact`
   (`_pytest/faulthandler.py:114`, `tryfirst`). `dev11` was launched with a repeating 150 s dumper and
   produced none. The 2026-08-25 attention debugging worked because that stall was in the call phase,
   before any exception. For a stall *after* a failure, attach gdb from outside — and to keep the
   traceback itself, flush `report.longrepr` at the end of the call phase, which is what `dev11` did
   and what made §6.1 possible. `REPORT.md` §6.3.

## Scratch kept outside the tree, described for reproducibility

`/tmp/gap2_run.sh` and `/tmp/gap2_run_short.sh` (run wrappers; the short one bounds case 8 at 420 s
because its teardown never returns), `/tmp/gap2_chain.sh` (the sequential driver for the last three
device steps), and `/tmp/gap2_diag/` with two diagnostic-only pytest plugins: `gap2_faultdump`
(repeating `faulthandler.dump_traceback_later`) and `gap2_eagerreport` (flushes a failure repr at the
end of the call phase). None of these belong in the repo. `tttv2_gap2_scratch/`'s status is ruled on in
`REPORT.md` §9.2.
