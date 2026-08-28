
## 2026-08-28 — `mb-coverage`, attempts 2 and 3 (device, step 7)

The first work-log entry this job has had: attempt 1 ran on a dead mesh, attempt 2
was killed by a reservation expiry, and neither reached this file.

Mesh healthy throughout — 32/32 in `/sys/class/tenstorrent` and in
`/dev/tenstorrent`, `test_partition_wh_galaxy.py` 5 passed in 13.66 s. `HF_HOME`
inherited **empty**; every harness script exports
`/localdev/ctr-apbernal/hf_data`. Attempt 3 ran as two agent invocations in one
driver run; the detached serial device queue survived the boundary and the mesh
was never idle.

**The exit gate, re-measured at this tree.** Eight of the nine lines pass. The
ninth — "existing 1D model contract and demo-contract host tests green" — fails
5 of 301, at the same five node ids attempt 1 recorded, and none of the five
packages appears anywhere in `bc6ad03bfc2..HEAD`, so Milestone B cannot be their
cause. No expectation was edited; `git diff --name-only bc6ad03bfc2..HEAD --
models/common/tests/models/ | grep -v galaxy` is empty.

```text
Llama teacher-forced 512/511    top-1 501/511 = 98.04%   top-5 511/511 = 100.00%
Qwen  teacher-forced 512        top-1 498/511 = 97.46%   top-5 511/511 = 100.00%
batch-32 demos, both models     32 slots, each its own prompt, no contamination
4K / 32K / 128K smokes          pass, both models, one run per geometry
prefix-cached vs uncached       pass, both models
_1d.py / llm_runtime changes    0 and 0, of 384 changed paths
llm_runtime host suite          1032 passed, 1 skipped   (never run before tonight)
```

`git diff --name-only 718997518ab..HEAD -- models/` returns **one** file, and it
is a step-7 *test* file that `test_full_model_wh_galaxy.py` and `demo.py` do not
import — so the gate logs are measurements of byte-identical source, not quotes
of an older tree. Every row in the report says which commit produced its number.

**Three new defects, all in shared Galaxy code, and they are stacked.**

* **D-C5** — `GalaxyColumnUserSelector.__call__` is a bare `ttnn.matmul` needing
  an INTERLEAVED input B; the *shared* recipe makes both models' decode logits
  WIDTH_SHARDED. Measured for Qwen and for Llama, same frame, same assertion.
* **D-C8** — with D-C5 removed at the call site, the same line fails
  `Kernel group cores do not match sub device cores`: the matmul builds its
  program over cores outside the loaded decode sub-device. Three fresh processes,
  byte-identical. **D-C5's one-line fix is necessary and not sufficient.**
  Together these block the whole of the brief's area 4, for both models.
* **D-C7** — the L1 a *closed, dereferenced, garbage-collected* model held is not
  returned: the second model in one process cannot create its global circular
  buffer, with 923776 of 1393472 bytes per bank still allocated. Measured on
  **Qwen**, the model that does *not* hit the L1 address clash. So L1 has two
  signatures and only the clash could yield to the teardown ordering the plan
  suggests.

**Area 1's headline claim turned out to be measurable after all.** It was recorded
as not expressible twice — D-C4 (the adaptor substitutes a default pool for
`None`, so there is no contiguous path) and D-C7 (a process gets one model, so
the two-pool substitute cannot run either). Splitting the recording from the
comparison answers both without moving the claim or the threshold: one device
process per pool records its 32-slot prefill and decode logits, one host case
compares them. Qwen: **all 32 slots agree at PCC ≥ 0.99 for prefill and decode**.
The guard was exercised — with a recording absent the comparison fails rather
than skips.

Full account: `tttv2_milestone_b_evidence/coverage/REPORT.md` §A3, run-by-run
index in `RESULTS_A3.md`, machine-extracted verdicts in `VERDICTS_A3.txt`.
Handoff at `tttv2_milestone_b_briefs/job3_completion_handoff_attempt3.md`.
