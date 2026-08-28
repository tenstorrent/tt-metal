
## How attempt 2 ran, and what "recorded" means here

One pytest process on the mesh at a time, never piped, driven by
`cov_seq2.sh` over a manifest; each cycle reaps only the PID it started, refuses
to signal anything whose `comm` is not python, and runs `tt-smi -glx_reset`
after any non-clean exit. Logs are `logs2/a2_*.log`, one per cycle, never
overwritten. `RESULTS_A2.md` is the run-by-run index, written as each cycle
finished.

**The cost that shaped the night.** Every test in these files builds its own
model, and a Llama 80-layer build from the *warm* device weight cache is ~5.5
minutes; a *cold* recipe is far worse — `a2_01` spent 26 minutes staging 723
weights because the 512-token prefill recipe had never been resolved at this
commit. So a 17-node-id file is a three-hour run, and the house rule "three runs
in fresh processes before any device claim" cannot be applied to all 36 step-7
device cases in one night. What attempt 2 did instead, and states per row:

* the **exit-gate** lines and the **headline** step-7 mechanisms get three fresh
  processes;
* the remaining step-7 cases get **one** process, and are recorded as
  *observed, not qualified*. A single pass is bringup, and this project's own
  history says a case that passes once has proved nothing.

Nothing was recorded as evidence at a run count it did not get. Where a row says
`1 run` that is a statement about how much you may lean on it.

### One node id per process, and the 55 minutes it took to learn that

Attempt 2 began by running whole files in one process, on the reasoning that a
mesh open costs 25 s and 8 node ids in one process saves seven of them. That is
wrong on this stack, and expensively so — see **D-C3**: the device weight cache
fingerprint contains `MeshDevice.id()`, which increments per test, so test 2 of a
file re-stages all 965 weight tensors (138 GB, 26 min) and test 3 does it again.
The first cycle was stopped at 00:18 for that reason, its two completed tests
kept, and everything after it re-queued **one node id per process**. In that
shape every run is 100% cache hits.

The queue runner (`cov_queue.sh`) also grew a disk guard when this was found: it
prunes only the `.tensorbin` files this job wrote, and halts rather than
continue, if `/proj_sw` falls below 300 GB / 150 GB free.
