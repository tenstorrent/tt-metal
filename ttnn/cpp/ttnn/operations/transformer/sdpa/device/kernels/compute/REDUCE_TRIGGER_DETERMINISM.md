# SDPA streaming `reduce_trigger`: determinism findings

Notes on the non-determinism behind issue
[#47911](https://github.com/tenstorrent/tt-metal/issues/47911)
(`ring_mla` / ring-joint SDPA on Blackhole), and why `can_reduce_trigger` is
disabled in `sdpa_ring_v2` in `compute_streaming.hpp`.

## Symptom

`test_ring_mla_determinism[ring_mla-mla_100k-q160-k320]` runs the op 10× with
identical inputs and a reused `persistent_output_buffer_kv`, and asserts the
outputs are **bit-exact** across iterations. It fails with iteration 1 differing
from iteration 0 by a small amount (`max diff ~0.02–0.06`).

## Reproduction

Two ways: **(A)** LLK asserts (below), the original CI repro; **(B)** a tactical NOP
delay that reproduces the ND on a *plain* build with no asserts — see
"Tactical NOP-delay repro" below, the recommended way to iterate.

### A — LLK asserts

```bash
TT_METAL_LLK_ASSERTS=1 scripts/run_safe_pytest.sh \
  tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_mla_determinism
```

- The failing CI job is the **nightly debug run "with LLK asserts"**
  (`TT_METAL_LLK_ASSERTS=1`; kernels build at runtime so no host rebuild needed).
- With LLK asserts it fails at iteration 1 within ~11 s, reliably.
- **A single run is enough to reproduce** — no need to loop. The test itself runs
  the op 10× internally and asserts bit-exactness, so one invocation
  (~11 s test time, ~24 s wall incl. device setup/reset) deterministically
  surfaces the failure. Observed 11/11 runs fail at iteration 1 (max diff varied
  ~0.02–0.14 run-to-run — the magnitude is nondeterministic, but the failure
  itself is reliable).
- **Without LLK asserts the behaviour was not observed** (a plain release build
  passed 200 iterations; a 1500-iteration run only hit the pytest timeout, never
  a determinism failure). LLK asserts insert instruction-level checks that shift
  the math/pack/unpack pipeline timing; QB2 hardware timing reaches the same
  window on its own.

## What `reduce_trigger` does

`reduce_trigger` (gated by `can_reduce_trigger`) is a utilization optimization
for the QK row-max reduce. The unpack MOP is split into two halves with a
PACK→UNPACK hardware-semaphore handshake (`semaphore::FPU_SFPU`):

- **Packer** (`blocked_matmul_and_pack`, `trigger_reduce=true`): after packing
  the **last** QK subblock, posts `FPU_SFPU`.
- **Unpacker** (`_llk_unpack_AB_reduce_block_max_row_runtime_`,
  `respect_trigger=true`): runs the **first-half** MOP immediately, then
  `t6_semaphore_wait_on_zero(FPU_SFPU)`, then runs the **second-half** MOP.
- Because the split is supposed to cover synchronization, `reduce_c_row_group`
  **skips** the `cb_wait_front` producer barrier when `respect_trigger` is true.

## The problem

The `FPU_SFPU` handshake only sits **between** the two MOP halves, so it can only
guard the **second half**. The **first half** is read by the unpacker with **no
producer→consumer synchronization at all** — the code implicitly relies on the
packer always running ahead of the unpacker.

That assumption holds for most pipeline timings (so release builds pass), but it
is not guaranteed. Under adverse pack/unpack timing (LLK asserts, or QB2's
timing) the unpacker reads the first half of `cb_qkt_im` before the packer has
written it to L1 → stale tiles → corrupted row-max → small, non-deterministic
softmax/output differences.

Note this can *only* be a compute-side issue: the inputs are identical every
iteration, so the ring all-gather reproduces identical bytes — a data-movement /
semaphore / persistent-buffer race could only ever produce identical reads.

## Findings (controlled experiments)

All under `TT_METAL_LLK_ASSERTS=1`, everything else equal:

| Change | Result |
|---|---|
| baseline (`reduce_trigger` on) | **FAIL** at iter 1, ~11 s, every run |
| post `t6_semaphore_post<NONE>` → `<STALL_PACK>` (guards 2nd half harder) | **FAIL** — not the post/second half |
| un-skip `cb_wait_front` in `reduce_c_row_group` (keep split MOP on) | **PASS** (30 iters) |
| disable `can_reduce_trigger` | **PASS** (100 iters) |
| baseline, no LLK asserts | PASS (200 iters) |

Post-fix: ring_mla determinism 100 iters PASS; `test_ring_mla_accuracy`
PCC 0.9996 / RMSE 0.0082 (unchanged); `test_ring_joint_attention_sdpa_determinism`
7/7 configs PASS — all under LLK asserts.

## Tactical NOP-delay repro (no LLK asserts) — the decisive proof

To settle "is this a real race or just an LLK-asserts timing artifact?", reproduce it
on a **plain build** by injecting a tactical delay with NOP instructions on the
compute threads (RISC-V `nop` and Tensix `TTI_NOP`), via a loop-based per-thread
inserter placed at the QK packer (right before the `cb_qkt_im` pack in
`blocked_matmul_and_pack`, gated to `transpose==true`). Use a **loop**, not unrolled
`.rept` — unrolling thousands of nops overflows the kernel config buffer ("Program
size too large"). No DPRINT in the inserter (DPRINT in the reduce hot path deadlocks
the ring-fabric op).

Delaying the **packer** (producer) widens the window for the unpacker's unguarded
first-half read. Results (`TT_METAL_LLK_ASSERTS` unset):

| Config | Result |
|---|---|
| buggy, **pack** TTI nops 1024 / 8192 / 65536 | **FAIL** — ND, max diff 2.75 → 3.97 |
| buggy, **pack** RISC nops 4096 | **FAIL** — ND, max diff 3.17 |
| buggy, **unpack** nops 8192 (consumer side, wrong direction) | PASS (suppresses) |
| **`can_reduce_trigger=false`** (fix) + pack nops 1024 | PASS |
| buggy, no nops (baseline, no asserts) | PASS |

This is conclusive:
- The ND reproduces on a plain build with *no asserts* — asserts were only ever one
  way to perturb timing into the window; a pack-thread delay is another.
- **Direction matches the root cause exactly:** only delaying the *producer* (pack)
  triggers it; delaying the *consumer* (unpack) never does. That is precisely a
  missing producer→consumer ordering.
- **The fix is not "just moving timing":** under a pack delay that *deterministically*
  breaks the buggy code, the fixed code (wait_front restored) stays correct, because
  `cb_wait_front` blocks until the data lands no matter how long the packer is delayed.

### Kept-in-tree repro (how to use it — no asserts)

The NOP perturbation is left in the source so the ND can be reproduced/iterated on a
plain build (no LLK asserts). It is **off** unless `SDPA_NOP_PERTURB` is defined.

- **Knobs** — top of `ring_joint_sdpa.cpp`:
  ```c
  #define SDPA_NOP_PERTURB 1   // comment out to disable the whole repro
  #define SDPA_NOP_U 0         // unpack nops (consumer side — does NOT trigger)
  #define SDPA_NOP_M 0         // math nops
  #define SDPA_NOP_P 4096      // pack nops  <-- this is the trigger
  #define SDPA_NOP_RISCV 0     // 0 = Tensix TTI_NOP, 1 = RISC-V nop
  ```
- **Mechanism / placement** — `sdpa_nops<N,riscv>()` + `sdpa_nop_perturb()` live in
  `compute_streaming.hpp`; the perturbation is called inside `blocked_matmul_and_pack`
  right before the QK `cb_qkt_im` pack, gated to `transpose==true` (QK only). It is a
  **loop** (not unrolled `.rept`) and emits **no DPRINT** — both matter (see below).
- **Run** (plain build): `scripts/run_safe_pytest.sh <test>` — fails 3/3 with
  `iteration 1 differs from iteration 0, max diff=2.75…3.97` (varying magnitude =
  genuine ND). Kernels rebuild at runtime, so just edit the `#define`s and re-run.
- **Threshold:** ~1024 TTI pack-nops (loop) is the knife-edge; 4096 has margin. Dial
  `SDPA_NOP_P` down toward ~1024 for smaller, more "natural-looking" diffs.
- **Gotchas (cost real device runs to learn):**
  - Use a **loop**, never unrolled `.rept N` — thousands of inline nops overflow the
    kernel config buffer ("Program size too large"), which looks like a failure but is
    a build error, not ND.
  - **No DPRINT** in the inserter — DPRINT on the compute threads of this ring-fabric
    op deadlocks it (CCL semaphore timeout, "device unrecoverable").
  - A *fixed* delay can also land in a stable-but-wrong regime (passes the determinism
    check while being numerically wrong); the failing magnitudes above are genuinely
    nondeterministic (iter1 ≠ iter0, varying run-to-run).

## Is disabling `can_reduce_trigger` "just changing timing"?

This was the key question — disabling the optimization changes a lot
(removes the split MOP, the `FPU_SFPU` post/wait, instruction counts), so on its
own it can't distinguish "fixed a real sync gap" from "perturbed timing into a
lucky window."

**No, it is not just timing — it restores a real barrier.** Two reasons:

1. **Code path.** `can_reduce_trigger=false` ⇒ `respect_trigger=false` ⇒
   `reduce_c_row_group` executes `CircularBuffer(in0_cb).wait_front(...)`. So
   disabling the optimization routes through the branch that contains the proper
   producer→consumer barrier.

2. **Isolation experiment.** The "un-skip `cb_wait_front`" run keeps
   `respect_trigger=true` — the split MOP, the `FPU_SFPU` handshake, and all of
   its timing-changing instructions still execute (and the asserts still run) —
   and only **adds back the barrier**. It passes. So with the split-MOP timing
   held constant, the barrier alone is what flips reliable-fail → pass.

`cb_wait_front` is **not a fixed delay**: it blocks on the producer's
`push_back` count, so its only effect is to enforce *producer-before-consumer*
ordering (a happens-before guarantee). If the data is already present it returns
immediately; if not, it blocks exactly until the data lands. A pure timing
perturbation has no such ordering guarantee. Therefore the failure is caused by a
genuine missing synchronization, i.e. a real (latent) race — not by the
optimization merely shifting timing.

## What is and isn't proven

- **Proven — real missing synchronization:** the first-half read has no producer
  barrier (structural fact in the source); restoring that barrier (two independent
  ways) deterministically fixes the failure while the split-reduce still computes
  correctly given synchronized input.
- **Proven — reachable without LLK asserts:** the tactical pack-thread NOP delay
  reproduces the ND on a plain build (no asserts), and a *producer*-side delay is what
  triggers it while a *consumer*-side delay never does — i.e. the window is reachable
  by ordinary pipeline-timing pressure, not only by assert instrumentation. (Earlier
  this was the one open gap; the NOP repro closed it.)
- **Proven — first divergence is the reduce, not its input:** the CB-hash probe shows
  the raw QK^T score bytes feeding the reduce are bit-identical across iterations
  (0/400 cores), while the post-reduce `exp(score − max)` numerators diverge (70/400).
- **Still open (probability, not mechanism):** how *often* QB2 production timing lands
  in the window on its own without any perturbation. The mechanism and reachability are
  settled; only the natural hit-rate is unquantified.

## Direct measurement: CB-hash ND-bisection probe (PR #43041) — done

Beyond the inference (structural source fact + barrier-restore isolation), the root
cause was **directly measured** with the CB-hash ND-bisection tool from
[PR #43041](https://github.com/tenstorrent/tt-metal/pull/43041) ("HASH LLK and Compute
API for ND Bisection"), merged to main 2026-06-05 and already in-tree:

- `tt_metal/hw/inc/api/compute/debug/cb_hash.h` — `hash_cb_trisc()` / `hash_cb_sfpu()`
- gated on the `DEBUG_CB_HASH` kernel define (zero overhead when off).

**Use `hash_cb_trisc(cb, n, label)`** — it runs as pure scalar RISC-V on the **UNPACK
thread** (the exact thread doing the racy read) and FNV-1a-32-hashes `n` tiles from the
CB's front (`fifo_rd_ptr`), DPRINTing `hash[label] cb=N tiles=n = 0x…`. It touches no
DEST/SFPU, so it doesn't disturb the part of the pipeline we trust.

Plan:
1. Build the SDPA compute kernel with `DEBUG_CB_HASH` (kernel define in the program
   factory — kernels build at runtime, no host rebuild).
2. Drop `hash_cb_trisc(cb_qkt_im, …)` immediately before the reduce's first-half MOP
   read in the `reduce_trigger` path (buggy `.hpp`), and hash the upstream QK-matmul
   output CB too.
3. Run the determinism test (one run = 10 internal iterations) and diff the hash lines.
   **A hash that changes across iterations for `cb_qkt_im` at the consumption point =
   the bytes feeding the row-max reduce are nondeterministic** — direct proof of the
   stale first-half read, and it localizes the ND to that specific CB + line rather than
   inferring it.

Caveats:
- **Heisenbug risk (most important):** the probe adds UNPACK scalar work + a heavy,
  serializing DPRINT — same class of timing perturbation as the LLK asserts but pushing
  the *opposite* way (it slows the unpacker, so it can stop it running ahead and **mask**
  the race). So a *changing* hash is strong positive evidence; a *stable* hash is **not**
  exoneration.
- **`hash_cb_sfpu` does not apply here:** it requires INT32 input, but `cb_qkt_im` holds
  float scores. Use the scalar `hash_cb_trisc` only (which is the right choice anyway —
  DEST/SFPU are not the suspect; L1 producer ordering is).

### Result (ran it — first divergence localized)

Practical notes from actually running the probe on this op:

- **DPRINT in the reduce hot path deadlocks the ring-fabric op** (CCL semaphore
  timeout, "device unrecoverable"). Workaround that worked: fold each firing's FNV
  hash into a per-core accumulator (scalar L1 read, **no print**) and emit **one**
  DPRINT line per dispatch at `kernel_main` exit. Filter `TT_METAL_DPRINT_RISCVS=TR0`
  (probe is UNPACK) to drop dispatch-core noise.
- **`TT_METAL_LLK_ASSERTS=1` is mandatory** — the probe must run under the same
  repro trigger; without asserts the op is deterministic and the whole comparison
  is meaningless.

With that, comparing the per-core accumulators across the test's reuse iterations:

| Probe point (settled L1 bytes, iter 0 vs iter 1) | Diverging cores |
|---|---|
| reduce **INPUT** — raw QK^T scores at the consumption point | **0 / 400** |
| **post-softmax** — `exp(score − max)` numerators in `cb_qkt_im` | **70 / 400** |

(bug still reproduced with the probe in place: output max diff ~0.04–0.06.)

**Interpretation — the first divergence is the row-max reduce, not its input.**
The raw QK^T score *bytes* feeding the reduce are bit-identical across iterations on
every core (independently re-confirms "inputs/all-gather reproduce identical bytes").
The divergence first appears *after* the reduce + `sub_exp`: the row-max is
nondeterministic, so `exp(score − max)` differs on 70 cores. This directly localizes
the ND to the `reduce_trigger` split-MOP read — matching the structural argument above.

Note the stale read is **not observable in settled input bytes**: a separate scalar
read sees deterministic data because the packer's write *does* land eventually — the
race is purely *when* the unpacker's first-half MOP samples L1, which no separate read
can reproduce. The probe therefore can't snapshot the stale read directly; it catches
the **effect** (a corrupted row-max) in the post-softmax numerators. This is the
strongest direct evidence short of the barrier-restore experiment, and is consistent
with it.

## Fix

Two options:

1. **Known-good mitigation:** disable `can_reduce_trigger` in `sdpa_ring_v2` (set to
   `false && …`, keeping the conditions so intent is documented and re-enabling is one
   edit). The non-trigger path already has the correct `cb_wait_front` barrier, so
   correctness is restored; the only cost is losing the first-half/second-half overlap
   on the reduce (small — the reduce is minor vs. the two matmuls). Verified above
   (100 iters PASS under asserts; pack-NOP delay also PASS; accuracy unchanged).

2. **Proper fix (preferred, in progress):** keep the optimization but make the packer
   publish the **first half** before the unpacker reads it — a separate `push_back` /
   handshake after the first-half subblocks, not a single all-at-once `push_back` after
   every subblock (which is why a partial first-half wait gives no overlap today).

**Current branch state:** the `false &&` mitigation is **not** applied; instead the
tree carries the kept-in-tree NOP repro (see above) so option 2 can be developed and
validated against an asserts-free reproduction. A correct option-2 fix should flip the
NOP repro from FAIL → PASS (same as the mitigation does), since the restored ordering
makes the pack delay harmless.

## Still latent elsewhere

The same pattern exists in the **non-ring** streaming path in
`compute_streaming.hpp` (`can_reduce_trigger` ~L1879 and
`can_reduce_trigger_padded` ~L1890). Left untouched: no determinism test covers
it and it is a hot path with perf risk. Worth the same treatment if bit-exact
determinism is required there.
