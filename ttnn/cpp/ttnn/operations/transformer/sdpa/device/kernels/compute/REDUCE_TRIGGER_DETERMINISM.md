# SDPA `reduce_trigger` determinism — root cause, repro, and fix (#47911)

Everything known about the nondeterminism behind issue
[#47911](https://github.com/tenstorrent/tt-metal/issues/47911)
(`ring_mla` / ring-joint SDPA on Blackhole): what `reduce_trigger` is, the producer→consumer
race it introduced, how to reproduce it with **and without** LLK asserts, the evidence, and
the implemented fix.

An interactive walkthrough of all of this lives next to this file:
**`reduce_trigger_race.html`** (open in a browser — tabs: back story, the race, the
semaphore before→after, the fix).

---

## TL;DR

- **Symptom:** `test_ring_mla_determinism[ring_mla-mla_100k-q160-k320]` runs the op 10× with
  identical inputs and asserts bit-exact outputs; it fails iter 1 ≠ iter 0.
- **Root cause:** the `reduce_trigger` optimization lets the row-max reduce read the QK^T
  scores in `cb_qkt_im` **before the packer has written them**, with no producer→consumer
  barrier on the first half. It's a classic two-thread shared-L1 race (**T2 pack → T0 unpack**).
- **Fix (implemented):** keep `reduce_trigger` enabled, but re-anchor its single `FPU_SFPU`
  semaphore — post it **after** the QK pack + mask + push (with `STALL_PACK`), and make the
  reduce **wait before both MOP halves** (not just the second). One semaphore, 1 post / 1 get,
  group_size-safe.
- **Status:** validated on Blackhole — race repro (asserts and NOP) now PASS, accuracy
  unchanged (PCC 0.9996 / RMSE 0.0082), 7/7 broader determinism configs PASS, no hang under
  producer- or consumer-lead perturbation. WH mirrored (needs CI). Perf A/B still owed.

---

## 1. Background — the scope (for someone new to this kernel)

The kernel computes **attention** (transformer SDPA) on Tenstorrent hardware, **streaming**
the keys/values a chunk at a time (flash-attention online softmax) and **ring**-rotating them
across devices.

**One Tensix core, three concurrent threads**, data flowing T0→T1→T2:

- **T0 · Unpack** — moves tiles from L1 into the `SrcA`/`SrcB` register files.
- **T1 · Math** — FPU (matmul) / SFPU (vector) compute into `DST`.
- **T2 · Pack** — moves `DST` results back out to L1.

They run independently and only stay ordered where code **explicitly** makes one wait for
another (a circular-buffer flag or a hardware semaphore). **Pack (T2) is the last pipeline
stage**, so it structurally *trails* the others — remember this; it's the crux.

**The attention inner loop, per key-chunk:**

```
1. QK^T  -> scores        (written to cb_qkt_im in L1)
2. row-max (reduce)       <-- the buggy step
3. exp = softmax numerators  (in place in cb_qkt_im)
4. ·V                     (QK^T @ V)
5. accumulate output      (online-softmax rescale)
```

Step 2 needs the **maximum score per query row** before `exp(score − max)` (numerically
stable softmax). The scores are produced by **T2 (pack)** into `cb_qkt_im` and read back by
**T0 (unpack)** for the reduce — so `cb_qkt_im` is a **producer→consumer handoff** that needs
synchronization.

### `cb_qkt_im` is an in-place scratchpad, not a classic FIFO

`cb_qkt_im` is read by **three** passes within a q-chunk (row-max reduce, in-place `sub_exp`,
then QK^T@V). So it's not produce-once/consume-once:

- Producer publishes with `cb_push_back_hold_wr_ptr` (advances the front/available count but
  **holds the write pointer**, so the three passes address the same region).
- The reduce's classic `cb_wait_front` exists only on the **non-trigger** branch
  (`reduce_c_row_group`, `if (!respect_trigger)`).
- A single `pop_front` frees it at the **end of the whole q-chunk**.

---

## 2. What `reduce_trigger` is (the optimization)

A **latency-hiding** optimization for the row-max reduce. Normally the reduce stalls on
`cb_wait_front(cb_qkt_im)` until *all* scores are packed, then reduces. `reduce_trigger`
instead **starts the reduce early and overlaps it with the tail of packing**:

1. **Splits the reduce's unpack MOP into two halves.** The LLK `mop_config` is programmed for
   `block_ct_dim/2` iterations, so one `ckernel_template::run()` walks the first half of the
   score columns and a second `run()` walks the rest.
2. **Drops `cb_wait_front`** and substitutes a cheap **hardware semaphore** handshake
   (`FPU_SFPU`): the packer posts it after the last subblock; the unpacker waits on it
   *between* the two halves.

Intended win: the first-half row-max unpack runs while the packer is still finishing the last
columns, instead of the unpack engine sitting idle. (Small — the reduce is unpack-bound and
dwarfed by the two matmuls — but it is a hot path.)

The columns are packed **in order** (subblock 0 → cols 0..sbw−1, …, last subblock → highest
cols), and the reduce reads them in order, which is what makes an early first-half read
*seem* safe.

---

## 3. The `FPU_SFPU` hardware semaphore (how the handshake works)

`FPU_SFPU` is one of the core's **T6 thread-semaphores** — a small counter visible to all
three threads. Defined in `tt_llk_*/common/inc/ckernel.h`:

| Compute call | HW op | Effect |
|---|---|---|
| `t6_semaphore_init(FPU_SFPU, 0, 1)` | `TTI_SEMINIT` | start at **0**, saturate at **1** |
| `t6_semaphore_post(FPU_SFPU)` | `TTI_SEMPOST` | **+1** (the signal). `<STALL_PACK>` first stalls until the pack engine's L1 writes drain, *then* posts |
| `t6_semaphore_wait_on_zero(FPU_SFPU)` | `TTI_SEMWAIT(…, STALL_ON_ZERO)` | **stall while value == 0**, proceed when nonzero. **Non-consuming** (does not change the value) |
| `t6_semaphore_get(FPU_SFPU)` | `TTI_SEMGET` | **−1** (consume) |

Signaling is cross-thread through this counter: T0's `wait_on_zero` holds the unpack unit
while the counter is 0; the instant **T2's `post`** makes it nonzero, the hardware releases
T0. No polling, no L1 traffic.

**The property the fix relies on:** `wait_on_zero` is non-consuming, so a single "sticky"
token (value 1) can satisfy **many** waits; only one `get` clears it. (Name is backwards:
"wait_on_zero" = *wait while zero, go when nonzero*.)

---

## 4. Root cause — the race (confirmed)

**The first half of the split reduce MOP reads `cb_qkt_im` with no producer→consumer barrier.**

In `_llk_unpack_AB_reduce_block_max_row_runtime_`
(`tt_metal/tt-llk/tt_llk_{blackhole,wormhole_b0}/llk_lib/experimental/llk_unpack_AB_reduce_custom_runtime.h`),
the original trigger path was:

```cpp
ckernel::ckernel_template::run();         // FIRST half: cols [0, block_ct_dim/2)  <-- NO WAIT
t6_semaphore_wait_on_zero(FPU_SFPU);      // the only sync — sits BETWEEN halves
ckernel::ckernel_template::run();         // SECOND half: cols [block_ct_dim/2, block_ct_dim)
```

So the unpacker **consumes the first part without waiting, and only waits for the second part**.
The single `FPU_SFPU` post (in `blocked_matmul_and_pack`, after the *last* QK subblock) guards
only the second half, and `reduce_c_row_group` skips `cb_wait_front` on the trigger path.

**Why the early read goes stale:** the design bet that the packer is always ahead, so the
early (first-half) columns are surely written by the time the reduce starts. But **T2 (pack)
is the last pipeline stage and trails** T0. When T0 finishes feeding the QK matmul and jumps
straight into the first-half read, T2 may still be packing those very columns → T0 latches
stale L1 → wrong row-max. Whether it wins or loses the race is decided by the cycle-level
T0/T2 interleaving, which shifts with any timing pressure → **nondeterministic**.

For the failing config: `block_ct_dim = Sk_chunk_t = 10`, `qkt_subblock_w = 2` (5 subblocks);
the first `run()` reads cols 0–4, written by subblocks 0–2, but the only post is after
subblock 4 — so cols 0–4 are read with no guarantee they're packed.

**A second, related defect:** the post was also anchored to the **wrong producer event**. The
reduce reads `cb_qkt_im` *after* an in-place mask stamp (causal/padding,
`begin/apply/end_mask_l1_accumulate`), but `FPU_SFPU` is posted **before** the mask. The only
event that dominates *all* writers (QK pack **and** mask) is the `cb_push_back_hold_wr_ptr`.

**It can only be a compute-side race:** inputs are identical each iteration, so the ring
all-gather reproduces identical bytes; a data-movement/semaphore/buffer race could only ever
reproduce identical reads.

---

## 5. Reproduction

### A — LLK asserts (the original CI repro)

```bash
TT_METAL_LLK_ASSERTS=1 scripts/run_safe_pytest.sh \
  tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_mla_determinism
```

- The failing CI job is the nightly debug run "with LLK asserts". One test run is enough (it
  loops the op 10× internally). Observed **11/11 fail** at iter 1, ~11 s; max-diff magnitude
  varied run-to-run (~0.02–0.14) — varying magnitude is the nondeterminism signature.
- Without asserts a plain release build passed 200+ iters — asserts are one timing
  perturbation that reaches the race window; QB2 hardware timing can too.

### B — Tactical NOP delay (no asserts) — kept in tree, OFF by default

Proves it is a real race, not an asserts artifact: inject a delay on the **PACK** thread to
widen the window on a plain build. Helper `sdpa_nops<N,riscv>()` / `sdpa_nop_perturb()` in
`compute_streaming.hpp`, called in `blocked_matmul_and_pack` right before the QK pack (gated
`transpose==true`); knobs `SDPA_NOP_{PERTURB,U,M,P,RISCV}` at the top of `ring_joint_sdpa.cpp`.

To use: uncomment `#define SDPA_NOP_PERTURB 1`, run the determinism test (no asserts). When
`SDPA_NOP_PERTURB` is undefined it compiles to nothing. Gotchas learned the hard way:

- Use a **loop**, not unrolled `.rept` — thousands of inline nops overflow the kernel config
  buffer ("Program size too large"), which looks like a failure but is a build error.
- **No DPRINT** in the inserter — DPRINT on the compute threads of this ring-fabric op
  deadlocks it (CCL semaphore "device unrecoverable" timeout).

Results on the **buggy** code (`TT_METAL_LLK_ASSERTS` unset):

| Config | Result |
|---|---|
| **pack** TTI nops 1024 / 8192 / 65536 | FAIL — ND, max diff 2.75 → 3.97 |
| **pack** RISC nops 4096 | FAIL — ND, max diff 3.17 |
| **unpack** nops 8192 (consumer side, wrong direction) | PASS (suppresses) |
| no nops (baseline, no asserts) | PASS |

Only delaying the **producer** (pack) triggers it; delaying the **consumer** (unpack) never
does — exactly a missing producer→consumer ordering.

---

## 6. Evidence chain (all consistent)

1. **CB-hash bisection** (PR #43041 `hash_cb_trisc`, in-tree). Fold a per-core FNV hash of
   `cb_qkt_im` at the reduce-input point and post-softmax, emit one line per dispatch (DPRINT
   in the hot path deadlocks the fabric — accumulate, print once). Result: reduce **input**
   bytes are **bit-identical** across iters (0/400 cores), but the **post-softmax**
   `exp(score − max)` numerators **diverge** (70/400). ⇒ ND is born at the reduce, not its
   input. (The stale read isn't visible in *settled* input bytes — the packer's write lands
   eventually; the race is purely *when* the MOP samples L1 — so the probe catches the
   **effect**, not the read.)
2. **NOP repro direction** — producer-delay triggers, consumer-delay doesn't (§5B).
3. **Barrier-restore isolation** — with all the split-MOP/`FPU_SFPU` instructions still
   running, restoring *only* the `cb_wait_front` flips reliable-fail → PASS.

---

## 7. The fix (implemented) — single re-anchored `FPU_SFPU` token

**Design:** keep `reduce_trigger` enabled and keep the split MOP; fix only the
synchronization. There is still **exactly one** semaphore.

1. **Remove** the post from `blocked_matmul_and_pack` (it predated the mask and only ordered
   the QK pack).
2. **Add** one `t6_semaphore_post<p_stall::STALL_PACK>(FPU_SFPU)` immediately after
   `cb_push_back_hold_wr_ptr` (gated `if (reduce_trigger)`), so the token dominates the QK
   pack **and** the in-place mask **and** the push; `STALL_PACK` guarantees those L1 writes
   are committed before the token appears.
3. In the LLK runtime, **wait before both halves** on the one sticky token; no `get` in
   runtime; keep the single `get` in `_uninit_`:
   ```cpp
   t6_semaphore_wait_on_zero<p_stall::STALL_UNPACK>(FPU_SFPU);   // NEW: guard the first half
   ckernel::ckernel_template::run();                            // first half
   t6_semaphore_wait_on_zero<p_stall::STALL_UNPACK>(FPU_SFPU);   // same sticky token (non-consuming)
   ckernel::ckernel_template::run();                            // second half
   ```
4. `t6_semaphore_init(FPU_SFPU, 0, 1)` unchanged (`max=1`). Mirror the LLK edit BH↔WH.

### "Did we remove one semaphore and add two?" — no

| | before (buggy) | after (fix) |
|---|---|---|
| semaphores | 1 (`FPU_SFPU`) | 1 — unchanged, `max=1` |
| **posts** (T2) | 1, after last QK pack, `<NONE>`, *before the mask* | 1, after QK + mask + push, `<STALL_PACK>` — **moved + stalled** |
| **waits** (T0) | 1 — only between the halves (first half unguarded) | **2** — before each half (same token, non-consuming) |
| **gets** (T0) | 1 (in uninit) | 1 (in uninit) — unchanged |
| token balance | 1 post / 1 get | 1 post / 1 get → balanced for any `group_size` |

The "2" is **two waits**, not two semaphores. Because `wait_on_zero` is non-consuming, one
post (token=1) satisfies both waits and the single `get` clears it.

### Why correct

Every region the reduce reads is now preceded by a wait on a token posted only after
`cb_push_back` — program-ordered after the last QK pack and the entire mask, with `STALL_PACK`
ensuring L1 visibility. Closes both missing edges (first-half→producer, both-halves→mask).
Bit-exact (only ordering added). Under the NOP/pack-delay repro, delaying T2 just delays the
post, so T0 stalls instead of reading stale L1.

### Why NOT a two-token (`max=2`) handshake

A multi-agent design pass first proposed two posts + a `get` inside the per-row runtime.
Adversarial review found it **deadlocks for `group_size = qkt_subblock_h ≥ 2`**: the reduce
runtime runs per row (`group_size×`) but uninit once; a consuming `get` in the per-row path
makes gets = `group_size+1` vs 2 posts → the semaphore underflows mid-loop → `wait_on_zero`
stalls forever → device hang. (It also violates the `reduce_custom.h` "posts must equal uninit
calls" contract, and `max=2` can saturate-and-lose a token when PACK leads.) The single-token
form above keeps the original 1-post/1-get counting (which is `group_size`-safe because the
waits don't consume) and avoids all of that.

### Files touched

- `ttnn/.../compute/compute_streaming.hpp` — remove post in `blocked_matmul_and_pack`; add the
  re-anchored `STALL_PACK` post after `cb_push_back_hold_wr_ptr`.
- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/experimental/llk_unpack_AB_reduce_custom_runtime.h`
  — leading `wait_on_zero` before the first half.
- `tt_metal/tt-llk/tt_llk_wormhole_b0/.../llk_unpack_AB_reduce_custom_runtime.h` — same (parity).

Signatures unchanged (`respect_trigger` stays a bool), so the llk_api / Compute API wrappers
need no signature changes.

---

## 8. Validation (Blackhole)

All with `can_reduce_trigger` **enabled**:

| Test | Before fix | After fix |
|---|---|---|
| NOP pack-delay repro (no asserts) | 3/3 FAIL | **3/3 PASS** |
| LLK asserts — `mla` determinism (CI repro) | 11/11 FAIL | **2/2 PASS** |
| Packer-lead stress (unpack nops 8192) | — | **PASS, no hang** |
| Accuracy (`mla`) | — | **PASS, PCC 0.9996 / RMSE 0.0082** (= baseline) |
| Broader determinism, 7 configs, asserts (incl. group_size>1) | — | **7/7 PASS** |

Determinism alone only proves bit-exactness; the accuracy PCC confirms the output is *correct*,
not deterministically-wrong.

---

## 9. Performance note (honest)

For **masked** configs (all production ring-MLA: causal/padding) the fix does **not** recover
the original first-half/packing overlap — the mask is a monolithic in-place pass between the
pack and the reduce over the same columns, so no early signal can honestly release the
first-half read before the mask completes. **That overlap was never actually safe with a mask
present.** The reduce now effectively starts at `push_back`. Estimated loss is low-single-digit
% (reduce is unpack-bound, dwarfed by the two matmuls). **A perf A/B is still owed** (perf
harness: `tests/nightly/sdpa_perf_utils.py`) to confirm it's within run-to-run noise.

Recovering true overlap would need a **column-phased mask split** (apply/signal the first-half
columns independently) — high effort/risk, low reward; deferred. For a genuine no-mask config
(`uses_lightweight_mask == false`) the overlap is still safe and could be kept as a follow-up.

---

## 10. Residual risks / follow-ups

- **Wormhole** edit is mirrored but unvalidated here (BH-only device) — needs CI coverage.
- Confirm `STALL_PACK` fully drains the mask's in-place L1-accumulate (validated empirically:
  passes the strongest repro; matches the existing `PACK_DONE` idiom at lines ~1518/1582/1611).
- **Non-ring + padded paths** (`sdpa_standard_v2`, `can_reduce_trigger` ~L1879 /
  `can_reduce_trigger_padded` ~L1890) share `reduce_c_row_group` / `blocked_matmul_and_pack` /
  `init_sdpa_streaming_semaphores`, so the LLK + post + init changes cover them — verify each
  post site sits after its own `push_back`.
- Perf A/B (§9), and the optional no-mask fast path that keeps the overlap.
- The NOP repro scaffolding is kept (gated off) for regression — remove it before final merge
  if undesired.
