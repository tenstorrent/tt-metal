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

For the failing config (`q160`/`k320`): `block_ct_dim = Sk_chunk_t = 10`, and
`determine_largest_subblock_size(Sq_chunk_t=5, Sk_chunk_t=10, dst_size=8)` picks
`qkt_subblock_w = 5` → **2 subblocks** (subblock 0 = tiles 0–4, subblock 1 = tiles 5–9).
The first `run()` reads the first half [0,5) = subblock 0, but the only post is after
subblock 1 (the last, `kt_subblock == kt_num_full_subblocks - 1`) — so subblock 0's tiles
are read with no guarantee they're packed.

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

---

## 11. Adversarial review of the fix (Pavle's objection) — verdict + Phase-2 mandate

A reviewer (Pavle) challenged both the diagnosis and the fix. A multi-agent investigation
(5 lenses → red/blue debate → judge, conf 0.8) ruled on each claim against the actual code:

| # | Claim | Verdict | Why (code-grounded) |
|---|-------|---------|---------------------|
| 1 | Sync only needed on the **last** MOP half (early columns already committed) | **WRONG** | Geometry premise is right — `run()#1` reads the *early* columns (LLK `outerloop=block_ct_dim/2`, Z-increment, single counter reset, no reset between halves). But the conclusion fails: the producer is **T2 pack**, the trailing stage, with no happens-before to the early-column read; deleting the first-half wait reverts to the old "run first half immediately" shape, and the re-anchored post fires *later* (after the whole mask), **widening** the stale window. Also the in-place mask rewrites those low columns *after* the QK packs, so "early columns committed" cannot hold on any masked (production) config. |
| 2 | Math-thread serialization (MM math before reduce math on T1) is a backstop | **WRONG** | Wrong thread pair. Hazard is **T2 pack → T0 unpack** on L1. T1 in-order math orders neither the T0 L1 read nor the T2 L1 write; the three threads' frontends are independent and sync only on CB flags / T6 sems. |
| 3 | The fix is just timing perturbation, not a real barrier | **WRONG** | The fix installs genuine ordering ops: release = `STALLWAIT(STALL_SYNC,STALL_PACK)+SEMPOST` (drains the pack engine; only emitted when `WaitRes!=NONE` — the *old* `p_stall::NONE` post emitted **no** drain), acquire = `SEMWAIT/STALL_ON_ZERO`. Corroborated by the wait_front isolation (count-based barrier flips FAIL→PASS). |
| 4 | Cleaner to **fuse mm+reduce in the LLK** (kill the cb_qkt_im L1 round-trip) | **HALF / RIGHT DIRECTION** | Correct long-term *performance* play (row-max is associative → foldable per sub-block), but **not** a refutation of the current fix and **not** a drop-in: the in-place mask must run on the full score row between QK and reduce, and `cb_qkt_im` is read 3× (reduce / sub_exp / @V), so fusing mm+reduce alone still needs the L1 materialization. |

**Bottom line:** the shipped fix is **correct** and minimal; the added first-half wait is
**load-bearing**; it is **not** timing perturbation. Pavle is wrong that it's incorrect — but
**right that it costs perf**: it concedes the first-half/pack overlap on every masked production
path (§9). That perf cost is the real open issue and is what Phase 2 must close.

### One honest residual (the only place Pavle isn't clearly wrong)
For a **no-mask** config (`uses_lightweight_mask == false`), with the post draining the whole
pack engine, the second-half wait *might* also order the first-half columns — so the first-half
wait *could* be redundant there. All masked production paths still need it. (Isolated by the
"no-mask second-half-wait-only" experiment below.)

### Decisive experiments (to run on HW — settle remaining uncertainty)
1. **First-half wait load-bearing?** Keep the re-anchored `STALL_PACK` post, revert *only* the LLK
   to second-half-wait-only (drop the first-half wait). Run determinism (+ PACK-NOP repro, P=4096)
   with/without LLK asserts. **Predict: FAILS** (ND returns) → first-half wait load-bearing, claim #1 wrong.
2. **Post re-anchor load-bearing?** Keep both LLK waits, revert the post to old per-last-subblock
   `p_stall::NONE`. **Predict: FAILS** → STALL_PACK re-anchor independently necessary.
3. **No-mask isolation:** on `uses_lightweight_mask==false`, second-half-wait-only + re-anchored post
   → isolates whether the first-half wait is redundant absent the mask.
4. **Perf A/B:** shipped fix vs pre-fix (parent `26fdd6e5a5a^`) on ring-MLA masked configs → quantify
   the conceded overlap loss; bounds the value of the Phase-2 fused work.

### Phase-2 mandate (chosen direction)
Keep the proven barrier for **correctness**; recover the conceded perf so the single-chip/QB
`SDPA_PERF_CHECKS` gates (±1% band) pass at-or-above main. Candidate mechanisms, cheapest→biggest:
(a) **no-mask fast path** keeps the overlap where safe; (b) **column-phased mask split** (signal
first-half columns independently); (c) **fused mm+reduce LLK primitive** (Pavle) — biggest, recovers
all paths. Phase 2 converges on the safest mechanism that meets the perf gate.

---

## 12. Measured perf A/B (the owed §9 numbers) — QuietBox ring4, BH

Clean A/B isolating the fix (revert the 3 correctness files: BH+WH LLK + `compute_streaming.hpp`;
keep `ring_joint_sdpa.cpp` at HEAD so the NOP repro stays OFF). `SDPA_PERF_CHECKS=1`, single samples:

| config (ring4) | pre-fix baseline | with fix | Δ util | Δ wall | perf gate (±1%) |
|---|---|---|---|---|---|
| `wan2_2_1xGLX` q288/k512 (**non-causal, no mask**) | 68.60% / 7.883 ms | 67.06% / 8.063 ms | **−1.54 pp** | **+2.3%** | pre-fix PASS → fix **FAIL** |
| `mla_100k` q160/k320 (causal, mask iter0 only) | 62.85% / 4.812 ms | 62.49% / 4.840 ms | −0.36 pp | +0.6% | pre-fix PASS → fix **FAIL (marginal/noise)** |
| ring_mla < separate-V | 4.674 < 4.804 ms ✓ | 4.720 < 4.850 ms ✓ | — | — | both PASS |

Also re-confirmed: determinism+asserts **PASS** with fix (21 s); **FAIL** unfixed with **NOP OFF**
(13 s) — i.e. the natural race reproduces without the artificial NOP amplifier.

**Conclusion:** the fix is a real perf regression on the perf gate. The dominant hit (`wan`,
−1.54 pp) is a **no-mask** config, so it is **pure lost pack↔reduce overlap** — the mask is not
even involved there. Recovering the first-half/pack overlap on no-mask iterations (and the
non-masked ring iterations of causal configs) is the highest-value, lowest-surface fix. This is
exactly the Phase-2 mandate (§11).

---

## 13. Phase-2 solution (chosen + validated design) — two-phase reduce_trigger handshake

**Goal:** recover the §12 perf loss (esp. wan −1.54pp, a no-mask config) WITHOUT reopening the race.

**Why the obvious options fail:**
- *Single-semaphore two-phase* (architect's first idea): impossible. FPU_SFPU is `SEMINIT(0,1)`, `wait_on_zero` is non-consuming, one `get` at uninit. Once any post sets the saturating token, ALL later non-consuming waits (run#2, every row) pass → second half loses its barrier.
- *Wait-elision on no-mask (mechanism a)*: `run#1(no wait); wait; run#2` is **exactly the pre-fix LLK shape**; run#1 is unguarded → reopens #47911 (pre-fix measured FAIL). Ruled out by construction.

**Chosen: TWO SEMAPHORES (two-phase), no-mask path only.**
- Borrow **`UNPACK_MATH_DONE` (T6 sem 6)** as `PHASE1_SEM` — the only T6 index with **zero use** in the SDPA compute call graph AND not firmware-inited (firmware inits 1,2,4,7). Precedent: DeepSeek `SFPU_FPU` alias. **PACK_DONE(4) rejected**: its Phase-2 poster (@V region) can race across the q_subblock boundary and prematurely satisfy a phase-1 wait.
- **Producer (PACK):** on the no-mask overlap path, post `PHASE1_SEM` (STALL_PACK) inside the kt loop right after subblock `first_half_last_sb = (active_Sk/2 − 1) / actual_sbw` packs — the subblock that *covers* the last first-half column. Exact half-boundary alignment is **not** required: committing a superset of `[0, active_Sk/2)` is safe (run#1's columns are a subset), so no `active_Sk % (2*actual_sbw)` constraint is imposed. Keep the existing `FPU_SFPU` post after `cb_push_back` (gates run#2 = late columns).
- **Consumer (LLK overlap branch):** `wait(PHASE1_SEM); run#1; wait(FPU_SFPU); run#2`. Non-consuming waits; **one `get` each at uninit, the phase-1 get CONDITIONAL on overlap** (keeps balance for any group_size; masked path never posts/waits/gets PHASE1).
- **Gate:** `overlap_first_half = ENABLE_REDUCE_FIRST_HALF_OVERLAP && reduce_trigger && no_mask_this_iter`, where `no_mask_this_iter` is hoisted from the existing mask branches (`use_provided_mask` / `uses_lightweight_mask` / `should_apply_lightweight_mask`, incl. straddle) as the single source of truth. `can_reduce_trigger` already guarantees `kt_num_full_subblocks ≥ 2` so a second half always exists. Masked → fall back **byte-identical** to the committed barrier.

**Determinism safety (by construction):** `t6_semaphore_post<STALL_PACK>` emits `STALLWAIT(STALL_SYNC,STALL_PACK)` before `SEMPOST`, so the phase-1 token isn't observable until the first-half packs commit to L1 — same drain argument as the committed full-row fix, applied to the first half. run#2 unchanged. Masked iters execute the identical committed instruction stream.

**Files:** LLK `_llk_unpack_AB_reduce_block_max_row_runtime_` + `_uninit_` (BH+WH); llk_api wrappers (BH+WH); `reduce_custom.h` (runtime + uninit); `compute_streaming.hpp` (init seminit, kt-loop phase-1 post + predicate, `reduce_c_row_group` plumbing). **Build gotcha (confirmed by compile failure):** `TTI_SEMWAIT`/`SEMPOST`/`SEMINIT` encode the semaphore as an **immediate**, so run#1's semaphore must be a compile-time constant — the LLK selects it with an `if (overlap_first_half) wait(UNPACK_MATH_DONE) else wait(FPU_SFPU)` literal branch, **not** a runtime `first_sem` variable (a runtime select fails with "impossible constraint in 'asm'"). A constexpr master switch `ENABLE_REDUCE_FIRST_HALF_OVERLAP` gates the whole thing for easy A/B / one-line rollback.

**Validation order (Phase 3 loop):** determinism+asserts on masked mla (must PASS, must stay byte-identical) → no-mask determinism + NOP repro (P=4096) PASS → group_size>1 no-hang → perf gate ≥ pre-fix → accuracy. Rollback: if any determinism gate fails, set the master switch off (= committed fix) and accept the regression.

---

## 14. Phase-2 validation results (QuietBox ring4, BH) — two-phase fix

Implemented (UNPACK_MATH_DONE phase-1 token, `ENABLE_REDUCE_FIRST_HALF_OVERLAP=true`, no-mask path).
Build note: `TTI_SEMWAIT` encodes the semaphore as an **immediate**, so the run()#1 semaphore must
be a compile-time constant — the LLK branches with literal sem constants (a runtime sem select fails
to compile with "impossible constraint in 'asm'").

**Goal 1 — determinism (must PASS with fix, FAIL without):**
- two-phase + `TT_METAL_LLK_ASSERTS=1`, `ring_mla-mla_100k-q160-k320` (exercises masked iter0 barrier
  AND no-mask iters overlap): **PASS** (21 s), reproduced 3× (no flakiness).
- two-phase + asserts + **NOP repro `SDPA_NOP_P=4096`** (pack-thread delay, the decisive race-widener):
  **PASS** — proves the phase-1 STALL_PACK post truly orders run()#1's first-half read.
- unfixed (revert the 3 correctness files), asserts, **NOP off**: **FAIL** (natural race).

**Goal 2 — perf (must be unchanged-or-better vs main; ±1% gate):**

| config (ring4) | pre-fix (main) | committed single-token fix | **two-phase** | gate |
|---|---|---|---|---|
| `wan2_2_1xGLX` q288/k512 (no mask) | 68.60% / 7.883 ms | 67.06% (−1.54pp) ❌ | **68.56% / 7.887 ms** ✓ | [68.21,69.59] |
| `mla_100k` q160/k320 (causal) | 62.85% / 4.812 ms | 62.49% (−0.36pp) ❌ | **62.76% / 4.820 ms** ✓ | [62.57,63.83] |
| ring_mla < separate-V | 4.674<4.804 | 4.720<4.850 | **4.690<4.809 ms** ✓ | — |

Two-phase recovers **+1.50pp on wan** (back to pre-fix within noise) and **+0.27pp on mla**; all perf
gates PASS. The recovery also confirms the overlap path engages at runtime.

**Goal 3 — accuracy:** `test_ring_mla_accuracy` + `test_ring_joint_attention_sdpa_accuracy` (wan2_2 1x/4x,
videogen ×3, mla_100k, mla_128k): **8/8 PASS**.

**Verdict:** the two-phase handshake keeps determinism fixed (Pavle's correctness objection answered:
the barrier is moved earlier, not removed) AND closes the perf regression he flagged. WH mirror is
code-identical (overlap branch + conditional get); WH CI still owed. Rollback is one line
(`ENABLE_REDUCE_FIRST_HALF_OVERLAP=false` → committed single-token fix).
