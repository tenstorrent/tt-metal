# Ring Joint SDPA — combining 2-injector K-mcast with uniform Q-chunks/core

## Goal (next session)

Land **both** features together **without perf regression**. The bar is the
2-injector branch's standalone number on `mla_100k`, `q=160`, `k=320`:

> **≥ 60.9 % math util** (4.966 ms duration) on Blackhole 4-device single-ring (Quiet Box).

The naive cherry-pick combination measured in the previous session lost
~1 pp vs 2-injector-alone — see the matrix below. That is the regression
to investigate and remove before merging.

## Test scope

We are **only** measuring `mla_100k` at `q=160 k=320`. Do not run the full
sweep — it adds ~50 s without changing the answer.

```bash
source python_env/bin/activate

# Perf (math util)
RING_JOINT_QK_FILTER=q160-k320 scripts/run_safe_pytest.sh \
  tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_create_perf_table \
  -k mla_100k

# Accuracy (PCC) — run after any non-trivial change to program factory or kernels
scripts/run_safe_pytest.sh \
  tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_accuracy \
  -k mla_100k
```

`RING_JOINT_QK_FILTER` was added in commit `819a4910e9e`. On branches that
predate it (e.g. the standalone 2-injector tip), drop the env var and grep
the `q160-k320` row out of the table.

Build before perf runs:

```bash
./build_metal.sh
```

Device-side kernels JIT-build at runtime — no build step needed for
kernel-only edits, but host-side program-factory changes do require
`build_metal.sh`.

## Results matrix (mla_100k, q=160, k=320, BH 4-dev single-ring)

All five runs used the same machine and harness. Cores, iters/core, ring
eff, pad waste, slot waste are identical across all runs (87/100, 280, 79 %,
0.0 %, 4.8 %); only duration / FPU util / math util change.

| # | Run | Q DRAM mid-barrier | 2-injector K mcast | Uniform Q-chunks/core | Commit | Duration | **Math util** | Δ vs baseline |
|---|-----|:-:|:-:|:-:|---|---:|---:|---:|
| 1 | Baseline (`main` parent of uniform-Q) | ✓ (already in main) | — | — | `b3e8eb62c64` | 5.181 ms | **58.4 %** | — |
| 2 | Uniform Q only | ✓ | — | ✓ | `819a4910e9e` | 5.154 ms | **58.7 %** | +0.3 pp |
| 3 | 2-injector only | ✓ | ✓ | — | `3ff8eb14481` (`skrstic/sdpa-2-mcast-injectors`) | 4.966 ms | **60.9 %** | +2.5 pp |
| 4 | Both combined (naive cherry-pick) | ✓ | ✓ | ✓ | `5242f1709ab` (`temp/2injectors-plus-uniform-q`) | 5.050 ms | **59.9 %** | +1.5 pp |
| 5 | Both combined (per-chain max_q fix) | ✓ | ✓ | ✓ | tip of `temp/2injectors-plus-uniform-q` | **4.966 ms** | **60.9 %** | **+2.5 pp** |

Run 5 recovers run 3's number exactly while keeping the uniform-Q plumbing
in the codebase.

## Fix applied (run 5)

Hypothesis #1 from the previous session was correct: forcing both halves
to pad to the global `q_chunks_per_core` made the bottom chain do extra
K-mcast handshakes that contended on NoC with the top chain's last iter.

The fix backs out only the parts of the uniform-Q change that pinned the
chain padding to the global maximum, and leaves the CT plumbing in place:

- **Distribution:** restored standalone-style `base + extras-to-first-cores`
  (cores 0..extras-1 get `base+1`, rest get `base`). Extras land in the top
  half by row-major ordering, so `top_max_q > bot_max_q` again — the bottom
  chain's `chain_max_q` shrinks below the global max.
- **K-mcast `configure_chain`:** restored the `chain_max_q` parameter and
  set both `kc.this_core_q_chunks` and `kc.next_core_q_chunks` to it
  (per-chain), restored the `k_chain_max_q[ci]` per-core RT vector.
- **Reader RT arg:** re-added `max_q_per_core` after the batch chain
  config (only when `k_uses_batch_chain`).
- **Reader outer loop:** reverted `loop_q_count` to the standalone RT form
  `(k_uses_batch_chain && batch_mcast_enabled) ? max_q_per_core : q_per_core`.
  Restored the `is_padded_iter = q_iter >= q_per_core` predicate, the
  `2 * k_chunk_tiles` reserve hack on padded iters, and the post-K-mcast
  `if (is_padded_iter) continue;` that skips Q/V on padded iters.
  Dropped the wrap-around `% total_q_chunks` indexing — only needed by the
  old phantom mechanism.
- **Compute & writer:** `global_q_end` now equals `global_q_start + real
  q_count` (not `+ q_chunks_per_core`), so they iterate real chunks only.
  Dropped the `is_phantom`/`end_seq_tile=0`/empty-stats-range plumbing and
  the wrap-around indexing introduced for uniform-Q.
- **Kept (intentional):** the `q_chunks_per_core` CT arg in all three
  kernels (informational) and the `this_core_q_chunks` field on
  `ChainConfig` (used by the V chain's `should_receive` gate).

## What each change does

### Change A — 2-injector K mcast (`3ff8eb14481`)

Splits the K mcast chain into **two row-half chains** to cut NoC
contention.

- Cores partitioned by logical `y`: top half (`y < grid_size.y / 2`) vs
  bottom half. Each half forms its own mcast rectangle.
- Each half picks its own injector — the core with most real Q work in
  that half. The two injectors are forced into **different physical
  columns** so their forwards don't stack on the same NoC column.
- Each chain pads its loop count to its own half's `max_q` (via
  per-core RT arg `k_chain_max_q[i]` in the standalone branch).
- Falls back to single-chain unicast when `grid_size.y < 2`, or when
  either half has no work.

### Change B — Uniform Q-chunks per core (`819a4910e9e`)

Makes the per-core Q-chunk loop count a **compile-time arg**, uniform
across all cores. Trailing cores execute *phantom iters* that reuse the
K-mcast padded-iter handshake but skip Q/V DRAM reads, CB pushes, and
DRAM writes.

- `q_chunks_per_core = ceil(total_q_chunks / num_cores)`.
- Reader's outer Q loop count is now CT (slot 26) for all paths.
- Phantom iter detection: `global_q_start + i >= total_q_chunks`.
- Drops RT arg `max_q_per_core` (no longer needed — loop count is CT).
- Compute & writer keep RT-bounded loops over real chunks; CB sync from
  reader keeps them aligned.

See `UNIFORM_Q_CHUNKS_PER_CORE.md` next to this file for the full
write-up.

## Conflict resolution applied in run 4

Both features touch the same K-mcast chain configuration block in
`ring_joint_sdpa_program_factory.cpp` (and `append_to_args` reader-arg
emission for the K chain). Resolution kept on the combined branch:

1. **Outer if/else chain:** added the `q_chunks_per_core == 0` early
   fallback before the `grid_size.y < 2` fallback (both run before the
   2-injector main path).
2. **`configure_chain` lambda:** dropped the `chain_max_q` parameter.
   Set `kc.this_core_q_chunks = q_chunks_per_core` for every
   participating core (uniform), and `kc.next_core_q_chunks =
   q_chunks_per_core` for both injectors. Removed the `k_chain_max_q[ci]
   = chain_max_q` write.
3. **`k_chain_max_q` vector:** removed entirely (declaration + per-core
   RT arg push).
4. **Reader RT args:** dropped the per-core padding push
   (`reader_args.push_back(k_chain_max_q[i])`); the reader kernel already
   reads `q_chunks_per_core` as a CT arg and does not look at the RT
   slot.

The combined branch lives at `temp/2injectors-plus-uniform-q`
(`5242f1709ab`).

## Hypotheses for the anti-synergy (where to look first)

The chains are now **half size** (top/bottom split) AND every core does
the **same loop count** (uniform). On 87/100 cores, this lockstep across
two halves may be hurting in one of these ways:

1. **Phantom iters in one half wait for the other half's real work.**
   Because both halves now iterate `q_chunks_per_core` (the *global*
   max), but in the standalone 2-injector branch each half padded only
   to its own half's max, a half whose cores all finished real work
   early may now spin on phantom iters of the K mcast handshake, holding
   the chain. Pre-combination, that half's chain finished sooner.
   → Try: per-half `q_chunks_per_core_half = ceil(half_total_q / half_cores)`,
   keeping uniform inside each half but not across halves.
2. **Injector selection now biased.** `pick_injector` picks max-real-work
   in each half. With uniform iter count, real-work distribution still
   varies per core — but the injector is now doing the same number of
   forwards as everyone else. If real work is concentrated outside the
   chosen injectors, the injectors do extra phantom forwards that add
   NoC pressure for no compute. → Confirm with `log_debug`s for
   real-work distribution per half.
3. **NoC pressure from phantom mcast handshakes.** Phantom iters still
   do the mcast handshake (by design — keeps chains in lockstep). With
   two chains running concurrently, the extra phantom handshakes may
   compete on the *same* NoC the K-mcast's two-chain split was meant to
   relieve. → Try: skip the K-mcast handshake on phantom iters when
   *all* cores in that chain are phantom on that iter. Requires per-iter
   "any-core-real" predicate, derivable at program-factory time.
4. **`this_core_q_chunks = q_chunks_per_core` over-pads receivers.**
   Standalone 2-injector set the chain max via the per-core RT arg; the
   merge replaced that with the global uniform count. Receivers may now
   block on extra mcast receives that the injector also fires (so it
   should be balanced), but worth verifying via the kernel-side iter
   logs.

## Definition of done

- ✅ Accuracy: 3/3 PASS at `mla_100k` q160-k{160,256,320}.
- ✅ Perf: math util **60.9 %** at `mla_100k` q=160 k=320 (matches run 3).
- ✅ No new compute-core / iters-per-core / ring-eff regressions vs run 3
  (87/100, 280, 79 %, 0.0 %, 4.8 % all identical).

## Repo handles

- `b3e8eb62c64` — baseline, parent of the uniform-Q commit on `main`.
- `819a4910e9e` — uniform Q on `main` (current `HEAD`).
- `3ff8eb14481` — 2-injector standalone tip
  (`skrstic/sdpa-2-mcast-injectors`).
- `5242f1709ab` — naive combined cherry-pick
  (`temp/2injectors-plus-uniform-q`). Starting point for the
  perf-recovery work.
