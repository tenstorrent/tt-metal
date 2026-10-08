## Purpose

This issue collects every mechanism we found that makes Wormhole LLK perf measurements unstable, bistable or wrong. Each mechanism has its own issue with the hardware cause, the proof in Versim (the Tensix RTL) and on the card, and the fix status. PRs that fix a mechanism should reference its issue.

Earlier reports: #58919 (code changes that do no work move results), #55169 (matmul TILE_LOOP bistability). The harness fixes are in #58068.

## Two problems, two mechanisms

**Problem 1. The same build gives a different value when we measure it again.** On main, a full Wormhole run gives the same value on rerun for only 68.7% of points.

**Problem 2. A change that does no work moves values.** One nop in `_llk_pack_init_` moved 14,750 PACK_ISOLATE points by more than 2% (full suite).

**M1. The code address of a TRISC loop sets its speed.** Three units of the TRISC front end see the address: the branch predictor, the branch-type cache and the instruction cache.

**M2. The four packers lock into a fast or a slow rhythm** on the DEST read crossbar. Small timing events while the packers run select the rhythm. A trigger at a random cycle gives problem 1; a trigger that moves with the code gives problem 2.

Blackhole shows neither M1 nor M2 (one packer; 0 TILE_LOOP points move with nops).

## Mechanisms

| Issue | Mechanism | Problem | Versim | Card | Status at #58068 head |
|---|---|---|---|---|---|
| {{M2}} | M2. Packer rhythm on the DEST read crossbar (fixed priority) | both | yes | yes | not changed; the pushes below are fixed one by one |
| {{M1A}} | M1a. Pack branch predictor aliasing | 2 | yes | yes | fixed for code outside the loop (barrier) |
| {{M1B}} | M1b. Branch-type cache, random replacement | 2 | yes (values swapped) | yes | partly: same start cycle per build (barrier) |
| {{M1C}} | M1c. Math instruction-cache set conflicts | 2 | yes | yes | avoided by layout pads |
| {{T6}} | L1 accesses during packing (host, BRISC, zone record) | 1 and 2 | yes (hand-placed read); random case not possible | yes | fixed (quiet L1, zone records) |
| {{T2}} | Pack loop's first tiles: own code fetch, instruction timing | 2 | yes | yes | fixed for code outside the loop (barrier) |
| {{T3}} | Idle threads run exit code in isolate windows | 2 | yes | yes | **open**; fix tested on the card |
| {{ZONE}} | Size of the profiler zone helpers | 2 | – | yes | **open** |
| {{T7}} | State left by the previous kernel | 1 (order) | runs did not finish | yes | fixed at head; warm-up not needed |

## How we test

- Card: one Wormhole lab card. Versim: `versim-wormhole-b0`, the same ELF as the card. Every quiet Versim run in these issues gives the card's cycle count.
- "A change that does no work": `-fpatchable-function-entry=N,N` puts N never-executed nops in front of every function (all threads, or one thread only).
- Switches for every experiment are on branch `nstojictt/p58-versim` (PR head 3ff45cbd1c7 plus `LLK_*` environment switches). The PR branch does not change.
- Full write-up with waveforms (internal): {{PAGE}}

## Review of the #58068 changes

We switched each change off at PR head, and tried to break the rest with changes that do no work (323 test cases, 1,346 TILE_LOOP values; "moves" = values that change by more than 2%).

| Change | Verdict | Why |
|---|---|---|
| 054096d8efa barrier restart, out-of-line INIT | keep, incomplete | Barrier off: 4 bytes in front of every function move 124 of 736 values; on: 36. It does not stop the idle threads of isolate run types ({{T3}}). |
| b58fbfbd6d2 host polls overlay registers | keep | Without it, identical runs differ by up to 27% ({{T6}}). |
| e02b20f56ee BRISC polls every 100 µs on NOPs | keep | Removes BRISC L1 reads from the window (54 of 736 values move once without it). The commit text should say "bias", not "noise": identical runs agree without it. |
| df14044db9e zone records after the end read | keep, fragile | Removes an L1 store from every window (202 values, up to 112%). One more instruction in `zone_reserve` moves 35 values ({{ZONE}}). |
| 3ff45cbd1c7 layout pads | keep | Without pads, a math_matmul MATH_ISOLATE config runs 27% slower from instruction-cache set conflicts, the same on the card and in Versim ({{M1C}}). The pads do not add stability against code moves. |
| 9a4902e0092, c015600bf22 fixed kernel address | remove | Redundant with the barrier: without it no value moves by more than 0.9%, and harness growth of 16 or 384 bytes moves nothing. Costs up to 2 KiB of code space. |
| b7747df1b7d, 75669afc17a warm-up pass | remove | Without it, 0 of 1,346 values move, in forward and in reverse test order ({{T7}}). |
| 73e829efa99 / c16630eb8a0 packer drain | no net change | Added, then removed. |

With the T3 fix tested in {{T3}} (a settle before the measured zone), code moves of all threads leave no isolate value moving: 75 → 25 moves, all 25 in L1_TO_L1 and L1_CONGESTION (up to 6.2%), where all threads really run.

Attacks that did not break PR head: rerun (0 moves), reverse order (0), warm-up off (0), all BRISC functions moved by 4 bytes (0), pack code moved by 4 bytes per function (3, L1_TO_L1 only).

Not covered here: the Blackhole-only commits (11e32e336c8, 30934efd06a, 53b93fd7e5d, d6bec175114) and Quasar (ff6d697921a).

**Other note.** The branch predictor keeps its contents across the barrier (only a reset clears it), so "every zone starts from the same state" is not exactly true. We found no value that changes because of it (with the pack predictor off, the same values move).
