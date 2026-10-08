Part of {{UMBRELLA}}.

## Summary

On Wormhole, the math TRISC can mispredict a different number of times for the same code, if the loop starts a few cycles earlier or later. The cause is the branch-type cache in front of the branch predictor: it replaces entries at random, and the random source is a free-running LFSR. A change that does no work in another thread changes when the math loop starts, and the MATH_ISOLATE value moves by about 2%.

## Hardware

Before the branch predictor is used, the fetch unit checks a small cache that remembers which addresses hold a branch. It has 16 entries and is fully associative. On a miss it replaces a random entry. The random source is a 5-bit LFSR that steps on every clock (period 31), whether or not the core runs. The math matmul loop executes 68 different addresses, so it misses this cache all the time, and the entry that each miss replaces depends on the clock cycle of the miss.

## Evidence

MATH_ISOLATE, matmul 2×1, Float16_b → Float32. The math code is identical in both builds; the only change is one nop in the pack init (the pack thread does no work in MATH_ISOLATE).

| | 0 nops | +1 pack nop |
|---|---|---|
| Card (cycles) | 113,830 | 111,290 |
| Versim (cycles) | 111,290 | 113,822 |
| Versim: math loop start | cycle 46,897 | cycle 46,902 |
| Versim: LFSR at loop start | `11100` | `10001` |
| Versim: math mispredicts over the loop | 9,719 | 11,248 |
| Versim: branch-type cache misses | 15,744 | 15,746 |

- The card and Versim show the same two values, but swapped between the builds. The state is picked by a few cycles of start timing, and the card and Versim differ by a few cycles before the kernel starts.
- Card, 41 MATH_ISOLATE configs with +1 pack nop: pack predictor off → 1 moves; math predictor off → 0 move; all predictors off → 0 move.
- Full suite (CI, our stack): with all three predictors off, +1 and +3 nops move no MATH_ISOLATE, UNPACK_ISOLATE or PACK_ISOLATE point by more than 2%.

## Where it shows

Problem 2. MATH_ISOLATE, UNPACK_ISOLATE and L1_TO_L1 of matmul, about ±2–5%. Production kernels have the same exposure.

## Fix status in #58068

- 054096d8efa releases all threads together from a flushed pipeline, so the start cycle repeats for the same build. A change that moves the start by a few cycles can still select the other state.
- The LFSR runs on every clock, so a barrier cannot reset it.

## Open

- Turning the TRISC predictors off in perf builds removes it, but it shifts many values once and moves the numbers away from production. A diagnostic flag is a possible option.
- Hardware: replacement that does not depend on the clock cycle.
