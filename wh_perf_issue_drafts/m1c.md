Part of {{UMBRELLA}}.

## Summary

The Wormhole math TRISC has a small instruction cache: 2 ways, 16-byte lines, 16 sets, so one way covers 256 bytes. If three hot code lines of a loop sit 256 bytes apart, they share one set and evict each other on every pass. The loop then reads code from L1 on every pass, and the kernel can be 27% slower. Which lines share a set depends only on the code address, so a change that does no work can create or remove the conflict.

The unpack and pack TRISCs have 64 sets (1 KiB per way), so a conflict needs three hot lines 1 KiB apart. We have seen it only on the math TRISC.

## Evidence

PR head 3ff45cbd1c7 with the barrier, math_matmul config936, MATH_ISOLATE (1 tile, Bfp8_b → Float16, LoFi). The layout pads of 3ff45cbd1c7 are on (the model's choice, math P = 224, Z = 20) or off.

| | Pads on | Pads off |
|---|---|---|
| Card (cycles) | 88,211 | 111,741 |
| Versim (cycles) | 88,211 | 111,741 |
| Cycles per loop pass (Versim) | 86.1 | 109.1 |
| Code lines read from L1 in 1,023 passes | 26 (first pass only) | 6,159 (6 per pass) |
| Instruction cache misses | 60 | 7,215 |
| Mispredicts | 8,183 | 8,185 |

Without the pads, three hot lines share set 12 (`0xa2c0`, `0xa3c0`, `0xa4c0`) and three share set 13 (`0xa2d0`, `0xa3d0`, `0xa4d0`). Each set keeps two of the three, so all six lines miss on every pass. The branch predictor is the same in both builds, so the difference is the instruction cache alone.

On the card, removing the pads moves 56 of 100 math_matmul MATH_ISOLATE configs by more than 0.5%, up to 26.7%.

## Where it shows

Problem 2. Mostly MATH_ISOLATE and L1_TO_L1 of math-heavy loops. Production kernels have the same exposure.

## Fix status in #58068

- 054096d8efa: the measured loop restarts from a 512 B aligned point, so a change outside the loop does not move the loop's lines against each other (the math sets repeat every 256 B).
- 3ff45cbd1c7: a model of the predictor and this cache picks NOP pads before and after the loop so that the hot lines do not conflict. This avoids the bad layout above. It does not add stability against code moves; it chooses a good layout once per build.
- A change inside the measured loop can still create a conflict; the pads are chosen again for the new code.

## Open

- The model must stay in sync with the real cache. Its geometry (2 ways, 16 sets on math, 64 on unpack and pack, 16-byte lines, least-recently-written replacement) matches the hardware.
