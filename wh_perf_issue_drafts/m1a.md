Part of {{UMBRELLA}}.

## Summary

On Wormhole, the pack TRISC loop runs at a speed that depends on the address of its branches. If two branches of the loop map to the same branch-predictor entry, both branches mispredict on every pass. When the pack core is the slower side, the measured value goes up by about 1 cycle per tile. A change that does no work (one nop that moves the loop by 4 bytes) can select the slow layout.

## Hardware

The TRISC branch predictor has 16 entries and no tag. The entry index is a hash of address bits 2 to 8 (we read the hash from the waveform: entry = `pc[2] | ~(pc[3]^pc[4])<<1 | (pc[5]^pc[7])<<2 | ~(pc[6]^pc[8])<<3`). An entry keeps a 2-bit counter and the last next address. A not-taken branch also counts as a mispredict when the stored next address is not `pc + 4`. Only a core reset clears the table.

## Evidence

Repro: one PACK_ISOLATE matmul config, 256 loop passes, the other threads idle, one nop added in `_llk_pack_init_`.

| Build | Inner-loop branch | Outer-loop branch | Entry writes (256 passes) | Card | Versim |
|---|---|---|---|---|---|
| 0 nops | `0xf1e4` → entry 11 | `0xf1ec` → entry 9 | 1 each | 9,070 | 9,070 |
| +1 nop | `0xf1ec` → entry 9 | `0xf1f4` → entry 9 | 256 each | 9,317 | 9,313 |

- Versim, over 240 passes: 0 mispredicts and 5,971 cycles of a full instruction queue (fast), against 480 mispredicts and 0 cycles of a full queue (slow). In the slow build the pack core no longer keeps the packers busy.
- Card: from the branch addresses of 53 builds, the hash above predicted the fast or slow result for all 53.
- Card: `DISABLE_RISC_BP_Disable_trisc` bit 2 (pack) makes the slow build fast (9,317 → 9,063). Bits 0 and 1 have no effect. The bit stays set after a run; write it at every boot.
- perf_pack pad 0 / pad 9: card and Versim 3,286 / 3,664; four loop branches share entry 12 in the slow build (313 against 126 mispredicts).
- Full suite (CI, #58068 before the barrier): one nop moved 14,750 PACK_ISOLATE points by more than 2%. With the pack predictor off: 3,415.

## Where it shows

Problem 2 only (a change that does no work moves values). The same build always gives the same value. PACK_ISOLATE and L1_TO_L1 when the pack core is the bottleneck. Production kernels have the same exposure.

## Fix status in #58068

- 054096d8efa (barrier restart from a 512 B aligned point, INIT out of line): the loop keeps its address mod 512, so a change outside the loop cannot change the hash. A change inside the loop still can.
- The predictor keeps its contents across the barrier (reset only). We found no case where this changes a value.
- Layout pads (3ff45cbd1c7) are not used on pack threads.

## Open

- A change inside the measured pack loop can still select the slow layout. This is a real code change, but the speed difference is not in the code.
- Hardware: a tagged or larger predictor would remove the aliasing.
