# weight_stream_decouple_v2 (Perf 17)

This ports Perf 16's `weight_stream_decouple` onto the tree that has the bank-run fill and resident weights. It only
matters where the weights still stream, i.e. where [K, N] does not fit in L1 (here: C_in 128/256 -> 512).

## Files

- `op_base/`: an untouched copy of the working-tree op.
- `op_copy/`: the experiment copy. Every knob is a module constant and defaults to off:
  - `WSD_KROT`: () / "all" / a tuple / "auto" / "auto2"
  - `WSD_SPLIT`, `WSD_SPLIT_TILE`, `WSD_FILL_LEAD`
  - `WSD_DIAG`, `WSD_NOREAD`: debug only
- `op_final/`: the integration candidate. It is `op_copy` minus the debug knobs, with the recommended settings as
  defaults (`WSD_KROT="auto"`, which is `op_copy`'s "auto2" rule; `WSD_SPLIT=True`; `WSD_FILL_LEAD="auto"`).
- `integration.patch`: `diff -ru op_base op_final`.
- `test_wsd2.py` + `run.sh <label> <cases> <variants> [ENV=..]`: the A/B harness.
  - Each variant is signposted `cmp|case|variant|i`, and `runs/<label>.txt` holds the median device ns.
  - `runs/<label>.jsonl` holds per variant: sha1, PCC vs torch, ulp diff vs the first variant, and the blocking.
  - `WSD_REPEAT=n` makes n passes in alternating order.
  - `run.sh` gives up while still waiting for the device lock (exit 75), and stops a run that is still going after
    `WSD_DEADLINE` seconds.
- `test_panoptic_guard.py`: every non-FLUX LOOSE case, op_base vs op_final, checked bit-exactly
  (`WSD_TEST=<it> WSD_NOPROFILE=1 run.sh pan 0 -`).

## Mechanism

The R1 weight multicast sends one round per K-block. `cb_weights` has depth 2, and every receiver acks every round, so
all ~108 cores stay within one round of each other. As a result they all reach their M-block boundary in the same
window: they store the block and refill the slab together. That DRAM burst stalls the stream for everyone. In the base
profile, P7 `compute_bias_add` waits 357 us on the output CB while the writer sits blocked in a weight receive.

- **Split-phase rounds (`WSD_SPLIT`).** While the store loop polls the output CB, the writer:
  - reserves a slot and acks it;
  - pushes a round once its flag lands;
  - on the sender, sends only once every ack is in.

  A store therefore never waits behind a round. On its own it is bit-identical and worth 1-2.5%.
- **K rotation (`WSD_KROT`).** Channel participant i walks every M-block's K-blocks starting at offset o_i, so it takes
  stream rounds o_i + 9m .. o_i + 9m + 8. The rounds before its first one and after its last one are popped unused.
  This staggers the M-block boundaries across the K cycle. The cost is `max(o)` extra rounds at compute pace. It is not
  bit-identical, because the fp32 L1-accumulation order changes.
- **Staggered first fill (`WSD_FILL_LEAD`).** A core with offset o starts its first slab fill only once its writer has
  issued round o - 1.

Rule "auto", measured:
- bk <= 4: 4 groups, spaced min(2, M - 1) apart, with lead 1 at spacing 2. No rotation at M = 1.
- bk > 4: offsets (0, 1, 2) from M >= 3. Split only below that.

## Results (Blackhole p150b, 110-core grid, bf16, HiFi4, fp32 DEST, bias, padding (0, 0, 1, 1), ROW_MAJOR DRAM; device kernel ns)

| case | M, bk | base | op_final | change |
|---|---|---|---|---|
| idx43 P7 258x512 128->512 | 5, 4 | 1681259 | 1454400 | -13.5% |
| idx47 P13 514x1024 128->512 | 17, 4 | 6389531 | 4944213 | -22.6% |
| idx38 P10 130x256 128->512 | 2, 4 | 445414 (med 3) | 419028 (med 3) | -5.9% |
| idx34 P15 66x128 128->512 | 1, 12 | 139487 | 140173 | flat (bit-identical) |
| idx31 / idx32 (M = 1) | 1, 3 | 25919 / 73996 | 25989 / 74560 | flat (bit-identical) |
| idx36/37/39/40 (resident) | - | - | - | flat, bit-identical (program unchanged) |
| 258x512 256->512 | 8, 8 | 2921688 | 2646984 | -9.4% |
| 130x256 256->512 | 2, 8 | 762607 | 745316 | -2.3% (bit-identical, split only) |
| 164x512 128->512 | 3, 4 | 1042008 | 957411 | -8.1% |
| 218x512 128->512 | 4, 4 | 1402764 | 1214833 | -13.4% |
| 98x512 256->512 | 3, 8 | 1126704 | 1067287 | -5.3% |
| 130x512 256->512 | 4, 8 | 1484133 | 1380961 | -7.0% |
| 1024x1024 8->128 pad 1 (single-device RM guard) | resident | 9335009 | 9313769 | flat, bit-identical |

- **Precision with rotation:** 0.002-0.003% of outputs differ, max |diff| 0.0156 (1 bf16 ulp at |x| in [2, 4)). PCC vs
  torch matches the base's to 1e-9 (0.9999985).
- **L1:** no change. The blocking solve is identical on every case (same bm, bk, depth and footprint).
- **Guards:**
  - Panoptic: 45/45 non-FLUX LOOSE outputs are bit-identical. All are sharded, and both changes are gated off for
    sharded input or output.
  - TILE input: not reached. Rotation needs ROW_MAJOR; split is ROW_MAJOR-only unless `WSD_SPLIT_TILE`.
