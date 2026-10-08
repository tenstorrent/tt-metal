Part of {{UMBRELLA}}.

## Summary

At #58068 head, the profiler zone helpers (`zone_reserve`, `zone_record`) sit in fixed, aligned slots in the harness code, so a change elsewhere cannot move them. But their own size still matters: one more instruction in `zone_reserve` moves 35 of 1,346 TILE_LOOP values by more than 2%, up to 17.8%. A harness change that touches the zone code is therefore not neutral for the numbers.

## Evidence (card, PR head)

Experiment switch `LLK_ZONE_RESERVE_NOPS=1` (branch `nstojictt/p58-versim`) adds one nop to the body of `zone_reserve`. 323 test cases, 1,346 values:

| Run type | Values that move > 2% |
|---|---|
| L1_CONGESTION[PACK] | 23 |
| L1_CONGESTION[UNPACK] | 7 |
| PACK_ISOLATE (sfpu_binop_scalar) | 5 |

With the settle fix of {{T3}} in place, the same change still moves 37 values (L1_CONGESTION 30, eltwise_binary MATH_ISOLATE 7, up to 14.1%), so it is a separate problem.

`zone_reserve` is exactly one 16-byte cache line. With one more instruction it spans two lines, so each thread reads one more code line from L1 just before its zone opens. In an earlier run, the same helper shifted by 4 bytes (still 16 bytes long, so also two lines) moved nine unpack_tilize PACK_ISOLATE values.

This is the same mechanism as {{T2}} and {{T3}}: a code-line read from L1 at the start of a measured loop selects the packer rhythm ({{M2}}).

## Fix status in #58068

df14044db9e moved both zone records out of the window, into the helpers. That removed an L1 store from every window (202 values, up to 112%), and it is correct to keep. The fragility is new with it.

## Possible fix

Keep the helper code that runs just before a zone opens in lines that are already in the cache, or read it once before the barrier invalidates the cache. Or give each helper a fixed size (padded slot), and check the perf values in CI when the helpers change.
