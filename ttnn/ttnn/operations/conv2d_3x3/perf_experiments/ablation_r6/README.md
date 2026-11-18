# conv2d_3x3 ablation hooks (Perf 6)

`ablation_hooks.patch` applies to `5f23b256d1a` (the round-6 base, end of Perf 5). It is the round-5 patch rebased onto
the Perf 5 graduations, plus hooks the round-5 patch could not have:

- `fill` also stubs the writer's windowed / BLOCK split fill reads (`fill_channel_slice`, `fill_one`);
- `scatter` stubs the ring reduce-scatter's forwarding writes (`noc_async_write_one_packet_with_trid` in the reader);
  the tokens and trid retire still run;
- `gather` also stubs the planned tap-direct gather (`TapDirectPlanned::unit`): every tap entry points at the unit's
  own slot;
- `matmul` also empties the raw ring walk's `ckernel::matmul_block` k loop (`SKIP_COMPUTE` does not reach it).

The first DM-first peel ran without the last two hooks and left the BLOCK cases at 31-58 us with "everything" stubbed;
the `dm2_*` rows in `peel_results.txt` are the re-run with them. HEIGHT numbers did not change (neither hook is on the
HEIGHT path).

```bash
git apply ttnn/ttnn/operations/conv2d_3x3/perf_experiments/ablation_r6/ablation_hooks.patch
ttnn/ttnn/operations/conv2d_3x3/perf_experiments/ablation_r6/run_peel.sh <label> gather,fill,weights
git apply -R ttnn/ttnn/operations/conv2d_3x3/perf_experiments/ablation_r6/ablation_hooks.patch
```

`peel_results.txt`: DEVICE KERNEL DURATION (us, zoned) per focus case in `test_conv2d_3x3_perf_focus` order:
stem1, stem2, stem3, res2, res3.0, res3.x, res4.0, res4.x, res5.0, res5.1, res5.2, aspp1, aspp2, aspp3,
sem dec res3 fc0, inst dec res3 fc0, sem dec res3 fc1, sem dec res2 fc0, inst dec res2 fc0, sem res2 fc1,
inst res2 fc1, center_head.1. `zone_compact.py <zone_report output>` prints one line per case.
