# Groupnorm format follow-up — 2026-09-09

After the first review's reconfiguration-mode corrections, independent targeted
checks exposed a regression in small sharded groupnorm reductions. At source
`1005cb6d975`, T028's legacy 8x4 and all-configuration selection produced **25
failed and 26 passed** cases (`reduce-migration-gi7dvima`). The failures were
numerical, across BF16/FP32 inputs, BF16/FP32 affine tensors and row-major/tiled
layouts; a representative case had PCC 0.805220 and relative Frobenius error 0.703.

The existing sharded masking stage leaves input/mask unpack formats active. Its
new mean reduction may select native ReduceTile for fewer than eight input tiles.
Applying the old native calls' NONE reconfiguration policy to this new call was
incorrect: the former manual accumulation had established the intermediate
operand formats before its reduction. Add plans already configure their own
operands, explaining why larger cases and the prior sanity selection passed.

The host helper now accepts an explicit first-native-call mode, and the sharded
factory requests INPUT. The output pack format is already the intermediate
format shared by the masked tensor and local statistics. Existing native calls
retain their caller-owned format behavior; Add retains its required reconfiguration.

SM006 now selects the existing four-core sharded BF16 case with a BF8 mask and
FP32 intermediates, whose four-tile group exercises native mean reduction. The
previous one-core C++ smoke used compatible operand formats and missed this
transition. It remains in full group T159. Sanity stays at **75 cases** (60 Python,
15 C++), including **61 N300 cases**; the full suite remains **178 groups / 18,844
known cases**. No tolerance, test body or full-suite selection was weakened.

Reproduction and post-fix verification:
```
python3 scripts/run_reduce_migration_tests.py --group T028 -- -k '(test_group_norm_with_block_sharded_v2_8x4_grid or test_group_norm_sharded_all_config) and legacy'
```

Build: `cmake --build build --target ttnn unit_tests_ttnn --parallel 8`, log
`/tmp/reduce-groupnorm-format-fix-build-20260909.log`, **passed**. The runtime
extension was refreshed atomically. The same unmodified T028 selection then
passed **51/51**, no skips (`reduce-migration-n9526jxz`), including all 25 cases
that failed before the fix.
SM005 and the replacement SM006 both passed (`reduce-migration-n1bwzklv`).

Review history: round 2 stopped without a verdict when its process disappeared.
Round 3 was deliberately stopped after these independent checks found the defect;
it also has no verdict. Their transcripts remain in generated/reduce_migration_reviews.
A fresh Opus 5/high review will follow the validated fix.
