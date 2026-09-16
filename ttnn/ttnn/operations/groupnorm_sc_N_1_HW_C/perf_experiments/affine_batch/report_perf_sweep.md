# groupnorm_sc_N_1_HW_C / perf_experiments/affine_batch — pass-2 affine build per column group

box=bh-qb-11-special-mstaletovic-for-reservation-93463  arch=Arch.BLACKHOLE  cores=1  placement=single-core sharded-L1 (pure compute)  metric=DEVICE KERNEL DURATION [ns], one fresh run per cell  iters=[1, 11]

Precision contract (fixed for every variant): fp32_dest_acc_en=True, HiFi4, approx=False, dst_full_sync=True; stats/E fp32 pages, gamma/beta bf16 row tiles, a/b fp32 full tiles.

`per-group ns` = (ns[iters_max] - ns[iters_min]) / (iters_max - iters_min): the launch-independent cost of one column group's affine build. `per-tile ns` = per-group / cols. `bit-identical` = vs the baseline variant's output.

| cols | Kg | variant | ns@iters=1 | ns@iters=11 | per-group ns | per-tile ns | speedup (per-group) | a bit-identical / max|da| | b bit-identical / max|db| |
|---:|---:|---|---:|---:|---:|---:|---:|---|---|
| 1 | 1 | baseline | 1585 | 11835 | 1025 | 1025 | 1.00x | True / 0.0e+00 | True / 0.0e+00 |
| 1 | 1 | batched | 1764 | 11787 | 1002 | 1002 | 1.02x | True / 0.0e+00 | True / 0.0e+00 |
| 1 | 1 | batched_fold | 1586 | 11798 | 1021 | 1021 | 1.00x | True / 0.0e+00 | False / 1.6e-03 |
| 1 | 1 | fused | 1484 | 11978 | 1049 | 1049 | 0.98x | True / 0.0e+00 | False / 7.0e-03 |
| 1 | 1 | fused_sfpu | 1582 | 12135 | 1055 | 1055 | 0.97x | True / 0.0e+00 | False / 4.5e-03 |
| 2 | 1 | baseline | 2509 | 22152 | 1964 | 982 | 1.00x | True / 0.0e+00 | True / 0.0e+00 |
| 2 | 1 | batched | 2108 | 17850 | 1574 | 787 | 1.25x | True / 0.0e+00 | True / 0.0e+00 |
| 2 | 1 | batched_fold | 2101 | 18433 | 1633 | 817 | 1.20x | True / 0.0e+00 | False / 1.6e-03 |
| 2 | 1 | fused | 2254 | 19761 | 1751 | 875 | 1.12x | True / 0.0e+00 | False / 6.2e-03 |
| 2 | 1 | fused_sfpu | 2225 | 19844 | 1762 | 881 | 1.11x | True / 0.0e+00 | False / 4.5e-03 |
| 4 | 1 | baseline | 4297 | 41653 | 3736 | 934 | 1.00x | True / 0.0e+00 | True / 0.0e+00 |
| 4 | 1 | batched | 3117 | 29627 | 2651 | 663 | 1.41x | True / 0.0e+00 | True / 0.0e+00 |
| 4 | 1 | batched_fold | 3256 | 30810 | 2755 | 689 | 1.36x | True / 0.0e+00 | False / 1.8e-03 |
| 4 | 1 | fused | 3519 | 34664 | 3114 | 779 | 1.20x | True / 0.0e+00 | False / 1.1e-02 |
| 4 | 1 | fused_sfpu | 3609 | 34736 | 3113 | 778 | 1.20x | True / 0.0e+00 | False / 3.4e-03 |
| 5 | 1 | baseline | 5160 | 51465 | 4630 | 926 | 1.00x | True / 0.0e+00 | True / 0.0e+00 |
| 5 | 1 | batched | 3723 | 35304 | 3158 | 632 | 1.47x | True / 0.0e+00 | True / 0.0e+00 |
| 5 | 1 | batched_fold | 3810 | 37053 | 3324 | 665 | 1.39x | True / 0.0e+00 | False / 1.8e-03 |
| 5 | 1 | fused | 4280 | 42010 | 3773 | 755 | 1.23x | True / 0.0e+00 | False / 1.1e-02 |
| 5 | 1 | fused_sfpu | 4255 | 42224 | 3797 | 759 | 1.22x | True / 0.0e+00 | False / 3.4e-03 |
| 8 | 1 | baseline | 7892 | 80802 | 7291 | 911 | 1.00x | True / 0.0e+00 | True / 0.0e+00 |
| 8 | 1 | batched | 5223 | 52668 | 4744 | 593 | 1.54x | True / 0.0e+00 | True / 0.0e+00 |
| 8 | 1 | batched_fold | 5496 | 55480 | 4998 | 625 | 1.46x | True / 0.0e+00 | False / 2.8e-03 |
| 8 | 1 | fused | 6282 | 64470 | 5819 | 727 | 1.25x | True / 0.0e+00 | False / 2.4e-02 |
| 8 | 1 | fused_sfpu | 6296 | 64787 | 5849 | 731 | 1.25x | True / 0.0e+00 | False / 6.4e-03 |
| 1 | 2 | baseline | 1745 | 13207 | 1146 | 1146 | 1.00x | True / 0.0e+00 | True / 0.0e+00 |
| 1 | 2 | batched | 1832 | 13235 | 1140 | 1140 | 1.01x | True / 0.0e+00 | True / 0.0e+00 |
| 1 | 2 | batched_fold | 1636 | 13045 | 1141 | 1141 | 1.00x | True / 0.0e+00 | False / 7.8e-04 |
| 1 | 2 | fused | 1563 | 13259 | 1170 | 1170 | 0.98x | True / 0.0e+00 | False / 8.6e-03 |
| 1 | 2 | fused_sfpu | 1617 | 13357 | 1174 | 1174 | 0.98x | True / 0.0e+00 | False / 2.7e-03 |
| 2 | 2 | baseline | 2766 | 24764 | 2200 | 1100 | 1.00x | True / 0.0e+00 | True / 0.0e+00 |
| 2 | 2 | batched | 2263 | 20186 | 1792 | 896 | 1.23x | True / 0.0e+00 | True / 0.0e+00 |
| 2 | 2 | batched_fold | 2343 | 20790 | 1845 | 922 | 1.19x | True / 0.0e+00 | False / 9.4e-04 |
| 2 | 2 | fused | 2410 | 21948 | 1954 | 977 | 1.13x | True / 0.0e+00 | False / 8.6e-03 |
| 2 | 2 | fused_sfpu | 2422 | 22083 | 1966 | 983 | 1.12x | True / 0.0e+00 | False / 2.7e-03 |
| 4 | 2 | baseline | 4750 | 46723 | 4197 | 1049 | 1.00x | True / 0.0e+00 | True / 0.0e+00 |
| 4 | 2 | batched | 3497 | 33971 | 3047 | 762 | 1.38x | True / 0.0e+00 | True / 0.0e+00 |
| 4 | 2 | batched_fold | 3621 | 35165 | 3154 | 789 | 1.33x | True / 0.0e+00 | False / 1.9e-03 |
| 4 | 2 | fused | 3949 | 39070 | 3512 | 878 | 1.20x | True / 0.0e+00 | False / 8.6e-03 |
| 4 | 2 | fused_sfpu | 4061 | 39278 | 3522 | 880 | 1.19x | True / 0.0e+00 | False / 5.1e-03 |
| 5 | 2 | baseline | 5775 | 57695 | 5192 | 1038 | 1.00x | True / 0.0e+00 | True / 0.0e+00 |
| 5 | 2 | batched | 4162 | 40716 | 3655 | 731 | 1.42x | True / 0.0e+00 | True / 0.0e+00 |
| 5 | 2 | batched_fold | 4328 | 42555 | 3823 | 765 | 1.36x | True / 0.0e+00 | False / 1.9e-03 |
| 5 | 2 | fused | 4721 | 47550 | 4283 | 857 | 1.21x | True / 0.0e+00 | False / 8.6e-03 |
| 5 | 2 | fused_sfpu | 4802 | 47779 | 4298 | 860 | 1.21x | True / 0.0e+00 | False / 5.1e-03 |
| 8 | 2 | baseline | 8805 | 91065 | 8226 | 1028 | 1.00x | True / 0.0e+00 | True / 0.0e+00 |
| 8 | 2 | batched | 5996 | 61373 | 5538 | 692 | 1.49x | True / 0.0e+00 | True / 0.0e+00 |
| 8 | 2 | batched_fold | 6252 | 64110 | 5786 | 723 | 1.42x | True / 0.0e+00 | False / 2.6e-03 |
| 8 | 2 | fused | 7039 | 73043 | 6600 | 825 | 1.25x | True / 0.0e+00 | False / 1.1e-02 |
| 8 | 2 | fused_sfpu | 7073 | 73461 | 6639 | 830 | 1.24x | True / 0.0e+00 | False / 4.7e-03 |
