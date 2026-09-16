# groupnorm_sc_N_1_HW_C / perf_experiments/affine_lane_form — post-finalize region (stats bcast + affine + apply)

box=bh-qb-11-special-mstaletovic-for-reservation-93463  arch=Arch.BLACKHOLE  cores=1  placement=single-core sharded-L1 (pure compute)  metric=DEVICE KERNEL DURATION [ns], one fresh run per cell  iters=[1, 6]

Precision contract (fixed for every variant): fp32_dest_acc_en=True, HiFi4, approx=False, dst_full_sync=True; stats/E/stats_T/a/b fp32 pages, x/y/gamma/beta bf16.

`per-region ns` = (ns[iters_max] - ns[iters_min]) / (iters_max - iters_min): the launch-independent cost of one image's stats bcast + affine build + apply (all chunks). `bit-identical` / `max|dy|` = vs the `batched` variant (the op today); `err` = max|y - torch fp32 reference|.

| cell | cols | Kg | chunk_rows x chunks (Ht) | gamma/beta | variant | ns@iters=1 | ns@iters=6 | per-region ns | speedup | bit-identical / max|dy| | max err vs torch | mean err vs torch |
|---|---:|---:|---|---|---|---:|---:|---:|---:|---|---:|---:|
| focus | 2 | 1 | 2 x 2 (3) | 1/1 | batched | 3601 | 18533 | 2986 | 1.00x | True / 0.0e+00 | 8.2e-02 | 3.58e-03 |
| focus | 2 | 1 | 2 x 2 (3) | 1/1 | batched_blk | 3624 | 18654 | 3006 | 0.99x | True / 0.0e+00 | 8.2e-02 | 3.58e-03 |
| focus | 2 | 1 | 2 x 2 (3) | 1/1 | lane_fullab | 3725 | 19081 | 3071 | 0.97x | True / 0.0e+00 | 8.2e-02 | 3.58e-03 |
| focus | 2 | 1 | 2 x 2 (3) | 1/1 | lane_d2a | 3081 | 15360 | 2456 | 1.22x | True / 0.0e+00 | 8.2e-02 | 3.58e-03 |
| focus | 2 | 1 | 2 x 2 (3) | 1/1 | lane_accadd | 2950 | 14421 | 2294 | 1.30x | False / 6.2e-02 | 8.2e-02 | 3.53e-03 |
| focus | 2 | 1 | 2 x 2 (3) | 1/1 | lane_l1 | 3256 | 15899 | 2529 | 1.18x | True / 0.0e+00 | 8.2e-02 | 3.58e-03 |
| gamma=1 beta=0 | 2 | 1 | 2 x 2 (3) | 1/0 | batched | 3497 | 17992 | 2899 | 1.00x | True / 0.0e+00 | 8.0e-02 | 3.10e-03 |
| gamma=1 beta=0 | 2 | 1 | 2 x 2 (3) | 1/0 | batched_blk | 3501 | 17987 | 2897 | 1.00x | True / 0.0e+00 | 8.0e-02 | 3.10e-03 |
| gamma=1 beta=0 | 2 | 1 | 2 x 2 (3) | 1/0 | lane_fullab | 3850 | 19534 | 3137 | 0.92x | True / 0.0e+00 | 8.0e-02 | 3.10e-03 |
| gamma=1 beta=0 | 2 | 1 | 2 x 2 (3) | 1/0 | lane_d2a | 3184 | 15853 | 2534 | 1.14x | True / 0.0e+00 | 8.0e-02 | 3.10e-03 |
| gamma=1 beta=0 | 2 | 1 | 2 x 2 (3) | 1/0 | lane_accadd | 3042 | 14843 | 2360 | 1.23x | False / 6.2e-02 | 8.0e-02 | 3.15e-03 |
| gamma=1 beta=0 | 2 | 1 | 2 x 2 (3) | 1/0 | lane_l1 | 3330 | 16304 | 2595 | 1.12x | True / 0.0e+00 | 8.0e-02 | 3.10e-03 |
| gamma=0 beta=0 | 2 | 1 | 2 x 2 (3) | 0/0 | batched | 3458 | 17618 | 2832 | 1.00x | True / 0.0e+00 | 3.5e-02 | 2.90e-03 |
| gamma=0 beta=0 | 2 | 1 | 2 x 2 (3) | 0/0 | batched_blk | 3523 | 17681 | 2832 | 1.00x | True / 0.0e+00 | 3.5e-02 | 2.90e-03 |
| gamma=0 beta=0 | 2 | 1 | 2 x 2 (3) | 0/0 | lane_fullab | 3770 | 19076 | 3061 | 0.93x | True / 0.0e+00 | 3.5e-02 | 2.90e-03 |
| gamma=0 beta=0 | 2 | 1 | 2 x 2 (3) | 0/0 | lane_d2a | 3047 | 15370 | 2465 | 1.15x | True / 0.0e+00 | 3.5e-02 | 2.90e-03 |
| gamma=0 beta=0 | 2 | 1 | 2 x 2 (3) | 0/0 | lane_accadd | 2876 | 14386 | 2302 | 1.23x | False / 6.2e-02 | 3.5e-02 | 2.92e-03 |
| gamma=0 beta=0 | 2 | 1 | 2 x 2 (3) | 0/0 | lane_l1 | 3310 | 15921 | 2522 | 1.12x | True / 0.0e+00 | 3.5e-02 | 2.90e-03 |
| affine cols=1 Kg=1 | 1 | 1 | 2 x 2 (3) | 1/1 | batched | 2622 | 12096 | 1895 | 1.00x | True / 0.0e+00 | 5.6e-02 | 3.74e-03 |
| affine cols=1 Kg=1 | 1 | 1 | 2 x 2 (3) | 1/1 | batched_blk | 2627 | 12117 | 1898 | 1.00x | True / 0.0e+00 | 5.6e-02 | 3.74e-03 |
| affine cols=1 Kg=1 | 1 | 1 | 2 x 2 (3) | 1/1 | lane_fullab | 2659 | 12306 | 1929 | 0.98x | True / 0.0e+00 | 5.6e-02 | 3.74e-03 |
| affine cols=1 Kg=1 | 1 | 1 | 2 x 2 (3) | 1/1 | lane_d2a | 2337 | 10241 | 1581 | 1.20x | True / 0.0e+00 | 5.6e-02 | 3.74e-03 |
| affine cols=1 Kg=1 | 1 | 1 | 2 x 2 (3) | 1/1 | lane_accadd | 2203 | 9679 | 1495 | 1.27x | False / 6.2e-02 | 5.6e-02 | 3.71e-03 |
| affine cols=1 Kg=1 | 1 | 1 | 2 x 2 (3) | 1/1 | lane_l1 | 2417 | 10528 | 1622 | 1.17x | True / 0.0e+00 | 5.6e-02 | 3.74e-03 |
| affine cols=2 Kg=1 | 2 | 1 | 2 x 2 (3) | 1/1 | batched | 3696 | 18665 | 2994 | 1.00x | True / 0.0e+00 | 8.2e-02 | 3.58e-03 |
| affine cols=2 Kg=1 | 2 | 1 | 2 x 2 (3) | 1/1 | batched_blk | 3660 | 18592 | 2986 | 1.00x | True / 0.0e+00 | 8.2e-02 | 3.58e-03 |
| affine cols=2 Kg=1 | 2 | 1 | 2 x 2 (3) | 1/1 | lane_fullab | 3753 | 19187 | 3087 | 0.97x | True / 0.0e+00 | 8.2e-02 | 3.58e-03 |
| affine cols=2 Kg=1 | 2 | 1 | 2 x 2 (3) | 1/1 | lane_d2a | 3117 | 15412 | 2459 | 1.22x | True / 0.0e+00 | 8.2e-02 | 3.58e-03 |
| affine cols=2 Kg=1 | 2 | 1 | 2 x 2 (3) | 1/1 | lane_accadd | 2939 | 14487 | 2310 | 1.30x | False / 6.2e-02 | 8.2e-02 | 3.53e-03 |
| affine cols=2 Kg=1 | 2 | 1 | 2 x 2 (3) | 1/1 | lane_l1 | 3307 | 15990 | 2537 | 1.18x | True / 0.0e+00 | 8.2e-02 | 3.58e-03 |
| affine cols=4 Kg=1 | 4 | 1 | 2 x 2 (3) | 1/1 | batched | 5793 | 31225 | 5086 | 1.00x | True / 0.0e+00 | 8.3e-02 | 3.60e-03 |
| affine cols=4 Kg=1 | 4 | 1 | 2 x 2 (3) | 1/1 | batched_blk | 5765 | 31365 | 5120 | 0.99x | True / 0.0e+00 | 8.3e-02 | 3.60e-03 |
| affine cols=4 Kg=1 | 4 | 1 | 2 x 2 (3) | 1/1 | lane_fullab | 6020 | 32921 | 5380 | 0.95x | True / 0.0e+00 | 8.3e-02 | 3.60e-03 |
| affine cols=4 Kg=1 | 4 | 1 | 2 x 2 (3) | 1/1 | lane_d2a | 4825 | 25545 | 4144 | 1.23x | True / 0.0e+00 | 8.3e-02 | 3.60e-03 |
| affine cols=4 Kg=1 | 4 | 1 | 2 x 2 (3) | 1/1 | lane_accadd | 4497 | 23789 | 3858 | 1.32x | False / 1.2e-01 | 9.2e-02 | 3.59e-03 |
| affine cols=4 Kg=1 | 4 | 1 | 2 x 2 (3) | 1/1 | lane_l1 | 5098 | 27553 | 4491 | 1.13x | True / 0.0e+00 | 8.3e-02 | 3.60e-03 |
| affine cols=5 Kg=1 | 5 | 1 | 2 x 2 (3) | 1/1 | batched | 6810 | 37699 | 6178 | 1.00x | True / 0.0e+00 | 9.2e-02 | 3.45e-03 |
| affine cols=5 Kg=1 | 5 | 1 | 2 x 2 (3) | 1/1 | batched_blk | 6915 | 37683 | 6154 | 1.00x | True / 0.0e+00 | 9.2e-02 | 3.45e-03 |
| affine cols=5 Kg=1 | 5 | 1 | 2 x 2 (3) | 1/1 | lane_fullab | 7212 | 39864 | 6530 | 0.95x | True / 0.0e+00 | 9.2e-02 | 3.45e-03 |
| affine cols=5 Kg=1 | 5 | 1 | 2 x 2 (3) | 1/1 | lane_d2a | 5666 | 30600 | 4987 | 1.24x | True / 0.0e+00 | 9.2e-02 | 3.45e-03 |
| affine cols=5 Kg=1 | 5 | 1 | 2 x 2 (3) | 1/1 | lane_accadd | 5304 | 28516 | 4642 | 1.33x | False / 1.2e-01 | 8.5e-02 | 3.43e-03 |
| affine cols=5 Kg=1 | 5 | 1 | 2 x 2 (3) | 1/1 | lane_l1 | 6080 | 33484 | 5481 | 1.13x | True / 0.0e+00 | 9.2e-02 | 3.45e-03 |
| affine cols=8 Kg=1 | 8 | 1 | 2 x 2 (3) | 1/1 | batched | 9991 | 56846 | 9371 | 1.00x | True / 0.0e+00 | 9.0e-02 | 3.42e-03 |
| affine cols=8 Kg=1 | 8 | 1 | 2 x 2 (3) | 1/1 | batched_blk | 10057 | 56867 | 9362 | 1.00x | True / 0.0e+00 | 9.0e-02 | 3.42e-03 |
| affine cols=8 Kg=1 | 8 | 1 | 2 x 2 (3) | 1/1 | lane_fullab | 10657 | 60555 | 9980 | 0.94x | True / 0.0e+00 | 9.0e-02 | 3.42e-03 |
| affine cols=8 Kg=1 | 8 | 1 | 2 x 2 (3) | 1/1 | lane_d2a | 8162 | 45660 | 7500 | 1.25x | True / 0.0e+00 | 9.0e-02 | 3.42e-03 |
| affine cols=8 Kg=1 | 8 | 1 | 2 x 2 (3) | 1/1 | lane_accadd | 7619 | 42439 | 6964 | 1.35x | False / 1.2e-01 | 9.0e-02 | 3.41e-03 |
| affine cols=8 Kg=1 | 8 | 1 | 2 x 2 (3) | 1/1 | lane_l1 | 9099 | 51310 | 8442 | 1.11x | True / 0.0e+00 | 9.0e-02 | 3.42e-03 |
| affine cols=1 Kg=2 | 1 | 2 | 2 x 2 (3) | 1/1 | batched | 2814 | 13531 | 2143 | 1.00x | True / 0.0e+00 | 4.2e-02 | 3.50e-03 |
| affine cols=1 Kg=2 | 1 | 2 | 2 x 2 (3) | 1/1 | batched_blk | 2819 | 13530 | 2142 | 1.00x | True / 0.0e+00 | 4.2e-02 | 3.50e-03 |
| affine cols=1 Kg=2 | 1 | 2 | 2 x 2 (3) | 1/1 | lane_fullab | 2713 | 12770 | 2011 | 1.07x | True / 0.0e+00 | 4.2e-02 | 3.50e-03 |
| affine cols=1 Kg=2 | 1 | 2 | 2 x 2 (3) | 1/1 | lane_d2a | 2354 | 10851 | 1699 | 1.26x | True / 0.0e+00 | 4.2e-02 | 3.50e-03 |
| affine cols=1 Kg=2 | 1 | 2 | 2 x 2 (3) | 1/1 | lane_accadd | 2299 | 10374 | 1615 | 1.33x | False / 6.2e-02 | 4.2e-02 | 3.50e-03 |
| affine cols=1 Kg=2 | 1 | 2 | 2 x 2 (3) | 1/1 | lane_l1 | 2507 | 11228 | 1744 | 1.23x | True / 0.0e+00 | 4.2e-02 | 3.50e-03 |
| affine cols=2 Kg=2 | 2 | 2 | 2 x 2 (3) | 1/1 | batched | 4001 | 20693 | 3338 | 1.00x | True / 0.0e+00 | 5.0e-02 | 3.38e-03 |
| affine cols=2 Kg=2 | 2 | 2 | 2 x 2 (3) | 1/1 | batched_blk | 4007 | 20730 | 3345 | 1.00x | True / 0.0e+00 | 5.0e-02 | 3.38e-03 |
| affine cols=2 Kg=2 | 2 | 2 | 2 x 2 (3) | 1/1 | lane_fullab | 3918 | 20407 | 3298 | 1.01x | True / 0.0e+00 | 5.0e-02 | 3.38e-03 |
| affine cols=2 Kg=2 | 2 | 2 | 2 x 2 (3) | 1/1 | lane_d2a | 3310 | 16616 | 2661 | 1.25x | True / 0.0e+00 | 5.0e-02 | 3.38e-03 |
| affine cols=2 Kg=2 | 2 | 2 | 2 x 2 (3) | 1/1 | lane_accadd | 3131 | 15647 | 2503 | 1.33x | False / 6.2e-02 | 5.0e-02 | 3.39e-03 |
| affine cols=2 Kg=2 | 2 | 2 | 2 x 2 (3) | 1/1 | lane_l1 | 3536 | 17156 | 2724 | 1.23x | True / 0.0e+00 | 5.0e-02 | 3.38e-03 |
| affine cols=4 Kg=2 | 4 | 2 | 2 x 2 (3) | 1/1 | batched | 6337 | 34614 | 5655 | 1.00x | True / 0.0e+00 | 5.1e-02 | 3.37e-03 |
| affine cols=4 Kg=2 | 4 | 2 | 2 x 2 (3) | 1/1 | batched_blk | 6297 | 34684 | 5677 | 1.00x | True / 0.0e+00 | 5.1e-02 | 3.37e-03 |
| affine cols=4 Kg=2 | 4 | 2 | 2 x 2 (3) | 1/1 | lane_fullab | 6443 | 35336 | 5779 | 0.98x | True / 0.0e+00 | 5.1e-02 | 3.37e-03 |
| affine cols=4 Kg=2 | 4 | 2 | 2 x 2 (3) | 1/1 | lane_d2a | 5204 | 27934 | 4546 | 1.24x | True / 0.0e+00 | 5.1e-02 | 3.37e-03 |
| affine cols=4 Kg=2 | 4 | 2 | 2 x 2 (3) | 1/1 | lane_accadd | 4872 | 26251 | 4276 | 1.32x | False / 6.2e-02 | 5.1e-02 | 3.36e-03 |
| affine cols=4 Kg=2 | 4 | 2 | 2 x 2 (3) | 1/1 | lane_l1 | 5481 | 29926 | 4889 | 1.16x | True / 0.0e+00 | 5.1e-02 | 3.37e-03 |
| affine cols=5 Kg=2 | 5 | 2 | 2 x 2 (3) | 1/1 | batched | 7477 | 41585 | 6822 | 1.00x | True / 0.0e+00 | 8.0e-02 | 3.27e-03 |
| affine cols=5 Kg=2 | 5 | 2 | 2 x 2 (3) | 1/1 | batched_blk | 7509 | 41604 | 6819 | 1.00x | True / 0.0e+00 | 8.0e-02 | 3.27e-03 |
| affine cols=5 Kg=2 | 5 | 2 | 2 x 2 (3) | 1/1 | lane_fullab | 7677 | 42914 | 7047 | 0.97x | True / 0.0e+00 | 8.0e-02 | 3.27e-03 |
| affine cols=5 Kg=2 | 5 | 2 | 2 x 2 (3) | 1/1 | lane_d2a | 6221 | 33621 | 5480 | 1.24x | True / 0.0e+00 | 8.0e-02 | 3.27e-03 |
| affine cols=5 Kg=2 | 5 | 2 | 2 x 2 (3) | 1/1 | lane_accadd | 5801 | 31506 | 5141 | 1.33x | False / 1.2e-01 | 6.5e-02 | 3.26e-03 |
| affine cols=5 Kg=2 | 5 | 2 | 2 x 2 (3) | 1/1 | lane_l1 | 6574 | 36510 | 5987 | 1.14x | True / 0.0e+00 | 8.0e-02 | 3.27e-03 |
| affine cols=8 Kg=2 | 8 | 2 | 2 x 2 (3) | 1/1 | batched | 10941 | 62527 | 10317 | 1.00x | True / 0.0e+00 | 7.4e-02 | 3.18e-03 |
| affine cols=8 Kg=2 | 8 | 2 | 2 x 2 (3) | 1/1 | batched_blk | 10936 | 62489 | 10311 | 1.00x | True / 0.0e+00 | 7.4e-02 | 3.18e-03 |
| affine cols=8 Kg=2 | 8 | 2 | 2 x 2 (3) | 1/1 | lane_fullab | 11419 | 65240 | 10764 | 0.96x | True / 0.0e+00 | 7.4e-02 | 3.18e-03 |
| affine cols=8 Kg=2 | 8 | 2 | 2 x 2 (3) | 1/1 | lane_d2a | 8947 | 50400 | 8291 | 1.24x | True / 0.0e+00 | 7.4e-02 | 3.18e-03 |
| affine cols=8 Kg=2 | 8 | 2 | 2 x 2 (3) | 1/1 | lane_accadd | 8437 | 47330 | 7779 | 1.33x | False / 6.2e-02 | 7.4e-02 | 3.18e-03 |
| affine cols=8 Kg=2 | 8 | 2 | 2 x 2 (3) | 1/1 | lane_l1 | 9873 | 56084 | 9242 | 1.12x | True / 0.0e+00 | 7.4e-02 | 3.18e-03 |
| apply 1x1 | 1 | 1 | 1 x 2 (2) | 1/1 | batched | 2504 | 10684 | 1636 | 1.00x | True / 0.0e+00 | 4.6e-02 | 3.72e-03 |
| apply 1x1 | 1 | 1 | 1 x 2 (2) | 1/1 | batched_blk | 2510 | 10690 | 1636 | 1.00x | True / 0.0e+00 | 4.6e-02 | 3.72e-03 |
| apply 1x1 | 1 | 1 | 1 x 2 (2) | 1/1 | lane_fullab | 2559 | 10642 | 1617 | 1.01x | True / 0.0e+00 | 4.6e-02 | 3.72e-03 |
| apply 1x1 | 1 | 1 | 1 x 2 (2) | 1/1 | lane_d2a | 2116 | 8978 | 1372 | 1.19x | True / 0.0e+00 | 4.6e-02 | 3.72e-03 |
| apply 1x1 | 1 | 1 | 1 x 2 (2) | 1/1 | lane_accadd | 2075 | 8659 | 1317 | 1.24x | False / 6.2e-02 | 4.6e-02 | 3.71e-03 |
| apply 1x1 | 1 | 1 | 1 x 2 (2) | 1/1 | lane_l1 | 2304 | 9436 | 1426 | 1.15x | True / 0.0e+00 | 4.6e-02 | 3.72e-03 |
| apply 2x2 | 2 | 1 | 2 x 2 (4) | 1/1 | batched | 4135 | 20821 | 3337 | 1.00x | True / 0.0e+00 | 8.2e-02 | 3.60e-03 |
| apply 2x2 | 2 | 1 | 2 x 2 (4) | 1/1 | batched_blk | 4016 | 20685 | 3334 | 1.00x | True / 0.0e+00 | 8.2e-02 | 3.60e-03 |
| apply 2x2 | 2 | 1 | 2 x 2 (4) | 1/1 | lane_fullab | 4083 | 21245 | 3432 | 0.97x | True / 0.0e+00 | 8.2e-02 | 3.60e-03 |
| apply 2x2 | 2 | 1 | 2 x 2 (4) | 1/1 | lane_d2a | 3371 | 17140 | 2754 | 1.21x | True / 0.0e+00 | 8.2e-02 | 3.60e-03 |
| apply 2x2 | 2 | 1 | 2 x 2 (4) | 1/1 | lane_accadd | 3154 | 15944 | 2558 | 1.30x | False / 6.2e-02 | 8.2e-02 | 3.54e-03 |
| apply 2x2 | 2 | 1 | 2 x 2 (4) | 1/1 | lane_l1 | 3608 | 17779 | 2834 | 1.18x | True / 0.0e+00 | 8.2e-02 | 3.60e-03 |
| apply 4x4 | 4 | 1 | 4 x 2 (8) | 1/1 | batched | 9385 | 52716 | 8666 | 1.00x | True / 0.0e+00 | 8.3e-02 | 3.61e-03 |
| apply 4x4 | 4 | 1 | 4 x 2 (8) | 1/1 | batched_blk | 9327 | 52741 | 8683 | 1.00x | True / 0.0e+00 | 8.3e-02 | 3.61e-03 |
| apply 4x4 | 4 | 1 | 4 x 2 (8) | 1/1 | lane_fullab | 9616 | 54315 | 8940 | 0.97x | True / 0.0e+00 | 8.3e-02 | 3.61e-03 |
| apply 4x4 | 4 | 1 | 4 x 2 (8) | 1/1 | lane_d2a | 7556 | 41983 | 6885 | 1.26x | True / 0.0e+00 | 8.3e-02 | 3.61e-03 |
| apply 4x4 | 4 | 1 | 4 x 2 (8) | 1/1 | lane_accadd | 6753 | 37463 | 6142 | 1.41x | False / 1.2e-01 | 9.2e-02 | 3.60e-03 |
| apply 4x4 | 4 | 1 | 4 x 2 (8) | 1/1 | lane_l1 | 8641 | 48546 | 7981 | 1.09x | True / 0.0e+00 | 8.3e-02 | 3.61e-03 |
| apply 8x1 | 1 | 1 | 8 x 2 (16) | 1/1 | batched | 4998 | 26068 | 4214 | 1.00x | True / 0.0e+00 | 8.9e-02 | 3.87e-03 |
| apply 8x1 | 1 | 1 | 8 x 2 (16) | 1/1 | batched_blk | 4921 | 26050 | 4226 | 1.00x | True / 0.0e+00 | 8.9e-02 | 3.87e-03 |
| apply 8x1 | 1 | 1 | 8 x 2 (16) | 1/1 | lane_fullab | 4990 | 26184 | 4239 | 0.99x | True / 0.0e+00 | 8.9e-02 | 3.87e-03 |
| apply 8x1 | 1 | 1 | 8 x 2 (16) | 1/1 | lane_d2a | 4596 | 23933 | 3867 | 1.09x | True / 0.0e+00 | 8.9e-02 | 3.87e-03 |
| apply 8x1 | 1 | 1 | 8 x 2 (16) | 1/1 | lane_accadd | 4139 | 20973 | 3367 | 1.25x | False / 1.2e-01 | 8.2e-02 | 3.85e-03 |
| apply 8x1 | 1 | 1 | 8 x 2 (16) | 1/1 | lane_l1 | 4468 | 23327 | 3772 | 1.12x | True / 0.0e+00 | 8.9e-02 | 3.87e-03 |
| apply 1x8 | 8 | 1 | 1 x 2 (2) | 1/1 | batched | 8598 | 48127 | 7906 | 1.00x | True / 0.0e+00 | 9.0e-02 | 3.41e-03 |
| apply 1x8 | 8 | 1 | 1 x 2 (2) | 1/1 | batched_blk | 8564 | 48121 | 7911 | 1.00x | True / 0.0e+00 | 9.0e-02 | 3.41e-03 |
| apply 1x8 | 8 | 1 | 1 x 2 (2) | 1/1 | lane_fullab | 9217 | 51779 | 8512 | 0.93x | True / 0.0e+00 | 9.0e-02 | 3.41e-03 |
| apply 1x8 | 8 | 1 | 1 x 2 (2) | 1/1 | lane_d2a | 7079 | 39233 | 6431 | 1.23x | True / 0.0e+00 | 9.0e-02 | 3.41e-03 |
| apply 1x8 | 8 | 1 | 1 x 2 (2) | 1/1 | lane_accadd | 6760 | 37071 | 6062 | 1.30x | False / 6.2e-02 | 9.0e-02 | 3.40e-03 |
| apply 1x8 | 8 | 1 | 1 x 2 (2) | 1/1 | lane_l1 | 7664 | 42784 | 7024 | 1.13x | True / 0.0e+00 | 9.0e-02 | 3.41e-03 |
| apply 4x8 | 8 | 1 | 4 x 2 (8) | 1/1 | batched | 17148 | 99641 | 16499 | 1.00x | True / 0.0e+00 | 9.6e-02 | 3.40e-03 |
| apply 4x8 | 8 | 1 | 4 x 2 (8) | 1/1 | batched_blk | 17161 | 99763 | 16520 | 1.00x | True / 0.0e+00 | 9.6e-02 | 3.40e-03 |
| apply 4x8 | 8 | 1 | 4 x 2 (8) | 1/1 | lane_fullab | 17793 | 103201 | 17082 | 0.97x | True / 0.0e+00 | 9.6e-02 | 3.40e-03 |
| apply 4x8 | 8 | 1 | 4 x 2 (8) | 1/1 | lane_d2a | 13361 | 77050 | 12738 | 1.30x | True / 0.0e+00 | 9.6e-02 | 3.40e-03 |
| apply 4x8 | 8 | 1 | 4 x 2 (8) | 1/1 | lane_accadd | 12047 | 68739 | 11338 | 1.46x | False / 1.2e-01 | 9.0e-02 | 3.39e-03 |
| apply 4x8 | 8 | 1 | 4 x 2 (8) | 1/1 | lane_l1 | 16086 | 93338 | 15450 | 1.07x | True / 0.0e+00 | 9.6e-02 | 3.40e-03 |
