# groupnorm_sc_N_1_HW_C / perf_experiments/reader_pass1_shadow — pass-1 reader schedule

box=bh-qb-11-special-mstaletovic-for-reservation-93463  arch=Arch.BLACKHOLE  metric=DEVICE KERNEL DURATION [ns] of the pass-1 reader + consumer-stub program (perf mode, zones compiled out); median of RPS_REPEATS fresh-launch runs per cell (each sample is one launch; device kernel time has no warm-up transient).

Pure dataflow reorder: every variant's x tiles / scaler / E^T pages are bit-identical to the baseline's (gated in check mode before every timing).

| case | cores | per-core block | variant | ns (median) | samples | speedup | r_pass1 end p50 / max us | r_x_barrier dur p50 / max us | r_memb_fill dur p50 / max us | r_zero_fill dur p50 / max us |
|---|---:|---|---|---:|---|---:|---|---|---|---|
| focus_1024x640_110c | 110 | Ht 4 x Ct 2, cols 2, chunk_rows 2, resident | baseline | 5537 | 5537, 5555, 5493 | 1.000x | 4.59 / 5.75 | 1.02 / 1.95 | 1.22 / 1.22 | - |
| focus_1024x640_110c | 110 | Ht 4 x Ct 2, cols 2, chunk_rows 2, resident | shadow | 4449 | 4440, 4449, 4488 | 1.245x | 3.46 / 4.66 | 0.47 / 1.21 | 0.94 / 0.94 | 0.41 / 0.41 |
| focus_1024x640_110c | 110 | Ht 4 x Ct 2, cols 2, chunk_rows 2, resident | shadow_trid | 4076 | 3993, 4121, 4076 | 1.358x | 3.43 / 4.35 | 0.29 / 1.09 | 0.94 / 0.95 | 0.42 / 0.44 |
| focus_1024x640_110c | 110 | Ht 4 x Ct 2, cols 2, chunk_rows 2, resident | shadow_fastlanes | 4198 | 4198, 4199, 4167 | 1.319x | 3.69 / 4.64 | 0.27 / 1.34 | 1.30 / 1.31 | 0.41 / 0.43 |
| focus_1024x640_110c | 110 | Ht 4 x Ct 2, cols 2, chunk_rows 2, resident | shadow_tightlanes | 4040 | 4021, 4040, 4041 | 1.371x | 3.00 / 4.12 | 0.44 / 0.99 | 0.52 / 0.54 | 0.41 / 0.42 |
| focus_1024x640_110c | 110 | Ht 4 x Ct 2, cols 2, chunk_rows 2, resident | shadow_lookahead | 3873 | 3858, 3873, 3895 | 1.430x | 3.26 / 4.19 | 0.20 / 1.16 | 0.94 / 0.94 | 0.42 / 0.43 |
| focus_1024x640_110c | 110 | Ht 4 x Ct 2, cols 2, chunk_rows 2, resident | shadow_all | 3833 | 3771, 3887, 3833 | 1.445x | 3.02 / 4.00 | 0.26 / 1.37 | 0.53 / 0.55 | 0.41 / 0.42 |
| floor_32x32_1c | 1 | Ht 1 x Ct 1, cols 1, chunk_rows 1, resident | baseline | 1400 | 1396, 1400, 1444 | 1.000x | - | - | - | - |
| floor_32x32_1c | 1 | Ht 1 x Ct 1, cols 1, chunk_rows 1, resident | shadow | 1046 | 1038, 1046, 1054 | 1.338x | - | - | - | - |
| floor_32x32_1c | 1 | Ht 1 x Ct 1, cols 1, chunk_rows 1, resident | shadow_trid | 1070 | 1075, 1067, 1070 | 1.308x | - | - | - | - |
| floor_32x32_1c | 1 | Ht 1 x Ct 1, cols 1, chunk_rows 1, resident | shadow_fastlanes | 1445 | 1445, 1445, 1453 | 0.969x | - | - | - | - |
| floor_32x32_1c | 1 | Ht 1 x Ct 1, cols 1, chunk_rows 1, resident | shadow_tightlanes | 955 | 955, 994, 952 | 1.466x | - | - | - | - |
| floor_32x32_1c | 1 | Ht 1 x Ct 1, cols 1, chunk_rows 1, resident | shadow_lookahead | 1111 | 1107, 1111, 1127 | 1.260x | - | - | - | - |
| floor_32x32_1c | 1 | Ht 1 x Ct 1, cols 1, chunk_rows 1, resident | shadow_all | 987 | 979, 996, 987 | 1.418x | - | - | - | - |
| manychunk_256x160_1c | 1 | Ht 8 x Ct 5, cols 5, chunk_rows 3, resident | baseline | 5834 | 5811, 5834, 5857 | 1.000x | - | - | - | - |
| manychunk_256x160_1c | 1 | Ht 8 x Ct 5, cols 5, chunk_rows 3, resident | shadow | 5609 | 5638, 5609, 5604 | 1.040x | - | - | - | - |
| manychunk_256x160_1c | 1 | Ht 8 x Ct 5, cols 5, chunk_rows 3, resident | shadow_trid | 5653 | 5653, 5750, 5647 | 1.032x | - | - | - | - |
| manychunk_256x160_1c | 1 | Ht 8 x Ct 5, cols 5, chunk_rows 3, resident | shadow_fastlanes | 6767 | 6776, 6767, 6761 | 0.862x | - | - | - | - |
| manychunk_256x160_1c | 1 | Ht 8 x Ct 5, cols 5, chunk_rows 3, resident | shadow_tightlanes | 5218 | 5214, 5225, 5218 | 1.118x | - | - | - | - |
| manychunk_256x160_1c | 1 | Ht 8 x Ct 5, cols 5, chunk_rows 3, resident | shadow_lookahead | 5301 | 5301, 5296, 5304 | 1.101x | - | - | - | - |
| manychunk_256x160_1c | 1 | Ht 8 x Ct 5, cols 5, chunk_rows 3, resident | shadow_all | 4906 | 4906, 4897, 4992 | 1.189x | - | - | - | - |
| manychunk_256x160_1c_stream | 1 | Ht 8 x Ct 5, cols 5, chunk_rows 3, streaming | baseline | 5710 | 5707, 5711, 5710 | 1.000x | - | - | - | - |
| manychunk_256x160_1c_stream | 1 | Ht 8 x Ct 5, cols 5, chunk_rows 3, streaming | shadow | 5507 | 5493, 5510, 5507 | 1.037x | - | - | - | - |
| manychunk_256x160_1c_stream | 1 | Ht 8 x Ct 5, cols 5, chunk_rows 3, streaming | shadow_trid | 5527 | 5527, 5516, 5563 | 1.033x | - | - | - | - |
| manychunk_256x160_1c_stream | 1 | Ht 8 x Ct 5, cols 5, chunk_rows 3, streaming | shadow_fastlanes | 6640 | 6640, 6679, 6630 | 0.860x | - | - | - | - |
| manychunk_256x160_1c_stream | 1 | Ht 8 x Ct 5, cols 5, chunk_rows 3, streaming | shadow_tightlanes | 5146 | 5146, 5090, 5199 | 1.110x | - | - | - | - |
| manychunk_256x160_1c_stream | 1 | Ht 8 x Ct 5, cols 5, chunk_rows 3, streaming | shadow_lookahead | 5528 | 5514, 5603, 5528 | 1.033x | - | - | - | - |
| manychunk_256x160_1c_stream | 1 | Ht 8 x Ct 5, cols 5, chunk_rows 3, streaming | shadow_all | 5095 | 5086, 5135, 5095 | 1.121x | - | - | - | - |
| rect4_256x256 | 4 | Ht 4 x Ct 4, cols 4, chunk_rows 2, resident | baseline | 3997 | 4056, 3987, 3997 | 1.000x | - | - | - | - |
| rect4_256x256 | 4 | Ht 4 x Ct 4, cols 4, chunk_rows 2, resident | shadow | 3723 | 3710, 3726, 3723 | 1.074x | - | - | - | - |
| rect4_256x256 | 4 | Ht 4 x Ct 4, cols 4, chunk_rows 2, resident | shadow_trid | 3773 | 3779, 3773, 3767 | 1.059x | - | - | - | - |
| rect4_256x256 | 4 | Ht 4 x Ct 4, cols 4, chunk_rows 2, resident | shadow_fastlanes | 4772 | 4751, 4772, 4778 | 0.838x | - | - | - | - |
| rect4_256x256 | 4 | Ht 4 x Ct 4, cols 4, chunk_rows 2, resident | shadow_tightlanes | 3269 | 3273, 3269, 3250 | 1.223x | - | - | - | - |
| rect4_256x256 | 4 | Ht 4 x Ct 4, cols 4, chunk_rows 2, resident | shadow_lookahead | 3734 | 3734, 3796, 3717 | 1.070x | - | - | - | - |
| rect4_256x256 | 4 | Ht 4 x Ct 4, cols 4, chunk_rows 2, resident | shadow_all | 3238 | 3236, 3238, 3248 | 1.234x | - | - | - | - |
| twocg_1024x1280_110c | 110 | Ht 4 x Ct 4, cols 2, chunk_rows 2, resident | baseline | 9498 | 9498, 9561, 9480 | 1.000x | - | - | - | - |
| twocg_1024x1280_110c | 110 | Ht 4 x Ct 4, cols 2, chunk_rows 2, resident | shadow | 7616 | 7804, 7616, 7544 | 1.247x | - | - | - | - |
| twocg_1024x1280_110c | 110 | Ht 4 x Ct 4, cols 2, chunk_rows 2, resident | shadow_trid | 6977 | 7214, 6977, 6967 | 1.361x | - | - | - | - |
| twocg_1024x1280_110c | 110 | Ht 4 x Ct 4, cols 2, chunk_rows 2, resident | shadow_fastlanes | 7418 | 7418, 7504, 7307 | 1.280x | - | - | - | - |
| twocg_1024x1280_110c | 110 | Ht 4 x Ct 4, cols 2, chunk_rows 2, resident | shadow_tightlanes | 7147 | 7147, 6952, 7234 | 1.329x | - | - | - | - |
| twocg_1024x1280_110c | 110 | Ht 4 x Ct 4, cols 2, chunk_rows 2, resident | shadow_lookahead | 7050 | 7184, 7006, 7050 | 1.347x | - | - | - | - |
| twocg_1024x1280_110c | 110 | Ht 4 x Ct 4, cols 2, chunk_rows 2, resident | shadow_all | 7065 | 7065, 7036, 7277 | 1.344x | - | - | - | - |
| threecg_64x768_1c | 1 | Ht 2 x Ct 24, cols 8, chunk_rows 2, resident | baseline | 16853 | 16887, 16853, 16823 | 1.000x | - | - | - | - |
| threecg_64x768_1c | 1 | Ht 2 x Ct 24, cols 8, chunk_rows 2, resident | shadow | 15998 | 16002, 15995, 15998 | 1.053x | - | - | - | - |
| threecg_64x768_1c | 1 | Ht 2 x Ct 24, cols 8, chunk_rows 2, resident | shadow_trid | 16046 | 16056, 16046, 16043 | 1.050x | - | - | - | - |
| threecg_64x768_1c | 1 | Ht 2 x Ct 24, cols 8, chunk_rows 2, resident | shadow_fastlanes | 20319 | 20319, 20328, 20319 | 0.829x | - | - | - | - |
| threecg_64x768_1c | 1 | Ht 2 x Ct 24, cols 8, chunk_rows 2, resident | shadow_tightlanes | 10490 | 10490, 10490, 10506 | 1.607x | - | - | - | - |
| threecg_64x768_1c | 1 | Ht 2 x Ct 24, cols 8, chunk_rows 2, resident | shadow_lookahead | 16082 | 16082, 16087, 16061 | 1.048x | - | - | - | - |
| threecg_64x768_1c | 1 | Ht 2 x Ct 24, cols 8, chunk_rows 2, resident | shadow_all | 10527 | 10517, 10527, 10536 | 1.601x | - | - | - | - |
