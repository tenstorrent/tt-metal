# DiT matmul blocking sweep + E2E gates, 2026-09-22, host DC16-2-BG-0203-u13-43 (BH Galaxy 4x8 ring)

`sweep_results_mm.csv`: every L1-feasible (core_grid, M_block, K_block, N_block, subblock) combo
for the eleven matmul shapes the 5B DiT requests at M=2336 (720p/81f) and M=1024 (480p/81f),
device kernel time per combo. Produced by `models/tt_dit/utils/sweep_mm_block_sizes.py`;
winners are registered in `models/tt_dit/pipelines/wan/pipeline_wan_ti2v_5b.py`
(`_register_5b_matmul_tables`). Analysis and per-shape gains: `models/tt_dit/models/Wan2_2_TI2V_nadim_opt.md` section 7.1.

## E2E perf gates with the full table (mean of 3, all pass)

Run as, one per Bash call, `source python_env/bin/activate; HF_HOME=/mnt/tt-data/teja/hf pytest <node> -sv --timeout=0`:

    == i2v_720p: PASS PASS PASS
       Text Encoding        mean   0.088 s  runs ['0.088', '0.088', '0.088']  spread 1.0%
       Image Encode (host)  mean   1.379 s  runs ['1.347', '1.377', '1.413']  spread 4.7%
       Denoising            mean  11.513 s  runs ['11.484', '11.526', '11.530']  spread 0.4%
       VAE Decoding         mean   0.964 s  runs ['0.995', '0.947', '0.950']  spread 5.0%
       Total Pipeline       mean  13.962 s  runs ['13.933', '13.951', '14.001']  spread 0.5%
    == t2v_480p: PASS PASS PASS
       Text Encoding        mean   0.089 s  runs ['0.089', '0.089', '0.087']  spread 2.0%
       Denoising            mean   5.667 s  runs ['5.677', '5.650', '5.673']  spread 0.5%
       VAE Decoding         mean   0.576 s  runs ['0.566', '0.604', '0.559']  spread 7.8%
       Total Pipeline       mean   6.338 s  runs ['6.339', '6.350', '6.326']  spread 0.4%
    == t2v_720p: PASS PASS PASS
       Text Encoding        mean   0.092 s  runs ['0.099', '0.088', '0.088']  spread 12.2%
       Denoising            mean  10.701 s  runs ['10.714', '10.653', '10.736']  spread 0.8%
       VAE Decoding         mean   0.959 s  runs ['0.959', '0.953', '0.963']  spread 1.0%
       Total Pipeline       mean  11.772 s  runs ['11.795', '11.713', '11.809']  spread 0.8%

Transformer PCC suite (`test_transformer_wan_ti2v_5b.py`): 3 passed, 2 pre-existing skips,
PCC 100.0000 / 99.9893 / 99.9894 %, identical to before the swept table.

Kernel-cache cost of the sweep: ~130 GB in `~/.cache/tt-metal-cache` across ~11 shapes; the
entries are unique blockings and are never reused, safe to prune.
