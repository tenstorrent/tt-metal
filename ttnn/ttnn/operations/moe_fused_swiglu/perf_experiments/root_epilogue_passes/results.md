# root_epilogue_passes — isolated bake-off (single core, compute-only)

box=bh-49-special-mstaletovic-for-reservation-52882  arch=BH  cores=1  placement=single-core sharded-L1  N=3 (median)  kernel-iters=20

Metric: DEVICE KERNEL DURATION [ns] per ONE root-epilogue evaluation (h = SiLU(gate_acc+last_gate_child) * (up_acc+last_up_child)).
`overhead` = per-iteration CB scaffolding only; `plain_add_x8` = the root's 8 plain blocked
48-tile reduce adds; `a_only_*` = stage (a) alone. L1 = op-side scratch the arm needs.

## m_eff=1, HN_PAD=6, block_tiles=6

| arm | median ns | ns net of scaffolding | speedup vs baseline | PCC | max abs diff vs same-kind anchor | op L1 scratch B |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 6826 | - | 1.000x | 0.999789 | 0 | 52,224 |
| hoist_rows | 6813 | - | 1.002x | 0.999789 | 0 | 52,224 |
| blk_packer | 6813 | - | 1.002x | 0.999789 | 0 | 52,224 |
| add_silu_chain | 6782 | - | 1.007x | 0.999789 | 0 | 52,224 |
| add_silu_chain_nr | 6761 | - | 1.010x | 0.999789 | 0 | 52,224 |
| fuse_silu_mul | 6775 | - | 1.008x | 0.999807 | 0.1562 | 0 |
| sigappx_mul | 2608 | - | 2.618x | 0.997727 | 0.3164 | 104,448 |
| sigappx_fused | 2466 | - | 2.768x | 0.997650 | 0.3164 | 52,224 |
| single_pass | 9679 | - | 0.705x | 0.999945 | 0.125 | 0 |

## m_eff=2, HN_PAD=6, block_tiles=12

| arm | median ns | ns net of scaffolding | speedup vs baseline | PCC | max abs diff vs same-kind anchor | op L1 scratch B |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 13143 | - | 1.000x | 0.999790 | 0 | 52,224 |
| hoist_rows | 12994 | - | 1.011x | 0.999790 | 0 | 52,224 |
| blk_packer | 13050 | - | 1.007x | 0.999790 | 0 | 52,224 |
| add_silu_chain | 12914 | - | 1.018x | 0.999790 | 0 | 52,224 |
| add_silu_chain_nr | 12891 | - | 1.020x | 0.999790 | 0 | 52,224 |
| fuse_silu_mul | 13187 | - | 0.997x | 0.999827 | 0.1562 | 0 |
| sigappx_mul | 4462 | - | 2.945x | 0.997914 | 0.375 | 104,448 |
| sigappx_fused | 4556 | - | 2.885x | 0.997865 | 0.375 | 52,224 |
| single_pass | 18932 | - | 0.694x | 0.999949 | 0.0625 | 0 |

## m_eff=4, HN_PAD=6, block_tiles=24

| arm | median ns | ns net of scaffolding | speedup vs baseline | PCC | max abs diff vs same-kind anchor | op L1 scratch B |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 25638 | - | 1.000x | 0.999788 | 0 | 52,224 |
| hoist_rows | 25330 | - | 1.012x | 0.999788 | 0 | 52,224 |
| blk_packer | 25380 | - | 1.010x | 0.999788 | 0 | 52,224 |
| add_silu_chain | 25333 | - | 1.012x | 0.999788 | 0 | 52,224 |
| add_silu_chain_nr | 25305 | - | 1.013x | 0.999788 | 0 | 52,224 |
| fuse_silu_mul | 25825 | - | 0.993x | 0.999824 | 0.1562 | 0 |
| sigappx_mul | 8192 | - | 3.129x | 0.997950 | 0.4023 | 104,448 |
| sigappx_fused | 8368 | - | 3.064x | 0.997923 | 0.4023 | 52,224 |
| single_pass | 37457 | - | 0.684x | 0.999946 | 0.09375 | 0 |

## m_eff=8, HN_PAD=6, block_tiles=48  <-- FOCUS

| arm | median ns | ns net of scaffolding | speedup vs baseline | PCC | max abs diff vs same-kind anchor | op L1 scratch B |
|---|---:|---:|---:|---:|---:|---:|
| overhead | 187 | 0 | 273.346x | - | - | 0 |
| plain_add_x8 | 11336 | 11149 | 4.501x | 0.999980 | 0 | 0 |
| a_only_baseline | 48192 | 48006 | 1.059x | 0.999957 | 0 | 0 |
| a_only_hoist_rows | 47527 | 47340 | 1.074x | 0.999957 | 0 | 0 |
| a_only_blk_packer | 47553 | 47367 | 1.073x | 0.999957 | 0 | 0 |
| a_only_chain | 47368 | 47181 | 1.077x | 0.999957 | 0 | 0 |
| a_only_sigappx | 12741 | 12554 | 4.004x | 0.997831 | 0.25 | 52,224 |
| baseline | 51020 | 50833 | 1.000x | 0.999793 | 0 | 52,224 |
| hoist_rows | 50334 | 50147 | 1.014x | 0.999793 | 0 | 52,224 |
| blk_packer | 50362 | 50175 | 1.013x | 0.999793 | 0 | 52,224 |
| add_silu_chain | 50199 | 50012 | 1.016x | 0.999793 | 0 | 52,224 |
| add_silu_chain_nr | 50167 | 49980 | 1.017x | 0.999793 | 0 | 52,224 |
| add_then_silu | 51822 | 51636 | 0.985x | 0.999764 | 0.09375 | 104,448 |
| add_then_silu_dr | 51774 | 51587 | 0.985x | 0.999764 | 0.09375 | 104,448 |
| sigacc_mul | 52078 | 51891 | 0.980x | 0.999695 | 0.1562 | 104,448 |
| sigappx_mul | 15681 | 15494 | 3.254x | 0.997836 | 0.5 | 104,448 |
| sigappx_fused | 16377 | 16190 | 3.115x | 0.997796 | 0.5 | 52,224 |
| fuse_silu_mul | 51240 | 51053 | 0.996x | 0.999828 | 0.125 | 0 |
| fuse_silu_mul_pw | 51271 | 51085 | 0.995x | 0.999828 | 0.125 | 0 |
| fuse_up_mul | 50987 | 50800 | 1.001x | 0.999810 | 0.125 | 52,224 |
| single_pass | 74525 | 74339 | 0.685x | 0.999946 | 0.125 | 0 |
| single_pass_sigappx | 61003 | 60817 | 0.836x | 0.998106 | 0.5 | 0 |

## m_eff=4, HN_PAD=4, block_tiles=16

| arm | median ns | ns net of scaffolding | speedup vs baseline | PCC | max abs diff vs same-kind anchor | op L1 scratch B |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 17346 | - | 1.000x | 0.999789 | 0 | 52,224 |
| hoist_rows | 17031 | - | 1.018x | 0.999789 | 0 | 52,224 |
| blk_packer | 17100 | - | 1.014x | 0.999789 | 0 | 52,224 |
| add_silu_chain | 17022 | - | 1.019x | 0.999789 | 0 | 52,224 |
| add_silu_chain_nr | 16995 | - | 1.021x | 0.999789 | 0 | 52,224 |
| fuse_silu_mul | 17321 | - | 1.001x | 0.999832 | 0.09375 | 0 |
| sigappx_mul | 5692 | - | 3.047x | 0.997871 | 0.5 | 104,448 |
| sigappx_fused | 5721 | - | 3.032x | 0.997844 | 0.5 | 52,224 |
| single_pass | 25105 | - | 0.691x | 0.999944 | 0.1875 | 0 |

## m_eff=8, HN_PAD=4, block_tiles=32

| arm | median ns | ns net of scaffolding | speedup vs baseline | PCC | max abs diff vs same-kind anchor | op L1 scratch B |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 34405 | - | 1.000x | 0.999788 | 0 | 52,224 |
| hoist_rows | 33684 | - | 1.021x | 0.999788 | 0 | 52,224 |
| blk_packer | 33712 | - | 1.021x | 0.999788 | 0 | 52,224 |
| add_silu_chain | 33617 | - | 1.023x | 0.999788 | 0 | 52,224 |
| add_silu_chain_nr | 33584 | - | 1.024x | 0.999788 | 0 | 52,224 |
| fuse_silu_mul | 34291 | - | 1.003x | 0.999829 | 0.125 | 0 |
| sigappx_mul | 10697 | - | 3.216x | 0.997965 | 0.5 | 104,448 |
| sigappx_fused | 11023 | - | 3.121x | 0.997913 | 0.5 | 52,224 |
| single_pass | 49808 | - | 0.691x | 0.999948 | 0.09375 | 0 |
