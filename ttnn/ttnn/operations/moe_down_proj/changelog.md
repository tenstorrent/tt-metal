# Changelog: moe_down_proj

## Phase 0 - Core Implementation
- **Date**: 2026-11-03
- **What was done**: Initial implementation via incremental pipeline (planner -> implementer -> verifier). Regime R1:
  2D split (expert-sorted rows over 9 m-groups, N over 12 n-lines, L layout, 108 / 110 cores). Each K round runs
  rotating-sender x and w multicasts and accumulates fp32 packer-L1 partials, followed by an untilize and a
  row-major store per wave. Counts are read on device only. One `ttnn.generic_op` per call.
- **SUPPORTED at Phase 0**: dtype=[bfloat8_b], layout=[TILE], fp32_dest_acc_en=[True], memory_layout=[INTERLEAVED],
  buffer_type=[L1], out_buffer_type=[L1]; no INPUT_TAGGERS, no EXCLUSIONS
- **Accuracy achieved** (LoFi default, 5 shapes, `test_moe_down_proj_precision_baseline.py`): PCC=0.99991,
  max_abs_err=0.279, rel_rms_err=1.90e-2 (HiFi2: PCC=0.999999, rel_rms 1.68e-3). The LoFi got/true ratio median of
  0.9865 is the fidelity's truncation bias; it is 1.0000 at HiFi2.
- **Device kernel time (us)**: t4096 L0..L5 173.2 / 174.5 / 175.0 / 178.9 / 173.7 / 173.1; t3072 140.0;
  t2048 113.7; t1024 102.8 (all under stock; targets 90.8 / 74.6)
- **Golden suite at Phase 0**: 27 supported_pass, 54 xfail_expected, 0 supported_fail / xpass_drift /
  xfail_wrong_mode (93 total; 12 regression tests untagged, passing)
- **Issues encountered** (verifier fixes): CB sizing was duplicated between the budget solve and the allocation, now
  one `cb_plan()`. A raw `llk_pack_reconfig_l1_acc` was replaced by `pack_reconfig_l1_acc`. The writer CT-arg
  declarations were reordered. The ledger's `cb_w_round` producer was corrected (writer RISC). No kernel bugs.
- **Tests added**: test_moe_down_proj.py (planner), test_moe_down_proj_perf.py (implementer),
  test_moe_down_proj_precision_baseline.py and test_moe_down_proj_layouts.py (verifier)
