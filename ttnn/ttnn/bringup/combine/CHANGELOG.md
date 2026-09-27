# combine (fork)

- Source: `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine`
- Source SHA: `67ca5f3af4860c04e7ce31e345107f51cb22aa9b`
- Python: `ttnn.bringup.*` (was `ttnn.experimental.deepseek_prefill.*`)
- Forked for: ernie45_d_p P3.1
- Used by: ernie45_d_p, gemma4_a4b_d_p, mimo_v2_6_d_p

Mechanical fork changes (fork_op.py): namespace `ttnn::operations::bringup`, CMake target `ttnn_op_bringup_combine`, kernel paths and includes pointing at this folder, Python prefix `ttnn.bringup.`.

## Changes

### Support a 1-device dispatch axis (no fabric)
- What: all fabric setup is gated on `use_fabric = num_links > 0 && dispatch-axis devices > 1`; otherwise the
  kernels' existing no-fabric build (`#ifdef DEST_CHIP_ID`) is used. No change for dispatch axes with more than one
  device.
- Why: with one chip per dispatch group every routed expert is local or absent, so no token crosses the fabric, but
  the host factory still looked up fabric neighbours (TT_FATAL "No neighbors found").
- Needed by: ernie45_d_p P3.1 (originally commit e30d505c4ae)
- Files: `device/combine_program_factory.cpp`
