# recurrent_chunk_scan (fork)

- Source: `ttnn/cpp/ttnn/operations/experimental/kda/recurrent_chunk_scan`
- Source SHA: `97176c045374eba173f900a04caf735caa9cabfa`
- Python: `ttnn.bringup.*` (was `ttnn.experimental.kda.*`)
- Forked for: glm53_flash_d_p_lb kda-exact-decay
- Used by: glm53_flash_d_p_lb

Mechanical fork changes (fork_op.py): namespace `ttnn::operations::bringup`, CMake target `ttnn_op_bringup_recurrent_chunk_scan`, kernel paths and includes pointing at this folder, Python prefix `ttnn.bringup.`.

## Changes

<!-- One entry per change, newest last:
### <short title>
- What: the change, and the switch or argument that turns it on (default = source behaviour).
- Why: the symptom it fixes or the feature it adds.
- Needed by: <model> <task>
- Files: <paths inside this folder>
-->

### Decay complement: an exact inter-chunk state decay (prototype, 2026-10-27)
- What: `decay_complement=True` (recurrent_chunk_scan and summarize_chunk_recurrence): `final_decay` is FLOAT32
  d = 1 - exp(G_last) (`ttnn.bringup.prepare_chunk_recurrence(precise_gate_factors=True, decay_complement=True)`),
  and the state update is S - S * d + U: S unpacked losslessly to DST from a second, exact copy of the state
  (`state_exact` from the reader, `state_exact_ring` packed next to `state_ring`), S * d an FPU broadcast multiply
  into DST, the subtraction and the addition of U (`state_update`, lossless to DST) SFPU ops on FP32. Default off =
  the source program.
- Why: the source applies alpha = exp(G_last) as alpha * S + U on the FPU, reading the FP32 state and alpha through
  TF32 source registers (truncated, 2^-11 relative per read). A slow KDA channel's alpha is ~0.9995 per 32 tokens, so
  each read shortens its memory by as much as its own decay (GLM-5.3-Flash layer 2 over 56k tokens: state 13-19% off,
  slowest heads 30-70%; CPU emulation of the roundings on the layer's real inputs, BREADCRUMBS 2026-10-27).
- Needed by: glm53_flash_d_p_lb kda-exact-decay
- Files: device/recurrent_chunk_scan_device_operation_types.hpp, device/*_device_operation.{hpp,cpp},
  device/*_program_factory.cpp (state_exact DFBs, UnpackToDest modes, DECAY_COMPLEMENT define),
  device/kernels/compute/recurrent_chunk_scan.cpp (exact_decay_update), device/kernels/dataflow/
  reader_recurrent_chunk_scan.cpp (the exact seed copy), recurrent_chunk_scan.{hpp,cpp}, recurrent_chunk_scan_nanobind.cpp;
  sources.cmake drops the source's ../kda_nanobind.cpp (main builds it).
