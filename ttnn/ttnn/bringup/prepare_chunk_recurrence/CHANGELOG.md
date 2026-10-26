# prepare_chunk_recurrence (fork)

- Source: `ttnn/cpp/ttnn/operations/experimental/kda/prepare_chunk_recurrence`
- Source SHA: `231c64be747a9487017ae0703c97c5d245712467`
- Python: `ttnn.bringup.*` (was `ttnn.experimental.kda.*`)
- Forked for: glm53_flash_d_p_lb kda-precise-gate
- Used by: glm53_flash_d_p_lb

Mechanical fork changes (fork_op.py): namespace `ttnn::operations::bringup`, CMake target `ttnn_op_bringup_prepare_chunk_recurrence`, kernel paths and includes pointing at this folder, Python prefix `ttnn.bringup.`.

## Changes

<!-- One entry per change, newest last:
### <short title>
- What: the change, and the switch or argument that turns it on (default = source behaviour).
- Why: the symptom it fixes or the feature it adds.
- Needed by: <model> <task>
- Files: <paths inside this folder>
-->

### Precise gate factors (re-cut from main's reworked op)
- What: `precise_gate_factors=True` forms the anchored exponents exactly: exp(G - G_last/2), exp(G_last/2 - G) and
  exp(G_last/2) are masked matmuls of the BF16 gate (mask entries +-scale/2, reader-filled `gate_center` /
  `gate_center_neg` / `gate_half`, FP32 accumulation) exponentiated in DST, never re-read through a source register;
  G_last is never published. Default off = the source op.
- Why: the source op re-reads FP32 G = cumsum(g) at TF32 in srcA/srcB (|G| ~ 150 at the -5 gate bound: a 0.125 step),
  biasing the anchored exponents (k_dec_t fast-decay rows ~5% low; exp(G_last) by squaring doubles it). GLM layer-0
  KDA component test, worst-head recurrent state rel L2: source op 0.0548 (fails <= 0.05), this 0.0269. GLM-5.3-Flash
  2x4 56k prefill: 4.91 s / 11482 tok/s (source op 11442): no cost.
- Needed by: glm53_flash_d_p_lb kda-precise-gate
- Files: prepare_chunk_recurrence.{hpp,cpp}, prepare_chunk_recurrence_nanobind.cpp, device/*_types.hpp,
  device/*_device_operation.{hpp,cpp}, device/*_program_factory.cpp (DFBs + PRECISE_GATE_FACTORS define + compute O2),
  device/kernels/dataflow/reader_prepare_chunk_recurrence.cpp (fill_precise_gate_tiles),
  device/kernels/compute/prepare_chunk_recurrence.cpp (prepare_gate_factors_precise); sources.cmake drops the
  source's ../kda_nanobind.cpp (main builds it).
