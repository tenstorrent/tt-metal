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

### Decay complement output (prototype, 2026-10-27)
- What: `decay_complement=True` (needs `precise_gate_factors` and an FP32 final_decay, `output_bf16_mask` bit 5
  clear): `final_decay` holds d = 1 - exp(G_last) instead of exp(G_last), formed in DST from the exact G_last (a fourth
  mask-matmul pass: gate_half @ g, x2, exp, 1 - x) and transposed to the [K, 1] column. Default off = the source.
- Why: exp(G_last) of a slow channel is ~0.9995; the source squares exp(G_last/2) after a TF32 copy and transposes
  it through srcA (TF32), and ttKDA stores it in bf16 (rounds to 1.0): either moves the channel's decay by up to its
  whole size. d keeps its relative precision through the TF32 transpose. Consumed by
  `ttnn.bringup.recurrent_chunk_scan(decay_complement=True)`.
- Needed by: glm53_flash_d_p_lb kda-exact-decay
- Files: device/*_types.hpp, device/*_device_operation.{hpp,cpp}, device/*_program_factory.cpp
  (decay_complement_rows DFB, DECAY_COMPLEMENT define), device/kernels/compute/prepare_chunk_recurrence.cpp,
  prepare_chunk_recurrence.{hpp,cpp}, prepare_chunk_recurrence_nanobind.cpp.
