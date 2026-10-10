# prepare_chunk_recurrence (fork)

- Source: `ttnn/cpp/ttnn/operations/experimental/kda/prepare_chunk_recurrence`
- Source SHA: `c6a7679fb8787196a37d6e9447b20d5fcc1598e4`
- Python: `ttnn.bringup.*` (was `ttnn.experimental.kda.*`)
- Forked for: glm53_flash_d_p_lb kda-precise-gate
- Used by: glm53_flash_d_p / glm53_flash_d_p_lb (tt/kda_attention.py, GLM_KDA_DECAY=fork, the default)

Mechanical fork changes (fork_op.py): namespace `ttnn::operations::bringup`, CMake target `ttnn_op_bringup_prepare_chunk_recurrence`, kernel paths and includes pointing at this folder, Python prefix `ttnn.bringup.`.

## Changes

<!-- One entry per change, newest last:
### <short title>
- What: the change, and the switch or argument that turns it on (default = source behaviour).
- Why: the symptom it fixes or the feature it adds.
- Needed by: <model> <task>
- Files: <paths inside this folder>
-->

### precise_gate_factors
- What: `precise_gate_factors=True` (default False: the source program, unchanged) forms the anchored gate exponents
  as exact matmuls of the gate with constant masks and exponentiates them in DST: exp(G - G_last/2) =
  exp(gate_center @ g), exp(G_last/2 - G) = exp(-gate_center @ g), exp(G_last/2) = exp(gate_half @ g). The reader
  fills the three masks (entries +-scale/2, exact). This replaces the source's FP32 subtraction G - G_last/2, its
  copies of G_last/2 and of the centered exponent into the SFPU, and the reciprocal; G_last itself is no longer formed.
- Why: FP32 operands are read at TF32 precision in the source registers (unpack-to-src), so at the -5 gate bound
  (|G| ~ 150 per 32-token chunk, a step of 0.125) the anchored exponents were off by up to ~0.06 and biased:
  k_dec_t's fast-decay rows ~5% low (GLM KDA state worst head 0.054 > 0.05), also q_decay / kd / intra. GLM
  previously recomputed k_dec_t with 14 ttnn ops after the op (+1.44 ms per layer); with this option the op is
  exact and faster than the source (0.471 vs 0.586 ms per GLM layer, LoudBox 2x4, 2560 rows per chip).
- Program size: the source program is within ~1.7 KB of the 70.6 KB kernel config buffer. With the option the
  compute kernel builds at -O2 (-O3 re-inlines the matmul init into matmul_blocks), q and k share one L2-norm reduce
  instantiation (k's scale an exact multiply by 1.0), and the new loop has one copy and one unpack reconfig.
- Accuracy (GLM layer 0 KDA component test vs the CPU golden): output rel L2 0.0072, state rel 0.0142, worst head
  0.0227 (source 0.0070 / 0.0149 / 0.0537 FAIL; GLM's ttnn-op recompute 0.0070 / 0.0140 / 0.0222).
- Needed by: glm53_flash_d_p_lb kda-precise-gate
- Files: device/prepare_chunk_recurrence_device_operation_types.hpp (attribute), prepare_chunk_recurrence*.{hpp,cpp},
  device/prepare_chunk_recurrence_device_operation.{hpp,cpp}, device/prepare_chunk_recurrence_program_factory.cpp
  (masks, define, -O2), device/kernels/compute/prepare_chunk_recurrence.cpp, device/kernels/dataflow/
  reader_prepare_chunk_recurrence.cpp; sources.cmake (dropped the source family's kda_nanobind.cpp)
