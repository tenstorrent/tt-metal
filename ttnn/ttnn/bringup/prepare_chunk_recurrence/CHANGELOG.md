# prepare_chunk_recurrence (fork)

- Source: `ttnn/cpp/ttnn/operations/experimental/kda/prepare_chunk_recurrence`
- Source SHA: `c6a7679fb8787196a37d6e9447b20d5fcc1598e4`
- Python: `ttnn.bringup.*` (was `ttnn.experimental.kda.*`)
- Forked for: glm53_flash_d_p_lb kda-precise-gate
- Used by: glm53_flash_d_p / glm53_flash_d_p_lb (tt/kda_attention.py, GLM_KDA_DECAY=fork, the default)
- Note (2026-10-19): glm53_flash_d_p_lb is the LoudBox bring-up that asked for the fork (task kda-precise-gate,
  `models/demos/glm53_flash_d_p_lb/bringup/BREADCRUMBS.md`); it has no `tt/` of its own. The only caller is
  `models/demos/glm53_flash_d_p/tt/kda_attention.py` (`_ForkPrepRecurrence`).

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

### Tests: fork test suite with a glm53_flash_d_p case (O.1, 2026-10-13)
- Model-case suite `tests/cases.py`, `tests/reference.py`, `tests/test_prepare_chunk_recurrence.py` (+ `tests/__init__.py`): GLM's call (2x2, SP axis 0, 2560 rows, 32 heads x 128, gate_scale -5, beta logits at column 16512, precise_gate_factors), seven outputs vs a float64 reference, plus determinism. No op change.

### Upstream sync to source @ 6b80daf43e3 (final #59918)
- What (owner: full sync): the fork was taken @ `c6a7679fb87` from an early, unreviewed snapshot of #59918 (+ #59473).
  Taken from upstream:
  - `2c3b1060209` #59918 (final): **API change**, `beta_logits_column_offset` is a mandatory uint32 (default 0), no
    longer optional. Beta must be BF16 pre-sigmoid logits, token-major `[1, T, columns]`; the FP32 pre-activated
    beta and the by-chunk `[H, N, 32, 1]` beta are removed (validation, the `beta_logits` / `beta_token_major`
    compile args and the compute kernel's `beta_logits` template arg are gone). Reader: the gate and ones/tril fills
    are one `fill_ones_and_tril`; padded rows are zeroed earlier (same result).
  - `4cd5e626d8d` #60899: program 70768 -> 60160 B (it had overflowed the 70656 B kernel config buffer by up to
    384 B after #59773): compute helpers no longer `inline`, `#pragma GCC unroll 1` on pack loops, out-of-line
    reader fills.
  - `f8fabeb9b9c` #60450: `chronology.hpp` moved to `kda/device/kernels/` (the 3 kernels' include; the old path no
    longer exists).
  - `298c06426a8` #61991: the reader includes `"api/tile_constants.h"` instead of `<tt-metalium/constants.hpp>`
    (`#error` in kernels now).
  - `4d35d9573e7` #52360 (compute helpers templated on the buffer type), `22793d66d28` #59941 (arch read from the
    device): mechanical.
- How: not merged forward. Started from fork_op.py's copy of the source @ `6b80daf43e3` and re-applied
  `precise_gate_factors` by hand. Host: the option argument / attribute / binding, and the program factory's
  `if (attrs.precise_gate_factors)` block (3 FP32 mask buffers, their bindings and unpack modes, the
  `PRECISE_GATE_FACTORS` define on reader and compute, compute at -O2), identical to the old fork's hunks.
  Kernels: `prepare_gate_factors_precise` (compute) and `fill_precise_gate_tiles` (reader) in upstream's new style
  (templated on the buffer type, not `inline`); the `g_last` release, the shared q/k L2-norm reduce and the call
  site under the define as before. Re-applied the fork's hand fixes: `using namespace
  ttnn::experimental::prim::kda_factory_detail` (the shared namespace, not `prim::bringup::`), `sources.cmake`
  without `../kda_nanobind.cpp`, CMakeLists formatting. Option off: the kernels reduce to upstream's token for token
  with the define absent (checked), and the host adds only the attribute (default false, in the reflected
  attributes, so in the program hash). No build or device run yet.
- Behaviour change: callers must pass BF16 token-major beta logits (GLM already does: column 16512, no caller
  change). Option off = the final #59918 + #60899 program. Option on: same math as before; it now sits on
  upstream's reader / compute helpers, so its code size differs from the measured fork.
- Tests: `tests/reference.py` takes `beta_logits_column_offset=0` by default like the op (it asserted on None
  before); the model case already passed 16512. `tests/source.yaml` unchanged (empty selection, made at
  `c6a7679fb87`).
- Must be re-measured after the build: program size with the option off and on (both must fit the 70656 B
  kernel config buffer; the -O2 was sized for the pre-#60899 program and may no longer be needed, and -O3 may now
  fit), accuracy (the model case in `tests/`, GLM's KDA component test: output / state / worst head vs the
  0.0072 / 0.0142 / 0.0227 above) and the op time (0.471 ms per GLM layer above).
- Needed by: rebase onto llk_helper_library `6b80daf43e3` (2026-10-27)
- Files: prepare_chunk_recurrence.{hpp,cpp}, prepare_chunk_recurrence_nanobind.cpp,
  device/prepare_chunk_recurrence_device_operation{.hpp,.cpp,_types.hpp}, device/prepare_chunk_recurrence_program_factory.cpp,
  device/kernels/compute/prepare_chunk_recurrence.cpp, device/kernels/dataflow/{reader,writer}_prepare_chunk_recurrence.cpp,
  tests/reference.py, README.md
