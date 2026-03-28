# Wormhole Streaming SDPA LLK Plan

## Goal

Port the Blackhole-only LLK features used by `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp` into the Wormhole LLK stack first, with LLK-side accuracy and perf coverage, and only then remove the remaining `#ifdef ARCH_BLACKHOLE` guards in the SDPA kernel.

## Reference Sources

Use these as the main external knowledge sources while implementing the Wormhole LLK ports:

- Local Tensix simulator: `/localdev/pjosipovic/ttsim-private`
  - Use this to confirm Tensix behavior when LLK intent is unclear, especially around replay-buffer execution, address modifiers, unpack/pack sequencing, semaphore interactions, and instruction-side effects.
- TT ISA documentation: `https://github.com/tenstorrent/tt-isa-documentation`
  - Use this to validate instruction semantics and register behavior at the ISA level, especially for MOP programming, replay, counter updates, and pack/unpack/math instructions.

Primary in-tree implementation references:

- Blackhole LLK implementations already used by the streaming SDPA kernel
- Existing Wormhole LLK matmul/pack/unpack implementations that will be extended rather than replaced

## Missing LLK Features In `compute_streaming.hpp`

These are the real Wormhole LLK gaps behind the current Blackhole-only branches.

| Missing feature | Where it shows up in `compute_streaming.hpp` | Current Blackhole path | Current Wormhole behavior |
| --- | --- | --- | --- |
| No-MOP blocked matmul | `blocked_matmul_and_pack`, `sdpa_inner_loop_step` Q@KT and QKT@V init/reinit sites | `matmul_block_no_mop`, `mm_no_mop_init_short`, `mm_no_mop_reinit_short` | Falls back to regular MOP-based matmul init/run |
| Blocked sub+bcast(cols) | `sub_exp_block_bcast_cols` | `sub_bcast_cols_init_short_custom`, `sub_tiles_bcast_cols_custom` | Falls back to tile-by-tile `sub_tiles_bcast_cols` loop |
| Multi-tile pack MOP / blocked pack | `blocked_matmul_and_pack`, `sub_exp_block_bcast_cols`, `salad_correct_fused`, `sdpa_inner_loop_step` pack reconfig points | `llk_pack_mop_config(..., num_tiles)` plus blocked `pack_tile<true>` usage | Only single-tile pack MOP exists, so Wormhole must pack tile-by-tile |

## Exact Kernel Sites To Replace After LLK Work Lands

- No-MOP matmul:
  - `compute_streaming.hpp:99-103`
  - `compute_streaming.hpp:735-739`
  - `compute_streaming.hpp:762-766`
  - `compute_streaming.hpp:930-934`
  - `compute_streaming.hpp:1038-1042`
- Blocked sub+bcast(cols):
  - `compute_streaming.hpp:224-228`
  - `compute_streaming.hpp:241-250`
- Multi-tile pack MOP / blocked pack:
  - `compute_streaming.hpp:112-120`
  - `compute_streaming.hpp:275-299`
  - `compute_streaming.hpp:323-326`
  - `compute_streaming.hpp:431-455`
  - `compute_streaming.hpp:467-468`
  - `compute_streaming.hpp:719-720`
  - `compute_streaming.hpp:823-824`
  - `compute_streaming.hpp:841-842`
  - `compute_streaming.hpp:896-897`

## Blackhole-Only Branches That Are Not LLK Gaps

- `SDPA_NOINLINE` near the file top is a Wormhole code-size workaround, not a missing LLK.
- `normalize_row_streaming` uses a Blackhole reciprocal fast path, but Wormhole already has generic `recip_tile(..., VectorMode::C)` support. That branch is optional cleanup/perf work, not a blocker for LLK parity.

## LLK Work Breakdown

### Phase 1: Wormhole packer parity for blocked pack

This is the first dependency because the SDPA kernel uses blocked packing in several places.

Files to update:

- `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_pack.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_pack_api.h`

Expected work:

- Add Wormhole support for `num_tiles` in `_llk_pack_mop_config_`.
- Thread `num_tiles` through `llk_pack_mop_config` and `llk_pack_init`, matching Blackhole API shape closely enough that SDPA can call the same interface on both arches.
- Keep the single-tile behavior identical for existing users.

LLK validation:

- Extend or unskip `tt_metal/third_party/tt_llk/tests/python_tests/test_pack_dest_bank.py`
- Reuse `tt_metal/third_party/tt_llk/tests/sources/pack_dest_bank_test.cpp`
- Add perf coverage by extending `perf_pack_dest_bank.py` or adding a small blocked-pack perf test if the existing harness is too broad

Acceptance for this phase:

- One `_llk_pack_` call can pack multiple consecutive tiles correctly on Wormhole.
- Existing Wormhole pack tests remain green.

### Phase 2: Port blocked sub+bcast(cols) custom LLKs

Files to add under Wormhole LLK:

- `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/experimental/llk_math_eltwise_binary_custom.h`
- `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/experimental/llk_unpack_AB_sub_bcast_col_custom.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/experimental/llk_math_eltwise_binary_custom_api.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/experimental/llk_unpack_AB_sub_bcast_col_custom_api.h`

Files to wire:

- `tt_metal/hw/inc/api/compute/experimental/sdpa_sub_custom.h`

Expected work:

- Port the blocked unpack flow that loads one broadcast tile and `ct_dim` source tiles in a single call.
- Port the custom math loop that consumes the blocked unpack stream without re-running the generic tile-by-tile setup.
- Keep the API identical to the Blackhole helpers so the SDPA kernel can use one call site.

LLK validation:

- Unskip or generalize `tt_metal/third_party/tt_llk/tests/python_tests/test_eltwise_bcast_col_custom.py`
- Unskip or generalize `tt_metal/third_party/tt_llk/tests/python_tests/perf_eltwise_bcast_col_custom.py`
- Reuse:
  - `tt_metal/third_party/tt_llk/tests/sources/multiple_tiles_eltwise_custom_test.cpp`
  - `tt_metal/third_party/tt_llk/tests/sources/eltwise_bcast_col_custom_perf.cpp`

Acceptance for this phase:

- Wormhole accuracy matches the existing golden path across the current `ct_dim` sweep.
- Perf beats the current tile-by-tile fallback for representative `ct_dim > 1`.

### Phase 3: Port no-MOP matmul custom LLKs

Files to add under Wormhole LLK:

- `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/experimental/llk_math_matmul_custom_no_mop.h`
- `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/experimental/llk_unpack_AB_matmul_custom.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/experimental/llk_math_matmul_custom_api.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/experimental/llk_unpack_AB_matmul_custom_api.h`

Files to wire:

- `tt_metal/hw/inc/api/compute/experimental/matmul_custom.h`

Expected work:

- Implement Wormhole equivalents of:
  - `mm_no_mop_init_short`
  - `matmul_block_no_mop`
  - `mm_no_mop_reinit_short`
- If needed for test parity, also port the light reinit path after sub:
  - `mm_no_mop_reinit_after_sub`
- Reuse Wormhole’s existing matmul addrmods and replay-buffer machinery where possible; the main delta is removing the per-call MOP programming/run overhead.

LLK validation:

- Unskip or generalize `tt_metal/third_party/tt_llk/tests/python_tests/test_matmul_custom.py`
- Unskip or generalize `tt_metal/third_party/tt_llk/tests/python_tests/test_sdpa_reinits.py`
- Reuse:
  - `tt_metal/third_party/tt_llk/tests/sources/matmul_custom_test.cpp`
  - `tt_metal/third_party/tt_llk/tests/sources/sdpa_reinits_test.cpp`
- Add perf coverage:
  - either extend `perf_math_matmul.py` / `perf_matmul.py`
  - or add dedicated `perf_matmul_custom.py` and matching source if the custom path needs separate reporting

Acceptance for this phase:

- Wormhole custom no-MOP matmul is correct across the existing matmul sweep.
- Reinit sequences used by SDPA are covered by LLK tests, not only by the higher-level kernel.

### Phase 4: SDPA integration after LLK parity

Only after the LLK submodule work above is in place:

- Remove or narrow the `ARCH_BLACKHOLE` guards in `compute_streaming.hpp` for:
  - no-MOP matmul
  - blocked sub+bcast
  - blocked pack
- Keep `SDPA_NOINLINE` and the reciprocal fast-path choice separate until Wormhole code-size and perf are measured.

Validation after integration:

- Run LLK accuracy/perf suites again.
- Run SDPA correctness and model-level perf on Wormhole with representative streaming shapes.

## Recommended Execution Order

1. Multi-tile pack MOP on Wormhole
2. Blocked sub+bcast(cols)
3. No-MOP matmul
4. `compute_streaming.hpp` integration

Why this order:

- Packer parity is a hard dependency for most of the blocked-pack branches.
- Blocked sub+bcast is the smallest isolated custom path and already has dedicated test/perf scaffolding.
- No-MOP matmul is the highest-risk port and benefits from pack/sub infrastructure already being in place.

## Risks / Open Questions

- Wormhole packer MOP programming differs from Blackhole, so the `num_tiles` port will likely be a Wormhole-specific implementation, not a text copy.
- Wormhole replay-buffer depth and instruction pressure may force a slightly different no-MOP matmul formulation than Blackhole.
- `test_sdpa_reinits.py` currently validates more than the exact streaming-kernel call pattern; decide early whether to port `mm_no_mop_reinit_after_sub` immediately or keep it as a second step.
- After LLK parity lands, the SDPA kernel may still need selective `noinline` or smaller helper factoring on Wormhole for code-size reasons.

## Definition Of Done

- Wormhole no longer skips the LLK tests that currently cover the needed Blackhole custom primitives:
  - `test_pack_dest_bank.py`
  - `test_eltwise_bcast_col_custom.py`
  - `perf_eltwise_bcast_col_custom.py`
  - `test_matmul_custom.py`
  - `test_sdpa_reinits.py`
- Wormhole LLK has arch-local implementations for the three missing feature groups above.
- `compute_streaming.hpp` can use the same LLK-level feature set on both Blackhole and Wormhole for the SDPA hot paths that are currently guarded.
