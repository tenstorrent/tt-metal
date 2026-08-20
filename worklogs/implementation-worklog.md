# Implementation worklog — `riverwu/m2-neat`

Living log of *this* implementation pass. Behavior lives in [`SPEC.md`](./SPEC.md); plan in [`IMPLEMENTATION.md`](./IMPLEMENTATION.md). Newest entry at the top.

---

## 2026-08-19 — Slice A started

Executing IMPLEMENTATION §7 Slice A only: `ScratchpadSpec` fields + `MakeProgramFromSpec` validation (SPEC §3.2 / §14). No tests, tokens, filegen, or `to_llk_mem_descriptor`.

Clarified before coding: geometry-without-format is Scratchpad-only. Compute DFB already requires format; DM-only DFB is not an LLK source.

Plan:

1. Add the three optional LLK fields to `scratchpad_spec.hpp` (same names/types/defaults as DFB; format comment is scratchpad-specific — binding ≠ LLK operand).
2. In `program_spec.cpp`, next to the existing DFB format checks: Scratchpad arch-support + geometry-without-format; DFB+Scratchpad invalid `FaceGeometry` and face-grid overflow (same predicates as `CircularBufferConfig::set_unpack_face_geometry` / `compute_num_faces_rc_dims`). Overflow only when both tile and face are set. Face-grid helper stays local here; Slice B extracts it.
3. Compile-check `unit_tests_api`.
4. One commit, push.

Fields and validation are in. `ninja -C build unit_tests_api` succeeded (including existing Scratchpad designated-init sites with no edits). Gen1 `ScratchpadAccessorBindingJITSmokeComputeKernel` still compiles and runs. Quasar mock tests fail in fixture SetUp (`dispatch_cores.empty()`) — pre-existing env, not this change.

Committing Slice A (host fields + validation + worklogs). No tests/filegen yet.
