# Worklog — id-free LLK operand metadata (runtime)

Living log for `riverwu/m2-neat`. Background: [`BACKGROUND.md`](./BACKGROUND.md).

Convention: newest session at the top. Record decisions, not just file dumps. Update this file in the same session as the decision — do not batch later.

---

## 2026-08-19 — Slice A executing

Implementor started Slice A (ScratchpadSpec fields + §14 host validation). Living implementation log: [`implementation-worklog.md`](./implementation-worklog.md).

---

## 2026-08-19 — Clarify SPEC §1: host knowledge of “needs format”

User: DFB with a compute kernel always needs format (it is an LLK operand). The same bytes on a compute-bound scratchpad need not be ingested by LLK, so the host cannot require metadata there.

[`SPEC.md`](./SPEC.md) §1: table rows for geometry-without-format are Scratchpad-only. DFB compute already requires format; DM-only DFB is not an LLK source. §14 reject table matches. Implementor misread the §1 Reject rows as a DFB DM-only rule.

---

## 2026-08-19 — Tests after A; compile + commit + push per slice

User: implement tests after Slice A, and only the host-side-compilable ones first. Every slice ends with a compile-check, commit, and push.

[`IMPLEMENTATION.md`](./IMPLEMENTATION.md) §7: **A** host fields/validation → **A.tests** (`tests.md` §2.1, §2.2 rejects, format-less accept) → **B** plumbing/filegen → **C** converter + remaining Gen1 fold / DRAM / hash cases. Held from A.tests: anything that needs `to_llk_mem_descriptor` or baked token facts.

---

## 2026-08-19 — §2.2 accepts also compile + `static_assert`

User: §2.2 will not dispatch, but compilation still runs — add a device `static_assert` that the values actually landed on the token.

[`tests.md`](./tests.md): rejects stay Quasar / `MakeProgramFromSpec` only. Format-bearing Scratchpad accepts move to Gen1: `MakeProgramFromSpec` then `CompileProgram` with the same kernel `static_assert` as §3.1 (cookbook shapes). Format-less accept still compiles `Scratchpad<T>(token)` only — no `to_llk_mem_descriptor` (UB). Dropped the duplicate Scratchpad rows from §3.3; that table is now DFB + MeshTensor.

---

## 2026-08-19 — Fresh test plan (`tests.md`)

User: rewrite [`tests.md`](./tests.md) from scratch (discard the prior attempt). Spec + background + the API / `to_llk_mem_descriptor` only; update this worklog. No other files.

Threw out the old plan (it had drifted with SPEC/IMPLEMENTATION swings and skipped DFB §14 geometry rejects). New plan is spec-driven:

- Still the Metal 2 mega file (`test_program_spec.cpp`): Quasar `MakeProgramFromSpec` for host, Gen1 `CompileProgram` + kernel `static_assert` for the converter. No sibling files, no silicon, no experimental ops.
- Host: add the missing `ScratchpadSpec` aggregate/hash `static_assert`s; Scratchpad §3.2 / §14 matrix (compute bind without format succeeds; geometry without format / bad `FaceGeometry` / face-grid overflow fail). Also the two DFB §14 cases that are **not** covered by “compute needs format”: invalid `FaceGeometry`, face-grid overflow.
- Device: cookbook of folded `TensorShape`s (default 32×32, FaceGeometry `{1,4}`, 16×32 vs 32×16). One compile case per object + the tensor wide/narrow pair. DRAM still `is_dram`. Format-less conversion stays untested (UB).
- Hash: format / tile on the token must change `compute_hash` (same pattern as scratchpad size).
- Explicitly out: address seams, `LLKOperand` weld, token-member reads, remaps, `unpack_modes`.

---

## 2026-08-19 — Keep the worklog current

User: keep updating [`WORKLOG.md`](./WORKLOG.md) every session.

Catch-up for sessions that only touched SPEC:

- Host reject table (§1 / §14): **only two DFB checks exist** at `MakeProgramFromSpec` (compute DFB needs format; format must be arch-supported). Scratchpad LLK fields, geometry-without-format, invalid `FaceGeometry`, and face-grid overflow are **not** implemented yet. Overflow exists later in JIT `compute_num_faces_rc_dims`, not at program construction. Invalid face geometry is a CB setter check, not DFB `ProgramSpec`.
- LLK does **not** reject a bad `LLKMemDescriptor` / `LLKOperand`. Types have no invariants. `Invalid` (`0xFF`) programmed into HW is masked to 4 bits → `Bfp2_b`. `SCALE_DATUM_SIZE` default is 1 byte/datum. `LLK_ASSERT` / sanitizer / Watcher are debug-only. That is why device-side invalid config is **UB**, not a converter fail.
- Intermediate SPEC swing (conversion “must fail”) was dropped; final lock is the UB entry below.

---

## 2026-08-19 — Filegen uses designated initializers

User: generated tokens should designated-init the LLK facts, not a positional brace list.

`LlkOperandMembers` stays a public aggregate. Filegen emits `.format` / `.face_r_dim` / `.face_c_dim` / `.num_faces_r_dim` / `.num_faces_c_dim` in declaration order. Device is C++17; gcc designated-init extension is the expected compile path.

---

## 2026-08-19 — Format-less conversion is UB; host owns illegal config

User, after refining SPEC: most errors are host configuration; the leftover (compute scratchpad with no format, then `to_llk_mem_descriptor`) is **UB**.

Aligned [`IMPLEMENTATION.md`](./IMPLEMENTATION.md) and [`tests.md`](./tests.md) to SPEC §12 / §14:

- Tokens stay untemplated. Extra ctor args when format is set. Converter always matches and reads members. No `HasLlk`, no converter `ASSERT`.
- Host `MakeProgramFromSpec` rejects unsupported format, geometry without format, bad `FaceGeometry`, face-grid overflow. Compute-bound format-less scratchpad stays legal.
- Do not test format-less conversion (not a compile fail, not a defined success). DRAM still `is_dram`. Added host tests for invalid FaceGeometry / face-grid overflow.

---

## 2026-08-19 — Device presence check is a nothing burger

User: today’s missing-format path for compute DFB is host `TT_FATAL`; unused slots silently `Invalid`. Device-side checking is not worth doing.

SPEC §12 rewritten: no `HasFormat`, no conversion `ASSERT`, no missing overload. Compute DFB without format never reaches the device. Format-less scratchpad conversion = unused `chlkc` slot (`Invalid` + default 32×32). Keep DRAM `is_dram` only. tests.md drops the format-less scratchpad compile-fail case.

---

## 2026-08-19 — No templated tokens; fault omission at conversion; host rejects illegal configs

User, after an implementation attempt: (1) binding token must not be horribly templated; (2) not giving LLK params — just fault?; (3) when to reject an LLK config.

SPEC updated (§8, §12, §14). Decisions:

- **No `HasLlk` / Format NTTPs** on DFB or Scratchpad tokens. They stay value types. Tensor keeps existing `CTA`/`CRTA` only.
- **Omission** (scratchpad / DM-only DFB with no format): legal on the host; `Scratchpad<T>` / `DataflowBuffer` still construct. `to_llk_mem_descriptor` **faults**. Not a missing overload. C++17 cannot `static_assert` on instance members; we will not add a bool NTTP to fake that. DRAM still `static_assert(!is_dram)`.
- **Illegal config** is host `MakeProgramFromSpec`: unsupported format, geometry without format, bad FaceGeometry, face grid vs tile overflow. Do **not** require format on compute-bound scratchpad. TensorParameter has no extra reject rules.

IMPLEMENTATION.md `HasLlk` plan is superseded; rewrite token shapes before coding. tests.md §3.2 no longer assumes `static_assert(HasLlk)`.

---

## 2026-08-19 — Conversion refusal parked for SPEC

User: had not thought §12 through in the spec; will resolve it there.

Agreed. IMPLEMENTATION §1.6 is a C++17 constraint note only (cannot `static_assert` on a parameter’s instance members; no `consteval`). Not a chosen mechanism (`HasLlk` / missing overload / …). Do not implement refusal until SPEC says how.

---

## 2026-08-19 — Tests live in the Metal 2 mega file; compile only

User: put the tests in the existing Metal 2.0 mega file; most should be “can this compile” via `static_assert`; no LLK integration.

Rewrote [`tests.md`](./tests.md). All cases go in `test_program_spec.cpp` (`unit_tests_api`). Dropped sibling files, `sources.cmake` additions, `test_kernels/` sources, and the BH datacopy / experimental-op suite.

- Host: add `ScratchpadSpec` to the existing `hashable_v` / `is_aggregate_v` lists; Scratchpad LLK-field validation next to the current Scratchpad / DFB-format blocks.
- Device: `ProgramSpecTestGen1` + inline source + `CompileProgram`. Kernel `static_assert`s the folded descriptor (default tile, FaceGeometry, 16×32 vs 32×16). Refusal is `CompileProgram` throw from the converter’s `static_assert` (sentinel / DRAM).
- `ScratchpadAccessorBindingJITSmokeComputeKernel` already covers format-less `Scratchpad<T>(token)`.

---

## 2026-08-19 — C++17: static_assert on `HasLlk`, not consteval / instance members

User: device is C++17 (no `consteval`). Can we inject a constexpr member and `static_assert` on it?

No for an **instance** member: `static_assert(token.has_llk_)` is ill-formed — parameters are not constant expressions, even in a `constexpr` function.

Yes for a **static** constexpr on the type. Same idea as `TensorBindingToken::args_t::is_dram`. Plan: `template <bool HasLlk = false>` on DFB/scratch tokens, `static constexpr bool has_llk_metadata = HasLlk`, converter does `static_assert(HasLlk)`. Format/shape stay instance members. Filegen emits `Token<true>` vs `Token<>`. No `consteval`.

Updated [`IMPLEMENTATION.md`](./IMPLEMENTATION.md) §1.6 / §2 / §5.1 and [`tests.md`](./tests.md) §3.2.

---

## 2026-08-19 — Token metadata are members, not NTTPs

User: do not put format/shape on the token as NTTPs; they should be member variables.

Updated [`IMPLEMENTATION.md`](./IMPLEMENTATION.md) §1–§5 and [`tests.md`](./tests.md) §3.2.

- Extra constexpr ctor on the existing value tokens: `{id, {format, face_r, face_c, nfr, nfc}}` (DFB), same trailing aggregate on scratch/tensor. Private `LlkOperandMembers`. Friend `to_llk_mem_descriptor`.
- `TensorBindingToken<CTA, CRTA>` keeps those NTTPs (address/layout seam). LLK facts are members. Accessor / `DataflowBuffer` / `Scratchpad<T>` signatures unchanged.
- Folding: `dfb::in` is already `constexpr`; reading members folds the same way `id_` does through `operator uint32_t()`.
- SPEC §12 “overload not available” cannot be SFINAE on one type. Conversion is `consteval` + sentinel `0xFF` / `static_assert(!is_dram)`. Negative JIT tests still fail `CompileProgram`; diagnostic is not “no matching function.”

---

## 2026-08-19 — Implementation plan (IMPLEMENTATION.md)

User: write an implementation.md for SPEC Parts I–II. Must cover Metal 2 layers, filegens, and additions alongside LLK APIs (not more behavior spec).

Wrote [`IMPLEMENTATION.md`](./IMPLEMENTATION.md). Plumbing choices SPEC left open:

- **Integer NTTPs on the token type** (HW `uint8_t` format + packed 4-byte shape). Not `ckernel::TensorShape` on the token — DM kernels compile the same token headers and must not include tt-llk.
- **`to_llk_mem_descriptor` reads the type, not `chlkc`.** DFB still keeps slot + `operator uint32_t()` for Gen1. Do not delete `set_dfb_data_fmt_and_tile`.
- **Sentinel format `0xFF`** + C++20 `requires` so format-less scratchpad / DM-only DFB / DRAM tensor have no matching overload (SPEC §12).
- **Slim token headers** (`dfb_binding_token.h`, `scratchpad_binding_token.h`) so the new `binding_token_llk.h` does not pull `dataflow_buffer.h` into every TRISC compile.
- **Host-only `LlkOperandFacts`** (SPEC §2 “normalized view”). Lift `host_data_format_to_hw` out of `genfiles.cpp` so tokens and `chlkc` arrays cannot drift. Share `compute_num_faces_rc_dims`’s per-operand split.
- Facts ride on binding handles → `JitBuildSettings` callbacks → `write_kernel_bindings_generated_header`. **Emulation re-emits that header** (`emulated_program_runner.cpp`) and must stay in lockstep. `compute_hash` gains the baked integers.
- Address stays on the memory object. Flag the DFB `get_read_ptr` vs `cb_read_address` (`fifo_rd_ptr - 1`) unit mismatch as the first thing a device test must verify.

Slices: **A** ScratchpadSpec + validation; **B** facts + handles + filegen + token NTTPs; **C** LLK overloads. Test files and cases follow [`tests.md`](./tests.md) (`test_llk_operand_metadata.cpp` / `test_llk_operand_hw.cpp` under Metal 2, not `unit_tests_llk`).

---

## 2026-08-19 — Test plan (`tests.md`)

User: write how we test the SPEC, given the API already seen. Tests live with Metal 2.0, not LLK.

Wrote [`tests.md`](./tests.md). Placement: new files under `tests/tt_metal/tt_metal/api/metal2_host_api/`, wired into `unit_tests_api` via `sources.cmake`. No cases in `tests/tt_metal/tt_metal/llk/` / `unit_tests_llk` (that suite owns CB-path id-free vs legacy). Do not dump into the 5k-line `test_program_spec.cpp`.

Layers:

- Host (`test_llk_operand_metadata.cpp`): Scratchpad aggregate/hash + the validation that differs from DFB (compute bind without format succeeds; geometry without format fails; arch-supported format). TensorParameter grows no fields; optional host mapper table if that helper appears.
- Mock-WH JIT (same file, `ProgramSpecTestGen1`): `to_llk_mem_descriptor` present/absent via `CompileProgram`; folded `static_assert`s for default tile, FaceGeometry override, 16×32 vs 32×16. Call the conversion, not BH-only ops.
- BH silicon (`test_llk_operand_hw.cpp`, `ProgramSpecHWTest`): one datacopy per object (DFB / format-bearing scratchpad / L1 tensor). Skip WH/Quasar.

Reuse `test_helpers.hpp` and existing scratchpad / LocalTensorAccessor tests as regression.

---

## 2026-08-19 — Freeze host + device behavior spec

User: host and device behavior have been sufficiently described; check.

Agreed. SPEC.md marked **Status: behavior specified** for Part I + Part II. Known non-goals left in §5 / §13 (token representation, `chlkc`, `unpack_modes` beyond DFB, `operator uint32_t()`, Bfp stride, LLK’s `uint8_t`/`DataFormat` split). Implicit token→`LLKMemDescriptor` conversion considered and rejected (existing `operator uint32_t()`, opacity, layering). Explicit `LLKMemDescriptor(token)` constructor is optional sugar, not required.

Next is implementation, not more behavior spec.

Follow-up: explicit `LLKMemDescriptor(token)` constructor parked — nice sugar, not now.

---

## 2026-08-19 — Device spec: BindingToken → LLKOperand (SPEC.md Part II)

User: device side next. Plumbing through BindingToken. #53193 already has `to_llk_mem_descriptor(Cb<CbId>)`. Token stays opaque (“construct A from this handle”). Next step is constructing LLKOperand’s compile-time half from it. Do not spec kernel-gen plumbing.

Wrote SPEC.md Part II. Decisions:

- BindingToken remains opaque. Kernels do not read format/shape/id. `operator uint32_t()` is not the id-free path.
- Reuse PR conversion: additive `to_llk_mem_descriptor` overloads on `DFBBindingToken` / `ScratchpadBindingToken` / `TensorBindingToken`. Takes the **token** (constexpr identity), not `DataflowBuffer`/`Scratchpad`/`LocalTensorAccessor` (runtime).
- Conversion returns `LLKMemDescriptor` only (the “what”). Address comes from the object already built from the same token (`get_read_ptr` / `get_base_address` / `get_bank_base_address`).
- Kernel welds them into `LLKOperand<Format, Shape>(addr)`, same pattern as `eltwise_copy_fp8_2_0.cpp`. A `make_llk_operand(token, addr)` helper is optional sugar.
- Format-less scratchpad: no conversion overload; `Scratchpad<T>(token)` still works. Closes host open Q1 without a ProgramSpec flag.
- Explicitly out of spec: NTTP vs sidecar vs leftover `chlkc` index on the token.

Follow-up: `static_cast<DataFormat>(in_desc.format)` in the PR kernel is LLK’s (`LLKMemDescriptor::format` is `uint8_t`; `LLKOperand` is parameterized by `enum class DataFormat`). Spec Part II no longer copies that cast into the Metal 2 kernel contract. Token → descriptor is ours; welding descriptor + address into `LLKOperand` is theirs.

---

## 2026-08-19 — Host-config spec (SPEC.md §0–§6)

User: start a real `spec.md`; host-side configuration first. The other two memory objects should carry the DFB LLK metadata (`dataflow_buffer_spec.hpp` 92–111). Dedicated struct is an open question; DFB refactor would churn use sites. Otherwise propose how to propagate.

Wrote [`SPEC.md`](./SPEC.md). Decisions:

- **No public nested struct** on the specs. DFB designated-init `.data_format_metadata = …` is hundreds of TTNN/test sites; C++20 designated init does not flatten nested/inherited members.
- **DFB:** leave the three fields.
- **ScratchpadSpec:** copy the same three optional fields (same names). Do *not* require format merely because a compute kernel binds the scratchpad (unlike DFB): binding ≠ LLK operand. Existing ~20 Scratchpad inits keep compiling.
- **TensorParameter:** do **not** duplicate format/tile — `spec.data_type()` / `spec.tile()` already have them (`datatype_to_dataformat_converter`). Add only optional `unpack_face_geometry_metadata`.
- Runtime-side normalized `{format, tile, face}` view is allowed later as impl, not host API.

Follow-up same day: user asked to double-check face geometry on MeshTensor. **Dropped the TensorParameter FaceGeometry field.** `Tile` already stores `face_shape`, `num_faces`, `narrow_tile`, `partial_face`; `compute_num_faces_rc_dims` builds `TensorShape` from that. `FaceGeometry` is a DFB/CB override for FIFO pages that are *not* a `Tile` (pool windows, conv scalar `{1,4}`). A MeshTensor page *is* `spec.tile()`. Tile is also strictly better: flat `num_faces==2` cannot distinguish 16×32 vs 32×16.

---

## 2026-08-19 — Close background; rename SPEC → BACKGROUND

User: this is the end of background gathering; tasks come in another conversation. Spill remaining context into the doc. Rename — it was never a spec of this branch’s work.

- Replaced `worklogs/SPEC.md` with `worklogs/BACKGROUND.md`.
- Dropped invented “runtime work breakdown” and “success criteria.”
- Kept LLK phases as *their* published plan, not ours.
- Filled remaining holes: CTA vs CRTA on tokens, FaceGeometry, unpack_modes / UnpackToDestEn, host DataFormat remaps, experimental pack out-of-order vs legacy auto-increment, `compute_kernel_hw_startup` still CB-id, Bfp stride limitation, TensorShape 2×1 vs 1×2 helper, untilize reconfig reason, binary format-free, git/PR targeting, example kernel paths.

---

## 2026-08-19 — Metal 2 host API + three memory objects

User: learn Program/Kernel/etc. from the Metal 2 host API; the three memory objects passed to device are DFB, MeshTensor, Scratchpad. After that, background should be clear.

Read `program_spec.hpp`, `kernel_spec.hpp`, `program.hpp`, `program_run_args.hpp`, `dataflow_buffer_spec.hpp`, `scratchpad_spec.hpp`, `tensor_parameter.hpp`, `node_coord.hpp`, `mesh_tensor.hpp`.

### Vocabulary (locked)

- **Node** = NOC endpoint in the grid (`NodeCoord`). RISC-V inside a node = core.
- **Kernel** = `kernel_main()` on baby RISC-V; compute or data-movement. `KernelSpec` = one compiled specialization. Instance = one copy per placed node.
- **WorkUnitSpec** = kernels that run together on a node set; placement is derived from this.
- **Program** = compiled executable enqueued to the device. `ProgramSpec` = immutable signature+body; `ProgramRunArgs` = per-enqueue arguments.

### Three L1/device memory objects

1. **DFB** — program-scope FIFO, one producer + one consumer per node. Token `dfb::`. Format/tile fields on the spec are the LLK leak. Credits stay on DFB.
2. **MeshTensor** — user-managed owning device allocation. Declared as `TensorParameter` (`TensorSpec`: dtype, layout, tile); value supplied at enqueue. Compute uses `LocalTensorAccessor` for the node-local L1 region. Token `tensor::` already type-NTTP but does not carry Format/Shape. DRAM tensors cannot be LocalTensorAccessor.
3. **Scratchpad** — program-scope private raw L1, no sync, size only. Token `scratch::`. No format/geometry on the spec.

All three can be compute L1 sources. Only DFB reaches LLK today, via CB slot. MeshTensor is the one whose host spec already has dtype+tile to pipe.

---

## 2026-08-19 — Base this work on #53193

User: build on the LLK branch. Create/push `riverwu/m2-neat-base` at the tip of #53193 for PR targeting; put `riverwu/m2-neat` on that base (no unique commits on current).

- Pushed `origin/riverwu/m2-neat-base` @ `eaf65b9ce23` (same as `rtawfik/l1spec-datacopy`).
- A literal `git rebase` of `m2-neat` (then at `main`) onto the older LLK tip tried to replay 157 main commits and hit a conflict. Aborted; `git reset --hard riverwu/m2-neat-base` instead, which is the empty-branch case.
- Restored `tt_metal/third_party/umd` to the LLK-recorded submodule commit. `worklogs/` still untracked.

`riverwu/m2-neat` was not pushed (no remote tracking before; user asked only to push the base). Future PRs from this branch should target `riverwu/m2-neat-base`, not `main`.

---

## 2026-08-19 — `LLKOperand` from PR #53193

### Why this session

Still background. User: current approach is to pipe metadata from BindingTokens; constexpr is critical to perf. Look at [PR #53193](https://github.com/tenstorrent/tt-metal/pull/53193); focus on `LLKOperand`’s declaration and what it needs. Do not switch `riverwu/m2-neat`.

### What I did

- Fetched `origin/rtawfik/l1spec-datacopy` into a detached worktree at `/workspace/.claude/worktrees/l1spec-datacopy` (`eaf65b9ce23`). This worktree stayed on `riverwu/m2-neat`.
- Read `tt_metal/hw/inc/api/compute/experimental/2_0/llk_mem_descriptor.h` and the compute/LLK consumers (`tile_move_copy`, `pack`, `tilize`, `eltwise_binary`, BH `data_format_derive.h`, example kernels).

### What `LLKOperand` needs (the contract)

Two types in `ckernel::experimental`:

```cpp
struct LLKMemDescriptor { uint8_t format; TensorShape shape; };

template <DataFormat Format, TensorShape Shape>
struct LLKOperand {
    uint32_t l1_address;  // only runtime member
    static constexpr LLKMemDescriptor descriptor = {uint8_t(Format), Shape};
};
```

- **What (constexpr / NTTP, must fold):** L1 `DataFormat` + `TensorShape` `{face_r_dim, face_c_dim=16, num_faces_r_dim, num_faces_c_dim}`. `-ftt-nttp` is required so per-format switches DCE. This is the perf constraint.
- **Where (runtime):** absolute L1 tile base. Address seam today is `cb_read_address` / `cb_write_address` (CB FIFO, `fifo_rd_ptr - 1 + page*index`).
- **Not on the operand:** CB id, register formats (derived in-LLK from `DESC.format` + dest-acc), FIFO wr_ptr as op state. Per-tile stride folds from `SCALE_DATUM_SIZE(Format, Shape)` (page == one linear tile).

Kernel pattern on the PR (still Phase 2 — CB arrays):

```cpp
constexpr auto in_cb = experimental::Cb<tt::CBIndex::c_0>{};
constexpr auto in_desc = experimental::to_llk_mem_descriptor(in_cb);  // indexes chlkc[id]
using InOp = experimental::LLKOperand<DataFormat(in_desc.format), in_desc.shape>;
experimental::copy_tile(InOp(in_cb.read_address()), 0);
```

`Cb<CbId>` puts identity in the **type**. `to_llk_mem_descriptor` is constexpr but still keyed on slot. Header comments future overloads: `to_llk_mem_descriptor(DFBBindingToken)` / `ScratchpadBindingToken` “once each exposes constexpr format + TensorShape.”

Today’s `DFBBindingToken` is only `{id}` + `operator uint32_t()`. Enough for a Phase 2 id→chlkc overload; **not** enough to drop the arrays. `TensorBindingToken<CTA, CRTA>` is the type-NTTP pattern to copy.

Host mapping we already have: `data_format_metadata` → Format; `tile_format_metadata` + `unpack_face_geometry_metadata` → `TensorShape` (same face-grid as `compute_num_faces_rc_dims`).

PR surface (BH only): datacopy, pack, tilize, pack_untilize, eltwise add/sub/mul. Binary is format-free at the op (geometry from A + two addresses). Math still takes `LLKMemDescriptor` even though it never touches L1.

### Spec updates

Rewrote §4–§5 around the real types (`to_llk_mem_descriptor`, not the issue’s `MemDescriptor` name). Recorded BindingToken pipe as the working assumption. Added open questions: token NTTP vs sidecar; host vs HW `DataFormat` encoding; Bfp stride limitation.

### Branch state

- This worktree: `riverwu/m2-neat` @ `main` (unchanged).
- Inspection worktree: `l1spec-datacopy` @ `eaf65b9` (leave in place for later diffs).

### Next

Wait for the user on what this branch implements. Background on `LLKOperand` is enough to start a token-pipe design.

---

## 2026-08-19 — Establish what this branch is for

### Why this session

`riverwu/m2-neat` is at `main` (`29b2bafa0f6`). No unique commits yet. Goal of the day: write down the background so later runtime work is aimed at a real contract, not at “make DFBs nicer.”

User framing (paraphrased): we work in `tt_metal/` runtime. Runtime owns the host specification of a Program. LLK is the device-side layer next to assembly. LLK needs operand metadata; that metadata was passed down as a hack through the CB interface (a FIFO meant for kernels to talk to each other). LLK is decoupling that metadata from CB id. Starting point: [issue #53456](https://github.com/tenstorrent/tt-metal/issues/53456).

### Sources

- [Issue #53456](https://github.com/tenstorrent/tt-metal/issues/53456) — id-free `LLKOperand` / `MemDescriptor`, three phases, Phase 0 comment from @rtawfik01 (2026-08-19).
- [PR #53193](https://github.com/tenstorrent/tt-metal/pull/53193) (draft) — experimental compute APIs with no CB id; branch `rtawfik/l1spec-datacopy`. Not merged; not in this worktree.
- Host DFB spec comments in `dataflow_buffer_spec.hpp` (open in editor): format / tile / face-geometry fields exist to feed LLK, only when a compute kernel is bound.
- Runtime compile path: `ProgramImpl::set_cb_data_fmt_and_tile` / `set_dfb_data_fmt_and_tile` → `tt_hlk_desc` → `jit_build_genfiles_descriptors`.
- Device: `copy_tile` / `pack_tile` still take `uint32_t` CB ids; `DFBBindingToken` implicitly converts to that id; Scratchpad / `LocalTensorAccessor` tokens do not.

### What we established

1. **The hack is real and two-sided.** CB id is both (a) the index into JIT `unpack_*`/`pack_*` arrays and (b) the handle for L1 FIFO pointers. LLK cannot fold format/geometry, and cannot treat Scratchpad or a local tensor as an operand, while that remains true.

2. **Metal 2.0 did not fix it; it renamed the FIFO.** `DataflowBufferSpec` still carries `data_format_metadata`, `tile_format_metadata`, `unpack_face_geometry_metadata`. At compile time `set_dfb_data_fmt_and_tile` casts `dfb->device_slot` to `CBIndex` and writes the same `hlk_desc` slots CBs use. `program_spec.cpp` documents that `unpack_modes` is translated into a `max_cbs`-long vector indexed by that slot — on WH, BH, *and* Quasar.

3. **BindingTokens are the Phase 3 hook that already exists.** `kernel_bindings_generated.h` emits `dfb::name`, `scratch::name`, `tensor::name`. Compute still does `copy_tile(dfb::in, …)` via `operator uint32_t()`. Scratchpad and LocalTensorAccessor already carry an L1 address (CRTA) and **do not** participate in `chlkc_descriptors.h`. That is the gap Phase 3 asks runtime to close.

4. **Host currently derives register formats.** `genfiles.cpp` `compute_data_formats()` turns L1 `DataFormat` into unpack src/dst and pack src/dst (dest-acc, exponent family, unpack-to-dest, Mx, Fp8). Issue #53456 wants that derivation inside arch LLK (`data_format_derive.h`). Phase 0 experimental APIs already stopped using JIT-inferred reg formats. Runtime validation (arch support, unpack-to-dest vs 16-bit dest, Blackhole Fp8 requiring `fp32_dest_acc_en`) is separate and should stay.

5. **This branch’s job is the runtime half of Phase 3**, with Phase 2 as a compatibility translator. We should not reimplement experimental compute APIs here; we should consume [#53193](https://github.com/tenstorrent/tt-metal/pull/53193) and then stop keying metadata on slot ids.

6. **DFB’s real job is unchanged.** Entry size, depth, producer/consumer identity, access patterns, borrowed L1, aliasing stay on `DataflowBufferSpec`. Only the LLK metadata piggyback is in scope.

### Branch state

- Worktree: `/workspace/.claude/worktrees/m2-neat`
- Branch: `riverwu/m2-neat`, clean, equal to `main`
- Artifacts added this session:
  - `worklogs/SPEC.md` (later renamed to `BACKGROUND.md`)
  - `worklogs/WORKLOG.md` (this file)

### Decisions (so far)

- Spec + worklog live under `worklogs/` in this worktree (internal planning; not a published metalium page).
- Follow LLK’s three phases; do not invent a fourth host-only metadata scheme.
- Keep legacy CB-id kernels working (Phase 2 translator). TTNN factory migration is out of scope.

### Open (carried into SPEC §7)

- Scratchpad / LocalTensorAccessor: host metadata vs call-site NTTPs?
- Math APIs: `LLKMemDescriptor` or shape+format only?
- When can `DFBBindingToken::operator uint32_t()` die?
- Confirm compute DFBs never need `fifo_page_size != tile size` once experimental APIs are the compute path.
- Minimum data-format matrix once JIT inference is gone.

### Next

- Track [#53193](https://github.com/tenstorrent/tt-metal/pull/53193) landing (experimental API shape is the device contract we bind to).
- If implementation starts here: inventory every `get_operand_*` / `unpack_*_format[id]` consumer in `tt_metal/hw/` so Phase 2 `to_mem_descriptor` has a closed list.
- Do not start deleting `set_dfb_data_fmt_and_tile` until Phase 3 token overloads exist.
