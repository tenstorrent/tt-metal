# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/sharded/sharded_to_interleaved`

Single device-operation directory:

- **`ShardedToInterleavedDeviceOperation`** (`device/sharded_to_interleaved_device_operation.{hpp,cpp}`)
  - `ShardedToInterleavedProgramFactory` (`device/sharded_to_interleaved_program_factory.cpp`) — one `descriptor` factory that selects kernels by input layout and dtype-conversion need.

**Kernels exercised** (all file-path-instantiated; the op owns none of its kernels):

| Role | Path | Selected when |
|---|---|---|
| Reader | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` | always |
| Writer (tiled) | `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp` | `input.layout() == TILE` |
| Writer (row-major) | `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | `input.layout() == ROW_MAJOR` |
| Compute (copy) | `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` | `convert_df` (input dtype ≠ output dtype; TILE only) |

**Config matrix** (the factory has three reachable shapes):

- **C1 — TILE, no conversion**: reader + tiled writer.
- **C2 — TILE, conversion** (`convert_df`): reader + tiled writer + compute.
- **C3 — ROW_MAJOR** (never converts; dtype-mismatch requires TILE per `validate_inputs`): reader + RM writer.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** *unpinnable* — this audit ran against `/localdev/edwinlee/metal2_audit.md` (not a repo-tracked file). The repo path `docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/` does not exist in this checkout, so `git log` yields no provenance hash. The Metal 2.0 `analyses/` triage docs (offset-base-pointers, 3rd-arg) and the `port/metal2_port.md` recipe are also absent from this checkout; findings below rely on first-hand code reads (the intended source of truth) and the live readiness sheet.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/sharded/sharded_to_interleaved` |
| **Overall** | **RED** — 2 independent, config-scoped gate blockers; clean subset **C1 (TILE, no conversion)** survives |
| **DOps / Factories** | `ShardedToInterleavedDeviceOperation` → `ShardedToInterleavedProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **No (RED)** — compute kernel `eltwise_copy.cpp` on legacy free-function CB API; config-scoped to C2. Small/self-contained, not broad Device 1.0. → Device 2.0 track |
| *Prereqs* — Cross-op escapes | Ok (no function-call escapes; all 4 kernels file-path-borrowed → port-together coupling, FYI) |
| *Feature Support* — overall | **GREEN** (all Appendix A entries N/A) |
| *Feature Support* — Variadic-CTA | N/A (all CTA reads at constexpr index 0) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (sheet; cross-check clean) |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD | N/A (not `WorkloadDescriptor`) |
| *TTNN Readiness* — Is safe to port? | Yes |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Runtime-args update | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` |
| *Port work* — Offset base pointer | **GATE (RED)** — RM writer, Type 2 (accessor-fed interior offset); config-scoped to C3 → ops team + framework/Audrey (flag early) |
| *Port work* — Tensor bindings (per binding) | input = clean (borrowed-memory DFB); output = Case 1 in C1/C2 (tiled), offset-gated in C3 |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none (no 3rd-arg sites) |
| *Port work* — CB endpoints | all legal 1:1 (per config) |

## Result

**RED at op level; subset C1 (TILE input, no dtype conversion) is clear.**

Two independent gates block the full-op port, each confined to one config branch of the single factory:

1. **Device 2.0 — compute kernel not migrated** (config **C2**: TILE + dtype conversion). `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` still uses the legacy free-function circular-buffer API. → **Device 2.0 migration team.** Cheap, self-contained; op returns for a re-audit once it lands.
2. **Offset base pointers — Type 2 accessor-fed interior offset** (config **C3**: ROW_MAJOR, width/block-sharded). The RM writer feeds `dst_addr + input_width_offset_bytes` as a `TensorAccessor` base; Metal 2.0's `tensor::name`-built accessor has no seam for a non-base start. → **ops team + framework/Audrey, flag early** (needs design discussion, not a mechanical port step).

The clean subset **C1** exercises only the reader and the tiled writer — both Device 2.0-compliant, clean-base — and is portable today on the readiness/feature/3rd-arg axes. **Caveat:** because all three configs live inside a *single* factory (kernel choice is a runtime branch on layout/dtype, not a separate `ProgramFactory`), the practical porting unit is the whole factory; the C1 subset cannot be shipped as an isolated factory port. It is offered to show the blockers are branch-local and small, so both clear quickly.

**Neither blocker is a permanent wall.** The Device 2.0 gap is a routine straggler-kernel migration (the same conversion already applied to sibling shared compute kernels, e.g. `transpose_wh.cpp`). The offset-base gap has a plausible resolution (below) pending an ops/framework decision.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** Readiness sheet ("TTNN Operations analysis", `dgomez@`, modified 2026-07-23) row:
  `data_movement/sharded/sharded_to_interleaved | ShardedToInterleavedDeviceOperation | ShardedToInterleavedProgramFactory | Concept=descriptor | Custom hash=no | Runtime-args update (get_dynamic_runtime_args)=no | Runtime-args update (PD override)=no | Pybind descriptor=no | Smuggled pointer=no | Is safe to port?=yes | Is able to port?=yes | Op-owned tensors?=(no) | TensorParameter relaxation=none`.
  Cross-check against code — all confirmed:
  - `Concept=descriptor` ✓ — `create_descriptor()` returns `tt::tt_metal::ProgramDescriptor` (`device/sharded_to_interleaved_program_factory.hpp:15`).
  - `Custom hash=no` ✓ — no `compute_program_hash` override anywhere in the op dir.
  - `Runtime-args update=no` ✓ — no `get_dynamic_runtime_args` / `override_runtime_arguments` in the op dir.
  - `Pybind descriptor=no` ✓ — `sharded_to_interleaved_nanobind.cpp` binds only the `sharded_to_interleaved` function; no `create_descriptor`/`nb::class_` of the device op.
  - `Op-owned tensors?=no` ✓ — consistent with `descriptor` concept (cross-column invariant holds).
  No spreadsheet conflict. **Target concept: `MetalV2FactoryConcept`** (no op-owned tensors).

- **Device 2.0 (every kernel used):** **RED**, one kernel, config-scoped to **C2**.

  Reader and both writers are fully Device 2.0-native: `Noc`, `DataflowBuffer`, `TensorAccessor`, `noc.async_write` / `wait_front` / `pop_front` (migrated in PRs #49392 / #49410). `get_tile_size(cb_id)` (tiled writer :26) is a **sanctioned** free function — not a violation. The compute kernel is the sole holdover:

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` | 20 | `cb_wait_front(tt::CBIndex::c_0, 1)` | none |
  | `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` | 21 | `cb_reserve_back(tt::CBIndex::c_16, 1)` | none |
  | `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` | 28 | `cb_pop_front(tt::CBIndex::c_0, 1)` | none |
  | `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` | 29 | `cb_push_back(tt::CBIndex::c_16, 1)` | none |

  **Sizing (for the Device 2.0 team):** *not* broad Device 1.0 — there are no addr-gen idioms (`InterleavedAddrGen`/`ShardedAddrGen`), no raw `noc_*`, no raw sem addresses. It is the FIFO-CB free-function API that the migration guide maps to `CircularBuffer`/`DataflowBuffer` object methods (`cb_wait_front(cb,n)` → `cb.wait_front(n)`, etc.). Slightly more than a 1-line in-scope-wrapper holdover only because **no wrapper object is currently in scope** — the fix constructs two wrappers (for `c_0` and `c_16`) and rewrites the four calls. This is exactly the conversion already applied to the sibling shared compute kernel `ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp` (`CircularBuffer cb_in(tt::CBIndex::c_0); cb_in.wait_front(1); …`). Route to the Device 2.0 track; the port whitelist cannot absorb it, so the op re-audits after the migration lands. **Owning family/pool:** the shared kernel pool `ttnn/cpp/ttnn/kernel/compute/` — co-consumers are `data_movement/copy` (SameMemoryConfig) and `data_movement/untilize_with_unpadding`, so the migration must land as one shared rewrite (see Team-only coupling).

- **Feature compatibility:** every Appendix A entry scanned; all **N/A** (no `GREEN` row exists — each entry is a gate-feature, and none is present).

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no `GlobalCircularBuffer` type, `.global_circular_buffer` field, `remote_cb`/`.remote_index`. The input CB is a plain Buffer-backed (borrowed-memory) CB — `cb.buffer = src_buffer` (`program_factory.cpp:41,147`) — which is a mechanical `DataflowBufferSpec::borrowed_from` translation, **not** a GCB and **not** an Appendix A entry. |
  | CBDescriptor `address_offset` (non-zero) | N/A | `push_s2i_cb_pair` never sets `.address_offset` (defaults 0); no `set_address_offset` anywhere. |
  | GlobalSemaphore | N/A | the op uses no semaphores of any kind. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | every `get_compile_time_arg_val` read is at constexpr index 0 (reader :13, tiled writer :22, RM writer :19, compute :12); `tensor_args_t` is a fixed pair (`input_tensor` + optional `preallocated_output`), no `std::vector<Tensor>`; no runtime-varying CTA loop. |

- **Offset base pointers:** **RED — Type 2 (accessor-fed interior offset)**, config-scoped to **C3** (ROW_MAJOR).

  **Address RTAs resolved.** The output buffer reaches both writers via a `Buffer*` binding, not a smuggled `->address()`: `writer_rt.push_back(dst_buffer)` (`program_factory.cpp:242` tiled, `:293` RM) → arg 0. The framework auto-registers this as a `BufferBinding` and patches it on cache hits, so it is correct-on-cache-hit today (consistent with the sheet's `Smuggled pointer=no`). The kernel then receives a raw `uint32_t` base; classify by what the kernel does with it:

  - **Tiled writer (C1/C2):** `TensorAccessor(dst_args, dst_addr)` (`writer_unary_sharded_blocks_interleaved_start_id.cpp:28`) — **clean base.** All addressing is by `page_id = tile_id` / `start_id` (tile indices, not byte offsets). No fold. → ordinary Case-1 port work (see Tensor bindings).
  - **RM writer (C3):** `TensorAccessor(dst_args, dst_addr + input_width_offset_bytes)` (`writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp:22`). `input_width_offset_bytes` = arg 5 = host `curr_idx_w` (`program_factory.cpp:298`), the byte column-offset of this core's shard block within each output stick. It is **non-zero on every core after the first** for WIDTH- and BLOCK-sharded RM (zero for HEIGHT-sharded RM, where a core spans the full row). The offset is baked into the **accessor base** — this is the [Type 2] shape: the offset *is* the accessor's start, not a relocatable trailing `+`, so Metal 2.0's `TensorAccessor(tensor::dst)` (base-only, args auto-built) has no seam to express it.

  **Why this gates even though there is no host-side fold.** The offset-base recognition model keys on host folds (`buffer()->address() + <expr>` computed in the factory). Here the host passes a **clean base + a separate scalar offset arg**, and the kernel adds them — which the recipe's four-outcomes table treats as the *already-split-out / GREEN* case (cf. `roll` DRAM_RM). **But that GREEN disposition presumes the split-out offset is consumed *raw* (Case 2), not fed to a `TensorAccessor` base.** Here it *is* fed to the accessor base, so the exact Type-2 Metal 2.0 wall applies regardless of how the offset was delivered. A mechanical Case-1 port (`TensorAccessor(tensor::dst)`) would **silently drop** `input_width_offset_bytes` → wrong numerics on cache hits, only in width/block-sharded RM. Per the operating principle (identify gaps; default conservative when the failure mode is silent mis-addressing), this is gated. See Recipe notes for the recognition-rule gap.

  **Plausible-but-unsettled resolution (for ops + framework):** the RM writer's `async_write` dst-args already carry an `.offset_bytes` field (currently `{.page_id = stick_id, .offset_bytes = 0}`, `:32`). Moving `input_width_offset_bytes` from the accessor base into that per-write `.offset_bytes`, with a clean-base `TensorAccessor(tensor::dst)`, is *likely* equivalent for interleaved addressing — but it is a kernel-logic change (off the mechanical porter whitelist) and its correctness across sharding/alignment cases needs framework confirmation. Alternative: a base-binding + kernel-side accessor construction from `tensor::dst.args` + `get_bank_base_address() + offset`. Either way this is a design decision, not a porter task — **flag early.**

  *Triage-doc cross-check:* the dated `analyses/2026-07-19_offset_base_pointers.md` is **absent from this checkout**, so it could not be consulted; this finding is from first-hand code read. Recommend the triage-doc owner reconcile whether `sharded_to_interleaved` (RM writer) is catalogued.

- **TensorAccessor 3rd argument:** **GREEN / none.** Both `TensorAccessor` constructions are 2-arg (`(dst_args, dst_addr)` / `(dst_args, dst_addr + offset)`) — no explicit page-size third argument at any site. (The 3rd-arg triage doc is also absent from this checkout, but no site fires the syntactic signal.)

- **CB endpoints (GATE-free):** all CBs are legal **1 producer + 1 consumer** on every node in every config — no self-loop, multi-binding, or dead CB. Device 2.0 idioms are intact for the scanned kernels (compute-kernel holdover does not obscure the census: its CB touches are plain FIFO `cb_wait_front`/`cb_pop_front`/`cb_reserve_back`/`cb_push_back`).

  | CB | Config | Producer | Consumer | Verdict |
  |---|---|---|---|---|
  | `c_0` (`src0_cb_index`, borrowed-memory, `cb.buffer=src_buffer`) | C1 (no convert) | reader `dfb.push_back` (`reader_unary_sharded.cpp:16`) | writer `dfb_out.wait_front`/`pop_front` (`out_cb_index==c_0`) | 1P+1C legal |
  | `c_0` | C2/C3 (convert / RM) | reader `dfb.push_back` | compute `cb_wait_front`/`cb_pop_front` (C2) · RM writer `wait_front`/`pop_front` (C3) | 1P+1C legal |
  | `c_16` (`out_cb_index`, convert only) | C2 | compute `cb_reserve_back`/`cb_push_back` | tiled writer `dfb_out.wait_front`/`pop_front` | 1P+1C legal |

## Port-work summary  *(reference; no brief issued on RED)*

- **Tensor bindings** (per binding):
  - **input_tensor** (`c_0`, borrowed-memory) — **clean** (borrowed-memory DFB read: the reader only `push_back`s the globally-allocated shard CB; no `TensorAccessor`). Port via `DataflowBufferSpec::borrowed_from`.
  - **output_tensor** (`dst_buffer`, delivered via `Buffer*` binding) — **Case 1** in C1/C2 (tiled writer, clean-base `TensorAccessor`) → express as `TensorParameter`, kernel builds `TensorAccessor(tensor::dst)`. In **C3** (RM) this same binding is **offset-gated** (Type 2, above), *not* a clean Case 1 — do not port it mechanically.
- **TensorParameter relaxation:** none.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** all legal 1:1 (per config, table above).

## Team-only

- **Out-of-directory coupling & donor shape:** roll-up **✓ clean** for function-call escapes (no kernel `#include`s an out-of-directory op helper; all kernel includes resolve to `tt_metal/*` LLK/HAL headers). The host factory `#include`s the in-family `sharded_common.hpp` for `calculate_starting_idx_h` — host code, no kernel-token bridging. **File-path kernel instantiation is the whole story here: the op owns none of its kernels.** Each borrowed kernel forms a Metal 2.0 **port-together set** (a shared kernel's CB→DFB / named-token rewrite is one change all co-borrowers must adopt together):

  | Borrowed kernel | Owning family / pool | Co-borrowers (same-repo scan) |
  |---|---|---|
  | `reader_unary_sharded.cpp` | `eltwise/unary` (cross-family) | broadly shared — typecast, tilize (×2), transpose_wh_sharded, untilize (×3), untilize_with_unpadding, sharded_to_interleaved_partial, slice_write (×2), + `experimental/quasar/*` variants |
  | `writer_unary_sharded_blocks_interleaved_start_id.cpp` | `data_movement/sharded` (in-family) | sharded_to_interleaved_partial, `experimental/quasar/sharded_to_interleaved` |
  | `writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | `data_movement/sharded` (in-family) | sharded_to_interleaved_partial, `experimental/quasar/sharded_to_interleaved` |
  | `eltwise_copy.cpp` | `ttnn/cpp/ttnn/kernel/compute/` (shared pool) | copy (SameMemoryConfig), untilize_with_unpadding, sharded_to_interleaved_partial |

  Note the same broadly-shared kernels are also used by `experimental/quasar/*` ops — out of scope for this Gen1 audit, but relevant to whoever sequences the shared-kernel rewrite. The Device 2.0 straggler `eltwise_copy.cpp` in the shared compute pool is itself a port-together concern: its Device 2.0 migration equally unblocks `copy` and `untilize_with_unpadding`.

- **Relaxation candidates (mined from custom hash):** none — the op has no custom hash.

- **TTNN factory analysis (sheet-derived + `file:line`):** current concept `descriptor`; no op-owned tensors; no pybind `create_descriptor`; no custom hash; no custom `override_runtime_arguments`; `Is safe to port? = yes`; target concept `MetalV2FactoryConcept`. All gate conjuncts absent → gate clears.

## Misc anomalies  *(team-only, non-gating; the port does not act on these)*

- **Dead RTA in the RM writer.** The factory pushes 7 writer RTAs for the RM path — arg 1 = `num_units_per_row` (`program_factory.cpp:294`) — but `writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` reads only indices 0,2,3,4,5,6 and never reads index 1. `num_units_per_row` is a dead RTA on the RM path.
- **`is_l1_aligned` is a hardcoded `true`** (`program_factory.cpp:55`), which makes the RM-path guard `if (is_blackhole or is_l1_aligned) { if (!dst_is_dram or is_l1_aligned) { … } }` (`:286-289`) unconditionally taken. Consequently `is_blackhole` (`:135`) and `dst_is_dram` (`:134`) are computed but effectively dead in this branch (`dst_is_dram` has no other use). Suspicious dead branch / forced constant.
- **`num_slices` / `slice_index` are always `1` / `0`** for this (non-partial) op — hardcoded at the launch site (`sharded_to_interleaved_device_operation.cpp:147`, `ShardedToInterleavedParams{…, 1, 0}`). `calculate_starting_idx_h(output, 1, 0)` therefore always yields the base start id; the slicing parameters are vestigial here (the real user is the separate `sharded_to_interleaved_partial` op). Not a bug, but dead generality carried into the hash-relevant attributes.

## Questions for the user

1. **Offset-base disposition on the RM writer:** the recognition rule as written (host-fold-only) would pass this site to TensorParameter analysis, where a mechanical Case-1 port silently drops the interior offset. I gated it conservatively as Type 2. Please confirm routing to ops + framework/Audrey, and whether the "move the offset into per-write `dst_args.offset_bytes`" resolution is acceptable/blessed (it is a kernel-logic change, off the mechanical porter whitelist).
2. **Missing checkout docs:** the Metal 2.0 `analyses/` triage docs and `port/metal2_port.md` are not present in this checkout (only `metal2_audit.md` and `DFB_PORTING_GUIDE.md` at `/localdev/edwinlee/`). The offset-base and 3rd-arg dated priors could not be consulted. Findings rely on first-hand reads; flagging so the triage-doc owner can reconcile whether `sharded_to_interleaved` (RM writer) is catalogued.

## Recipe notes  *(friction with the audit recipe itself)*

- **Recognition-rule gap: kernel-side offset fed to a `TensorAccessor` base.** The [Offset base pointers] subject recognizes only *host-side* folds (`buffer()->address() + <expr>` in the factory), and its four-outcomes table classifies "clean base + separate scalar offset arg, added in the kernel" as GREEN → hand to [TensorParameter analysis]. But that GREEN disposition is only correct when the split-out offset is consumed **raw** (Case 2). When the kernel adds the offset and feeds the result to a `TensorAccessor` **base** (as this op's RM writer does), the Type-2 Metal 2.0 wall applies identically, yet neither subject catches it: the offset gate sees "no host fold," and TensorParameter analysis would call it Case 1 and silently drop the offset. Suggest the offset-base recognition add a kernel-side clause ("an `->address()`/base RTA plus a separately-delivered offset that are **summed and passed as a `TensorAccessor` base** is Type 2, regardless of where the sum is computed"), and that the four-outcomes "No fold → clean → TensorParameter analysis" bullet be qualified with "provided the base reaches the accessor unmodified." The `roll` DRAM_RM precedent cited for the GREEN case should state explicitly whether its split-out offset is raw-consumed or accessor-fed.
- **Compute-kernel scope of the Device 2.0 gate is implicit.** The gate text and migration guide are framed as "Data Movement," which invites the reading that compute kernels are out of scope for CB-wrapper migration. In practice compute kernels *are* being migrated (`transpose_wh.cpp`, bcast compute kernels in the DFB PRs use `CircularBuffer`), and `cb_wait_front`/`cb_pop_front`/`cb_reserve_back`/`cb_push_back` in a compute kernel are gate violations. A one-line note that the CB-FIFO free functions gate in compute kernels too (only `get_tile_size` / `get_local_cb_interface` are sanctioned) would remove the ambiguity.
