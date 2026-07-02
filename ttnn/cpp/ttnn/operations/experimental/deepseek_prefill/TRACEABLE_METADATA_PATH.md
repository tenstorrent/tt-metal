# Trace-safe metadata path for the chunked-prefill MLA ops

## Why

The Kimi/DeepSeek chunked-prefill `transformer.forward()` is captured as a ttnn trace to collapse the
per-op host-dispatch (op2op) gaps. A trace records the device command stream **once** and replays it; any
op argument baked into the program at capture is frozen, so an op that takes a **per-chunk scalar**
(e.g. the running KV length) cannot be replayed across chunks — the captured value would be wrong for
every chunk but the one captured.

Fix: move each per-chunk scalar out of the host dispatch path and into a small **metadata DRAM tensor**
that the op reads **on-device**. The program no longer depends on the per-chunk value, so one captured
trace replays across all chunks. The metadata tensor is the runner's `h2d_socket_sync` payload:

```
metadata: uint32 DRAM tensor, replicated across the mesh, canonical layout
          [slot_id, actual_start, actual_end]      (3 words, 12 bytes)
            index 0 = slot_id        (= cache_user_id)
            index 1 = actual_start   (= kv_actual_isl, the prior valid KV length, tokens)
            index 2 = actual_end     (pad-zero boundary)
```

Produced by `ttnn.H2DStreamService` + `ttnn.experimental.deepseek_prefill.inbound_socket_service_sync`
(see `models/demos/deepseek_v3_d_p/tt/runners/prefill_runner.py`,
`tt/runners/runner_utils.py::build_h2d_service`).

## Dual signature (per op)

Every converted op keeps its **original scalar signature** and gains a second **metadata** overload,
selected by whether a `metadata` tensor is passed. Both produce identical device results. The scalar
form stays the default for existing callers (e.g. `tt/mla/mla.py`); the metadata form is opt-in and is
the trace-safe one.

## Per-op status

| Op | Per-chunk value(s) → metadata index | Consumer kernel | Status |
|----|--------------------------------------|-----------------|--------|
| `update_padded_kv_cache` | `slot_idx`→0, `kv_actual_global`→1 | writer (dataflow) | **done** |
| `rotary_embedding_indexed` (Q + KV rope) | `kv_actual_global`→1 | reader (dataflow) | **done** |
| `zero_padded_kv_cache` | `slot_idx`→0, `valid_global`(=actual_end)→2 | reader + writer (dataflow) | **done** |
| `ring_mla` | `kv_cache_batch_idx`→0, `kv_actual_isl`→1, `logical_n`=`actual_start`+chunk | SDPA reader + all-gather reader (+ compute, task 4) | **in progress** |

`ring_mla` is harder: `kv_actual_isl` drives host-side ring-iteration masks / Q-mapping / valid-page
counts baked into runtime args, and its **compute** kernel needs derived values — needs its own design.

### ring_mla migration plan (incremental)

`ring_mla` is `ttnn.transformer.ring_mla` → the `ring_joint_sdpa` device op, with a fused
`ring_attention_all_gather_async` sub-program. A single optional `metadata` tensor is threaded through
(`tensor_args.metadata`, in the program hash via `has_metadata()`); the public API gains a `metadata=`
kwarg (single optional, not a C++ overload — the signature is too large). Each per-chunk scalar is
migrated on-device one at a time; a bit-exact `metadata == scalar` test
(`test_ring_mla_metadata_matches_scalar_*` in `tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py`)
gates each step, with `META_PATH_HOST_SCALARS` tracking which scalars are not yet on-device.

**Step 1 — `kv_cache_batch_idx` → `metadata[0]` (slot_id). DONE + verified bit-exact (slot 0/1) on 8×4.**
It is consumed in TWO kernels: the fused **all-gather reader** (gather offset
`input_batch_base = slot * num_heads * Ht * Wt`) and the **SDPA reader** (`ring_joint_reader.cpp`
local-KV slot, `kv_batch`). "Indexed mode" turns on when `kv_cache_batch_idx.has_value() || has_metadata()`
(host op + validation + all-gather validate). On the metadata path the host passes no slot scalar; each
reader reads `slot_id` from `metadata[0]` on-device.
- All-gather reader: `has_metadata` compile flag + dedicated meta CB (`c_in3`) + `HasMeta`-dependent
  `TensorAccessorArgs` offset (fallback to a VALID unused accessor offset, NOT 0 — offset 0 fails the
  accessor's internal `static_assert` at JIT time) + metadata DRAM address as a per-core runtime arg.
- SDPA reader: `slot_from_metadata` compile flag (slot 32, accessors shift to 33) + metadata accessor
  inserted between the tensor accessors and the chain-semaphore compile args (chain base offset shifted,
  flag-dependent) + metadata DRAM address as **common runtime arg 0** (raw address, mirrors update_cache).
- **Gotchas hit + fixed (both cost a device hang / JIT abort):** (1) the `TensorAccessorArgs<0>` fallback
  static_assert above; (2) a NoC read into a kernel **stack buffer hangs** — the destination must be a
  real L1 CB address. The SDPA reader reads into `cb_q_in`'s L1 scratch (free before the main loop fills
  it). Localized with `DPRINT` + `TT_METAL_DPRINT_CORES=all` (AG post-read fired, SDPA post-read didn't).
`has_metadata=false` ⟹ no metadata accessor appended ⟹ existing programs bit-identical (all other
ring-attention callers unaffected; verified by the scalar regression tests).

**Step 2 — `kv_actual_isl` / `logical_n` → `metadata[1]` (+ chunk_size_global). IN PROGRESS (the hard one).**
PIVOTAL CONSTRAINT: the **compute** kernel reads `logical_nt`, the 4 q-mapping tiles, and
`active_ring_iter_mask` as runtime args (`ring_joint_sdpa.cpp` lines ~116-121) and **cannot NoC-read
DRAM** — so it cannot read metadata itself. Therefore the broadcast design is REQUIRED (not optional):
the **SDPA reader** reads `actual_start` from `metadata[1]`, computes all derived values on-device, and
hands them to the writer + compute via a shared L1 region + a semaphore.

Per-chunk input is just `kv_actual_isl` → `logical_n = kv_actual_isl + chunk_global` → `logical_nt =
div_up(logical_n, 32)`; everything else is a pure function of `logical_nt` + static config:
- `logical_nt` — consumed by reader, writer, compute.
- `build_kv_pad_q_mapping` (factory `:257-305`) → 4 tiles (q_pre_wrap_start/count, q_post_wrap_start,
  q_valid_count) — consumed by **compute** (runtime args 117-120).
- `build_ring_work_plan_impl` (factory `:192-242`) → `active_ring_iter_mask` (reader/writer/compute) +
  `single_valid_kv_chunk_mask` (writer). Uses `RingIdSequencer` (shared device-usable struct in
  `ring_id_sequencer.hpp`) + `kv_global_tile_for_host_ring_plan` (`:174` → `chunked_kv_global_tile_for_local`
  in `chunked_prefill_utils.hpp`) compared against `logical_nt`.
- all-gather `gather_valid_Ht = ceil(logical_n/chunk_global) * (n_local_q/32)` — recomputed in the
  all-gather reader (already reads metadata for the slot).

DONE so far (committed): the device derivation header (`ring_joint_kv_pad_derivation.hpp`) and the host
rotation-enablement (`kv_pad_rotation_enabled = has_kv_pad_rotation() || (has_metadata() && is_chunked())`,
host q-mapping guarded on `kv_actual_isl.has_value()`, `compute_gather_valid_Ht` gate widened,
`kv_pad_from_metadata` local marked `[[maybe_unused]]`). No regression.

EXACT RESUME SPECIFICS for the kernel sync wiring (verified by inspection):
- Producer CB: add `cb_kv_pad_derived = allocate_cb(64, 1, UInt32)` (factory ~line 1441 area, via the
  `allocate_cb` lambda); append its id to `reader_cb_compile_time_args` (reader reads it at
  `cb_arg_offset + 3`) AND to `cb_compile_time_args` (compute reads it at `cb_arg_offset + 23`;
  compute `cb_arg_offset = 49`).
- `kv_pad_from_metadata` compile flag: reader add at fixed slot 33 (bump reader accessors `TensorAccessorArgs<33>` → `<34>`, which auto-shifts meta_args + chains_base_offset); writer + compute add at the end of their fixed compile-arg lists (bump their cb_arg_offset by 1).
- Reader ring params for `build_ring_work_masks_device`: `fused_op_receiver.seq.{ring_index, ring_size,
  expected[0]=backward, expected[1]=forward}` (constructed at reader line ~277, before the metadata read).
  chunk_global = `q_chunk_group_tile_count * 32`; kv_actual_tile_count = `metadata[1] / 32`.
- Reader: when kv_pad_from_metadata, read metadata[1] (already reads metadata[0] for slot), compute
  logical_nt/masks/q-mapping, override its own `logical_nt`+`active_ring_iter_mask` (mutable, lines
  275-276), and `cb_reserve_back`+write 6 u32+`cb_push_back` to cb_kv_pad_derived.
- Compute: when flag, `cb_wait_front(cb_kv_pad_derived,1)`, read 6 u32 via
  `reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb))`, override its rt-arg
  logical_nt/q-mapping/active_ring_iter_mask (lines 116-121), `cb_pop_front(1)`.
- Writer: when flag, re-read metadata[1] (needs metadata addr via emplace_common_runtime_args like the
  reader), recompute logical_nt + masks, override (lines 453-455).
- All-gather gather_valid_Ht on-device: DEFER (host value correct for the non-trace bit-exact test;
  needed only for actual trace replay).
- Test: rotation scenarios; metadata path passes kv_actual_isl=None (host can't compute q-mapping →
  kernels must → discriminating).

**RESOLVED + WORKING (task 4 core).** The full `kv_actual_isl`/`logical_n` metadata path is bit-exact vs
the scalar path: `test_ring_mla_metadata_matches_scalar_rotation[kv64/kv256/kv320]` pass on 8×4 (the
metadata path drops `kv_actual_isl`, so the q-mapping is derived SOLELY on-device). The compute kernel
reads the reader-produced scalars from `cb_kv_pad_derived` via **`ckernel::read_tile_value(cb, tile,
elem)`** (`api/compute/cb_api.h`) — the TRISC-safe UNPACK-mailbox CB read, mirroring `sparse_sdpa`'s
`cb_ctrl` (thanks to that op for the pattern). `read_tile_value` touches `cb_interface` only inside
`UNPACK({})` (trisc0) and mailboxes the value to MATH/PACK, so it links on all threads — the earlier
`CircularBuffer::get_read_ptr()` referenced `cb_interface` on MATH/PACK (trisc1), which does not link.
`element_offset` is a uint32 index (4 B), independent of CB format. REMAINING (trace-replay only, NOT
needed for the non-trace bit-exact test, since the host supplies correct logical_nt/masks/gather bound
from the passed `logical_n`): (1) WRITER recompute logical_nt + masks from metadata[1] (lines 453-455);
(2) all-gather reader recompute `gather_valid_Ht` from metadata. Until those land, an actual captured
trace would freeze the writer's masks + the gather extent.

(Historical blocker, now resolved:) the compute kernel reading the reader-produced CB initially failed to
LINK on TRISC: `undefined reference to cb_interface` (from `CircularBuffer::get_read_ptr()` /
`get_local_cb_interface` on trisc1/MATH). Fixed by `read_tile_value` as above. It's gated by `if constexpr (kv_pad_from_metadata)`, so it's discarded on the
scalar/indexed path (committed tests stay green) and only fails when the rotation metadata path is
active (the uncommitted `test_ring_mla_metadata_matches_scalar_rotation`). `cb_interface` is
`extern`/firmware-provided; the ring_joint compute kernel's `--just-symbols` weakened elf doesn't supply
it (the matmul `bmm_..._gathered` compute kernel DOES use `get_local_cb_interface` successfully — diff
its build/includes). CANDIDATE FIXES (pick one): (a) replicate whatever lets the matmul compute kernel
link `cb_interface`; (b) AVOID cb_interface in compute — pass `cb_kv_pad_derived`'s L1 base address from
the host (it knows CB addresses) as a compute compile/runtime arg, and read via a plain
`volatile tt_l1_ptr uint32_t*` at that address (no cb_interface lookup); (c) read the scalars through the
LLK tile path. Option (b) looks cleanest/most-portable. Reader producer + host gating + compute-consume
scaffold are all COMMITTED and dormant (no regression); only the compute CB-read mechanism is unresolved.

Plan: (1) new shared device header porting `compute_logical_nt` + `build_kv_pad_q_mapping_device` +
`build_ring_work_masks_device` (the static derivation params — k_chunk_tile_count, kv_local_padded_Nt,
num_local_k_chunks, q_chunk_group_tile_count, q_local_padded_Nt, num_joint_k_chunks, joint_seq_len,
kernel_chunked, kv_pad_rotation_enabled, kernel_is_causal, device_index, is_balanced — pass as the
reader's compile args). (2) Reader: read metadata[1], compute, write ~8 u32 to an L1 scratch CB, inc a
new "derived-ready" semaphore. (3) Writer + compute: wait the semaphore, read the values from the L1 CB
instead of their runtime args. (4) Host: alloc the L1 CB + semaphore; on metadata path stop pushing the
derived runtime args and don't require the kv_actual_isl scalar. (5) Test: drop `kv_actual_isl` from
`META_PATH_HOST_SCALARS` + add the rotation scenarios (aligned_min / midchip_straddle / lastchip /
rot_partial / multislab / allfull). This is a 3-kernel synchronized handoff — the most delicate change;
a sync bug hangs the device (recover `tt-smi -glx_reset`).

## Implementation pattern (shared by all three done ops)

Host (`device/<op>_device_operation.{hpp,cpp}`):
- `tensor_args_t.metadata` is `std::optional<Tensor>`; `operation_attributes_t` keeps the scalar(s),
  used only on the scalar path (0 on the metadata path).
- `compute_program_hash` includes `metadata.has_value()` so the scalar and metadata programs never
  collide. The per-chunk **values** are never hashed on either path (one cached program per layer).
- `create_descriptor` adds a `has_metadata` compile-time flag to the consumer kernel(s), appends a
  metadata `TensorAccessorArgs` (only when present), allocates a small L1 metadata-scratch CB, and puts
  the metadata tensor's raw DRAM address (else the scalar) in a common runtime arg.
- `MeshWorkloadFactory::override_runtime_arguments` patches that common arg on cache hits — the metadata
  address (metadata path) or the scalar(s) (scalar path) — since the buffer-binding fast path leaves
  raw-address/scalar common args stale otherwise.
- Two public overloads (top-level `.cpp`) + two nanobind `ttnn::overload_t` overloads, disambiguated by
  the differing positional arg type (`int` scalar vs `Tensor` metadata).

Consumer kernel (dataflow):
- Body is `template <bool HasMeta>` called from `kernel_main()` as `run_x<get_compile_time_arg_val(flag)>()`.
  This is required: `if constexpr` inside the non-template `kernel_main` would still **instantiate** the
  discarded branch's non-dependent templates and fail to compile. Inside the template, the metadata
  branch is genuinely discarded for the scalar program.
- The metadata `TensorAccessorArgs<offset>` offset is made **dependent on `HasMeta`**
  (`HasMeta ? cache_args.next_compile_time_args_offset() : 0`) so the scalar program never names an
  out-of-range compile-time-arg index (a fixed offset there static-asserts "Index out of range").
- Metadata read: `noc.async_read(meta_accessor, meta_cb, N, {.page_id=0})` (N = 8 or 12 B — only the
  needed indices), then `CoreLocalMem<volatile uint32_t>` to extract the fields.

### `zero_padded_kv_cache` — note the compute kernel

This op has a **compute** kernel (masks the partial boundary tile), and compute kernels cannot NoC-read
DRAM. Rather than a reader→compute control handoff (the `unified_routed_expert_ffn` UNPACK→mailbox
pattern), the kernels use an **unconditional CB protocol**: the reader always pushes `src`+`mask`, the
compute always multiplies exactly `Wt` tiles (`Wt` is a structural common arg readable by all three
compute threads), and the writer always pops the `out` tiles but only writes them back on the chip that
owns the partial (discarding them otherwise). The compute kernel is therefore path-agnostic and needs no
per-chunk value at all — no control CB, no mailbox, no multi-thread coordination. Only the dataflow
reader/writer read the metadata.

## Tests

Each op has an H2D-service equivalence test that proves **metadata path == scalar path, bit-exact**,
driving the metadata from a **real** `H2DStreamService` + `inbound_socket_service_sync` (not a hand-built
tensor), on an 8×4 mesh with `fabric_config=FABRIC_2D`:
- `models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_deepseek_prefill_update_padded_kv_cache.py::test_update_padded_kv_cache_metadata_matches_scalar`
- `models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_deepseek_prefill_rotary_embedding_indexed.py::test_rotary_embedding_indexed_metadata_matches_scalar`
- `models/demos/deepseek_v3_d_p/tests/test_zero_padded_kv_cache.py::test_zero_padded_kv_cache_metadata_matches_scalar`

The existing scalar-path op tests are the regression that the scalar path is unchanged.

## Verified (8×4 Blackhole, Kimi K2.6)

- update_padded_kv_cache: equivalence 3/3 dtypes bit-exact; op regression 6 passed.
- rotary_embedding_indexed: equivalence 2/2 offsets bit-exact; scalar regression 4 passed.
- zero_padded_kv_cache: equivalence 4/4 (740 / 2600 / 4512 windows, slot 0/1) bit-exact; scalar
  regression 10 passed.
- End-to-end `test_kimi_prefill_transformer_chunked_trace_kv_pcc` (L10, model's scalar callers): passed,
  min KV-cache PCC = 0.993906.

## Gotcha

A kernel JIT-compile failure segfaults the test process and can wedge an active ethernet core
("Timed out while waiting for active ethernet core … become active again"). Recover with
`tt-smi -glx_reset` before retrying.
