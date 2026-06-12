# Port Plan — nlp_create_qkv_heads

Port plan for `nlp_create_qkv_heads`, ported from `ProgramDescriptor` to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

**Scope of this pass:** the **Sharded** factory is ported to `create_program_spec`. The
**Interleaved** factory stays on legacy `create_descriptor` (grounded stop — see
[Deferred / Flagged](#deferred--flagged)). Mixed-concept variant; the framework dispatches per-factory.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (both `Interleaved` and `Sharded`, two `create_descriptor`s).
- Variants: `program_factory_t = std::variant<Interleaved, Sharded>`, chosen by
  `select_program_factory` on `input_tensor.is_sharded()`.
- Custom `compute_program_hash`: **none** — default reflection hash. (`validate_on_program_cache_hit`
  is present but empty, not a custom hash.)

### Variant: Sharded (ported)

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|
| reader | `reader_tm_tile_layout_nlp_create_qkv_heads_sharded.cpp` | `q_cores` | `{q_output_cb_index(16), k_output_cb_index(17)}` | 18 + noc-coord arrays (head_size, num_q_heads, …, q_base_addr, q_start_addr, q_offset, read_kv_heads, …, kv_base_addr, kv_start_addr, num_kv_tiles, num_x, then x[]/y[] via `get_arg_addr(19)`) | none (legacy) | none | Reader |
| writer | same source (bound twice) | `q_cores` | `{q_output_cb_index(16), v_output_cb_index(18)}` | same 18 + noc arrays (kv slots carry V) | none | none | Writer |

(No compute kernel on this path; the op's validate enforces `transpose_k_heads == false` for sharded.)

#### CBs
| index | total_size | core_ranges | data_format | page_size | borrowed buffer |
|---|---|---|---|---|---|
| c_16 (q_out) | q_num_tiles · single_tile_size | q_cores | input dtype | single_tile_size | **`std::get<0>(output).buffer()`** |
| c_17 (k_out) | k_num_tiles · single_tile_size | k_cores | input dtype | single_tile_size | **`std::get<1>(output).buffer()`** |
| c_18 (v_out) | v_num_tiles · single_tile_size | v_cores (= q_cores) | input dtype | single_tile_size | **`std::get<2>(output).buffer()`** |

All three are borrowed-memory, write-only address-source (fake) CBs: kernel grabs base via
`get_write_ptr()`, does explicit NoC unicast reads into borrowed L1, no real FIFO.

#### Semaphores
none

#### Tensor accessors
| host site | originating Tensor | legacy form | Metal 2.0 |
|---|---|---|---|
| reader/writer RTAs 6,7 (`q_base_addr`,`q_start_addr`) | `input_tensor_q` | `buffer()->address()` baked into RTA + per-core NoC walk (Case 2) | `TensorParameter`/`TensorBinding input_q` + `get_bank_base_address()` |
| reader RTA 15,16 / writer overwrite (`kv_base/start_addr`) | `input_tensor_kv` (when present) else `input_tensor_q` | `buffer()->address()` baked into RTA | conditional `input_kv` binding (gated `READ_FROM_INPUT_TENSOR_KV`) else derived from `input_q` base |

The sharded kernel uses **no `TensorAccessorArgs`** — reads are explicit `{noc_x,noc_y,addr}` unicasts.

#### Work split
- Driver: not `split_work_to_cores`. `num_cores = max(q_cores.num_cores(), k_cores.num_cores())`;
  per-core head counts derived from shard-spec core counts. Each output core hosts up to one Q head
  group split across the two RISCs (`per_risc0_out_q_heads`, `per_risc1_out_q_heads`).

### Variant: Interleaved (NOT ported — grounded stop)

#### Kernels
| unique_id | source | notes |
|---|---|---|
| reader | `reader_tm_tile_layout_nlp_create_qkv_heads.cpp` | in-dir; CB ids 0/1, `TensorAccessorArgs`, Buffer* RTA |
| writer | `writer_tm_tile_layout_nlp_create_qkv_heads.cpp` | in-dir; CB ids 1/16, `TensorAccessorArgs`, Buffer* RTAs |
| compute (×1–2) | **`ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp`** | **cross-op**, only when `transpose_k_heads` |

#### CBs
c_1 (always), c_0 + c_16 (only when `transpose_k_heads`). Not borrowed.

### Cross-op kernels
- **`ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp`** — outside the op dir, shared by 4 ops
  (`nlp_create_qkv_heads`, `nlp_create_qkv_heads_boltz`, `nlp_create_qkv_heads_vit`,
  `split_query_key_value_and_split_heads`). Bound only by the Interleaved factory's
  `transpose_k_heads` path. **Blocks the Interleaved port** — see Deferred / Flagged.

### Flags
- The Interleaved reader's `Buffer*`-RTA (`reader_rt.push_back(in0_buffer)`/`in1_buffer`) and the
  writer's `q/k/v_buffer` RTAs are Case-1 candidates — but they are not ported in this pass because
  the factory is blocked on the cross-op compute kernel (porting one kernel of an atomic factory
  unit is not a buildable sub-target).

## TTNN ProgramFactory
- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept` (Sharded factory).
- **Custom `compute_program_hash`**: none.
- **Implementation notes**: Mixed-concept variant — `Sharded` on `ProgramSpecFactoryConcept`,
  `Interleaved` left on `ProgramDescriptorFactoryConcept`. Both factory structs coexist in the one
  device-op `program_factory_t` variant; the framework dispatches per-factory.

## Planned Spec Shape (Sharded)
- **KernelSpecs**: 2 (`reader`, `writer`) — same source bound twice (the legacy idiom: reader binds
  CB16+CB17, writer binds CB16+CB18).
- **DataflowBufferSpecs**: 3 borrowed (`q_out`←Q_OUT, `k_out`←K_OUT, `v_out`←V_OUT).
- **SemaphoreSpecs**: none.
- **TensorParameters**: `input_q` (+ `input_kv` conditionally), plus `q_output`/`k_output`/`v_output`
  backing the borrowed DFBs.
- **WorkUnitSpecs**: 1 (`{reader, writer}` on `q_cores`).

## Preserved Multiplicity
none — no `split_work_to_cores` per-group CTA multiplicity in the sharded factory. (The "two
kernel descriptors of the same source" here is the reader/writer RISC split, modeled 1:1 as two
KernelSpecs; not a per-group-CTA work split.)

## Dropped Plumbing (Sharded)
| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader/writer CTA slot 0 (`q_output_cb_index`) | CB index 16 | `DFBBinding(Q_OUT_DFB, "q_out")` |
| reader CTA slot 1 (`k_output_cb_index`) | CB index 17 | `DFBBinding(K_OUT_DFB, "kv_out")` (self-loop) |
| writer CTA slot 1 (`v_output_cb_index`) | CB index 18 | `DFBBinding(V_OUT_DFB, "kv_out")` (self-loop) |
| reader/writer RTA 6 `q_base_addr` | `input.buffer()->address()` | `TensorBinding(input_q)` + `get_bank_base_address()`; RTA dropped |
| reader/writer RTA 7 `q_start_addr` | `q_base_addr + offset` | base from accessor + named RTA `q_start_offset` (kernel-side add) |
| reader RTA 15/16 `kv_base/start_addr` | `input_kv.buffer()->address()` (or `q_base_addr + offset`) | conditional `input_kv` binding (or `input_q` base) + named RTAs `kv_base_offset`/`kv_start_offset` |
| writer RTA 15/16 (V) | same | same, V offsets |
| noc-coord arrays via `get_arg_addr(19)`/`get_arg_addr(19+num_x)` | positional RTA tail read by pointer | `num_common_runtime_varargs` + `get_common_vararg(i)` (broadcast `[x.., y..]`) |
| all positional CTAs/RTAs | positional | named throughout |

## Applied Patterns
- **Borrowed-memory DFB** (`borrowed_from`): q_out/k_out/v_out borrow the output shard buffers.
- **Fake CB → self-loop DFB** (interim workaround): k_out self-looped on reader (PRODUCER+CONSUMER),
  v_out self-looped on writer; q_out split reader=PRODUCER / writer=CONSUMER (shared across the two
  co-located kernels, mirroring the reference `nlp_concat_heads_decode`).
- **Case-2 base-pointer bridge**: `get_bank_base_address()` for `input_q` (and `input_kv`).
- **Conditional / optional binding**: `input_kv` TensorBinding + `READ_FROM_INPUT_TENSOR_KV` define,
  `#ifdef`-gated kernel-side accessor.
- **Host-computed base-pointer offset → kernel-side addition**: the RTA offsets added to the
  accessor base on the kernel side (RTA-borne variant, mirroring the reference port).
- **Common runtime varargs**: NoC coordinate arrays.

## Deferred / Flagged
- **Interleaved factory — grounded stop (cross-op kernel).** The `transpose_k_heads` path binds the
  out-of-dir, multi-consumer `ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp`, which is not
  Metal-2.0-ready: it reads a **positional** CTA `get_compile_time_arg_val(0)` (would `static_assert`
  at JIT once the host emits only named args) and **hardcodes** physical CB indices
  `tt::CBIndex::c_0`/`c_16` rather than taking `uint32_t` CB ids as parameters (so `dfb::name` cannot
  be threaded in via the implicit `operator uint32_t`). Making it work requires *editing the kernel*,
  which is out of the porter's scope. The reader/writer of this factory also coordinate with the
  compute kernel through fixed physical CB indices (k routes through CB0→compute→CB16). Per the
  cross-op kernel rule this is a grounded stop: the Interleaved factory remains on legacy
  `create_descriptor`. See `METAL2_PORT_REPORT.md` → Handoff points.
- No YELLOW audit items.
