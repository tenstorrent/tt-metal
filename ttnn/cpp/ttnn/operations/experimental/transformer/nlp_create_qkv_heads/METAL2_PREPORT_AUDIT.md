# Pre-Port Audit — nlp_create_qkv_heads

Feasibility audit for porting `nlp_create_qkv_heads` from the legacy `ProgramDescriptor`
API to Metal 2.0 `ProgramSpecFactoryConcept`. Device op: `NlpCreateHeadsDeviceOperation`
(`ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads/`).

## Scope

Two program factories live in the one device-op (`program_factory_t = std::variant<Interleaved, Sharded>`),
selected by `select_program_factory` on `input_tensor.is_sharded()`:

- `Interleaved` — DRAM-interleaved I/O. Uses reader/writer dataflow kernels **and** the shared
  cross-op compute kernel `transpose_wh.cpp` when `transpose_k_heads == true` (the pybind default).
- `Sharded` — L1 height-sharded I/O. Reader/writer only (the same `*_sharded.cpp` source bound twice).
  No compute kernel. The op's own validate enforces `transpose_k_heads == false` on this path
  (`nlp_create_qkv_heads_device_operation.cpp:106`).

Each factory is its own port unit (recipe: "the atomic unit of a port is one ProgramFactory").

## TTNN ProgramFactory

### Concept
`ProgramSpecFactoryConcept` (the only portable concept today).

### Fit
- Single vs multi-program: **single** — one `ProgramSpec` stamped across the mesh, per factory.
- Op-owned device resources: **none** — every tensor referenced is reachable from
  `tensor_args` (`input_tensor_q`, `input_tensor_kv`) or `tensor_return_value` (the `q,k,v` tuple).
  No factory-allocated scratch `MeshTensor`s or `GlobalSemaphore`s.
- Tensor-arg matching: strict (default; no relaxation applied or warranted).
- Legacy-to-Metal-2.0 shape: 1:1 with legacy per factory.

### Custom compute_program_hash
**none** — the device op uses the default reflection-based hash. (`validate_on_program_cache_hit`
exists but is empty; not a custom hash.)

### Stop signals
- **Interleaved factory: BLOCKED on a cross-op shared compute kernel.** Its `transpose_k_heads`
  path binds `ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp` — a shared kernel-pool file *outside*
  the op directory, used by 4 ops (this op, `nlp_create_qkv_heads_boltz`, `nlp_create_qkv_heads_vit`,
  `split_query_key_value_and_split_heads`). It is **not** Metal-2.0-ready: it (a) reads a *positional*
  CTA `get_compile_time_arg_val(0)` (the host emits only named args post-port → JIT `static_assert`),
  and (b) hardcodes physical CB indices `tt::CBIndex::c_0` / `tt::CBIndex::c_16` rather than taking
  `uint32_t` CB ids as parameters — so `dfb::name` cannot be threaded into it via the implicit
  `operator uint32_t`. Making it work would require *editing the kernel*, which is out of the porter's
  scope (out-of-dir, multi-consumer). Per the cross-op kernel rule this is a **grounded stop**:
  the Interleaved factory remains on legacy `create_descriptor`; the shared kernel needs Metal-2.0
  prep by its owner before this factory can port. Mixed-concept variant (one factory Metal 2.0,
  one legacy) is sanctioned and the framework dispatches per-factory.
- **Sharded factory: GREEN.** No compute kernel. The reader/writer kernel
  (`reader_tm_tile_layout_nlp_create_qkv_heads_sharded.cpp`) is in-dir and uses only CB-id CTAs,
  base-address RTAs (Case-2 NoC walk), and `get_arg_addr`-based noc-coordinate arrays — all of
  which have documented Metal 2.0 translations.

## Overall

**GREEN for the Sharded factory; grounded-stop for the Interleaved factory** (blocked on the
out-of-dir `transpose_wh.cpp`). Port the Sharded factory; leave Interleaved on legacy.
