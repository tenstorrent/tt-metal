# L1 Ledger: nlp_create_qkv_heads_decode

Target: Quasar (`.claude/references/targets/quasar.md`), ttsim. Schema: `.claude/references/l1-footprint-discipline.md`.

**Op-allocated buffers: none.** The program declares no dataflow buffer, no scratchpad and no semaphore. The gathering kernel's NoC reads land directly in the three output shards, which are tensor-owned (allocated by `ttnn.allocate_tensor_on_device` from the specs of rule R5 before the dispatch) and bound to the kernel as tensors (`LocalTensorAccessor`). They are listed so that every byte the op touches on the core is accounted for; they cost the op's budget nothing.

Block axes (from `op_design.md` Blocking Model): `users`, `heads`, `head_dim_tiles`, `face_col`, `out_tile_row`.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `q_shard` (tensor `q`, not a CB) | `(q_rows/32) * head_dim_tiles`, page `out_page_bytes` | equal to capacity: it is the result | `{users: streams → 1 per core (R5), heads: spans → nh rows of q_rows, head_dim_tiles: spans → head_dim_tiles, face_col: spans → 2, out_tile_row: spans → q_rows/32}` | input dtype (bf16 Phase 0; fp32 TARGET) — dataflow-only payload, never in DEST | `gather` (NoC reads) | none in program (host / next op) | whole program and after (it is the output) | none — it is a distinct output tensor (R5: "each is a tensor of its own"), concurrently live with `k_shard`, `v_shard` |
| `k_shard` (tensor `k`) | `(kv_rows/32) * head_dim_tiles`, page `out_page_bytes` | equal to capacity | `{users: streams → 1, heads: spans → nkv rows of kv_rows, head_dim_tiles: spans → head_dim_tiles, face_col: spans → 2, out_tile_row: spans → kv_rows/32}` | input dtype, dataflow-only | `gather` | none in program | whole program and after | none — distinct output tensor, concurrent with q, v |
| `v_shard` (tensor `v`) | `(kv_rows/32) * head_dim_tiles`, page `out_page_bytes` | equal to capacity | as `k_shard` | input dtype, dataflow-only | `gather` | none in program | whole program and after | none — distinct output tensor, concurrent with q, k |

Capacity vs live set: capacity exceeds the *data* rows (nh of q_rows, nkv of kv_rows) by the tile padding of the head axis; that padding is fixed by the output contract (R5 shard `[ceil(n/32)*32, head_dim]`), not by this op. A staging buffer that would sit between the input and the shards was removed by Rule 3 pattern 1: the gather's destination is the output.

### Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `q_rows` | `ceil(nh/32)*32` ∈ {32, 64} (TARGET `head_rows`); 32 in Phase 0 | `tag_head_rows` ∈ SUPPORTED (validate) |
| `kv_rows` | `ceil(nkv/32)*32` ≤ `q_rows` | X3 (`nkv ≤ nh`) |
| `head_dim_tiles` | `W / (32*(nh+2*nkv))`; ≤ 8 over INPUTS (8192-wide row, 16+2·8 sections → hd 256); bounded in general by `W/96` | X4, X5 (validate) |
| `out_page_bytes` | tile size (2,048 B bf16, 4,096 B fp32), aligned to `ttnn.get_l1_alignment()` | dtype ∈ SUPPORTED |
| per-core budget | `ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes`; value on this target not recorded (no predicate of this op uses it: op-allocated footprint is 0) | — |

### Total per-core footprint

- Op-allocated: **0 B** (scales with no knob).
- Tensor-owned on the user's core: `(q_rows + 2*kv_rows)/32 * head_dim_tiles * out_page_bytes`. It scales with the output contract only, not with any block knob (`heads_per_block`, `GATHER_THREADS` change no byte). Llama bf16: (32+32+32)/32 · 2 · 2,048 = 12,288 B; Qwen bf16: 3 · 4 · 2,048 = 24,576 B; worst INPUTS × TARGET (`[1,1,1,8192]`, 16/8 heads, fp32): 3 · 8 · 4,096 = 98,304 B — all well under Quasar's 4 MiB L1.

## Data-movement budget

Chosen split: users → cores (pinned by R5), all of a user's heads gathered on its core straight into its shards (regime A). Phase 0 input is interleaved in L1.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input | 0 (Phase 0, L1 input); 1 in the deferred A-DRAM regime (each used row byte once; bf16 runs read as their 64 B aligned window → 2× the run bytes) | only row u of each tile row is read (R3), each run exactly once; no second pass | each used byte crosses the NoC once from its bank to the user's core (loopback when the page is in that core's bank): `W * es` bytes in `2 * input_width_tiles` reads. Padding rows (31 of 32 at users = 1) never move. |
| q, k, v | 0 | written in place in the user's L1 by the reads themselves | none |

Totals per tier (bf16): DRAM 0 B. NoC: Llama 6,144 B in 192 reads of 32 B; Qwen 12,288 B in 384 reads of 32 B (fp32: twice the bytes, same read count). Core-local copies: 0.

> Cheapest-traffic split considered: users → cores with direct gather into the shards (regime A) — it is the minimum: each used input byte crosses the NoC once, outputs zero times, no DRAM. Implemented: the same split. The alternatives (whole-tile staging B: 32× the NoC bytes; two-hop C: +1 local crossing per byte; heads-across-cores D: 2 NoC crossings per byte) are `rejected` rows in the regime table.
