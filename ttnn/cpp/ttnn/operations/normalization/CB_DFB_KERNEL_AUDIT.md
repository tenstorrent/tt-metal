# CB→DFB Kernel Audit: `normalization` family

**Date:** 2026-07-13
**Op root:** `ttnn/cpp/ttnn/operations/normalization/`

**Scope:** Device kernels for `batch_norm`, `groupnorm`, `layernorm`, `layernorm_distributed`, `rmsnorm`, `rmsnorm_distributed`, `softmax`, plus shared `kernel_util/compute/memory.h`. Host `packer_l1_acc` config destructuring in program factories is **not** a kernel field access — excluded.

## Overall verdict: RED (driven solely by `groupnorm`)

**Summary:** `softmax`, `batch_norm`, `rmsnorm`, and the non-Welford `layernorm*` paths are canonical Class 1 → portable. The **Welford family** (layernorm / layernorm_distributed / groupnorm welford kernels) reads the `cb_reciprocals` LUT via `get_pointer_to_cb_data` / `get_tile_address` in the shared `kernel_util/compute/memory.h` → **Portable (prereq: LTA)** on 1xx and **2xx-blocked** until the Quasar DFB read API lands (single `memory.h` fix). The one hard blocker is **`groupnorm_zero_fill.hpp`**, which reads `fifo_size` off `get_local_cb_interface` — a field with **no getter today** → **Blocked** until `get_total_buffer_size_bytes()`/ring-span getter lands (file to Almeet). That makes the family rollup RED.

## Per-op rollup

| Sub-op | Rollup | Driver |
|--------|--------|--------|
| `batch_norm` | GREEN | Canonical FIFO only |
| `softmax` | GREEN | Canonical FIFO only (`packer_l1_acc` is host config) |
| `rmsnorm` | GREEN | Canonical FIFO (reuses layernorm non-welford kernels) |
| `rmsnorm_distributed` | GREEN | Canonical FIFO only |
| `layernorm` | YELLOW | Welford `cb_reciprocals` → LTA + `memory.h` `get_tile_address` (2xx) |
| `layernorm_distributed` | YELLOW | Same Welford LUT pattern (`layernorm_pre_allgather_welford.cpp`) |
| `groupnorm` | **RED** | `groupnorm_zero_fill.hpp` `fifo_size` read (no getter) **+** Welford LUT |

## CB portability (notable rows; unlisted CBs are Class 1 → Portable both arches)

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_reciprocals` | 6 | `layernorm/.../layernorm_welford.cpp` (106), `layernorm_sharded_welford.cpp` (288), `layernorm_large_tensor_welford.cpp` (393), `layernorm_distributed/.../layernorm_pre_allgather_welford.cpp` (75), `groupnorm/.../welford_groupnorm.cpp` (247) via `kernel_util/compute/memory.h` | Portable (prereq: LTA) | sync-free borrowed reciprocal LUT → **LocalTensorAccessor** (replaces `get_pointer_to_cb_data`) | Blocked (runtime) | `memory.h` `get_tile_address` needs Quasar DFB read API (single fix unblocks all Welford kernels) |
| `cb_ex_external` (zero-filled) | 6 | `groupnorm/.../groupnorm_zero_fill.hpp` (38,40), incl. by non-welford mcast reader | Blocked | **GATE:** `get_local_cb_interface(cb).fifo_size` read — **no getter today** → file issue to Almeet (`get_total_buffer_size_bytes()` / ring-span). Not a mechanical swap. | Blocked | same |

## GATE hits (must be empty to merge)

- `groupnorm/device/kernels/dataflow/groupnorm_zero_fill.hpp:38,40` — `auto& iface = get_local_cb_interface(cb_id); ... iface.fifo_size` read — **no existing getter**. File to Almeet before porting groupnorm; alternative is to pass the backing byte size in as a compile-time/runtime arg (removes the field read entirely) or migrate `cb_ex_external` to `ScratchpadSpec`.

## Blocked on runtime (2xx rollup)

- `get_tile_address` / `read_tile_value` on Quasar DFB → all Welford `cb_reciprocals` reads (`memory.h`). Single `memory.h` fix unblocks layernorm + layernorm_distributed + groupnorm welford kernels. 1xx path is clear (LTA).
- `get_total_buffer_size_bytes()` / ring-span getter → `groupnorm_zero_fill` `fifo_size`. Needed on both arches (this is the RED driver).

## Notes

- The `memory.h` `get_tile_address` path is **QUASAR-BLOCKED, not GATE** — it is a sanctioned DFB read API path, not `get_local_cb_interface` field access. On 1xx the `cb_reciprocals` LUT ports via **LocalTensorAccessor** (host `TensorBinding` + kernel ctor).
- `reduction/generic` and this family share the same "one mechanical getter" theme, but groupnorm's `fifo_size` has **no** getter yet — that is the only true blocker across both families you asked about.
