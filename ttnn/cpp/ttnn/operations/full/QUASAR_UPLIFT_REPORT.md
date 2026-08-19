# Quasar Uplift Report — `ttnn/cpp/ttnn/operations/full`

Recipe followed: `docs/source/ttnn/ttnn/ai/quasar_porting.md`, plus the canonical passes it
extends (`ai/audit/quasar_audit.md`, `ai/post_port/semantic/dm_self_loop_dfbs.md`,
`ai/post_port/style/sync_free_dfbs.md` on branch `akertesz/op-porting-recipe`).

## Status: RED — stop, do not force the uplift through

The op is a clean Metal 2.0 op on Gen1 and needs exactly one Gen2 change: its three fill-value
buffers are **data-movement self-loop DFBs**, which Gen2 rejects. The sanctioned fix for that shape
is a `Scratchpad`, and **two capabilities the fix depends on are missing from the sanctioned Quasar
API**. Both belong in the runtime API, not in this op, so the uplift stops here per the recipe's
RED-stop rule ("A required capability is missing from the sanctioned Quasar API — flag the missing
feature; do **not** hand-roll an op-level interface").

## Files changed

**None.** Zero-diff. Nothing was moved, renamed, or copied; the op stays in
`ttnn/cpp/ttnn/operations/full/` in the `ttnn::operations::full` namespace. Nothing was written to
`experimental/quasar/` and no `::qsr` namespace was invented.

## Parity claim

Trivially satisfied: the diff is empty, so **WH and BH keep the original path bit-for-bit**. No
device run was needed to establish that. See "Test commands" below for the confirmation runs.

---

## 1. Pre-check: is the op Metal 2.0 on Gen1? Yes

| Check | Result |
|---|---|
| Factory is `create_program_artifacts` / `ProgramArtifacts` | Yes, all three factories |
| Kernels use `dfb::` / `args::` / `tensor::` | Yes |
| No `CBIndex::c_`, no positional `get_arg_val`, no address-RTA `TensorAccessorArgs` | Confirmed — zero occurrences in the op directory |
| Every DFB has a valid `data_format_metadata` | Yes ([interleaved](device/full_program_factory_interleaved.cpp#L62-L69), [sharded](device/full_program_factory_sharded.cpp#L56-L61), [nd-sharded](device/full_program_factory_nd_sharded.cpp#L56-L61)) |
| `opt_level` matches the resolved legacy value | Yes — field left absent on all three DM kernels, which resolves to O2, the legacy DM default |
| No `get_local_cb_interface().fifo_page_size` read | Confirmed — zero occurrences |
| No `disable_dfb_implicit_sync_for_all` / `disable_implicit_sync_for` | Confirmed — zero occurrences |
| No `evil_set_read_ptr` / `evil_set_write_ptr` | Confirmed — zero occurrences |
| No `borrowed_from` DFB, no `dfb_run_overrides` | Confirmed — zero occurrences |
| Non-zero-init semaphores (`quasar_audit.md` check 2) | **N/A** — the op declares no semaphores at all |

So the op starts where this recipe expects: an already-ported Gen1 Metal 2.0 op.

## 2. The blocker: three DM self-loop DFBs

Every one of the op's three program factories declares a one-entry fill-value buffer and binds it to
a **single data-movement kernel as both `PRODUCER` and `CONSUMER`**:

| Factory | DFB spec(s) | Binding kernel(s) |
|---|---|---|
| [full_program_factory_interleaved.cpp:74-87](device/full_program_factory_interleaved.cpp#L74-L87) | `fill_value_writer`, `fill_value_reader` | `writer`, `reader` (each binds its own, both roles) |
| [full_program_factory_sharded.cpp:76-86](device/full_program_factory_sharded.cpp#L76-L86) | `fill_value` | `writer` (both roles) |
| [full_program_factory_nd_sharded.cpp:76-86](device/full_program_factory_nd_sharded.cpp#L76-L86) | `fill_value` | `writer` (both roles) |

This is rejected at program creation on Gen2 by
[program_spec.cpp:1425-1439](../../../../../tt_metal/impl/metal2_host_api/program_spec.cpp#L1425-L1439):

> `DataflowBuffer '{}' is self-looped by data-movement kernel '{}' (bound as both PRODUCER and
> CONSUMER). Self-loop DFBs are not supported for data-movement kernels on Gen2 architectures.
> Consider using a scratchpad or LocalTensorAccessor instead.`

So the op **cannot run on Quasar at all today** — it fails before any kernel is even compiled. This
is a hard blocker, not a latent one.

The shape is genuine and intentional: the kernel builds one page holding the fill value, then NoC-writes
that page to every output page it owns. It is the only toucher of that page, and the source comments
say so ([writer_full.cpp:25-27](device/kernels/writer_full.cpp#L25-L27)). The recipe's survey
confirms it is a site for the DFB → `Scratchpad` conversion on every count except one:

- every binder takes both roles ✓
- every binder is data movement ✓ (`create_writer_datamovement_config` / `create_reader_datamovement_config`)
- `borrowed_from` unset ✓
- no runtime size override ✓
- `num_entries = 1`, so both FIFO indices stay `0` — no stride, no wrap, the simplest possible translation ✓
- **but** the kernel uses the buffer as a `noc.async_write_zeros` destination ✗ — see blocker A

## 3. Blocker A — `Noc::async_write_zeros` accepts no `Scratchpad` destination

For the common all-zero fill, each kernel zeroes the page with one NoC transaction instead of a
per-element store loop, via the shared helper
[full_kernel_common.hpp:19-23](device/kernels/full_kernel_common.hpp#L19-L23):

```cpp
inline void zero_buffer(const DataflowBuffer& dfb, uint32_t bytes) {
    Noc noc;
    noc.async_write_zeros(dfb, bytes);
    noc.write_zeros_l1_barrier();
}
```

`async_write_zeros`'s local-L1 overload hard-rejects anything that is not a `CircularBuffer` or a
`DataflowBuffer`, on **both** generations:

- [internal/tt-2xx/noc_zero_l1.inl:18-23](../../../../../tt_metal/hw/inc/internal/tt-2xx/noc_zero_l1.inl#L18-L23)
- [internal/tt-1xx/noc_zero_l1.inl:12-17](../../../../../tt_metal/hw/inc/internal/tt-1xx/noc_zero_l1.inl#L12-L17)

`dm_self_loop_dfbs.md` lists this use as an explicit stop, for exactly this reason. The three ways
around it are all closed:

1. **Drop the zero fast path** (the per-element loop already writes bit-identical zeros for all
   three supported dtypes). This changes WH/BH un-guarded, which is itself a RED-stop condition and
   a decision for the op owner, not for the uplift.
2. **Keep a DFB on Gen1 and a `Scratchpad` on Gen2**, guarded. Blocked by blocker B below.
3. **Re-inline the zeroing in the op** (a NoC loopback read from `MEM_ZEROS_BASE`). Forbidden:
   §7 says do not hand-roll an op-level equivalent of a missing device API, §11 says zero-fill goes
   through `async_write_zeros()` because Quasar has no `MEM_ZEROS_BASE`, and this is precisely the
   code that PR #45450 ("Create arch-agnostic zeroing APIs") removed from this very kernel.

**Missing feature to flag:** a `Scratchpad` destination overload of
`Noc::async_write_zeros` (local-L1 overload 1). Both implementations already resolve the destination
through `noc_traits_t<Dst>::dst_addr<LOCAL_L1>`, which `Scratchpad` provides
([scratchpad.h:194-200](../../../../../tt_metal/hw/inc/api/scratchpad.h#L194-L200)) — so the
`static_assert` is the only thing in the way. Until it is relaxed, **no op whose DM self-loop buffer
is zero-filled can take the sanctioned `Scratchpad` fix.**

## 4. Blocker B — `Scratchpad` has no Quasar-safe path for CPU writes the NoC later reads

This is the deeper of the two, and it is the reason option 2 above does not rescue the uplift either.

The op's entire mechanism is: **the RISC core stores the fill value into an SRAM page, then the NoC
reads that page** ([writer_full.cpp:31-67](device/kernels/writer_full.cpp#L31-L67)). On Quasar the DM
core sits behind a write-back L2, and the cacheable and uncached views of node SRAM are two distinct
address windows
([dev_mem_map.h:32-36](../../../../../tt_metal/hw/inc/internal/tt-2xx/quasar/dev_mem_map.h#L32-L36)):

```c
#define MEM_L1_BASE 0x0
#define MEM_L1_SIZE (4 * 1024 * 1024)
#define MEM_L1_UNCACHED_BASE (MEM_L1_BASE + MEM_L1_SIZE)  // upper 4MBs bypass cache
```

Every sanctioned abstraction that lets a kernel poke SRAM the hardware also touches routes CPU
access through the **uncached** window on Quasar DM, and hands the NoC the cached address:

| Abstraction | How it handles it |
|---|---|
| `DataflowBuffer` | `get_write_ptr()` / `get_read_ptr()` add `L1_UNCACHED_OFFSET`; `get_noc_write_addr()` does not ([dataflow_buffer.h:330-383](../../../../../tt_metal/hw/inc/api/dataflow/dataflow_buffer.h#L330-L383)) |
| `Semaphore` | `local_l1_addr_ += MEM_L1_UNCACHED_BASE` ([noc_semaphore.h:46](../../../../../tt_metal/hw/inc/api/dataflow/noc_semaphore.h#L46)) |
| `Noc` | `l1_cached_view()` maps an uncached address back before it reaches a NoC API ([noc.h:104-113](../../../../../tt_metal/hw/inc/api/dataflow/noc.h#L104-L113)) |
| **`Scratchpad`** | **nothing.** `get_base_address()` returns the cached view, `operator[]` writes through it, and `noc_traits_t` hands the NoC that same cached address ([scratchpad.h:89-123, 177-200](../../../../../tt_metal/hw/inc/api/scratchpad.h#L89-L200)) |

My reading of this is that `scratchpad[i] = v;` followed by `noc.async_write(scratchpad, …)` is not
cache-safe on Quasar DM: the stores can sit dirty in L2 while the NoC reads stale SRAM. I have not
run it — the op cannot get that far — so treat this as a strong structural inference rather than an
observed failure. Two things corroborate it: the `Scratchpad` HW tests are deliberately gated to
Gen1 only ([test_scratchpad_hw.cpp:58-62](../../../../../tests/tt_metal/tt_metal/api/metal2_host_api/test_scratchpad_hw.cpp#L58-L62),
"the device-side kernel here uses the Gen1 L1-readback idiom"), and §12 of the recipe states the
general rule: "DM caches aren't auto-coherent with L1 producers."

The only mitigations available are `flush_l2_cache_range` / the raw `MEM_L1_UNCACHED_BASE` offset,
both of which live in `internal/tt-2xx/` and would be a hand-rolled op-level device interface —
forbidden by §7, and the kind of thing §8.1 shows needs an `ARCH_QUASAR` guard even inside the
runtime's own headers.

**Missing feature to flag:** `Scratchpad` needs the same Quasar-DM cache story `DataflowBuffer`
already has — an uncached accessor, a scoped write lock, or documentation stating the region is
coherent and why. Note that `LocalTensorAccessor`, the legalizer's other suggestion, has the same
gap ([local_tensor_accessor.h](../../../../../tt_metal/hw/inc/api/tensor/local_tensor_accessor.h)) and
in any case requires an SRAM-resident tensor, so it cannot serve the interleaved/DRAM path at all.

Until this is resolved, **`Scratchpad` cannot replace any DM self-loop DFB that the kernel fills with
CPU stores** — which is the majority of the shape, and the whole of this op.

## 5. §7–§8 gotchas: considered, and what came of each

| Gotcha | Applies? |
|---|---|
| §6 DM self-loop DFB rejected on Gen2 | **Yes — the blocker.** See §2 above |
| §7 `disable_dfb_implicit_sync_*` must not be set | Not set; nothing to do. The Gen2 implicit-sync default is what `noc.async_write(dfb, …)` would use |
| §7 non-zero-init semaphores | N/A — no semaphores |
| §7 Int32-only / no uint16-uint32 device format | N/A — the op supports BFLOAT16, INT32, FLOAT32 only ([full_device_operation.cpp:26-30](device/full_device_operation.cpp#L26-L30)), all available on Quasar. The `uint16_t` / `uint32_t` in the kernels are host-side C++ element types for the store loop, not device formats |
| §7 `compute_kernel_hw_startup` once, tilize/pack init, DEST wrap | N/A — the op has no compute kernel |
| §7 RM shard width must be 16-byte aligned | **Deferred, not applied.** The op does support ROW_MAJOR, so this may well fire, but §2 is explicit that §7–§8 fixes are reactive: apply one only when its symptom actually fires. No device run was possible, so nothing was added speculatively |
| §8.1 `-Werror=int-to-pointer-cast` on a `(void*)(uint32_t)` SRAM address | **Yes, latent.** All three kernels do `reinterpret_cast<uint16_t*>(write_addr)` (and the `uint32_t` / `float` variants) on a `uint32_t` address, e.g. [writer_full.cpp:36-53](device/kernels/writer_full.cpp#L36-L53). Quasar pointers are 64-bit, so these fail to compile there. Not fixed, because the op never reaches JIT — the §2 blocker fires first, and a fix that cannot be built or run is not worth landing blind. The established mainline idiom is `CoreLocalMem<T>(write_addr)`, which is correct on both generations |
| §8.2 hangs / credit stalls, §8.3 wrong output / PCC, §8.4 LLK init, §8.5 HW bugs | **Not reachable.** Every entry needs a device run to trigger its symptom. None was possible |
| §11 NoC / multicast, mcast corner clamps | N/A — the op does no multicast and does not construct NoC coordinates |
| §11 pad the W-dim tail as well as H | N/A — the op writes whole pages, and the sharded factories already clamp both the width and height tails to valid pages ([full_program_factory_sharded.cpp:116-122](device/full_program_factory_sharded.cpp#L116-L122)) |
| §11 zero-fill via `async_write_zeros()` | Already the case — and it is what blocker A turns on |

## 6. Deferred / follow-up items

**For the runtime / DFB API owners (each its own PR, deliberately not bundled here):**

1. **`Noc::async_write_zeros` local-L1 overload should accept a `Scratchpad` destination.** One
   `static_assert` in each of `internal/tt-{1xx,2xx}/noc_zero_l1.inl`; the address resolution already
   works. Blocks every DM-self-loop conversion whose buffer is zero-filled.
2. **`Scratchpad` needs a defined Quasar-DM cache story for CPU writes later read by the NoC** —
   matching what `DataflowBuffer::get_write_ptr()` and `scoped_write_lock()` already provide. This is
   the wider of the two: it blocks the sanctioned DM-self-loop fix for any op that builds its buffer
   contents with RISC stores. Worth feeding back into `quasar_audit.md` as a general check, since it
   is not specific to `full`.
3. **`Scratchpad` HW coverage on Gen2.** `test_scratchpad_hw.cpp` is Gen1-only by construction. A
   Quasar test that CPU-writes a scratchpad and then NoC-reads it would settle item 2 by measurement.

**For the op owner, once the above land:**

4. **The conversion itself**, which is small and already scoped: delete the three
   `DataflowBufferSpec`s and their six `DFBBinding`s; add a `ScratchpadSpec` per page with
   `size_per_node = page_size * 1` registered on `spec.scratchpads`, plus one `ScratchpadBinding`
   each; drop the four FIFO calls in each kernel (both indices are constant `0` at `num_entries = 1`,
   so there is no stride and no wrap to write); pass the scratchpad straight to `noc.async_write` as
   it already passes the DFB. `scratchpad_bindings` sits between `semaphore_bindings` and
   `tensor_bindings` in `KernelSpec`, so the designated initializers keep their declaration order.
   Two details to carry: the interleaved factory binds its reader page **conditionally** on
   `has_reader` ([full_program_factory_interleaved.cpp:123-137](device/full_program_factory_interleaved.cpp#L123-L137)),
   and the `spec.scratchpads` registration must carry the same guard (a declared-but-unbound
   `ScratchpadSpec` is a `TT_FATAL` at program creation); and `data_format_metadata` drops with no
   counterpart, which is sound here because every use of these buffers is a raw address grab or a NoC
   operand — nothing consults the declared format.
5. **Whether to keep the all-zero fast path at all** (blocker A, option 1). If item 1 does not land,
   the op owner's call is between a WH/BH perf change and carrying two buffer designs behind a guard.
   Bit-exactness is not at stake either way: the per-element loop writes the same zeros.
6. **The `int-to-pointer-cast` fix** in all three kernels (§5 table), best done as part of item 4 so
   it can actually be compiled for Quasar and tested.
7. **Two unrelated tidy-ups noticed in passing, not touched:** the `aligned_page_size` compile-time
   arg is passed by both sharded factories but read by neither kernel; and the kernels write
   `s.get_aligned_page_size()` bytes out of a page whose buffer entry is only `page_size` bytes, so
   when those differ the tail of each NoC write sources neighbouring SRAM into the output page's
   alignment padding. Harmless today (the padding is not tensor data) and pre-existing on Gen1.

## Test commands

Nothing here needs a Quasar run — there is nothing to run yet. These are the WH/BH confirmations
that the zero-diff parity claim holds, and the commands that will be needed once the uplift can
proceed:

```bash
# WH/BH parity (expected: unchanged pass — the diff is empty)
source python_env/bin/activate && pytest tests/ttnn/unit_tests/operations/data_movement/test_full.py -v

# Broader coverage of the same op via the creation API
source python_env/bin/activate && pytest tests/ttnn/unit_tests/operations/data_movement/test_creation.py -v

# Once the uplift proceeds, force JIT because the kernels change, and run asserts on first
source python_env/bin/activate && TT_METAL_FORCE_JIT_COMPILE=1 TT_METAL_LLK_ASSERTS=1 \
    pytest tests/ttnn/unit_tests/operations/data_movement/test_full.py -v
```

## Definition-of-done checklist

- [x] Op left **in place** in its existing directory and namespace; nothing copied into `experimental/quasar/`, no `::qsr`
- [x] Factory is `create_program_artifacts` / `ProgramArtifacts`; kernels use `dfb::` / `args::` / `tensor::`
- [x] Each kernel's `opt_level` matches the resolved legacy value (absent → O2 on DM)
- [x] Every DFB has a valid `data_format_metadata`; no `fifo_page_size` reads
- [ ] **Sync-free / DM self-loop DFBs converted to `Scratchpad` — BLOCKED (blockers A and B)**
- [x] No `disable_dfb_implicit_sync_for_all` / `disable_implicit_sync_for`
- [x] No non-zero-init semaphore dependency (no semaphores at all)
- [x] BH and WH keep the original path — zero diff
- [ ] **Quasar builds and runs — NO. Fails at program creation on the DM self-loop legalizer check**
- [x] No DIAG / debug leftovers (nothing was added)
- [x] Missing core dependencies flagged for dedicated PRs rather than silently bundled (§6, items 1–3)
- [x] This report written with status, changed files, and the parity claim; RED-stop conditions checked
