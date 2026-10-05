// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// D2.0 fork of kv_cache/.../reader_fill_cache_interleaved_start_id.cpp, so deepseek_prefill can move to
// D2.0 without dragging the (still Device 1.x) kv_cache op along.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/kernels/dataflow/cache_dataflow_helpers.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/pack_scaled_fp8_kv_cache/packed_kv_layout.hpp"

namespace packed = ttnn::operations::experimental::deepseek_prefill::pack_scaled_fp8_kv_cache;

namespace {

// Append contiguous tensor pages to a reserved CB range. Caller handles the barrier and CB publication.
template <typename Accessor>
inline void read_pages(
    Noc& noc,
    const Accessor& source,
    const CircularBuffer& cb,
    uint32_t first_page,
    uint32_t page_count,
    uint32_t page_bytes,
    uint32_t destination_page = 0) {
    for (uint32_t page = 0; page < page_count; ++page) {
        noc.async_read(
            source,
            cb,
            page_bytes,
            {.page_id = first_page + page},
            {.offset_bytes = (destination_page + page) * page_bytes});
    }
}

}  // namespace

// `kv_actual_global` reaches this kernel the same two ways the writer's does, selected by the
// `has_metadata` compile-time flag: read on-device from the 1-element uint32 tensor whose raw address is
// common arg 7 (metadata/traceable path), or straight out of common arg 7 as a value (scalar path). The
// reader NEEDS the real value even under tp_axis -- its source-row mapping is derived from the chunk
// start -- which is why the metadata path had to grow this read before tp_axis could be allowed on it.
//
// Templated on HasMeta so `if constexpr` DISCARDS the unused branch: kernel_main is not a template, so
// an `if constexpr` there would still instantiate the metadata TensorAccessor and fail to compile the
// scalar program, which carries no metadata accessor args.
template <bool HasMeta, bool HasRope, bool HasScaled, bool TiledRope>
static void run_reader() {
    // Index 8 follows the eight common args in update_padded_kv_cache_device_operation.cpp,
    // where reader_kernel.emplace_common_runtime_args({src_buffer}) appends the source binding.
    const uint32_t src_addr = get_common_arg_val<uint32_t>(8);
    const uint32_t num_pages = get_arg_val<uint32_t>(0);
    const uint32_t core_blocks_written = get_arg_val<uint32_t>(1);

    const uint32_t linear_coord = get_common_arg_val<uint32_t>(0);
    const uint32_t linear_factor = get_common_arg_val<uint32_t>(1);
    const uint32_t chunk_local_t = get_common_arg_val<uint32_t>(2);  // stripe: page-rows THIS chip writes
    const uint32_t input_Ht = get_common_arg_val<uint32_t>(3);       // window: page-rows the input holds
    const uint32_t sp_factor = get_common_arg_val<uint32_t>(4);
    const uint32_t tp_factor = get_common_arg_val<uint32_t>(5);
    const uint32_t Wt = get_common_arg_val<uint32_t>(6);
    // Common arg 7 is the value on the scalar path and the 1-element tensor's ADDRESS on the metadata
    // path; resolved below, next to the mapping that consumes it.

    constexpr uint32_t tile_height = get_compile_time_arg_val(0);
    // [1] = has_metadata (consumed as the HasMeta template param), [2] = metadata scratch CB (0 on the
    // scalar path), [3] = untilize, [4..5] = split widths in tiles, [6] = scaled-FP8 packing.
    // [7] = tiled RoPE CB (0 if absent). Accessors start at 8: source, optional RoPE/scales, then optional metadata.
    constexpr auto src_args = TensorAccessorArgs<8>();

    constexpr uint32_t cb_id_in0 = 0;
    CircularBuffer cb_in0(cb_id_in0);

#ifdef INPUT_SHARDED
    cb_in0.reserve_back(num_pages);
    cb_in0.push_back(num_pages);
#else
    Noc noc;
    uint32_t kv_actual_global;
    if constexpr (HasMeta) {
        // Same on-device read the writer performs, so both derive the chunk start from ONE value and
        // cannot disagree on a non-window-aligned start. Own scratch CB: the writer is doing its own
        // metadata reads into kMetaCbIndex concurrently on this core.
        constexpr uint32_t cb_id_meta = get_compile_time_arg_val(2);
        // Gate the offset on HasMeta so this is a *dependent* template-id -- the scalar program has no
        // metadata accessor and must not name a fixed out-of-range compile-arg offset here.
        constexpr uint32_t kMetaArgsOffset = [] {
            if constexpr (HasRope || HasScaled) {
                constexpr auto rope_args =
                    TensorAccessorArgs<(HasRope || HasScaled) ? src_args.next_compile_time_args_offset() : 0>();
                if constexpr (HasScaled) {
                    constexpr auto scale_args =
                        TensorAccessorArgs<HasScaled ? rope_args.next_compile_time_args_offset() : 0>();
                    return scale_args.next_compile_time_args_offset();
                } else {
                    return rope_args.next_compile_time_args_offset();
                }
            } else {
                return src_args.next_compile_time_args_offset();
            }
        }();
        CircularBuffer cb_meta(cb_id_meta);
        cb_meta.reserve_back(1);
        kv_actual_global =
            kv_cache_dataflow::read_metadata<kMetaArgsOffset>(noc, cb_meta, get_common_arg_val<uint32_t>(7));
        cb_meta.push_back(1);
    } else {
        kv_actual_global = get_common_arg_val<uint32_t>(7);  // per-call; patched on cache hits
    }

    // Source-row mapping. The TP-replicated input holds one SP chip's whole window (input_Ht rows, in the
    // writer's rotated order) and this chip owns one chunk_local_t stripe of it. Inverting that rotation
    // gives src(j) = base + j, plus `jump` once j leaves the stripe -- two runs only on the start's chip.
    const uint32_t start_t = kv_actual_global / tile_height;  // chunk start, in page-rows
    // Chunk start within this chip's stripe / its SP group's window; nonzero only on the chip that holds it
    // (the stripe test is the writer's own expression, so this matches its update_idxt).
    const uint32_t start_in_stripe =
        (linear_coord == (start_t / chunk_local_t) % linear_factor) ? start_t % chunk_local_t : 0;
    const uint32_t start_in_window =
        (linear_coord / tp_factor == (start_t / input_Ht) % sp_factor) ? start_t % input_Ht : 0;
    // + input_Ht keeps the numerator positive (start_in_window < input_Ht); sum < 2*input_Ht, one mod.
    const uint32_t base =
        ((linear_coord % tp_factor) * chunk_local_t + start_in_stripe + input_Ht - start_in_window) % input_Ht;
    const uint32_t jump = chunk_local_t * (tp_factor - 1);  // skip the other TP stripes; 0 at tp_factor == 1

    constexpr uint32_t onetile = 1;
    const auto s = TensorAccessor(src_args, src_addr);
    // CB page size, NOT get_tile_size(): in ROW_MAJOR a page is one token row, and a tile-size read
    // overruns the CB into the writer's metadata scratch. The writer derives its page bytes the same way.
    const uint32_t src_page_bytes = get_local_cb_interface(cb_id_in0).fifo_page_size;

    const uint32_t num_blocks = HasRope ? num_pages : num_pages / Wt;
    for (uint32_t block = 0; block < num_blocks; ++block) {
        const uint32_t j = core_blocks_written + block;
        const uint32_t source_row = base + j + (j + start_in_stripe >= chunk_local_t ? jump : 0);
        if constexpr (HasScaled) {
            constexpr auto rope_args = TensorAccessorArgs<HasScaled ? src_args.next_compile_time_args_offset() : 0>();
            constexpr auto scale_args = TensorAccessorArgs<HasScaled ? rope_args.next_compile_time_args_offset() : 0>();
            const auto rope = TensorAccessor(rope_args, get_common_arg_val<uint32_t>(9));
            const auto scales = TensorAccessor(scale_args, get_common_arg_val<uint32_t>(10));
            // Field pages stay aligned; tiled RoPE uses its own untilize pipeline.
            constexpr uint32_t rows_per_block = TiledRope ? tile_height : 1;
            constexpr uint32_t pages_per_row = TiledRope ? 2 : 3;
            constexpr uint32_t block_pages = rows_per_block * pages_per_row;
            cb_in0.reserve_back(block_pages);
            for (uint32_t row = 0; row < rows_per_block; ++row) {
                const uint32_t token = source_row * rows_per_block + row;
                const uint32_t offset = row * pages_per_row * src_page_bytes;
                noc.async_read(s, cb_in0, packed::LATENT_WIDTH, {.page_id = token}, {.offset_bytes = offset});
                noc.async_read(
                    scales,
                    cb_in0,
                    packed::SCALE_WIDTH * sizeof(float),
                    {.page_id = token},
                    {.offset_bytes = offset + src_page_bytes});
                if constexpr (!TiledRope) {
                    noc.async_read(
                        rope,
                        cb_in0,
                        packed::ROPE_WIDTH * sizeof(uint16_t),
                        {.page_id = token},
                        {.offset_bytes = offset + 2 * src_page_bytes});
                }
            }
            if constexpr (TiledRope) {
                constexpr uint32_t rope_cb_id = get_compile_time_arg_val(7);
                constexpr uint32_t rope_tiles = get_compile_time_arg_val(5);
                CircularBuffer rope_cb(rope_cb_id);
                rope_cb.reserve_back(rope_tiles);
                read_pages(noc, rope, rope_cb, source_row * rope_tiles, rope_tiles, get_tile_size(rope_cb_id));
                noc.async_read_barrier();
                rope_cb.push_back(rope_tiles);
            } else {
                noc.async_read_barrier();
            }
            cb_in0.push_back(block_pages);
        } else if constexpr (HasRope) {
            constexpr auto rope_args =
                TensorAccessorArgs<(HasRope || HasScaled) ? src_args.next_compile_time_args_offset() : 0>();
            const auto rope = TensorAccessor(rope_args, get_common_arg_val<uint32_t>(9));
            constexpr uint32_t latent_wt = get_compile_time_arg_val(4);
            constexpr uint32_t rope_wt = get_compile_time_arg_val(5);
            constexpr uint32_t block_tiles = latent_wt + rope_wt;
            // Append both tile rows in L1 for the standard untilize compute kernel.
            cb_in0.reserve_back(block_tiles);
            read_pages(noc, s, cb_in0, source_row * latent_wt, latent_wt, src_page_bytes);
            read_pages(noc, rope, cb_in0, source_row * rope_wt, rope_wt, src_page_bytes, latent_wt);
            noc.async_read_barrier();
            cb_in0.push_back(block_tiles);
        } else {
            for (uint32_t page = 0; page < Wt; ++page) {
                cb_in0.reserve_back(onetile);
                read_pages(noc, s, cb_in0, source_row * Wt + page, onetile, src_page_bytes);
                noc.async_read_barrier();
                cb_in0.push_back(onetile);
            }
        }
    }
#endif  // INPUT_SHARDED
}

void kernel_main() {
    constexpr bool has_metadata = get_compile_time_arg_val(1) != 0;
    constexpr bool has_rope = get_compile_time_arg_val(3) != 0;
    constexpr bool has_scaled = get_compile_time_arg_val(6) != 0;
    constexpr bool tiled_rope = get_compile_time_arg_val(7) != 0;
    run_reader<has_metadata, has_rope, has_scaled, tiled_rope>();
}
