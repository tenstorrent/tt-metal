// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_codegen_supported.hpp"

#include <algorithm>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/data_movement/common/common.hpp"
#include "untilize_codegen_cb_plan.hpp"
#include "untilize_codegen_device_operation.hpp"

namespace ttnn::operations::data_movement::untilize_codegen {

uint32_t usable_l1_bytes(const tt::tt_metal::distributed::MeshDevice& device) {
    return device.l1_size_per_core() - device.allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
}

// Correctness scope of the codegen path: TILE input, interleaved (non-sharded) input AND
// requested output memory config, dtype bfloat16, bfloat8_b, int32 or uint32.
//
// Tile-alignment is dtype-conditional. Non-tile-aligned bfloat16 / int32 / uint32 are in scope --
// build_with_unpadding() in the program factory strips the padding directly, and those dtypes are
// their own RM representation. Non-tile-aligned bfloat8_b is not: serving it would require first
// casting bf8_b -> bf16, a step this implementation does not have, so it routes to native.
//
// float32 is left out: its pack_untilize is only exact with the input CB unpacked straight to the
// 32-bit DEST (UnpackToDestFp32), which the program factory does not configure. int32 / uint32 do
// not need that mode: tt_metal/jit_build/data_format.cpp applies it to a Float32 source only, so
// native's UnpackToDest setting for them is a no-op and fp32_dest_acc_en alone makes their unpack
// exact.
bool supported_by_codegen(const Tensor& input, const tt::tt_metal::MemoryConfig& output_mem_config) {
    using tt::tt_metal::DataType;
    using tt::tt_metal::Layout;
    namespace plan = ttnn::prim::untilize_codegen_detail;

    if (input.layout() != Layout::TILE) {
        return false;
    }
    // Every page-geometry term here and in the program factory is a hardcoded 32x32 quantity --
    // the per-dtype tile_size below, and padded_grid's wt / total_tile_rows. An off-default tile
    // shape changes both the buffer's page size and its page count, so that plan mis-addresses it
    // in both directions. A transposed tile keeps both but permutes the datums the untilize kernels lift
    // out of each face, and nothing here configures the unpacker for that permutation. Declining
    // is not a correctness guarantee for either case: native reads the tile but does not serve an
    // off-default one reliably either. This only stops codegen claiming support it lacks.
    const auto tile = input.tensor_spec().tile();
    if (tile.get_height() != tt::constants::TILE_HEIGHT || tile.get_width() != tt::constants::TILE_WIDTH ||
        tile.get_transpose_within_face() || tile.get_transpose_of_faces()) {
        return false;
    }
    const DataType dtype = input.dtype();
    if (dtype != DataType::BFLOAT16 && dtype != DataType::BFLOAT8_B && dtype != DataType::INT32 &&
        dtype != DataType::UINT32) {
        return false;
    }
    if (input.is_sharded()) {
        return false;
    }
    if (output_mem_config.is_sharded()) {
        return false;
    }

    const uint32_t tile_size = plan::planning_tile_size(dtype, untilize_output_dtype(dtype));
    constexpr uint64_t kWideChunkThreshold = 800'000;

    // Mirrors build_column_parallel + plan_cb_depths: that builder's CB plan is sized by the
    // busiest core's tile count, not by Wt. A case whose single-buffer plan (cb_in + cb_out, one
    // slot per tile each) cannot fit even an empty core's L1 is out of scope for every codegen
    // builder no matter what else is resident, so reject it here and let "auto" pick native.
    // This is a static bound only -- whether the plan fits the L1 that is actually free right now
    // is codegen_cb_plan_fits_live_l1's business (routing) and the program factory's (CB tier).
    auto column_parallel_plan_fits = [&](uint32_t wt) {
        auto* device = input.device();
        auto grid = device->compute_with_storage_grid_size();
        auto split = tt::tt_metal::split_work_to_cores(grid, wt, /*row_wise=*/true);
        uint32_t max_tiles_per_core =
            std::max(std::get<4>(split), std::get<3>(split).empty() ? 0u : std::get<5>(split));
        return 2ull * max_tiles_per_core * tile_size <= usable_l1_bytes(*device);
    };

    // The same physical grid the program factory dispatches on (Wt and the tile-row count are
    // padded, tile-grid quantities; padded_shape is always tile-rounded, so Wt is also the
    // ceil(W / TILE_W) the unpadding path needs).
    const auto g = plan::padded_grid(input);

    if (!g.tile_aligned) {
        // bf8_b has no RM form: native casts it to bf16 first, a step this implementation does
        // not have.
        if (dtype == DataType::BFLOAT8_B) {
            return false;
        }
        // The with-unpadding path has no column-parallel fallback, so ANY row wide enough to
        // overflow the L1 budget -- regardless of total_tile_rows -- is out of scope and goes to
        // native.
        return 2ull * g.wt * tile_size <= kWideChunkThreshold;
    }

    // Tile-aligned path: mirrors ops/untilize/untilize.py's wide-tensor guard for
    // build_untilize_tile -- a multi-tile-row input wide enough that a single tile-row would
    // overflow the slice+concat chunking threshold (~800KB for two double-buffered CBs at the
    // dtype's tile size) is routed to a slice -> untilize -> concat cascade over unrelated builder
    // entries, out of scope here.
    if (g.total_tile_rows > 1 && 2ull * g.wt * tile_size > kWideChunkThreshold) {
        return false;
    }
    // A single tile-row skips the chunk cascade (build_column_parallel splits Wt across the grid
    // in one dispatch instead), so it is bounded by that builder's own per-core plan, not by the
    // whole-row threshold above.
    if (g.total_tile_rows == 1 && g.wt > 1 && !column_parallel_plan_fits(g.wt)) {
        return false;
    }

    return true;
}

bool codegen_cb_plan_fits_live_l1(const Tensor& input, const tt::tt_metal::MemoryConfig& output_mem_config) {
    using tt::tt_metal::Layout;
    namespace plan = ttnn::prim::untilize_codegen_detail;

    if (input.storage_type() != ttnn::StorageType::DEVICE || !input.is_allocated()) {
        return true;
    }

    const ttnn::prim::UntilizeCodegenOperationAttributes attrs{.output_mem_config = output_mem_config};
    const ttnn::prim::UntilizeCodegenTensorArgs tensor_args{.input = input};

    // The codegen op allocates its output (create_output_tensors) before it samples live L1 for
    // the hash and the program; routing runs before that allocation, so reserve the output's
    // per-core L1 footprint here to see the budget those two will (issue #21358 for the native op).
    const auto out_spec = ttnn::prim::UntilizeCodegenDeviceOperation::compute_output_specs(attrs, tensor_args);
    // require_constructible: the spec was just built from these same (shape, dtype, layout, mem_config) by
    // compute_output_specs, so a construction failure here is a bug, not a case to wave through with a 0 B
    // reservation that would make this gate more permissive than the program factory it predicts.
    const uint32_t pending_l1_output_bytes = get_pending_l1_output_reservation(
        input,
        out_spec.padded_shape(),
        output_mem_config,
        out_spec.data_type(),
        Layout::ROW_MAJOR,
        /*require_constructible=*/true);

    return plan::choose_codegen_cb_plan(attrs, tensor_args, pending_l1_output_bytes).tier !=
           plan::CodegenCbPlan::Native;
}

bool supported_execution_controls(bool use_multicore, const std::optional<CoreRangeSet>& sub_core_grids) {
    return use_multicore && !sub_core_grids.has_value();
}

// Perf-demote ledger: cases supported_by_codegen() accepts (correct under codegen) that a device
// kernel-duration comparison found slower than native. Populate only from device kernel
// durations, never from an end-to-end timing: dispatch overhead dominates these single-digit-
// microsecond kernels, so an end-to-end number says nothing about the kernel.
//
// Demoted: a tile-aligned, one-tile-wide int32/uint32 input written to DRAM with 256 or more tile
// rows, when the input is in DRAM on any arch, or in L1 on blackhole. Measured as forced codegen
// over native (device kernel duration) at 263 tile rows: on N300 a DRAM input loses 16-17% while
// an L1 input wins 21%; on P150 a DRAM input loses 2-3% and an L1 input 5-6%. On P150 the same
// DRAM-to-DRAM case, swept by height (median of 5): 2% slower at 255-256 rows, 10-12% slower at
// 384-512, level at 1024, and 6-19% faster at 32-240 rows with one 2% dip at 192, so the bound
// marks where codegen stops winning rather than a cliff. L1 outputs (8-20% faster at 263 rows on
// both archs), bf16 (24% faster at 256 rows) and rows two or more tiles wide (7-8% faster at
// 256-512 rows) stay on codegen. Non-tile-aligned inputs are not demoted: native serves them
// through its unpadding path, several times slower.
bool is_demoted(const Tensor& input, const tt::tt_metal::MemoryConfig& output_mem_config) {
    using tt::tt_metal::BufferType;
    using tt::tt_metal::DataType;
    namespace plan = ttnn::prim::untilize_codegen_detail;
    constexpr uint64_t kNarrow32BitDramTileRowLimit = 256;

    const DataType dtype = input.dtype();
    if ((dtype != DataType::INT32 && dtype != DataType::UINT32) ||
        output_mem_config.buffer_type() != BufferType::DRAM) {
        return false;
    }
    const auto g = plan::padded_grid(input);
    if (!g.tile_aligned || g.wt != 1 || g.total_tile_rows < kNarrow32BitDramTileRowLimit) {
        return false;
    }
    return input.memory_config().buffer_type() == BufferType::DRAM || input.device()->arch() == tt::ARCH::BLACKHOLE;
}

}  // namespace ttnn::operations::data_movement::untilize_codegen
