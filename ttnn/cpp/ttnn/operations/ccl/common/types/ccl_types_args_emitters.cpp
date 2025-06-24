// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include <tt-metalium/buffer_types.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include <tt-metalium/device.hpp>
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"

using namespace tt::tt_metal;

namespace ttnn {
namespace ccl {

args_list_t emit_runtime_args(WorkerEdmInterfaceArgs const& edm_interface_args) {
    return {
        edm_interface_args.edm_noc_x,
        edm_interface_args.edm_noc_y,
        reinterpret_cast<uint32_t>(edm_interface_args.edm_buffer_base_address),
        reinterpret_cast<uint32_t>(edm_interface_args.edm_semaphore_address),
        edm_interface_args.num_buffers_per_channel};
}

args_list_t emit_compile_time(WorkerEdmInterfaceArgs const& edm_interface_args) { return {}; }

args_list_t legacy_emit_address_generator_runtime_args(
    const tt::tt_metal::IDevice* const d, const tt::tt_metal::Tensor& t) {
    args_list_t args;
    switch (t.buffer()->buffer_layout()) {
        case tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED:
        case tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED:
        case tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED:
            return ShardedAddrGenArgBuilder::emit_rt_args(d, t);
            break;

        case tt::tt_metal::TensorMemoryLayout::INTERLEAVED:
            TT_ASSERT(t.buffer()->page_size() != 1024);
            // For now we won't emit args for interleaved here... assume these are passed in elsewhere
            // This is during some transitionary period
            return {};

            break;

        default:
            TT_ASSERT(
                false,
                "Tried emitting address generator args for an unsupported type{}. Consider adding the missing support "
                "or using a supported tensor memory layout (width sharded, height sharded, block sharded, interleaved",
                t.buffer()->buffer_layout());
            return {};
    };
}

args_list_t emit_address_generator_runtime_args(const tt::tt_metal::IDevice* const d, const tt::tt_metal::Tensor& t) {
    args_list_t args;
    switch (t.buffer()->buffer_layout()) {
        case tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED:
        case tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED:
        case tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED: return shard_builder::generate_run_time_args(t); break;

        case tt::tt_metal::TensorMemoryLayout::INTERLEAVED:
            TT_ASSERT(t.buffer()->page_size() != 1024);
            // For now we won't emit args for interleaved here... assume these are passed in elsewhere
            // This is during some transitionary period
            return {};

            break;

        default:
            TT_ASSERT(
                false,
                "Tried emitting address generator args for an unsupported type{}. Consider adding the missing support "
                "or using a supported tensor memory layout (width sharded, height sharded, block sharded, interleaved",
                t.buffer()->buffer_layout());
            return {};
    };
}

args_list_t legacy_emit_address_generator_compile_time_args(const tt::tt_metal::Tensor& t) {
    switch (t.buffer()->buffer_layout()) {
        case tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED:
        case tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED:
        case tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED: return ShardedAddrGenArgBuilder::emit_ct_args(t); break;

        case tt::tt_metal::TensorMemoryLayout::INTERLEAVED: return {}; break;

        default:
            TT_ASSERT(
                false,
                "Tried emitting address generator args for an unsupported type{}. Consider adding the missing support "
                "or using a supported tensor memory layout (width sharded, height sharded, block sharded, interleaved",
                t.buffer()->buffer_layout());
            return {};
    }
    TT_ASSERT(false);
}

args_list_t emit_address_generator_compile_time_args(const tt::tt_metal::Tensor& t) {
    switch (t.buffer()->buffer_layout()) {
        case tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED:
        case tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED:
        case tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED:
            return shard_builder::generate_compile_time_args(t);
            break;

        case tt::tt_metal::TensorMemoryLayout::INTERLEAVED: return {}; break;

        default:
            TT_ASSERT(
                false,
                "Tried emitting address generator args for an unsupported type{}. Consider adding the missing support "
                "or using a supported tensor memory layout (width sharded, height sharded, block sharded, interleaved",
                t.buffer()->buffer_layout());
            return {};
    }
    TT_ASSERT(false);
}

std::pair<CoreCoord, CoreCoord> shard_grid_from_shard_spec(const ShardSpec& shard_spec) {
    auto const& core_range = shard_spec.grid.bounding_box();
    log_trace(
        tt::LogOp,
        "SHARD CORE_RANGE: start_x:{} start_y:{} end_x:{} end_y:{}",
        core_range.start_coord.x,
        core_range.start_coord.y,
        core_range.end_coord.x,
        core_range.end_coord.y);
    log_trace(tt::LogOp, "grid_size: {}", shard_spec.grid.num_cores());

    return {core_range.start_coord, core_range.end_coord};
}

// non-transposed - always row-major layout
// vec<logical row -> noc row>, vec<logicacal col -> noc col>
static std::pair<std::vector<uint32_t>, std::vector<uint32_t>> shard_noc_cores_from_shard_spec(
    IDevice const* d, const ShardSpec& shard_spec) {
    TT_ASSERT(d != nullptr);
    auto const& core_range = shard_spec.grid.bounding_box();
    std::vector<uint32_t> logical_to_noc_row_map;
    std::vector<uint32_t> logical_to_noc_col_map;
    for (uint32_t y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
        CoreCoord noc_core = d->virtual_core_from_logical_core(CoreCoord(0, y), CoreType::WORKER);
        logical_to_noc_row_map.push_back(noc_core.y);
    }
    for (uint32_t x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
        CoreCoord noc_core = d->virtual_core_from_logical_core(CoreCoord(x, 0), CoreType::WORKER);
        logical_to_noc_col_map.push_back(noc_core.x);
    }

    return {logical_to_noc_row_map, logical_to_noc_col_map};
}

std::vector<uint32_t> ShardedAddrGenArgBuilder::emit_rt_args(IDevice const* d, Tensor const& t) {
    std::vector<uint32_t> args;
    auto const& [row_map, col_map] = shard_noc_cores_from_shard_spec(d, t.shard_spec().value());
    args.push_back(row_map.size());
    for (uint32_t i = 0; i < row_map.size(); i++) {
        args.push_back(row_map.at(i));
    }
    args.push_back(col_map.size());
    for (uint32_t i = 0; i < col_map.size(); i++) {
        args.push_back(col_map.at(i));
    }

    return args;
}

std::vector<uint32_t> ShardedAddrGenArgBuilder::emit_ct_args(Tensor const& t) {
    std::vector<uint32_t> args;
    TT_ASSERT(t.is_sharded());
    auto const& [pages_per_shard_y, pages_per_shard_x] = t.buffer()->shard_spec().shape_in_pages();
    auto const& [shard_grid_start, shard_grid_end] = shard_grid_from_shard_spec(t.shard_spec().value());
    bool shard_grid_transposed = shard_grid_is_transposed(t);
    TT_FATAL(
        t.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
            t.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED ||
            t.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "ShardedAddrGenArgBuilder::emit_ct_args was invoked with a tensor containing an unsupported (Sharded) Tensor "
        "Memory Layout: {}",
        t.memory_config().memory_layout());
    // shard_grid_height (cores)
    args.push_back(shard_grid_end.y - shard_grid_start.y + 1);
    TT_FATAL(args.back() > 0, "Passed shard_grid height == 0 to sharded addrgen, which is invalid");
    // shard_grid_width (cores)
    args.push_back(shard_grid_end.x - shard_grid_start.x + 1);
    TT_FATAL(args.back() > 0, "Passed shard_grid width == 0 to sharded addrgen, which is invalid");
    // shard_grid_start_y
    args.push_back(shard_grid_start.y);
    // shard_grid_start_x
    args.push_back(shard_grid_start.x);
    // pages_per_shard_y
    args.push_back(pages_per_shard_y);
    TT_FATAL(args.back() > 0, "Passed pages per shard y == 0 to sharded addrgen, which is invalid");
    // pages_per_shard_x
    args.push_back(pages_per_shard_x);
    TT_FATAL(args.back() > 0, "Passed pages per shard x == 0 to sharded addrgen, which is invalid");
    // transposed grid
    args.push_back(static_cast<uint32_t>(shard_grid_transposed));

    return args;
}

bool ShardedAddrGenArgBuilder::shard_grid_is_transposed(Tensor const& t) {
    TT_FATAL(
        t.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
            t.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED ||
            t.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "ShardedAddrGenArgBuilder::emit_ct_args was invoked with a tensor containing an unsupported (Sharded) Tensor "
        "Memory Layout: {}",
        t.memory_config().memory_layout());
    bool shard_grid_transposed =
        ((t.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED &&
          t.shard_spec()->orientation == ShardOrientation::ROW_MAJOR) ||
         ((t.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
           t.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) &&
          t.shard_spec()->orientation == ShardOrientation::COL_MAJOR));
    return shard_grid_transposed;
}

void ShardedAddrGenArgBuilder::log_sharded_tensor_kernel_args(Tensor const& t, std::string const& prefix) {
    auto const& [pages_per_shard_y, pages_per_shard_x] = t.buffer()->shard_spec().shape_in_pages();
    auto const& [shard_grid_start, shard_grid_end] = shard_grid_from_shard_spec(t.shard_spec().value());
    bool shard_grid_transposed = shard_grid_is_transposed(t);

    TT_ASSERT(pages_per_shard_y > 0);
    TT_ASSERT(pages_per_shard_x > 0);
    log_trace(tt::LogOp, "\t{}_shard_grid_height: {}", prefix, shard_grid_end.y - shard_grid_start.y + 1);
    log_trace(tt::LogOp, "\t{}_shard_grid_width: {}", prefix, shard_grid_end.x - shard_grid_start.x + 1);
    log_trace(tt::LogOp, "\t{}_shard_grid_start_y: {}", prefix, shard_grid_start.y);
    log_trace(tt::LogOp, "\t{}_shard_grid_start_x: {}", prefix, shard_grid_start.x);
    log_trace(tt::LogOp, "\t{}_pages_per_shard_y: {}", prefix, pages_per_shard_y);
    log_trace(tt::LogOp, "\t{}_pages_per_shard_x: {}", prefix, pages_per_shard_x);
    log_trace(tt::LogOp, "\t{}_transposed_grid: {}", prefix, static_cast<uint32_t>(shard_grid_transposed));
}

}  // namespace ccl
}  // namespace ttnn
