// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "toy_scaled_add_program_factory.hpp"

#include <bit>
#include <string>
#include <utility>

#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "kernels/toy_scaled_add_args.hpp"
#include "toy_scaled_add_common.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::toy_scaled_add {

using namespace tt::tt_metal;
using namespace ::toy_scaled_add;

namespace {

// The kernels stay with the Python package that first ran them; CMakeLists.txt installs them at this
// same root-relative path for packaged runtimes.
constexpr const char* kKernelDir = "ttnn/ttnn/operations/toy_scaled_add/kernels/";

std::string kernel_path(const char* name) { return std::string(kKernelDir) + name; }

// A circular buffer's page is one tile of the tensor it carries: its dtype, its tile, its size.
CBFormatDescriptor tile_format(uint8_t buffer_index, const Tensor& t) {
    return CBFormatDescriptor{
        .buffer_index = buffer_index,
        .data_format = datatype_to_dataformat_converter(t.dtype()),
        .page_size = detail::tile_bytes(t),
        .tile = TileDescriptor(t.tensor_spec().tile()),
    };
}

CBDescriptor stream_cb(uint8_t buffer_index, const Tensor& t, uint32_t num_tiles, const CoreRangeSet& cores) {
    return CBDescriptor{
        .total_size = num_tiles * detail::tile_bytes(t),
        .core_ranges = cores,
        .format_descriptors = {tile_format(buffer_index, t)},
    };
}

// A circular buffer over this core's shard of `t`: the kernels read and write the tensor in place.
CBDescriptor shard_cb(uint8_t buffer_index, const Tensor& t, uint32_t shard_tiles, const CoreRangeSet& cores) {
    return CBDescriptor{
        .total_size = shard_tiles * detail::tile_bytes(t),
        .core_ranges = cores,
        .format_descriptors = {tile_format(buffer_index, t)},
        .buffer = t.buffer(),
    };
}

KernelDescriptor::Defines gamma_defines(const ToyScaledAddInputs& t) {
    if (t.gamma.has_value()) {
        return {{"TOY_SCALED_ADD_HAS_GAMMA", "1"}};
    }
    return {};
}

// A null Buffer* stands for an absent optional tensor: as a runtime arg it writes 0, and its tensor
// accessor args are a placeholder of the usual layout, so the slots and offsets after it do not move.
Buffer* gamma_buffer(const ToyScaledAddInputs& t) { return t.gamma.has_value() ? t.gamma->buffer() : nullptr; }

// Every CB id this op uses must exist on the device it runs on.
void check_cb_ids() {
    TT_FATAL(
        cb::GAMMA < hal::get_arch_num_circular_buffers(),
        "toy_scaled_add: circular-buffer id {} exceeds the {} circular buffers of this architecture",
        cb::GAMMA,
        hal::get_arch_num_circular_buffers());
}

KernelDescriptor compute_kernel(
    const ToyScaledAddParams& attrs, const ToyScaledAddInputs& t, const CoreRangeSet& cores, uint32_t width_tiles) {
    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(t.a.device()->arch(), attrs.compute_kernel_config);
    KernelDescriptor compute;
    compute.kernel_source = kernel_path("compute.cpp");
    compute.core_ranges = cores;
    compute.named_compile_time_args = {{"Wt", width_tiles}};
    compute.defines = gamma_defines(t);
    // alpha is a common runtime arg, not a compile-time arg: one compiled program serves every alpha.
    compute.common_runtime_args = {std::bit_cast<uint32_t>(attrs.alpha)};
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode,
    };
    return compute;
}

// The per-core arguments: the block of tile-rows each core owns. Fixed by the shapes in the cache key,
// so only a miss writes them.
void add_core_args(ProgramDescriptor& desc, const CoreCoord& core, uint32_t row_start, uint32_t num_rows) {
    for (auto& kernel : desc.kernels) {
        kernel.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{row_start, num_rows});
    }
}

void reserve_core_args(ProgramDescriptor& desc, size_t num_cores) {
    for (auto& kernel : desc.kernels) {
        kernel.runtime_args.reserve(num_cores);
    }
}

}  // namespace

// ---------------------------------------------------------------------------------------------------
// Interleaved
// ---------------------------------------------------------------------------------------------------

ProgramDescriptor InterleavedProgramFactory::create_descriptor(
    const ToyScaledAddParams& attrs, const ToyScaledAddInputs& t, Tensor& output) {
    check_cb_ids();
    const auto [num_rows, width_tiles] = detail::tile_grid(t.a);
    const CoreCoord grid = t.a.device()->compute_with_storage_grid_size();
    const auto [num_cores, all_cores, group_1, group_2, rows_1, rows_2] =
        split_work_to_cores(grid, num_rows, /*row_wise=*/true);

    ProgramDescriptor desc;
    desc.cbs.push_back(stream_cb(cb::A, t.a, detail::kStreamCbTiles, all_cores));
    desc.cbs.push_back(stream_cb(cb::B, t.b, detail::kStreamCbTiles, all_cores));
    desc.cbs.push_back(stream_cb(cb::OUT, output, detail::kStreamCbTiles, all_cores));
    if (t.gamma.has_value()) {
        desc.cbs.push_back(stream_cb(cb::GAMMA, *t.gamma, width_tiles, all_cores));
    }

    KernelDescriptor reader;
    reader.kernel_source = kernel_path("reader_interleaved.cpp");
    reader.core_ranges = all_cores;
    reader.named_compile_time_args = {{"Wt", width_tiles}};
    reader.defines = gamma_defines(t);
    TensorAccessorArgs(*t.a.buffer()).append_to(reader.compile_time_args);
    TensorAccessorArgs(*t.b.buffer()).append_to(reader.compile_time_args);
    TensorAccessorArgs(gamma_buffer(t)).append_to(reader.compile_time_args);
    // Tensor addresses go in as Buffer*: the descriptor writes each buffer's current address and marks
    // the slot as a tensor address. This factory's override_runtime_arguments rewrites these slots on
    // every hit; an op without that hook gets them patched by the framework through the same marks.
    reader.emplace_common_runtime_args({t.a.buffer(), t.b.buffer(), gamma_buffer(t)});
    reader.config = ReaderConfigDescriptor{};

    KernelDescriptor writer;
    writer.kernel_source = kernel_path("writer_interleaved.cpp");
    writer.core_ranges = all_cores;
    writer.named_compile_time_args = {{"Wt", width_tiles}};
    TensorAccessorArgs(*output.buffer()).append_to(writer.compile_time_args);
    writer.emplace_common_runtime_args({output.buffer()});
    writer.config = WriterConfigDescriptor{};

    TT_FATAL(
        reader.common_runtime_args.size() == reader_arg::COUNT &&
            writer.common_runtime_args.size() == writer_arg::COUNT,
        "toy_scaled_add: common runtime argument layout is out of sync with toy_scaled_add_args.hpp");

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(compute_kernel(attrs, t, all_cores, width_tiles));

    reserve_core_args(desc, num_cores);
    uint32_t row_start = 0;
    for (const auto& [group, rows] : {std::pair{group_1, rows_1}, std::pair{group_2, rows_2}}) {
        for (const CoreCoord& core : corerange_to_cores(group, std::nullopt, /*row_wise=*/true)) {
            add_core_args(desc, core, row_start, rows);
            row_start += rows;
        }
    }
    return desc;
}

void InterleavedProgramFactory::override_runtime_arguments(
    Program& program,
    const ToyScaledAddParams& attrs,
    const ToyScaledAddInputs& t,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // A cache hit keeps the work split and every per-core argument: the key fixes them. What can
    // differ is in the common runtime args — one write per slot, whatever the core count.
    //
    // An op whose per-call values sit in per-core args instead refreshes them by walking each kernel's
    // grid of per-core arg vectors in place, skipping cores without args:
    //     auto& grid = GetRuntimeArgs(program, kernel_index::READER);  // by reference: no copy per hit
    //     for (auto& column : grid) { for (auto& args : column) { if (args.size() > 0) { args[SLOT] = value; } } }
    auto& reader = GetCommonRuntimeArgs(program, kernel_index::READER);
    reader[reader_arg::A_ADDR] = t.a.buffer()->address();
    reader[reader_arg::B_ADDR] = t.b.buffer()->address();
    reader[reader_arg::GAMMA_ADDR] = t.gamma.has_value() ? t.gamma->buffer()->address() : 0u;
    GetCommonRuntimeArgs(program, kernel_index::WRITER)[writer_arg::OUT_ADDR] = output.buffer()->address();
    GetCommonRuntimeArgs(program, kernel_index::COMPUTE)[compute_arg::ALPHA_BITS] =
        std::bit_cast<uint32_t>(attrs.alpha);
}

// ---------------------------------------------------------------------------------------------------
// Height-sharded
// ---------------------------------------------------------------------------------------------------

ProgramDescriptor HeightShardedProgramFactory::create_descriptor(
    const ToyScaledAddParams& attrs, const ToyScaledAddInputs& t, Tensor& output) {
    check_cb_ids();
    const auto [num_rows, width_tiles] = detail::tile_grid(t.a);
    const ShardSpec& shard_spec = *t.a.memory_config().shard_spec();
    const CoreRangeSet& cores = shard_spec.grid;
    const uint32_t shard_rows = shard_spec.shape[0] / tt::constants::TILE_HEIGHT;
    const uint32_t shard_tiles = shard_rows * width_tiles;

    ProgramDescriptor desc;
    desc.cbs.push_back(shard_cb(cb::A, t.a, shard_tiles, cores));
    desc.cbs.push_back(shard_cb(cb::B, t.b, shard_tiles, cores));
    desc.cbs.push_back(shard_cb(cb::OUT, output, shard_tiles, cores));
    if (t.gamma.has_value()) {
        desc.cbs.push_back(stream_cb(cb::GAMMA, *t.gamma, width_tiles, cores));
    }

    KernelDescriptor reader;
    reader.kernel_source = kernel_path("reader_sharded.cpp");
    reader.core_ranges = cores;
    reader.named_compile_time_args = {{"Wt", width_tiles}};
    reader.defines = gamma_defines(t);
    TensorAccessorArgs(gamma_buffer(t)).append_to(reader.compile_time_args);
    reader.emplace_common_runtime_args({gamma_buffer(t)});
    reader.config = ReaderConfigDescriptor{};

    KernelDescriptor writer;
    writer.kernel_source = kernel_path("writer_sharded.cpp");
    writer.core_ranges = cores;
    writer.named_compile_time_args = {{"Wt", width_tiles}};
    writer.config = WriterConfigDescriptor{};

    TT_FATAL(
        reader.common_runtime_args.size() == sharded_reader_arg::COUNT,
        "toy_scaled_add: common runtime argument layout is out of sync with toy_scaled_add_args.hpp");

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(compute_kernel(attrs, t, cores, width_tiles));

    // Shard i holds tile-rows [i * shard_rows, (i + 1) * shard_rows), walked in the shard orientation;
    // the last shard may be partial and trailing cores may hold none.
    const bool row_wise = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    const auto shard_cores = corerange_to_cores(cores, std::nullopt, row_wise);
    reserve_core_args(desc, shard_cores.size());
    uint32_t row_start = 0;
    for (const CoreCoord& core : shard_cores) {
        const uint32_t rows = std::min(shard_rows, num_rows - row_start);
        add_core_args(desc, core, row_start, rows);
        row_start += rows;
    }
    return desc;
}

void HeightShardedProgramFactory::override_runtime_arguments(
    Program& program,
    const ToyScaledAddParams& attrs,
    const ToyScaledAddInputs& t,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // The shards move with the tensors: point the shard-backed circular buffers at the current ones,
    // matched by buffer index. gamma's address and alpha are common runtime args.
    for (const auto& circular_buffer : program.circular_buffers()) {
        if (!circular_buffer->globally_allocated()) {
            continue;
        }
        const auto& indices = circular_buffer->buffer_indices();
        const Tensor& backing = indices.contains(cb::A) ? t.a : indices.contains(cb::B) ? t.b : output;
        UpdateDynamicCircularBufferAddress(program, circular_buffer->id(), *backing.buffer());
    }
    GetCommonRuntimeArgs(program, kernel_index::READER)[sharded_reader_arg::GAMMA_ADDR] =
        t.gamma.has_value() ? t.gamma->buffer()->address() : 0u;
    GetCommonRuntimeArgs(program, kernel_index::COMPUTE)[compute_arg::ALPHA_BITS] =
        std::bit_cast<uint32_t>(attrs.alpha);
}

}  // namespace ttnn::operations::toy_scaled_add
