// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <ctime>
#include <random>
#include <string>

#include "bernoulli_device_operation.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::bernoulli {

using namespace tt;
using namespace tt::tt_metal;

namespace {
std::mt19937 rng(std::time(nullptr));
std::uniform_int_distribution<uint32_t> dist(1, 1 << 20);
uint32_t get_random_seed() { return dist(rng); }

// Work split used by create_descriptor (cache miss) and override_runtime_arguments (cache hit).
struct BernoulliWorkSplit {
    uint32_t num_cores = 0;
    CoreRangeSet all_cores;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t units_per_core_group_1 = 0;
    uint32_t units_per_core_group_2 = 0;
    std::vector<CoreCoord> cores;
};

BernoulliWorkSplit bernoulli_work_split(const Tensor& output) {
    auto grid = output.device()->compute_with_storage_grid_size();
    uint32_t units_to_divide = output.physical_volume() / constants::TILE_HW;
    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);
    auto cores = grid_to_cores(num_cores, grid.x, grid.y);
    return {
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        units_per_core_group_1,
        units_per_core_group_2,
        std::move(cores)};
}

// Per-core work assignment, single-sourced so create_descriptor and override_runtime_arguments can
// never drift on core-group selection or tile_offset accumulation.
struct BernoulliCoreWork {
    CoreCoord core;
    uint32_t units_per_core;
    uint32_t tile_offset;
};
std::vector<BernoulliCoreWork> bernoulli_core_layout(const BernoulliWorkSplit& ws) {
    std::vector<BernoulliCoreWork> layout;
    layout.reserve(ws.cores.size());
    uint32_t tile_offset = 0;
    for (const auto& core : ws.cores) {
        uint32_t units_per_core;
        if (ws.core_group_1.contains(core)) {
            units_per_core = ws.units_per_core_group_1;
        } else if (ws.core_group_2.contains(core)) {
            units_per_core = ws.units_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }
        layout.push_back({core, units_per_core, tile_offset});
        tile_offset += units_per_core;
    }
    return layout;
}

// Per-core seed; shared so the miss-build and the hit-patch produce identical values.
uint32_t bernoulli_seed_for_core(const BernoulliDeviceOperation::operation_attributes_t& attrs, uint32_t i) {
    return attrs.seed != 0 ? attrs.seed + i : get_random_seed();
}
}  // namespace

ProgramDescriptor BernoulliDeviceOperation::BernoulliProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const Tensor& input = tensor_args.input;

    IDevice* device = output.device();
    const auto ws = bernoulli_work_split(output);
    const auto& all_cores = ws.all_cores;

    // ---- Circular buffers ----

    ProgramDescriptor desc;

    constexpr uint32_t num_tiles = 2;
    auto in_data_format = datatype_to_dataformat_converter(input.dtype());
    const uint32_t in_dtype_tile_size = tile_size(in_data_format);
    constexpr uint32_t in_cb_id = CBIndex::c_0;

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_tiles * in_dtype_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = in_cb_id,
            .data_format = in_data_format,
            .page_size = in_dtype_tile_size,
        }}},
    });

    const uint32_t float32_tile_size = tile_size(tt::DataFormat::Float32);
    constexpr uint32_t intermed_cb_id = CBIndex::c_24;

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_tiles * float32_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = intermed_cb_id,
            .data_format = tt::DataFormat::Float32,
            .page_size = float32_tile_size,
        }}},
    });

    auto out_data_format = datatype_to_dataformat_converter(output.dtype());
    const uint32_t out_dtype_tile_size = tile_size(out_data_format);
    constexpr uint32_t intermed1_cb_id = CBIndex::c_25;

    desc.cbs.push_back(CBDescriptor{
        .total_size = 1 * out_dtype_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = intermed1_cb_id,
            .data_format = out_data_format,
            .page_size = out_dtype_tile_size,
        }}},
    });

    // ---- Kernels ----

    const std::string kernels_dir_path = "ttnn/cpp/ttnn/operations/bernoulli/device/kernels/";

    // Reader kernel
    std::vector<uint32_t> reader_compile_time_args{in_cb_id};
    TensorAccessorArgs(*input.buffer()).append_to(reader_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = kernels_dir_path + "reader_bernoulli.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.config = ReaderConfigDescriptor{};

    // Writer kernel
    std::vector<uint32_t> writer_compile_time_args{in_cb_id, intermed_cb_id, intermed1_cb_id};
    TensorAccessorArgs(*output.buffer()).append_to(writer_compile_time_args);

    KernelDescriptor::Defines writer_defines;
    switch (input.dtype()) {
        case DataType::BFLOAT16: writer_defines.emplace_back("INPUT_DTYPE_BFLOAT16", "1"); break;
        case DataType::FLOAT32: writer_defines.emplace_back("INPUT_DTYPE_FLOAT32", "1"); break;
        default: break;
    }
    switch (output.dtype()) {
        case DataType::BFLOAT16: writer_defines.emplace_back("OUTPUT_DTYPE_BFLOAT16", "1"); break;
        case DataType::FLOAT32: writer_defines.emplace_back("OUTPUT_DTYPE_FLOAT32", "1"); break;
        default: break;
    }

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = kernels_dir_path + "writer_bernoulli.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.defines = writer_defines;
    writer_desc.config = WriterConfigDescriptor{};

    // Compute kernel
    const std::vector<uint32_t> compute_compile_time_args{intermed_cb_id};

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = kernels_dir_path + "compute_bernoulli.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = compute_compile_time_args;
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = true,  // must always be true, otherwise generated floats are in [0.4, 0.5]
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode,
    };

    // ---- Runtime args per core ----

    const auto layout = bernoulli_core_layout(ws);
    for (uint32_t i = 0; i < layout.size(); ++i) {
        const auto& [core, units_per_core, tile_offset] = layout[i];

        // Register input/output addresses as Buffer* bindings for the cache-miss build; both are
        // re-applied on every cache hit via override_runtime_arguments().
        reader_desc.emplace_runtime_args(core, {input.buffer(), tile_offset, units_per_core});

        // seed is DYNAMIC (excluded from compute_program_hash): baked here for the cache-miss
        // build, re-applied on every cache hit via override_runtime_arguments().
        const uint32_t seed = bernoulli_seed_for_core(operation_attributes, i);
        compute_desc.runtime_args.emplace_back(
            core, KernelDescriptor::CoreRuntimeArgs{seed, tile_offset, units_per_core});

        writer_desc.emplace_runtime_args(core, {output.buffer(), tile_offset, units_per_core});
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

void BernoulliDeviceOperation::BernoulliProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // Re-derive every per-dispatch arg on each cache hit from the same builder create_descriptor uses:
    // the compute seed and the reader/writer buffer addresses. override replaces resolve_bindings, so
    // the addresses are ours to re-apply too. Push order in create_descriptor: reader 0, writer 1,
    // compute 2.
    constexpr uint32_t reader_kernel_idx = 0;
    constexpr uint32_t writer_kernel_idx = 1;
    constexpr uint32_t compute_kernel_idx = 2;

    const auto ws = bernoulli_work_split(output);
    const uint32_t in_addr = tensor_args.input.buffer()->address();
    const uint32_t out_addr = output.buffer()->address();

    const auto layout = bernoulli_core_layout(ws);
    for (uint32_t i = 0; i < layout.size(); ++i) {
        const auto& [core, units_per_core, tile_offset] = layout[i];
        const uint32_t seed = bernoulli_seed_for_core(operation_attributes, i);

        auto& reader_args = tt::tt_metal::GetRuntimeArgs(program, reader_kernel_idx, core);
        reader_args[0] = in_addr;
        reader_args[1] = tile_offset;
        reader_args[2] = units_per_core;

        auto& writer_args = tt::tt_metal::GetRuntimeArgs(program, writer_kernel_idx, core);
        writer_args[0] = out_addr;
        writer_args[1] = tile_offset;
        writer_args[2] = units_per_core;

        auto& compute_args = tt::tt_metal::GetRuntimeArgs(program, compute_kernel_idx, core);
        compute_args[0] = seed;
        compute_args[1] = tile_offset;
        compute_args[2] = units_per_core;
    }
}

}  // namespace ttnn::operations::bernoulli
