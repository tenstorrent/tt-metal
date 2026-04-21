// SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstring>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/tensor/types.hpp"
#include "randn_device_operation.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::randn {

using namespace tt;
using namespace tt::tt_metal;

std::uniform_int_distribution distribution(1, std::numeric_limits<int32_t>::max());

auto get_random_seed(std::mt19937& rng) -> uint32_t { return distribution(rng); }

static constexpr const char* WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/randn/device/kernels/writer_standard_normal.cpp";
static constexpr const char* COMPUTE_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/randn/device/kernels/compute_standard_normal.cpp";

namespace {
// Cached work split + grid + compute config — avoid recomputing on every dispatch.
struct WorkSplitCache {
    IDevice* device = nullptr;
    CoreCoord grid{0, 0};
    uint32_t units = 0;
    uint32_t num_cores = 0;
    CoreRangeSet all_cores;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t units_per_core_group_1 = 0;
    uint32_t units_per_core_group_2 = 0;
    std::vector<CoreCoord> cores;

    // Cached compute kernel config — depends only on arch + config, constant across dispatches
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    bool math_approx_mode = false;
    bool dst_full_sync_en = false;

    // Cached program hash — structure is identical for same device/shape/dtype
    std::size_t program_hash = 0;

    bool matches(IDevice* dev, uint32_t u) const { return device == dev && units == u; }

    void update(IDevice* dev, uint32_t u) {
        device = dev;
        units = u;
        grid = dev->compute_with_storage_grid_size();
        auto [nc, ac, cg1, cg2, upcg1, upcg2] = split_work_to_cores(grid, u);
        num_cores = nc;
        all_cores = std::move(ac);
        core_group_1 = std::move(cg1);
        core_group_2 = std::move(cg2);
        units_per_core_group_1 = upcg1;
        units_per_core_group_2 = upcg2;
        cores = grid_to_cores(num_cores, grid.x, grid.y);
        program_hash = 0;  // invalidate hash on structural change
    }
};
static thread_local WorkSplitCache work_split_cache;
}  // namespace

ProgramDescriptor RandnDeviceOperation::create_descriptor(
    const operation_attributes_t& operation_attributes,
    [[maybe_unused]] const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    IDevice* device = output.device();

    uint32_t units_to_divide = output.physical_volume() / constants::TILE_HW;
    if (!work_split_cache.matches(device, units_to_divide)) {
        work_split_cache.update(device, units_to_divide);

        // Cache compute kernel config — arch and config are constant for the lifetime of the device
        auto [mf, mam, fp32, pl1, dfs] =
            get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);
        work_split_cache.math_fidelity = mf;
        work_split_cache.math_approx_mode = mam;
        work_split_cache.dst_full_sync_en = dfs;
    }
    const auto& all_cores = work_split_cache.all_cores;
    const auto& core_group_1 = work_split_cache.core_group_1;
    const auto& units_per_core_group_1 = work_split_cache.units_per_core_group_1;
    const auto& units_per_core_group_2 = work_split_cache.units_per_core_group_2;
    const auto& cores = work_split_cache.cores;
    const auto num_cores_total = cores.size();

    DataType output_dtype = output.dtype();
    auto out_data_format = datatype_to_dataformat_converter(output_dtype);
    const uint32_t dtype_tile_size = tile_size(out_data_format);

    constexpr uint32_t in_out_num_tiles = 2;
    constexpr uint32_t dst_cb_id = CBIndex::c_0;

    ProgramDescriptor desc;

    desc.cbs.push_back(CBDescriptor{
        .total_size = in_out_num_tiles * dtype_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = dst_cb_id,
            .data_format = out_data_format,
            .page_size = dtype_tile_size,
        }}},
    });

    // Writer kernel
    KernelDescriptor::CompileTimeArgs writer_ct_args;
    writer_ct_args.reserve(8);
    writer_ct_args.push_back(dst_cb_id);
    TensorAccessorArgs(*output.buffer()).append_to(writer_ct_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = WRITER_KERNEL_PATH;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};
    writer_desc.runtime_args.reserve(num_cores_total);

    // Compute kernel
    KernelDescriptor compute_desc;
    compute_desc.kernel_source = COMPUTE_KERNEL_PATH;
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = {dst_cb_id};
    if (output_dtype == DataType::BFLOAT16) {
        compute_desc.defines.emplace_back("OUTPUT_DTYPE_BFLOAT16", "1");
    }
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = work_split_cache.math_fidelity,
        .fp32_dest_acc_en = true,  // if fp32_dest_acc_en set to false a precision error may occur which makes
                                   // generated number out of range [from, to)
        .dst_full_sync_en = work_split_cache.dst_full_sync_en,
        .math_approx_mode = work_split_cache.math_approx_mode,
    };
    compute_desc.runtime_args.reserve(num_cores_total);

    const uint32_t output_addr = output.buffer()->address();

    // Runtime args
    std::mt19937 rng = operation_attributes.seed.has_value() ? std::mt19937(*operation_attributes.seed)
                                                             : std::mt19937(std::time(nullptr));

    uint32_t tile_offset = 0;
    for (const auto& core : cores) {
        uint32_t units_per_core = core_group_1.contains(core) ? units_per_core_group_1 : units_per_core_group_2;

        compute_desc.runtime_args.emplace_back(
            core, KernelDescriptor::CoreRuntimeArgs{get_random_seed(rng), units_per_core});

        writer_desc.runtime_args.emplace_back(
            core, KernelDescriptor::CoreRuntimeArgs{output_addr, tile_offset, units_per_core});

        tile_offset += units_per_core;
    }

    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    // Cache hash — descriptor structure never changes for same device/shape/dtype
    if (work_split_cache.program_hash == 0) {
        work_split_cache.program_hash = std::hash<ProgramDescriptor>{}(desc);
    }
    desc.custom_program_hash = work_split_cache.program_hash;

    return desc;
}

}  // namespace ttnn::operations::randn
