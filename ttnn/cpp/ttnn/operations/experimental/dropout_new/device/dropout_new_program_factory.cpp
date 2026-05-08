// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <bit>

#include "dropout_new_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::experimental::prim {

namespace {

constexpr const char* WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/experimental/dropout/device/kernels/dataflow/writer_dropout_interleaved_start_id.cpp";
constexpr const char* READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/experimental/dropout/device/kernels/dataflow/reader_dropout_interleaved_start_id.cpp";
constexpr const char* COMPUTE_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/experimental/dropout/device/kernels/compute/dropout_kernel.cpp";

constexpr uint32_t SRC0_CB_INDEX = tt::CBIndex::c_0;
constexpr uint32_t OUTPUT_CB_INDEX = tt::CBIndex::c_2;

constexpr uint32_t NUM_INPUT_TILES = 2;
constexpr uint32_t NUM_OUTPUT_TILES = 2;

uint32_t get_effective_seed(const DropoutNewParams& args, const Tensor& output) {
    if (!args.use_per_device_seed) {
        return args.seed;
    }
    return args.seed + output.device()->id();
}

}  // namespace

tt::tt_metal::ProgramDescriptor DropoutNewDeviceOperation::create_descriptor(
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::constants;

    const auto& input = tensor_args.input;
    auto* device = input.device();

    const DataFormat data_fmt_in = datatype_to_dataformat_converter(input.dtype());
    const DataFormat data_fmt_out = datatype_to_dataformat_converter(output.dtype());

    const uint32_t single_tile_size_in = tile_size(data_fmt_in);
    const uint32_t single_tile_size_out = tile_size(data_fmt_out);
    const uint32_t num_tiles = input.physical_volume() / TILE_HW;

    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
            split_work_to_cores(compute_with_storage_grid_size, num_tiles);

    ProgramDescriptor desc;

    desc.cbs.push_back(CBDescriptor{
        .total_size = NUM_INPUT_TILES * single_tile_size_in,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = SRC0_CB_INDEX,
            .data_format = data_fmt_in,
            .page_size = single_tile_size_in,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = NUM_OUTPUT_TILES * single_tile_size_out,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = OUTPUT_CB_INDEX,
            .data_format = data_fmt_out,
            .page_size = single_tile_size_out,
        }}},
    });

    auto* src_buffer = input.buffer();
    KernelDescriptor::CompileTimeArgs reader_compile_args = {SRC0_CB_INDEX};
    TensorAccessorArgs(src_buffer).append_to(reader_compile_args);

    auto* dst_buffer = output.buffer();
    KernelDescriptor::CompileTimeArgs writer_compile_args = {OUTPUT_CB_INDEX};
    TensorAccessorArgs(dst_buffer).append_to(writer_compile_args);

    const uint32_t uscale = std::bit_cast<uint32_t>(args.scale);
    const uint32_t prob_int = static_cast<uint32_t>(static_cast<double>(INT_MAX) * args.prob);
    const bool math_approx_mode = false;

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = READER_KERNEL_PATH;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_args);
    reader_desc.config = ReaderConfigDescriptor{};
    reader_desc.runtime_args.reserve(num_cores);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = WRITER_KERNEL_PATH;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_args);
    writer_desc.config = WriterConfigDescriptor{};
    writer_desc.runtime_args.reserve(num_cores);

    KernelDescriptor compute_group_1_desc;
    compute_group_1_desc.kernel_source = COMPUTE_KERNEL_PATH;
    compute_group_1_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_group_1_desc.core_ranges = core_group_1;
    compute_group_1_desc.compile_time_args = {
        num_tiles_per_core_group_1,
        1,
        prob_int,
        uscale,
    };
    compute_group_1_desc.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = false,
        .dst_full_sync_en = false,
        .math_approx_mode = math_approx_mode,
    };
    compute_group_1_desc.runtime_args.reserve(num_cores);

    KernelDescriptor compute_group_2_desc;
    const bool has_core_group_2 = !core_group_2.ranges().empty();
    if (has_core_group_2) {
        compute_group_2_desc.kernel_source = COMPUTE_KERNEL_PATH;
        compute_group_2_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_group_2_desc.core_ranges = core_group_2;
        compute_group_2_desc.compile_time_args = {
            num_tiles_per_core_group_2,
            1,
            prob_int,
            uscale,
        };
        compute_group_2_desc.config = ComputeConfigDescriptor{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .dst_full_sync_en = false,
            .math_approx_mode = math_approx_mode,
        };
        compute_group_2_desc.runtime_args.reserve(num_cores);
    }

    const uint32_t effective_seed = get_effective_seed(args, output);

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
            compute_group_1_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{effective_seed});
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
            TT_FATAL(
                has_core_group_2, "Second core group runtime args must not be populated when core group 2 is empty");
            compute_group_2_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{effective_seed});
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        reader_desc.runtime_args.emplace_back(
            core, KernelDescriptor::CoreRuntimeArgs{src_buffer->address(), num_tiles_per_core, num_tiles_written});

        writer_desc.runtime_args.emplace_back(
            core, KernelDescriptor::CoreRuntimeArgs{dst_buffer->address(), num_tiles_per_core, num_tiles_written});

        num_tiles_written += num_tiles_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_group_1_desc));
    if (has_core_group_2) {
        desc.kernels.push_back(std::move(compute_group_2_desc));
    }

    return desc;
}

}  // namespace ttnn::experimental::prim
