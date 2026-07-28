// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "bcast_multi_core_hw_program_factory.hpp"

#include <optional>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/data_movement/bcast/bcast_types.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;
using namespace tt::constants;

tt::tt_metal::ProgramDescriptor BcastMultiCoreHWProgramFactory::create_descriptor(
    const BcastParams& operation_attributes, const BcastInputs& tensor_args, Tensor& tensor_return_value) {
    const Tensor& a = tensor_args.input_a;
    const Tensor& b = tensor_args.input_b;
    Tensor& output = tensor_return_value;

    const auto& ashape = a.padded_shape();
    const auto& bshape = b.padded_shape();
    const uint32_t N = ashape.rank() >= 4 ? ashape[-4] : 1;
    const uint32_t C = ashape.rank() >= 3 ? ashape[-3] : 1;
    const uint32_t H = ashape[-2];
    const uint32_t W = ashape[-1];
    const uint32_t bN = bshape.rank() >= 4 ? bshape[-4] : 1;
    const uint32_t bC = bshape.rank() >= 3 ? bshape[-3] : 1;
    const uint32_t NC = N * C;

    const auto& tile = a.tensor_spec().tile();

    const uint32_t Wt = W / tile.get_width();
    const uint32_t Ht = H / tile.get_height();
    const uint32_t HtWt = Ht * Wt;

    const uint32_t num_tensor_tiles = NC * Ht * Wt;

    const uint32_t bnc1 = (bN * bC == 1);

    IDevice* device = a.device();

    std::optional<ShardSpec> shard_spec = std::nullopt;
    const bool src0_sharded = a.memory_config().is_sharded();
    const bool output_sharded = output.memory_config().is_sharded();
    if (src0_sharded) {
        shard_spec = a.shard_spec().value();
    } else if (output_sharded) {
        shard_spec = output.shard_spec().value();
    }

    const tt::DataFormat src0_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    const tt::DataFormat src1_cb_data_format = datatype_to_dataformat_converter(b.dtype());
    const tt::DataFormat dst_cb_data_format = datatype_to_dataformat_converter(output.dtype());

    const uint32_t src0_single_tile_size = tile.get_tile_size(src0_cb_data_format);
    const uint32_t src1_single_tile_size = tile.get_tile_size(src1_cb_data_format);
    const uint32_t dst_single_tile_size = tile.get_tile_size(dst_cb_data_format);

    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    const uint32_t num_cores_total = num_cores_x * num_cores_y;
    const auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_tensor_tiles);
    (void)num_cores;

    Buffer* src0_buffer = a.buffer();
    Buffer* src1_buffer = b.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    const uint32_t src0_cb_index = 0;
    const uint32_t num_input_tiles = 2;
    uint32_t num_tiles_per_shard = 0;
    if (shard_spec.has_value()) {
        num_tiles_per_shard = shard_spec.value().shape[0] * shard_spec.value().shape[1] / tile.get_tile_hw();
        num_tiles_per_core_group_1 = num_tiles_per_shard;
        num_tiles_per_core_group_2 = 0;
        all_cores = shard_spec.value().grid;
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet();
    }

    const uint32_t num_input_tiles_cb0 = src0_sharded ? num_tiles_per_shard : num_input_tiles;

    const uint32_t src1_cb_index = 1;
    const uint32_t output_cb_index = tt::CBIndex::c_16;
    const uint32_t num_output_tiles = output_sharded ? num_tiles_per_shard : 2;

    ProgramDescriptor desc;

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles_cb0 * src0_single_tile_size,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = src0_cb_data_format,
            .page_size = src0_single_tile_size,
            .tile = TileDescriptor(tile),
        }}},
        .buffer = src0_sharded ? src0_buffer : nullptr,
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * src1_single_tile_size,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src1_cb_index),
            .data_format = src1_cb_data_format,
            .page_size = src1_single_tile_size,
            .tile = TileDescriptor(tile),
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * dst_single_tile_size,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = dst_cb_data_format,
            .page_size = dst_single_tile_size,
            .tile = TileDescriptor(tile),
        }}},
        .buffer = output_sharded ? dst_buffer : nullptr,
    });

    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    std::map<std::string, std::string> reader_defines;
    std::vector<uint32_t> reader_compile_time_args;
    std::map<std::string, std::string> bcast_compute_defines =
        bcast_op_utils::get_defines(BcastOpDim::HW, operation_attributes.math_op);
    if (bnc1) {
        reader_defines["BCAST_SCALAR"] = "1";
        bcast_compute_defines["BCAST_SCALAR"] = "1";
    }
    if (src0_sharded) {
        reader_defines["IN0_SHARDED"] = "1";
    } else {
        TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);
    }
    TensorAccessorArgs(*src1_buffer).append_to(reader_compile_time_args);

    static constexpr const char* READER_KERNEL_PATH =
        "ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/dataflow/"
        "reader_bcast_hw_interleaved_partitioned.cpp";
    static constexpr const char* WRITER_KERNEL_PATH =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
    static constexpr const char* BCAST_KERNEL_PATH =
        "ttnn/cpp/ttnn/operations/data_movement/bcast/device/kernels/compute/bcast_hw.cpp";

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = READER_KERNEL_PATH;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_device_cores;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.defines = {reader_defines.begin(), reader_defines.end()};
    reader_desc.config = ReaderConfigDescriptor{};

    std::map<std::string, std::string> writer_defines;
    if (output_sharded) {
        writer_defines["OUT_SHARDED"] = "1";
    }
    KernelDescriptor writer_desc;
    writer_desc.kernel_source = WRITER_KERNEL_PATH;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_device_cores;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.defines = {writer_defines.begin(), writer_defines.end()};
    writer_desc.config = WriterConfigDescriptor{};

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = BCAST_KERNEL_PATH;
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_device_cores;
    compute_desc.defines = {bcast_compute_defines.begin(), bcast_compute_defines.end()};
    compute_desc.config = ComputeConfigDescriptor{};

    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores_total; i++) {
        const CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tensor_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tensor_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tensor_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            reader_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs(7, 0));
            compute_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{1, 1, 0});
            writer_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs(3, 0));
            continue;
        }

        reader_desc.emplace_runtime_args(
            core,
            {src0_buffer,  // 0
             src1_buffer,
             num_tensor_tiles_per_core,
             HtWt,
             num_tiles_read / HtWt * HtWt,
             num_tiles_read % HtWt,
             bnc1 ? 0u : num_tiles_read / HtWt});

        compute_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                1,                         // B
                1,                         // Ht
                num_tensor_tiles_per_core  // Wt
            });

        writer_desc.emplace_runtime_args(
            core,
            {
                dst_buffer,
                num_tensor_tiles_per_core,
                num_tiles_read,
            });
        num_tiles_read += num_tensor_tiles_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::prim
