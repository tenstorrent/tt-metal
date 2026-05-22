// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operations/experimental/deepseek_moe_post_combine_tilize/device/deepseek_moe_post_combine_tilize_program_factory.hpp"

#include "ttnn/operations/cb_utils.hpp"

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

tt::tt_metal::ProgramDescriptor DeepseekMoEPostCombineTilizeProgramFactory::create_descriptor(
    const DeepseekMoEPostCombineTilizeParams&,
    const DeepseekMoEPostCombineTilizeInputs& tensor_args,
    ttnn::Tensor& tensor_return_value) {
    ProgramDescriptor desc;

    /*
     * Tensors
     */
    const ttnn::Tensor& input_tensor = tensor_args.input_tensor;
    uint32_t input_row_page_size = input_tensor.buffer()->page_size();
    const auto& input_shape = input_tensor.padded_shape();
    const uint32_t input_rank = input_shape.rank();

    const ttnn::Tensor& output_tensor = tensor_return_value;
    const uint32_t output_tile_page_size = output_tensor.buffer()->page_size();
    const auto& output_shape = output_tensor.padded_shape();

    /*
     * Shard spec
     */
    const auto output_nd_shard_spec = output_tensor.memory_config().nd_shard_spec().value();
    const uint32_t output_shard_width = output_nd_shard_spec.shard_shape[-1];
    const uint32_t output_shard_width_tiles = output_shard_width / tt::constants::TILE_WIDTH;
    const uint32_t output_shard_width_bytes = output_shard_width * output_tensor.element_size();

    const CoreRangeSet op_cores = output_nd_shard_spec.grid;

    uint32_t upper_dims = 1;
    for (uint32_t dim = 0; dim < input_rank - 1; ++dim) {
        upper_dims *= input_shape[dim];
    }

    const uint32_t output_num_shards_wide = output_shape[-1] / output_shard_width;
    const uint32_t output_num_shards_high = upper_dims / output_nd_shard_spec.shard_shape[-2];

    /*
     * CBs
     */
    const uint32_t tilize_input_cb_id = tt::CBIndex::c_0;
    const tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    desc.cbs.push_back(CBDescriptor{
        .total_size = tt::constants::TILE_HEIGHT * output_shard_width_bytes,
        .core_ranges = op_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tilize_input_cb_id,
            .data_format = data_format,
            .page_size = output_shard_width_bytes,
        }}},
    });

    // Sharded tilize-output CB bound to the output buffer.
    const uint32_t tilize_output_cb_id = tt::CBIndex::c_1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = output_shard_width_tiles * output_tile_page_size,
        .core_ranges = op_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = tilize_output_cb_id,
            .data_format = data_format,
            .page_size = output_tile_page_size,
        }}},
        .buffer = output_tensor.buffer(),
    });

    /*
     * Kernels
     */

    // reader
    KernelDescriptor::NamedCompileTimeArgs reader_named_ct_args = {
        {"tilize_input_cb_id", tilize_input_cb_id},
        {"input_row_page_size", input_row_page_size},
        {"bytes_to_read_per_row", output_shard_width_bytes},
    };

    std::vector<uint32_t> reader_ct_args = {};
    tt::tt_metal::TensorAccessorArgs(*input_tensor.buffer()).append_to(reader_ct_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_moe_post_combine_tilize/device/kernels/"
        "deepseek_moe_post_combine_tilize_reader.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = op_cores;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.named_compile_time_args = std::move(reader_named_ct_args);
    reader_desc.opt_level = KernelBuildOptLevel::O2;
    reader_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::NOC_0,
        .noc_mode = NOC_MODE::DM_DEDICATED_NOC,
    };

    // compute
    KernelDescriptor::NamedCompileTimeArgs compute_named_ct_args = {
        {"tilize_input_cb_id", tilize_input_cb_id},
        {"tilize_output_cb_id", tilize_output_cb_id},
        {"num_tiles", output_shard_width_tiles},
    };
    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_moe_post_combine_tilize/device/kernels/"
        "deepseek_moe_post_combine_tilize_compute.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = op_cores;
    compute_desc.named_compile_time_args = std::move(compute_named_ct_args);
    compute_desc.config = ComputeConfigDescriptor{};

    // writer
    KernelDescriptor::NamedCompileTimeArgs writer_named_ct_args = {
        {"tilize_output_cb_id", tilize_output_cb_id},
        {"num_tiles", output_shard_width_tiles},
    };
    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_moe_post_combine_tilize/device/kernels/"
        "deepseek_moe_post_combine_tilize_writer.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = op_cores;
    writer_desc.named_compile_time_args = std::move(writer_named_ct_args);
    writer_desc.opt_level = KernelBuildOptLevel::O2;
    writer_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::NOC_1,
        .noc_mode = NOC_MODE::DM_DEDICATED_NOC,
    };

    // ---- Per-core runtime args (reader only) ----
    const bool is_row_major_shard_orientation = output_nd_shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    std::vector<tt::tt_metal::CoreCoord> cores =
        corerange_to_cores(op_cores, std::nullopt, is_row_major_shard_orientation);
    reader_desc.runtime_args.reserve(cores.size());
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const auto& core = cores[i];

        uint32_t intra_row_byte_offset;
        uint32_t row_page_offset;
        if (is_row_major_shard_orientation) {
            intra_row_byte_offset = (i % output_num_shards_wide) * output_shard_width_bytes;
            row_page_offset = (i / output_num_shards_wide) * tt::constants::TILE_HEIGHT;
        } else {
            intra_row_byte_offset = (i / output_num_shards_high) * output_shard_width_bytes;
            row_page_offset = (i % output_num_shards_high) * tt::constants::TILE_HEIGHT;
        }
        reader_desc.runtime_args.emplace_back(
            core, std::vector<uint32_t>{intra_row_byte_offset, row_page_offset, input_tensor.buffer()->address()});
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(compute_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::experimental::prim
