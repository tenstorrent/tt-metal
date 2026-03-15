// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operations/experimental/deepseek_moe_post_combine_tilize/device/deepseek_moe_post_combine_tilize_program_factory.hpp"

#include "ttnn/operations/cb_utils.hpp"

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

DeepseekMoEPostCombineTilizeProgramFactory::cached_program_t DeepseekMoEPostCombineTilizeProgramFactory::create(
    const DeepseekMoEPostCombineTilizeParams&,
    const DeepseekMoEPostCombineTilizeInputs& tensor_args,
    ttnn::Tensor& tensor_return_value) {
    auto program = Program();

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
    uint32_t tilize_input_cb_id = tt::CBIndex::c_0;
    tt::tt_metal::create_cb(
        tilize_input_cb_id,
        program,
        op_cores,
        output_shard_width_bytes,
        tt::constants::TILE_HEIGHT,
        tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype()));

    uint32_t tilize_output_cb_id = tt::CBIndex::c_1;
    auto output_cb = tt::tt_metal::create_cb(
        tilize_output_cb_id,
        program,
        op_cores,
        output_tile_page_size,
        output_shard_width_tiles,
        tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype()),
        output_tensor.buffer());
    tt::tt_metal::CBHandle sharded_output_cb_handle = std::get<1>(output_cb);

    /*
     * Kernels
     */

    // reader
    std::unordered_map<std::string, uint32_t> reader_named_ct_args = {
        {"tilize_input_cb_id", tilize_input_cb_id},
        {"input_row_page_size", input_row_page_size},
        {"bytes_to_read_per_row", output_shard_width_bytes},
    };

    std::vector<uint32_t> reader_ct_args = {};
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_ct_args);

    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_moe_post_combine_tilize/device/kernels/"
        "deepseek_moe_post_combine_tilize_reader.cpp",
        op_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::NOC_0,
            .noc_mode = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC,
            .compile_args = reader_ct_args,
            .defines = {},
            .named_compile_args = reader_named_ct_args,
            .opt_level = tt::tt_metal::KernelBuildOptLevel::O2});

    // compute
    std::unordered_map<std::string, uint32_t> compute_named_ct_args = {
        {"tilize_input_cb_id", tilize_input_cb_id},
        {"tilize_output_cb_id", tilize_output_cb_id},
        {"num_tiles", output_shard_width_tiles},
    };
    tt::tt_metal::KernelHandle compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_moe_post_combine_tilize/device/kernels/"
        "deepseek_moe_post_combine_tilize_compute.cpp",
        op_cores,
        tt::tt_metal::ComputeConfig{.named_compile_args = compute_named_ct_args});

    // writer
    std::unordered_map<std::string, uint32_t> writer_named_ct_args = {
        {"tilize_output_cb_id", tilize_output_cb_id},
        {"num_tiles", output_shard_width_tiles},
    };
    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_moe_post_combine_tilize/device/kernels/"
        "deepseek_moe_post_combine_tilize_writer.cpp",
        op_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::NOC_1,
            .noc_mode = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC,
            .compile_args = {},
            .defines = {},
            .named_compile_args = writer_named_ct_args,
            .opt_level = tt::tt_metal::KernelBuildOptLevel::O2});

    bool is_row_major_shard_orientation = output_nd_shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    std::vector<tt::tt_metal::CoreCoord> cores =
        corerange_to_cores(op_cores, std::nullopt, is_row_major_shard_orientation);
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const auto& core = cores[i];

        // reader
        uint32_t intra_row_byte_offset;
        uint32_t row_page_offset;
        if (is_row_major_shard_orientation) {
            intra_row_byte_offset = (i % output_num_shards_wide) * output_shard_width_bytes;
            row_page_offset = (i / output_num_shards_wide) * tt::constants::TILE_HEIGHT;
        } else {
            intra_row_byte_offset = (i / output_num_shards_high) * output_shard_width_bytes;
            row_page_offset = (i % output_num_shards_high) * tt::constants::TILE_HEIGHT;
        }
        std::vector<uint32_t> reader_rt_args = {
            intra_row_byte_offset, row_page_offset, input_tensor.buffer()->address()};
        SetRuntimeArgs(program, reader_kernel_id, core, reader_rt_args);
    }

    return cached_program_t{
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .compute_kernel_id = compute_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .sharded_output_cb_handle = sharded_output_cb_handle,
         .cores = cores}};
}

void DeepseekMoEPostCombineTilizeProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const DeepseekMoEPostCombineTilizeParams&,
    const DeepseekMoEPostCombineTilizeInputs& tensor_args,
    ttnn::Tensor& tensor_return_value) {
    const ttnn::Tensor& input_tensor = tensor_args.input_tensor;
    const ttnn::Tensor& output_tensor = tensor_return_value;

    auto& program = cached_program.program;
    const tt::tt_metal::KernelHandle& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    const std::vector<tt::tt_metal::CoreCoord>& cores = cached_program.shared_variables.cores;

    tt::tt_metal::UpdateDynamicCircularBufferAddress(
        program, cached_program.shared_variables.sharded_output_cb_handle, *output_tensor.buffer());

    for (uint32_t i = 0; i < cores.size(); ++i) {
        auto& reader_args = GetRuntimeArgs(program, reader_kernel_id, cores[i]);
        reader_args[2] = input_tensor.buffer()->address();
    }
}

}  // namespace ttnn::experimental::prim
