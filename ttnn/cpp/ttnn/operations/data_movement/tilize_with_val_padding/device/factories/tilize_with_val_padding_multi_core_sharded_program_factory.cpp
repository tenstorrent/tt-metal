// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_with_val_padding_multi_core_sharded_program_factory.hpp"

#include <cmath>

#include "ttnn/operations/cb_utils.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/data_movement/tilize_with_val_padding/device/factories/tilize_with_val_padding_factory_helper.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

TilizeWithValPaddingMultiCoreShardedFactory::cached_program_t TilizeWithValPaddingMultiCoreShardedFactory::create(
    const operation_attributes_t& operation_attributes, const Tensor& input_tensor, const Tensor& output_tensor) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    const Tensor& a = input_tensor;
    const Tensor& output = output_tensor;
    auto pad_value = operation_attributes.pad_value;
    bool src_sharded = a.memory_config().is_sharded();
    bool out_sharded = output.memory_config().is_sharded();

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    bool fp32_llk_acc = a.dtype() == DataType::FLOAT32;

    auto input_shard_spec = a.shard_spec().value();
    auto output_shard_spec = output.shard_spec().value();

    auto all_cores = output_shard_spec.grid;

    uint32_t num_batches = output.physical_volume() / (output.padded_shape()[-2] * output.padded_shape()[-1]);

    uint32_t num_input_rows = input_shard_spec.shape[0];
    uint32_t input_shard_width_bytes = input_shard_spec.shape[1] * a.element_size();
    uint32_t ntiles_per_core = output_shard_spec.shape[0] * output_shard_spec.shape[1] / TILE_HW;
    uint32_t ntiles_per_batch = ntiles_per_core / num_batches;
    uint32_t ntiles_per_block = output_shard_spec.shape[1] / TILE_WIDTH;
    uint32_t nblocks_per_core = output_shard_spec.shape[0] / TILE_HEIGHT;
    uint32_t num_padded_rows = output.padded_shape()[-2] - a.padded_shape()[-2];

    auto [src0_cb_index, cb_src0] = create_cb(
        tt::CBIndex::c_1,
        program,
        all_cores,
        input_shard_width_bytes,
        num_input_rows,
        input_cb_data_format,
        src_sharded ? a.buffer() : nullptr);

    auto [src1_cb_index, cb_src1] = create_cb(
        tt::CBIndex::c_0, program, all_cores, input_single_tile_size, ntiles_per_batch * 2, input_cb_data_format);

    auto [src2_cb_index, cb_src2] =
        create_cb(tt::CBIndex::c_2, program, all_cores, input_shard_width_bytes, 1, input_cb_data_format);

    auto [output_cb_index, cb_output] = create_cb(
        tt::CBIndex::c_16,
        program,
        all_cores,
        output_single_tile_size,
        ntiles_per_core,
        output_cb_data_format,
        out_sharded ? output.buffer() : nullptr);

    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    /** reader
     */
    KernelHandle unary_reader_kernel_id;
    std::vector<uint32_t> reader_ct_args = {
        (std::uint32_t)src0_cb_index,
        (std::uint32_t)src1_cb_index,
        (std::uint32_t)src2_cb_index,
    };

    unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/kernels/dataflow/"
        "reader_unary_pad_height_width_sharded.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));

    /** writer
     */
    KernelHandle unary_writer_kernel_id;
    std::vector<uint32_t> writer_ct_args = {
        output_cb_index,
    };
    unary_writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp",
        all_cores,
        WriterDataMovementConfig(writer_ct_args));

    /** compute
     */
    std::vector<uint32_t> compute_args = {
        (uint32_t)nblocks_per_core,  // per_core_block_cnt
        (uint32_t)ntiles_per_block,  // per_block_ntiles
    };

    CreateKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/tilize.cpp",
        all_cores,
        ComputeConfig{.fp32_dest_acc_en = fp32_llk_acc, .compile_args = compute_args});

    uint32_t packed_pad_value = detail::get_packed_value(a, pad_value);

    const std::array reader_rt_args = {
        num_input_rows,
        input_shard_width_bytes,
        (num_input_rows / num_batches) * input_shard_width_bytes,
        ntiles_per_batch,
        num_padded_rows,
        num_batches,
        packed_pad_value};
    tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, all_cores, reader_rt_args);

    const std::array writer_rt_args = {ntiles_per_core};
    tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, all_cores, writer_rt_args);

    return cached_program_t(
        std::move(program),
        shared_variables_t{
            .reader_kernel_id = unary_reader_kernel_id,
            .writer_kernel_id = unary_writer_kernel_id,
            .cb_src0 = cb_src0,
            .cb_output = cb_output});
}

void TilizeWithValPaddingMultiCoreShardedFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const Tensor& input_tensor,
    const Tensor& output_tensor) {
    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;

    auto* src_buffer = input_tensor.buffer();
    auto* dst_buffer = output_tensor.buffer();

    UpdateDynamicCircularBufferAddress(program, shared_variables.cb_src0, *src_buffer);
    UpdateDynamicCircularBufferAddress(program, shared_variables.cb_output, *dst_buffer);
}

}  // namespace ttnn::prim
