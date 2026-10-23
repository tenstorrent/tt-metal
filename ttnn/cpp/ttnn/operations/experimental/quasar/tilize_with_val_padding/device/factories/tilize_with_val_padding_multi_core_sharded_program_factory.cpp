// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_with_val_padding_multi_core_sharded_program_factory.hpp"

#include <cmath>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/experimental/quasar/tilize_with_val_padding/device/factories/tilize_with_val_padding_factory_helper.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim::qsr {

ProgramDescriptor TilizeWithValPaddingMultiCoreShardedFactory::create_descriptor(
    const TilizeWithValPaddingParams& operation_attributes, const Tensor& input_tensor, Tensor& tensor_return_value) {
    const Tensor& a = input_tensor;
    const Tensor& output = tensor_return_value;
    auto pad_value = operation_attributes.pad_value;
    bool src_sharded = a.memory_config().is_sharded();
    bool out_sharded = output.memory_config().is_sharded();

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    bool fp32_llk_acc = a.dtype() == DataType::FLOAT32 || a.dtype() == DataType::FP8_E4M3 ||
                        output.dtype() == DataType::FP8_E4M3 || output.dtype() == DataType::BFLOAT8_B;

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

    Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    const uint32_t src0_cb_index = tt::CBIndex::c_1;
    const uint32_t src1_cb_index = tt::CBIndex::c_0;
    const uint32_t src2_cb_index = tt::CBIndex::c_2;
    const uint32_t output_cb_index = tt::CBIndex::c_16;

    ProgramDescriptor desc;

    // Sharded input CB — globally allocated to the input buffer; framework patches
    // the CB address on cache hits via cb.buffer.
    {
        CBDescriptor cb_src0;
        cb_src0.total_size = num_input_rows * input_shard_width_bytes;
        cb_src0.core_ranges = all_cores;
        cb_src0.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = input_cb_data_format,
            .page_size = input_shard_width_bytes,
        });
        if (src_sharded) {
            cb_src0.buffer = a.buffer();
        }
        desc.cbs.push_back(std::move(cb_src0));
    }

    desc.cbs.push_back(CBDescriptor{
        .total_size = ntiles_per_batch * 2 * input_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src1_cb_index),
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = input_shard_width_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src2_cb_index),
            .data_format = input_cb_data_format,
            .page_size = input_shard_width_bytes,
        }}},
    });

    // Sharded output CB — globally allocated to the output buffer.
    {
        CBDescriptor cb_output;
        cb_output.total_size = ntiles_per_core * output_single_tile_size;
        cb_output.core_ranges = all_cores;
        cb_output.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = output_cb_data_format,
            .page_size = output_single_tile_size,
        });
        if (out_sharded) {
            cb_output.buffer = dst_buffer;
        }
        desc.cbs.push_back(std::move(cb_output));
    }

    /** reader
     */
    std::vector<uint32_t> reader_ct_args = {
        static_cast<uint32_t>(src0_cb_index),
        static_cast<uint32_t>(src1_cb_index),
        static_cast<uint32_t>(src2_cb_index),
    };

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/quasar/tilize_with_val_padding/device/kernels/dataflow/"
        "reader_unary_pad_height_width_sharded.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.config = ReaderConfigDescriptor{};

    /** writer
     */
    std::vector<uint32_t> writer_ct_args = {output_cb_index};
    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/quasar/tilize_with_val_padding/device/kernels/dataflow/"
        "writer_unary_sharded.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};

    /** compute
     */
    std::vector<uint32_t> compute_args = {
        static_cast<uint32_t>(nblocks_per_core),  // per_core_block_cnt
        static_cast<uint32_t>(ntiles_per_block),  // per_block_ntiles
    };

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (fp32_llk_acc) {
        unpack_to_dest_mode[tt::CBIndex::c_0] = UnpackToDestMode::UnpackToDestFp32;
    }

    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/quasar/tilize_with_val_padding/device/kernels/compute/tilize.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = std::move(compute_args);
    compute_desc.config = ComputeConfigDescriptor{
        .fp32_dest_acc_en = fp32_llk_acc,
        .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
    };

    uint32_t packed_pad_value = detail::get_packed_value(a, pad_value);

    // Sharded readers/writers: the CBs themselves carry the buffer bindings, so no
    // Buffer* slot is needed in runtime args.
    for (const auto& core : corerange_to_cores(all_cores)) {
        reader_desc.emplace_runtime_args(
            core,
            {num_input_rows,
             input_shard_width_bytes,
             (num_input_rows / num_batches) * input_shard_width_bytes,
             ntiles_per_batch,
             num_padded_rows,
             num_batches,
             packed_pad_value});
        writer_desc.emplace_runtime_args(core, {ntiles_per_core});
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::prim::qsr
