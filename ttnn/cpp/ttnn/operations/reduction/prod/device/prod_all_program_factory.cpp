// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "prod_all_device_operation.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::prim {

using namespace tt;
using namespace tt::tt_metal;

ProgramDescriptor ProdAllDeviceOperation::ProdAllProgramFactory::create_descriptor(
    const ProdAllParams& /*operation_attributes*/, const ProdAllInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& input = tensor_args.input.mesh_tensor();
    const auto& output = tensor_return_value.mesh_tensor();

    ProgramDescriptor desc;

    CoreRange core({0, 0}, {0, 0});
    CoreRangeSet core_ranges(core);

    DataFormat in_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    DataFormat out_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t in_single_tile_size = tile_size(in_cb_data_format);
    uint32_t out_single_tile_size = tile_size(out_cb_data_format);

    uint32_t num_tiles = input.physical_volume() / input.tensor_spec().tile().get_tile_hw();

    TT_FATAL(num_tiles > 0, "Prod_all workload num_tiles must be > 0, got {}", num_tiles);

    uint32_t num_input_tiles = 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * in_single_tile_size,
        .core_ranges = core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_0),
            .data_format = in_cb_data_format,
            .page_size = in_single_tile_size,
        }}},
    });

    constexpr uint32_t output_cb_index = CBIndex::c_3;
    uint32_t num_output_tiles = 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * out_single_tile_size,
        .core_ranges = core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = out_cb_data_format,
            .page_size = out_single_tile_size,
        }}},
    });

    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(input).append_to(reader_compile_time_args);
    std::vector<uint32_t> writer_compile_time_args = {static_cast<uint32_t>(output_cb_index)};
    TensorAccessorArgs(output).append_to(writer_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_ranges;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = core_ranges;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    std::vector<uint32_t> compute_kernel_args = {
        num_tiles,  // per_core_block_cnt
        1           // per_core_block_size
    };

    bool fp32_dest_acc_en = true;
    // On Wormhole B0, HiFi4 must not be combined with fp32_dest_acc_en due to a hardware bug
    // (see tenstorrent/tt-metal#38306); drop to HiFi3 only on that arch. Other architectures keep HiFi4.
    const bool needs_wh_fp32_workaround =
        fp32_dest_acc_en && tensor_args.input.device()->arch() == tt::ARCH::WORMHOLE_B0;
    const auto math_fidelity =
        needs_wh_fp32_workaround ? tt::tt_metal::MathFidelity::HiFi3 : tt::tt_metal::MathFidelity::HiFi4;
    bool math_approx_mode = true;
    KernelDescriptor compute_desc;
    compute_desc.kernel_source = "ttnn/cpp/ttnn/operations/reduction/prod/device/kernels/compute/prod_all.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = core_ranges;
    compute_desc.compile_time_args = std::move(compute_kernel_args);
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = false,
        .math_approx_mode = math_approx_mode,
    };

    CoreCoord core_coord = {0, 0};
    reader_desc.emplace_runtime_args(core_coord, {input, num_tiles, 0u});
    writer_desc.emplace_runtime_args(core_coord, {output, /*num_tiles=*/1u, 0u});

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::prim
