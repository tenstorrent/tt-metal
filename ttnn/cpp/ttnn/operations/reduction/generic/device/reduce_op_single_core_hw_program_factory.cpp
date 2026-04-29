// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_op_device_operation.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <bit>
#include <cmath>
#include <map>

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor ReduceDeviceOperation::ReduceSingleCoreHwProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    const auto& a = tensor_args;
    auto& output = tensor_return_value;
    const auto& shape = a.padded_shape();
    uint32_t W = shape[3], H = shape[2], NC = shape[1] * shape[0];
    const uint32_t tile_height = a.tensor_spec().tile().get_height();
    const uint32_t tile_width = a.tensor_spec().tile().get_width();
    const uint32_t tile_hw = a.tensor_spec().tile().get_tile_hw();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(a.device()->arch(), operation_attributes.compute_kernel_config);

    uint32_t Wt = W / tile_width;
    uint32_t Ht = H / tile_height;

    // The single-core HW path uses REDUCE_SCALAR mode, which applies the
    // scaler twice internally (once per dimension). Here we compensate with
    // sqrt(scaler). However, sqrt of a negative number is NaN, so negative scalers
    // must not reach this code path. Instead negative scalers are handled via the two-step
    // W-then-H path where the scaler is applied once (see the reduce function in reduce_op.cpp).
    TT_FATAL(operation_attributes.scaler >= 0, "Scalar must be non-negative");
    float scaler = std::sqrt(operation_attributes.scaler);

    uint32_t num_tensor_tiles = NC * H * W / tile_hw;

    CoreCoord selected_core_coord = {0, 0};
    if (operation_attributes.sub_core_grids.has_value() && !operation_attributes.sub_core_grids->ranges().empty()) {
        const auto& r = operation_attributes.sub_core_grids->ranges().front();
        selected_core_coord = r.start_coord;
    }
    CoreRange core(selected_core_coord, selected_core_coord);
    CoreRangeSet core_set(core);

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);

    tt::DataFormat scaler_cb_data_format =
        src0_cb_data_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    uint32_t scaler_single_tile_size = tt::tile_size(scaler_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    tt_metal::Buffer* src0_buffer = a.buffer();

    // This should allocate a DRAM buffer on the device
    tt_metal::Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device");

    ProgramDescriptor desc;

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * src0_single_tile_size,
        .core_ranges = core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = src0_cb_data_format,
            .page_size = src0_single_tile_size,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = scaler_single_tile_size,
        .core_ranges = core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_2),
            .data_format = scaler_cb_data_format,
            .page_size = scaler_single_tile_size,
        }}},
    });

    uint32_t output_cb_index = tt::CBIndex::c_3;
    uint32_t num_output_tiles = 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * dst_single_tile_size,
        .core_ranges = core_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = dst_cb_data_format,
            .page_size = dst_single_tile_size,
        }}},
    });

    // For min/max with non-unity scalar, the GMPOOL hardware path only respects the scaler's
    // exponent, so the device reduces with scaler=1.0 and the user scalar is applied after the
    // reduction via SFPU mul_unary_tile inside the compute kernel.
    const bool use_post_mul = operation_attributes.post_mul_scaler != 1.0f;
    uint32_t post_mul_scaler_bits = std::bit_cast<uint32_t>(operation_attributes.post_mul_scaler);

    std::vector<uint32_t> reader_compile_time_args = {std::bit_cast<uint32_t>(scaler)};
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);

    if (operation_attributes.negate) {
        uint32_t acc_cb_index = tt::CBIndex::c_4;
        uint32_t num_acc_tiles = 1;
        desc.cbs.push_back(CBDescriptor{
            .total_size = num_acc_tiles * dst_single_tile_size,
            .core_ranges = core_set,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(acc_cb_index),
                .data_format = dst_cb_data_format,
                .page_size = dst_single_tile_size,
            }}},
        });

        uint32_t inv_cb_index = tt::CBIndex::c_5;
        uint32_t num_inv_tiles = 1;
        desc.cbs.push_back(CBDescriptor{
            .total_size = num_inv_tiles * dst_single_tile_size,
            .core_ranges = core_set,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(inv_cb_index),
                .data_format = dst_cb_data_format,
                .page_size = dst_single_tile_size,
            }}},
        });
    }

    std::vector<uint32_t> writer_compile_time_args = {output_cb_index};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    std::map<std::string, std::string> reduce_defines =
        reduce_op_utils::get_defines(operation_attributes.math_op, tt::tt_metal::ReduceOpDim::HW);
    if (use_post_mul) {
        reduce_defines["REDUCE_POST_MUL"] = "1";
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
        "reader_unary_reduce_universal_start_id.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_set;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.defines = {reduce_defines.begin(), reduce_defines.end()};
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = core_set;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.config = WriterConfigDescriptor{};

    std::vector<uint32_t> compute_kernel_args = {
        Ht,                    // Ht
        Wt,                    // Wt
        NC,                    // NC
        post_mul_scaler_bits,  // packed fp32 user scalar (only used if REDUCE_POST_MUL is set)
    };

    const std::string compute_kernel =
        std::string("ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce") +
        (operation_attributes.negate ? "_hw_neg" : "") + ".cpp";

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = compute_kernel;
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = core_set;
    compute_desc.compile_time_args = compute_kernel_args;
    compute_desc.defines = {reduce_defines.begin(), reduce_defines.end()};
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
    };

    reader_desc.emplace_runtime_args(selected_core_coord, {a.buffer(), num_tensor_tiles, 0u});

    TT_FATAL(Ht != 0 && Wt != 0, "Height and width in tiles must be non-zero (Ht={}, Wt={}, H={}, W={})", Ht, Wt, H, W);
    uint32_t out_dim_divider = Ht * Wt;

    writer_desc.emplace_runtime_args(selected_core_coord, {output.buffer(), num_tensor_tiles / out_dim_divider, 0u});

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::prim
