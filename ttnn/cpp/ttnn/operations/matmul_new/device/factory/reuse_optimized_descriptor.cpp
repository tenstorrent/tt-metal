// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reuse_optimized_descriptor.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/tensor/shape/shape.hpp"

using namespace tt;
using namespace tt::constants;

namespace ttnn::prim::matmul_new_detail {

tt::tt_metal::ProgramDescriptor ReuseOptimizedDescriptorFactory::create_descriptor(
    const MatmulParams& operation_attributes,
    const MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    using namespace tt::tt_metal;

    const auto& program_config =
        std::get<operations::matmul::MatmulMultiCoreReuseProgramConfig>(operation_attributes.program_config.value());

    TT_FATAL(operation_attributes.output_dtype.has_value(), "Output dtype should have been provided");
    TT_FATAL(operation_attributes.compute_kernel_config.has_value(), "Compute kernel config should have been provided");
    TT_FATAL(operation_attributes.bcast_batch.has_value(), "Bcast batch should have been provided");

    const auto& a = tensor_args.input_tensors.at(0);
    const auto& b = tensor_args.input_tensors.at(1);
    auto& output = tensor_return_value.at(0);

    bool bcast_batch = operation_attributes.bcast_batch.value();
    bool transpose_a = operation_attributes.transpose_a;
    bool transpose_b = operation_attributes.transpose_b;
    bool untilize_out = operation_attributes.untilize_out;
    auto compute_with_storage_grid_size = program_config.compute_with_storage_grid_size;
    uint32_t in0_block_w = program_config.in0_block_w;
    uint32_t out_subblock_h = program_config.out_subblock_h;
    uint32_t out_subblock_w = program_config.out_subblock_w;
    uint32_t per_core_M = program_config.per_core_M;
    uint32_t per_core_N = program_config.per_core_N;

    const auto& ashape = operations::matmul::utilities::get_matmul_tensor_padded_shape(a, transpose_a);
    const auto& bshape = operations::matmul::utilities::get_matmul_tensor_padded_shape(b, transpose_b);
    auto in0_tile = operations::matmul::utilities::get_matmul_tile(a, transpose_a);
    auto in1_tile = operations::matmul::utilities::get_matmul_tile(b, transpose_b);

    tt::DataFormat in0_data_format = datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat in1_data_format = datatype_to_dataformat_converter(b.dtype());
    tt::DataFormat output_data_format = datatype_to_dataformat_converter(operation_attributes.output_dtype.value());

    IDevice* device = a.device();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config.value());

    uint32_t B = get_batch_size(ashape);
    uint32_t M = operations::matmul::utilities::get_M_dim(ashape, in0_tile, false);
    uint32_t K = operations::matmul::utilities::get_K_dim(ashape, in0_tile);
    uint32_t N = operations::matmul::utilities::get_N_dim(bshape, in1_tile);

    const auto ashape_logical = operations::matmul::utilities::get_matmul_tensor_logical_shape(a, transpose_a);
    const auto in0_last_ktile_w = ashape_logical[-1] % in0_tile.get_width();

    // ---- Derived parameters ----
    uint32_t batch_scale_factor = per_core_M > M ? per_core_M / M : 1;
    uint32_t per_core_M_per_batch = per_core_M > M ? M : per_core_M;
    uint32_t num_blocks = (K / in0_block_w);
    bool packer_l1_acc_en = packer_l1_acc && (num_blocks > 2);

    tt::DataFormat interm0_data_format = packer_l1_acc_en
                                             ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                                             : (fp32_dest_acc_en ? tt::DataFormat::Float32 : output_data_format);

    bool in0_transpose_tile = in0_tile.get_transpose_of_faces() && in0_tile.get_transpose_within_face();
    bool in1_transpose_tile = in1_tile.get_transpose_of_faces() && in1_tile.get_transpose_within_face();

    auto output_tile = tt::tt_metal::Tile({in0_tile.get_height(), in1_tile.get_width()});
    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    uint32_t output_single_tile_size = output_tile.get_tile_size(output_data_format);
    uint32_t interm0_single_tile_size = output_tile.get_tile_size(interm0_data_format);

    Buffer* in0_buffer = a.buffer();
    Buffer* in1_buffer = b.buffer();
    Buffer* out_buffer = output.buffer();
    bool in0_is_sharded = a.is_sharded();
    bool in1_is_sharded = b.is_sharded();
    bool output_is_sharded = output.is_sharded();

    uint32_t in0_block_num_tiles = per_core_M_per_batch * in0_block_w;
    uint32_t in0_CB_tiles = in0_block_num_tiles;
    if (in0_is_sharded) {
        in0_CB_tiles = per_core_M * K;
    } else {
        in0_CB_tiles *= 2;
    }
    uint32_t in0_CB_size = in0_CB_tiles * in0_single_tile_size;
    uint32_t in1_block_num_tiles = per_core_N * in0_block_w;
    uint32_t in1_CB_tiles = in1_block_num_tiles;
    if (in1_is_sharded) {
        in1_CB_tiles *= num_blocks * batch_scale_factor;
    } else {
        in1_CB_tiles *= 2;
    }
    uint32_t in1_CB_size = in1_CB_tiles * in1_single_tile_size;
    uint32_t out_block_tiles = per_core_M * per_core_N;
    uint32_t out_CB_tiles = out_block_tiles;
    uint32_t out_CB_size = out_CB_tiles * output_single_tile_size;
    uint32_t interm0_CB_size = out_CB_tiles * interm0_single_tile_size;

    uint32_t in0_num_subblocks = (per_core_M_per_batch / out_subblock_h);
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;
    uint32_t in1_num_subblocks = (per_core_N / out_subblock_w);
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;
    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;
    uint32_t out_num_subblocks_h = per_core_M_per_batch / out_subblock_h;
    uint32_t out_num_subblocks_w = in1_num_subblocks;
    uint32_t num_tiles_per_block_out = per_core_M_per_batch * per_core_N;
    uint32_t num_output_blocks_total = (B * M / per_core_M) * (N / per_core_N);

    std::optional<ShardSpec> shard_spec = std::nullopt;
    if (in0_is_sharded) {
        shard_spec = a.shard_spec().value();
    } else if (in1_is_sharded) {
        shard_spec = b.shard_spec().value();
    } else if (output_is_sharded) {
        shard_spec = output.shard_spec().value();
    }

    uint32_t num_cores = 0, num_blocks_per_core_group_1 = 0, num_blocks_per_core_group_2 = 0;
    CoreRangeSet all_cores, core_group_1, core_group_2;

    CoreCoord core_range = compute_with_storage_grid_size;
    if (shard_spec.has_value()) {
        all_cores = shard_spec.value().grid;
        num_cores = all_cores.num_cores();
        core_group_1 = all_cores;
        num_blocks_per_core_group_1 = num_output_blocks_total / num_cores * batch_scale_factor;
    } else {
        std::tie(
            num_cores,
            all_cores,
            core_group_1,
            core_group_2,
            num_blocks_per_core_group_1,
            num_blocks_per_core_group_2) = split_work_to_cores(core_range, num_output_blocks_total);
        num_blocks_per_core_group_1 *= batch_scale_factor;
        num_blocks_per_core_group_2 *= batch_scale_factor;
    }
    uint32_t g1_numcores = core_group_1.num_cores();

    // ---- Build ProgramDescriptor ----
    ProgramDescriptor desc;

    // -- Circular Buffers --
    // CB src0
    {
        CBDescriptor cb;
        cb.total_size = in0_CB_size;
        cb.core_ranges = all_cores;
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_0),
            .data_format = in0_data_format,
            .page_size = in0_single_tile_size,
            .tile = TileDescriptor(in0_tile),
        });
        if (in0_is_sharded) {
            cb.buffer = in0_buffer;
        }
        desc.cbs.push_back(std::move(cb));
    }
    // CB src1
    {
        CBDescriptor cb;
        cb.total_size = in1_CB_size;
        cb.core_ranges = all_cores;
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_1),
            .data_format = in1_data_format,
            .page_size = in1_single_tile_size,
            .tile = TileDescriptor(in1_tile),
        });
        if (in1_is_sharded) {
            cb.buffer = in1_buffer;
        }
        desc.cbs.push_back(std::move(cb));
    }
    // CB output (and possibly interm0 sharing)
    {
        uint32_t output_cb_index = tt::CBIndex::c_4;
        uint32_t interm0_cb_index = tt::CBIndex::c_5;

        if ((interm0_data_format != output_data_format) || (untilize_out && (in1_num_subblocks > 1))) {
            // Separate output and interm0 CBs
            desc.cbs.push_back(CBDescriptor{
                .total_size = out_CB_size,
                .core_ranges = CoreRangeSet({all_cores}),
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(output_cb_index),
                    .data_format = output_data_format,
                    .page_size = output_single_tile_size,
                    .tile = TileDescriptor(output_tile),
                }}},
                .buffer = output_is_sharded ? out_buffer : nullptr,
            });
            desc.cbs.push_back(CBDescriptor{
                .total_size = interm0_CB_size,
                .core_ranges = CoreRangeSet({all_cores}),
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(interm0_cb_index),
                    .data_format = interm0_data_format,
                    .page_size = interm0_single_tile_size,
                    .tile = TileDescriptor(output_tile),
                }}},
            });
        } else {
            // Shared buffer for output and interm0
            CBDescriptor cb;
            cb.total_size = out_CB_size;
            cb.core_ranges = CoreRangeSet({all_cores});
            cb.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(output_cb_index),
                .data_format = output_data_format,
                .page_size = output_single_tile_size,
                .tile = TileDescriptor(output_tile),
            });
            cb.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(interm0_cb_index),
                .data_format = interm0_data_format,
                .page_size = interm0_single_tile_size,
                .tile = TileDescriptor(output_tile),
            });
            if (output_is_sharded) {
                cb.buffer = out_buffer;
            }
            desc.cbs.push_back(std::move(cb));
        }
    }
    // Intermediate CB read workaround for Blackhole alignment
    bool in0_needs_intermediate_cb_read = false;
    bool in1_needs_intermediate_cb_read = false;
    if (device->arch() == tt::ARCH::BLACKHOLE) {
        in0_needs_intermediate_cb_read = ((in0_single_tile_size % 64) != 0);
        in1_needs_intermediate_cb_read = ((in1_single_tile_size % 64) != 0);
    }
    if (in1_needs_intermediate_cb_read) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = in1_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_9),
                .data_format = in1_data_format,
                .page_size = in1_single_tile_size,
                .tile = TileDescriptor(in1_tile),
            }}},
        });
    }
    if (in0_needs_intermediate_cb_read) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = in0_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_8),
                .data_format = in0_data_format,
                .page_size = in0_single_tile_size,
                .tile = TileDescriptor(in0_tile),
            }}},
        });
    }
    if (in0_transpose_tile) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = in0_CB_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(tt::CBIndex::c_10),
                .data_format = in0_data_format,
                .page_size = in0_single_tile_size,
                .tile = TileDescriptor(in0_tile),
            }}},
        });
    }

    // -- Kernels --
    const auto in0_tensor_stride_w = transpose_a ? M : 1;
    const auto in0_tensor_stride_h = transpose_a ? 1 : K;
    const auto in0_tensor_next_block_stride = in0_block_w * in0_tensor_stride_w;
    const auto in1_tensor_stride_w = transpose_b ? K : 1;
    const auto in1_tensor_stride_h = transpose_b ? 1 : N;
    const auto in1_tensor_next_block_stride = in0_block_w * in1_tensor_stride_h;

    // Reader kernel
    std::vector<uint32_t> reader_compile_time_args = {
        (uint32_t)in0_tensor_stride_w,
        (uint32_t)in0_tensor_stride_h,
        (uint32_t)in0_tensor_next_block_stride,
        (uint32_t)in0_block_w,
        (uint32_t)per_core_M_per_batch,
        (uint32_t)in0_block_num_tiles,
        (uint32_t)in0_last_ktile_w,
        (uint32_t)num_blocks,
        (uint32_t)bcast_batch,
        (uint32_t)(M * K),
    };
    TensorAccessorArgs(*in0_buffer).append_to(reader_compile_time_args);

    KernelDescriptor::Defines reader_defines;
    if (in0_is_sharded) {
        reader_defines.emplace_back("IN0_SHARDED", "1");
    }
    if (in0_needs_intermediate_cb_read) {
        reader_defines.emplace_back("INTERMEDIATE_CB_READ", "1");
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0.cpp";
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.defines = reader_defines;
    reader_desc.config = ReaderConfigDescriptor{};

    // Writer kernel
    std::vector<uint32_t> writer_compile_time_args = {
        (uint32_t)in1_tensor_stride_w,
        (uint32_t)in1_tensor_stride_h,
        (uint32_t)in1_tensor_next_block_stride,
        (uint32_t)per_core_N,
        (uint32_t)in0_block_w,
        (uint32_t)in1_block_num_tiles,
        (uint32_t)num_blocks,
        (uint32_t)bcast_batch,
        (uint32_t)(K * N),
        (uint32_t)1,
        (uint32_t)N,
        (uint32_t)out_subblock_w,
        (uint32_t)(out_subblock_h * N),
        (uint32_t)out_subblock_w,
        (uint32_t)out_subblock_h,
        (uint32_t)(out_subblock_w * out_subblock_h),
        (uint32_t)out_num_subblocks_w,
        (uint32_t)out_num_subblocks_h,
        (uint32_t)(M * N),
    };
    TensorAccessorArgs(*in1_buffer).append_to(writer_compile_time_args);
    TensorAccessorArgs(*out_buffer).append_to(writer_compile_time_args);

    KernelDescriptor::Defines writer_defines;
    if (in1_is_sharded) {
        writer_defines.emplace_back("IN1_SHARDED", "1");
    }
    if (output_is_sharded) {
        writer_defines.emplace_back("OUT_SHARDED", "1");
    }
    if (in1_needs_intermediate_cb_read) {
        writer_defines.emplace_back("INTERMEDIATE_CB_READ", "1");
    }

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_writer_bmm_tile_layout_in1.cpp";
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.defines = writer_defines;
    writer_desc.config = WriterConfigDescriptor{};

    // Compute kernel defines
    KernelDescriptor::Defines compute_defines;
    if (packer_l1_acc_en) {
        compute_defines.emplace_back("PACKER_L1_ACC", "1");
    }
    if (fp32_dest_acc_en) {
        compute_defines.emplace_back("FP32_DEST_ACC_EN", "1");
    }
    if (in1_transpose_tile) {
        compute_defines.emplace_back("IN1_TRANSPOSE_TILE", "1");
    }
    {
        std::map<std::string, std::string> stagger_defines;
        ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
            device->arch(), num_cores, stagger_defines);
        ttnn::operations::compute_throttle_utils::throttle_mm_perf(
            device->arch(),
            num_cores,
            stagger_defines,
            ttnn::get_throttle_level(operation_attributes.compute_kernel_config.value()));
        for (auto& [k, v] : stagger_defines) {
            compute_defines.emplace_back(k, v);
        }
    }

    // Compute kernel - group 1
    std::vector<uint32_t> compute_args_1 = {
        in0_block_w,
        in0_num_subblocks,
        in0_block_num_tiles,
        in0_subblock_num_tiles,
        in1_num_subblocks,
        in1_block_num_tiles,
        in1_per_core_w,
        num_blocks,
        1,
        1,
        out_subblock_h,
        out_subblock_w,
        out_subblock_num_tiles,
        num_blocks_per_core_group_1,
        out_block_tiles,
        untilize_out ? 1u : 0u,
        0u,  // get_batch_from_reader = false
        in0_transpose_tile ? 1u : 0u,
    };

    KernelDescriptor compute_desc_1;
    compute_desc_1.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp";
    compute_desc_1.core_ranges = core_group_1;
    compute_desc_1.compile_time_args = compute_args_1;
    compute_desc_1.defines = compute_defines;
    compute_desc_1.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
    };

    bool has_group_2 = !core_group_2.ranges().empty();
    KernelDescriptor compute_desc_2;
    if (has_group_2) {
        std::vector<uint32_t> compute_args_2 = {
            in0_block_w,
            in0_num_subblocks,
            in0_block_num_tiles,
            in0_subblock_num_tiles,
            in1_num_subblocks,
            in1_block_num_tiles,
            in1_per_core_w,
            num_blocks,
            1,
            1,
            out_subblock_h,
            out_subblock_w,
            out_subblock_num_tiles,
            num_blocks_per_core_group_2,
            out_block_tiles,
            untilize_out ? 1u : 0u,
            0u,
            in0_transpose_tile ? 1u : 0u,
        };
        compute_desc_2.kernel_source =
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp";
        compute_desc_2.core_ranges = core_group_2;
        compute_desc_2.compile_time_args = compute_args_2;
        compute_desc_2.defines = compute_defines;
        compute_desc_2.config = ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
        };
    }

    // -- Runtime args per core --
    bool row_major = false;
    if (shard_spec.has_value()) {
        row_major = shard_spec.value().orientation == ShardOrientation::ROW_MAJOR;
    }
    const auto cores = grid_to_cores(num_cores, core_range.x, core_range.y, row_major);

    uint32_t m_blocks_per_batch = M / per_core_M_per_batch;
    uint32_t n_blocks_per_batch = N / per_core_N;
    uint32_t blocks_per_batch = m_blocks_per_batch * n_blocks_per_batch;

    uint32_t in0_batch_stride = M * K;
    uint32_t in1_batch_stride = K * N;
    uint32_t in0_m_block_stride = per_core_M_per_batch * (transpose_a ? 1 : K);
    uint32_t in1_n_block_stride = per_core_N * (transpose_b ? K : 1);

    for (uint32_t i = 0, num_blocks_written = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];
        uint32_t num_output_blocks_per_core =
            i < g1_numcores ? num_blocks_per_core_group_1 : num_blocks_per_core_group_2;

        uint32_t start_batch = num_blocks_written / blocks_per_batch;
        uint32_t block_within_batch = num_blocks_written % blocks_per_batch;
        uint32_t start_m_block = block_within_batch / n_blocks_per_batch;
        uint32_t start_n_block = block_within_batch % n_blocks_per_batch;

        uint32_t in0_start_tile_id = (start_batch * in0_batch_stride) + (start_m_block * in0_m_block_stride);
        uint32_t in1_start_tile_id =
            (bcast_batch ? 0 : (start_batch * in1_batch_stride)) + (start_n_block * in1_n_block_stride);

        reader_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                (uint32_t)in0_buffer->address(),
                in0_start_tile_id,
                num_output_blocks_per_core,
            });

        writer_desc.runtime_args.emplace_back(
            core,
            KernelDescriptor::CoreRuntimeArgs{
                (uint32_t)in1_buffer->address(),
                in1_start_tile_id,
                num_output_blocks_per_core,
                (uint32_t)out_buffer->address(),
                num_blocks_written * num_tiles_per_block_out,
            });

        num_blocks_written += num_output_blocks_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc_1));
    if (has_group_2) {
        desc.kernels.push_back(std::move(compute_desc_2));
    }

    return desc;
}

}  // namespace ttnn::prim::matmul_new_detail
