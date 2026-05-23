// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "group_attn_matmul_program_factory.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

tt::tt_metal::ProgramDescriptor GroupAttnMatmulProgramFactory::create_descriptor(
    const GroupAttnMatmulParams& operation_attributes,
    const GroupAttnMatmulInputs& tensor_args,
    Tensor& tensor_return_value) {
    ProgramDescriptor desc;

    const auto& a = tensor_args.input_tensor_a;
    const auto& b = tensor_args.input_tensor_b;
    auto& output = tensor_return_value;

    const auto& ashape = a.padded_shape();
    const auto& bshape = b.padded_shape();

    tt::tt_metal::IDevice* device = a.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    tt::DataFormat in0_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat in1_data_format = tt::tt_metal::datatype_to_dataformat_converter(b.dtype());
    tt::DataFormat interm_data_format = fp32_dest_acc_en and in0_data_format == tt::DataFormat::Float32
                                            ? tt::DataFormat::Float32
                                            : tt::DataFormat::Float16_b;
    tt::DataFormat output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t in0_single_tile_size = tt::tile_size(in0_data_format);
    uint32_t in1_single_tile_size = tt::tile_size(in1_data_format);
    uint32_t interm_single_tile_size = tt::tile_size(interm_data_format);
    uint32_t output_single_tile_size = tt::tile_size(output_data_format);

    if (in0_data_format == tt::DataFormat::Float32 or in1_data_format == tt::DataFormat::Float32 or
        output_data_format == tt::DataFormat::Float32) {
        TT_FATAL(fp32_dest_acc_en == true, "when inputs/output are in fp32 format, fp32_dest_acc_en must be set");
    }

    tt::tt_metal::Buffer* src0_buffer = a.buffer();
    tt::tt_metal::Buffer* src1_buffer = b.buffer();
    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    // Load kernels on all device cores (cached program covers shape variation via
    // CB-size re-application on cache hit).
    CoreCoord device_compute_with_storage_grid = device->compute_with_storage_grid_size();
    CoreRangeSet all_device_cores{
        CoreRange({0, 0}, {device_compute_with_storage_grid.x - 1, device_compute_with_storage_grid.y - 1})};

    // ---- Shape-derived quantities (these vary across cache hits) ----
    const bool transpose_hw_bool = operation_attributes.transpose_hw.value_or(false);
    const uint32_t num_tokens_val = operation_attributes.num_tokens.value_or(0);
    uint32_t Q_HEADS = ashape[1];
    uint32_t KV_HEADS = bshape[1];
    uint32_t Mt = ashape[2] / TILE_HEIGHT;
    uint32_t Kt = ashape[3] / TILE_WIDTH;
    uint32_t in1_Kt = transpose_hw_bool ? Kt : bshape[2] / TILE_HEIGHT;
    uint32_t Nt = transpose_hw_bool ? num_tokens_val / TILE_HEIGHT : bshape[3] / TILE_WIDTH;
    uint32_t MtKt = Mt * Kt;
    uint32_t MtNt = Mt * Nt;
    uint32_t in1_KtNt = transpose_hw_bool ? bshape[2] / TILE_HEIGHT * in1_Kt : in1_Kt * Nt;
    uint32_t in1_CKtNt = KV_HEADS * in1_KtNt;

    // ---- Matmul/blocking params ----
    constexpr uint32_t out_subblock_h = 1;
    constexpr uint32_t in1_num_subblocks = 1;
    const uint32_t out_block_w = operation_attributes.out_subblock_w;
    const uint32_t in0_block_w = Kt;
    const uint32_t in0_block_num_tiles = in0_block_w * out_subblock_h;
    const uint32_t in0_subblock_num_tiles = in0_block_num_tiles;
    const uint32_t in1_block_num_tiles_per_kv_heads = in0_block_w * operation_attributes.out_subblock_w;
    const uint32_t in1_block_num_tiles = KV_HEADS * in1_block_num_tiles_per_kv_heads;
    const uint32_t in1_num_blocks = ((Nt - 1) / out_block_w) + 1;
    const uint32_t out_subblock_num_tiles = out_subblock_h * out_block_w;
    const uint32_t intermediate_num_tiles = out_subblock_num_tiles;
    const uint32_t in1_per_core_w = in1_num_subblocks * out_block_w;
    const uint32_t in1_block_w_tile_bytes = operation_attributes.out_subblock_w * in1_single_tile_size;
    const uint32_t ONE_ROW_BFLOAT16_BYTES =
        fp32_dest_acc_en and in0_data_format == tt::DataFormat::Float32 ? 128u : 64u;
    const uint32_t bfloat16_row_bytes = ONE_ROW_BFLOAT16_BYTES * out_block_w;

    // ---- Work distribution ----
    uint32_t num_active_cores = std::max(Q_HEADS, TILE_HEIGHT);
    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_output_blocks_per_core_group_1,
         num_output_blocks_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(
                operation_attributes.compute_with_storage_grid_size, num_active_cores, operation_attributes.row_major);
    TT_FATAL(
        num_output_blocks_per_core_group_1 == 1 and num_output_blocks_per_core_group_2 == 0,
        "Group attention matmul only supports one q_heads per core. Increase compute grid size to at least have as "
        "many cores as q_heads!");

    // ---- Mcast topology (first 32 cores are senders) ----
    CoreRangeSet mcast_receiver_cores = num_cores_to_corerangeset(
        Q_HEADS, operation_attributes.compute_with_storage_grid_size, operation_attributes.row_major);
    CoreRange mcast_receiver_cores_bounding_box = mcast_receiver_cores.bounding_box();
    uint32_t mcast_num_dests = mcast_receiver_cores.num_cores();
    uint32_t mcast_num_cores = mcast_receiver_cores_bounding_box.size();
    CoreCoord top_left_core = mcast_receiver_cores_bounding_box.start_coord;
    CoreCoord bottom_right_core = mcast_receiver_cores_bounding_box.end_coord;
    CoreCoord top_left_core_physical = device->worker_core_from_logical_core(top_left_core);
    CoreCoord bottom_right_core_physical = device->worker_core_from_logical_core(bottom_right_core);

    CoreCoord mcast_sender_grid =
        ((CoreRangeSet)num_cores_to_corerangeset(
             TILE_HEIGHT, operation_attributes.compute_with_storage_grid_size, operation_attributes.row_major))
            .bounding_box()
            .grid_size();
    std::vector<uint32_t> in1_mcast_sender_noc_x;
    std::vector<uint32_t> in1_mcast_sender_noc_y;
    in1_mcast_sender_noc_x.reserve(mcast_sender_grid.x);
    in1_mcast_sender_noc_y.reserve(mcast_sender_grid.y);
    for (uint32_t core_idx_x = 0; core_idx_x < mcast_sender_grid.x; ++core_idx_x) {
        in1_mcast_sender_noc_x.push_back(device->worker_core_from_logical_core({core_idx_x, 0}).x);
    }
    for (uint32_t core_idx_y = 0; core_idx_y < mcast_sender_grid.y; ++core_idx_y) {
        in1_mcast_sender_noc_y.push_back(device->worker_core_from_logical_core({0, core_idx_y}).y);
    }

    // ---- Semaphores (workload-scoped: IDs 0/1 reserved across all_device_cores) ----
    constexpr uint32_t in1_mcast_sender_semaphore_id = 0;
    constexpr uint32_t in1_mcast_receiver_semaphore_id = 1;
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = in1_mcast_sender_semaphore_id,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = all_device_cores,
        .initial_value = INVALID,
    });
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = in1_mcast_receiver_semaphore_id,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = all_device_cores,
        .initial_value = INVALID,
    });

    // ---- Circular buffers (sharded variants use CBDescriptor::buffer so the
    // framework patches the dynamic address on cache hit; CB total_size and
    // page_size are NOT patched on cache hit — sizing that varies with input
    // shape is folded into compute_program_hash() via padded_shape). ----
    const bool in0_is_sharded = a.is_sharded();
    const bool in1_is_sharded = b.is_sharded();
    const bool output_is_sharded = output.is_sharded();

    constexpr uint8_t src0_cb_index = tt::CBIndex::c_0;
    {
        const uint32_t cb0_num_input_tiles = in0_is_sharded ? (a.shard_spec().value().numel() / TILE_HW) : in0_block_w;
        desc.cbs.push_back(CBDescriptor{
            .total_size = cb0_num_input_tiles * in0_single_tile_size,
            .core_ranges = all_device_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = src0_cb_index,
                .data_format = in0_data_format,
                .page_size = in0_single_tile_size,
            }}},
            .buffer = in0_is_sharded ? src0_buffer : nullptr,
        });
    }

    constexpr uint8_t src1_cb_index = tt::CBIndex::c_1;
    {
        const uint32_t cb1_num_input_tiles = 2 * in1_block_num_tiles;
        desc.cbs.push_back(CBDescriptor{
            .total_size = cb1_num_input_tiles * in1_single_tile_size,
            .core_ranges = all_device_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = src1_cb_index,
                .data_format = in1_data_format,
                .page_size = in1_single_tile_size,
            }}},
        });
    }

    constexpr uint8_t src2_cb_index = tt::CBIndex::c_2;
    if (in1_is_sharded) {
        const uint32_t cb2_num_input_tiles = b.shard_spec().value().numel() / TILE_HW;
        desc.cbs.push_back(CBDescriptor{
            .total_size = cb2_num_input_tiles * in1_single_tile_size,
            .core_ranges = all_device_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = src2_cb_index,
                .data_format = in1_data_format,
                .page_size = in1_single_tile_size,
            }}},
            .buffer = src1_buffer,
        });
    }

    constexpr uint8_t cb_intermed0_index = tt::CBIndex::c_3;
    {
        const uint32_t interm_cb_num_tiles = 2 * intermediate_num_tiles;
        desc.cbs.push_back(CBDescriptor{
            .total_size = interm_cb_num_tiles * interm_single_tile_size,
            .core_ranges = all_device_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = cb_intermed0_index,
                .data_format = interm_data_format,
                .page_size = interm_single_tile_size,
            }}},
        });
    }

    constexpr uint8_t cb_intermed1_index = tt::CBIndex::c_4;
    desc.cbs.push_back(CBDescriptor{
        .total_size = MtNt * interm_single_tile_size,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_intermed1_index,
            .data_format = interm_data_format,
            .page_size = interm_single_tile_size,
        }}},
    });

    constexpr uint8_t output_cb_index = tt::CBIndex::c_5;
    {
        const uint32_t num_output_tiles = output_is_sharded ? (output.shard_spec().value().numel() / TILE_HW) : MtNt;
        desc.cbs.push_back(CBDescriptor{
            .total_size = num_output_tiles * output_single_tile_size,
            .core_ranges = all_device_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = output_cb_index,
                .data_format = output_data_format,
                .page_size = output_single_tile_size,
            }}},
            .buffer = output_is_sharded ? dst_buffer : nullptr,
        });
    }

    // ---- Kernels ----
    std::vector<uint32_t> reader_compile_time_args = {
        static_cast<uint32_t>(transpose_hw_bool),
        static_cast<uint32_t>(operation_attributes.row_major),
        operation_attributes.out_subblock_w,
    };
    tt::tt_metal::TensorAccessorArgs(*src1_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {
        static_cast<uint32_t>(output_cb_index),
        operation_attributes.out_subblock_w,
        intermediate_num_tiles,
    };
    tt::tt_metal::TensorAccessorArgs(*src0_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    KernelDescriptor::Defines reader_defines;
    KernelDescriptor::Defines writer_defines;
    if (in0_is_sharded) {
        writer_defines.emplace_back("IN0_SHARDED", "1");
    }
    if (in1_is_sharded) {
        reader_defines.emplace_back("IN1_SHARDED", "1");
    }
    if (output_is_sharded) {
        writer_defines.emplace_back("OUT_SHARDED", "1");
    }

    tt::tt_metal::NOC reader_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());
    const bool reader_noc_is_NOC_0 = reader_noc == tt::tt_metal::NOC::NOC_0;
    tt::tt_metal::NOC writer_noc = reader_noc_is_NOC_0 ? tt::tt_metal::NOC::NOC_1 : tt::tt_metal::NOC::NOC_0;

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/device/kernels/dataflow/"
        "reader_mcast_transformer_group_attn_matmul.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_device_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.defines = std::move(reader_defines);
    reader_desc.config = DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
        .noc = reader_noc,
    };

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/device/kernels/dataflow/"
        "writer_transformer_group_attn_matmul.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_device_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.defines = std::move(writer_defines);
    writer_desc.config = DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
        .noc = writer_noc,
    };

    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/device/kernels/compute/"
        "transformer_group_attn_matmul.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_device_cores;
    compute_desc.compile_time_args = {
        static_cast<uint32_t>(transpose_hw_bool),
        operation_attributes.out_subblock_w,
        out_subblock_num_tiles,
        intermediate_num_tiles,
    };
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
    };

    // ---- Per-core runtime args ----
    uint32_t Nt_bytes = Nt * in1_single_tile_size;
    uint32_t out_last_subblock_w = Nt % out_block_w == 0 ? out_block_w : Nt % out_block_w;
    uint32_t in1_last_block_w_tile_read_bytes = out_last_subblock_w * in1_single_tile_size;
    uint32_t in1_last_block_addr_skip =
        (operation_attributes.out_subblock_w - out_last_subblock_w) * in1_single_tile_size;
    uint32_t bfloat16_Nt_bytes = ONE_ROW_BFLOAT16_BYTES * Nt;
    uint32_t bfloat16_last_row_bytes_read = ONE_ROW_BFLOAT16_BYTES * out_last_subblock_w;

    CoreRange all_cores_bounding_box = all_cores.bounding_box();
    std::vector<CoreCoord> cores = grid_to_cores_with_noop(
        all_cores_bounding_box.end_coord.x,
        all_cores_bounding_box.end_coord.y,
        device_compute_with_storage_grid.x,
        device_compute_with_storage_grid.y,
        operation_attributes.row_major);
    uint32_t g1_numcores = core_group_1.num_cores();

    reader_desc.runtime_args.reserve(cores.size());
    writer_desc.runtime_args.reserve(cores.size());
    compute_desc.runtime_args.reserve(cores.size());

    uint32_t num_blocks_written = 0;
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];
        const uint32_t num_output_blocks_per_core =
            i < num_cores ? (i < g1_numcores ? num_output_blocks_per_core_group_1 : num_output_blocks_per_core_group_2)
                          : 0;

        const uint32_t kv_heads_id = KV_HEADS == 0 ? 0u : i / std::max<uint32_t>(Q_HEADS / KV_HEADS, 1u);
        const uint32_t has_work_for_q_heads = i < Q_HEADS;
        const uint32_t has_work_for_mcast_kv_heads = i < num_active_cores;
        const uint32_t mcast_num_cores_for_core = mcast_num_cores - static_cast<uint32_t>(i < mcast_num_cores);
        const uint32_t in1_mcast_num_dests =
            Q_HEADS < TILE_HEIGHT ? std::min(mcast_num_cores_for_core, num_active_cores - 1) : mcast_num_dests - 1;

        std::vector<uint32_t> reader_runtime_args = {
            has_work_for_mcast_kv_heads,
            has_work_for_q_heads,
            src1_buffer->address(),
            Mt,
            Nt,
            KV_HEADS,
            in1_CKtNt,
            in1_CKtNt * TILE_HEIGHT,
            num_output_blocks_per_core,
            0,  // in1_start_id

            in0_block_w,
            out_block_w,
            in1_num_subblocks,
            in1_num_blocks,
            in1_block_num_tiles,

            Nt_bytes,
            in1_block_w_tile_bytes,
            out_last_subblock_w,
            in1_last_block_w_tile_read_bytes,
            in1_last_block_addr_skip,

            static_cast<uint32_t>(reader_noc_is_NOC_0 ? top_left_core_physical.x : bottom_right_core_physical.x),
            static_cast<uint32_t>(reader_noc_is_NOC_0 ? top_left_core_physical.y : bottom_right_core_physical.y),
            static_cast<uint32_t>(reader_noc_is_NOC_0 ? bottom_right_core_physical.x : top_left_core_physical.x),
            static_cast<uint32_t>(reader_noc_is_NOC_0 ? bottom_right_core_physical.y : top_left_core_physical.y),
            in1_mcast_num_dests,
            mcast_num_cores_for_core,
            mcast_num_cores,
            in1_mcast_sender_semaphore_id,
            in1_mcast_receiver_semaphore_id,
            in1_block_num_tiles * in1_single_tile_size,
            i,  // in1_mcast_sender_id
            static_cast<uint32_t>(in1_mcast_sender_noc_x.size()),
            static_cast<uint32_t>(in1_mcast_sender_noc_y.size()),
        };
        reader_runtime_args.insert(
            reader_runtime_args.end(), in1_mcast_sender_noc_x.begin(), in1_mcast_sender_noc_x.end());
        reader_runtime_args.insert(
            reader_runtime_args.end(), in1_mcast_sender_noc_y.begin(), in1_mcast_sender_noc_y.end());
        reader_desc.runtime_args.emplace_back(core, std::move(reader_runtime_args));

        std::vector<uint32_t> writer_runtime_args = {
            has_work_for_q_heads,
            src0_buffer->address(),
            dst_buffer->address(),
            Mt,
            Kt,
            Nt,
            MtKt,
            num_output_blocks_per_core,
            num_blocks_written * MtKt,
            num_blocks_written * MtNt,

            in0_block_w,
            in1_num_subblocks,
            in1_num_blocks,
            MtNt,

            bfloat16_row_bytes,
            bfloat16_Nt_bytes,
            bfloat16_last_row_bytes_read,
        };
        writer_desc.runtime_args.emplace_back(core, std::move(writer_runtime_args));

        std::vector<uint32_t> compute_runtime_args = {
            has_work_for_q_heads,
            num_output_blocks_per_core,
            Mt,
            kv_heads_id * in1_block_num_tiles_per_kv_heads,
            (KV_HEADS - kv_heads_id) * in1_block_num_tiles_per_kv_heads,

            in0_block_w,
            out_subblock_h,
            in1_num_subblocks,
            in1_num_blocks,
            in0_block_num_tiles,
            in1_block_num_tiles,
            MtNt,

            in0_subblock_num_tiles,
            in1_per_core_w,
        };
        compute_desc.runtime_args.emplace_back(core, std::move(compute_runtime_args));

        num_blocks_written += num_output_blocks_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::experimental::prim
