// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_matmul_program_factory.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"
#include <algorithm>
#include <utility>

#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/compute_throttle_utils.hpp"

using namespace tt;
using ttnn::operations::unary::UnaryOpType;
using ttnn::operations::unary::UnaryWithParam;

namespace ttnn::experimental::prim {

namespace ring_matmul_helpers {

enum class CORE_TYPE : uint32_t { IDLE_CORE = 0, WORKER_CORE = 1, HOP_CORE = 2 };

uint32_t get_preferred_noc(
    const ttnn::CoreCoord src,
    const ttnn::CoreCoord dst,
    const tt_metal::IDevice* device,
    const bool use_dedicated_noc = false) {
    /*
        NOC0: Preferred +x -> +y
        NOC1: Preferred -y -> -x
    */

    uint32_t src_x = src.x, src_y = src.y;
    uint32_t dst_x = dst.x, dst_y = dst.y;

    uint32_t MAX_X = device->grid_size().x;
    uint32_t MAX_Y = device->grid_size().y;

    uint32_t dist_right = src_x <= dst_x ? dst_x - src_x : MAX_X - src_x + dst_x;
    uint32_t dist_left = src_x < dst_x ? src_x + MAX_X - dst_x : src_x - dst_x;

    uint32_t dist_bottom = src_y <= dst_y ? dst_y - src_y : MAX_Y - src_y + dst_y;
    uint32_t dist_top = src_y < dst_y ? src_y + MAX_Y - dst_y : src_y - dst_y;

    uint32_t dist_noc_0 = dist_right + dist_bottom;
    uint32_t dist_noc_1 = dist_top + dist_left;

    uint32_t noc = dist_noc_0 < dist_noc_1 ? 0 : 1;

    return use_dedicated_noc ? 1 : noc;
}

}  // namespace ring_matmul_helpers

RingMatmulProgramFactory::shared_variables_t ring_matmul_create_program(
    tt::tt_metal::Program& program,
    const Tensor& a,
    const std::vector<Tensor>& b_tensors,
    const std::optional<const Tensor>& bias,
    const std::vector<Tensor>& output_tensors,
    const RingMatmulConfig& config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    std::optional<operations::unary::UnaryWithParam> fused_activation,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    bool untilize_out,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb) {
    using tt::tt_metal::num_cores_to_corerangeset;

    const auto& b = b_tensors[0];
    const auto num_output_cb = output_tensors.size();
    const auto batch = b_tensors.size();

    /* Input validation */
    TT_FATAL(
        a.is_sharded(),
        "Ring matmul requires input_tensor_a to be sharded across cores. Got memory_config: {}",
        a.memory_config());
    TT_FATAL(b.buffer() != nullptr, "Input tensor B must be allocated in a buffer on device");
    TT_FATAL(!output_tensors.empty(), "Ring matmul requires at least one output tensor");

    /* Get data buffers early */
    tt_metal::IDevice* device = a.device();
    tt_metal::Buffer* in0_buffer = a.buffer();
    tt_metal::Buffer* in1_buffer = b.buffer();

    const bool in1_is_dram_interleaved = in1_buffer->is_dram() && !b.is_sharded();
    const bool in1_is_dram_sharded = in1_buffer->is_dram() && b.is_sharded() && !global_cb.has_value();

    /* Core setup */
    constexpr bool row_major = true;
    CoreRangeSet all_worker_cores = a.shard_spec().value().grid;
    CoreRangeSet hop_cores = config.hop_cores;
    CoreRangeSet non_idle_cores = all_worker_cores.merge(hop_cores);
    CoreRangeSet all_cores = non_idle_cores;
    std::vector<CoreRange> non_idle_cores_vec;
    auto subdevice_cores = device->worker_cores(
        tt::tt_metal::HalProgrammableCoreType::TENSIX,
        sub_device_id.has_value() ? *sub_device_id : device->get_sub_device_ids().at(0));

    for (const auto& cr : subdevice_cores.ranges()) {
        auto intersection = non_idle_cores.intersection(cr);
        if (!intersection.empty()) {
            non_idle_cores_vec.push_back(intersection.bounding_box());
        }
    }
    all_cores = CoreRangeSet(non_idle_cores_vec);

    std::vector<CoreRange> ring_list = all_worker_cores.ranges();
    std::vector<CoreRange> hop_list = hop_cores.ranges();
    ring_list.insert(ring_list.end(), hop_list.begin(), hop_list.end());

    CoreRangeSet ring_cores = CoreRangeSet(ring_list);
    const uint32_t num_cores = all_worker_cores.num_cores();
    const uint32_t ring_size = num_cores;

    uint32_t num_hop_cores = hop_cores.num_cores();
    bool use_hop_cores = num_hop_cores > 0;

    /* Get data formats and other tile info */
    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(b.dtype());
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output_tensors[0].dtype());

    auto in0_tile = operations::matmul::utilities::get_matmul_tile(a, /*transpose_a=*/false);
    auto in1_tile = operations::matmul::utilities::get_matmul_tile(b, /*transpose_b=*/false);
    auto output_tile = tt::tt_metal::Tile({in0_tile.get_height(), in1_tile.get_width()});

    tt_metal::Buffer* bias_buffer = nullptr;
    tt::DataFormat bias_data_format = tt::DataFormat::Bfp8_b;
    if (bias.has_value()) {
        bias_buffer = bias->buffer();
        bias_data_format = tt_metal::datatype_to_dataformat_converter(bias->dtype());
    }

    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    uint32_t output_single_tile_size = output_tile.get_tile_size(output_data_format);

    const auto& a_shape = operations::matmul::utilities::get_matmul_tensor_padded_shape(a, /*transpose_a=*/false);
    const auto& b_shape = operations::matmul::utilities::get_matmul_tensor_padded_shape(b, /*transpose_b=*/false);

    uint32_t M = a_shape[-2];  // padded height
    uint32_t K = a_shape[-1];  // padded width (inner dimension)
    uint32_t N = b_shape[-1];  // padded width

    uint32_t in0_block_w = config.in0_block_w;
    uint32_t out_subblock_h = config.out_subblock_h;
    uint32_t out_subblock_w = config.out_subblock_w;
    uint32_t per_core_M = config.per_core_M;
    uint32_t per_core_N = config.per_core_N;
    uint32_t num_global_cb_receivers = config.num_global_cb_receivers;

    /* Compute kernel configuration */
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    uint32_t num_blocks = K / in0_block_w;
    bool packer_l1_acc_en = packer_l1_acc && num_blocks > 1;
    tt::DataFormat interm0_data_format = packer_l1_acc_en
                                             ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                                             : (fp32_dest_acc_en ? tt::DataFormat::Float32 : output_data_format);
    uint32_t interm0_single_tile_size = output_tile.get_tile_size(interm0_data_format);
    bool use_global_cb = global_cb.has_value();

    /* Inner dim padding */
    const uint32_t Kt_pad = in0_buffer->shard_spec().shape()[1] / in0_tile.get_width() * num_cores;
    uint32_t in0_shard_width_in_tiles = in0_buffer->shard_spec().shape()[1] / in0_tile.get_width();
    uint32_t in0_CB_tiles = per_core_M * in0_shard_width_in_tiles;
    uint32_t in0_CB_size = in0_CB_tiles * in0_single_tile_size;

    /* in1 CB configuration */
    uint32_t in1_shard_height_in_tiles = 0;
    uint32_t in1_shard_width_in_tiles = 0;
    uint32_t in1_CB_tiles = 0;

    const auto& bshape = operations::matmul::utilities::get_matmul_tensor_padded_shape(b, /*transpose=*/false);
    uint32_t in1_tensor_width_in_tiles = bshape[-1] / in1_tile.get_width();

    if (in1_is_dram_sharded || in1_is_dram_interleaved) {
        in1_CB_tiles = 2 * in0_shard_width_in_tiles * per_core_N;  // Double buffered
    } else {
        in1_shard_height_in_tiles = in1_buffer->shard_spec().shape()[0] / in1_tile.get_height();
        in1_shard_width_in_tiles = in1_buffer->shard_spec().shape()[1] / in1_tile.get_width() / num_global_cb_receivers;
        in1_CB_tiles = in1_shard_height_in_tiles * in1_shard_width_in_tiles;
    }
    uint32_t in1_CB_size = in1_CB_tiles * in1_single_tile_size;

    /* in2 - other shards */
    uint32_t in2_single_tile_size = in0_single_tile_size;
    uint32_t in2_CB_tiles = (ring_size - 1) * in0_CB_tiles;
    uint32_t in2_CB_size = in2_CB_tiles * in2_single_tile_size;

    /* out */
    uint32_t out_block_tiles = per_core_M * per_core_N;
    uint32_t out_CB_tiles = out_block_tiles;
    uint32_t out_CB_size = out_CB_tiles * output_single_tile_size;
    uint32_t interm0_CB_size = out_CB_tiles * interm0_single_tile_size;

    uint32_t K_ = K;
    std::vector<uint32_t> unpadded_in0_shard_widths_in_tiles(num_cores, 0);
    for (uint32_t i = 0; i < num_cores && K_ > 0; ++i) {
        unpadded_in0_shard_widths_in_tiles[i] = std::min(K_, in0_shard_width_in_tiles);
        K_ -= unpadded_in0_shard_widths_in_tiles[i];
    }

    /* Semaphores */
    auto in0_signal_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);

    /* Subblock calculations */
    uint32_t in0_num_subblocks = (per_core_M / out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;
    uint32_t in1_num_subblocks = per_core_N / out_subblock_w;
    uint32_t in1_block_height_in_tiles = in0_block_w;
    uint32_t in1_block_num_tiles = out_subblock_w * in1_block_height_in_tiles * in1_num_subblocks;
    uint32_t in1_block_size_bytes = in1_block_num_tiles * in1_single_tile_size;
    uint32_t in1_tensor_size_bytes = in1_block_num_tiles * num_blocks * in1_single_tile_size;
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;
    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    /* Create circular buffers */
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    tt_metal::CircularBufferConfig src0_cb_config =
        tt_metal::CircularBufferConfig(in0_CB_size, {{src0_cb_index, in0_data_format}})
            .set_page_size(src0_cb_index, in0_single_tile_size)
            .set_tile_dims(src0_cb_index, in0_tile)
            .set_globally_allocated_address(*in0_buffer);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, src0_cb_config);

    uint32_t src1_cb_index = tt::CBIndex::c_1;
    tt::tt_metal::CBHandle cb_src1;
    uint32_t remote_cb_index = tt::CBIndex::c_31;
    if (use_global_cb) {
        uint32_t in1_block_size_bytes = in1_single_tile_size * in1_block_num_tiles;
        tt_metal::CircularBufferConfig remote_cb_config =
            tt_metal::CircularBufferConfig((global_cb->size() / in1_block_size_bytes) * in1_block_size_bytes);
        remote_cb_config.remote_index(remote_cb_index)
            .set_page_size(in1_block_size_bytes)
            .set_data_format(in1_data_format);
        remote_cb_config.index(src1_cb_index).set_page_size(in1_single_tile_size).set_data_format(in1_data_format);
        cb_src1 = tt_metal::experimental::CreateCircularBuffer(program, all_cores, remote_cb_config, *global_cb);
    } else {
        tt_metal::CircularBufferConfig src1_cb_config =
            tt_metal::CircularBufferConfig(in1_CB_size, {{src1_cb_index, in1_data_format}})
                .set_page_size(src1_cb_index, in1_single_tile_size)
                .set_tile_dims(src1_cb_index, in1_tile);
        if (!in1_is_dram_interleaved && !in1_is_dram_sharded) {
            src1_cb_config = src1_cb_config.set_globally_allocated_address(*in1_buffer);
        }
        cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, src1_cb_config);
    }

    uint32_t src2_cb_index = tt::CBIndex::c_2;
    tt_metal::CircularBufferConfig src2_cb_config =
        tt_metal::CircularBufferConfig(in2_CB_size, {{src2_cb_index, in0_data_format}})
            .set_page_size(src2_cb_index, in2_single_tile_size)
            .set_tile_dims(src2_cb_index, in0_tile);
    tt_metal::CreateCircularBuffer(program, all_cores, src2_cb_config);

    uint32_t sync_cb_index = tt::CBIndex::c_3;
    uint32_t sync_cb_size_bytes = 16;
    tt_metal::CircularBufferConfig sync_cb_config =
        tt_metal::CircularBufferConfig(sync_cb_size_bytes, {{sync_cb_index, DataFormat::UInt16}})
            .set_page_size(sync_cb_index, sync_cb_size_bytes);
    tt_metal::CreateCircularBuffer(program, all_cores, sync_cb_config);

    uint32_t sync_cb2_index = tt::CBIndex::c_4;
    uint32_t sync_cb2_size_bytes = 16;
    tt_metal::CircularBufferConfig sync_cb2_config =
        tt_metal::CircularBufferConfig(sync_cb2_size_bytes, {{sync_cb2_index, DataFormat::UInt16}})
            .set_page_size(sync_cb2_index, sync_cb2_size_bytes);
    tt_metal::CreateCircularBuffer(program, all_cores, sync_cb2_config);

    uint32_t output_cb_index = tt::CBIndex::c_5;
    uint32_t interm0_cb_index = tt::CBIndex::c_6;
    tt_metal::CircularBufferConfig interm0_cb_config =
        tt_metal::CircularBufferConfig(0, {{interm0_cb_index, interm0_data_format}});
    tt_metal::CircularBufferConfig output_cb_config =
        tt_metal::CircularBufferConfig(0, {{output_cb_index, output_data_format}});

    if ((interm0_data_format != output_data_format) || (untilize_out && (in1_num_subblocks > 1))) {
        std::map<uint8_t, tt::DataFormat> interm0_cb_data_format_spec{
            {interm0_cb_index, interm0_data_format},
        };
        interm0_cb_config = tt_metal::CircularBufferConfig(interm0_CB_size, interm0_cb_data_format_spec)
                                .set_page_size(interm0_cb_index, interm0_single_tile_size)
                                .set_tile_dims(interm0_cb_index, output_tile);
        tt_metal::CreateCircularBuffer(program, all_cores, interm0_cb_config);

        for (uint32_t i = 0; i < output_tensors.size(); ++i) {
            const auto& out_buffer = output_tensors[i].buffer();
            output_cb_index += i * 2;
            TT_FATAL(
                output_cb_index <= tt::CBIndex::c_31,
                "Output circular buffer index {} exceeds maximum value {}",
                output_cb_index,
                tt::CBIndex::c_31);
            std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
                {output_cb_index, output_data_format},
            };
            output_cb_config = tt_metal::CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
                                   .set_page_size(output_cb_index, output_single_tile_size)
                                   .set_tile_dims(output_cb_index, output_tile)
                                   .set_globally_allocated_address(*out_buffer);
            tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);
        }
    } else {
        for (uint32_t i = 0; i < output_tensors.size(); ++i) {
            const auto& out_buffer = output_tensors[i].buffer();
            output_cb_index += i * 2;
            interm0_cb_index += i * 2;
            TT_FATAL(
                output_cb_index <= tt::CBIndex::c_31,
                "Output circular buffer index {} exceeds maximum value {}",
                output_cb_index,
                tt::CBIndex::c_31);
            TT_FATAL(
                interm0_cb_index <= tt::CBIndex::c_31,
                "Interm circular buffer index {} exceeds maximum value {}",
                interm0_cb_index,
                tt::CBIndex::c_31);
            std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
                {output_cb_index, output_data_format}, {interm0_cb_index, interm0_data_format}};
            output_cb_config = tt_metal::CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
                                   .set_page_size(output_cb_index, output_single_tile_size)
                                   .set_page_size(interm0_cb_index, interm0_single_tile_size)
                                   .set_tile_dims(output_cb_index, output_tile)
                                   .set_tile_dims(interm0_cb_index, output_tile)
                                   .set_globally_allocated_address(*out_buffer);
            tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);
        }
    }

    /* Kernel defines */
    std::map<std::string, std::string> mm_kernel_defines;
    std::map<std::string, std::string> mm_in1_kernel_defines;

    if (use_global_cb) {
        mm_in1_kernel_defines["ENABLE_GLOBAL_CB"] = "1";
        mm_kernel_defines["ENABLE_GLOBAL_CB"] = "1";
    }

    if (fused_activation.has_value()) {
        if (fused_activation.value().op_type == UnaryOpType::RELU) {
            mm_kernel_defines["PACK_RELU"] = "1";
        } else {
            using ttnn::operations::unary::utils::get_defines;
            mm_kernel_defines.merge(
                get_defines(fused_activation.value().op_type, fused_activation.value().params, "ACTIVATION", "i"));
        }
    }
    if (packer_l1_acc_en) {
        mm_kernel_defines["PACKER_L1_ACC"] = "1";
    }
    if (fp32_dest_acc_en) {
        mm_kernel_defines["FP32_DEST_ACC_EN"] = "1";
    }

    ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
        device->arch(), num_cores, mm_kernel_defines);
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(), num_cores, mm_kernel_defines, ttnn::get_throttle_level(compute_kernel_config));

    /* Kernel compile-time args */
    std::vector<uint32_t> in0_sender_compile_time_args = {
        (std::uint32_t)in0_shard_width_in_tiles,
        (std::uint32_t)per_core_M,
        (std::uint32_t)batch,
        (std::uint32_t)ring_size,
        (std::uint32_t)in0_signal_semaphore_id,
        (std::uint32_t)src0_cb_index,
        (std::uint32_t)src2_cb_index,
    };

    std::vector<uint32_t> in1_sender_writer_compile_time_args = {
        (std::uint32_t)in1_is_dram_interleaved,
        (std::uint32_t)in1_is_dram_sharded,
        (std::uint32_t)in1_block_height_in_tiles,
        (std::uint32_t)per_core_N,
        (std::uint32_t)in1_tensor_width_in_tiles,
        (std::uint32_t)num_blocks,
        (std::uint32_t)batch,
        (std::uint32_t)src1_cb_index,
        (std::uint32_t)sync_cb_index,
        (std::uint32_t)sync_cb2_index,
        (std::uint32_t)remote_cb_index,
        (std::uint32_t)0,  // fused_op_signaler
    };
    tt::tt_metal::TensorAccessorArgs(*in1_buffer).append_to(in1_sender_writer_compile_time_args);

    /* Compute kernel args */
    const uint32_t out_block_num_subblocks = out_block_tiles / out_subblock_num_tiles;
    TT_FATAL(
        out_block_num_subblocks == 1 || !untilize_out,
        "untilize_out is not supported for cases that out_block_num_subblocks > 1");
    std::vector<uint32_t> compute_kernel_args = {
        in0_block_w,
        in0_num_subblocks,
        in0_block_num_tiles,
        in0_subblock_num_tiles,
        in1_num_subblocks,
        in1_block_num_tiles,
        in1_block_size_bytes,
        in1_tensor_size_bytes,
        in1_per_core_w,
        num_blocks,
        out_subblock_h,
        out_subblock_w,
        out_subblock_num_tiles,
        batch,
        out_block_tiles,
        untilize_out,
        in1_is_dram_interleaved,
        in1_is_dram_sharded,
        src0_cb_index,
        src1_cb_index,
        src2_cb_index,
        sync_cb_index,
        sync_cb2_index,
    };
    compute_kernel_args.push_back(compute_kernel_args.size() + 1);
    for (uint32_t i = 0; i < num_output_cb; ++i) {
        compute_kernel_args.push_back(output_cb_index + i * 2);
    }
    for (uint32_t i = 0; i < num_output_cb; ++i) {
        compute_kernel_args.push_back(interm0_cb_index + i * 2);
    }

    /* NOC configuration */
    tt_metal::NOC in0_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    tt_metal::NOC in1_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());
    bool use_dedicated_noc = true;
    tt_metal::NOC_MODE noc_mode =
        use_dedicated_noc ? tt_metal::NOC_MODE::DM_DEDICATED_NOC : tt_metal::NOC_MODE::DM_DYNAMIC_NOC;

    /* Create kernels - point to ring_matmul specific kernels */
    auto mm_kernel_in0_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ring_matmul/device/kernels/dataflow/in0_reader.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = in0_noc,
            .noc_mode = noc_mode,
            .compile_args = in0_sender_compile_time_args});

    auto mm_kernel_in1_sender_writer_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ring_matmul/device/kernels/dataflow/in1_reader.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = in1_noc,
            .noc_mode = noc_mode,
            .compile_args = in1_sender_writer_compile_time_args,
            .defines = mm_in1_kernel_defines});

    auto mm_kernel = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ring_matmul/device/kernels/compute/ring_mm_compute.cpp",
        all_cores,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args,
            .defines = mm_kernel_defines});

    /* Runtime args setup */
    // Mapping from worker core y-coordinate (and column group) to DRAM bank IDs.
    // The DRAM banks are split into two column groups (left and right halves of the chip).
    // On Wormhole: hardcoded mapping; banks 0-3 left column (x <= 3), banks 4-11 right column.
    // On Blackhole: dynamically derived from optimal DRAM bank API; banks split at x <= 7.
    std::map<uint32_t, uint32_t> worker_y_to_dram_bank_first_col;
    std::map<uint32_t, uint32_t> worker_y_to_dram_bank_second_col;
    uint32_t first_col_max_x = device->arch() == tt::ARCH::WORMHOLE_B0 ? 3 : 7;
    uint32_t num_receiver_cores_per_dram = ring_size / in1_buffer->shard_spec().grid().num_cores();
    if (in1_is_dram_sharded) {
        if (device->arch() == tt::ARCH::WORMHOLE_B0) {
            worker_y_to_dram_bank_first_col[0] = 1;
            worker_y_to_dram_bank_first_col[4] = 2;
            worker_y_to_dram_bank_first_col[5] = 3;
            worker_y_to_dram_bank_first_col[9] = 0;

            worker_y_to_dram_bank_second_col[0] = 4;
            worker_y_to_dram_bank_second_col[1] = 6;
            worker_y_to_dram_bank_second_col[2] = 9;
            worker_y_to_dram_bank_second_col[4] = 10;
            worker_y_to_dram_bank_second_col[5] = 11;
            worker_y_to_dram_bank_second_col[6] = 8;
            worker_y_to_dram_bank_second_col[7] = 7;
            worker_y_to_dram_bank_second_col[9] = 5;
        } else {
            // Dynamically derive mapping from optimal DRAM bank API
            auto optimal_dram_workers = device->get_optimal_dram_bank_to_logical_worker_assignment(in1_noc);
            uint32_t num_banks = optimal_dram_workers.size();
            uint32_t banks_in_first_col = num_banks / 2;

            std::vector<std::pair<uint32_t, uint32_t>> first_col_anchors;   // (y, bank_id)
            std::vector<std::pair<uint32_t, uint32_t>> second_col_anchors;  // (y, bank_id)

            for (uint32_t bank = 0; bank < num_banks; ++bank) {
                const auto& core = optimal_dram_workers[bank];
                if (bank < banks_in_first_col) {
                    first_col_anchors.push_back({core.y, bank});
                } else {
                    second_col_anchors.push_back({core.y, bank});
                }
            }

            // Sort anchors by y-coordinate for nearest-neighbor lookup
            auto sort_by_y = [](const auto& a, const auto& b) { return a.first < b.first; };
            std::sort(first_col_anchors.begin(), first_col_anchors.end(), sort_by_y);
            std::sort(second_col_anchors.begin(), second_col_anchors.end(), sort_by_y);

            // Helper to find nearest bank for a given y-coordinate
            auto find_nearest_bank = [](uint32_t y,
                                        const std::vector<std::pair<uint32_t, uint32_t>>& anchors) -> uint32_t {
                if (anchors.empty()) {
                    return 0;  // Fallback
                }
                uint32_t best_bank = anchors[0].second;
                uint32_t best_dist = std::abs((int)y - (int)anchors[0].first);
                for (const auto& [anchor_y, bank] : anchors) {
                    uint32_t dist = std::abs((int)y - (int)anchor_y);
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_bank = bank;
                    }
                }
                return best_bank;
            };

            // Build complete maps for all possible y-coordinates (0 to max worker y)
            auto compute_grid = device->compute_with_storage_grid_size();
            for (uint32_t y = 0; y < compute_grid.y; ++y) {
                if (!first_col_anchors.empty()) {
                    worker_y_to_dram_bank_first_col[y] = find_nearest_bank(y, first_col_anchors);
                }
                if (!second_col_anchors.empty()) {
                    worker_y_to_dram_bank_second_col[y] = find_nearest_bank(y, second_col_anchors);
                }
            }
        }
    }

    uint32_t bank_id = 0;
    std::vector<uint32_t> bank_ids;

    auto all_cores_vec = corerange_to_cores(all_cores, std::nullopt, row_major);
    auto worker_cores_vec = corerange_to_cores(all_worker_cores, std::nullopt, row_major);
    auto hop_cores_vec = corerange_to_cores(hop_cores, std::nullopt, row_major);

    for (auto core : all_cores_vec) {
        auto all_worker_cores_iter = std::find(worker_cores_vec.begin(), worker_cores_vec.end(), core);
        auto hop_cores_iter = std::find(hop_cores_vec.begin(), hop_cores_vec.end(), core);
        bool core_is_in_all_worker_cores = all_worker_cores_iter != worker_cores_vec.end();
        bool core_is_in_hop_cores = hop_cores_iter != hop_cores_vec.end();
        if (!use_hop_cores) {
            core_is_in_hop_cores = false;
        }

        if (!core_is_in_all_worker_cores && !core_is_in_hop_cores) {
            auto core_type = ring_matmul_helpers::CORE_TYPE::IDLE_CORE;

            std::vector<uint32_t> mm_kernel_in0_args;
            mm_kernel_in0_args.push_back((std::uint32_t)core_type);
            tt_metal::SetRuntimeArgs(program, mm_kernel_in0_id, core, mm_kernel_in0_args);

            std::vector<uint32_t> mm_kernel_in1_sender_writer_args;
            mm_kernel_in1_sender_writer_args.push_back((std::uint32_t)core_type);
            tt_metal::SetRuntimeArgs(program, mm_kernel_in1_sender_writer_id, core, mm_kernel_in1_sender_writer_args);

            std::vector<uint32_t> mm_kernel_args;
            mm_kernel_args.push_back((std::uint32_t)core_type);
            tt_metal::SetRuntimeArgs(program, mm_kernel, core, mm_kernel_args);
        }
    }

    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& core = worker_cores_vec[i];
        const auto& core_noc = device->worker_core_from_logical_core(core);

        auto core_type = ring_matmul_helpers::CORE_TYPE::WORKER_CORE;
        CoreCoord next_core;
        if (i == 0 && use_hop_cores) {
            next_core = hop_cores_vec[0];
        } else {
            uint32_t next_i = i == 0 ? num_cores - 1 : i - 1;
            next_core = worker_cores_vec[next_i % num_cores];
        }
        const auto& next_core_noc = device->worker_core_from_logical_core(next_core);
        uint32_t noc = ring_matmul_helpers::get_preferred_noc(core_noc, next_core_noc, device, use_dedicated_noc);

        std::vector<uint32_t> mm_in0_args = {
            (std::uint32_t)core_type,
            i,
            next_core_noc.x,
            next_core_noc.y,
            noc,
            (std::uint32_t)false,
        };
        mm_in0_args.insert(
            mm_in0_args.end(), unpadded_in0_shard_widths_in_tiles.begin(), unpadded_in0_shard_widths_in_tiles.end());
        tt_metal::SetRuntimeArgs(program, mm_kernel_in0_id, core, mm_in0_args);

        std::vector<uint32_t> mm_in1_args = {
            (std::uint32_t)core_type,
            in1_buffer->address(),  // in1_tensor_addr
            i,                      // ring_idx
        };
        if (in1_is_dram_sharded) {
            // Look up bank_id based on core.y and which column group core.x belongs to
            if (core.x <= first_col_max_x) {
                auto it = worker_y_to_dram_bank_first_col.find(core.y);
                if (it == worker_y_to_dram_bank_first_col.end()) {
                    log_info(
                        tt::LogOp,
                        "ERROR: Worker core ({}, {}) y={} NOT FOUND in first-col map! Available y values:",
                        core.x,
                        core.y,
                        core.y);
                    for (const auto& [y, bank] : worker_y_to_dram_bank_first_col) {
                        log_info(tt::LogOp, "  y={}", y);
                    }
                }
                bank_id = it->second;
            } else {
                auto it = worker_y_to_dram_bank_second_col.find(core.y);
                if (it == worker_y_to_dram_bank_second_col.end()) {
                    log_info(
                        tt::LogOp,
                        "ERROR: Worker core ({}, {}) y={} NOT FOUND in second-col map! Available y values:",
                        core.x,
                        core.y,
                        core.y);
                    for (const auto& [y, bank] : worker_y_to_dram_bank_second_col) {
                        log_info(tt::LogOp, "  y={}", y);
                    }
                }
                bank_id = it->second;
            }

            uint32_t dram_read_offset = 0;
            /* TODO: This is a temporary solution to handle the dram read offset for the wormhole b0. */
            /* TODO: The dram read offset is x coordinate dependent for wormhole because all usage on wormhole assumes
             * input core range is column major, whereas blackhole usage is row major*/
            /* TODO: The correct behaviour is that first core next to dram bank always has offset 0 and then offset
             * increases by 1 for each core in the same row*/
            /* TODO: This logic should be removed once all usage of ring matmul asserts that the input core ranges are
             * arranged in row major order*/
            if (device->arch() == tt::ARCH::WORMHOLE_B0) {
                if (core.x % 2 == 0) {
                    dram_read_offset = 1;
                }
            } else {
                // For iterating through ring matmul cores in row major order
                dram_read_offset = i % num_receiver_cores_per_dram;
            }

            bank_ids.push_back(bank_id);
            uint32_t vc = 0;
            for (uint32_t j = 0; j < i; ++j) {
                auto core_prev = worker_cores_vec[j];
                if (core_prev.y == core.y) {
                    vc = (vc + 1) & 0x3;
                }
            }
            mm_in1_args.push_back((std::uint32_t)bank_id);
            mm_in1_args.push_back((std::uint32_t)vc);
            mm_in1_args.push_back((std::uint32_t)dram_read_offset);
        }
        tt_metal::SetRuntimeArgs(program, mm_kernel_in1_sender_writer_id, core, mm_in1_args);

        std::vector<uint32_t> mm_kernel_compute_args = {
            (std::uint32_t)core_type,
            i,
        };
        mm_kernel_compute_args.insert(
            mm_kernel_compute_args.end(),
            unpadded_in0_shard_widths_in_tiles.begin(),
            unpadded_in0_shard_widths_in_tiles.end());
        tt_metal::SetRuntimeArgs(program, mm_kernel, core, mm_kernel_compute_args);
    }

    for (uint32_t i = 0; i < num_hop_cores; ++i) {
        bool end_of_hop = i == num_hop_cores - 1;
        auto core_type = ring_matmul_helpers::CORE_TYPE::HOP_CORE;
        const auto& core = hop_cores_vec[i];
        const auto& core_noc = device->worker_core_from_logical_core(core);

        CoreCoord next_core = end_of_hop ? worker_cores_vec[num_cores - 1] : hop_cores_vec[i + 1];
        const auto& next_core_noc = device->worker_core_from_logical_core(next_core);
        uint32_t noc = ring_matmul_helpers::get_preferred_noc(core_noc, next_core_noc, device, use_dedicated_noc);

        std::vector<uint32_t> mm_in0_args = {
            (std::uint32_t)core_type,
            0,
            next_core_noc.x,
            next_core_noc.y,
            noc,
            (std::uint32_t)end_of_hop,
        };
        tt_metal::SetRuntimeArgs(program, mm_kernel_in0_id, core, mm_in0_args);

        std::vector<uint32_t> mm_kernel_in1_sender_writer_args;
        mm_kernel_in1_sender_writer_args.push_back((std::uint32_t)core_type);
        tt_metal::SetRuntimeArgs(program, mm_kernel_in1_sender_writer_id, core, mm_kernel_in1_sender_writer_args);

        std::vector<uint32_t> mm_kernel_args;
        mm_kernel_args.push_back((std::uint32_t)core_type);
        tt_metal::SetRuntimeArgs(program, mm_kernel, core, mm_kernel_args);
    }

    return RingMatmulProgramFactory::shared_variables_t{
        {mm_kernel_in0_id, mm_kernel_in1_sender_writer_id, mm_kernel}, worker_cores_vec, num_cores, use_hop_cores};
}

RingMatmulProgramFactory::cached_program_t RingMatmulProgramFactory::create(
    const RingMatmulParams& operation_attributes,
    const RingMatmulInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    TT_FATAL(operation_attributes.config.has_value(), "RingMatmul operation requires a config to be provided");

    tt_metal::Program program{};
    auto shared_vars = ring_matmul_create_program(
        program,
        tensor_args.input_tensor_a,
        tensor_args.input_tensors_b,
        tensor_args.bias_tensor,
        tensor_return_value,
        operation_attributes.config.value(),
        operation_attributes.compute_kernel_config,
        operation_attributes.fused_activation,
        operation_attributes.sub_device_id,
        operation_attributes.untilize_out,
        operation_attributes.global_cb);

    return {std::move(program), std::move(shared_vars)};
}

void RingMatmulProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const RingMatmulParams& operation_attributes,
    const RingMatmulInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    // Update tensor addresses and buffer pointers
    auto* src_buffer_a = tensor_args.input_tensor_a.buffer();
    auto* src_buffer_b = tensor_args.input_tensors_b[0].buffer();
    std::optional<tt::tt_metal::Buffer*> bias_buffer = std::nullopt;
    if (tensor_args.bias_tensor.has_value()) {
        bias_buffer = tensor_args.bias_tensor.value().buffer();
    }

    auto& in1_runtime_args = GetRuntimeArgs(cached_program.program, cached_program.shared_variables.kernel_ids[1]);
    for (const auto& core : cached_program.shared_variables.cores) {
        auto& writer_runtime_args = in1_runtime_args[core.x][core.y];
        writer_runtime_args[1] = src_buffer_b->address();
    }
}

}  // namespace ttnn::experimental::prim
