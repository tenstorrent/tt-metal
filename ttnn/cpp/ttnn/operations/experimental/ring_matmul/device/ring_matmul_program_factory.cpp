// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_matmul_device_operation.hpp"
#include <tt-metalium/math.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/compute_throttle_utils.hpp"
#include <algorithm>
#include <tuple>
#include <utility>
#include <vector>

using namespace tt;
using namespace tt::constants;
using ttnn::operations::unary::UnaryOpType;
using ttnn::operations::unary::UnaryWithParam;

namespace ttnn::operations::experimental::ring_matmul {

namespace {

enum class CORE_TYPE : uint32_t { IDLE_CORE = 0, WORKER_CORE = 1, HOP_CORE = 2 };

uint32_t get_preferred_noc(
    const ttnn::CoreCoord src,
    const ttnn::CoreCoord dst,
    const tt_metal::IDevice* device,
    const bool use_dedicated_noc = false) {
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

}  // namespace

RingMatmulDeviceOperation::ProgramFactory::cached_program_t RingMatmulDeviceOperation::ProgramFactory::create(
    const operation_attributes_t& attrs, const tensor_args_t& tensors, tensor_return_value_t& output_tensor) {
    auto* device = tensors.input_tensor.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), attrs.compute_kernel_config);

    if (attrs.config.has_value()) {
        fp32_dest_acc_en = attrs.config->fp32_dest_acc_en;
        packer_l1_acc = attrs.config->packer_l1_acc;
        dst_full_sync_en = attrs.config->dst_full_sync_en;
    }

    auto in0_buffer = tensors.input_tensor.buffer();
    auto in1_buffer = tensors.weight_tensor.buffer();
    auto out_buffer = output_tensor.buffer();

    auto in0_data_format = tt::tt_metal::datatype_to_dataformat_converter(tensors.input_tensor.dtype());
    auto in1_data_format = tt::tt_metal::datatype_to_dataformat_converter(tensors.weight_tensor.dtype());
    auto output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());

    auto in0_tile = tensors.input_tensor.tensor_spec().tile();
    auto in1_tile = tensors.weight_tensor.tensor_spec().tile();
    auto output_tile = output_tensor.tensor_spec().tile();

    auto in0_shape = tensors.input_tensor.padded_shape();
    auto in1_shape = tensors.weight_tensor.padded_shape();

    uint32_t M = in0_shape[-2] / in0_tile.get_tile_shape()[0];
    uint32_t K = in0_shape[-1] / in0_tile.get_tile_shape()[1];
    uint32_t N = in1_shape[-1] / in1_tile.get_tile_shape()[1];

    uint32_t in0_block_w = attrs.config.has_value() ? attrs.config->in0_block_w : 1;
    uint32_t out_subblock_h = attrs.config.has_value() ? attrs.config->out_subblock_h : 1;
    uint32_t out_subblock_w = attrs.config.has_value() ? attrs.config->out_subblock_w : 1;
    uint32_t per_core_M = attrs.config.has_value() ? attrs.config->per_core_M : M;
    uint32_t per_core_N = attrs.config.has_value() ? attrs.config->per_core_N : N;

    const bool in1_is_dram_interleaved = in1_buffer->is_dram() && !tensors.weight_tensor.is_sharded();
    const bool in1_is_dram_sharded =
        in1_buffer->is_dram() && tensors.weight_tensor.is_sharded() && !attrs.global_cb.has_value();

    constexpr bool row_major = true;
    CoreRangeSet all_worker_cores = tensors.input_tensor.shard_spec().value().grid;
    CoreRangeSet non_idle_cores = all_worker_cores.merge(attrs.hop_cores);
    CoreRangeSet all_cores = non_idle_cores;
    std::vector<CoreRange> non_idle_cores_vec;
    auto subdevice_cores = device->worker_cores(
        tt::tt_metal::HalProgrammableCoreType::TENSIX,
        attrs.sub_device_id.has_value() ? *attrs.sub_device_id : device->get_sub_device_ids().at(0));
    if (attrs.restricted_cores.has_value()) {
        subdevice_cores = subdevice_cores.subtract(attrs.restricted_cores.value());
    }
    for (auto& cr : subdevice_cores.ranges()) {
        auto intersection = non_idle_cores.intersection(cr);
        if (!intersection.empty()) {
            non_idle_cores_vec.push_back(intersection.bounding_box());
        }
    }
    all_cores = CoreRangeSet(non_idle_cores_vec);
    std::vector<CoreRange> ring_list = all_worker_cores.ranges();
    std::vector<CoreRange> hop_list = attrs.hop_cores.ranges();
    ring_list.insert(ring_list.end(), hop_list.begin(), hop_list.end());

    const uint32_t num_cores = all_worker_cores.num_cores();
    const uint32_t ring_size = num_cores;

    uint32_t num_hop_cores = attrs.hop_cores.num_cores();
    bool use_hop_cores = num_hop_cores > 0;

    const uint32_t Kt_pad = in0_buffer->shard_spec().shape()[1] / in0_tile.get_tile_shape()[1] * num_cores;
    in0_block_w = Kt_pad / num_cores;

    uint32_t num_blocks = Kt_pad / in0_block_w;
    bool packer_l1_acc_en = packer_l1_acc && num_blocks > 1;

    bool use_global_cb = attrs.global_cb.has_value();

    tt::DataFormat interm0_data_format = packer_l1_acc_en
                                             ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                                             : (fp32_dest_acc_en ? tt::DataFormat::Float32 : output_data_format);

    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    uint32_t output_single_tile_size = output_tile.get_tile_size(output_data_format);
    uint32_t interm0_single_tile_size = output_tile.get_tile_size(interm0_data_format);

    uint32_t in0_shard_width_in_tiles = in0_buffer->shard_spec().shape()[1] / in0_tile.get_tile_shape()[1];
    uint32_t in0_CB_tiles = per_core_M * in0_shard_width_in_tiles;
    uint32_t in0_CB_size = in0_CB_tiles * in0_single_tile_size;

    uint32_t in1_shard_height_in_tiles = 0;
    uint32_t in1_shard_width_in_tiles = 0;
    uint32_t in1_CB_tiles = 0;
    uint32_t in1_tensor_width_in_tiles = tensors.weight_tensor.padded_shape()[-1] / in1_tile.get_tile_shape()[1];

    if (in1_is_dram_sharded || in1_is_dram_interleaved) {
        in1_CB_tiles = 2 * in0_shard_width_in_tiles * per_core_N;
    } else {
        in1_shard_height_in_tiles = in1_buffer->shard_spec().shape()[0] / in1_tile.get_tile_shape()[0];
        in1_shard_width_in_tiles =
            in1_buffer->shard_spec().shape()[1] / in1_tile.get_tile_shape()[1] / attrs.num_global_cb_receivers;
        in1_CB_tiles = in1_shard_height_in_tiles * in1_shard_width_in_tiles;
    }
    uint32_t in1_CB_size = in1_CB_tiles * in1_single_tile_size;

    uint32_t per_core_N_size_bytes = per_core_N * in1_single_tile_size;
    uint32_t max_packet_size = 8192;
    uint32_t in1_block_page_size = per_core_N_size_bytes > max_packet_size ? max_packet_size : per_core_N_size_bytes;
    uint32_t in1_block_page_size_last =
        per_core_N_size_bytes > max_packet_size ? per_core_N_size_bytes % max_packet_size : per_core_N_size_bytes;
    uint32_t in1_block_width_num_pages = (per_core_N_size_bytes + in1_block_page_size - 1) / in1_block_page_size;
    uint32_t in1_shard_width_in_dram = 0;
    if (in1_is_dram_sharded) {
        in1_shard_width_in_dram = in1_buffer->shard_spec().shape()[1] / in1_tile.get_tile_shape()[1];
    }

    uint32_t in2_single_tile_size = in0_single_tile_size;
    uint32_t in2_CB_tiles = (ring_size - 1) * in0_CB_tiles;
    uint32_t in2_CB_size = in2_CB_tiles * in2_single_tile_size;

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

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    auto in0_signal_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);

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

    uint32_t base_cb_index = tt::CBIndex::c_0;
    uint32_t src0_cb_index = base_cb_index;
    tt_metal::CircularBufferConfig src0_cb_config =
        tt_metal::CircularBufferConfig(in0_CB_size, {{src0_cb_index, in0_data_format}})
            .set_page_size(src0_cb_index, in0_single_tile_size)
            .set_tile_dims(src0_cb_index, in0_tile)
            .set_globally_allocated_address(*in0_buffer);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, src0_cb_config);

    uint32_t src1_cb_index = base_cb_index + 1;
    tt::tt_metal::CBHandle cb_src1;
    uint32_t remote_cb_index = tt::CBIndex::c_31;
    if (use_global_cb) {
        uint32_t in1_block_size_bytes_gcb = in1_single_tile_size * in1_block_num_tiles;
        tt_metal::CircularBufferConfig remote_cb_config = tt_metal::CircularBufferConfig(
            (attrs.global_cb->size() / in1_block_size_bytes_gcb) * in1_block_size_bytes_gcb);
        remote_cb_config.remote_index(remote_cb_index)
            .set_page_size(in1_block_size_bytes_gcb)
            .set_data_format(in1_data_format);
        remote_cb_config.index(src1_cb_index).set_page_size(in1_single_tile_size).set_data_format(in1_data_format);
        cb_src1 = tt_metal::experimental::CreateCircularBuffer(program, all_cores, remote_cb_config, *attrs.global_cb);
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

    uint32_t src2_cb_index = base_cb_index + 2;
    tt_metal::CircularBufferConfig src2_cb_config =
        tt_metal::CircularBufferConfig(in2_CB_size, {{src2_cb_index, in0_data_format}})
            .set_page_size(src2_cb_index, in2_single_tile_size)
            .set_tile_dims(src2_cb_index, in0_tile);
    tt_metal::CreateCircularBuffer(program, all_cores, src2_cb_config);

    uint32_t sync_cb_index = base_cb_index + 3;
    uint32_t sync_cb_size_bytes = 16;
    tt_metal::CircularBufferConfig sync_cb_config =
        tt_metal::CircularBufferConfig(sync_cb_size_bytes, {{sync_cb_index, DataFormat::UInt16}})
            .set_page_size(sync_cb_index, sync_cb_size_bytes);
    tt_metal::CreateCircularBuffer(program, all_cores, sync_cb_config);

    uint32_t sync_cb2_index = base_cb_index + 4;
    uint32_t sync_cb2_size_bytes = 16;
    tt_metal::CircularBufferConfig sync_cb2_config =
        tt_metal::CircularBufferConfig(sync_cb2_size_bytes, {{sync_cb2_index, DataFormat::UInt16}})
            .set_page_size(sync_cb2_index, sync_cb2_size_bytes);
    tt_metal::CreateCircularBuffer(program, all_cores, sync_cb2_config);

    uint32_t output_cb_index = base_cb_index + 5;
    uint32_t interm0_cb_index = base_cb_index + 6;

    std::vector<tt::tt_metal::CBHandle> cb_outputs;
    uint32_t output_cb_idx = output_cb_index;
    uint32_t interm0_cb_idx = interm0_cb_index;

    if ((interm0_data_format != output_data_format) || (attrs.untilize_out && (in1_num_subblocks > 1))) {
        std::map<uint8_t, tt::DataFormat> interm0_cb_data_format_spec{{interm0_cb_idx, interm0_data_format}};
        tt_metal::CircularBufferConfig interm0_cb_config =
            tt_metal::CircularBufferConfig(interm0_CB_size, interm0_cb_data_format_spec)
                .set_page_size(interm0_cb_idx, interm0_single_tile_size)
                .set_tile_dims(interm0_cb_idx, output_tile);
        tt_metal::CreateCircularBuffer(program, all_cores, interm0_cb_config);

        std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{{output_cb_idx, output_data_format}};
        tt_metal::CircularBufferConfig output_cb_config =
            tt_metal::CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
                .set_page_size(output_cb_idx, output_single_tile_size)
                .set_tile_dims(output_cb_idx, output_tile)
                .set_globally_allocated_address(*out_buffer);
        auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);
        cb_outputs.push_back(cb_output);
    } else {
        std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
            {output_cb_idx, output_data_format}, {interm0_cb_idx, interm0_data_format}};
        tt_metal::CircularBufferConfig output_cb_config =
            tt_metal::CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
                .set_page_size(output_cb_idx, output_single_tile_size)
                .set_page_size(interm0_cb_idx, interm0_single_tile_size)
                .set_tile_dims(output_cb_idx, output_tile)
                .set_tile_dims(interm0_cb_idx, output_tile)
                .set_globally_allocated_address(*out_buffer);
        auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);
        cb_outputs.push_back(cb_output);
    }

    std::vector<uint32_t> in0_sender_compile_time_args = {
        (std::uint32_t)in0_shard_width_in_tiles,
        (std::uint32_t)per_core_M,
        (std::uint32_t)1,
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
        (std::uint32_t)1,
        (std::uint32_t)in1_block_page_size,
        (std::uint32_t)in1_block_page_size_last,
        (std::uint32_t)in1_block_width_num_pages,
        (std::uint32_t)in1_shard_width_in_dram,
        (std::uint32_t)src1_cb_index,
        (std::uint32_t)sync_cb_index,
        (std::uint32_t)sync_cb2_index,
        (std::uint32_t)remote_cb_index,
        (std::uint32_t)false,
    };
    tt::tt_metal::TensorAccessorArgs(*in1_buffer).append_to(in1_sender_writer_compile_time_args);

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
        1,
        out_block_tiles,
        attrs.untilize_out,
        in1_is_dram_interleaved,
        in1_is_dram_sharded,
        src0_cb_index,
        src1_cb_index,
        src2_cb_index,
        sync_cb_index,
        sync_cb2_index,
    };
    compute_kernel_args.push_back(compute_kernel_args.size() + 1);
    compute_kernel_args.push_back(output_cb_idx);
    compute_kernel_args.push_back(interm0_cb_idx);

    std::map<std::string, std::string> mm_in1_kernel_defines;
    std::map<std::string, std::string> mm_kernel_defines;

    if (use_global_cb) {
        mm_in1_kernel_defines["ENABLE_GLOBAL_CB"] = "1";
        mm_kernel_defines["ENABLE_GLOBAL_CB"] = "1";
    }

    if (attrs.fused_activation.has_value()) {
        if (attrs.fused_activation.value().op_type == UnaryOpType::RELU) {
            mm_kernel_defines["PACK_RELU"] = "1";
        } else {
            using ttnn::operations::unary::utils::get_defines;
            mm_kernel_defines.merge(get_defines(
                attrs.fused_activation.value().op_type, attrs.fused_activation.value().params, "ACTIVATION", "i"));
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

    tt_metal::NOC in0_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    tt_metal::NOC in1_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());

    bool use_dedicated_noc = true;
    tt_metal::NOC_MODE noc_mode =
        use_dedicated_noc ? tt_metal::NOC_MODE::DM_DEDICATED_NOC : tt_metal::NOC_MODE::DM_DYNAMIC_NOC;

    auto mm_kernel_in0_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_ring_all_gather.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = in0_noc,
            .noc_mode = noc_mode,
            .compile_args = in0_sender_compile_time_args});

    auto mm_kernel_in1_sender_writer_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_ring_all_gather.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = in1_noc,
            .noc_mode = noc_mode,
            .compile_args = in1_sender_writer_compile_time_args,
            .defines = mm_in1_kernel_defines});

    auto mm_kernel = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp",
        all_cores,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args,
            .defines = mm_kernel_defines});

    auto all_cores_vec = corerange_to_cores(all_cores, std::nullopt, row_major);
    auto worker_cores_vec = corerange_to_cores(all_worker_cores, std::nullopt, row_major);
    auto hop_cores_vec = corerange_to_cores(attrs.hop_cores, std::nullopt, row_major);

    for (uint32_t i = 0; i < all_cores_vec.size(); ++i) {
        auto core = all_cores_vec[i];

        auto all_worker_cores_iter = std::find(worker_cores_vec.begin(), worker_cores_vec.end(), core);
        auto hop_cores_iter = std::find(hop_cores_vec.begin(), hop_cores_vec.end(), core);
        bool core_is_in_all_worker_cores = all_worker_cores_iter != worker_cores_vec.end();
        bool core_is_in_hop_cores = hop_cores_iter != hop_cores_vec.end();
        if (!use_hop_cores) {
            core_is_in_hop_cores = false;
        }

        if (!core_is_in_all_worker_cores && !core_is_in_hop_cores) {
            auto core_type = CORE_TYPE::IDLE_CORE;
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

    std::map<uint32_t, uint32_t> worker_coord_y_to_dram_bank_first_col_mapping;
    std::map<uint32_t, uint32_t> worker_coord_y_to_dram_bank_second_col_mapping;
    if (in1_is_dram_sharded) {
        if (device->arch() == tt::ARCH::WORMHOLE_B0) {
            worker_coord_y_to_dram_bank_first_col_mapping[0] = 1;
            worker_coord_y_to_dram_bank_first_col_mapping[4] = 2;
            worker_coord_y_to_dram_bank_first_col_mapping[5] = 3;
            worker_coord_y_to_dram_bank_first_col_mapping[9] = 0;

            worker_coord_y_to_dram_bank_second_col_mapping[0] = 4;
            worker_coord_y_to_dram_bank_second_col_mapping[1] = 6;
            worker_coord_y_to_dram_bank_second_col_mapping[2] = 9;
            worker_coord_y_to_dram_bank_second_col_mapping[4] = 10;
            worker_coord_y_to_dram_bank_second_col_mapping[5] = 11;
            worker_coord_y_to_dram_bank_second_col_mapping[6] = 8;
            worker_coord_y_to_dram_bank_second_col_mapping[7] = 7;
            worker_coord_y_to_dram_bank_second_col_mapping[9] = 5;
        } else if (device->arch() == tt::ARCH::BLACKHOLE) {
            TT_THROW("ring_matmul currently not supporting blackhole when in1 is dram sharded");
        } else {
            TT_THROW("ring_matmul currently not supporting this device arch");
        }
    }

    uint32_t bank_id = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        bool send_to_hop_core = i == 0 && use_hop_cores;
        const auto& core = worker_cores_vec[i];
        const auto& core_noc = device->worker_core_from_logical_core(core);

        auto core_type = CORE_TYPE::WORKER_CORE;
        CoreCoord next_core;
        if (send_to_hop_core) {
            next_core = hop_cores_vec[0];
        } else {
            uint32_t next_i = i == 0 ? num_cores - 1 : i - 1;
            next_core = worker_cores_vec[next_i % num_cores];
        }
        const auto& next_core_noc = device->worker_core_from_logical_core(next_core);
        uint32_t noc = get_preferred_noc(core_noc, next_core_noc, device, use_dedicated_noc);

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

        std::vector<uint32_t> mm_in1_args = {(std::uint32_t)core_type, in1_buffer->address(), i};
        if (in1_is_dram_sharded) {
            if (core.x <= 3) {
                bank_id = worker_coord_y_to_dram_bank_first_col_mapping[core.y];
            } else {
                bank_id = worker_coord_y_to_dram_bank_second_col_mapping[core.y];
            }
            uint32_t dram_read_offset = 0;
            if (core.x % 2 == 0) {
                dram_read_offset = 1;
            }
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

        std::vector<uint32_t> mm_kernel_compute_args = {(std::uint32_t)core_type, i};
        mm_kernel_compute_args.insert(
            mm_kernel_compute_args.end(),
            unpadded_in0_shard_widths_in_tiles.begin(),
            unpadded_in0_shard_widths_in_tiles.end());
        tt_metal::SetRuntimeArgs(program, mm_kernel, core, mm_kernel_compute_args);
    }

    for (uint32_t i = 0; i < num_hop_cores; ++i) {
        bool end_of_hop = i == num_hop_cores - 1;

        auto core_type = CORE_TYPE::HOP_CORE;
        const auto& core = hop_cores_vec[i];
        const auto& core_noc = device->worker_core_from_logical_core(core);

        CoreCoord next_core = end_of_hop ? worker_cores_vec[num_cores - 1] : hop_cores_vec[i + 1];
        const auto& next_core_noc = device->worker_core_from_logical_core(next_core);
        uint32_t noc = get_preferred_noc(core_noc, next_core_noc, device, use_dedicated_noc);

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

    std::vector<tt::tt_metal::CBHandle> shared_cbs = {cb_src0, cb_src1};
    shared_cbs.insert(shared_cbs.end(), cb_outputs.begin(), cb_outputs.end());

    return {
        std::move(program),
        {.kernels = {mm_kernel_in1_sender_writer_id},
         .cbs = shared_cbs,
         .cores = all_cores_vec,
         .num_cores = num_cores}};
}

void RingMatmulDeviceOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& attrs,
    const tensor_args_t& tensors,
    tensor_return_value_t& output_tensor) {
    auto& program = cached_program.program;
    auto& shared_vars = cached_program.shared_variables;

    auto in1_addr = tensors.weight_tensor.buffer()->address();
    auto& in1_runtime_args = GetRuntimeArgs(program, shared_vars.kernels.at(0));

    for (uint32_t i = 0; i < shared_vars.num_cores; ++i) {
        CoreCoord core = shared_vars.cores.at(i);
        auto& args = in1_runtime_args[core.x][core.y];
        if (args.size() > 1) {
            args[1] = in1_addr;
        }
    }
}

}  // namespace ttnn::operations::experimental::ring_matmul
