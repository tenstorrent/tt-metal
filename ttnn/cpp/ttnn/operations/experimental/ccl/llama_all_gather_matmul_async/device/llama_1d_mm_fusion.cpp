// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_device_operation.hpp"
#include "ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_1d_mm_fusion.hpp"

#include <algorithm>
#include <utility>

#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operation.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/tensor/shape/shape.hpp"

using namespace tt;

using ttnn::operations::unary::UnaryOpType;
using ttnn::operations::unary::UnaryWithParam;

namespace ttnn::operations::llama_agmm_fusion_helpers {

enum class CORE_TYPE : uint32_t { IDLE_CORE = 0, WORKER_CORE = 1, HOP_CORE = 2 };

static ttnn::prim::matmul_mcast_1d_common_override_variables_t
process_agmm_fusion_program_and_create_override_variables(
    tt_metal::Program& program,
    const tt::tt_metal::Tensor& /*a*/,
    const std::vector<tt::tt_metal::Tensor>& b_tensors,
    tt_metal::IDevice* device,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode,
    bool packer_l1_acc,
    bool dst_full_sync_en,
    CoreCoord /*compute_with_storage_grid_size*/,
    ttnn::operations::compute_throttle_utils::ThrottleLevel throttle_level,
    uint32_t base_cb_index,
    uint32_t /*B*/,
    uint32_t /*M*/,
    uint32_t /*N*/,
    uint32_t /*K*/,
    bool /*bcast_batch*/,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    std::optional<UnaryWithParam> fused_activation,
    const CoreRangeSet& hop_cores,
    tt_metal::Buffer* in0_buffer,
    tt_metal::Buffer* in1_buffer,
    std::vector<tt_metal::Buffer*> out_buffers,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const tt::tt_metal::Tile& output_tile,
    tt::DataFormat in0_data_format,
    tt::DataFormat in1_data_format,
    tt::DataFormat output_data_format,
    bool untilize_out,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
    uint32_t num_global_cb_receivers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<CoreRangeSet> restricted_cores,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& fused_op_signaler) {
    const auto& b = b_tensors[0];
    const auto num_output_cb = out_buffers.size();
    const auto batch = b_tensors.size();
    const bool in1_is_dram_interleaved = in1_buffer->is_dram() && !b.is_sharded();
    const bool in1_is_dram_sharded =
        in1_buffer->is_dram() && b.is_sharded() && !global_cb.has_value();  // read from DRAM directly

    /* Core setup */
    constexpr bool row_major = true;
    CoreRangeSet all_worker_cores = b.shard_spec().value().grid;
    CoreRangeSet non_idle_cores = all_worker_cores.merge(hop_cores);
    CoreRangeSet all_cores = non_idle_cores;
    std::vector<CoreRange> non_idle_cores_vec;
    auto subdevice_cores = device->worker_cores(
        tt::tt_metal::HalProgrammableCoreType::TENSIX,
        sub_device_id.has_value() ? *sub_device_id : device->get_sub_device_ids().at(0));
    if (restricted_cores.has_value()) {
        subdevice_cores = subdevice_cores.subtract(restricted_cores.value());
    }
    for (const auto& cr : subdevice_cores.ranges()) {
        auto intersection = non_idle_cores.intersection(cr);
        if (!intersection.empty()) {
            non_idle_cores_vec.push_back(intersection.bounding_box());
        }
    }
    // all_cores = CoreRangeSet(non_idle_cores_vec);
    std::vector<CoreRange> ring_list = all_worker_cores.ranges();
    std::vector<CoreRange> hop_list = hop_cores.ranges();
    ring_list.insert(ring_list.end(), hop_list.begin(), hop_list.end());

    CoreRangeSet ring_cores = CoreRangeSet(ring_list);
    all_cores = ring_cores;
    const uint32_t num_cores = all_worker_cores.num_cores();
    const uint32_t ring_size =
        fused_op_signaler->ring_size;  // use ccl ring size instead of num_cores = local core ring size for fused op
    const uint32_t ring_index = fused_op_signaler->start_ring_index;

    uint32_t num_hop_cores = hop_cores.num_cores();
    bool use_hop_cores = num_hop_cores > 0;

    /* Inner dim - no padding needed for multicast approach */
    const uint32_t Kt_total = in0_buffer->shard_spec().shape()[1] / in0_tile.get_tile_shape()[1];
    constexpr uint32_t num_multicast_steps = 4;
    in0_block_w = Kt_total / num_multicast_steps;  // Each step sends 1/4 of K dimension

    uint32_t num_blocks = num_multicast_steps;  // Always 4 blocks now
    // Only enable packer l1 accumulation when there are spills, otherwise
    // unnecessary overhead for reconfigs are added
    bool packer_l1_acc_en = packer_l1_acc && num_blocks > 1;

    bool use_global_cb = global_cb.has_value();

    // if fp32 enabled then we pack fp32 in l1, if not, then we pack fp16 in l1
    tt::DataFormat interm0_data_format = packer_l1_acc_en
                                             ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                                             : (fp32_dest_acc_en ? tt::DataFormat::Float32 : output_data_format);

    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    uint32_t output_single_tile_size = output_tile.get_tile_size(output_data_format);
    uint32_t interm0_single_tile_size = output_tile.get_tile_size(interm0_data_format);

    /* in0 - each multicast step sends 1/4 of data to all cores */
    uint32_t multicast_chunk_width_in_tiles = Kt_total / num_multicast_steps;  // 1/4 of K
    uint32_t in0_CB_tiles =
        per_core_M * multicast_chunk_width_in_tiles * num_multicast_steps;  // Buffer for all 4 chunk
    uint32_t in0_CB_size = in0_CB_tiles * in0_single_tile_size;

    /* in1 */
    uint32_t in1_shard_height_in_tiles = 0;
    uint32_t in1_shard_width_in_tiles = 0;
    uint32_t in1_CB_tiles = 0;
    uint32_t in1_tensor_width_in_tiles = b.padded_shape()[-1] / in1_tile.get_tile_shape()[1];

    if (in1_is_dram_sharded || in1_is_dram_interleaved) {
        in1_CB_tiles = 2 * multicast_chunk_width_in_tiles * per_core_N;  // Double buffered
    } else {
        in1_shard_height_in_tiles = in1_buffer->shard_spec().shape()[0] / in1_tile.get_tile_shape()[0];
        in1_shard_width_in_tiles =
            in1_buffer->shard_spec().shape()[1] / in1_tile.get_tile_shape()[1] / num_global_cb_receivers;
        in1_CB_tiles = in1_shard_height_in_tiles * in1_shard_width_in_tiles;
    }
    uint32_t in1_CB_size = in1_CB_tiles * in1_single_tile_size;

    // get the max page size based on num tiles
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

    /* in2 - not needed for multicast approach since all cores receive same data */
    uint32_t in2_single_tile_size = in0_single_tile_size;
    uint32_t in2_CB_size = 0;

    /* out */
    uint32_t out_block_tiles = per_core_M * per_core_N;
    uint32_t out_CB_tiles = out_block_tiles;  // No double buffer
    uint32_t out_CB_size = out_CB_tiles * output_single_tile_size;
    uint32_t interm0_CB_size = out_CB_tiles * interm0_single_tile_size;

    // No need for unpadded widths since no padding and uniform 1/4 chunks
    // All multicast steps send exactly the same amount: multicast_chunk_width_in_tiles

    /* semaphores */
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

    /* Create circular buffers */
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

    uint32_t src2_cb_index = base_cb_index + 2;
    tt_metal::CircularBufferConfig src2_cb_config =
        tt_metal::CircularBufferConfig(in2_CB_size, {{src2_cb_index, in0_data_format}})
            .set_page_size(src2_cb_index, in2_single_tile_size)
            .set_tile_dims(src2_cb_index, in0_tile);
    // auto cb_src2 = tt_metal::CreateCircularBuffer(program, all_cores, src2_cb_config);

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

    uint32_t output_cb_index = base_cb_index + 5;  // output operands start at index 16
    uint32_t interm0_cb_index = base_cb_index + 6;
    tt_metal::CircularBufferConfig interm0_cb_config =
        tt_metal::CircularBufferConfig(0, {{interm0_cb_index, interm0_data_format}});
    tt_metal::CircularBufferConfig output_cb_config =
        tt_metal::CircularBufferConfig(0, {{output_cb_index, output_data_format}});
    std::vector<tt::tt_metal::CBHandle> cb_outputs;
    std::vector<tt::tt_metal::CBHandle> output_cb_indices;
    std::vector<tt::tt_metal::CBHandle> interm_cb_indices;

    if ((interm0_data_format != output_data_format) || (untilize_out && (in1_num_subblocks > 1))) {
        // interm0
        std::map<uint8_t, tt::DataFormat> interm0_cb_data_format_spec{
            {interm0_cb_index, interm0_data_format},
        };
        interm0_cb_config = tt_metal::CircularBufferConfig(interm0_CB_size, interm0_cb_data_format_spec)
                                .set_page_size(interm0_cb_index, interm0_single_tile_size)
                                .set_tile_dims(interm0_cb_index, output_tile);

        tt_metal::CreateCircularBuffer(program, all_cores, interm0_cb_config);

        for (uint32_t i = 0; i < out_buffers.size(); ++i) {
            const auto& out_buffer = out_buffers[i];
            output_cb_index += i * 2;  // 5, 7, 9...
            TT_FATAL(
                output_cb_index <= tt::CBIndex::c_31,
                "Output circular buffer index {} exceeds maximum value {}",
                output_cb_index,
                tt::CBIndex::c_31);
            // output
            std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
                {output_cb_index, output_data_format},
            };
            output_cb_config = tt_metal::CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
                                   .set_page_size(output_cb_index, output_single_tile_size)
                                   .set_tile_dims(output_cb_index, output_tile)
                                   .set_globally_allocated_address(*out_buffer);
            auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);
            cb_outputs.push_back(cb_output);
            output_cb_indices.push_back(output_cb_index);
            interm_cb_indices.push_back(interm0_cb_index);
        }
    } else {
        for (uint32_t i = 0; i < out_buffers.size(); ++i) {
            const auto& out_buffer = out_buffers[i];
            output_cb_index += i * 2;   // 5, 7, 9...
            interm0_cb_index += i * 2;  // 6, 8, 10...
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
            // share buffer
            std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
                {output_cb_index, output_data_format}, {interm0_cb_index, interm0_data_format}};
            output_cb_config = tt_metal::CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
                                   .set_page_size(output_cb_index, output_single_tile_size)
                                   .set_page_size(interm0_cb_index, interm0_single_tile_size)
                                   .set_tile_dims(output_cb_index, output_tile)
                                   .set_tile_dims(interm0_cb_index, output_tile)
                                   .set_globally_allocated_address(*out_buffer);
            auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);
            cb_outputs.push_back(cb_output);
            output_cb_indices.push_back(output_cb_index);
            interm_cb_indices.push_back(interm0_cb_index);
        }
    }

    /* Compile time args for multicast receiver */
    std::vector<uint32_t> in0_multicast_receiver_compile_time_args = {
        static_cast<std::uint32_t>(multicast_chunk_width_in_tiles),  // 1/4 of K per step
        static_cast<std::uint32_t>(per_core_M),                      // in0_shard_height_in_tiles
        static_cast<std::uint32_t>(batch),                           // batch
        static_cast<std::uint32_t>(num_multicast_steps),             // 4 steps instead of ring_size
        static_cast<std::uint32_t>(in0_signal_semaphore_id),
        static_cast<std::uint32_t>(src0_cb_index),
        static_cast<std::uint32_t>(src2_cb_index),  // Keep for compatibility, though not used
        static_cast<std::uint32_t>(fused_op_signaler->fused_op_receiver_signal_semaphores[0]),  // Step 0
        static_cast<std::uint32_t>(fused_op_signaler->fused_op_receiver_signal_semaphores[1]),  // Step 1
        static_cast<std::uint32_t>(fused_op_signaler->fused_op_receiver_signal_semaphores[2]),  // Step 2
        static_cast<std::uint32_t>(fused_op_signaler->fused_op_receiver_signal_semaphores[3]),  // Step 3
        static_cast<std::uint32_t>(ring_index),                                                 // first chunk index
        static_cast<std::uint32_t>((ring_index - 1 + ring_size) % ring_size),                   // second chunk index
        static_cast<std::uint32_t>((ring_index - 2 + ring_size) % ring_size),                   // third chunk index
        static_cast<std::uint32_t>((ring_index - 3 + ring_size) % ring_size),                   // fourth chunk index
    };

    std::vector<uint32_t> in1_sender_writer_compile_time_args = {
        static_cast<std::uint32_t>(in1_is_dram_interleaved),    // in1_is_dram_interleaved
        static_cast<std::uint32_t>(in1_is_dram_sharded),        // in1_is_dram_sharded
        static_cast<std::uint32_t>(in1_block_height_in_tiles),  // in1_block_height_in_tiles
        static_cast<std::uint32_t>(per_core_N),                 // in1_block_width_in_tiles
        static_cast<std::uint32_t>(in1_tensor_width_in_tiles),  // in1_tensor_width_in_tiles
        static_cast<std::uint32_t>(num_blocks),                 // num_blocks
        static_cast<std::uint32_t>(batch),                      // batch
        static_cast<std::uint32_t>(in1_block_page_size),
        static_cast<std::uint32_t>(in1_block_page_size_last),
        static_cast<std::uint32_t>(in1_block_width_num_pages),
        static_cast<std::uint32_t>(in1_shard_width_in_dram),
        static_cast<std::uint32_t>(src1_cb_index),
        static_cast<std::uint32_t>(sync_cb_index),
        static_cast<std::uint32_t>(sync_cb2_index),
        static_cast<std::uint32_t>(remote_cb_index),
        static_cast<std::uint32_t>(0),  // no need to signaler for the fused op
    };
    tt::tt_metal::TensorAccessorArgs(*in1_buffer).append_to(in1_sender_writer_compile_time_args);

    /* compute kernel args */
    const uint32_t out_block_num_subblocks = out_block_tiles / out_subblock_num_tiles;
    TT_FATAL(
        out_block_num_subblocks == 1 || !untilize_out,
        "untilize_out is not supported for cases that out_block_num_subblocks > 1");
    std::vector<uint32_t> compute_kernel_args = {
        in0_block_w,             // in0_block_w (now 1/4 of K)
        in0_num_subblocks,       // in0_num_subblocks
        in0_block_num_tiles,     // in0_block_num_tiles
        in0_subblock_num_tiles,  // in0_subblock_num_tiles

        in1_num_subblocks,      // in1_num_subblocks
        in1_block_num_tiles,    // in1_block_num_tiles
        in1_block_size_bytes,   // in1_block_size_bytes
        in1_tensor_size_bytes,  // in1_tensor_size_bytes
        in1_per_core_w,         // in1_per_core_w

        num_multicast_steps,  // Always 4 steps instead of variable num_blocks

        out_subblock_h,          // out_subblock_h
        out_subblock_w,          // out_subblock_w
        out_subblock_num_tiles,  // out_subblock_num_tiles
        batch,                   // batch
        out_block_tiles,         // out_block_num_tiles

        untilize_out,             // untilize_out
        in1_is_dram_interleaved,  // in1_is_dram_interleaved
        in1_is_dram_sharded,      // in1_is_dram_sharded
        src0_cb_index,
        src1_cb_index,
        src2_cb_index,  // Keep for compatibility though not used
        sync_cb_index,
        sync_cb2_index,
    };
    compute_kernel_args.push_back(compute_kernel_args.size() + 1);  // The CT index of the output_cbs
    for (uint32_t i = 0; i < num_output_cb; ++i) {
        compute_kernel_args.push_back(output_cb_indices[i]);
    }
    for (uint32_t i = 0; i < num_output_cb; ++i) {
        compute_kernel_args.push_back(interm_cb_indices[i]);
    }

    /* Kernel defines */
    std::map<std::string, std::string> mm_in1_kernel_defines;
    std::map<std::string, std::string> mm_kernel_defines;

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
        device->arch(), num_cores, mm_kernel_defines, throttle_level);

    // in1 is the reader of weights/output writer, and we choose to make it use the optimized reader noc
    tt_metal::NOC in0_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    tt_metal::NOC in1_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());

    bool use_dedicated_noc = true;
    tt_metal::NOC_MODE noc_mode =
        use_dedicated_noc ? tt_metal::NOC_MODE::DM_DEDICATED_NOC : tt_metal::NOC_MODE::DM_DYNAMIC_NOC;

    // Init the signaler
    if (fused_op_signaler.has_value() && fused_op_signaler.value().fused_op_type ==
                                             ttnn::experimental::ccl::MatmulFusedOpSignalerType::LLAMA_REDUCE_SCATTER) {
        ttnn::experimental::ccl::MatmulFusedOpSignaler& signaler = fused_op_signaler.value();
        signaler.init_llama_rs_cores_mm(all_cores, program, device, 0);
    }
    /* Create the kernels */
    auto mm_kernel_in0_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/kernels/"
        "reader_bmm_tile_layout_in0_ring_all_gather.cpp",  // Keep same kernel name
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = in0_noc,
            .noc_mode = noc_mode,
            .compile_args = in0_multicast_receiver_compile_time_args});  // NEW ARGS
    // Each core needs to signal to all RS cores, need to get a count of how many cores are in all_cores
    auto mm_kernel_in1_sender_writer_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/kernels/"
        "reader_bmm_tile_layout_in1_ring_all_gather.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = in1_noc,
            .noc_mode = noc_mode,
            .compile_args = in1_sender_writer_compile_time_args,
            .defines = mm_in1_kernel_defines});

    auto mm_kernel = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/kernels/compute/"
        "bmm_large_block_zm_fused_bias_activation_gathered.cpp",
        all_cores,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args,
            .defines = mm_kernel_defines});

    // for all the cores in the rect grid, we send one rt arg to determine if they are worker core
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

        if (!core_is_in_all_worker_cores && !core_is_in_hop_cores) {  // not worker core and not hop core
            auto core_type = CORE_TYPE::IDLE_CORE;                    // idle core
            // in0
            std::vector<uint32_t> mm_kernel_in0_args;
            mm_kernel_in0_args.push_back(static_cast<std::uint32_t>(core_type));
            tt_metal::SetRuntimeArgs(program, mm_kernel_in0_id, core, mm_kernel_in0_args);

            // in1
            std::vector<uint32_t> mm_kernel_in1_sender_writer_args;
            mm_kernel_in1_sender_writer_args.push_back(static_cast<std::uint32_t>(core_type));
            if (fused_op_signaler.has_value() &&
                fused_op_signaler.value().fused_op_type ==
                    ttnn::experimental::ccl::MatmulFusedOpSignalerType::LLAMA_REDUCE_SCATTER) {
                ttnn::experimental::ccl::MatmulFusedOpSignaler& signaler = fused_op_signaler.value();
                signaler.push_llama_rs_rt_args_for_mm(mm_kernel_in1_sender_writer_args, core, in1_noc, device);
            }

            tt_metal::SetRuntimeArgs(program, mm_kernel_in1_sender_writer_id, core, mm_kernel_in1_sender_writer_args);

            // compute
            std::vector<uint32_t> mm_kernel_args;
            mm_kernel_args.push_back(static_cast<std::uint32_t>(core_type));
            tt_metal::SetRuntimeArgs(program, mm_kernel, core, mm_kernel_args);
        }
    }

    /* Runtime args */
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
            TT_THROW("ring gather MM currently not supporting blackhole when in1 is dram sharded");
        } else {
            TT_THROW("ring gather MM currently not supporting this device arch");
        }
    }

    uint32_t bank_id = 0;
    std::vector<uint32_t> bank_ids;
    for (uint32_t i = 0; i < num_cores; ++i) {  // runtime args for mm cores
        const auto& core = worker_cores_vec[i];
        /* in0 - multicast receiver setup (no ring topology needed) */
        auto core_type = CORE_TYPE::WORKER_CORE;  // worker core

        std::vector<uint32_t> mm_in0_args = {
            static_cast<std::uint32_t>(core_type),
            static_cast<std::uint32_t>(ring_index),           // Core index for multicast addressing
            static_cast<std::uint32_t>(num_multicast_steps),  // 4 steps
            // No next_core coordinates needed for multicast reception
            // Multicast sender coordinates would be determined elsewhere
        };

        // No need for unpadded widths array since no padding and uniform chunks
        // Add fused op semaphores directly
        tt_metal::SetRuntimeArgs(program, mm_kernel_in0_id, core, mm_in0_args);

        /* in1 */
        std::vector<uint32_t> mm_in1_args = {
            static_cast<std::uint32_t>(core_type),
            in1_buffer->address(),  // in1_tensor_addr
            i,                      // ring_idx
        };
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
            bank_ids.push_back(bank_id);
            uint32_t vc = 0;
            for (uint32_t j = 0; j < i; ++j) {
                auto core_prev = worker_cores_vec[j];
                if (core_prev.y == core.y) {
                    vc = (vc + 1) & 0x3;
                }
            }
            mm_in1_args.push_back(static_cast<std::uint32_t>(bank_id));
            mm_in1_args.push_back(static_cast<std::uint32_t>(vc));
            mm_in1_args.push_back(static_cast<std::uint32_t>(dram_read_offset));
        }
        if (fused_op_signaler.has_value() &&
            fused_op_signaler.value().fused_op_type ==
                ttnn::experimental::ccl::MatmulFusedOpSignalerType::LLAMA_REDUCE_SCATTER) {
            ttnn::experimental::ccl::MatmulFusedOpSignaler& signaler = fused_op_signaler.value();
            signaler.push_llama_rs_rt_args_for_mm(mm_in1_args, core, in1_noc, device);
        }
        tt_metal::SetRuntimeArgs(program, mm_kernel_in1_sender_writer_id, core, mm_in1_args);

        /* compute */
        std::vector<uint32_t> mm_kernel_compute_args = {
            static_cast<std::uint32_t>(core_type),
            ring_index,  // core_idx (not ring_idx anymore)
        };
        // No need for unpadded widths since all steps process uniform 1/4 chunks

        tt_metal::SetRuntimeArgs(program, mm_kernel, core, mm_kernel_compute_args);
    }

    // Runtime args for hop cores
    for (uint32_t i = 0; i < num_hop_cores; ++i) {
        auto core_type = CORE_TYPE::HOP_CORE;  // hop core
        const auto& core = hop_cores_vec[i];

        /* in0 - hop cores not needed for multicast, but keeping for compatibility */
        std::vector<uint32_t> mm_in0_args = {
            static_cast<std::uint32_t>(core_type),
            static_cast<std::uint32_t>(i),  // Core index
            // Hop cores may not be needed for multicast approach
        };
        tt_metal::SetRuntimeArgs(program, mm_kernel_in0_id, core, mm_in0_args);

        // in1
        std::vector<uint32_t> mm_kernel_in1_sender_writer_args;
        mm_kernel_in1_sender_writer_args.push_back(static_cast<std::uint32_t>(core_type));
        if (fused_op_signaler.has_value() &&
            fused_op_signaler.value().fused_op_type ==
                ttnn::experimental::ccl::MatmulFusedOpSignalerType::LLAMA_REDUCE_SCATTER) {
            ttnn::experimental::ccl::MatmulFusedOpSignaler& signaler = fused_op_signaler.value();
            signaler.push_llama_rs_rt_args_for_mm(mm_kernel_in1_sender_writer_args, core, in1_noc, device);
        }
        tt_metal::SetRuntimeArgs(program, mm_kernel_in1_sender_writer_id, core, mm_kernel_in1_sender_writer_args);

        // compute
        std::vector<uint32_t> mm_kernel_args;
        mm_kernel_args.push_back(static_cast<std::uint32_t>(core_type));
        tt_metal::SetRuntimeArgs(program, mm_kernel, core, mm_kernel_args);
    }
    std::vector<tt::tt_metal::CBHandle> shared_cbs = {cb_src0, cb_src1};
    shared_cbs.insert(shared_cbs.end(), cb_outputs.begin(), cb_outputs.end());

    return ttnn::prim::matmul_mcast_1d_common_override_variables_t{
        {mm_kernel_in1_sender_writer_id},
        shared_cbs,
        false,
        CoreCoord{0, 0},
        all_cores_vec,
        0,
        ttnn::prim::Matmul1DType::GATHER_IN0};
}  // end of process_agmm_fusion_program_and_create_override_variables
}  // namespace ttnn::operations::llama_agmm_fusion_helpers

namespace ttnn::operations::llama_matmul {

void override_agmm_fusion_program_parameters(
    const ttnn::prim::matmul_mcast_1d_common_override_variables_t& override_variables,
    const ttnn::prim::MatmulParams& operation,
    tt_metal::Program& program,
    const std::vector<tt::tt_metal::Tensor>& input_tensors,
    const std::vector<std::optional<const tt::tt_metal::Tensor>>& /*optional_input_tensors*/,
    const std::vector<tt::tt_metal::Tensor>& output_tensors) {
    const auto& global_cb = operation.global_cb;

    auto* src_buffer_a = input_tensors[0].buffer();
    auto* src_buffer_b = input_tensors[1].buffer();

    bool src0_sharded = input_tensors[0].is_sharded();
    bool src1_sharded = input_tensors[1].is_sharded();
    bool out_sharded = output_tensors[0].is_sharded();

    // Manually unroll sender core
    if (src0_sharded) {
        UpdateDynamicCircularBufferAddress(program, override_variables.cbs[0], *src_buffer_a);
    }
    if (src1_sharded) {
        if (!global_cb.has_value() && !src_buffer_b->is_dram()) {
            UpdateDynamicCircularBufferAddress(program, override_variables.cbs[1], *src_buffer_b);
        }
    }
    if (out_sharded) {
        for (uint32_t i = 0; i < override_variables.cbs.size() - 2; ++i) {
            // cbs 0 and 1 contain cb_src0 and cb_src1
            // the rest contains the actual output cbs
            const auto& cb_output = override_variables.cbs[i + 2];
            const auto& out_buffer = output_tensors[i].buffer();
            UpdateDynamicCircularBufferAddress(program, cb_output, *out_buffer);
        }
    }

    if (not src1_sharded) {
        auto& writer_runtime_args_by_core = GetRuntimeArgs(program, override_variables.kernels.at(0));
        for (const auto& core : override_variables.cores) {
            auto& writer_runtime_args = writer_runtime_args_by_core[core.x][core.y];

            /* in1 */
            writer_runtime_args[1] = src_buffer_b->address();
        }
    }
}

static ttnn::prim::matmul_mcast_1d_common_override_variables_t matmul_multi_core_agmm_fusion_(
    tt_metal::Program& program,
    const Tensor& a,
    const std::vector<Tensor>& b_tensors,
    const std::optional<const Tensor>& bias,
    const std::vector<Tensor>& output_tensors,
    bool bcast_batch,
    CoreCoord compute_with_storage_grid_size,
    DeviceComputeKernelConfig compute_kernel_config,
    ttnn::operations::compute_throttle_utils::ThrottleLevel throttle_level,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t /*out_block_h*/,
    uint32_t /*out_block_w*/,
    uint32_t per_core_M,
    uint32_t per_core_N,
    bool fuse_batch,
    const std::optional<UnaryWithParam>& fused_activation,
    bool mcast_in0,
    bool gather_in0,
    const CoreRangeSet& hop_cores,
    bool untilize_out,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& fused_op_signaler,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
    uint32_t num_global_cb_receivers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    uint32_t start_cb_index,
    std::optional<CoreRangeSet> restricted_cores) {
    // Validate that only GATHER_IN0 is supported
    TT_FATAL(!mcast_in0, "Only GATHER_IN0 is supported. MCAST_IN0 has been removed.");
    TT_FATAL(gather_in0, "Only GATHER_IN0 is supported. This function requires gather_in0=true.");

    const auto& b = b_tensors[0];
    const auto& output = output_tensors[0];

    TT_FATAL(output_tensors.size() == b_tensors.size(), "number of outputs must match number of inputs b");

    const auto& ashape = Shape{
        1,
        1,
        a.memory_config().nd_shard_spec()->shard_shape[-2],
        a.memory_config().nd_shard_spec()->shard_shape[-1]};  // one shard of aggregated tensor will be in0
    const auto& bshape = b.padded_shape();
    auto in0_tile = a.tensor_spec().tile();
    auto in1_tile = b.tensor_spec().tile();
    // cannot use the output tensor tile directly as that might be changed by user override
    auto in0_tile_shape = in0_tile.get_tile_shape();
    auto in1_tile_shape = in1_tile.get_tile_shape();
    auto output_tile = tt::tt_metal::Tile({in0_tile_shape[0], in1_tile_shape[1]});

    // CB dataformats
    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());          // in0
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(b.dtype());          // in1
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());  // output

    if (bias.has_value()) {
        const auto& c = bias.value();
        TT_FATAL(
            c.storage_type() == StorageType::DEVICE,
            "Bias tensor storage type must be DEVICE but got {}",
            c.storage_type());
        TT_FATAL(a.device() == c.device(), "Operands to matmul need to be on the same device!");
        TT_FATAL(c.buffer() != nullptr, "Operands to matmul need to be allocated in buffers on device!");
    }

    tt_metal::IDevice* device = a.device();

    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    tt_metal::Buffer* in0_buffer = a.buffer();
    tt_metal::Buffer* in1_buffer = b.buffer();
    TT_FATAL(
        in0_buffer->size() % in0_single_tile_size == 0,
        "Input buffer 0 size ({}) must be divisible by single tile size ({})",
        in0_buffer->size(),
        in0_single_tile_size);
    TT_FATAL(
        in1_buffer->size() % in1_single_tile_size == 0,
        "Input buffer 1 size ({}) must be divisible by single tile size ({})",
        in1_buffer->size(),
        in1_single_tile_size);

    TT_FATAL(
        ashape[-1] == bshape[-2],
        "Dimension K (A.shape[-1] and B.shape[-2]) must match for A and B in bmm_op");  // A.K == B.K
    TT_FATAL(
        ashape[-2] % in0_tile_shape[0] == 0,
        "Input A height ({}) must be divisible by tile height ({})",
        ashape[-2],
        in0_tile_shape[0]);
    TT_FATAL(
        ashape[-1] % in0_tile_shape[1] == 0,
        "Input A width ({}) must be divisible by tile width ({})",
        ashape[-1],
        in0_tile_shape[1]);
    TT_FATAL(
        bshape[-2] % in1_tile_shape[0] == 0,
        "Input B height ({}) must be divisible by tile height ({})",
        bshape[-2],
        in1_tile_shape[0]);
    TT_FATAL(
        bshape[-1] % in1_tile_shape[1] == 0,
        "Input B width ({}) must be divisible by tile width ({})",
        bshape[-1],
        in1_tile_shape[1]);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    ////////////////////////////////////////////////////////////////////////////
    //                      Matmul Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // NOTE: Pads matmul input dims to 512 x 512 multiples (ie. multiples of 16*32 x 16*32)
    // NOTE: Maximum number of tiles in output is 120 * 16^2 = 30,720 (eg. [1, 1, 5120, 6144])
    uint32_t B = get_batch_size(ashape);
    uint32_t Mt = ashape[-2] / in0_tile_shape[0];
    uint32_t Kt = ashape[-1] / in0_tile_shape[1];
    uint32_t Nt = bshape[-1] / in1_tile_shape[1];

    if (fuse_batch) {
        Mt = B * Mt;
        B = 1;
    }
    TT_FATAL(
        Kt % in0_block_w == 0,
        "K dimension in tiles ({}) must be divisible by input block width ({})",
        Kt,
        in0_block_w);

    // This should allocate a DRAM buffer on the device
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores = num_cores_x * num_cores_y;

    // Calculate number of blocks along x and y; tensor dims are padded up to 512
    uint32_t num_blocks_y = ((Mt - 1) / per_core_M) + 1;
    uint32_t num_blocks_x = ((Nt - 1) / per_core_N) + 1;
    uint32_t num_blocks_total = num_blocks_y * num_blocks_x;

    // TODO: Max used grid can actually exceed mcast receiver grid if in0 is sharded
    // TODO: Move these validates to op validate and properly check for this
    TT_FATAL(
        num_blocks_total <= num_cores,
        "Number of blocks exceeds number of cores: {} blocks > {} cores",
        num_blocks_total,
        num_cores);

    if (!gather_in0) {
        TT_FATAL(hop_cores.empty(), "Hop cores are not supported for any mode besides gather_in0.");
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////

    // Only GATHER_IN0 is supported now
    std::vector<tt_metal::Buffer*> out_buffers;
    out_buffers.reserve(output_tensors.size());
    for (const auto& output_tensor : output_tensors) {
        out_buffers.push_back(output_tensor.buffer());
    }
    return llama_agmm_fusion_helpers::process_agmm_fusion_program_and_create_override_variables(
        program,
        a,
        b_tensors,
        device,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode,
        packer_l1_acc,
        dst_full_sync_en,
        compute_with_storage_grid_size,
        throttle_level,
        start_cb_index,
        B,
        Mt,
        Nt,
        Kt,
        bcast_batch,
        in0_block_w,
        out_subblock_h,
        out_subblock_w,
        per_core_M,
        per_core_N,
        fused_activation,
        hop_cores,
        in0_buffer,
        in1_buffer,
        out_buffers,
        in0_tile,
        in1_tile,
        output_tile,
        in0_data_format,
        in1_data_format,
        output_data_format,
        untilize_out,
        global_cb,
        num_global_cb_receivers,
        sub_device_id,
        std::move(restricted_cores),
        fused_op_signaler);
}

ttnn::prim::matmul_mcast_1d_common_override_variables_t matmul_multi_core_agmm_fusion_helper(
    tt_metal::Program& program,
    const Tensor& a,
    const std::vector<Tensor>& b_tensors,
    const std::optional<const Tensor>& bias,
    const std::vector<Tensor>& output_tensors,
    bool broadcast_batch,
    DeviceComputeKernelConfig compute_kernel_config,
    const matmul::MatmulProgramConfig& program_config,
    bool untilize_out,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& fused_op_signaler,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    uint32_t start_cb_index,
    std::optional<CoreRangeSet> restricted_cores) {
    matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig config =
        std::get<matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>(program_config);

    // Validate that only GATHER_IN0 is supported
    TT_FATAL(!config.mcast_in0, "Only GATHER_IN0 is supported. MCAST_IN0 has been removed.");
    TT_FATAL(config.gather_in0, "Only GATHER_IN0 is supported. This function requires gather_in0=true.");

    return matmul_multi_core_agmm_fusion_(
        program,
        a,
        b_tensors,
        bias,
        output_tensors,
        broadcast_batch,
        config.compute_with_storage_grid_size,
        compute_kernel_config,
        ttnn::get_throttle_level(compute_kernel_config),
        config.in0_block_w,
        config.out_subblock_h,
        config.out_subblock_w,
        config.out_block_h,
        config.out_block_w,
        config.per_core_M,
        config.per_core_N,
        config.fuse_batch,
        config.fused_activation,
        config.mcast_in0,
        config.gather_in0,
        config.hop_cores,
        untilize_out,
        fused_op_signaler,
        global_cb,
        config.num_global_cb_receivers,
        sub_device_id,
        start_cb_index,
        std::move(restricted_cores));
}

}  // namespace ttnn::operations::llama_matmul
