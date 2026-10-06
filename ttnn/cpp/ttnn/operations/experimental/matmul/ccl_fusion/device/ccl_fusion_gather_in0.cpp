// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ProgramDescriptor form of matmul's gather_in0 (ring) 1D builder, for llama_reduce_scatter_matmul, which builds the
// matmul next to its reduce-scatter kernels in one per-device program and signals the reduce-scatter through the
// llama MatmulFusedOpSignaler (positional runtime args, ttnn/operations/ccl/ccl_op_fusion.hpp). in1 may be fed by a
// GlobalCircularBuffer (the prefetcher), which a ProgramDescriptor can attach but Metal 2.0 cannot yet.
//
// Translated from matmul's legacy process_gather_in0_program_and_create_override_variables (still used by matmul's
// own MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory): same CBs (the GCB-backed remote CB included), kernels,
// compile-time and runtime args; tensor-backed CBs carry their tensors and the in1 address is a buffer binding, so a
// WorkloadDescriptor op built from it needs no matmul-specific cache-hit refresh. Binds ccl_fusion copies of the three
// gather kernels. Delete once llama_reduce_scatter_matmul is on Metal 2.0.

#include "ttnn/operations/experimental/matmul/ccl_fusion/device/ccl_fusion_gather_in0.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"
#include <algorithm>
#include <map>
#include <utility>
#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/experimental/prefetcher_pipe.hpp>
#include "ttnn/prefetcher_pipe.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/operations/matmul/shared_with_host/activation_type.hpp"

using namespace tt;
using tt::tt_metal::KernelBuildOptLevel;
using tt::tt_metal::MeshTensor;
using ttnn::operations::unary::UnaryOpType;
using ttnn::operations::unary::UnaryWithParam;

namespace ttnn::prim::ccl_fusion {

namespace reuse_mcast_1d_optimized_helpers {

uint32_t get_preferred_noc(
    const ttnn::CoreCoord src,
    const ttnn::CoreCoord dst,
    const tt_metal::distributed::MeshDevice& device,
    const bool use_dedicated_noc = false) {
    /*
        NOC0: Preferred +x -> +y
        NOC1: Preferred -y -> -x
    */

    uint32_t src_x = src.x, src_y = src.y;
    uint32_t dst_x = dst.x, dst_y = dst.y;

    uint32_t MAX_X = device.grid_size().x;
    uint32_t MAX_Y = device.grid_size().y;

    // Get the wrapped distances
    uint32_t dist_right = src_x <= dst_x ? dst_x - src_x : MAX_X - src_x + dst_x;
    uint32_t dist_left = src_x < dst_x ? src_x + MAX_X - dst_x : src_x - dst_x;

    uint32_t dist_bottom = src_y <= dst_y ? dst_y - src_y : MAX_Y - src_y + dst_y;
    uint32_t dist_top = src_y < dst_y ? src_y + MAX_Y - dst_y : src_y - dst_y;

    uint32_t dist_noc_0 = dist_right + dist_bottom;
    uint32_t dist_noc_1 = dist_top + dist_left;

    uint32_t noc = dist_noc_0 < dist_noc_1 ? 0 : 1;

    // Debug print if needed
    // std::cout << "src: (" << src_x << ", " << src_y << "), dst: (" << dst_x << ", " << dst_y << "), noc: " << noc <<
    // std::endl;

    return use_dedicated_noc ? 1 : noc;
}

enum class CORE_TYPE : uint32_t { IDLE_CORE = 0, WORKER_CORE = 1, HOP_CORE = 2 };

void create_program_gather_in0_descriptor(
    tt::tt_metal::ProgramDescriptor& desc,
    const ttnn::Tensor& a,
    const std::vector<ttnn::Tensor>& b_tensors,
    const tt_metal::distributed::MeshDevice& device,
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
    uint32_t K,
    bool /*bcast_batch*/,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    std::optional<UnaryWithParam> fused_activation,
    const CoreRangeSet& hop_cores,
    const MeshTensor& in0_tensor,
    const MeshTensor& in1_tensor,
    std::vector<std::reference_wrapper<const tt::tt_metal::MeshTensor>> out_buffers,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const tt::tt_metal::Tile& output_tile,
    tt::DataFormat in0_data_format,
    tt::DataFormat in1_data_format,
    tt::DataFormat output_data_format,
    bool untilize_out,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
    uint32_t num_global_cb_receivers,
    bool stream_in1,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<CoreRangeSet> restricted_cores,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& fused_op_signaler) {
    const auto& b = b_tensors[0];
    const auto num_output_cb = out_buffers.size();
    const auto batch = b_tensors.size();
    const bool in1_is_dram_interleaved = in1_tensor.memory_config().is_dram() && !b.is_sharded();
    const bool in1_is_dram_sharded =
        in1_tensor.memory_config().is_dram() && b.is_sharded() && !global_cb.has_value();  // read from DRAM directly

    /* Core setup */
    constexpr bool row_major = true;
    CoreRangeSet all_worker_cores = a.shard_spec().value().grid;
    CoreRangeSet non_idle_cores = all_worker_cores.merge(hop_cores);
    CoreRangeSet all_cores = non_idle_cores;
    std::vector<CoreRange> non_idle_cores_vec;
    non_idle_cores_vec.reserve(non_idle_cores.ranges().size());
    auto subdevice_cores = device.worker_cores(
        tt::tt_metal::HalProgrammableCoreType::TENSIX,
        sub_device_id.has_value() ? *sub_device_id : device.get_sub_device_ids().at(0));
    if (restricted_cores.has_value()) {
        subdevice_cores = subdevice_cores.subtract(restricted_cores.value());
    }
    for (const auto& cr : subdevice_cores.ranges()) {
        auto intersection = non_idle_cores.intersection(cr);
        if (intersection.empty()) {
            continue;
        }
        bool is_rectangular = cr.end_coord.x > cr.start_coord.x && cr.end_coord.y > cr.start_coord.y;
        if (is_rectangular) {
            non_idle_cores_vec.push_back(intersection.bounding_box());
        } else {
            for (const auto& ir : intersection.ranges()) {
                non_idle_cores_vec.push_back(ir);
            }
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

    /* Inner dim padding */
    const uint32_t Kt_pad = in0_tensor.shard_spec()->shape[1] / in0_tile.get_width() * num_cores;
    in0_block_w = Kt_pad / num_cores;

    uint32_t num_blocks = Kt_pad / in0_block_w;
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

    /* in0 */
    uint32_t in0_shard_width_in_tiles = in0_tensor.shard_spec()->shape[1] / in0_tile.get_width();
    uint32_t in0_CB_tiles = per_core_M * in0_shard_width_in_tiles;
    uint32_t in0_CB_size = in0_CB_tiles * in0_single_tile_size;

    /* in1 */
    uint32_t in1_shard_height_in_tiles = 0;
    uint32_t in1_shard_width_in_tiles = 0;
    uint32_t in1_CB_tiles = 0;

    const auto& bshape = operations::matmul::utilities::get_matmul_tensor_padded_shape(b, /*transpose=*/false);
    uint32_t in1_tensor_width_in_tiles = bshape[-1] / in1_tile.get_width();

    if (in1_is_dram_sharded || in1_is_dram_interleaved) {
        in1_CB_tiles = 2 * in0_shard_width_in_tiles * per_core_N;  // Double buffered
    } else if (use_global_cb) {
        // in1 is fed via the remote GCB CB (created from global_cb->size() below), so this
        // local in1_CB_size is dead. Skip reading the weight's legacy shard_spec — a
        // receiver-contiguous weight is allocated with an NdShardSpec (no legacy shard_spec),
        // and the per-receiver block geometry is fully described by per_core_N / the GCB page.
        in1_shard_height_in_tiles = in0_block_w;
        in1_shard_width_in_tiles = per_core_N;
        in1_CB_tiles = in1_shard_height_in_tiles * in1_shard_width_in_tiles;
    } else {
        in1_shard_height_in_tiles = in1_tensor.shard_spec()->shape[0] / in1_tile.get_height();
        in1_shard_width_in_tiles = in1_tensor.shard_spec()->shape[1] / in1_tile.get_width() / num_global_cb_receivers;
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
        in1_shard_width_in_dram = in1_tensor.shard_spec()->shape[1] / in1_tile.get_width();
    }

    /* in2 */
    uint32_t in2_single_tile_size = in0_single_tile_size;
    uint32_t in2_CB_tiles = (ring_size - 1) * in0_CB_tiles;  // All shards except local
    uint32_t in2_CB_size = std::max(in2_CB_tiles, 1u) * in2_single_tile_size;

    /* out */
    uint32_t out_block_tiles = per_core_M * per_core_N;
    uint32_t out_CB_tiles = out_block_tiles;  // No double buffer
    uint32_t out_CB_size = out_CB_tiles * output_single_tile_size;
    uint32_t interm0_CB_size = out_CB_tiles * interm0_single_tile_size;

    uint32_t K_ = K;
    std::vector<uint32_t> unpadded_in0_shard_widths_in_tiles(num_cores, 0);
    for (uint32_t i = 0; i < num_cores && K_ > 0; ++i) {
        unpadded_in0_shard_widths_in_tiles[i] = std::min(K_, in0_shard_width_in_tiles);
        K_ -= unpadded_in0_shard_widths_in_tiles[i];
    }

    /* semaphores */
    // The fused CCL op has already placed semaphores in desc; take an id that is free on these cores.
    auto in0_signal_semaphore_id = ttnn::experimental::ccl::add_semaphore_descriptor(desc, all_cores, INVALID);

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
            .set_globally_allocated_address(in0_tensor);
    desc.cbs.push_back(ttnn::experimental::ccl::make_cb_descriptor(src0_cb_config, all_cores, &in0_tensor));

    uint32_t src1_cb_index = base_cb_index + 1;
    uint32_t remote_cb_index = tt::CBIndex::c_31;
    if (use_global_cb) {
        uint32_t in1_block_size_bytes = in1_single_tile_size * in1_block_num_tiles;
        tt_metal::CircularBufferConfig remote_cb_config =
            tt_metal::CircularBufferConfig((global_cb->size() / in1_block_size_bytes) * in1_block_size_bytes);
        remote_cb_config.remote_index(remote_cb_index)
            .set_page_size(in1_block_size_bytes)
            .set_data_format(in1_data_format);
        remote_cb_config.index(src1_cb_index).set_page_size(in1_single_tile_size).set_data_format(in1_data_format);
        desc.cbs.push_back(ttnn::experimental::ccl::make_cb_descriptor(
            remote_cb_config, all_cores, nullptr, nullptr, &global_cb.value()));
    } else {
        tt_metal::CircularBufferConfig src1_cb_config =
            tt_metal::CircularBufferConfig(in1_CB_size, {{src1_cb_index, in1_data_format}})
                .set_page_size(src1_cb_index, in1_single_tile_size)
                .set_tile_dims(src1_cb_index, in1_tile);
        if (!in1_is_dram_interleaved && !in1_is_dram_sharded) {
            src1_cb_config = src1_cb_config.set_globally_allocated_address(in1_tensor);
        }
        desc.cbs.push_back(ttnn::experimental::ccl::make_cb_descriptor(
            src1_cb_config, all_cores, (!in1_is_dram_interleaved && !in1_is_dram_sharded) ? &in1_tensor : nullptr));
    }

    uint32_t src2_cb_index = base_cb_index + 2;
    tt_metal::CircularBufferConfig src2_cb_config =
        tt_metal::CircularBufferConfig(in2_CB_size, {{src2_cb_index, in0_data_format}})
            .set_page_size(src2_cb_index, in2_single_tile_size)
            .set_tile_dims(src2_cb_index, in0_tile);
    desc.cbs.push_back(ttnn::experimental::ccl::make_cb_descriptor(src2_cb_config, all_cores));

    // Streaming pipelines one block ahead (lookahead 1), so up to this many blocks are in flight at
    // once: the GCB must hold this many blocks/receiver (checked below) and the reader's cumulative
    // remote_cb_wait_front peaks at this count. Bumping it would also require generalizing the
    // reader's one-behind ack loop. (The reader recycles GCB slots off the in1 CB's engine-accurate
    // consumer ack, so streaming needs no separate compute-done credit CB.)
    constexpr uint32_t kStreamingInFlightBlocks = 2;

    uint32_t sync_cb_index = base_cb_index + 3;
    // Compute->reader release signal: one 16 B page (one credit). Only the batched global-CB path
    // uses it (signals once per layer); streaming recycles GCB slots off the in1 CB's own consumer
    // ack and needs no credit here.
    constexpr uint32_t sync_cb_page_bytes = 16;
    uint32_t sync_cb_size_bytes = sync_cb_page_bytes;
    tt_metal::CircularBufferConfig sync_cb_config =
        tt_metal::CircularBufferConfig(sync_cb_size_bytes, {{sync_cb_index, DataFormat::UInt16}})
            .set_page_size(sync_cb_index, sync_cb_page_bytes);
    desc.cbs.push_back(ttnn::experimental::ccl::make_cb_descriptor(sync_cb_config, all_cores));

    uint32_t sync_cb2_index = base_cb_index + 4;
    uint32_t sync_cb2_size_bytes = 16;
    tt_metal::CircularBufferConfig sync_cb2_config =
        tt_metal::CircularBufferConfig(sync_cb2_size_bytes, {{sync_cb2_index, DataFormat::UInt16}})
            .set_page_size(sync_cb2_index, sync_cb2_size_bytes);
    desc.cbs.push_back(ttnn::experimental::ccl::make_cb_descriptor(sync_cb2_config, all_cores));

    uint32_t output_cb_index = base_cb_index + 5;  // output operands start at index 16
    uint32_t interm0_cb_index = base_cb_index + 6;
    tt_metal::CircularBufferConfig interm0_cb_config =
        tt_metal::CircularBufferConfig(0, {{interm0_cb_index, interm0_data_format}});
    tt_metal::CircularBufferConfig output_cb_config =
        tt_metal::CircularBufferConfig(0, {{output_cb_index, output_data_format}});
    std::vector<tt::tt_metal::CBHandle> cb_outputs;
    cb_outputs.reserve(out_buffers.size());
    std::vector<tt::tt_metal::CBHandle> output_cb_indices;
    output_cb_indices.reserve(out_buffers.size());
    std::vector<tt::tt_metal::CBHandle> interm_cb_indices;
    interm_cb_indices.reserve(out_buffers.size());

    if ((interm0_data_format != output_data_format) || (untilize_out && (in1_num_subblocks > 1))) {
        // interm0
        std::map<uint8_t, tt::DataFormat> interm0_cb_data_format_spec{
            {interm0_cb_index, interm0_data_format},
        };
        interm0_cb_config = tt_metal::CircularBufferConfig(interm0_CB_size, interm0_cb_data_format_spec)
                                .set_page_size(interm0_cb_index, interm0_single_tile_size)
                                .set_tile_dims(interm0_cb_index, output_tile);

        desc.cbs.push_back(ttnn::experimental::ccl::make_cb_descriptor(interm0_cb_config, all_cores));

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
                                   .set_globally_allocated_address(out_buffer);
            desc.cbs.push_back(
                ttnn::experimental::ccl::make_cb_descriptor(output_cb_config, all_cores, &out_buffer.get()));
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
                                   .set_globally_allocated_address(out_buffer);
            desc.cbs.push_back(
                ttnn::experimental::ccl::make_cb_descriptor(output_cb_config, all_cores, &out_buffer.get()));
            output_cb_indices.push_back(output_cb_index);
            interm_cb_indices.push_back(interm0_cb_index);
        }
    }

    /* Compile time args */
    std::vector<uint32_t> in0_sender_compile_time_args = {
        (std::uint32_t)in0_shard_width_in_tiles,
        (std::uint32_t)per_core_M,  // in0_shard_height_in_tiles
        (std::uint32_t)batch,       // batch
        (std::uint32_t)ring_size,   // ring_size
        (std::uint32_t)in0_signal_semaphore_id,
    };

    std::vector<uint32_t> in1_sender_writer_compile_time_args = {
        (std::uint32_t)in1_is_dram_interleaved,    // in1_is_dram_interleaved
        (std::uint32_t)in1_is_dram_sharded,        // in1_is_dram_sharded
        (std::uint32_t)in1_block_height_in_tiles,  // in1_block_height_in_tiles
        (std::uint32_t)per_core_N,                 // in1_block_width_in_tiles
        (std::uint32_t)in1_tensor_width_in_tiles,  // in1_tensor_width_in_tiles
        (std::uint32_t)num_blocks,                 // num_blocks
        (std::uint32_t)batch,                      // batch
        (std::uint32_t)in1_block_page_size,
        (std::uint32_t)in1_block_page_size_last,
        (std::uint32_t)in1_block_width_num_pages,
        (std::uint32_t)in1_shard_width_in_dram,
        (std::uint32_t)fused_op_signaler.has_value(),
    };
    tt::tt_metal::TensorAccessorArgs(in1_tensor).append_to(in1_sender_writer_compile_time_args);

    /* compute kernel args */
    const uint32_t out_block_num_subblocks = out_block_tiles / out_subblock_num_tiles;
    TT_FATAL(
        out_block_num_subblocks == 1 || !untilize_out,
        "untilize_out is not supported for cases that out_block_num_subblocks > 1");
    std::vector<uint32_t> compute_kernel_args = {
        in0_block_w,             // in0_block_w
        in0_num_subblocks,       // in0_num_subblocks
        in0_block_num_tiles,     // in0_block_num_tiles
        in0_subblock_num_tiles,  // in0_subblock_num_tiles

        in1_num_subblocks,      // in1_num_subblocks
        in1_block_num_tiles,    // in1_block_num_tiles
        in1_block_size_bytes,   // in1_block_size_bytes
        in1_tensor_size_bytes,  // in1_tensor_size_bytes
        in1_per_core_w,         // in1_per_core_w

        num_blocks,  // num_blocks

        out_subblock_h,          // out_subblock_h
        out_subblock_w,          // out_subblock_w
        out_subblock_num_tiles,  // out_subblock_num_tiles
        batch,                   // batch
        out_block_tiles,         // out_block_num_tiles

        untilize_out,             // untilize_out
        in1_is_dram_interleaved,  // in1_is_dram_interleaved
        in1_is_dram_sharded,      // in1_is_dram_sharded
    };
    std::unordered_map<std::string, uint32_t> compute_named_compile_args = {
        {"cb_in0", src0_cb_index},
        {"cb_in1", src1_cb_index},
        {"cb_in2", src2_cb_index},
        {"cb_sync", sync_cb_index},
        {"cb_sync2", sync_cb2_index},
    };
    for (uint32_t i = 0; i < num_output_cb; ++i) {
        compute_named_compile_args["cb_mm_out_" + std::to_string(i)] = output_cb_indices[i];
    }
    for (uint32_t i = 0; i < num_output_cb; ++i) {
        compute_named_compile_args["cb_mm_partials_" + std::to_string(i)] = interm_cb_indices[i];
    }

    /* Kernel defines */
    std::map<std::string, std::string> mm_in1_kernel_defines;
    std::map<std::string, std::string> mm_kernel_defines;

    if (use_global_cb) {
        mm_in1_kernel_defines["ENABLE_GLOBAL_CB"] = "1";
        mm_kernel_defines["ENABLE_GLOBAL_CB"] = "1";
        if (stream_in1) {
            // Consume in1 blocks in ring-rotated FIFO order as they arrive (matching a
            // streaming prefetcher) instead of waiting for the whole tensor. The reader pipelines
            // one block ahead, so two blocks are in flight at once; the GCB must hold at least 2
            // blocks/receiver or the reader deadlocks waiting for the prefetcher to deliver a
            // block whose slot only frees after a later ack.
            const uint32_t resident_blocks = global_cb->size() / (in1_block_num_tiles * in1_single_tile_size);
            TT_FATAL(
                resident_blocks >= kStreamingInFlightBlocks,
                "stream_in1 pipelines {} in1 blocks per receiver in flight, so the global circular buffer must "
                "hold at least that many blocks/receiver, but it holds only {}; increase the GCB window.",
                kStreamingInFlightBlocks,
                resident_blocks);
            mm_in1_kernel_defines["STREAMING_IN1"] = "1";
            mm_kernel_defines["STREAMING_IN1"] = "1";
        }
    } else {
        TT_FATAL(!stream_in1, "stream_in1 requires a DRAM-sender global circular buffer (use_global_cb)");
    }

    if (fused_activation.has_value()) {
        const auto& activation = fused_activation.value();
        const auto& op_type = activation.op_type;
        if (op_type == UnaryOpType::RELU) {
            mm_kernel_defines["PACK_RELU"] = "1";
        } else {
            mm_kernel_defines["SFPU_ACTIVATION"] = "1";
            using ttnn::operations::matmul::utilities::get_activation_params;
            const auto params = get_activation_params(activation);
            compute_named_compile_args["activation_type"] = static_cast<uint32_t>(params.type);
            compute_named_compile_args["activation_param0"] = params.param0;
            compute_named_compile_args["activation_param1"] = params.param1;
            compute_named_compile_args["activation_param2"] = params.param2;
        }
    }
    if (packer_l1_acc_en) {
        mm_kernel_defines["PACKER_L1_ACC"] = "1";
    }
    if (fp32_dest_acc_en) {
        mm_kernel_defines["FP32_DEST_ACC_EN"] = "1";
    }
    ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
        device.arch(), num_cores, mm_kernel_defines);
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device.arch(), num_cores, mm_kernel_defines, throttle_level);

    // in1 is the reader of weights/output writer, and we choose to make it use the optimized reader noc
    tt_metal::NOC in0_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device.arch());
    tt_metal::NOC in1_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device.arch());

    bool use_dedicated_noc = true;
    tt_metal::NOC_MODE noc_mode =
        use_dedicated_noc ? tt_metal::NOC_MODE::DM_DEDICATED_NOC : tt_metal::NOC_MODE::DM_DYNAMIC_NOC;

    // Init the signaler
    if (fused_op_signaler.has_value()) {
        ttnn::experimental::ccl::MatmulFusedOpSignaler& signaler = fused_op_signaler.value();
        signaler.init_llama_rs_cores_mm(all_cores, desc, &device, 0);
    }
    /* Create the kernels */
    const size_t mm_kernel_in0_id = ttnn::experimental::ccl::add_kernel_descriptor(
        desc,
        "ttnn/cpp/ttnn/operations/experimental/matmul/ccl_fusion/device/kernels/dataflow/"
        "reader_bmm_tile_layout_in0_ring_all_gather.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = in0_noc,
            .noc_mode = noc_mode,
            .compile_args = in0_sender_compile_time_args,
            .named_compile_args = {{"cb_in0", src0_cb_index}, {"cb_in2", src2_cb_index}}});
    // Each core needs to signal to all RS cores, need to get a count of how many cores are in all_cores
    const size_t mm_kernel_in1_sender_writer_id = ttnn::experimental::ccl::add_kernel_descriptor(
        desc,
        "ttnn/cpp/ttnn/operations/experimental/matmul/ccl_fusion/device/kernels/dataflow/"
        "reader_bmm_tile_layout_in1_ring_all_gather.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = in1_noc,
            .noc_mode = noc_mode,
            .compile_args = in1_sender_writer_compile_time_args,
            .defines = mm_in1_kernel_defines,
            .named_compile_args = {
                {"cb_in1", src1_cb_index},
                {"cb_sync", sync_cb_index},
                {"cb_sync2", sync_cb2_index},
                {"cb_remote", remote_cb_index}}});

    // fp32 K-partials (interm0_cb_index) are reloaded into DEST between blocks via
    // copy_block_matmul_partials; mark the CB UnpackToDestFp32 so the reload goes directly to
    // DEST instead of through SrcA (which would truncate the fp32 partial to TF32).
    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    if (fp32_dest_acc_en && interm0_data_format == tt::DataFormat::Float32) {
        unpack_to_dest_mode[interm0_cb_index] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }

    const size_t mm_kernel = ttnn::experimental::ccl::add_kernel_descriptor(
        desc,
        "ttnn/cpp/ttnn/operations/experimental/matmul/ccl_fusion/device/kernels/compute/"
        "bmm_large_block_zm_fused_bias_activation_gathered.cpp",
        all_cores,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args,
            .defines = mm_kernel_defines,
            .named_compile_args = compute_named_compile_args});

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
            mm_kernel_in0_args.push_back((std::uint32_t)core_type);
            desc.kernels[mm_kernel_in0_id].runtime_args.emplace_back(core, mm_kernel_in0_args);

            // in1
            std::vector<uint32_t> mm_kernel_in1_sender_writer_args;
            mm_kernel_in1_sender_writer_args.push_back((std::uint32_t)core_type);
            if (fused_op_signaler.has_value()) {
                ttnn::experimental::ccl::MatmulFusedOpSignaler& signaler = fused_op_signaler.value();
                signaler.push_llama_rs_rt_args_for_mm(mm_kernel_in1_sender_writer_args, core, in1_noc, &device);
            }

            desc.kernels[mm_kernel_in1_sender_writer_id].runtime_args.emplace_back(
                core, mm_kernel_in1_sender_writer_args);

            // compute
            std::vector<uint32_t> mm_kernel_args;
            mm_kernel_args.push_back((std::uint32_t)core_type);
            desc.kernels[mm_kernel].runtime_args.emplace_back(core, mm_kernel_args);
        }
    }

    /* Runtime args */
    // Mapping from worker core y-coordinate (and column group) to DRAM bank IDs.
    // The DRAM banks are split into two column groups (left and right halves of the chip).
    // On Wormhole: hardcoded mapping; banks 0-3 left column (x <= 3), banks 4-11 right column.
    // On Blackhole: dynamically derived from optimal DRAM bank API; banks split at x <= 6.
    std::map<uint32_t, uint32_t> worker_y_to_dram_bank_first_col;
    std::map<uint32_t, uint32_t> worker_y_to_dram_bank_second_col;
    uint32_t first_col_max_x = device.arch() == tt::ARCH::WORMHOLE_B0 ? 3 : 7;
    uint32_t num_receiver_cores_per_dram = 2;  // default to 2 for wormhole b0 and blackhole
    if (in1_is_dram_sharded) {
        num_receiver_cores_per_dram = ring_size / in1_tensor.shard_spec()->grid.num_cores();
        if (device.arch() == tt::ARCH::WORMHOLE_B0) {
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
            auto optimal_dram_workers = device.get_optimal_dram_bank_to_logical_worker_assignment(in1_noc);
            uint32_t num_banks = optimal_dram_workers.size();
            uint32_t banks_in_first_col = num_banks / 2;

            std::vector<std::pair<uint32_t, uint32_t>> first_col_anchors;   // (y, bank_id)
            std::vector<std::pair<uint32_t, uint32_t>> second_col_anchors;  // (y, bank_id)
            first_col_anchors.reserve(banks_in_first_col);
            second_col_anchors.reserve(num_banks - banks_in_first_col);

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
            auto compute_grid = device.compute_with_storage_grid_size();
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
    bank_ids.reserve(num_cores);
    for (uint32_t i = 0; i < num_cores; ++i) {
        bool send_to_hop_core = i == 0 && use_hop_cores;
        const auto& core = worker_cores_vec[i];
        const auto& core_noc = device.worker_core_from_logical_core(core);

        /* in0 */
        auto core_type = CORE_TYPE::WORKER_CORE;  // worker core
        CoreCoord next_core;
        if (send_to_hop_core) {
            next_core = hop_cores_vec[0];  // Send to first hop core
        } else {
            uint32_t next_i = i == 0 ? num_cores - 1 : i - 1;
            next_core = worker_cores_vec[next_i % num_cores];
        }
        const auto& next_core_noc = device.worker_core_from_logical_core(next_core);
        uint32_t noc = get_preferred_noc(core_noc, next_core_noc, device, use_dedicated_noc);

        std::vector<uint32_t> mm_in0_args = {
            (std::uint32_t)core_type,
            i,                // ring_index
            next_core_noc.x,  // next_core_noc_x
            next_core_noc.y,  // next_core_noc_y
            noc,
            (std::uint32_t)false,  // end_of_hop
        };

        mm_in0_args.insert(
            mm_in0_args.end(), unpadded_in0_shard_widths_in_tiles.begin(), unpadded_in0_shard_widths_in_tiles.end());
        desc.kernels[mm_kernel_in0_id].runtime_args.emplace_back(core, mm_in0_args);

        /* in1 */
        // Slot 1 (in1_tensor_addr) is a placeholder here; it becomes a buffer binding on in1 below.
        constexpr size_t in1_tensor_addr_slot = 1;
        std::vector<uint32_t> mm_in1_args = {
            (std::uint32_t)core_type,
            0,  // in1_tensor_addr
            i,  // ring_idx
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
            if (device.arch() == tt::ARCH::WORMHOLE_B0) {
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
        if (fused_op_signaler.has_value()) {
            ttnn::experimental::ccl::MatmulFusedOpSignaler& signaler = fused_op_signaler.value();
            signaler.push_llama_rs_rt_args_for_mm(mm_in1_args, core, in1_noc, &device);
        }
        {
            tt::tt_metal::KernelDescriptor::RTArgList in1_args;
            in1_args.reserve(mm_in1_args.size());
            for (size_t arg = 0; arg < mm_in1_args.size(); ++arg) {
                if (arg == in1_tensor_addr_slot) {
                    in1_args.push_back(in1_tensor);
                } else {
                    in1_args.push_back(mm_in1_args[arg]);
                }
            }
            desc.kernels[mm_kernel_in1_sender_writer_id].emplace_runtime_args(core, in1_args);
        }

        /* compute */
        std::vector<uint32_t> mm_kernel_compute_args = {
            (std::uint32_t)core_type,
            i,  // ring_idx
        };
        mm_kernel_compute_args.insert(
            mm_kernel_compute_args.end(),
            unpadded_in0_shard_widths_in_tiles.begin(),
            unpadded_in0_shard_widths_in_tiles.end());

        desc.kernels[mm_kernel].runtime_args.emplace_back(core, mm_kernel_compute_args);
    }

    // Runtime args for hop cores
    for (uint32_t i = 0; i < num_hop_cores; ++i) {
        bool end_of_hop = i == num_hop_cores - 1;

        auto core_type = CORE_TYPE::HOP_CORE;  // hop core
        const auto& core = hop_cores_vec[i];
        const auto& core_noc = device.worker_core_from_logical_core(core);

        /* in0 */
        CoreCoord next_core = end_of_hop ? worker_cores_vec[num_cores - 1] : hop_cores_vec[i + 1];
        const auto& next_core_noc = device.worker_core_from_logical_core(next_core);
        uint32_t noc = get_preferred_noc(core_noc, next_core_noc, device, use_dedicated_noc);

        std::vector<uint32_t> mm_in0_args = {
            (std::uint32_t)core_type,
            0,                // ring_index
            next_core_noc.x,  // next_core_noc_x
            next_core_noc.y,  // next_core_noc_y
            noc,
            (std::uint32_t)end_of_hop,  // end_of_hop
        };
        desc.kernels[mm_kernel_in0_id].runtime_args.emplace_back(core, mm_in0_args);

        // in1
        std::vector<uint32_t> mm_kernel_in1_sender_writer_args;
        mm_kernel_in1_sender_writer_args.push_back((std::uint32_t)core_type);
        if (fused_op_signaler.has_value()) {
            ttnn::experimental::ccl::MatmulFusedOpSignaler& signaler = fused_op_signaler.value();
            signaler.push_llama_rs_rt_args_for_mm(mm_kernel_in1_sender_writer_args, core, in1_noc, &device);
        }
        desc.kernels[mm_kernel_in1_sender_writer_id].runtime_args.emplace_back(core, mm_kernel_in1_sender_writer_args);

        // compute
        std::vector<uint32_t> mm_kernel_args;
        mm_kernel_args.push_back((std::uint32_t)core_type);
        desc.kernels[mm_kernel].runtime_args.emplace_back(core, mm_kernel_args);
    }
}

}  // namespace reuse_mcast_1d_optimized_helpers

static void matmul_multi_core_reuse_mcast_1d_gather_in0_descriptor_(
    tt::tt_metal::ProgramDescriptor& desc,
    const Tensor& a,
    const std::vector<Tensor>& b_tensors,
    const std::optional<const Tensor>& bias,
    const std::vector<Tensor>& output_tensors,
    bool bcast_batch,
    bool transpose_a,
    bool transpose_b,
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
    bool /*mcast_in0*/,
    bool gather_in0,
    const CoreRangeSet& hop_cores,
    bool untilize_out,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& fused_op_signaler,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
    uint32_t num_global_cb_receivers,
    bool stream_in1,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    uint32_t start_cb_index,
    std::optional<CoreRangeSet> restricted_cores) {
    const auto& b = b_tensors[0];
    const auto& output = output_tensors[0].mesh_tensor();

    TT_FATAL(output_tensors.size() == b_tensors.size(), "number of outputs must match number of inputs b");

    const auto& ashape = operations::matmul::utilities::get_matmul_tensor_padded_shape(a, transpose_a);
    const auto& bshape = operations::matmul::utilities::get_matmul_tensor_padded_shape(b, transpose_b);
    auto in0_tile = operations::matmul::utilities::get_matmul_tile(a, transpose_a);
    auto in1_tile = operations::matmul::utilities::get_matmul_tile(b, transpose_b);
    // cannot use the output tensor tile directly as that might be changed by user override
    auto output_tile = tt::tt_metal::Tile({in0_tile.get_height(), in1_tile.get_width()});

    // CB dataformats
    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());  // in0
    const MeshTensor& in1_tensor = b.mesh_tensor();
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(in1_tensor.dtype());  // in1
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());   // output

    ttsl::optional_reference<const MeshTensor> bias_mesh_tensor;
    [[maybe_unused]] tt::DataFormat bias_data_format = tt::DataFormat::Bfp8_b;  // bias; doesn't matter if bias=nullptr
    if (bias.has_value()) {
        const auto& c = bias.value();
        TT_FATAL(
            c.storage_type() == StorageType::DEVICE,
            "Bias tensor must be on device, got storage type: {}",
            c.storage_type());
        TT_FATAL(a.device() == c.device(), "Operands to matmul need to be on the same device!");

        bias_mesh_tensor = c.mesh_tensor();

        bias_data_format = tt_metal::datatype_to_dataformat_converter(c.dtype());
    }

    const tt_metal::distributed::MeshDevice& device = a.mesh_tensor().device();

    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    const MeshTensor& in0_tensor = a.mesh_tensor();
    TT_FATAL(
        in0_tensor.mesh_buffer().device_local_size() % in0_single_tile_size == 0,
        "Input A buffer size ({}) must be divisible by single tile size ({})",
        in0_tensor.mesh_buffer().device_local_size(),
        in0_single_tile_size);
    TT_FATAL(
        in1_tensor.mesh_buffer().device_local_size() % in1_single_tile_size == 0,
        "Input B buffer size ({}) must be divisible by single tile size ({})",
        in1_tensor.mesh_buffer().device_local_size(),
        in1_single_tile_size);

    TT_FATAL(
        ashape[-1] == bshape[-2],
        "Dimension K (A.shape[-1] and B.shape[-2]) must match for A and B in bmm_op");  // A.K == B.K
    TT_FATAL(
        ashape[-2] % in0_tile.get_height() == 0,
        "A.shape[-2] ({}) must be divisible by tile height ({})",
        ashape[-2],
        in0_tile.get_height());
    TT_FATAL(
        ashape[-1] % in0_tile.get_width() == 0,
        "A.shape[-1] ({}) must be divisible by tile width ({})",
        ashape[-1],
        in0_tile.get_width());
    TT_FATAL(
        bshape[-2] % in1_tile.get_height() == 0,
        "B.shape[-2] ({}) must be divisible by tile height ({})",
        bshape[-2],
        in1_tile.get_height());
    TT_FATAL(
        bshape[-1] % in1_tile.get_width() == 0,
        "B.shape[-1] ({}) must be divisible by tile width ({})",
        bshape[-1],
        in1_tile.get_width());

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device.arch(), compute_kernel_config);

    ////////////////////////////////////////////////////////////////////////////
    //                      Matmul Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // NOTE: Pads matmul input dims to 512 x 512 multiples (ie. multiples of 16*32 x 16*32)
    // NOTE: Maximum number of tiles in output is 120 * 16^2 = 30,720 (eg. [1, 1, 5120, 6144])
    const auto in0_B = fuse_batch ? 1 : get_batch_size(ashape);
    [[maybe_unused]] const auto in1_B = fuse_batch ? 1 : get_batch_size(bshape);
    const auto Mt = operations::matmul::utilities::get_M_dim(ashape, in0_tile, fuse_batch);
    const auto Kt = operations::matmul::utilities::get_K_dim(ashape, in0_tile);
    const auto Nt = operations::matmul::utilities::get_N_dim(bshape, in1_tile);

    TT_FATAL(Kt % in0_block_w == 0, "Kt ({}) must be divisible by in0_block_w ({})", Kt, in0_block_w);

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
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////

    TT_FATAL(
        gather_in0,
        "CCL-fused gather_in0 builder called with a non-gather_in0 1D program config; use the mcast_in0 builder");
    {
        TT_FATAL(
            !transpose_a,
            "Transpose A is ({}) not supported for gather_in0, please use a different program configuration",
            transpose_a);
        TT_FATAL(
            !transpose_b,
            "Transpose B is ({}) not supported for gather_in0, please use a different program configuration",
            transpose_b);
        std::vector<std::reference_wrapper<const tt::tt_metal::MeshTensor>> out_buffers;
        out_buffers.reserve(output_tensors.size());
        for (const auto& output_tensor : output_tensors) {
            out_buffers.push_back(output_tensor.mesh_tensor());
        }
        reuse_mcast_1d_optimized_helpers::create_program_gather_in0_descriptor(
            desc,
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
            in0_B,
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
            in0_tensor,
            in1_tensor,
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
            stream_in1,
            sub_device_id,
            std::move(restricted_cores),
            fused_op_signaler);
    }
}

void matmul_multi_core_reuse_mcast_1d_gather_in0_helper(
    tt::tt_metal::ProgramDescriptor& desc,
    const Tensor& a,
    const std::vector<Tensor>& b_tensors,
    const std::optional<const Tensor>& bias,
    const std::vector<Tensor>& output_tensors,
    bool broadcast_batch,
    DeviceComputeKernelConfig compute_kernel_config,
    const operations::matmul::MatmulProgramConfig& program_config,
    bool untilize_out,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& fused_op_signaler,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    uint32_t start_cb_index,
    std::optional<CoreRangeSet> restricted_cores) {
    auto config = std::get<operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>(program_config);

    if (!config.allowed_worker_cores.has_value()) {
        log_warning(
            tt::LogOp,
            "matmul_multi_core_reuse_mcast_1d_gather_in0_helper: program_config.allowed_worker_cores not populated; "
            "auto-populating from compute_with_storage_grid_size. Callers that bypass ttnn::prim::matmul() (e.g. "
            "CCL fused ops) should invoke ttnn::operations::matmul::normalize_program_config() on the program "
            "config first. This will become a hard error in a future release.");
        config.allowed_worker_cores = CoreRangeSet(CoreRange(
            CoreCoord(0, 0),
            CoreCoord(config.compute_with_storage_grid_size.x - 1, config.compute_with_storage_grid_size.y - 1)));
    }
    auto resolved_grid = config.allowed_worker_cores.value().bounding_box().grid_size();

    matmul_multi_core_reuse_mcast_1d_gather_in0_descriptor_(
        desc,
        a,
        b_tensors,
        bias,
        output_tensors,
        broadcast_batch,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        resolved_grid,
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
        config.stream_in1,
        sub_device_id,
        start_cb_index,
        std::move(restricted_cores));
}

}  // namespace ttnn::prim::ccl_fusion
