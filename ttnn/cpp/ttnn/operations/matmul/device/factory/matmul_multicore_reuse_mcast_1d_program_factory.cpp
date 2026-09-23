// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"
#include <algorithm>
#include <utility>

#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/prefetcher_pipe.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/operations/matmul/shared_with_host/activation_type.hpp"

using namespace tt;

using ttnn::operations::unary::UnaryOpType;
using ttnn::operations::unary::UnaryWithParam;

using tt::tt_metal::MeshTensor;

using tt::tt_metal::KernelBuildOptLevel;
using tt::tt_metal::experimental::AddRuntimeArgsForNode;
using tt::tt_metal::experimental::DataflowBufferSpec;
using tt::tt_metal::experimental::DataMovementGen1Config;
using tt::tt_metal::experimental::DFBBinding;
using tt::tt_metal::experimental::DFBEndpointType;
using tt::tt_metal::experimental::DFBSpecName;
using tt::tt_metal::experimental::double_buffer_dest;
using tt::tt_metal::experimental::Group;
using tt::tt_metal::experimental::KernelRunArgs;
using tt::tt_metal::experimental::KernelSpec;
using tt::tt_metal::experimental::KernelSpecName;
using tt::tt_metal::experimental::PrefetcherPipeArgument;
using tt::tt_metal::experimental::PrefetcherPipeParameter;
using tt::tt_metal::experimental::PrefetcherPipeParamName;
using tt::tt_metal::experimental::ProgramRunArgs;
using tt::tt_metal::experimental::ProgramSpec;
using tt::tt_metal::experimental::SemaphoreBinding;
using tt::tt_metal::experimental::SemaphoreSpec;
using tt::tt_metal::experimental::SemaphoreSpecName;
using tt::tt_metal::experimental::TensorBinding;
using tt::tt_metal::experimental::TensorParameter;
using tt::tt_metal::experimental::TensorParamName;
using tt::tt_metal::experimental::unpack_modes;
using tt::tt_metal::experimental::WorkUnitSpec;

namespace ttnn::prim {

namespace reuse_mcast_1d_optimized_helpers {

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

MatmulMultiCoreReuseMcast1DProgramFactory::shared_variables_t process_mcast_in0_program_and_create_override_variables(
    tt_metal::Program& program,
    const ttnn::Tensor& a,
    tt_metal::IDevice* device,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode,
    bool packer_l1_acc,
    CoreCoord compute_with_storage_grid_size,
    ttnn::operations::compute_throttle_utils::ThrottleLevel throttle_level,
    uint32_t B,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    bool bcast_batch,
    bool transpose_a,
    bool transpose_b,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t out_block_h,
    uint32_t out_block_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    std::optional<UnaryWithParam> fused_activation,
    const MeshTensor& in0_tensor,
    const MeshTensor& in1_tensor,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
    ttsl::optional_reference<const MeshTensor> bias_tensor,
    const MeshTensor& out_tensor,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const tt::tt_metal::Tile& bias_tile,
    const tt::tt_metal::Tile& output_tile,
    tt::DataFormat in0_data_format,
    tt::DataFormat in1_data_format,
    tt::DataFormat bias_data_format,
    tt::DataFormat output_data_format,
    bool in0_is_sharded,
    bool in1_is_sharded,
    bool bias_is_sharded,
    bool output_is_sharded,
    bool untilize_out,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& fused_op_signaler,
    bool row_broadcast_bias = true,
    CoreCoord sub_device_start_core = {0, 0}) {
    using tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids;

    const bool use_global_cb = global_cb.has_value();
    const bool in1_is_locally_sharded = in1_is_sharded && !use_global_cb;

    // currently only support transpose of the full tile
    bool in0_transpose_tile = in0_tile.get_transpose_of_faces() && in0_tile.get_transpose_within_face();
    bool in1_transpose_tile = in1_tile.get_transpose_of_faces() && in1_tile.get_transpose_within_face();

    bool fuse_op = fused_op_signaler.has_value();

    uint32_t num_blocks = K / in0_block_w;
    // Only enable packer l1 accumulation when there are spills, otherwise
    // unnecessary overhead for reconfigs are added
    bool packer_l1_acc_en = packer_l1_acc && num_blocks > 1;

    // if fp32 enabled then we pack fp32 in l1, if not, then we pack fp16 in l1
    tt::DataFormat interm0_data_format = packer_l1_acc_en
                                             ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                                             : (fp32_dest_acc_en ? tt::DataFormat::Float32 : output_data_format);

    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    uint32_t bias_single_tile_size = bias_tile.get_tile_size(bias_data_format);
    uint32_t output_single_tile_size = output_tile.get_tile_size(output_data_format);
    uint32_t interm0_single_tile_size = output_tile.get_tile_size(interm0_data_format);

    bool do_not_inplace_interm0_out_CB = output_is_sharded && (per_core_M != out_block_h);

    uint32_t in0_block_h = out_block_h;
    uint32_t in1_block_w = out_block_w;
    uint32_t in0_num_blocks_y = per_core_M / out_block_h;
    uint32_t in1_num_blocks_x = per_core_N / out_block_w;
    uint32_t out_num_blocks_x = in1_num_blocks_x;
    uint32_t out_num_blocks_y = in0_num_blocks_y;

    uint32_t in0_block_tiles = in0_block_h * in0_block_w;
    uint32_t in0_CB_tiles = in0_block_tiles;
    if (B * num_blocks > 1) {
        in0_CB_tiles *= operations::matmul::utilities::MCAST_INPUT_BUFFERING_DEPTH;
    }
    // Tiles whose size is not a multiple of the DRAM alignment (e.g. bfp8 32x16 = 544B on
    // Blackhole's 64B alignment) are padded to it in DRAM. The interleaved reader copies tiles at
    // the padded stride, so the in0/in1/bias CBs must hold pages at the aligned stride and the
    // reader/unpacker walk tiles at the same stride. No-op when already aligned (all bf16 tiles,
    // 32-wide bfp8, Wormhole). Replaces the staging-CB workaround. Sharded CBs are backed by the
    // tensor buffer and keep their natural page size.
    const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();
    uint32_t in0_aligned_tile_size =
        in0_is_sharded ? in0_single_tile_size : tt::align(in0_single_tile_size, dram_alignment);
    uint32_t in1_aligned_tile_size = (in1_is_locally_sharded || use_global_cb)
                                         ? in1_single_tile_size
                                         : tt::align(in1_single_tile_size, dram_alignment);
    // Bias CB pages must be padded to the DRAM alignment so the reader's L1 write stride
    // matches the DRAM page stride (e.g. 64B on Blackhole for a 32B (1,16) bf16 bias tile).
    // Mirrors in0/in1 above. Sharded bias is backed by the L1 tensor buffer and keeps its
    // natural page size. No-op on Wormhole and for tiles already >= dram_alignment.
    uint32_t bias_aligned_tile_size =
        bias_is_sharded ? bias_single_tile_size : tt::align(bias_single_tile_size, dram_alignment);
    uint32_t in0_CB_size = in0_CB_tiles * in0_aligned_tile_size;

    uint32_t in2_block_tiles = 0;
    uint32_t in0_shard_width_in_tiles = 0;
    uint32_t in0_shard_height_in_tiles = 0;
    if (in0_is_sharded) {
        in0_shard_width_in_tiles = in0_tensor.shard_spec()->shape[1] / in0_tile.get_width();
        in0_shard_height_in_tiles = in0_tensor.shard_spec()->shape[0] / in0_tile.get_height();
        in2_block_tiles = per_core_M * in0_shard_width_in_tiles;
    }
    uint32_t in2_CB_tiles = in2_block_tiles;
    uint32_t in2_CB_size = in2_CB_tiles * in0_single_tile_size;

    uint32_t in1_block_tiles = out_block_w * in0_block_w;
    uint32_t in1_CB_tiles = in1_block_tiles;
    if (B * num_blocks > 1) {
        in1_CB_tiles *= operations::matmul::utilities::MCAST_INPUT_BUFFERING_DEPTH;
    }
    if (in1_is_locally_sharded) {
        uint32_t in1_shard_height_in_tiles = in1_tensor.shard_spec()->shape[0] / in1_tile.get_height();
        in1_CB_tiles = per_core_N * in1_shard_height_in_tiles;
    }

    uint32_t in1_CB_size = in1_CB_tiles * in1_aligned_tile_size;

    uint32_t out_block_tiles = out_block_h * out_block_w;
    uint32_t out_shard_tiles = per_core_M * per_core_N;
    uint32_t out_CB_tiles = out_block_tiles;  // No double buffer
    if (output_is_sharded) {
        out_CB_tiles = out_shard_tiles;
    }
    uint32_t out_CB_size = out_CB_tiles * output_single_tile_size;
    uint32_t interm0_CB_tiles = out_block_tiles;  // No double buffer
    uint32_t interm0_CB_size = interm0_CB_tiles * interm0_single_tile_size;

    uint32_t in3_block_tiles = out_block_w;
    uint32_t in3_CB_tiles = in3_block_tiles;  // No double buffer
    uint32_t in3_CB_size = in3_CB_tiles * bias_aligned_tile_size;

    CoreCoord start_core = sub_device_start_core;
    uint32_t start_core_x = start_core.x;
    uint32_t start_core_y = start_core.y;
    uint32_t num_cores_c = compute_with_storage_grid_size.x;

    // The matmul region is the rectangle of size `compute_with_storage_grid_size`
    // anchored at `start_core`. Callers must ensure this rectangle lies entirely
    // within the active sub-device's worker cores (validated upstream). Using a
    // CoreRangeSet here lets num_cores_to_corerangeset_in_subcoregrids honour the
    // start offset when the sub-device is not anchored at (0, 0).
    CoreRangeSet matmul_core_rect(CoreRange(
        start_core,
        CoreCoord(
            start_core_x + compute_with_storage_grid_size.x - 1, start_core_y + compute_with_storage_grid_size.y - 1)));

    uint32_t num_blocks_y = ((M - 1) / per_core_M) + 1;
    uint32_t num_blocks_x = ((N - 1) / per_core_N) + 1;
    uint32_t num_blocks_total = num_blocks_y * num_blocks_x;
    uint32_t num_cores_with_work = num_blocks_total;

    // mcast_in0 broadcasts a single M-block across all cores; only the start_core gets in0 sender
    // runtime args (with output_idx_y == 0), so M must fit within per_core_M.
    TT_FATAL(
        num_blocks_y == 1,
        "matmul_multicore_reuse_mcast_1d requires num_blocks_y == 1 (M <= per_core_M) for mcast_in0. "
        "Got M={}, per_core_M={}, num_blocks_y={}.",
        M,
        per_core_M,
        num_blocks_y);

    uint32_t in0_sender_num_cores = in0_is_sharded ? a.shard_spec().value().grid.num_cores() : 1;
    uint32_t num_cores = in0_is_sharded ? std::max(num_cores_with_work, in0_sender_num_cores) : num_cores_with_work;

    constexpr bool row_major = true;
    CoreRangeSet all_cores =
        num_cores_to_corerangeset_in_subcoregrids(start_core, num_cores, matmul_core_rect, row_major);

    CoreRangeSet in0_mcast_sender_cores =
        num_cores_to_corerangeset_in_subcoregrids(start_core, in0_sender_num_cores, matmul_core_rect, row_major);
    CoreCoord in0_mcast_sender_cores_grid = in0_mcast_sender_cores.bounding_box().grid_size();

    CoreRangeSet all_cores_with_work =
        num_cores_to_corerangeset_in_subcoregrids(start_core, num_cores_with_work, matmul_core_rect, row_major);
    if (use_global_cb) {
        TT_FATAL(
            global_cb->receiver_cores() == all_cores_with_work,
            "mcast_in0 global_cb receivers {} must exactly match output worker cores {}",
            global_cb->receiver_cores(),
            all_cores_with_work);
    }
    CoreRange in0_mcast_receiver_cores_bounding_box = all_cores_with_work.bounding_box();
    uint32_t in0_mcast_receiver_num_cores = in0_mcast_receiver_cores_bounding_box.size();  // always mcast to full grid
    uint32_t in0_mcast_receiver_num_dests = std::min(
        in0_mcast_receiver_num_cores,
        num_cores);  // should always be number of cores in receiver grid up to number of active cores

    CoreRangeSet in0_mcast_cores_with_work_and_in_receiver_grid;
    CoreRangeSet in0_mcast_cores_without_work_and_in_receiver_grid;
    CoreRangeSet in0_mcast_cores_without_work_and_not_in_receiver_grid;
    CoreRangeSet in0_mcast_receivers;
    std::vector<uint32_t> in0_mcast_noc_x;
    std::vector<uint32_t> in0_mcast_noc_y;
    if (in0_is_sharded) {
        in0_mcast_cores_with_work_and_in_receiver_grid = all_cores_with_work;

        if (in0_mcast_receiver_num_dests > num_cores_with_work) {
            const uint32_t in0_mcast_cores_without_work_and_in_receiver_grid_num_cores =
                in0_mcast_receiver_num_dests - num_cores_with_work;
            uint32_t core_idx_x = num_cores_with_work % num_cores_c;
            uint32_t core_idx_y = num_cores_with_work / num_cores_c;
            CoreCoord start_core = {(std::size_t)start_core_x + core_idx_x, (std::size_t)start_core_y + core_idx_y};
            in0_mcast_cores_without_work_and_in_receiver_grid = num_cores_to_corerangeset_in_subcoregrids(
                start_core, in0_mcast_cores_without_work_and_in_receiver_grid_num_cores, matmul_core_rect, row_major);
        }

        if (in0_sender_num_cores > in0_mcast_receiver_num_dests) {
            const uint32_t in0_mcast_cores_without_work_and_not_in_receiver_grid_num_cores =
                in0_sender_num_cores - in0_mcast_receiver_num_dests;
            uint32_t core_idx_x = in0_mcast_receiver_num_dests % num_cores_c;
            uint32_t core_idx_y = in0_mcast_receiver_num_dests / num_cores_c;
            CoreCoord start_core = {(std::size_t)start_core_x + core_idx_x, (std::size_t)start_core_y + core_idx_y};
            in0_mcast_cores_without_work_and_not_in_receiver_grid = num_cores_to_corerangeset_in_subcoregrids(
                start_core,
                in0_mcast_cores_without_work_and_not_in_receiver_grid_num_cores,
                matmul_core_rect,
                row_major);
        }

        in0_mcast_noc_x.reserve(in0_mcast_sender_cores_grid.x);
        in0_mcast_noc_y.reserve(in0_mcast_sender_cores_grid.y);
        for (uint32_t core_idx_x = 0; core_idx_x < in0_mcast_sender_cores_grid.x; ++core_idx_x) {
            in0_mcast_noc_x.push_back(
                device->worker_core_from_logical_core({start_core_x + core_idx_x, start_core_y}).x);
        }
        for (uint32_t core_idx_y = 0; core_idx_y < in0_mcast_sender_cores_grid.y; ++core_idx_y) {
            in0_mcast_noc_y.push_back(
                device->worker_core_from_logical_core({start_core_x, start_core_y + core_idx_y}).y);
        }
    } else {
        in0_mcast_cores_with_work_and_in_receiver_grid = CoreRangeSet({CoreRange(start_core, start_core)});
        if (in0_mcast_receiver_num_cores > 1) {
            // Check against the actual rectangle width (instead of bare grid_size.x-1) so that
            // sub-devices anchored away from (0, 0) wrap correctly.
            auto receiver_start_core = compute_with_storage_grid_size.x > 1 ? CoreCoord{start_core.x + 1, start_core.y}
                                                                            : CoreCoord{start_core.x, start_core.y + 1};
            in0_mcast_receivers = num_cores_to_corerangeset_in_subcoregrids(
                receiver_start_core, num_cores - 1, matmul_core_rect, row_major);
        }
    }

    // Mcast args
    auto in0_mcast_sender_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto in0_mcast_receiver_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);

    CoreCoord top_left_core = in0_mcast_receiver_cores_bounding_box.start_coord;
    CoreCoord bottom_right_core = in0_mcast_receiver_cores_bounding_box.end_coord;
    auto top_left_core_physical = device->worker_core_from_logical_core(top_left_core);
    auto bottom_right_core_physical = device->worker_core_from_logical_core(bottom_right_core);

    uint32_t in0_num_subblocks = (out_block_h / out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
    const auto& a_shape_logical = operations::matmul::utilities::get_matmul_tensor_logical_shape(a, transpose_a);
    // When transpose_a is true, the K dimension maps to the row dimension of the raw tile,
    // which is already zero-padded during tile layout conversion. pad_last_ktile operates on
    // columns, so applying it would incorrectly zero valid data that becomes output rows
    // after the compute kernel transposes the tile.
    const auto in0_last_ktile_w = transpose_a ? 0 : a_shape_logical[-1] % in0_tile.get_width();
    const auto in0_last_ktile_h = transpose_a ? a_shape_logical[-1] % in0_tile.get_width() : 0;
    TT_FATAL(
        in0_last_ktile_w == 0 || in0_last_ktile_h == 0,
        "At most one of in0_last_ktile_w ({}) and in0_last_ktile_h ({}) can be non-zero",
        in0_last_ktile_w,
        in0_last_ktile_h);

    const auto& a_padded_shape = operations::matmul::utilities::get_matmul_tensor_padded_shape(a, transpose_a);
    const uint32_t M_per_batch = a_padded_shape[-2] / in0_tile.get_height();
    const auto [in0_tensor_stride_w, in0_tensor_stride_h] =
        operations::matmul::utilities::get_in0_transpose_strides(M, M_per_batch, transpose_a, K);
    const auto in0_tensor_next_block_stride = in0_block_w * in0_tensor_stride_w;
    const auto in0_tensor_next_h_dim_block_stride = in0_block_h * in0_tensor_stride_h;
    const auto in0_tensor_start_tile_id_stride = per_core_M * in0_tensor_stride_h;

    const auto in1_tensor_stride_w = transpose_b ? K : 1;
    const auto in1_tensor_stride_h = transpose_b ? 1 : N;
    const auto in1_tensor_next_block_stride = in0_block_w * in1_tensor_stride_h;
    const auto in1_tensor_next_w_dim_block_stride = in1_block_w * in1_tensor_stride_w;
    const auto in1_tensor_start_tile_id_stride = per_core_N * in1_tensor_stride_w;

    std::vector<uint32_t> in0_sender_compile_time_args;
    if (in0_is_sharded) {
        in0_sender_compile_time_args = {
            (std::uint32_t)1,  // core_has_output_block_work
            (std::uint32_t)1,  // core_in_in0_receiver_mcast_grid

            (std::uint32_t)in0_block_num_tiles,                         // in0_block_num_tiles
            (std::uint32_t)in0_block_num_tiles * in0_single_tile_size,  // in0_block_size_bytes
            (std::uint32_t)in0_last_ktile_w,
            (std::uint32_t)in0_last_ktile_h,

            // in0/in1 common args
            (std::uint32_t)num_blocks,        // num_blocks
            (std::uint32_t)out_num_blocks_x,  // num_blocks_x
            (std::uint32_t)out_num_blocks_y,  // num_blocks_y
            // in0 mcast args
            (std::uint32_t)in0_mcast_sender_semaphore_id,
            (std::uint32_t)in0_mcast_receiver_semaphore_id,
            (std::uint32_t)in0_mcast_receiver_num_dests,  // in0_mcast_num_dests
            (std::uint32_t)in0_mcast_receiver_num_cores,  // in0_mcast_num_cores
            (std::uint32_t)(in0_mcast_sender_cores_grid.x),
            (std::uint32_t)(in0_mcast_sender_cores_grid.y),
            (std::uint32_t)(false),
            (std::uint32_t)(in0_shard_width_in_tiles),
            (std::uint32_t)(in0_shard_height_in_tiles),
            (std::uint32_t)(in0_block_w),
            (std::uint32_t)in0_block_h,  // in0_block_h

            // batch args
            (std::uint32_t)B  // batch
        };
    } else {
        in0_sender_compile_time_args = {
            // in0 tensor args
            (std::uint32_t)in0_tensor_stride_w,
            (std::uint32_t)in0_tensor_stride_h,
            (std::uint32_t)in0_tensor_next_block_stride,
            (std::uint32_t)in0_tensor_next_h_dim_block_stride,
            // in0 block args
            (std::uint32_t)in0_block_w,          // in0_block_w
            (std::uint32_t)in0_block_h,          // in0_block_h
            (std::uint32_t)in0_block_num_tiles,  // in0_block_num_tiles
            (std::uint32_t)in0_last_ktile_w,
            (std::uint32_t)in0_last_ktile_h,
            (std::uint32_t)false,  // extract_shard_sub_blocks (not used for interleaved)
            (std::uint32_t)0,      // shard_width_in_tiles (not used for interleaved)
            (std::uint32_t)0,      // shard_height_in_tiles (not used for interleaved)
            // in0/in1 common args
            (std::uint32_t)num_blocks,        // num_blocks
            (std::uint32_t)out_num_blocks_x,  // num_blocks_x
            (std::uint32_t)out_num_blocks_y,  // num_blocks_y
            // in0 mcast args
            (std::uint32_t)in0_mcast_sender_semaphore_id,
            (std::uint32_t)in0_mcast_receiver_semaphore_id,
            (std::uint32_t)num_cores - 1,                     // in0_mcast_num_dests
            (std::uint32_t)in0_mcast_receiver_num_cores - 1,  // in0_mcast_num_cores
            // batch args
            (std::uint32_t)M * K,  // MtKt
            (std::uint32_t)B,      // batch
            (std::uint32_t)B,      // batch
            (std::uint32_t)false,  // reuse_in0_in_CB
            // sparsity args
            (std::uint32_t)0,     // batchB
            (std::uint32_t)0,     // sparsity_pagesize (placeholder since sparsity not used in this case)
            (std::uint32_t)true,  // bcast_A
            (std::uint32_t)false  // get_batch_from_reader
        };
    }
    in0_sender_compile_time_args.push_back((std::uint32_t)(fuse_op && fused_op_signaler->is_all_gather()));
    tt::tt_metal::TensorAccessorArgs(in0_tensor).append_to(in0_sender_compile_time_args);
    tt::tt_metal::TensorAccessorArgs().append_to(in0_sender_compile_time_args);  // placeholder for sparsity
    in0_sender_compile_time_args.push_back((std::uint32_t)0);  // num_batch_compute (unused, sparsity disabled)

    std::vector<uint32_t> in1_sender_writer_compile_time_args = {
        // READER
        // in1 tensor args
        (std::uint32_t)in1_tensor_stride_w,
        (std::uint32_t)in1_tensor_stride_h,
        (std::uint32_t)in1_tensor_next_block_stride,
        (std::uint32_t)in1_tensor_next_w_dim_block_stride,
        // in1 block args
        (std::uint32_t)in1_block_w,                // in1_block_w
        (std::uint32_t)in0_block_w,                // in1_block_h
        (std::uint32_t)in1_block_w * in0_block_w,  // in1_block_num_tiles
        // in0/in1 common args
        (std::uint32_t)num_blocks,        // num_blocks
        (std::uint32_t)out_num_blocks_x,  // out_num_blocks_x
        (std::uint32_t)out_num_blocks_y,  // out_num_blocks_y
        // in1 mcast args
        (std::uint32_t)0,
        (std::uint32_t)0,
        (std::uint32_t)0,  // in1_mcast_num_dests
        (std::uint32_t)0,  // in1_mcast_num_cores
        // batch args
        (std::uint32_t)K * N,        // KtNt
        (std::uint32_t)B,            // batch
        (std::uint32_t)bcast_batch,  // bcast_B
        // sparsity args
        (std::uint32_t)0,  // batchB
        (std::uint32_t)0,  // sparsity_pagesize (placeholder since sparsity not used in this case)

        // WRITER
        // out tensor args
        (std::uint32_t)1,                   // out_tensor_stride_w
        (std::uint32_t)N,                   // out_tensor_stride_h
        (std::uint32_t)out_subblock_w,      // out_tensor_next_subblock_stride_w
        (std::uint32_t)out_subblock_h * N,  // out_tensor_next_subblock_stride_h
        (std::uint32_t)out_block_w,         // out_tensor_next_w_dim_block_stride
        (std::uint32_t)out_block_h * N,     // out_tensor_next_h_dim_block_stride
        // out subblock args
        (std::uint32_t)out_subblock_w,                     // out_subblock_w
        (std::uint32_t)out_subblock_h,                     // out_subblock_h
        (std::uint32_t)(out_subblock_w * out_subblock_h),  // out_subblocks_w * out_subblocks_h
        // batch args
        (std::uint32_t)M * N  // MtNt
    };
    if (bias_tensor.has_value()) {
        in1_sender_writer_compile_time_args.push_back((std::uint32_t)1);  // in3_tensor_stride_w
    } else {
        in1_sender_writer_compile_time_args.push_back(0);  // Placeholder; not used
    }

    in1_sender_writer_compile_time_args.push_back((std::uint32_t)(fuse_op && fused_op_signaler->is_all_gather()));
    in1_sender_writer_compile_time_args.push_back((std::uint32_t)(fuse_op && fused_op_signaler->is_reduce_scatter()));
    in1_sender_writer_compile_time_args.push_back((std::uint32_t)false);  // compact_output

    // Append TensorAccessorArgs
    tt::tt_metal::TensorAccessorArgs(in1_tensor).append_to(in1_sender_writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs().append_to(in1_sender_writer_compile_time_args);  // placeholder for sparsity
    tt::tt_metal::TensorAccessorArgs(out_tensor).append_to(in1_sender_writer_compile_time_args);
    if (bias_tensor.has_value()) {
        tt::tt_metal::TensorAccessorArgs(*bias_tensor).append_to(in1_sender_writer_compile_time_args);
    }

    std::vector<uint32_t> in0_receiver_compile_time_args = {
        // in0 block args
        (std::uint32_t)in0_block_num_tiles,  // in0_block_num_tiles
        // in0/in1 common args
        (std::uint32_t)num_blocks,        // num_blocks
        (std::uint32_t)out_num_blocks_x,  // out_num_blocks_x
        (std::uint32_t)out_num_blocks_y,  // out_num_blocks_y
        // in0 mcast args
        (std::uint32_t)in0_mcast_sender_semaphore_id,
        (std::uint32_t)in0_mcast_receiver_semaphore_id,
        // batch args
        (std::uint32_t)B,     // batch
        (std::uint32_t)false  // get_batch_from_reader
    };

    std::map<std::string, std::string> mm_kernel_defines;
    std::map<std::string, std::string> mm_kernel_in0_sender_writer_defines;
    std::map<std::string, std::string> mm_kernel_in1_sender_writer_defines;
    if (bias_tensor.has_value()) {
        mm_kernel_defines["FUSE_BIAS"] = "1";
        mm_kernel_in1_sender_writer_defines["FUSE_BIAS"] = "1";
    }
    if (fused_activation.has_value()) {
        if (fused_activation.value().op_type == UnaryOpType::RELU) {
            mm_kernel_defines["PACK_RELU"] = "1";
        } else {
            mm_kernel_defines["SFPU_ACTIVATION"] = "1";
        }
    }
    if (packer_l1_acc_en) {
        mm_kernel_defines["PACKER_L1_ACC"] = "1";
    }
    if (fp32_dest_acc_en) {
        mm_kernel_defines["FP32_DEST_ACC_EN"] = "1";
    }
    if (in1_transpose_tile) {
        mm_kernel_defines["IN1_TRANSPOSE_TILE"] = "1";
    }

    ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
        device->arch(), num_cores, mm_kernel_defines);
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(), num_cores, mm_kernel_defines, throttle_level);

    if (in1_is_locally_sharded) {
        mm_kernel_in1_sender_writer_defines["IN1_SHARDED"] = "1";
    }
    if (use_global_cb) {
        mm_kernel_in1_sender_writer_defines["ENABLE_GLOBAL_CB"] = "1";
    }

    if (bias_is_sharded) {
        mm_kernel_in1_sender_writer_defines["BIAS_SHARDED"] = "1";
    }

    if (output_is_sharded) {
        mm_kernel_in1_sender_writer_defines["OUT_SHARDED"] = "1";
    }

    // TODO: SKIP_MCAST flag isn't used for the sharded reader kernel because internal mcast logic already works without
    // skipping We can use this flag to turn off unnecessary mcast overhead if necessary
    if (in0_mcast_receiver_num_cores == 1) {
        mm_kernel_in0_sender_writer_defines["SKIP_MCAST"] = "1";
    }

    mm_kernel_in1_sender_writer_defines["SKIP_MCAST"] = "1";

    // Intermediate CB read
    /*
    Blackhole architecture alignment issue workaround for tiny tiles:

    Problem: When reading tiny tiles from DRAM to circular buffers (CB), address alignment
    issues occur. DRAM tile addresses are 64-byte aligned within each block, but L1 CB
    addresses are not necessarily aligned due to non-64-byte-aligned page sizes.

    Example scenario:
    - Two consecutive 544-byte tiles (16x32 tile of dtype bfloat8_b) stored on different DRAM banks
    - CB configured with size=2 to hold both tiles

    Result:
    - Tile 0: DRAM Bank 0, Address 64    → CB L1 Address 0   (64-byte aligned ✓)
    - Tile 1: DRAM Bank 1, Address 64    → CB L1 Address 544 (not 64-byte aligned ✗)

    Solution: Use an intermediate single-tile CB as a staging area. Read each tile into
    the intermediate CB first, then copy to the destination CB. This ensures proper
    alignment at the cost of additional memory bandwidth overhead.

    Note: This workaround should only be used for this specific alignment issue case.
    */

    // in1 is the reader of weights/output writer, and we choose to make it use the optimized reader noc
    tt_metal::NOC in0_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    tt_metal::NOC in1_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());

    if (fuse_op && fused_op_signaler->is_all_gather()) {
        // Create semaphores
        fused_op_signaler->init_fused_op(
            program,
            device,
            in0_mcast_sender_cores,
            in0_is_sharded ? ttnn::experimental::ccl::FusedOpSignalerMode::SINGLE
                           : ttnn::experimental::ccl::FusedOpSignalerMode::MULTI);
    }

    auto mm_kernel_in0_mcast_cores_with_work_and_in_receiver_grid_id = tt_metal::CreateKernel(
        program,
        in0_is_sharded
            ? "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
              "reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp"
            : "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp",
        in0_mcast_cores_with_work_and_in_receiver_grid,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = in0_noc,
            .compile_args = in0_sender_compile_time_args,
            .defines = mm_kernel_in0_sender_writer_defines,
            .named_compile_args = {
                {"cb_in0", tt::CBIndex::c_0},
                {"cb_in0_sharded", tt::CBIndex::c_2},
                {"cb_l1_array", tt::CBIndex::c_6},
                {"cb_sparsity", tt::CBIndex::c_6},
                // Indexed/gather mode is sparse-matmul-only; 0 disables it here. The reader reads this
                // name unconditionally, so every factory building it must pass it. (This block also
                // serves the block-sharded reader selected above, which simply ignores it.)
                {"num_active", 0},
            }});

    tt::tt_metal::KernelHandle mm_kernel_in0_mcast_cores_without_work_and_in_receiver_grid_id = 0;
    tt::tt_metal::KernelHandle mm_kernel_in0_mcast_cores_without_work_and_not_in_receiver_grid_id = 0;
    if (in0_is_sharded) {
        if (in0_mcast_cores_without_work_and_in_receiver_grid.num_cores() > 0) {
            in0_sender_compile_time_args[0] = 0;  // core_has_output_block_work
            in0_sender_compile_time_args[1] = 1;  // core_in_in0_receiver_mcast_grid
            mm_kernel_in0_mcast_cores_without_work_and_in_receiver_grid_id = tt_metal::CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
                "reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp",
                in0_mcast_cores_without_work_and_in_receiver_grid,
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1,
                    .noc = in0_noc,
                    .compile_args = in0_sender_compile_time_args,
                    .defines = mm_kernel_in0_sender_writer_defines,
                    .named_compile_args = {
                        {"cb_in0", tt::CBIndex::c_0},
                        {"cb_in0_sharded", tt::CBIndex::c_2},
                        {"cb_l1_array", tt::CBIndex::c_6},
                    }});
        }
        if (in0_mcast_cores_without_work_and_not_in_receiver_grid.num_cores() > 0) {
            in0_sender_compile_time_args[0] = 0;  // core_has_output_block_work
            in0_sender_compile_time_args[1] = 0;  // core_in_in0_receiver_mcast_grid
            mm_kernel_in0_mcast_cores_without_work_and_not_in_receiver_grid_id = tt_metal::CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
                "reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp",
                in0_mcast_cores_without_work_and_not_in_receiver_grid,
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1,
                    .noc = in0_noc,
                    .compile_args = in0_sender_compile_time_args,
                    .defines = mm_kernel_in0_sender_writer_defines,
                    .named_compile_args = {
                        {"cb_in0", tt::CBIndex::c_0},
                        {"cb_in0_sharded", tt::CBIndex::c_2},
                        {"cb_l1_array", tt::CBIndex::c_6},
                    }});
        }
    }

    tt::tt_metal::KernelHandle mm_kernel_in0_receiver_id = 0;
    if (!in0_is_sharded and in0_mcast_receivers.num_cores() > 0) {
        mm_kernel_in0_receiver_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_receiver.cpp",
            in0_mcast_receivers,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = in0_noc,
                .compile_args = in0_receiver_compile_time_args,
                .named_compile_args = {
                    {"cb_in0", tt::CBIndex::c_0},
                }});
    }

    auto mm_kernel_in1_sender_writer_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp",
        all_cores_with_work,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = in1_noc,
            .compile_args = in1_sender_writer_compile_time_args,
            .defines = mm_kernel_in1_sender_writer_defines,
            .named_compile_args = {
                {"cb_in1", tt::CBIndex::c_1},
                {"cb_bias", tt::CBIndex::c_3},
                {"cb_out", tt::CBIndex::c_4},
                {"cb_sparsity", tt::CBIndex::c_7},
                {"num_active", 0},  // indexed/gather mode: sparse_matmul only (0 = disabled)
            }});

    // Compute kernel compile time args

    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;

    uint32_t in1_num_subblocks = (out_block_w / out_subblock_w);
    uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks;
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    std::vector<uint32_t> compute_kernel_args = {
        in0_block_w,             // in0_block_w
        in0_num_subblocks,       // in0_num_subblocks
        in0_block_num_tiles,     // in0_block_num_tiles
        in0_subblock_num_tiles,  // in0_subblock_num_tiles

        in1_num_subblocks,    // in1_num_subblocks
        in1_block_num_tiles,  // in1_block_num_tiles
        in1_per_core_w,       // in1_per_core_w

        num_blocks,        // num_blocks
        out_num_blocks_x,  // out_num_blocks_x
        out_num_blocks_y,  // out_num_blocks_y

        out_subblock_h,          // out_subblock_h
        out_subblock_w,          // out_subblock_w
        out_subblock_num_tiles,  // out_subblock_num_tiles
        B,                       // batch
        out_block_tiles,         // out_block_num_tiles

        untilize_out,  // untilize_out
        false,         // get_batch_from_reader
        in0_transpose_tile,
    };
    if (bias_tensor.has_value()) {
        compute_kernel_args.push_back(row_broadcast_bias ? 1u : 0u);
    }

    constexpr auto cb_intermed0 = tt::CBIndex::c_5;
    std::unordered_map<std::string, uint32_t> compute_named_compile_args = {
        {"cb_in0", tt::CBIndex::c_0},
        {"cb_in1", tt::CBIndex::c_1},
        {"cb_bias", tt::CBIndex::c_3},
        {"cb_out", tt::CBIndex::c_4},
        {"cb_intermed0", cb_intermed0},
        {"cb_in0_transposed", tt::CBIndex::c_10},
        {"bias_ntiles", in1_per_core_w},
    };

    if (fused_activation.has_value() && fused_activation.value().op_type != UnaryOpType::RELU) {
        using ttnn::operations::matmul::utilities::get_activation_params;
        const auto& activation = fused_activation.value();
        const auto params = get_activation_params(activation);
        compute_named_compile_args["activation_type"] = static_cast<uint32_t>(params.type);
        compute_named_compile_args["activation_param0"] = params.param0;
        compute_named_compile_args["activation_param1"] = params.param1;
        compute_named_compile_args["activation_param2"] = params.param2;
    }

    // The fused bias add reads the partials CB as an FPU operand (SrcA), so UnpackToDestFp32 cannot be
    // set on cb_intermed0 directly when bias is present. In that case the cross-block reload instead
    // copies through cb_intermed0_alias, a second buffer index over the same SRAM carrying
    // UnpackToDestFp32, while the bias add keeps reading cb_intermed0 via SrcA. The alias index is
    // handed to the compute kernel through the MM_PARTIALS_RELOAD_ALIAS_CB define, which selects the
    // alias reload path there. Without bias the reload reads cb_intermed0 and the flag is set on it.
    constexpr auto cb_intermed0_alias = tt::CBIndex::c_7;
    const bool bias_reload_alias =
        fp32_dest_acc_en && interm0_data_format == tt::DataFormat::Float32 && bias_tensor.has_value();
    if (bias_reload_alias) {
        mm_kernel_defines["MM_PARTIALS_RELOAD_ALIAS_CB"] = std::to_string(static_cast<uint32_t>(cb_intermed0_alias));
    }
    // When accumulating in fp32 (fp32_dest_acc_en) with the K reduction split across blocks,
    // the intermediate partials CB (cb_intermed0) holds Float32 and is reloaded into DEST
    // between blocks by copy_block_matmul_partials. Unless the reload's CB view is marked
    // UnpackToDestFp32, that reload is routed through SrcA and rounded to TF32 (10 mantissa bits),
    // so the fp32 partial loses precision on every block boundary. The flag is set on the alias
    // when bias forces a separate SrcA view of cb_intermed0 (see above), else on cb_intermed0 itself.
    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    if (fp32_dest_acc_en && interm0_data_format == tt::DataFormat::Float32) {
        const uint32_t cb_to_mark =
            bias_reload_alias ? static_cast<uint32_t>(cb_intermed0_alias) : static_cast<uint32_t>(cb_intermed0);
        unpack_to_dest_mode[cb_to_mark] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }

    // Create compute kernel
    // bool fp32_dest_acc_en = false;
    // Gelu currently has better accuracy when run in approx mode
    // bool math_approx_mode = false;
    tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp",
        all_cores_with_work,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args,
            .defines = mm_kernel_defines,
            .named_compile_args = compute_named_compile_args});

    // Create circular buffers
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    tt_metal::CircularBufferConfig src0_cb_config =
        tt_metal::CircularBufferConfig(in0_CB_size, {{src0_cb_index, in0_data_format}})
            .set_page_size(src0_cb_index, in0_aligned_tile_size)
            .set_tile_dims(src0_cb_index, in0_tile);
    tt_metal::CreateCircularBuffer(program, all_cores, src0_cb_config);
    log_debug(
        LogOp,
        "CB {} :: PS = {}, NP = {}, TOTAL = {}",
        src0_cb_index,
        in0_single_tile_size,
        in0_CB_size / in0_single_tile_size,
        in0_CB_size);

    uint32_t src1_cb_index = tt::CBIndex::c_1;
    tt::tt_metal::CBHandle cb_src1 = 0;
    uint32_t in1_remote_cb_size = 0;
    if (use_global_cb) {
        const uint32_t remote_cb_index = tt::CBIndex::c_31;
        const uint32_t in1_block_size_bytes = in1_block_tiles * in1_single_tile_size;
        // Floor the GCB to a whole number of in1 K-block pages, as the gather_in0 path does. The
        // caller sizes the GCB in bytes and need not know this op's page size; any bytes past the
        // last whole page are simply unused rather than a hard error. The reader streams through a
        // two-page window (remote_cb_wait_front of 1 then 2), so that much has to survive the floor —
        // op validation checks this too, but a GCB built via the raw factory skips that path.
        in1_remote_cb_size = tt::round_down(global_cb->size(), in1_block_size_bytes);
        TT_FATAL(
            in1_remote_cb_size >= 2 * in1_block_size_bytes,
            "mcast_in0 global_cb size {} holds {} whole in1 K-block pages of {} B; the reader needs at least 2",
            global_cb->size(),
            in1_remote_cb_size / in1_block_size_bytes,
            in1_block_size_bytes);
        tt_metal::CircularBufferConfig remote_cb_config(in1_remote_cb_size);
        remote_cb_config.remote_index(remote_cb_index)
            .set_page_size(in1_block_size_bytes)
            .set_data_format(in1_data_format);
        remote_cb_config.index(src1_cb_index)
            .set_page_size(in1_single_tile_size)
            .set_data_format(in1_data_format)
            .set_tile_dims(in1_tile);
        cb_src1 =
            tt_metal::experimental::CreateCircularBuffer(program, all_cores_with_work, remote_cb_config, *global_cb);
    } else {
        tt_metal::CircularBufferConfig src1_cb_config =
            tt_metal::CircularBufferConfig(in1_CB_size, {{src1_cb_index, in1_data_format}})
                .set_page_size(src1_cb_index, in1_aligned_tile_size)
                .set_tile_dims(src1_cb_index, in1_tile);
        if (in1_is_locally_sharded) {
            src1_cb_config = src1_cb_config.set_globally_allocated_address(in1_tensor);
        }
        cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, src1_cb_config);
    }
    log_debug(
        LogOp,
        "CB {} :: PS = {}, NP = {}, TOTAL = {}",
        src1_cb_index,
        in1_single_tile_size,
        use_global_cb ? in1_remote_cb_size / in1_single_tile_size : in1_CB_size / in1_single_tile_size,
        use_global_cb ? in1_remote_cb_size : in1_CB_size);

    uint32_t src2_cb_index = tt::CBIndex::c_2;
    tt::tt_metal::CBHandle cb_src2 = 0;
    if (in0_is_sharded) {
        tt_metal::CircularBufferConfig src2_cb_config =
            tt_metal::CircularBufferConfig(in2_CB_size, {{src2_cb_index, in0_data_format}})
                .set_page_size(src2_cb_index, in0_single_tile_size)
                .set_globally_allocated_address(in0_tensor)
                .set_tile_dims(src2_cb_index, in0_tile);
        cb_src2 = tt_metal::CreateCircularBuffer(program, all_cores, src2_cb_config);
        log_debug(
            LogOp,
            "CB {} :: PS = {}, NP = {}, TOTAL = {}",
            src2_cb_index,
            in0_single_tile_size,
            in2_CB_size / in0_single_tile_size,
            in2_CB_size);

        // Local L1 to store temp vars
        uint32_t l1_cb_index = tt::CBIndex::c_6;
        tt::tt_metal::CircularBufferConfig cb_for_l1_array_config =
            tt::tt_metal::CircularBufferConfig(32 * 2, {{l1_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(l1_cb_index, 32 * 2);
        tt_metal::CreateCircularBuffer(program, all_cores, cb_for_l1_array_config);
    }

    uint32_t output_cb_index = tt::CBIndex::c_4;
    uint32_t interm0_cb_index = tt::CBIndex::c_5;
    tt_metal::CircularBufferConfig interm0_cb_config =
        tt_metal::CircularBufferConfig(0, {{interm0_cb_index, interm0_data_format}});
    tt_metal::CircularBufferConfig output_cb_config =
        tt_metal::CircularBufferConfig(0, {{output_cb_index, output_data_format}});

    if (do_not_inplace_interm0_out_CB || (interm0_data_format != output_data_format) ||
        (untilize_out && (in1_num_subblocks > 1))) {
        // output
        std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
            {output_cb_index, output_data_format},
        };
        output_cb_config = tt_metal::CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
                               .set_page_size(output_cb_index, output_single_tile_size)
                               .set_tile_dims(output_cb_index, output_tile);
        // interm0
        std::map<uint8_t, tt::DataFormat> interm0_cb_data_format_spec{
            {interm0_cb_index, interm0_data_format},
        };
        // Alias over the same SRAM, marked UnpackToDestFp32, for the bias reload (see above).
        if (bias_reload_alias) {
            interm0_cb_data_format_spec[static_cast<uint8_t>(cb_intermed0_alias)] = interm0_data_format;
        }
        interm0_cb_config = tt_metal::CircularBufferConfig(interm0_CB_size, interm0_cb_data_format_spec)
                                .set_page_size(interm0_cb_index, interm0_single_tile_size)
                                .set_tile_dims(interm0_cb_index, output_tile);
        if (bias_reload_alias) {
            interm0_cb_config =
                interm0_cb_config.set_page_size(static_cast<uint8_t>(cb_intermed0_alias), interm0_single_tile_size)
                    .set_tile_dims(static_cast<uint8_t>(cb_intermed0_alias), output_tile);
        }

        tt_metal::CreateCircularBuffer(program, CoreRangeSet({all_cores}), interm0_cb_config);
        log_debug(
            LogOp,
            "CB {} :: PS = {}, NP = {}, TOTAL = {}",
            interm0_cb_index,
            interm0_single_tile_size,
            interm0_CB_size / interm0_single_tile_size,
            interm0_CB_size);
    } else {
        // share buffer
        std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
            {output_cb_index, output_data_format}, {interm0_cb_index, interm0_data_format}};
        // Alias over the same SRAM, marked UnpackToDestFp32, for the bias reload (see above).
        if (bias_reload_alias) {
            output_cb_data_format_spec[static_cast<uint8_t>(cb_intermed0_alias)] = interm0_data_format;
        }
        output_cb_config = tt_metal::CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
                               .set_page_size(output_cb_index, output_single_tile_size)
                               .set_page_size(interm0_cb_index, interm0_single_tile_size)
                               .set_tile_dims(output_cb_index, output_tile)
                               .set_tile_dims(interm0_cb_index, output_tile);
        if (bias_reload_alias) {
            output_cb_config =
                output_cb_config.set_page_size(static_cast<uint8_t>(cb_intermed0_alias), interm0_single_tile_size)
                    .set_tile_dims(static_cast<uint8_t>(cb_intermed0_alias), output_tile);
        }
    }

    if (output_is_sharded) {
        output_cb_config = output_cb_config.set_globally_allocated_address(out_tensor);
    }
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);
    log_debug(
        LogOp,
        "CB {} :: PS = {}, NP = {}, TOTAL = {}",
        output_cb_index,
        output_single_tile_size,
        out_CB_size / output_single_tile_size,
        out_CB_size);

    tt_metal::CBHandle cb_src3 = 0;
    if (bias_tensor.has_value()) {
        uint32_t src3_cb_index = tt::CBIndex::c_3;
        tt_metal::CircularBufferConfig cb_src3_config =
            tt_metal::CircularBufferConfig(in3_CB_size, {{src3_cb_index, bias_data_format}})
                .set_page_size(src3_cb_index, bias_aligned_tile_size)
                .set_tile_dims(src3_cb_index, bias_tile);

        if (bias_is_sharded) {
            cb_src3_config = cb_src3_config.set_globally_allocated_address(*bias_tensor);
        }

        cb_src3 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src3_config);
        log_debug(
            LogOp,
            "CB {} :: PS = {}, NP = {}, TOTAL = {}",
            src3_cb_index,
            bias_aligned_tile_size,
            in3_CB_size / bias_aligned_tile_size,
            in3_CB_size);
    }

    // Intermediate CB read

    // Transpose CB for input0
    if (in0_transpose_tile) {
        const uint32_t in0_transpose_cb_index = tt::CBIndex::c_10;
        auto in0_transpose_cb_config =
            tt_metal::CircularBufferConfig(in0_CB_size, {{in0_transpose_cb_index, in0_data_format}})
                .set_page_size(in0_transpose_cb_index, in0_aligned_tile_size)
                .set_tile_dims(in0_transpose_cb_index, in0_tile);
        tt_metal::CreateCircularBuffer(program, all_cores, in0_transpose_cb_config);
    }

    // Parameters for last row, col, or block. mcast_in0 replicates M across all cores
    // (num_blocks_y == 1), so any M-direction padding lives entirely within a single h-block.
    uint32_t last_per_core_N = N % per_core_N == 0 ? per_core_N : N % per_core_N;
    uint32_t last_out_block_w = last_per_core_N % out_block_w == 0 ? out_block_w : last_per_core_N % out_block_w;
    uint32_t last_out_num_blocks_w = ((last_per_core_N - 1) / out_block_w) + 1;
    uint32_t last_block_num_nonzero_subblocks_w = ((last_out_block_w - 1) / out_subblock_w) + 1;
    uint32_t last_subblock_of_last_block_w =
        last_out_block_w % out_subblock_w == 0 ? out_subblock_w : last_out_block_w % out_subblock_w;
    uint32_t last_block_padded_subblock_tiles_addr_skip =
        output_single_tile_size * (out_subblock_w - last_subblock_of_last_block_w);
    uint32_t last_block_padded_block_tiles_w_skip =
        (out_subblock_w * out_subblock_h) * (out_block_w / out_subblock_w - last_block_num_nonzero_subblocks_w);

    // M-direction padding when M < per_core_M. With num_blocks_y == 1 the last (only) h-block
    // holds last_out_block_h valid tile rows out of out_block_h.
    uint32_t in0_last_per_core_M = M < per_core_M ? M : per_core_M;
    uint32_t in0_last_out_block_h =
        in0_last_per_core_M % out_block_h == 0 ? out_block_h : in0_last_per_core_M % out_block_h;
    uint32_t in0_last_block_num_nonzero_subblocks_h = ((in0_last_out_block_h - 1) / out_subblock_h) + 1;
    uint32_t in0_last_subblock_of_last_block_h =
        in0_last_out_block_h % out_subblock_h == 0 ? out_subblock_h : in0_last_out_block_h % out_subblock_h;
    uint32_t in0_last_block_padded_block_tiles_h_skip =
        (out_block_h / out_subblock_h - in0_last_block_num_nonzero_subblocks_h) * (out_block_w * out_subblock_h);

    CoreCoord start_core_noc = top_left_core_physical;
    CoreCoord end_core_noc = bottom_right_core_physical;
    if (in0_noc == tt::tt_metal::NOC::NOC_1) {
        std::swap(start_core_noc, end_core_noc);
    }

    const auto& cores = corerange_to_cores(all_cores, std::nullopt, row_major);
    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& core = cores[i];
        uint32_t output_idx_x = i % num_blocks_x;
        uint32_t output_idx_y = i / num_blocks_x;

        if (in0_is_sharded) {
            std::vector<uint32_t> mm_in0_sender_args;
            mm_in0_sender_args.reserve(5 + in0_mcast_noc_x.size() + in0_mcast_noc_y.size());
            mm_in0_sender_args.push_back(i);
            mm_in0_sender_args.push_back(start_core_noc.x);
            mm_in0_sender_args.push_back(start_core_noc.y);
            mm_in0_sender_args.push_back(end_core_noc.x);
            mm_in0_sender_args.push_back(end_core_noc.y);
            mm_in0_sender_args.insert(mm_in0_sender_args.end(), in0_mcast_noc_x.begin(), in0_mcast_noc_x.end());
            mm_in0_sender_args.insert(mm_in0_sender_args.end(), in0_mcast_noc_y.begin(), in0_mcast_noc_y.end());

            if (fuse_op && fused_op_signaler->is_all_gather()) {
                fused_op_signaler->push_matmul_fused_op_rt_args(mm_in0_sender_args, false);
            }

            if (i < num_cores_with_work) {
                tt_metal::SetRuntimeArgs(
                    program,
                    mm_kernel_in0_mcast_cores_with_work_and_in_receiver_grid_id,
                    core,
                    mm_in0_sender_args);  // RISCV_0_default
            } else if (i < in0_mcast_receiver_num_dests) {
                tt_metal::SetRuntimeArgs(
                    program,
                    mm_kernel_in0_mcast_cores_without_work_and_in_receiver_grid_id,
                    core,
                    mm_in0_sender_args);  // RISCV_0_default
            } else {
                tt_metal::SetRuntimeArgs(
                    program,
                    mm_kernel_in0_mcast_cores_without_work_and_not_in_receiver_grid_id,
                    core,
                    mm_in0_sender_args);  // RISCV_0_default
            }
        }
        // in0 sender and in1 sender
        else if (core == start_core) {
            std::vector<uint32_t> mm_in0_sender_args = {
                // in0 tensor args
                (std::uint32_t)in0_tensor.address(),
                (std::uint32_t)in0_tensor_start_tile_id_stride * output_idx_y,  // in0_tensor_start_tile_id
                // in0 mcast args
                (std::uint32_t)start_core_noc.x,  // in0_mcast_dest_noc_start_x
                (std::uint32_t)start_core_noc.y,  // in0_mcast_dest_noc_start_y
                (std::uint32_t)end_core_noc.x,    // in0_mcast_dest_noc_end_x
                (std::uint32_t)end_core_noc.y,    // in0_mcast_dest_noc_end_y

                // padding args
                (std::uint32_t)in0_last_out_block_h,  // last_block_h

                // sparsity args
                (std::uint32_t)0,  // sparsity_addr
            };

            if (fuse_op && fused_op_signaler->is_all_gather()) {
                fused_op_signaler->push_matmul_fused_op_rt_args(mm_in0_sender_args, false);
            }

            tt_metal::SetRuntimeArgs(
                program,
                mm_kernel_in0_mcast_cores_with_work_and_in_receiver_grid_id,
                core,
                mm_in0_sender_args);  // RISCV_0_default
        }
        // in0 receiver and in 1 sender
        else {
            std::vector<uint32_t> mm_in0_receiver_args = {
                // in0 mcast args
                (std::uint32_t)top_left_core_physical.x,  // in0_mcast_sender_noc_x
                (std::uint32_t)top_left_core_physical.y   // in0_mcast_sender_noc_y
            };
            tt_metal::SetRuntimeArgs(
                program, mm_kernel_in0_receiver_id, core, mm_in0_receiver_args);  // RISCV_1_default
        }
        if (i < num_cores_with_work) {
            std::vector<uint32_t> mm_in1_sender_writer_args = {
                // READER
                // in1 tensor args
                (std::uint32_t)in1_tensor.address(),
                (std::uint32_t)in1_tensor_start_tile_id_stride * output_idx_x,  // in1_tensor_start_tile_id
                // in1 mcast args
                (std::uint32_t)0,  // in1_mcast_dest_noc_start_x
                (std::uint32_t)0,  // in1_mcast_dest_noc_start_y
                (std::uint32_t)0,  // in1_mcast_dest_noc_end_x
                (std::uint32_t)0,  // in1_mcast_dest_noc_end_y

                // sparsity args
                (std::uint32_t)0,  // sparsity_addr

                // WRITER
                // out tensor args
                (std::uint32_t)out_tensor.address(),
                ((std::uint32_t)output_idx_x * per_core_N) +
                    (output_idx_y * per_core_M * N)  // out_tensor_start_tile_id
            };

            if (output_idx_x == num_blocks_x - 1) {
                // padding args (READER)
                mm_in1_sender_writer_args.push_back(last_out_block_w);

                // padding args (WRITER)
                mm_in1_sender_writer_args.push_back(in0_last_block_num_nonzero_subblocks_h);
                mm_in1_sender_writer_args.push_back(in0_last_subblock_of_last_block_h);
                mm_in1_sender_writer_args.push_back(in0_last_block_padded_block_tiles_h_skip);
                mm_in1_sender_writer_args.push_back(out_block_w / out_subblock_w);  // out_num_nonzero_subblocks_w
                mm_in1_sender_writer_args.push_back(last_block_num_nonzero_subblocks_w);
                mm_in1_sender_writer_args.push_back(last_subblock_of_last_block_w);
                mm_in1_sender_writer_args.push_back(last_block_padded_subblock_tiles_addr_skip);
                mm_in1_sender_writer_args.push_back(last_block_padded_block_tiles_w_skip);
            } else {
                // padding args (READER)
                mm_in1_sender_writer_args.push_back(out_block_w);

                // padding args (WRITER)
                mm_in1_sender_writer_args.push_back(in0_last_block_num_nonzero_subblocks_h);
                mm_in1_sender_writer_args.push_back(in0_last_subblock_of_last_block_h);
                mm_in1_sender_writer_args.push_back(in0_last_block_padded_block_tiles_h_skip);
                mm_in1_sender_writer_args.push_back(out_block_w / out_subblock_w);  // out_num_nonzero_subblocks_w
                mm_in1_sender_writer_args.push_back(out_block_w / out_subblock_w);
                mm_in1_sender_writer_args.push_back(out_subblock_w);
                mm_in1_sender_writer_args.push_back(0);
                mm_in1_sender_writer_args.push_back(0);
            }

            // Bias base address; patched on program-cache hit by override_mcast_in0_program_parameters (idx 18).
            mm_in1_sender_writer_args.push_back(
                bias_tensor.has_value() ? (std::uint32_t)bias_tensor->address() : 0);  // smuggled-rta-ok
            mm_in1_sender_writer_args.push_back(
                bias_tensor.has_value() ? (std::uint32_t)per_core_N * output_idx_x : 0);  // in3_tensor_start_tile_id
            if (!output_is_sharded) {
                if (output_idx_x == num_blocks_x - 1) {
                    mm_in1_sender_writer_args.push_back(last_out_num_blocks_w);
                } else {
                    mm_in1_sender_writer_args.push_back(out_num_blocks_x);
                }
            }

            if (fuse_op && fused_op_signaler->is_all_gather()) {
                fused_op_signaler->push_matmul_fused_op_rt_args(mm_in1_sender_writer_args, true);
            }

            tt_metal::SetRuntimeArgs(
                program, mm_kernel_in1_sender_writer_id, core, mm_in1_sender_writer_args);  // RISCV_0_default
        }
    }

    return MatmulMultiCoreReuseMcast1DProgramFactory::shared_variables_t{
        {mm_kernel_in0_mcast_cores_with_work_and_in_receiver_grid_id, mm_kernel_in1_sender_writer_id},
        {cb_src1, cb_src2, cb_src3, cb_output},
        false,
        start_core,
        cores,
        num_cores_with_work,
        ttnn::prim::Matmul1DType::MCAST_IN0};
}

MatmulMultiCoreReuseMcast1DProgramFactory::shared_variables_t process_mcast_in1_program_and_create_override_variables(
    tt_metal::Program& program,
    const ttnn::Tensor& a,
    tt_metal::IDevice* device,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode,
    bool packer_l1_acc,
    CoreCoord compute_with_storage_grid_size,
    ttnn::operations::compute_throttle_utils::ThrottleLevel throttle_level,
    uint32_t in0_B,
    uint32_t in1_B,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    bool bcast_batch,
    bool transpose_a,
    bool transpose_b,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t out_block_h,
    uint32_t out_block_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    std::optional<UnaryWithParam> fused_activation,
    const MeshTensor& in0_tensor,
    const MeshTensor& in1_tensor,
    ttsl::optional_reference<const MeshTensor> bias_tensor,
    const MeshTensor& out_tensor,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const tt::tt_metal::Tile& bias_tile,
    const tt::tt_metal::Tile& output_tile,
    tt::DataFormat in0_data_format,
    tt::DataFormat in1_data_format,
    tt::DataFormat bias_data_format,
    tt::DataFormat output_data_format,
    bool in0_is_sharded,
    bool output_is_sharded,
    bool untilize_out,
    bool row_broadcast_bias = true,
    CoreCoord sub_device_start_core = {0, 0}) {
    // currently only support transpose of the full tile
    bool in0_transpose_tile = in0_tile.get_transpose_of_faces() && in0_tile.get_transpose_within_face();
    bool in1_transpose_tile = in1_tile.get_transpose_of_faces() && in1_tile.get_transpose_within_face();

    bool fuse_op = false;

    uint32_t num_blocks = K / in0_block_w;
    // Only enable packer l1 accumulation when there are num_blocks > 2, otherwise
    // unnecessary overhead for reconfigs are added. Last iteration of l1 accumulation
    // does a spill and reload, so need more than 2 blocks to use l1 acc for packer
    // For bias, last iteration of l1 acc remains in intermediate buffer, does not spill and reload
    bool packer_l1_acc_en = packer_l1_acc && (((bias_tensor.has_value()) && num_blocks > 1) || (num_blocks > 2));

    // if fp32 enabled then we pack fp32 in l1, if not, then we pack fp16 in l1
    tt::DataFormat interm0_data_format = packer_l1_acc_en
                                             ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                                             : (fp32_dest_acc_en ? tt::DataFormat::Float32 : output_data_format);

    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    uint32_t bias_single_tile_size = bias_tile.get_tile_size(bias_data_format);
    uint32_t output_single_tile_size = output_tile.get_tile_size(output_data_format);
    uint32_t interm0_single_tile_size = output_tile.get_tile_size(interm0_data_format);

    bool do_not_inplace_interm0_out_CB = output_is_sharded && (per_core_M != out_block_h);

    uint32_t in0_block_h = out_block_h;
    uint32_t in1_block_w = out_block_w;
    uint32_t in0_num_blocks_y = per_core_M / out_block_h;
    uint32_t in1_num_blocks_x = per_core_N / out_block_w;
    uint32_t out_num_blocks_x = in1_num_blocks_x;
    uint32_t out_num_blocks_y = in0_num_blocks_y;

    uint32_t in0_block_tiles = in0_block_h * in0_block_w;
    uint32_t in0_CB_tiles = in0_block_tiles;

    bool reuse_in0_in_CB =
        ((in0_B == 1 && in1_B > 1) && !in0_is_sharded && !output_is_sharded && !bcast_batch &&
         !fused_activation.has_value());

    if (reuse_in0_in_CB) {
        in0_CB_tiles = per_core_M * num_blocks * in0_block_w;
    } else if (in0_is_sharded) {
        in0_CB_tiles = num_blocks * per_core_M * in0_block_w * in0_B;
    } else if (in0_B * num_blocks > 1) {
        in0_CB_tiles = in0_CB_tiles * 2;  // double buffer
    }
    // Tiles whose size is not a multiple of the DRAM alignment (e.g. bfp8 32x16 = 544B on
    // Blackhole's 64B alignment) are padded to it in DRAM. The interleaved reader copies tiles at
    // the padded stride, so the in0/in1/bias CBs must hold pages at the aligned stride and the
    // reader/unpacker walk tiles at the same stride. No-op when already aligned (all bf16 tiles,
    // 32-wide bfp8, Wormhole). Replaces the staging-CB workaround. Sharded CBs are backed by the
    // tensor buffer and keep their natural page size.
    const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();
    uint32_t in0_aligned_tile_size =
        in0_is_sharded ? in0_single_tile_size : tt::align(in0_single_tile_size, dram_alignment);
    uint32_t in1_aligned_tile_size = tt::align(in1_single_tile_size, dram_alignment);
    // Bias CB pages must be padded to the DRAM alignment so the reader's L1 write stride
    // matches the DRAM page stride (e.g. 64B on Blackhole for a 32B (1,16) bf16 bias tile).
    // Mirrors in0/in1 above and the dram_sharded factory. No-op on Wormhole and for
    // tiles already >= dram_alignment.
    uint32_t bias_aligned_tile_size = tt::align(bias_single_tile_size, dram_alignment);
    uint32_t in0_CB_size = in0_CB_tiles * in0_aligned_tile_size;

    const auto& a_shape_logical = operations::matmul::utilities::get_matmul_tensor_logical_shape(a, transpose_a);
    // When transpose_a is true, the K dimension maps to the row dimension of the raw tile,
    // which is already zero-padded during tile layout conversion. pad_last_ktile operates on
    // columns, so applying it would incorrectly zero valid data that becomes output rows
    // after the compute kernel transposes the tile.
    const auto in0_last_ktile_w = transpose_a ? 0 : a_shape_logical[-1] % in0_tile.get_width();
    const auto in0_last_ktile_h = transpose_a ? a_shape_logical[-1] % in0_tile.get_width() : 0;
    TT_FATAL(
        in0_last_ktile_w == 0 || in0_last_ktile_h == 0,
        "At most one of in0_last_ktile_w ({}) and in0_last_ktile_h ({}) can be non-zero",
        in0_last_ktile_w,
        in0_last_ktile_h);

    bool extract_shard_sub_blocks = false;
    uint32_t in0_shard_height_in_tiles = 0;
    uint32_t in0_shard_width_in_tiles = 0;
    if (in0_is_sharded) {
        in0_shard_height_in_tiles = in0_tensor.shard_spec()->shape[0] / in0_tile.get_height();
        in0_shard_width_in_tiles = in0_tensor.shard_spec()->shape[1] / in0_tile.get_width();
        // NOTE: Criteria for extract_shard_sub_blocks is different from mcast in0
        // In the reader kernel, always need to copy to cb0 even for height=1 shards since we may not always do mcast
        // In mcast in0 sharded reader kernel, this is handled by mcast with loopback src
        // For mcast in1, if we don't need to extract_shard_sub_blocks, set the sharded in0 cb to cb0
        // For mcast in0, sharded in0 cb is always cb2
        if (in0_shard_width_in_tiles / in0_block_w > 1) {
            extract_shard_sub_blocks = true;
        }
    }
    uint32_t in2_CB_tiles = in0_block_tiles;
    uint32_t in2_CB_size = in2_CB_tiles * in0_single_tile_size;

    uint32_t in1_block_tiles = out_block_w * in0_block_w;
    uint32_t in1_CB_tiles = in1_block_tiles;
    if (in1_B * num_blocks > 1) {
        in1_CB_tiles = in1_CB_tiles * 2;  // double buffer
    }

    uint32_t in1_CB_size = in1_CB_tiles * in1_aligned_tile_size;

    uint32_t out_block_tiles = out_block_h * out_block_w;
    uint32_t out_shard_tiles = per_core_M * per_core_N;
    uint32_t out_CB_tiles = out_block_tiles;  // No double buffer
    if (output_is_sharded) {
        out_CB_tiles = out_shard_tiles;
    }
    uint32_t out_CB_size = out_CB_tiles * output_single_tile_size;
    uint32_t interm0_CB_tiles = out_block_tiles;  // No double buffer
    uint32_t interm0_CB_size = interm0_CB_tiles * interm0_single_tile_size;

    uint32_t in3_block_tiles = out_block_w;
    uint32_t in3_CB_tiles = in3_block_tiles;  // No double buffer
    uint32_t in3_CB_size = in3_CB_tiles * bias_aligned_tile_size;

    CoreCoord start_core = sub_device_start_core;

    // The matmul region is the rectangle of size `compute_with_storage_grid_size`
    // anchored at `start_core`. Callers must ensure this rectangle lies entirely
    // within the active sub-device's worker cores (validated upstream).
    CoreRangeSet matmul_core_rect(CoreRange(
        start_core,
        CoreCoord(
            start_core.x + compute_with_storage_grid_size.x - 1, start_core.y + compute_with_storage_grid_size.y - 1)));

    uint32_t num_blocks_y = ((M - 1) / per_core_M) + 1;
    uint32_t num_blocks_x = ((N - 1) / per_core_N) + 1;
    uint32_t num_blocks_total = num_blocks_y * num_blocks_x;
    uint32_t num_cores = num_blocks_total;

    TT_FATAL(
        num_blocks_x == 1,
        "mcast_in1 requires N ({}) to fit within one per_core_N block ({}); got num_blocks_x={}",
        N,
        per_core_N,
        num_blocks_x);
    TT_FATAL(
        ((N - 1) / out_block_w) + 1 == out_num_blocks_x,
        "mcast_in1 requires the logical N tail to be in the final internal W block; got N={}, per_core_N={}, "
        "out_block_w={}",
        N,
        per_core_N,
        out_block_w);
    TT_FATAL(
        num_blocks_y != 1 || ((M - 1) / out_block_h) + 1 == out_num_blocks_y,
        "a single-Y mcast_in1 sender requires the logical M tail to be in the final internal H block; got M={}, "
        "per_core_M={}, out_block_h={}",
        M,
        per_core_M,
        out_block_h);
    TT_FATAL(
        num_blocks_y != 1 || M % out_block_h == 0 || out_num_blocks_y == 1,
        "a single-Y mcast_in1 sender supports a partial final H block only with one internal H block; got M={}, "
        "per_core_M={}, out_block_h={}",
        M,
        per_core_M,
        out_block_h);

    constexpr bool row_major = true;
    CoreRangeSet all_cores =
        tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(start_core, num_cores, matmul_core_rect, row_major);
    CoreRange in1_mcast_receiver_cores_bounding_box = all_cores.bounding_box();
    uint32_t in1_mcast_receiver_num_cores = in1_mcast_receiver_cores_bounding_box.size();  // always mcast to full grid

    CoreRange in1_mcast_sender(start_core, start_core);
    CoreRangeSet in1_mcast_receivers;
    if (in1_mcast_receiver_num_cores > 1) {
        // Check against the actual rectangle width (instead of bare grid_size.x-1) so that
        // sub-devices anchored away from (0, 0) wrap correctly.
        auto receiver_start_core = compute_with_storage_grid_size.x > 1 ? CoreCoord{start_core.x + 1, start_core.y}
                                                                        : CoreCoord{start_core.x, start_core.y + 1};
        in1_mcast_receivers = tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(
            receiver_start_core, num_cores - 1, matmul_core_rect, row_major);
    }

    // Mcast args
    auto in1_mcast_sender_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto in1_mcast_receiver_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);

    CoreCoord top_left_core = in1_mcast_receiver_cores_bounding_box.start_coord;
    CoreCoord bottom_right_core = in1_mcast_receiver_cores_bounding_box.end_coord;
    auto top_left_core_physical = device->worker_core_from_logical_core(top_left_core);
    auto bottom_right_core_physical = device->worker_core_from_logical_core(bottom_right_core);

    const auto& a_padded_shape = operations::matmul::utilities::get_matmul_tensor_padded_shape(a, transpose_a);
    const uint32_t M_per_batch = a_padded_shape[-2] / in0_tile.get_height();
    const auto [in0_tensor_stride_w, in0_tensor_stride_h] =
        operations::matmul::utilities::get_in0_transpose_strides(M, M_per_batch, transpose_a, K);
    const auto in0_tensor_next_block_stride = in0_block_w * in0_tensor_stride_w;
    const auto in0_tensor_next_h_dim_block_stride = in0_block_h * in0_tensor_stride_h;
    const auto in0_tensor_start_tile_id_stride = per_core_M * in0_tensor_stride_h;

    const auto in1_tensor_stride_w = transpose_b ? K : 1;
    const auto in1_tensor_stride_h = transpose_b ? 1 : N;
    const auto in1_tensor_next_block_stride = in0_block_w * in1_tensor_stride_h;
    const auto in1_tensor_next_w_dim_block_stride = in1_block_w * in1_tensor_stride_w;
    const auto in1_tensor_start_tile_id_stride = per_core_N * in1_tensor_stride_w;

    std::vector<uint32_t> in0_sender_compile_time_args = {
        // in0 tensor args
        (std::uint32_t)in0_tensor_stride_w,
        (std::uint32_t)in0_tensor_stride_h,
        (std::uint32_t)in0_tensor_next_block_stride,
        (std::uint32_t)in0_tensor_next_h_dim_block_stride,
        // in0 block args
        (std::uint32_t)in0_block_w,                // in0_block_w
        (std::uint32_t)in0_block_h,                // in0_block_h
        (std::uint32_t)in0_block_w * in0_block_h,  // in0_block_num_tiles
        (std::uint32_t)in0_last_ktile_w,
        (std::uint32_t)in0_last_ktile_h,

        (std::uint32_t)extract_shard_sub_blocks,
        (std::uint32_t)in0_shard_width_in_tiles,
        (std::uint32_t)in0_shard_height_in_tiles,
        // in0/in1 common args
        (std::uint32_t)num_blocks,        // num_blocks
        (std::uint32_t)out_num_blocks_x,  // out_num_blocks_x
        (std::uint32_t)out_num_blocks_y,  // out_num_blocks_y
        // in0 mcast args
        (std::uint32_t)0,
        (std::uint32_t)0,
        (std::uint32_t)0,  // in0_mcast_num_dests
        (std::uint32_t)0,  // in0_mcast_num_cores
        // batch args
        (std::uint32_t)M * K,            // MtKt
        (std::uint32_t)in0_B,            // batch
        (std::uint32_t)in1_B,            // batch
        (std::uint32_t)reuse_in0_in_CB,  // reuse_in0_in_CB
        // sparsity args
        (std::uint32_t)0,     // batchB
        (std::uint32_t)0,     // sparsity_pagesize (placeholder since sparsity not used in this case)
        (std::uint32_t)true,  // bcast_A
        (std::uint32_t)false  // get_batch_from_reader
    };
    in0_sender_compile_time_args.push_back((std::uint32_t)fuse_op);
    tt::tt_metal::TensorAccessorArgs(in0_tensor).append_to(in0_sender_compile_time_args);
    tt::tt_metal::TensorAccessorArgs().append_to(in0_sender_compile_time_args);  // placeholder for sparsity
    in0_sender_compile_time_args.push_back((std::uint32_t)0);  // num_batch_compute (unused, sparsity disabled)

    std::vector<uint32_t> in1_sender_writer_compile_time_args = {
        // READER
        // in1 tensor args
        (std::uint32_t)in1_tensor_stride_w,
        (std::uint32_t)in1_tensor_stride_h,
        (std::uint32_t)in1_tensor_next_block_stride,
        (std::uint32_t)in1_tensor_next_w_dim_block_stride,
        // in1 block args
        (std::uint32_t)in1_block_w,                // in1_block_w
        (std::uint32_t)in0_block_w,                // in1_block_h
        (std::uint32_t)in1_block_w * in0_block_w,  // in1_block_num_tiles
        // in0/in1 common args
        (std::uint32_t)num_blocks,        // num_blocks
        (std::uint32_t)out_num_blocks_x,  // out_num_blocks_x
        (std::uint32_t)out_num_blocks_y,  // out_num_blocks_y
        // in1 mcast args
        (std::uint32_t)in1_mcast_sender_semaphore_id,
        (std::uint32_t)in1_mcast_receiver_semaphore_id,
        (std::uint32_t)num_cores - 1,                     // in1_mcast_num_dests
        (std::uint32_t)in1_mcast_receiver_num_cores - 1,  // in1_mcast_num_cores
        // batch args
        (std::uint32_t)K * N,                              // KtNt
        (std::uint32_t)(reuse_in0_in_CB ? in1_B : in0_B),  // batch
        (std::uint32_t)bcast_batch,                        // bcast_B
        // sparsity args
        (std::uint32_t)0,  // batchB
        (std::uint32_t)0,  // sparsity_pagesize (placeholder since sparsity not used in this case)

        // WRITER
        // out tensor args
        (std::uint32_t)1,                   // out_tensor_stride_w
        (std::uint32_t)N,                   // out_tensor_stride_h
        (std::uint32_t)out_subblock_w,      // out_tensor_next_subblock_stride_w
        (std::uint32_t)out_subblock_h * N,  // out_tensor_next_subblock_stride_h
        (std::uint32_t)out_block_w,         // out_tensor_next_w_dim_block_stride
        (std::uint32_t)out_block_h * N,     // out_tensor_next_h_dim_block_stride
        // out subblock args
        (std::uint32_t)out_subblock_w,                     // out_subblock_w
        (std::uint32_t)out_subblock_h,                     // out_subblock_h
        (std::uint32_t)(out_subblock_w * out_subblock_h),  // out_subblocks_w * out_subblocks_h
        // batch args
        (std::uint32_t)M * N  // MtNt
    };

    if (bias_tensor.has_value()) {
        in1_sender_writer_compile_time_args.push_back((std::uint32_t)1);  // in3_tensor_stride_w
    } else {
        in1_sender_writer_compile_time_args.push_back(0);  // Placeholder; not used
    }

    in1_sender_writer_compile_time_args.push_back((std::uint32_t)fuse_op);
    in1_sender_writer_compile_time_args.push_back((std::uint32_t)fuse_op);
    in1_sender_writer_compile_time_args.push_back((std::uint32_t)false);  // compact_output

    // Append TensorAccessorArgs
    tt::tt_metal::TensorAccessorArgs(in1_tensor).append_to(in1_sender_writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs().append_to(in1_sender_writer_compile_time_args);  // placeholder for sparsity
    tt::tt_metal::TensorAccessorArgs(out_tensor).append_to(in1_sender_writer_compile_time_args);
    if (bias_tensor.has_value()) {
        tt::tt_metal::TensorAccessorArgs(*bias_tensor).append_to(in1_sender_writer_compile_time_args);
    }

    std::vector<uint32_t> in1_receiver_writer_compile_time_args = {
        // READER
        // in1 block args
        (std::uint32_t)in1_block_w * in0_block_w,  // in1_block_num_tiles
        // in0/in1 common args
        (std::uint32_t)num_blocks,        // num_blocks
        (std::uint32_t)out_num_blocks_x,  // out_num_blocks_x
        (std::uint32_t)out_num_blocks_y,  // out_num_blocks_y
        // in1 mcast args
        (std::uint32_t)in1_mcast_sender_semaphore_id,
        (std::uint32_t)in1_mcast_receiver_semaphore_id,
        // batch args
        (std::uint32_t)(reuse_in0_in_CB ? in1_B : in0_B),  // batch

        // WRITER
        // out tensor args
        (std::uint32_t)1,                   // out_tensor_stride_w
        (std::uint32_t)N,                   // out_tensor_stride_h
        (std::uint32_t)out_subblock_w,      // out_tensor_next_subblock_stride_w
        (std::uint32_t)out_subblock_h * N,  // out_tensor_next_subblock_stride_h
        (std::uint32_t)out_block_w,         // out_tensor_next_w_dim_block_stride
        (std::uint32_t)out_block_h * N,     // out_tensor_next_h_dim_block_stride
        // out subblock args
        (std::uint32_t)out_subblock_w,                     // out_subblock_w
        (std::uint32_t)out_subblock_h,                     // out_subblock_h
        (std::uint32_t)(out_subblock_w * out_subblock_h),  // out_subblocks_w * out_subblocks_h
        // batch args
        (std::uint32_t)M * N  // MtNt
    };

    if (bias_tensor.has_value()) {
        in1_receiver_writer_compile_time_args.push_back((std::uint32_t)in1_block_w);
    } else {
        in1_receiver_writer_compile_time_args.push_back(0);  // Placeholder; not used
    }
    in1_receiver_writer_compile_time_args.push_back((std::uint32_t)fuse_op);
    tt::tt_metal::TensorAccessorArgs(out_tensor).append_to(in1_receiver_writer_compile_time_args);

    std::map<std::string, std::string> mm_kernel_defines;
    std::map<std::string, std::string> mm_kernel_in0_sender_defines;
    std::map<std::string, std::string> mm_kernel_in1_sender_writer_defines;
    std::map<std::string, std::string> mm_kernel_in1_receiver_writer_defines;
    if (bias_tensor.has_value()) {
        mm_kernel_defines["FUSE_BIAS"] = "1";
        mm_kernel_in1_sender_writer_defines["FUSE_BIAS"] = "1";
        mm_kernel_in1_receiver_writer_defines["FUSE_BIAS"] = "1";
    }
    if (fused_activation.has_value()) {
        if (fused_activation.value().op_type == UnaryOpType::RELU) {
            mm_kernel_defines["PACK_RELU"] = "1";
        } else {
            mm_kernel_defines["SFPU_ACTIVATION"] = "1";
        }
    }
    if (packer_l1_acc_en) {
        mm_kernel_defines["PACKER_L1_ACC"] = "1";
    }
    if (fp32_dest_acc_en) {
        mm_kernel_defines["FP32_DEST_ACC_EN"] = "1";
    }
    if (in1_transpose_tile) {
        mm_kernel_defines["IN1_TRANSPOSE_TILE"] = "1";
    }

    ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
        device->arch(), num_cores, mm_kernel_defines);
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(), num_cores, mm_kernel_defines, throttle_level);

    if (in0_is_sharded) {
        mm_kernel_in0_sender_defines["IN0_SHARDED"] = "1";
    }
    if (output_is_sharded) {
        mm_kernel_in1_sender_writer_defines["OUT_SHARDED"] = "1";
        mm_kernel_in1_receiver_writer_defines["OUT_SHARDED"] = "1";
    }

    mm_kernel_in0_sender_defines["SKIP_MCAST"] = "1";

    if (in1_mcast_receiver_num_cores == 1) {
        mm_kernel_in1_sender_writer_defines["SKIP_MCAST"] = "1";
    }

    // Intermediate CB read
    /*
    Blackhole architecture alignment issue workaround for tiny tiles:

    Problem: When reading tiny tiles from DRAM to circular buffers (CB), address alignment
    issues occur. DRAM tile addresses are 64-byte aligned within each block, but L1 CB
    addresses are not necessarily aligned due to non-64-byte-aligned page sizes.

    Example scenario:
    - Two consecutive 544-byte tiles (16x32 tile of dtype bfloat8_b) stored on different DRAM banks
    - CB configured with size=2 to hold both tiles

    Result:
    - Tile 0: DRAM Bank 0, Address 64    → CB L1 Address 0   (64-byte aligned ✓)
    - Tile 1: DRAM Bank 1, Address 64    → CB L1 Address 544 (not 64-byte aligned ✗)

    Solution: Use an intermediate single-tile CB as a staging area. Read each tile into
    the intermediate CB first, then copy to the destination CB. This ensures proper
    alignment at the cost of additional memory bandwidth overhead.

    Note: This workaround should only be used for this specific alignment issue case.
    */

    // in1 is the reader of weights/output writer, and we choose to make it use the optimized reader noc
    tt_metal::NOC in0_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    tt_metal::NOC in1_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());

    auto mm_kernel_in0_sender_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = in0_noc,
            .compile_args = in0_sender_compile_time_args,
            .defines = mm_kernel_in0_sender_defines,
            .named_compile_args = {
                {"cb_in0", tt::CBIndex::c_0},
                {"cb_in0_sharded", tt::CBIndex::c_2},
                {"cb_sparsity", tt::CBIndex::c_6},
                {"num_active", 0},  // indexed/gather mode: sparse_matmul only (0 = disabled)
            }});

    auto mm_kernel_in1_sender_writer_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp",
        in1_mcast_sender,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = in1_noc,
            .compile_args = in1_sender_writer_compile_time_args,
            .defines = mm_kernel_in1_sender_writer_defines,
            .named_compile_args = {
                {"cb_in1", tt::CBIndex::c_1},
                {"cb_bias", tt::CBIndex::c_3},
                {"cb_out", tt::CBIndex::c_4},
                {"cb_sparsity", tt::CBIndex::c_7},
                {"num_active", 0},  // indexed/gather mode: sparse_matmul only (0 = disabled)
            }});

    tt::tt_metal::KernelHandle mm_kernel_in1_receiver_writer_id = 0;
    if (in1_mcast_receivers.num_cores() > 0) {
        mm_kernel_in1_receiver_writer_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
            "reader_bmm_tile_layout_in1_receiver_writer_padding.cpp",
            in1_mcast_receivers,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = in1_noc,
                .compile_args = in1_receiver_writer_compile_time_args,
                .defines = mm_kernel_in1_receiver_writer_defines,
                .named_compile_args = {
                    {"cb_in1", tt::CBIndex::c_1},
                    {"cb_bias", tt::CBIndex::c_3},
                    {"cb_out", tt::CBIndex::c_4},
                }});
    }

    // Compute kernel compile time args

    uint32_t in0_num_subblocks = (out_block_h / out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;

    uint32_t in1_num_subblocks = (out_block_w / out_subblock_w);
    uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks;
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    std::vector<uint32_t> compute_kernel_args = {
        in0_block_w,             // in0_block_w
        in0_num_subblocks,       // in0_num_subblocks
        in0_block_num_tiles,     // in0_block_num_tiles
        in0_subblock_num_tiles,  // in0_subblock_num_tiles

        in1_num_subblocks,    // in1_num_subblocks
        in1_block_num_tiles,  // in1_block_num_tiles
        in1_per_core_w,       // in1_per_core_w

        num_blocks,        // num_blocks
        out_num_blocks_x,  // out_num_blocks_x
        out_num_blocks_y,  // out_num_blocks_y

        out_subblock_h,                     // out_subblock_h
        out_subblock_w,                     // out_subblock_w
        out_subblock_num_tiles,             // out_subblock_num_tiles
        (reuse_in0_in_CB ? in1_B : in0_B),  // batch
        out_block_tiles,                    // out_block_num_tiles

        untilize_out,  // untilize_out
        false,         // get_batch_from_reader
        in0_transpose_tile,
    };
    if (bias_tensor.has_value()) {
        compute_kernel_args.push_back(row_broadcast_bias ? 1u : 0u);
    }

    // Setup named compile args
    std::unordered_map<std::string, uint32_t> compute_named_compile_args = {
        {"cb_in0", tt::CBIndex::c_0},
        {"cb_in1", tt::CBIndex::c_1},
        {"cb_bias", tt::CBIndex::c_3},
        {"cb_out", tt::CBIndex::c_4},
        {"cb_intermed0", tt::CBIndex::c_5},
        {"cb_in0_transposed", tt::CBIndex::c_10},
        {"bias_ntiles", in1_per_core_w},
    };

    // Add activation type if needed
    if (fused_activation.has_value() && fused_activation.value().op_type != UnaryOpType::RELU) {
        using ttnn::operations::matmul::utilities::get_activation_params;
        const auto& activation = fused_activation.value();
        const auto params = get_activation_params(activation);
        compute_named_compile_args["activation_type"] = static_cast<uint32_t>(params.type);
        compute_named_compile_args["activation_param0"] = params.param0;
        compute_named_compile_args["activation_param1"] = params.param1;
        compute_named_compile_args["activation_param2"] = params.param2;
    }

    // The fused bias add reads the partials CB as an FPU operand (SrcA), so UnpackToDestFp32 cannot be
    // set on cb_intermed0 directly when bias is present. In that case the cross-block reload instead
    // copies through cb_intermed0_alias, a second buffer index over the same SRAM carrying
    // UnpackToDestFp32, while the bias add keeps reading cb_intermed0 via SrcA. The alias index is
    // handed to the compute kernel through the MM_PARTIALS_RELOAD_ALIAS_CB define, which selects the
    // alias reload path there. Without bias the reload reads cb_intermed0 and the flag is set on it.
    constexpr auto cb_intermed0 = tt::CBIndex::c_5;
    constexpr auto cb_intermed0_alias = tt::CBIndex::c_7;
    const bool bias_reload_alias =
        fp32_dest_acc_en && interm0_data_format == tt::DataFormat::Float32 && bias_tensor.has_value();
    if (bias_reload_alias) {
        mm_kernel_defines["MM_PARTIALS_RELOAD_ALIAS_CB"] = std::to_string(static_cast<uint32_t>(cb_intermed0_alias));
    }
    // When accumulating in fp32 (fp32_dest_acc_en) with the K reduction split across blocks, the
    // intermediate partials CB (cb_intermed0) holds Float32 and is reloaded into DEST between blocks
    // by copy_block_matmul_partials. Unless the reload's CB view is marked UnpackToDestFp32, that
    // reload is routed through SrcA and rounded to TF32 (10 mantissa bits), so the fp32 partial loses
    // precision on every block boundary. The flag is set on the alias when bias forces a separate SrcA
    // view of cb_intermed0 (see above), else on cb_intermed0 itself.
    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    if (fp32_dest_acc_en && interm0_data_format == tt::DataFormat::Float32) {
        const uint32_t cb_to_mark =
            bias_reload_alias ? static_cast<uint32_t>(cb_intermed0_alias) : static_cast<uint32_t>(cb_intermed0);
        unpack_to_dest_mode[cb_to_mark] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }

    // Create compute kernel
    // bool fp32_dest_acc_en = false;
    // Gelu currently has better accuracy when run in approx mode
    // bool math_approx_mode = false;
    tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp",
        all_cores,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args,
            .defines = mm_kernel_defines,
            .named_compile_args = compute_named_compile_args});

    // Create circular buffers
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    tt_metal::CircularBufferConfig src0_cb_config =
        tt_metal::CircularBufferConfig(in0_CB_size, {{src0_cb_index, in0_data_format}})
            .set_page_size(src0_cb_index, in0_aligned_tile_size)
            .set_tile_dims(src0_cb_index, in0_tile);
    if (in0_is_sharded and not extract_shard_sub_blocks) {
        src0_cb_config = src0_cb_config.set_globally_allocated_address(in0_tensor);
    }
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, src0_cb_config);
    log_debug(
        LogOp,
        "CB {} :: PS = {}, NP = {}, TOTAL = {}",
        src0_cb_index,
        in0_single_tile_size,
        in0_CB_size / in0_single_tile_size,
        in0_CB_size);

    uint32_t src2_cb_index = tt::CBIndex::c_2;
    tt::tt_metal::CBHandle cb_src2 = 0;
    if (in0_is_sharded and extract_shard_sub_blocks) {  // in0_is_sharded is technically redundant
        tt_metal::CircularBufferConfig src2_cb_config =
            tt_metal::CircularBufferConfig(in2_CB_size, {{src2_cb_index, in0_data_format}})
                .set_page_size(src2_cb_index, in0_single_tile_size)
                .set_globally_allocated_address(in0_tensor)
                .set_tile_dims(src2_cb_index, in0_tile);
        cb_src2 = tt_metal::CreateCircularBuffer(program, all_cores, src2_cb_config);
        log_debug(
            LogOp,
            "CB {} :: PS = {}, NP = {}, TOTAL = {}",
            src2_cb_index,
            in0_single_tile_size,
            in2_CB_size / in0_single_tile_size,
            in2_CB_size);
    }

    uint32_t src1_cb_index = tt::CBIndex::c_1;
    tt_metal::CircularBufferConfig src1_cb_config =
        tt_metal::CircularBufferConfig(in1_CB_size, {{src1_cb_index, in1_data_format}})
            .set_page_size(src1_cb_index, in1_aligned_tile_size)
            .set_tile_dims(src1_cb_index, in1_tile);
    tt_metal::CreateCircularBuffer(program, all_cores, src1_cb_config);
    log_debug(
        LogOp,
        "CB {} :: PS = {}, NP = {}, TOTAL = {}",
        src1_cb_index,
        in1_single_tile_size,
        in1_CB_size / in1_single_tile_size,
        in1_CB_size);

    uint32_t output_cb_index = tt::CBIndex::c_4;
    uint32_t interm0_cb_index = tt::CBIndex::c_5;
    tt_metal::CircularBufferConfig interm0_cb_config =
        tt_metal::CircularBufferConfig(0, {{interm0_cb_index, interm0_data_format}});
    tt_metal::CircularBufferConfig output_cb_config =
        tt_metal::CircularBufferConfig(0, {{output_cb_index, output_data_format}});

    if (do_not_inplace_interm0_out_CB || (interm0_data_format != output_data_format) ||
        (untilize_out && (in1_num_subblocks > 1))) {
        // output
        std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
            {output_cb_index, output_data_format},
        };
        output_cb_config = tt_metal::CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
                               .set_page_size(output_cb_index, output_single_tile_size)
                               .set_tile_dims(output_cb_index, output_tile);
        // interm0
        std::map<uint8_t, tt::DataFormat> interm0_cb_data_format_spec{
            {interm0_cb_index, interm0_data_format},
        };
        // Alias over the same SRAM, marked UnpackToDestFp32, for the bias reload (see above).
        if (bias_reload_alias) {
            interm0_cb_data_format_spec[static_cast<uint8_t>(cb_intermed0_alias)] = interm0_data_format;
        }
        interm0_cb_config = tt_metal::CircularBufferConfig(interm0_CB_size, interm0_cb_data_format_spec)
                                .set_page_size(interm0_cb_index, interm0_single_tile_size)
                                .set_tile_dims(interm0_cb_index, output_tile);
        if (bias_reload_alias) {
            interm0_cb_config =
                interm0_cb_config.set_page_size(static_cast<uint8_t>(cb_intermed0_alias), interm0_single_tile_size)
                    .set_tile_dims(static_cast<uint8_t>(cb_intermed0_alias), output_tile);
        }

        tt_metal::CreateCircularBuffer(program, CoreRangeSet({all_cores}), interm0_cb_config);
        log_debug(
            LogOp,
            "CB {} :: PS = {}, NP = {}, TOTAL = {}",
            interm0_cb_index,
            interm0_single_tile_size,
            interm0_CB_size / interm0_single_tile_size,
            interm0_CB_size);
    } else {
        // share buffer
        std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
            {output_cb_index, output_data_format}, {interm0_cb_index, interm0_data_format}};
        // Alias over the same SRAM, marked UnpackToDestFp32, for the bias reload (see above).
        if (bias_reload_alias) {
            output_cb_data_format_spec[static_cast<uint8_t>(cb_intermed0_alias)] = interm0_data_format;
        }
        output_cb_config = tt_metal::CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
                               .set_page_size(output_cb_index, output_single_tile_size)
                               .set_page_size(interm0_cb_index, interm0_single_tile_size)
                               .set_tile_dims(output_cb_index, output_tile)
                               .set_tile_dims(interm0_cb_index, output_tile);
        if (bias_reload_alias) {
            output_cb_config =
                output_cb_config.set_page_size(static_cast<uint8_t>(cb_intermed0_alias), interm0_single_tile_size)
                    .set_tile_dims(static_cast<uint8_t>(cb_intermed0_alias), output_tile);
        }
    }

    if (output_is_sharded) {
        output_cb_config = output_cb_config.set_globally_allocated_address(out_tensor);
    }
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);
    log_debug(
        LogOp,
        "CB {} :: PS = {}, NP = {}, TOTAL = {}",
        output_cb_index,
        output_single_tile_size,
        out_CB_size / output_single_tile_size,
        out_CB_size);

    if (bias_tensor.has_value()) {
        uint32_t src3_cb_index = tt::CBIndex::c_3;
        tt_metal::CircularBufferConfig cb_src3_config =
            tt_metal::CircularBufferConfig(in3_CB_size, {{src3_cb_index, bias_data_format}})
                .set_page_size(src3_cb_index, bias_aligned_tile_size)
                .set_tile_dims(src3_cb_index, bias_tile);
        tt_metal::CreateCircularBuffer(program, all_cores, cb_src3_config);
        log_debug(
            LogOp,
            "CB {} :: PS = {}, NP = {}, TOTAL = {}",
            src3_cb_index,
            bias_aligned_tile_size,
            in3_CB_size / bias_aligned_tile_size,
            in3_CB_size);
    }

    // Intermediate CB read

    if (in0_transpose_tile) {
        const uint32_t in0_transpose_cb_index = tt::CBIndex::c_10;
        auto in0_transpose_cb_config =
            tt_metal::CircularBufferConfig(in0_CB_size, {{in0_transpose_cb_index, in0_data_format}})
                .set_page_size(in0_transpose_cb_index, in0_aligned_tile_size)
                .set_tile_dims(in0_transpose_cb_index, in0_tile);
        tt_metal::CreateCircularBuffer(program, all_cores, in0_transpose_cb_config);
    }

    // Parameters for last row, col, or block
    uint32_t last_per_core_M = M % per_core_M == 0 ? per_core_M : M % per_core_M;
    uint32_t last_out_block_h = last_per_core_M % out_block_h == 0 ? out_block_h : last_per_core_M % out_block_h;
    uint32_t last_out_num_blocks_h = ((last_per_core_M - 1) / out_block_h) + 1;
    uint32_t last_block_num_nonzero_subblocks_h = ((last_out_block_h - 1) / out_subblock_h) + 1;
    uint32_t last_subblock_of_last_block_h =
        last_out_block_h % out_subblock_h == 0 ? out_subblock_h : last_out_block_h % out_subblock_h;
    uint32_t last_block_padded_block_tiles_h_skip =
        (out_block_h / out_subblock_h - last_block_num_nonzero_subblocks_h) * (out_block_w * out_subblock_h);

    // W-dim padding parameters for the last block in X. Mirrors the mcast_in0
    // factory (see `process_mcast_in0_program_and_create_override_variables`).
    // Without these, the receiver-writer emits full per_core_N-wide writes for
    // the last X block even when N is not divisible by per_core_N, sending
    // pages past the tensor's logical extent and producing OOB writes past the
    // L1 allocation on the far banks.
    uint32_t last_per_core_N = N % per_core_N == 0 ? per_core_N : N % per_core_N;
    uint32_t last_out_block_w = last_per_core_N % out_block_w == 0 ? out_block_w : last_per_core_N % out_block_w;
    uint32_t last_out_num_blocks_w = ((last_per_core_N - 1) / out_block_w) + 1;
    uint32_t last_block_num_nonzero_subblocks_w = ((last_out_block_w - 1) / out_subblock_w) + 1;
    uint32_t last_subblock_of_last_block_w =
        last_out_block_w % out_subblock_w == 0 ? out_subblock_w : last_out_block_w % out_subblock_w;
    uint32_t last_block_padded_subblock_tiles_addr_skip =
        output_single_tile_size * (out_subblock_w - last_subblock_of_last_block_w);
    uint32_t last_block_padded_block_tiles_w_skip =
        (out_subblock_w * out_subblock_h) * (out_block_w / out_subblock_w - last_block_num_nonzero_subblocks_w);

    CoreCoord start_core_noc = bottom_right_core_physical;
    CoreCoord end_core_noc = top_left_core_physical;
    if (in1_noc == tt::tt_metal::NOC::NOC_0) {
        std::swap(start_core_noc, end_core_noc);
    }

    const auto& cores = corerange_to_cores(all_cores, std::nullopt, row_major);
    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& core = cores[i];
        uint32_t output_idx_x = i / num_blocks_y;
        uint32_t output_idx_y = i % num_blocks_y;

        // in0 sender and in1 sender
        if (core == start_core) {
            // The sender can independently be the last block in either dimension.
            bool last_y = (output_idx_y == num_blocks_y - 1);
            bool last_x = (output_idx_x == num_blocks_x - 1);
            std::vector<uint32_t> mm_in1_sender_writer_args = {
                // READER
                // in1 tensor args
                (std::uint32_t)in1_tensor.address(),
                (std::uint32_t)in1_tensor_start_tile_id_stride * output_idx_x,  // in1_tensor_start_tile_id
                // in1 mcast args
                (std::uint32_t)start_core_noc.x,  // in1_mcast_dest_noc_start_x
                (std::uint32_t)start_core_noc.y,  // in1_mcast_dest_noc_start_y
                (std::uint32_t)end_core_noc.x,    // in1_mcast_dest_noc_end_x
                (std::uint32_t)end_core_noc.y,    // in1_mcast_dest_noc_end_y

                // sparsity args
                (std::uint32_t)0,  // sparsity_addr

                // WRITER
                // out tensor args
                (std::uint32_t)out_tensor.address(),
                ((std::uint32_t)output_idx_x * per_core_N) +
                    (output_idx_y * per_core_M * N),  // out_tensor_start_tile_id

                // padding args (READER)
                (std::uint32_t)(last_x ? last_out_block_w : out_block_w),  // last_block_w
                // padding args (WRITER)
                (std::uint32_t)(last_y ? last_block_num_nonzero_subblocks_h : out_block_h / out_subblock_h),
                (std::uint32_t)(last_y ? last_subblock_of_last_block_h : out_subblock_h),
                (std::uint32_t)(last_y ? last_block_padded_block_tiles_h_skip : 0),
                (std::uint32_t)out_block_w / out_subblock_w,
                (std::uint32_t)(last_x ? last_block_num_nonzero_subblocks_w : out_block_w / out_subblock_w),
                (std::uint32_t)(last_x ? last_subblock_of_last_block_w : out_subblock_w),
                (std::uint32_t)(last_x ? last_block_padded_subblock_tiles_addr_skip : 0),
                (std::uint32_t)(last_x ? last_block_padded_block_tiles_w_skip : 0)};

            if (bias_tensor.has_value()) {
                // Bias base address; patched on program-cache hit by override_mcast_in1_program_parameters (idx 18).
                mm_in1_sender_writer_args.push_back((std::uint32_t)bias_tensor->address());  // smuggled-rta-ok
                mm_in1_sender_writer_args.push_back(
                    (std::uint32_t)per_core_N * output_idx_x);  // in3_tensor_start_tile_id
            } else {
                mm_in1_sender_writer_args.push_back(0);
                mm_in1_sender_writer_args.push_back(0);
            }
            if (!output_is_sharded) {
                mm_in1_sender_writer_args.push_back(last_x ? last_out_num_blocks_w : out_num_blocks_x);
            }

            tt_metal::SetRuntimeArgs(
                program, mm_kernel_in1_sender_writer_id, core, mm_in1_sender_writer_args);  // RISCV_1_default
        }
        // in0 sender and in1 receiver
        else {
            std::vector<uint32_t> mm_in1_receiver_writer_args = {
                // READER
                // in1 mcast args
                (std::uint32_t)top_left_core_physical.x,  // in1_mcast_sender_noc_x
                (std::uint32_t)top_left_core_physical.y,  // in1_mcast_sender_noc_y

                // WRITER
                // out tensor args
                (std::uint32_t)out_tensor.address(),  // out_tensor_addr
                ((std::uint32_t)output_idx_x * per_core_N) +
                    (output_idx_y * per_core_M * N)  // out_tensor_start_tile_id
            };

            {
                // padding args (WRITER): H-dim tail depends on output_idx_y ==
                // num_blocks_y - 1; W-dim tail depends on output_idx_x ==
                // num_blocks_x - 1. The two dimensions are independent.
                bool last_y = (output_idx_y == num_blocks_y - 1);
                bool last_x = (output_idx_x == num_blocks_x - 1);
                mm_in1_receiver_writer_args.push_back(out_block_h / out_subblock_h);
                mm_in1_receiver_writer_args.push_back(
                    last_y ? last_block_num_nonzero_subblocks_h : out_block_h / out_subblock_h);
                mm_in1_receiver_writer_args.push_back(last_y ? last_subblock_of_last_block_h : out_subblock_h);
                mm_in1_receiver_writer_args.push_back(last_y ? last_block_padded_block_tiles_h_skip : 0);
                mm_in1_receiver_writer_args.push_back(out_block_w / out_subblock_w);
                mm_in1_receiver_writer_args.push_back(
                    last_x ? last_block_num_nonzero_subblocks_w : out_block_w / out_subblock_w);
                mm_in1_receiver_writer_args.push_back(last_x ? last_subblock_of_last_block_w : out_subblock_w);
                mm_in1_receiver_writer_args.push_back(last_x ? last_block_padded_subblock_tiles_addr_skip : 0);
                mm_in1_receiver_writer_args.push_back(last_x ? last_block_padded_block_tiles_w_skip : 0);
            }
            if (!output_is_sharded) {
                mm_in1_receiver_writer_args.push_back(
                    output_idx_y == num_blocks_y - 1 ? last_out_num_blocks_h : out_num_blocks_y);
                mm_in1_receiver_writer_args.push_back(
                    output_idx_x == num_blocks_x - 1 ? last_out_num_blocks_w : out_num_blocks_x);
            }

            tt_metal::SetRuntimeArgs(
                program,
                mm_kernel_in1_receiver_writer_id,
                core,
                mm_in1_receiver_writer_args);  // RISCV_0_default
        }
        std::vector<uint32_t> mm_in0_sender_args = {
            // in0 tensor args
            (std::uint32_t)in0_tensor.address(),
            (std::uint32_t)in0_tensor_start_tile_id_stride * output_idx_y,  // in0_tensor_start_tile_id
            // in0 mcast args
            (std::uint32_t)0,  // in0_mcast_dest_noc_start_x
            (std::uint32_t)0,  // in0_mcast_dest_noc_start_y
            (std::uint32_t)0,  // in0_mcast_dest_noc_end_x
            (std::uint32_t)0,  // in0_mcast_dest_noc_end_y

            // padding args
            (std::uint32_t)per_core_M,  // last_block_h

            // sparsity args
            (std::uint32_t)0,  // sparsity_addr
        };
        tt_metal::SetRuntimeArgs(program, mm_kernel_in0_sender_id, core, mm_in0_sender_args);  // RISCV_1_default
    }
    return MatmulMultiCoreReuseMcast1DProgramFactory::shared_variables_t{
        {mm_kernel_in0_sender_id, mm_kernel_in1_sender_writer_id, mm_kernel_in1_receiver_writer_id},
        {cb_src0, cb_src2, cb_output},
        extract_shard_sub_blocks,
        start_core,
        cores,
        0,
        ttnn::prim::Matmul1DType::MCAST_IN1};
}

enum class CORE_TYPE : uint32_t { IDLE_CORE = 0, WORKER_CORE = 1, HOP_CORE = 2 };

MatmulMultiCoreReuseMcast1DProgramFactory::shared_variables_t process_gather_in0_program_and_create_override_variables(
    tt_metal::Program& program,
    const ttnn::Tensor& a,
    const std::vector<ttnn::Tensor>& b_tensors,
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
    auto subdevice_cores = device->worker_cores(
        tt::tt_metal::HalProgrammableCoreType::TENSIX,
        sub_device_id.has_value() ? *sub_device_id : device->get_sub_device_ids().at(0));
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
            .set_globally_allocated_address(in0_tensor);
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
            src1_cb_config = src1_cb_config.set_globally_allocated_address(in1_tensor);
        }
        cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, src1_cb_config);
    }

    uint32_t src2_cb_index = base_cb_index + 2;
    tt_metal::CircularBufferConfig src2_cb_config =
        tt_metal::CircularBufferConfig(in2_CB_size, {{src2_cb_index, in0_data_format}})
            .set_page_size(src2_cb_index, in2_single_tile_size)
            .set_tile_dims(src2_cb_index, in0_tile);
    tt_metal::CreateCircularBuffer(program, all_cores, src2_cb_config);

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
                                   .set_globally_allocated_address(out_buffer);
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
                                   .set_globally_allocated_address(out_buffer);
            auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);
            cb_outputs.push_back(cb_output);
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
    if (fused_op_signaler.has_value()) {
        ttnn::experimental::ccl::MatmulFusedOpSignaler& signaler = fused_op_signaler.value();
        signaler.init_llama_rs_cores_mm(all_cores, program, device, 0);
    }
    /* Create the kernels */
    auto mm_kernel_in0_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_ring_all_gather.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = in0_noc,
            .noc_mode = noc_mode,
            .compile_args = in0_sender_compile_time_args,
            .named_compile_args = {{"cb_in0", src0_cb_index}, {"cb_in2", src2_cb_index}}});
    // Each core needs to signal to all RS cores, need to get a count of how many cores are in all_cores
    auto mm_kernel_in1_sender_writer_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_ring_all_gather.cpp",
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

    auto mm_kernel = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp",
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
            tt_metal::SetRuntimeArgs(program, mm_kernel_in0_id, core, mm_kernel_in0_args);

            // in1
            std::vector<uint32_t> mm_kernel_in1_sender_writer_args;
            mm_kernel_in1_sender_writer_args.push_back((std::uint32_t)core_type);
            if (fused_op_signaler.has_value()) {
                ttnn::experimental::ccl::MatmulFusedOpSignaler& signaler = fused_op_signaler.value();
                signaler.push_llama_rs_rt_args_for_mm(mm_kernel_in1_sender_writer_args, core, in1_noc, device);
            }

            tt_metal::SetRuntimeArgs(program, mm_kernel_in1_sender_writer_id, core, mm_kernel_in1_sender_writer_args);

            // compute
            std::vector<uint32_t> mm_kernel_args;
            mm_kernel_args.push_back((std::uint32_t)core_type);
            tt_metal::SetRuntimeArgs(program, mm_kernel, core, mm_kernel_args);
        }
    }

    /* Runtime args */
    // Mapping from worker core y-coordinate (and column group) to DRAM bank IDs.
    // The DRAM banks are split into two column groups (left and right halves of the chip).
    // On Wormhole: hardcoded mapping; banks 0-3 left column (x <= 3), banks 4-11 right column.
    // On Blackhole: dynamically derived from optimal DRAM bank API; banks split at x <= 6.
    std::map<uint32_t, uint32_t> worker_y_to_dram_bank_first_col;
    std::map<uint32_t, uint32_t> worker_y_to_dram_bank_second_col;
    uint32_t first_col_max_x = device->arch() == tt::ARCH::WORMHOLE_B0 ? 3 : 7;
    uint32_t num_receiver_cores_per_dram = 2;  // default to 2 for wormhole b0 and blackhole
    if (in1_is_dram_sharded) {
        num_receiver_cores_per_dram = ring_size / in1_tensor.shard_spec()->grid.num_cores();
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
    bank_ids.reserve(num_cores);
    for (uint32_t i = 0; i < num_cores; ++i) {
        bool send_to_hop_core = i == 0 && use_hop_cores;
        const auto& core = worker_cores_vec[i];
        const auto& core_noc = device->worker_core_from_logical_core(core);

        /* in0 */
        auto core_type = CORE_TYPE::WORKER_CORE;  // worker core
        CoreCoord next_core;
        if (send_to_hop_core) {
            next_core = hop_cores_vec[0];  // Send to first hop core
        } else {
            uint32_t next_i = i == 0 ? num_cores - 1 : i - 1;
            next_core = worker_cores_vec[next_i % num_cores];
        }
        const auto& next_core_noc = device->worker_core_from_logical_core(next_core);
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
        tt_metal::SetRuntimeArgs(program, mm_kernel_in0_id, core, mm_in0_args);

        /* in1 */
        std::vector<uint32_t> mm_in1_args = {
            (std::uint32_t)core_type,
            in1_tensor.address(),  // in1_tensor_addr
            i,                     // ring_idx
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
        if (fused_op_signaler.has_value()) {
            ttnn::experimental::ccl::MatmulFusedOpSignaler& signaler = fused_op_signaler.value();
            signaler.push_llama_rs_rt_args_for_mm(mm_in1_args, core, in1_noc, device);
        }
        tt_metal::SetRuntimeArgs(program, mm_kernel_in1_sender_writer_id, core, mm_in1_args);

        /* compute */
        std::vector<uint32_t> mm_kernel_compute_args = {
            (std::uint32_t)core_type,
            i,  // ring_idx
        };
        mm_kernel_compute_args.insert(
            mm_kernel_compute_args.end(),
            unpadded_in0_shard_widths_in_tiles.begin(),
            unpadded_in0_shard_widths_in_tiles.end());

        tt_metal::SetRuntimeArgs(program, mm_kernel, core, mm_kernel_compute_args);
    }

    // Runtime args for hop cores
    for (uint32_t i = 0; i < num_hop_cores; ++i) {
        bool end_of_hop = i == num_hop_cores - 1;

        auto core_type = CORE_TYPE::HOP_CORE;  // hop core
        const auto& core = hop_cores_vec[i];
        const auto& core_noc = device->worker_core_from_logical_core(core);

        /* in0 */
        CoreCoord next_core = end_of_hop ? worker_cores_vec[num_cores - 1] : hop_cores_vec[i + 1];
        const auto& next_core_noc = device->worker_core_from_logical_core(next_core);
        uint32_t noc = get_preferred_noc(core_noc, next_core_noc, device, use_dedicated_noc);

        std::vector<uint32_t> mm_in0_args = {
            (std::uint32_t)core_type,
            0,                // ring_index
            next_core_noc.x,  // next_core_noc_x
            next_core_noc.y,  // next_core_noc_y
            noc,
            (std::uint32_t)end_of_hop,  // end_of_hop
        };
        tt_metal::SetRuntimeArgs(program, mm_kernel_in0_id, core, mm_in0_args);

        // in1
        std::vector<uint32_t> mm_kernel_in1_sender_writer_args;
        mm_kernel_in1_sender_writer_args.push_back((std::uint32_t)core_type);
        if (fused_op_signaler.has_value()) {
            ttnn::experimental::ccl::MatmulFusedOpSignaler& signaler = fused_op_signaler.value();
            signaler.push_llama_rs_rt_args_for_mm(mm_kernel_in1_sender_writer_args, core, in1_noc, device);
        }
        tt_metal::SetRuntimeArgs(program, mm_kernel_in1_sender_writer_id, core, mm_kernel_in1_sender_writer_args);

        // compute
        std::vector<uint32_t> mm_kernel_args;
        mm_kernel_args.push_back((std::uint32_t)core_type);
        tt_metal::SetRuntimeArgs(program, mm_kernel, core, mm_kernel_args);
    }
    std::vector<tt::tt_metal::CBHandle> shared_cbs = {cb_src0, cb_src1};
    shared_cbs.insert(shared_cbs.end(), cb_outputs.begin(), cb_outputs.end());

    return MatmulMultiCoreReuseMcast1DProgramFactory::shared_variables_t{
        {mm_kernel_in1_sender_writer_id},
        std::move(shared_cbs),
        false,
        CoreCoord{0, 0},
        std::move(worker_cores_vec),
        0,
        ttnn::prim::Matmul1DType::GATHER_IN0};
}

inline void override_mcast_in1_program_parameters(
    tt_metal::Program& program,
    const MatmulMultiCoreReuseMcast1DProgramFactory::shared_variables_t& override_variables,
    const ttnn::prim::MatmulInputs& tensor_args,
    const std::vector<ttnn::Tensor>& output_tensors) {
    const auto& input_tensors = tensor_args.input_tensors;
    const auto& optional_input_tensors = tensor_args.optional_input_tensors;

    TT_FATAL(
        input_tensors.size() + optional_input_tensors.size() == 3,
        "mcast in1 requires 3 input tensors, {} + {} = {} provided",
        input_tensors.size(),
        optional_input_tensors.size(),
        optional_input_tensors.size() + input_tensors.size());
    TT_FATAL(
        output_tensors.size() == 1, "matmul mcast in1 requires 1 output tensor, {} provided", output_tensors.size());

    const MeshTensor& src_a_tensor = input_tensors.at(0).mesh_tensor();
    const MeshTensor& src_b_tensor = input_tensors.at(1).mesh_tensor();
    const auto& bias_tensor = optional_input_tensors.at(0);

    ttsl::optional_reference<const MeshTensor> bias_mesh_tensor;
    if (bias_tensor.has_value()) {
        bias_mesh_tensor = bias_tensor.value().mesh_tensor();
    }

    const MeshTensor& dst_tensor = output_tensors.at(0).mesh_tensor();

    bool src0_sharded = input_tensors[0].is_sharded();
    bool out_sharded = output_tensors[0].is_sharded();

    auto& reader_runtime_args_by_core = GetRuntimeArgs(program, override_variables.kernels.at(0));

    // Manually unroll sender core
    {
        // in0 sender
        auto& reader_runtime_args =
            reader_runtime_args_by_core[override_variables.start_core.x][override_variables.start_core.y];
        reader_runtime_args[0] = src_a_tensor.address();

        // in1 sender
        auto& sender_writer_runtime_args =
            GetRuntimeArgs(program, override_variables.kernels.at(1), override_variables.start_core);
        sender_writer_runtime_args[0] = src_b_tensor.address();
        sender_writer_runtime_args[7] = dst_tensor.address();
        if (bias_tensor.has_value()) {
            sender_writer_runtime_args[18] = bias_mesh_tensor->address();
        }
    }

    auto& receiver_writer_runtime_args_by_core = GetRuntimeArgs(program, override_variables.kernels.at(2));

    for (uint32_t i = 1; i < override_variables.cores.size(); ++i) {
        const CoreCoord& core = override_variables.cores[i];

        auto& reader_runtime_args = reader_runtime_args_by_core[core.x][core.y];

        auto& writer_runtime_args = receiver_writer_runtime_args_by_core[core.x][core.y];

        // in0 sender
        reader_runtime_args[0] = src_a_tensor.address();
        // in1 receiver
        writer_runtime_args[2] = dst_tensor.address();
    }

    if (src0_sharded) {
        if (override_variables.extract_shard_sub_blocks) {
            UpdateDynamicCircularBufferAddress(program, override_variables.cbs.at(1), src_a_tensor);
        } else {
            UpdateDynamicCircularBufferAddress(program, override_variables.cbs.at(0), src_a_tensor);
        }
    }

    if (out_sharded) {
        UpdateDynamicCircularBufferAddress(program, override_variables.cbs.at(2), dst_tensor);
    }
}

static void override_mcast_in0_program_parameters(
    tt_metal::Program& program,
    const MatmulMultiCoreReuseMcast1DProgramFactory::shared_variables_t& override_variables,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
    const ttnn::prim::MatmulInputs& tensor_args,
    const std::vector<ttnn::Tensor>& output_tensors) {
    const auto& input_tensors = tensor_args.input_tensors;
    const auto& optional_input_tensors = tensor_args.optional_input_tensors;

    TT_FATAL(
        input_tensors.size() + optional_input_tensors.size() == 3,
        "mcast in0 requires 3 input tensors, {} + {} = {} provided",
        input_tensors.size(),
        optional_input_tensors.size(),
        optional_input_tensors.size() + input_tensors.size());
    TT_FATAL(
        output_tensors.size() == 1, "matmul mcast in0 requires 1 output tensor, {} provided", output_tensors.size());

    const MeshTensor& src_a_tensor = input_tensors.at(0).mesh_tensor();
    const MeshTensor& src_b_tensor = input_tensors.at(1).mesh_tensor();
    const auto& bias_tensor = optional_input_tensors.at(0);

    ttsl::optional_reference<const MeshTensor> bias_mesh_tensor;
    if (bias_tensor.has_value()) {
        bias_mesh_tensor = bias_tensor.value().mesh_tensor();
    }

    const MeshTensor& dst_tensor = output_tensors.at(0).mesh_tensor();

    bool src0_sharded = input_tensors[0].is_sharded();
    bool src1_sharded = input_tensors[1].is_sharded();
    bool out_sharded = output_tensors[0].is_sharded();

    // Manually unroll sender core
    if (src0_sharded) {
        UpdateDynamicCircularBufferAddress(program, override_variables.cbs.at(1), src_a_tensor);
    } else {
        // in0 sender
        auto& reader_sender_runtime_args =
            GetRuntimeArgs(program, override_variables.kernels.at(0), override_variables.start_core);
        reader_sender_runtime_args[0] = src_a_tensor.address();
    }

    if (src1_sharded) {
        // cbs[0] is cb_src1. For the receiver-contiguous GCB path it is a GlobalCircularBuffer whose
        // address is owned by the GCB (not tensor-backed), so the tensor overload of
        // UpdateDynamicCircularBufferAddress would TT_FATAL on a program-cache hit. Skip it there,
        // mirroring the gather_in0 override.
        if (!global_cb.has_value()) {
            UpdateDynamicCircularBufferAddress(program, override_variables.cbs.at(0), src_b_tensor);
        }
    }

    if (bias_tensor.has_value() && bias_tensor.value().is_sharded()) {
        UpdateDynamicCircularBufferAddress(program, override_variables.cbs.at(2), *bias_mesh_tensor);
    }

    auto& writer_runtime_args_by_core = GetRuntimeArgs(program, override_variables.kernels.at(1));

    for (uint32_t i = 0; i < override_variables.num_cores_with_work; ++i) {
        const auto& core = override_variables.cores[i];

        auto& writer_runtime_args = writer_runtime_args_by_core[core.x][core.y];

        // in1 sender
        writer_runtime_args[0] = src_b_tensor.address();
        writer_runtime_args[7] = dst_tensor.address();
        if (bias_tensor.has_value()) {
            writer_runtime_args[18] = bias_mesh_tensor->address();
        }
    }

    if (out_sharded) {
        UpdateDynamicCircularBufferAddress(program, override_variables.cbs.at(3), dst_tensor);
    }
}

inline void override_gather_in0_program_parameters(
    tt_metal::Program& program,
    const MatmulMultiCoreReuseMcast1DProgramFactory::shared_variables_t& override_variables,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
    const ttnn::prim::MatmulInputs& tensor_args,
    const std::vector<ttnn::Tensor>& output_tensors) {
    const auto& input_tensors = tensor_args.input_tensors;

    const MeshTensor& src_a_tensor = input_tensors[0].mesh_tensor();
    const MeshTensor& src_b_tensor = input_tensors[1].mesh_tensor();

    bool src0_sharded = input_tensors[0].is_sharded();
    bool src1_sharded = input_tensors[1].is_sharded();
    bool out_sharded = output_tensors[0].is_sharded();

    // Manually unroll sender core
    if (src0_sharded) {
        UpdateDynamicCircularBufferAddress(program, override_variables.cbs[0], src_a_tensor);
    }
    if (src1_sharded) {
        if (!global_cb.has_value() && !src_b_tensor.memory_config().is_dram()) {
            UpdateDynamicCircularBufferAddress(program, override_variables.cbs[1], src_b_tensor);
        }
    }
    if (out_sharded) {
        for (uint32_t i = 0; i < override_variables.cbs.size() - 2; ++i) {
            // cbs 0 and 1 contain cb_src0 and cb_src1
            // the rest contains the actual output cbs
            const auto& cb_output = override_variables.cbs[i + 2];
            const MeshTensor& out_tensor = output_tensors[i].mesh_tensor();
            UpdateDynamicCircularBufferAddress(program, cb_output, out_tensor);
        }
    }

    // Update in1 tensor address for all worker cores.
    // Note: override_variables.cores only contains worker cores (not hop/idle cores),
    // so it's safe to unconditionally update index [1] which holds in1_tensor_addr.
    auto& writer_runtime_args_by_core = GetRuntimeArgs(program, override_variables.kernels.at(0));
    for (const auto& core : override_variables.cores) {
        auto& writer_runtime_args = writer_runtime_args_by_core[core.x][core.y];

        /* in1 */
        writer_runtime_args[1] = src_b_tensor.address();
    }
}

void override_program_parameters(
    const MatmulMultiCoreReuseMcast1DProgramFactory::shared_variables_t& override_variables,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
    Program& program,
    const ttnn::prim::MatmulInputs& tensor_args,
    const std::vector<ttnn::Tensor>& tensor_return_value) {
    switch (override_variables.type) {
        case ttnn::prim::Matmul1DType::MCAST_IN0:
            override_mcast_in0_program_parameters(
                program, override_variables, global_cb, tensor_args, tensor_return_value);
            break;
        case ttnn::prim::Matmul1DType::GATHER_IN0: {
            override_gather_in0_program_parameters(
                program, override_variables, global_cb, tensor_args, tensor_return_value);
            break;
        }
        case ttnn::prim::Matmul1DType::MCAST_IN1:
            override_mcast_in1_program_parameters(program, override_variables, tensor_args, tensor_return_value);
            break;
    }
}

static ttnn::device_operation::ProgramArtifacts create_program_mcast_in0_artifacts(
    const ttnn::Tensor& a,
    tt_metal::IDevice* device,
    bool fp32_dest_acc_en,
    bool packer_l1_acc,
    CoreCoord compute_with_storage_grid_size,
    ttnn::operations::compute_throttle_utils::ThrottleLevel throttle_level,
    const DeviceComputeKernelConfig& compute_kernel_config,
    uint32_t in0_B,
    uint32_t in1_B,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    bool bcast_batch,
    bool transpose_a,
    bool transpose_b,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t out_block_h,
    uint32_t out_block_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    std::optional<UnaryWithParam> fused_activation,
    const MeshTensor& in0_tensor,
    const MeshTensor& in1_tensor,
    ttsl::optional_reference<const MeshTensor> bias_tensor,
    const MeshTensor& out_tensor,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const tt::tt_metal::Tile& bias_tile,
    const tt::tt_metal::Tile& output_tile,
    tt::DataFormat in0_data_format,
    tt::DataFormat in1_data_format,
    tt::DataFormat bias_data_format,
    tt::DataFormat output_data_format,
    bool in0_is_sharded,
    bool in1_is_sharded,
    bool bias_is_sharded,
    bool output_is_sharded,
    bool untilize_out,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& fused_op_signaler,
    bool row_broadcast_bias = true,
    CoreCoord sub_device_start_core = {0, 0},
    const std::vector<std::shared_ptr<tt::tt_metal::experimental::PrefetcherPipe>>& prefetcher_pipes = {}) {
    using tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids;

    // currently only support transpose of the full tile
    bool in0_transpose_tile = in0_tile.get_transpose_of_faces() && in0_tile.get_transpose_within_face();
    bool in1_transpose_tile = in1_tile.get_transpose_of_faces() && in1_tile.get_transpose_within_face();

    bool fuse_op = fused_op_signaler.has_value();

    // PrefetcherPipe delivery: the Tensor prefetcher writes each worker's in1 K-blocks straight into a
    // PrefetcherPipe ring on that worker, and cb_in1 is a relay laid over the rings, so in1 is neither
    // read from DRAM nor multicast here.
    const bool use_prefetcher_pipes = !prefetcher_pipes.empty();
    if (use_prefetcher_pipes) {
        // The weight is DRAM-sharded for the prefetcher, but this program never reads it: in1 comes
        // from the pipes, so none of the sharded-in1 handling below applies.
        in1_is_sharded = false;
        TT_FATAL(
            !transpose_b,
            "matmul mcast_in0 over prefetcher_pipes does not support transpose_b: the pipes deliver the weight's "
            "K-blocks in its DRAM layout");
    }

    uint32_t num_blocks = K / in0_block_w;
    // Only enable packer l1 accumulation when there are spills, otherwise
    // unnecessary overhead for reconfigs are added
    bool packer_l1_acc_en = packer_l1_acc && num_blocks > 1;

    // if fp32 enabled then we pack fp32 in l1, if not, then we pack fp16 in l1
    tt::DataFormat interm0_data_format = packer_l1_acc_en
                                             ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                                             : (fp32_dest_acc_en ? tt::DataFormat::Float32 : output_data_format);

    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    uint32_t bias_single_tile_size = bias_tile.get_tile_size(bias_data_format);

    // Tiles whose size is not a multiple of the DRAM alignment (e.g. bfp8 32x16 = 544B on
    // Blackhole's 64B alignment) are padded to it in DRAM. The interleaved reader copies tiles at
    // the padded stride, so the in0/in1/bias CBs must hold pages at the aligned stride and the
    // reader/unpacker walk tiles at the same stride. No-op when already aligned (all bf16 tiles,
    // 32-wide bfp8, Wormhole). Replaces the staging-CB workaround. Sharded buffers are backed by the
    // tensor buffer and keep their natural entry size.
    const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();
    uint32_t in0_aligned_tile_size =
        in0_is_sharded ? in0_single_tile_size : tt::align(in0_single_tile_size, dram_alignment);
    uint32_t in1_aligned_tile_size =
        in1_is_sharded ? in1_single_tile_size : tt::align(in1_single_tile_size, dram_alignment);
    // Bias buffer entries must be padded to the DRAM alignment so the reader's L1 write stride
    // matches the DRAM page stride (e.g. 64B on Blackhole for a 32B (1,16) bf16 bias tile).
    // Mirrors in0/in1 above. Sharded bias is backed by the L1 tensor buffer and keeps its
    // natural entry size. No-op on Wormhole and for tiles already >= dram_alignment.
    uint32_t bias_aligned_tile_size =
        bias_is_sharded ? bias_single_tile_size : tt::align(bias_single_tile_size, dram_alignment);
    uint32_t output_single_tile_size = output_tile.get_tile_size(output_data_format);
    uint32_t interm0_single_tile_size = output_tile.get_tile_size(interm0_data_format);

    bool do_not_inplace_interm0_out_dfb = output_is_sharded && (per_core_M != out_block_h);

    uint32_t in0_block_h = out_block_h;
    uint32_t in1_block_w = out_block_w;
    uint32_t in0_num_blocks_y = per_core_M / out_block_h;
    uint32_t in1_num_blocks_x = per_core_N / out_block_w;
    uint32_t out_num_blocks_x = in1_num_blocks_x;
    uint32_t out_num_blocks_y = in0_num_blocks_y;

    uint32_t in0_block_tiles = in0_block_h * in0_block_w;
    uint32_t in0_dfb_tiles = in0_block_tiles;
    if (in0_B * num_blocks > 1) {
        in0_dfb_tiles *= operations::matmul::utilities::MCAST_INPUT_BUFFERING_DEPTH;
    }
    uint32_t in0_dfb_size = in0_dfb_tiles * in0_aligned_tile_size;

    uint32_t in2_block_tiles = 0;
    uint32_t in0_shard_width_in_tiles = 0;
    uint32_t in0_shard_height_in_tiles = 0;
    if (in0_is_sharded) {
        in0_shard_width_in_tiles = in0_tensor.shard_spec()->shape[1] / in0_tile.get_width();
        in0_shard_height_in_tiles = in0_tensor.shard_spec()->shape[0] / in0_tile.get_height();
        in2_block_tiles = per_core_M * in0_shard_width_in_tiles;
    }
    uint32_t in2_dfb_tiles = in2_block_tiles;
    uint32_t in2_dfb_size = in2_dfb_tiles * in0_single_tile_size;

    uint32_t in1_block_tiles = out_block_w * in0_block_w;
    uint32_t in1_dfb_tiles = in1_block_tiles;
    if (in1_B * num_blocks > 1) {
        in1_dfb_tiles *= operations::matmul::utilities::MCAST_INPUT_BUFFERING_DEPTH;
    }
    if (in1_is_sharded) {
        uint32_t in1_shard_height_in_tiles = in1_tensor.shard_spec()->shape[0] / in1_tile.get_height();
        in1_dfb_tiles = per_core_N * in1_shard_height_in_tiles;
    }

    uint32_t in1_dfb_size = in1_dfb_tiles * in1_aligned_tile_size;

    uint32_t out_block_tiles = out_block_h * out_block_w;
    uint32_t out_shard_tiles = per_core_M * per_core_N;
    uint32_t out_dfb_tiles = out_block_tiles;  // No double buffer
    if (output_is_sharded) {
        out_dfb_tiles = out_shard_tiles;
    }
    uint32_t out_dfb_size = out_dfb_tiles * output_single_tile_size;
    uint32_t interm0_dfb_tiles = out_block_tiles;  // No double buffer
    uint32_t interm0_dfb_size = interm0_dfb_tiles * interm0_single_tile_size;

    uint32_t in3_block_tiles = out_block_w;
    uint32_t in3_dfb_tiles = in3_block_tiles;  // No double buffer
    uint32_t in3_dfb_size = in3_dfb_tiles * bias_aligned_tile_size;

    CoreCoord start_core = sub_device_start_core;
    uint32_t start_core_x = start_core.x;
    uint32_t start_core_y = start_core.y;
    uint32_t num_cores_c = compute_with_storage_grid_size.x;

    // The matmul region is the rectangle of size `compute_with_storage_grid_size`
    // anchored at `start_core`. Callers must ensure this rectangle lies entirely
    // within the active sub-device's worker cores (validated upstream).
    CoreRangeSet matmul_core_rect(CoreRange(
        start_core,
        CoreCoord(
            start_core_x + compute_with_storage_grid_size.x - 1, start_core_y + compute_with_storage_grid_size.y - 1)));

    uint32_t num_blocks_y = ((M - 1) / per_core_M) + 1;
    uint32_t num_blocks_x = ((N - 1) / per_core_N) + 1;
    uint32_t num_blocks_total = num_blocks_y * num_blocks_x;
    uint32_t num_cores_with_work = num_blocks_total;

    // mcast_in0 broadcasts a single M-block across all cores; only the start_core gets in0 sender
    // runtime args (with output_idx_y == 0), so M must fit within per_core_M.
    TT_FATAL(
        num_blocks_y == 1,
        "matmul_multicore_reuse_mcast_1d requires num_blocks_y == 1 (M <= per_core_M) for mcast_in0. "
        "Got M={}, per_core_M={}, num_blocks_y={}.",
        M,
        per_core_M,
        num_blocks_y);

    uint32_t in0_sender_num_cores = in0_is_sharded ? a.shard_spec().value().grid.num_cores() : 1;
    uint32_t num_cores = in0_is_sharded ? std::max(num_cores_with_work, in0_sender_num_cores) : num_cores_with_work;

    constexpr bool row_major = true;
    CoreRangeSet all_cores =
        num_cores_to_corerangeset_in_subcoregrids(start_core, num_cores, matmul_core_rect, row_major);

    CoreRangeSet in0_mcast_sender_cores =
        num_cores_to_corerangeset_in_subcoregrids(start_core, in0_sender_num_cores, matmul_core_rect, row_major);
    CoreCoord in0_mcast_sender_cores_grid = in0_mcast_sender_cores.bounding_box().grid_size();

    CoreRangeSet all_cores_with_work =
        num_cores_to_corerangeset_in_subcoregrids(start_core, num_cores_with_work, matmul_core_rect, row_major);
    CoreRange in0_mcast_receiver_cores_bounding_box = all_cores_with_work.bounding_box();
    uint32_t in0_mcast_receiver_num_cores = in0_mcast_receiver_cores_bounding_box.size();  // always mcast to full grid
    uint32_t in0_mcast_receiver_num_dests = std::min(
        in0_mcast_receiver_num_cores,
        num_cores);  // should always be number of cores in receiver grid up to number of active cores

    CoreRangeSet in0_mcast_cores_with_work_and_in_receiver_grid;
    CoreRangeSet in0_mcast_cores_without_work_and_in_receiver_grid;
    CoreRangeSet in0_mcast_cores_without_work_and_not_in_receiver_grid;
    CoreRangeSet in0_mcast_receivers;
    std::vector<uint32_t> in0_mcast_noc_x;
    std::vector<uint32_t> in0_mcast_noc_y;
    if (in0_is_sharded) {
        in0_mcast_cores_with_work_and_in_receiver_grid = all_cores_with_work;

        if (in0_mcast_receiver_num_dests > num_cores_with_work) {
            const uint32_t in0_mcast_cores_without_work_and_in_receiver_grid_num_cores =
                in0_mcast_receiver_num_dests - num_cores_with_work;
            uint32_t core_idx_x = num_cores_with_work % num_cores_c;
            uint32_t core_idx_y = num_cores_with_work / num_cores_c;
            CoreCoord start_core = {(std::size_t)start_core_x + core_idx_x, (std::size_t)start_core_y + core_idx_y};
            in0_mcast_cores_without_work_and_in_receiver_grid = num_cores_to_corerangeset_in_subcoregrids(
                start_core, in0_mcast_cores_without_work_and_in_receiver_grid_num_cores, matmul_core_rect, row_major);
        }

        if (in0_sender_num_cores > in0_mcast_receiver_num_dests) {
            const uint32_t in0_mcast_cores_without_work_and_not_in_receiver_grid_num_cores =
                in0_sender_num_cores - in0_mcast_receiver_num_dests;
            uint32_t core_idx_x = in0_mcast_receiver_num_dests % num_cores_c;
            uint32_t core_idx_y = in0_mcast_receiver_num_dests / num_cores_c;
            CoreCoord start_core = {(std::size_t)start_core_x + core_idx_x, (std::size_t)start_core_y + core_idx_y};
            in0_mcast_cores_without_work_and_not_in_receiver_grid = num_cores_to_corerangeset_in_subcoregrids(
                start_core,
                in0_mcast_cores_without_work_and_not_in_receiver_grid_num_cores,
                matmul_core_rect,
                row_major);
        }

        in0_mcast_noc_x.reserve(in0_mcast_sender_cores_grid.x);
        in0_mcast_noc_y.reserve(in0_mcast_sender_cores_grid.y);
        for (uint32_t core_idx_x = 0; core_idx_x < in0_mcast_sender_cores_grid.x; ++core_idx_x) {
            in0_mcast_noc_x.push_back(
                device->worker_core_from_logical_core({start_core_x + core_idx_x, start_core_y}).x);
        }
        for (uint32_t core_idx_y = 0; core_idx_y < in0_mcast_sender_cores_grid.y; ++core_idx_y) {
            in0_mcast_noc_y.push_back(
                device->worker_core_from_logical_core({start_core_x, start_core_y + core_idx_y}).y);
        }
    } else {
        in0_mcast_cores_with_work_and_in_receiver_grid = CoreRangeSet({CoreRange(start_core, start_core)});
        if (in0_mcast_receiver_num_cores > 1) {
            // Check against the actual rectangle width (instead of bare grid_size.x-1) so that
            // sub-devices anchored away from (0, 0) wrap correctly.
            auto receiver_start_core = compute_with_storage_grid_size.x > 1 ? CoreCoord{start_core.x + 1, start_core.y}
                                                                            : CoreCoord{start_core.x, start_core.y + 1};
            in0_mcast_receivers = num_cores_to_corerangeset_in_subcoregrids(
                receiver_start_core, num_cores - 1, matmul_core_rect, row_major);
        }
    }

    CoreCoord top_left_core = in0_mcast_receiver_cores_bounding_box.start_coord;
    CoreCoord bottom_right_core = in0_mcast_receiver_cores_bounding_box.end_coord;
    auto top_left_core_physical = device->worker_core_from_logical_core(top_left_core);
    auto bottom_right_core_physical = device->worker_core_from_logical_core(bottom_right_core);

    uint32_t in0_num_subblocks = (out_block_h / out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
    const auto& a_shape_logical = operations::matmul::utilities::get_matmul_tensor_logical_shape(a, transpose_a);
    // When transpose_a is true, the K dimension maps to the row dimension of the raw tile,
    // which is already zero-padded during tile layout conversion. pad_last_ktile operates on
    // columns, so applying it would incorrectly zero valid data that becomes output rows
    // after the compute kernel transposes the tile.
    const auto in0_last_ktile_w = transpose_a ? 0 : a_shape_logical[-1] % in0_tile.get_width();
    const auto in0_last_ktile_h = transpose_a ? a_shape_logical[-1] % in0_tile.get_width() : 0;
    TT_FATAL(
        in0_last_ktile_w == 0 || in0_last_ktile_h == 0,
        "At most one of in0_last_ktile_w ({}) and in0_last_ktile_h ({}) can be non-zero",
        in0_last_ktile_w,
        in0_last_ktile_h);

    const auto& a_padded_shape = operations::matmul::utilities::get_matmul_tensor_padded_shape(a, transpose_a);
    const uint32_t M_per_batch = a_padded_shape[-2] / in0_tile.get_height();
    const auto [in0_tensor_stride_w, in0_tensor_stride_h] =
        operations::matmul::utilities::get_in0_transpose_strides(M, M_per_batch, transpose_a, K);
    const auto in0_tensor_next_block_stride = in0_block_w * in0_tensor_stride_w;
    const auto in0_tensor_next_h_dim_block_stride = in0_block_h * in0_tensor_stride_h;
    const auto in0_tensor_start_tile_id_stride = per_core_M * in0_tensor_stride_h;

    const auto in1_tensor_stride_w = transpose_b ? K : 1;
    const auto in1_tensor_stride_h = transpose_b ? 1 : N;
    const auto in1_tensor_next_block_stride = in0_block_w * in1_tensor_stride_h;
    const auto in1_tensor_next_w_dim_block_stride = in1_block_w * in1_tensor_stride_w;
    const auto in1_tensor_start_tile_id_stride = per_core_N * in1_tensor_stride_w;

    // create_program_artifacts always passes std::nullopt for the fused-op signaler, so the CCL
    // fused-op path is unreachable here. The Metal 2.0 kernel forks do not carry it: MatmulOpReceiver
    // and OpSignaler read positional runtime args from outside this op's directory, which named
    // arguments cannot supply. Fail loudly rather than silently building a program without it.
    TT_FATAL(!fuse_op, "matmul_multicore_reuse_mcast_1d: the Metal 2.0 path does not support fused CCL ops");

    // ------------------------------------------------------------------
    // Spec-scope resource names. Declared function-local rather than at file scope: the matmul
    // factory .cpp files share one unity-build target, so file-scope constants with these names
    // would collide as sibling factories are ported.
    // ------------------------------------------------------------------
    const KernelSpecName IN0_SENDER{"in0_sender"};
    const KernelSpecName IN0_NO_WORK_IN_RECV{"in0_no_work_in_receiver"};
    const KernelSpecName IN0_NO_WORK_NOT_IN_RECV{"in0_no_work_not_in_receiver"};
    const KernelSpecName IN0_RECEIVER{"in0_receiver"};
    const KernelSpecName IN1_SENDER_WRITER{"in1_sender_writer"};
    const KernelSpecName COMPUTE{"compute"};

    const DFBSpecName IN0_DFB{"in0"};
    const DFBSpecName IN0_RELAY_DFB{"in0_relay"};
    const DFBSpecName IN1_DFB{"in1"};
    const DFBSpecName IN0_SHARDED_DFB{"in0_sharded"};
    const DFBSpecName OUT_DFB{"out"};
    const DFBSpecName INTERM0_DFB{"intermed0"};
    const DFBSpecName INTERM0_ALIAS_DFB{"intermed0_reload_alias"};
    const DFBSpecName BIAS_DFB{"bias"};
    const DFBSpecName IN0_TRANSPOSED_DFB{"in0_transposed"};

    const TensorParamName IN0{"in0"};
    const TensorParamName IN1{"in1"};
    const TensorParamName OUTPUT{"output"};
    const TensorParamName BIAS{"bias"};

    const SemaphoreSpecName SENDER_SEM{"in0_mcast_sender"};
    const SemaphoreSpecName RECEIVER_SEM{"in0_mcast_receiver"};

    // ------------------------------------------------------------------
    // Compute-kernel derived sizes (unchanged from the legacy factory)
    // ------------------------------------------------------------------
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;

    uint32_t in1_num_subblocks = (out_block_w / out_subblock_w);
    uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks;
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    // The fused bias add reads the partials buffer as an FPU operand (SrcA), so UnpackToDestFp32
    // cannot be set on it directly when bias is present. In that case the cross-block reload instead
    // copies through a second buffer aliasing the same SRAM that does carry UnpackToDestFp32, while
    // the bias add keeps reading the partials buffer via SrcA. The alias is handed to the compute
    // kernel as the MM_PARTIALS_RELOAD_ALIAS define plus its own binding, which selects the alias
    // reload path there. Without bias the reload reads the partials buffer and the flag is set on it.
    const bool bias_reload_alias =
        fp32_dest_acc_en && interm0_data_format == tt::DataFormat::Float32 && bias_tensor.has_value();

    // ------------------------------------------------------------------
    // Defines
    // ------------------------------------------------------------------
    std::map<std::string, std::string> mm_kernel_defines;
    if (use_prefetcher_pipes) {
        mm_kernel_defines["ENABLE_PREFETCHER_PIPE"] = "1";
    }
    std::map<std::string, std::string> mm_kernel_in0_sender_writer_defines;
    std::map<std::string, std::string> mm_kernel_in1_sender_writer_defines;
    if (use_prefetcher_pipes) {
        mm_kernel_in1_sender_writer_defines["ENABLE_PREFETCHER_PIPE"] = "1";
    }
    if (bias_tensor.has_value()) {
        mm_kernel_defines["FUSE_BIAS"] = "1";
        mm_kernel_in1_sender_writer_defines["FUSE_BIAS"] = "1";
    }
    if (fused_activation.has_value()) {
        if (fused_activation.value().op_type == UnaryOpType::RELU) {
            mm_kernel_defines["PACK_RELU"] = "1";
        } else {
            mm_kernel_defines["SFPU_ACTIVATION"] = "1";
        }
    }
    if (packer_l1_acc_en) {
        mm_kernel_defines["PACKER_L1_ACC"] = "1";
    }
    if (fp32_dest_acc_en) {
        mm_kernel_defines["FP32_DEST_ACC_EN"] = "1";
    }
    if (in1_transpose_tile) {
        mm_kernel_defines["IN1_TRANSPOSE_TILE"] = "1";
    }
    // in0_transpose_tile selects which buffer the compute kernel matmuls from. It arrives as a
    // define rather than a compile-time arg because the in0_transposed buffer is only bound when
    // the transpose is wanted, and a ternary over the two buffer names would name-look-up the
    // unbound one.
    if (in0_transpose_tile) {
        mm_kernel_defines["IN0_TRANSPOSE_TILE"] = "1";
    }
    if (bias_reload_alias) {
        mm_kernel_defines["MM_PARTIALS_RELOAD_ALIAS"] = "1";
    }

    ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
        device->arch(), num_cores, mm_kernel_defines);
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(), num_cores, mm_kernel_defines, throttle_level);

    if (in1_is_sharded) {
        mm_kernel_in1_sender_writer_defines["IN1_SHARDED"] = "1";
    }

    if (bias_is_sharded) {
        mm_kernel_in1_sender_writer_defines["BIAS_SHARDED"] = "1";
    }

    if (output_is_sharded) {
        mm_kernel_in1_sender_writer_defines["OUT_SHARDED"] = "1";
    }

    // TODO: SKIP_MCAST flag isn't used for the sharded reader kernel because internal mcast logic already works without
    // skipping We can use this flag to turn off unnecessary mcast overhead if necessary
    if (in0_mcast_receiver_num_cores == 1) {
        mm_kernel_in0_sender_writer_defines["SKIP_MCAST"] = "1";
    }

    mm_kernel_in1_sender_writer_defines["SKIP_MCAST"] = "1";

    // in1 is the reader of weights/output writer, and we choose to make it use the optimized reader noc
    tt_metal::NOC in0_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    tt_metal::NOC in1_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());

    ////////////////////////////////////////////////////////////////////////////
    //                      Dataflow buffers
    ////////////////////////////////////////////////////////////////////////////
    // The output and partials buffers share one L1 region unless the shapes or formats force them
    // apart; when they share, they are aliases of each other rather than independent buffers.
    const bool separate_out_and_interm0 = do_not_inplace_interm0_out_dfb ||
                                          (interm0_data_format != output_data_format) ||
                                          (untilize_out && (in1_num_subblocks > 1));
    // Every member of an alias group must have the same total backing size, so the partials buffer
    // (and its alias) are sized against whichever region they actually sit in.
    const uint32_t interm0_total_size = separate_out_and_interm0 ? interm0_dfb_size : out_dfb_size;

    Group<DataflowBufferSpec> dataflow_buffers;

    // The alias group: {partials, partials alias?} when the output has its own region, otherwise
    // {output, partials, partials alias?} because all of them sit on the one region.
    Group<DFBSpecName> alias_group;
    if (!separate_out_and_interm0) {
        alias_group.push_back(OUT_DFB);
    }
    alias_group.push_back(INTERM0_DFB);
    if (bias_reload_alias) {
        alias_group.push_back(INTERM0_ALIAS_DFB);
    }
    // A single-member "group" is not aliasing at all.
    const bool has_alias_group = alias_group.size() > 1;
    // Legacy put every shared index on ONE CBDescriptor whose single `tensor` field backed the whole
    // region, so when the partials share the output's region and the output is sharded, they are
    // backed by the output tensor too. The alias legality rule wants the same thing: either no member
    // of a group borrows, or all of them borrow from the same TensorParameter.
    const std::optional<TensorParamName> alias_group_borrow =
        (!separate_out_and_interm0 && output_is_sharded) ? std::optional<TensorParamName>(OUTPUT) : std::nullopt;
    auto alias_with_others = [&](const DFBSpecName& self) {
        Group<DFBSpecName> others;
        // A DFB that is not a member of the group aliases nothing. The output is exactly that case
        // when it has its own region: filtering only `self` would hand it the partials as aliases
        // while the partials never name it back, and the transitivity rule rejects a group whose
        // members disagree on their membership.
        const bool self_in_group = std::find(alias_group.begin(), alias_group.end(), self) != alias_group.end();
        if (has_alias_group && self_in_group) {
            for (const auto& name : alias_group) {
                if (name != self) {
                    others.push_back(name);
                }
            }
        }
        return others;
    };

    // Which in0 senders land on nodes that own a shard but produce no output block. Computed here
    // because the relay buffer below is conditional on it; reused at the kernel-spec sites.
    const bool has_in0_no_work_in_receiver_kernel =
        in0_is_sharded && in0_mcast_cores_without_work_and_in_receiver_grid.num_cores() > 0;
    const bool has_in0_no_work_not_in_receiver_kernel =
        in0_is_sharded && in0_mcast_cores_without_work_and_not_in_receiver_grid.num_cores() > 0;
    const bool has_in0_relay_dfb = has_in0_no_work_in_receiver_kernel || has_in0_no_work_not_in_receiver_kernel;

    // in0
    DataflowBufferSpec in0_dfb_spec{
        .unique_id = IN0_DFB,
        .entry_size = in0_aligned_tile_size,
        .num_entries = in0_dfb_size / in0_aligned_tile_size,
        .data_format_metadata = in0_data_format,
        .tile_format_metadata = in0_tile,
    };

    // The in0 multicast relay buffer.
    //
    // in0 is a plain per-node FIFO: the sender fills a slot (the payload arrives by NoC, from this
    // node or a peer) and compute drains it. The nodes that own a K-slice but no output block have
    // no compute, so they are not part of that FIFO -- yet they still multicast into it, and a
    // multicast writes one L1 offset on every destination. The sender derives that offset from a
    // local cursor it advances in step with the receivers, so its buffer must sit at in0's offset.
    // Hence a second DFB, self-looped by the senders on those nodes, with in0's geometry.
    //
    // The two offsets coincide because this pair is declared before any other DFB, so each starts at
    // its own allocator's base. That is the allocator's behaviour, not a declared property --
    // inserting any DFB on the work nodes ahead of in0 would part them by that DFB's size. Metal 2.0
    // cannot state the requirement yet (alias_with is the mechanism, but it requires members to
    // cover identical nodes; #56887 lifts that); until it can, KEEP THIS PAIR FIRST.
    //
    // Both no-work senders share this one relay: they run the same source on disjoint node sets, so
    // its producer and consumer sets are equal and every node still hosts exactly one of each.
    DataflowBufferSpec in0_relay_dfb_spec{
        .unique_id = IN0_RELAY_DFB,
        .entry_size = in0_dfb_spec.entry_size,
        .num_entries = in0_dfb_spec.num_entries,
        .data_format_metadata = in0_data_format,
        .tile_format_metadata = in0_tile,
    };
    dataflow_buffers.push_back(std::move(in0_dfb_spec));
    if (has_in0_relay_dfb) {
        dataflow_buffers.push_back(std::move(in0_relay_dfb_spec));
    }

    TT_FATAL(
        dataflow_buffers.front().unique_id == IN0_DFB && dataflow_buffers.size() == (has_in0_relay_dfb ? 2u : 1u),
        "in0{} must be the first DataflowBufferSpec(s) declared, so that each starts at its own "
        "allocator's base and the two share an L1 offset; got {} spec(s) with '{}' first",
        has_in0_relay_dfb ? " and its multicast relay" : "",
        dataflow_buffers.size(),
        dataflow_buffers.front().unique_id.get());

    // in1. Under PrefetcherPipe delivery it is a relay over the pipes' rings, one entry per K-block:
    // the prefetcher writes whole K-blocks, and compute addresses the tiles inside one
    // (matmul_block_in1_at). Every pipe is declared as a parameter and bound by the in1 reader under
    // one accessor; each worker holds exactly one pipe's receiver.
    Group<PrefetcherPipeParameter> prefetcher_pipe_parameters;
    Group<PrefetcherPipeParamName> prefetcher_pipe_names;
    const uint32_t in1_pipe_entry_size = in1_block_tiles * in1_single_tile_size;
    if (use_prefetcher_pipes) {
        const CoreRangeSet pipe_receivers =
            tt::tt_metal::experimental::prefetcher_pipe_receiver_cores(prefetcher_pipes);
        TT_FATAL(
            pipe_receivers.num_cores() == all_cores_with_work.num_cores() &&
                pipe_receivers.intersection(all_cores_with_work).num_cores() == all_cores_with_work.num_cores(),
            "matmul mcast_in0 over prefetcher_pipes needs the pipes' receivers to be exactly the {} workers that "
            "compute an output block ({}), but they are {}. Receiver i in row-major order computes output columns "
            "[i * per_core_N, (i + 1) * per_core_N).",
            all_cores_with_work.num_cores(),
            all_cores_with_work.str(),
            pipe_receivers.str());
        // Validation has checked the ring is a whole number of these K-blocks, and at least two.
        const uint32_t ring_size = prefetcher_pipes.front()->ring_size();
        for (size_t p = 0; p < prefetcher_pipes.size(); ++p) {
            const PrefetcherPipeParamName name{fmt::format("in1_prefetcher_pipe_{}", p)};
            prefetcher_pipe_names.push_back(name);
            prefetcher_pipe_parameters.push_back(PrefetcherPipeParameter{
                .unique_id = name,
                .receivers = prefetcher_pipes[p]->receiver_cores(),
                .ring_size = ring_size,
                .entry_size = in1_pipe_entry_size,
            });
        }
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = IN1_DFB,
            .entry_size = in1_pipe_entry_size,
            .num_entries = ring_size / in1_pipe_entry_size,
            .data_format_metadata = in1_data_format,
            .tile_format_metadata = in1_tile,
            .advanced_options = {.prefetcher_pipe_relays = prefetcher_pipe_names},
        });
    } else {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = IN1_DFB,
            .entry_size = in1_aligned_tile_size,
            .num_entries = in1_dfb_size / in1_aligned_tile_size,
            .data_format_metadata = in1_data_format,
            .tile_format_metadata = in1_tile,
            .borrowed_from = in1_is_sharded ? std::optional<TensorParamName>(IN1) : std::nullopt,
        });
    }

    // in0 sharded: the resident in0 shard the block-sharded sender multicasts out of.
    if (in0_is_sharded) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = IN0_SHARDED_DFB,
            .entry_size = in0_single_tile_size,
            .num_entries = in2_dfb_size / in0_single_tile_size,
            .data_format_metadata = in0_data_format,
            .tile_format_metadata = in0_tile,
            .borrowed_from = IN0,
        });
    }

    // Legacy also allocated a 64-byte "local L1 to store temp vars" buffer here (CB index 6) and
    // handed its index to the block-sharded readers. No kernel ever read it, so it has no endpoints
    // and no behaviour; it is dropped rather than translated.

    // output
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = out_dfb_size / output_single_tile_size,
        .data_format_metadata = output_data_format,
        .tile_format_metadata = output_tile,
        .borrowed_from = output_is_sharded ? std::optional<TensorParamName>(OUTPUT) : std::nullopt,
        .advanced_options = {.alias_with = alias_with_others(OUT_DFB)},
    });

    // partials
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = INTERM0_DFB,
        .entry_size = interm0_single_tile_size,
        .num_entries = interm0_total_size / interm0_single_tile_size,
        .data_format_metadata = interm0_data_format,
        .tile_format_metadata = output_tile,
        .borrowed_from = alias_group_borrow,
        .advanced_options = {.alias_with = alias_with_others(INTERM0_DFB)},
    });

    // partials alias over the same SRAM, marked UnpackToDest, for the bias reload (see above).
    if (bias_reload_alias) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = INTERM0_ALIAS_DFB,
            .entry_size = interm0_single_tile_size,
            .num_entries = interm0_total_size / interm0_single_tile_size,
            .data_format_metadata = interm0_data_format,
            .tile_format_metadata = output_tile,
            .borrowed_from = alias_group_borrow,
            .advanced_options = {.alias_with = alias_with_others(INTERM0_ALIAS_DFB)},
        });
    }

    // bias
    if (bias_tensor.has_value()) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = BIAS_DFB,
            .entry_size = bias_aligned_tile_size,
            .num_entries = in3_dfb_size / bias_aligned_tile_size,
            .data_format_metadata = bias_data_format,
            .tile_format_metadata = bias_tile,
            .borrowed_from = bias_is_sharded ? std::optional<TensorParamName>(BIAS) : std::nullopt,
        });
    }

    // in0 transposed
    if (in0_transpose_tile) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = IN0_TRANSPOSED_DFB,
            .entry_size = in0_aligned_tile_size,
            .num_entries = in0_dfb_size / in0_aligned_tile_size,
            .data_format_metadata = in0_data_format,
            .tile_format_metadata = in0_tile,
        });
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Semaphores
    ////////////////////////////////////////////////////////////////////////////
    Group<SemaphoreSpec> semaphores = {
        SemaphoreSpec{.unique_id = SENDER_SEM, .target_nodes = all_cores},
        SemaphoreSpec{.unique_id = RECEIVER_SEM, .target_nodes = all_cores},
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      Kernels
    ////////////////////////////////////////////////////////////////////////////
    const auto in0_sender_hw_config =
        DataMovementGen1Config{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = in0_noc};
    const auto in1_sender_hw_config =
        DataMovementGen1Config{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = in1_noc};

    Group<KernelSpec> kernels;

    // ---- in0 sender ------------------------------------------------------
    // Two different sources, selected by whether in0 is sharded; both fill the in0 buffer and
    // multicast it to the receiver grid.
    const std::string in0_sender_source =
        in0_is_sharded ? "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
                         "reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded_metal2.cpp"
                       : "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
                         "reader_bmm_tile_layout_in0_sender_padding_metal2.cpp";

    // Common bindings for every instance of the in0 sender. in0_dfb is the multicast staging buffer
    // this instance works through: in0 itself on the nodes that feed compute, the co-located relay on
    // the nodes that only send. The accessor name is "in0" either way, so the kernel source does not
    // distinguish them.
    auto in0_sender_dfb_bindings = [&](bool core_has_output_block_work, const DFBSpecName& in0_dfb) {
        Group<DFBBinding> b = {
            DFBBinding{
                .dfb_spec_name = in0_dfb,
                .accessor_name = "in0",
                .endpoint_type = DFBEndpointType::PRODUCER,
            },
        };
        if (!core_has_output_block_work) {
            // These nodes run no compute and no writer, so nothing downstream drains the buffer. The
            // sender pops its own tiles instead (the `if constexpr (!core_has_output_block_work)` pop
            // at the bottom of the block loop), which keeps the write pointer in lockstep with the
            // cores that do have work -- the multicast depends on every participant agreeing on it.
            // One toucher doing both halves is a self-loop, which is why this is the relay buffer and
            // not in0: in0's consumer is compute, and a role cannot mix kernel kinds.
            b.push_back(DFBBinding{
                .dfb_spec_name = in0_dfb,
                .accessor_name = "in0",
                .endpoint_type = DFBEndpointType::CONSUMER,
            });
        }
        if (in0_is_sharded) {
            // The resident shard is read through a raw pointer, never through the FIFO, so the one
            // kernel that touches it carries both endpoints.
            b.push_back(DFBBinding{
                .dfb_spec_name = IN0_SHARDED_DFB,
                .accessor_name = "in0_sharded",
                .endpoint_type = DFBEndpointType::PRODUCER,
            });
            b.push_back(DFBBinding{
                .dfb_spec_name = IN0_SHARDED_DFB,
                .accessor_name = "in0_sharded",
                .endpoint_type = DFBEndpointType::CONSUMER,
            });
        }
        return b;
    };
    const Group<SemaphoreBinding> in0_mcast_sem_bindings = {
        SemaphoreBinding{.semaphore_spec_name = SENDER_SEM, .accessor_name = "in0_mcast_sender"},
        SemaphoreBinding{.semaphore_spec_name = RECEIVER_SEM, .accessor_name = "in0_mcast_receiver"},
    };

    // The block-sharded sender walks per-sender mcast NOC coordinate lists whose lengths are
    // compile-time args rather than source literals, so the lists stay runtime varargs: the num_x
    // x-coordinates first, then the num_y y-coordinates.
    const uint32_t in0_sender_num_varargs =
        in0_is_sharded ? static_cast<uint32_t>(in0_mcast_noc_x.size() + in0_mcast_noc_y.size()) : 0u;

    // Per-instance compile-time args. The block-sharded source is instantiated three times over
    // disjoint core sets, differing only on the two flags below.
    auto make_in0_sender_cta = [&](bool core_has_output_block_work, bool core_in_receiver_grid) {
        KernelSpec::CompileTimeArgs cta;
        if (in0_is_sharded) {
            cta = {
                {"core_has_output_block_work", static_cast<uint32_t>(core_has_output_block_work)},
                {"core_in_in0_receiver_mcast_grid", static_cast<uint32_t>(core_in_receiver_grid)},
                {"in0_block_num_tiles", in0_block_num_tiles},
                {"in0_block_size_bytes", in0_block_num_tiles * in0_single_tile_size},
                {"in0_last_ktile_w", static_cast<uint32_t>(in0_last_ktile_w)},
                {"in0_last_ktile_h", static_cast<uint32_t>(in0_last_ktile_h)},
                {"num_blocks_inner_dim", num_blocks},
                {"num_blocks_w_dim", out_num_blocks_x},
                {"num_blocks_h_dim", out_num_blocks_y},
                {"in0_mcast_num_dests", in0_mcast_receiver_num_dests},
                {"in0_mcast_num_cores", in0_mcast_receiver_num_cores},
                {"num_x", static_cast<uint32_t>(in0_mcast_sender_cores_grid.x)},
                {"num_y", static_cast<uint32_t>(in0_mcast_sender_cores_grid.y)},
                {"transpose_mcast", 0u},
                {"shard_width_in_tiles", in0_shard_width_in_tiles},
                {"shard_height_in_tiles", in0_shard_height_in_tiles},
                {"in0_block_w", in0_block_w},
                {"in0_block_h", in0_block_h},
                {"batch", in0_B},
            };
        } else {
            cta = {
                {"in0_tensor_stride_w", static_cast<uint32_t>(in0_tensor_stride_w)},
                {"in0_tensor_stride_h", static_cast<uint32_t>(in0_tensor_stride_h)},
                {"in0_tensor_next_inner_dim_block_stride", static_cast<uint32_t>(in0_tensor_next_block_stride)},
                {"in0_tensor_next_h_dim_block_stride", static_cast<uint32_t>(in0_tensor_next_h_dim_block_stride)},
                {"in0_block_w", in0_block_w},
                {"in0_block_h", in0_block_h},
                {"in0_block_num_tiles", in0_block_num_tiles},
                {"in0_last_ktile_w", static_cast<uint32_t>(in0_last_ktile_w)},
                {"in0_last_ktile_h", static_cast<uint32_t>(in0_last_ktile_h)},
                {"shard_width_in_tiles", 0u},
                {"shard_height_in_tiles", 0u},
                {"num_blocks_inner_dim", num_blocks},
                {"num_blocks_w_dim", out_num_blocks_x},
                {"num_blocks_h_dim", out_num_blocks_y},
                {"in0_mcast_num_dests", num_cores - 1},
                {"in0_mcast_num_cores", in0_mcast_receiver_num_cores - 1},
                {"MtKt", M * K},
                {"in0_B", in0_B},
                {"in1_B", in1_B},
                {"in0_reuse_in_dfb", 0u},
                {"batchB", 0u},
                {"bcast_A", 1u},
                {"get_batch_from_reader", 0u},
                {"num_active", 0u},
            };
        }
        return cta;
    };

    // The block-sharded source reads shard_width/height and the mcast lists; the interleaved one
    // reads the in0 tensor. Only the interleaved source binds a tensor accessor.
    auto make_in0_sender_spec = [&](const KernelSpecName& unique_id,
                                    bool core_has_output_block_work,
                                    bool core_in_receiver_grid,
                                    const DFBSpecName& in0_dfb) {
            KernelSpec k{
                .unique_id = unique_id,
                .source = in0_sender_source,
                .compiler_options =
                    {
                        .defines = KernelSpec::CompilerOptions::Defines(mm_kernel_in0_sender_writer_defines),
                    },
                .dfb_bindings = in0_sender_dfb_bindings(core_has_output_block_work, in0_dfb),
                .semaphore_bindings = in0_mcast_sem_bindings,
                .compile_time_args = make_in0_sender_cta(core_has_output_block_work, core_in_receiver_grid),
                .hw_config = in0_sender_hw_config,
            };
            if (in0_is_sharded) {
                k.runtime_arg_schema = {
                    .runtime_arg_names =
                        {"sender_id",
                         "in0_mcast_dest_noc_start_x",
                         "in0_mcast_dest_noc_start_y",
                         "in0_mcast_dest_noc_end_x",
                         "in0_mcast_dest_noc_end_y"},
                };
                k.advanced_options.num_runtime_varargs = in0_sender_num_varargs;
            } else {
                k.tensor_bindings = {
                    TensorBinding{.tensor_parameter_name = IN0, .accessor_name = "in0"},
                };
                k.runtime_arg_schema = {
                    .runtime_arg_names =
                        {"in0_tensor_start_tile_id",
                         "in0_mcast_dest_noc_start_x",
                         "in0_mcast_dest_noc_start_y",
                         "in0_mcast_dest_noc_end_x",
                         "in0_mcast_dest_noc_end_y",
                         "last_block_h"},
                };
            }
            return k;
        };

    kernels.push_back(make_in0_sender_spec(IN0_SENDER, true, true, IN0_DFB));

    // Both no-work senders work the relay buffer, not in0 (see its declaration above).
    if (has_in0_no_work_in_receiver_kernel) {
        kernels.push_back(make_in0_sender_spec(IN0_NO_WORK_IN_RECV, false, true, IN0_RELAY_DFB));
    }
    if (has_in0_no_work_not_in_receiver_kernel) {
        kernels.push_back(make_in0_sender_spec(IN0_NO_WORK_NOT_IN_RECV, false, false, IN0_RELAY_DFB));
    }

    // ---- in0 receiver ----------------------------------------------------
    const bool has_in0_receiver_kernel = !in0_is_sharded && in0_mcast_receivers.num_cores() > 0;
    if (has_in0_receiver_kernel) {
        kernels.push_back(KernelSpec{
            .unique_id = IN0_RECEIVER,
            .source = "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
                      "reader_bmm_tile_layout_in0_receiver_metal2.cpp",
            .dfb_bindings =
                {
                    DFBBinding{
                        .dfb_spec_name = IN0_DFB,
                        .accessor_name = "in0",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },
                },
            .semaphore_bindings = in0_mcast_sem_bindings,
            .compile_time_args =
                {
                    {"in0_block_num_tiles", in0_block_num_tiles},
                    {"num_blocks_inner_dim", num_blocks},
                    {"num_blocks_w_dim", out_num_blocks_x},
                    {"num_blocks_h_dim", out_num_blocks_y},
                    {"batch", in0_B},
                    {"get_batch_from_reader", 0u},
                },
            .runtime_arg_schema =
                {
                    .runtime_arg_names = {"in0_mcast_sender_noc_x", "in0_mcast_sender_noc_y"},
                },
            .hw_config = in0_sender_hw_config,
        });
    }

    // ---- in1 sender / output writer --------------------------------------
    // On this path the in1 multicast is skipped entirely (SKIP_MCAST), so the kernel's semaphore
    // objects are constructed but never used; they are still bound because the kernel constructs
    // them unconditionally.
    {
        KernelSpec in1_sender{
            .unique_id = IN1_SENDER_WRITER,
            .source =
                "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
                "reader_bmm_tile_layout_in1_sender_writer_padding_metal2.cpp",
            .compiler_options =
                {
                    .defines = KernelSpec::CompilerOptions::Defines(mm_kernel_in1_sender_writer_defines),
                },
            .dfb_bindings =
                {
                    DFBBinding{
                        .dfb_spec_name = IN1_DFB,
                        .accessor_name = "in1",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },
                    DFBBinding{
                        .dfb_spec_name = OUT_DFB,
                        .accessor_name = "out",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                },
            .semaphore_bindings =
                {
                    SemaphoreBinding{.semaphore_spec_name = SENDER_SEM, .accessor_name = "in1_mcast_sender"},
                    SemaphoreBinding{.semaphore_spec_name = RECEIVER_SEM, .accessor_name = "in1_mcast_receiver"},
                },
            .tensor_bindings =
                {
                    TensorBinding{.tensor_parameter_name = IN1, .accessor_name = "in1"},
                    TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "out"},
                },
            .compile_time_args =
                {
                    {"in1_tensor_stride_w", static_cast<uint32_t>(in1_tensor_stride_w)},
                    {"in1_tensor_stride_h", static_cast<uint32_t>(in1_tensor_stride_h)},
                    {"in1_tensor_next_block_stride", static_cast<uint32_t>(in1_tensor_next_block_stride)},
                    {"in1_tensor_next_w_dim_block_stride", static_cast<uint32_t>(in1_tensor_next_w_dim_block_stride)},
                    {"in1_block_w", in1_block_w},
                    {"in1_block_h", in0_block_w},
                    {"in1_block_num_tiles", in1_block_w * in0_block_w},
                    {"num_blocks_inner_dim", num_blocks},
                    {"num_blocks_w_dim", out_num_blocks_x},
                    {"num_blocks_h_dim", out_num_blocks_y},
                    {"in1_mcast_num_dests", 0u},
                    {"in1_mcast_num_cores", 0u},
                    {"KtNt", K * N},
                    {"batch", in0_B},
                    {"bcast_B", static_cast<uint32_t>(bcast_batch)},
                    {"batchB", 0u},
                    {"out_tensor_stride_w", 1u},
                    {"out_tensor_stride_h", N},
                    {"out_tensor_next_subblock_stride_w", out_subblock_w},
                    {"out_tensor_next_subblock_stride_h", out_subblock_h * N},
                    {"out_tensor_next_w_dim_block_stride", out_block_w},
                    {"out_tensor_next_h_dim_block_stride", out_block_h * N},
                    {"out_subblock_w", out_subblock_w},
                    {"out_subblock_h", out_subblock_h},
                    {"out_subblock_tile_count", out_subblock_w * out_subblock_h},
                    {"MtNt", M * N},
                    {"compact_output", 0u},
                    {"num_active", 0u},
                },
            .hw_config = in1_sender_hw_config,
        };

        Group<std::string> in1_sender_rta_names = {
            "in1_tensor_start_tile_id",
            "in1_mcast_dest_noc_start_x",
            "in1_mcast_dest_noc_start_y",
            "in1_mcast_dest_noc_end_x",
            "in1_mcast_dest_noc_end_y",
            "out_tensor_start_tile_id",
            "last_block_w",
            "out_num_nonzero_subblocks_h",
            "out_last_subblock_h",
            "padded_block_tiles_h_skip",
            "out_num_nonzero_subblocks_w",
            "out_last_num_nonzero_subblocks_w",
            "out_last_subblock_w",
            "padded_subblock_tiles_addr_skip",
            "padded_block_tiles_w_skip",
        };
        if (bias_tensor.has_value()) {
            in1_sender.dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = BIAS_DFB,
                .accessor_name = "bias",
                .endpoint_type = DFBEndpointType::PRODUCER,
            });
            in1_sender.tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = BIAS, .accessor_name = "bias"});
            in1_sender.compile_time_args.insert({"in3_tensor_stride_w", 1u});
            in1_sender_rta_names.push_back("in3_tensor_start_tile_id");
        }
        if (use_prefetcher_pipes) {
            // The pipes' receiver kernel: one accessor over every pipe, one of which is present on each
            // worker. It publishes each delivered entry to cb_in1 and acks it once compute is done.
            in1_sender.advanced_options.prefetcher_pipe_bindings = {
                {.pipe_parameter_names = prefetcher_pipe_names, .accessor_name = "in1"}};
        }
        if (!output_is_sharded) {
            in1_sender_rta_names.push_back("last_num_blocks_w_dim");
        }
        in1_sender.runtime_arg_schema = {.runtime_arg_names = std::move(in1_sender_rta_names)};
        kernels.push_back(std::move(in1_sender));
    }

    // ---- compute ----------------------------------------------------------
    {
        // The op resolves a TTNN ComputeKernelConfig, so translate that rather than building a Metal
        // config by hand.
        auto compute_hw = ttnn::to_compute_hardware_config(device->arch(), compute_kernel_config);
        // The legacy factory resolves dst_full_sync_en but never passes it to either descriptor
        // builder, so the descriptor default applied and this op has always ignored the knob. The
        // TTNN helper reads the resolved config and would hand the caller's value back, which would
        // change behaviour, so pin the legacy-default result. Preserved deliberately, not a fix.
        double_buffer_dest(compute_hw) = true;

        // When accumulating in fp32 with the K reduction split across blocks, the partials buffer
        // holds Float32 and is reloaded into DEST between blocks. Unless the reload's view is marked
        // UnpackToDest, that reload is routed through SrcA and rounded to TF32 (10 mantissa bits), so
        // the fp32 partial loses precision on every block boundary and accuracy degrades as the
        // number of K-blocks grows. The flag goes on the alias when bias forces a separate SrcA view
        // of the partials buffer (see above), else on the partials buffer itself. Metal 2.0 also
        // *requires* an explicit entry for a Float32 buffer a compute kernel consumes with
        // enable_32_bit_dest on, which is exactly this case.
        //
        // That requirement covers EVERY Float32 buffer this kernel consumes, not just the partials:
        // omitting one is a TT_FATAL at program build, where legacy defaulted silently. Legacy held
        // Default (= UnpackToSrc) everywhere except the partials view marked below, so reproduce
        // that -- one UnpackToDest, UnpackToSrc for the rest. The entries must track the bindings
        // below: add a Float32 buffer to the compute kernel and it needs one here too.
        if (fp32_dest_acc_en) {
            const DFBSpecName marked = bias_reload_alias ? INTERM0_ALIAS_DFB : INTERM0_DFB;
            const bool mark = interm0_data_format == tt::DataFormat::Float32;
            auto add_if_float32 = [&](const DFBSpecName& name, tt::DataFormat fmt) {
                if (fmt != tt::DataFormat::Float32) {
                    return;
                }
                unpack_modes(compute_hw).emplace(
                    name,
                    (mark && name == marked) ? tt::tt_metal::UnpackMode::UnpackToDest
                                             : tt::tt_metal::UnpackMode::UnpackToSrc);
            };
            add_if_float32(IN0_DFB, in0_data_format);
            add_if_float32(IN1_DFB, in1_data_format);
            add_if_float32(INTERM0_DFB, interm0_data_format);
            if (bias_reload_alias) {
                add_if_float32(INTERM0_ALIAS_DFB, interm0_data_format);
            }
            if (bias_tensor.has_value()) {
                add_if_float32(BIAS_DFB, bias_data_format);
            }
            if (in0_transpose_tile) {
                add_if_float32(IN0_TRANSPOSED_DFB, in0_data_format);
            }
        }

        Group<DFBBinding> compute_dfb_bindings = {
            DFBBinding{
                .dfb_spec_name = IN0_DFB,
                .accessor_name = "in0",
                .endpoint_type = DFBEndpointType::CONSUMER,
            },
            DFBBinding{
                .dfb_spec_name = IN1_DFB,
                .accessor_name = "in1",
                .endpoint_type = DFBEndpointType::CONSUMER,
            },
            DFBBinding{
                .dfb_spec_name = OUT_DFB,
                .accessor_name = "out",
                .endpoint_type = DFBEndpointType::PRODUCER,
            },
            // The partials buffer is filled and re-read by this kernel alone.
            DFBBinding{
                .dfb_spec_name = INTERM0_DFB,
                .accessor_name = "intermed0",
                .endpoint_type = DFBEndpointType::PRODUCER,
            },
            DFBBinding{
                .dfb_spec_name = INTERM0_DFB,
                .accessor_name = "intermed0",
                .endpoint_type = DFBEndpointType::CONSUMER,
            },
        };
        if (bias_reload_alias) {
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = INTERM0_ALIAS_DFB,
                .accessor_name = "intermed0_reload_alias",
                .endpoint_type = DFBEndpointType::PRODUCER,
            });
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = INTERM0_ALIAS_DFB,
                .accessor_name = "intermed0_reload_alias",
                .endpoint_type = DFBEndpointType::CONSUMER,
            });
        }
        if (bias_tensor.has_value()) {
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = BIAS_DFB,
                .accessor_name = "bias",
                .endpoint_type = DFBEndpointType::CONSUMER,
            });
        }
        if (in0_transpose_tile) {
            // in0 is read from the in0 buffer, transposed, and written into the transposed buffer,
            // which this same kernel then matmuls from.
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = IN0_TRANSPOSED_DFB,
                .accessor_name = "in0_transposed",
                .endpoint_type = DFBEndpointType::PRODUCER,
            });
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = IN0_TRANSPOSED_DFB,
                .accessor_name = "in0_transposed",
                .endpoint_type = DFBEndpointType::CONSUMER,
            });
        }

        KernelSpec::CompileTimeArgs compute_cta = {
            {"in0_block_w", in0_block_w},
            {"in0_num_subblocks", in0_num_subblocks},
            {"in0_block_num_tiles", in0_block_num_tiles},
            {"in0_subblock_num_tiles", in0_subblock_num_tiles},
            {"in1_num_subblocks", in1_num_subblocks},
            {"in1_block_num_tiles", in1_block_num_tiles},
            {"in1_block_w", in1_per_core_w},
            {"num_blocks_inner_dim", num_blocks},
            {"num_blocks_w_dim", out_num_blocks_x},
            {"num_blocks_h_dim", out_num_blocks_y},
            {"out_subblock_h", out_subblock_h},
            {"out_subblock_w", out_subblock_w},
            {"out_subblock_num_tiles", out_subblock_num_tiles},
            {"batch", in0_B},
            {"out_block_num_tiles", out_block_tiles},
            {"untilize_out", static_cast<uint32_t>(untilize_out)},
            {"get_batch_from_reader", 0u},
            {"bias_ntiles", in1_per_core_w},
        };
        if (bias_tensor.has_value()) {
            // true: row-0 broadcast ([N] / [...,1,N]); false: elementwise add_tiles.
            compute_cta.insert({"row_broadcast_bias", static_cast<uint32_t>(row_broadcast_bias ? 1u : 0u)});
        }
        if (fused_activation.has_value() && fused_activation.value().op_type != UnaryOpType::RELU) {
            using ttnn::operations::matmul::utilities::get_activation_params;
            const auto params = get_activation_params(fused_activation.value());
            compute_cta.insert({"activation_type", static_cast<uint32_t>(params.type)});
            compute_cta.insert({"activation_param0", params.param0});
            compute_cta.insert({"activation_param1", params.param1});
            compute_cta.insert({"activation_param2", params.param2});
        }

        kernels.push_back(KernelSpec{
            .unique_id = COMPUTE,
            .source = "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/"
                      "bmm_large_block_zm_fused_bias_activation_metal2.cpp",
            .compiler_options =
                {
                    .defines = KernelSpec::CompilerOptions::Defines(mm_kernel_defines),
                    // A legacy ComputeConfigDescriptor with no opt_level resolves to O3; Metal 2.0's
                    // CompilerOptions defaults to O2, so it has to be stated.
                    .opt_level = KernelBuildOptLevel::O3,
                },
            .dfb_bindings = std::move(compute_dfb_bindings),
            .compile_time_args = std::move(compute_cta),
            .hw_config = compute_hw,
        });
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Work units
    ////////////////////////////////////////////////////////////////////////////
    Group<WorkUnitSpec> work_units;
    work_units.push_back(WorkUnitSpec{
        .name = "in0_senders_with_work",
        .kernels = {IN0_SENDER, IN1_SENDER_WRITER, COMPUTE},
        .target_nodes = in0_mcast_cores_with_work_and_in_receiver_grid,
    });
    if (has_in0_no_work_in_receiver_kernel) {
        work_units.push_back(WorkUnitSpec{
            .name = "in0_senders_no_work_in_receiver_grid",
            .kernels = {IN0_NO_WORK_IN_RECV},
            .target_nodes = in0_mcast_cores_without_work_and_in_receiver_grid,
        });
    }
    if (has_in0_no_work_not_in_receiver_kernel) {
        work_units.push_back(WorkUnitSpec{
            .name = "in0_senders_no_work_outside_receiver_grid",
            .kernels = {IN0_NO_WORK_NOT_IN_RECV},
            .target_nodes = in0_mcast_cores_without_work_and_not_in_receiver_grid,
        });
    }
    if (has_in0_receiver_kernel) {
        work_units.push_back(WorkUnitSpec{
            .name = "in0_receivers",
            .kernels = {IN0_RECEIVER, IN1_SENDER_WRITER, COMPUTE},
            .target_nodes = in0_mcast_receivers,
        });
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Tensor parameters
    ////////////////////////////////////////////////////////////////////////////
    Group<TensorParameter> tensor_parameters = {
        TensorParameter{.unique_id = IN0, .spec = in0_tensor.tensor_spec()},
        TensorParameter{.unique_id = IN1, .spec = in1_tensor.tensor_spec()},
        TensorParameter{.unique_id = OUTPUT, .spec = out_tensor.tensor_spec()},
    };
    if (bias_tensor.has_value()) {
        tensor_parameters.push_back(TensorParameter{.unique_id = BIAS, .spec = bias_tensor->tensor_spec()});
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Runtime args (per-core loop)
    ////////////////////////////////////////////////////////////////////////////
    // Parameters for last row, col, or block. mcast_in0 replicates M across all cores
    // (num_blocks_y == 1), so any M-direction padding lives entirely within a single h-block.
    uint32_t last_per_core_N = N % per_core_N == 0 ? per_core_N : N % per_core_N;
    uint32_t last_out_block_w = last_per_core_N % out_block_w == 0 ? out_block_w : last_per_core_N % out_block_w;
    uint32_t last_out_num_blocks_w = ((last_per_core_N - 1) / out_block_w) + 1;
    uint32_t last_block_num_nonzero_subblocks_w = ((last_out_block_w - 1) / out_subblock_w) + 1;
    uint32_t last_subblock_of_last_block_w =
        last_out_block_w % out_subblock_w == 0 ? out_subblock_w : last_out_block_w % out_subblock_w;
    uint32_t last_block_padded_subblock_tiles_addr_skip =
        output_single_tile_size * (out_subblock_w - last_subblock_of_last_block_w);
    uint32_t last_block_padded_block_tiles_w_skip =
        (out_subblock_w * out_subblock_h) * (out_block_w / out_subblock_w - last_block_num_nonzero_subblocks_w);

    // M-direction padding when M < per_core_M. With num_blocks_y == 1 the last (only) h-block
    // holds last_out_block_h valid tile rows out of out_block_h.
    uint32_t in0_last_per_core_M = M < per_core_M ? M : per_core_M;
    uint32_t in0_last_out_block_h =
        in0_last_per_core_M % out_block_h == 0 ? out_block_h : in0_last_per_core_M % out_block_h;
    uint32_t in0_last_block_num_nonzero_subblocks_h = ((in0_last_out_block_h - 1) / out_subblock_h) + 1;
    uint32_t in0_last_subblock_of_last_block_h =
        in0_last_out_block_h % out_subblock_h == 0 ? out_subblock_h : in0_last_out_block_h % out_subblock_h;
    uint32_t in0_last_block_padded_block_tiles_h_skip =
        (out_block_h / out_subblock_h - in0_last_block_num_nonzero_subblocks_h) * (out_block_w * out_subblock_h);

    CoreCoord start_core_noc = top_left_core_physical;
    CoreCoord end_core_noc = bottom_right_core_physical;
    if (in0_noc == tt::tt_metal::NOC::NOC_1) {
        std::swap(start_core_noc, end_core_noc);
    }

    KernelRunArgs in0_sender_run_args{.kernel = IN0_SENDER};
    KernelRunArgs in0_no_work_in_recv_run_args{.kernel = IN0_NO_WORK_IN_RECV};
    KernelRunArgs in0_no_work_not_in_recv_run_args{.kernel = IN0_NO_WORK_NOT_IN_RECV};
    KernelRunArgs in0_receiver_run_args{.kernel = IN0_RECEIVER};
    KernelRunArgs in1_sender_run_args{.kernel = IN1_SENDER_WRITER};

    const auto& cores = corerange_to_cores(all_cores, std::nullopt, row_major);
    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& core = cores[i];
        uint32_t output_idx_x = i % num_blocks_x;
        uint32_t output_idx_y = i / num_blocks_x;

        if (in0_is_sharded) {
            std::vector<uint32_t> varargs;
            varargs.reserve(in0_mcast_noc_x.size() + in0_mcast_noc_y.size());
            varargs.insert(varargs.end(), in0_mcast_noc_x.begin(), in0_mcast_noc_x.end());
            varargs.insert(varargs.end(), in0_mcast_noc_y.begin(), in0_mcast_noc_y.end());

            auto& dst = (i < num_cores_with_work)            ? in0_sender_run_args
                        : (i < in0_mcast_receiver_num_dests) ? in0_no_work_in_recv_run_args
                                                             : in0_no_work_not_in_recv_run_args;
            AddRuntimeArgsForNode(
                dst.runtime_arg_values,
                core,
                {{"sender_id", i},
                 {"in0_mcast_dest_noc_start_x", static_cast<uint32_t>(start_core_noc.x)},
                 {"in0_mcast_dest_noc_start_y", static_cast<uint32_t>(start_core_noc.y)},
                 {"in0_mcast_dest_noc_end_x", static_cast<uint32_t>(end_core_noc.x)},
                 {"in0_mcast_dest_noc_end_y", static_cast<uint32_t>(end_core_noc.y)}});
            dst.advanced_options.runtime_varargs[core] = std::move(varargs);
        }
        // in0 sender and in1 sender
        else if (core == start_core) {
            AddRuntimeArgsForNode(
                in0_sender_run_args.runtime_arg_values,
                core,
                {{"in0_tensor_start_tile_id", static_cast<uint32_t>(in0_tensor_start_tile_id_stride * output_idx_y)},
                 {"in0_mcast_dest_noc_start_x", static_cast<uint32_t>(start_core_noc.x)},
                 {"in0_mcast_dest_noc_start_y", static_cast<uint32_t>(start_core_noc.y)},
                 {"in0_mcast_dest_noc_end_x", static_cast<uint32_t>(end_core_noc.x)},
                 {"in0_mcast_dest_noc_end_y", static_cast<uint32_t>(end_core_noc.y)},
                 {"last_block_h", in0_last_out_block_h}});
        }
        // in0 receiver and in1 sender
        else {
            AddRuntimeArgsForNode(
                in0_receiver_run_args.runtime_arg_values,
                core,
                {{"in0_mcast_sender_noc_x", static_cast<uint32_t>(top_left_core_physical.x)},
                 {"in0_mcast_sender_noc_y", static_cast<uint32_t>(top_left_core_physical.y)}});
        }

        if (i < num_cores_with_work) {
            const bool last_x = (output_idx_x == num_blocks_x - 1);
            AddRuntimeArgsForNode(
                in1_sender_run_args.runtime_arg_values,
                core,
                {{"in1_tensor_start_tile_id", static_cast<uint32_t>(in1_tensor_start_tile_id_stride * output_idx_x)},
                 {"in1_mcast_dest_noc_start_x", 0u},
                 {"in1_mcast_dest_noc_start_y", 0u},
                 {"in1_mcast_dest_noc_end_x", 0u},
                 {"in1_mcast_dest_noc_end_y", 0u},
                 {"out_tensor_start_tile_id", (output_idx_x * per_core_N) + (output_idx_y * per_core_M * N)},
                 // padding args (READER)
                 {"last_block_w", last_x ? last_out_block_w : out_block_w},
                 // padding args (WRITER)
                 {"out_num_nonzero_subblocks_h", in0_last_block_num_nonzero_subblocks_h},
                 {"out_last_subblock_h", in0_last_subblock_of_last_block_h},
                 {"padded_block_tiles_h_skip", in0_last_block_padded_block_tiles_h_skip},
                 {"out_num_nonzero_subblocks_w", out_block_w / out_subblock_w},
                 {"out_last_num_nonzero_subblocks_w",
                  last_x ? last_block_num_nonzero_subblocks_w : out_block_w / out_subblock_w},
                 {"out_last_subblock_w", last_x ? last_subblock_of_last_block_w : out_subblock_w},
                 {"padded_subblock_tiles_addr_skip", last_x ? last_block_padded_subblock_tiles_addr_skip : 0u},
                 {"padded_block_tiles_w_skip", last_x ? last_block_padded_block_tiles_w_skip : 0u}});
            if (bias_tensor.has_value()) {
                in1_sender_run_args.runtime_arg_values["in3_tensor_start_tile_id"][core] = per_core_N * output_idx_x;
            }
            if (!output_is_sharded) {
                in1_sender_run_args.runtime_arg_values["last_num_blocks_w_dim"][core] =
                    last_x ? last_out_num_blocks_w : out_num_blocks_x;
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Assemble
    ////////////////////////////////////////////////////////////////////////////
    ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(in0_sender_run_args));
    if (has_in0_no_work_in_receiver_kernel) {
        run_args.kernel_run_args.push_back(std::move(in0_no_work_in_recv_run_args));
    }
    if (has_in0_no_work_not_in_receiver_kernel) {
        run_args.kernel_run_args.push_back(std::move(in0_no_work_not_in_recv_run_args));
    }
    if (has_in0_receiver_kernel) {
        run_args.kernel_run_args.push_back(std::move(in0_receiver_run_args));
    }
    run_args.kernel_run_args.push_back(std::move(in1_sender_run_args));

    run_args.tensor_args = {
        {IN0, in0_tensor},
        {IN1, in1_tensor},
        {OUTPUT, out_tensor},
    };
    if (bias_tensor.has_value()) {
        run_args.tensor_args.emplace(BIAS, *bias_tensor);
    }
    for (size_t p = 0; p < prefetcher_pipes.size(); ++p) {
        run_args.advanced_options.prefetcher_pipe_args.emplace(
            prefetcher_pipe_names[p], PrefetcherPipeArgument{*prefetcher_pipes[p]});
    }

    ProgramSpec spec{
        .name = "matmul_multi_core_reuse_mcast_1d_in0",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .semaphores = std::move(semaphores),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = std::move(work_units),
        .advanced_options = {.prefetcher_pipe_parameters = std::move(prefetcher_pipe_parameters)},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

static ttnn::device_operation::ProgramArtifacts create_program_mcast_in1_artifacts(
    const ttnn::Tensor& a,
    tt_metal::IDevice* device,
    bool fp32_dest_acc_en,
    bool packer_l1_acc,
    CoreCoord compute_with_storage_grid_size,
    ttnn::operations::compute_throttle_utils::ThrottleLevel throttle_level,
    const DeviceComputeKernelConfig& compute_kernel_config,
    uint32_t in0_B,
    uint32_t in1_B,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    bool bcast_batch,
    bool transpose_a,
    bool transpose_b,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t out_block_h,
    uint32_t out_block_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    std::optional<UnaryWithParam> fused_activation,
    const MeshTensor& in0_tensor,
    const MeshTensor& in1_tensor,
    ttsl::optional_reference<const MeshTensor> bias_tensor,
    const MeshTensor& out_tensor,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const tt::tt_metal::Tile& bias_tile,
    const tt::tt_metal::Tile& output_tile,
    tt::DataFormat in0_data_format,
    tt::DataFormat in1_data_format,
    tt::DataFormat bias_data_format,
    tt::DataFormat output_data_format,
    bool in0_is_sharded,
    bool output_is_sharded,
    bool untilize_out,
    bool row_broadcast_bias = true,
    CoreCoord sub_device_start_core = {0, 0}) {
    // currently only support transpose of the full tile
    bool in0_transpose_tile = in0_tile.get_transpose_of_faces() && in0_tile.get_transpose_within_face();
    bool in1_transpose_tile = in1_tile.get_transpose_of_faces() && in1_tile.get_transpose_within_face();

    bool fuse_op = false;

    uint32_t num_blocks = K / in0_block_w;
    // Only enable packer l1 accumulation when there are num_blocks > 2, otherwise
    // unnecessary overhead for reconfigs are added. Last iteration of l1 accumulation
    // does a spill and reload, so need more than 2 blocks to use l1 acc for packer
    // For bias, last iteration of l1 acc remains in intermediate buffer, does not spill and reload
    bool packer_l1_acc_en = packer_l1_acc && (((bias_tensor.has_value()) && num_blocks > 1) || (num_blocks > 2));

    // if fp32 enabled then we pack fp32 in l1, if not, then we pack fp16 in l1
    tt::DataFormat interm0_data_format = packer_l1_acc_en
                                             ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                                             : (fp32_dest_acc_en ? tt::DataFormat::Float32 : output_data_format);

    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    uint32_t bias_single_tile_size = bias_tile.get_tile_size(bias_data_format);

    // Tiles whose size is not a multiple of the DRAM alignment (e.g. bfp8 32x16 = 544B on
    // Blackhole's 64B alignment) are padded to it in DRAM. The interleaved reader copies tiles at
    // the padded stride, so the in0/in1/bias CBs must hold pages at the aligned stride and the
    // reader/unpacker walk tiles at the same stride. No-op when already aligned (all bf16 tiles,
    // 32-wide bfp8, Wormhole). Replaces the staging-CB workaround. Sharded buffers are backed by the
    // tensor buffer and keep their natural entry size.
    const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();
    uint32_t in0_aligned_tile_size =
        in0_is_sharded ? in0_single_tile_size : tt::align(in0_single_tile_size, dram_alignment);
    uint32_t in1_aligned_tile_size = tt::align(in1_single_tile_size, dram_alignment);
    // Bias buffer entries must be padded to the DRAM alignment so the reader's L1 write stride
    // matches the DRAM page stride (e.g. 64B on Blackhole for a 32B (1,16) bf16 bias tile).
    // Mirrors in0/in1 above and the dram_sharded factory. No-op on Wormhole and for
    // tiles already >= dram_alignment.
    uint32_t bias_aligned_tile_size = tt::align(bias_single_tile_size, dram_alignment);
    uint32_t output_single_tile_size = output_tile.get_tile_size(output_data_format);
    uint32_t interm0_single_tile_size = output_tile.get_tile_size(interm0_data_format);

    bool do_not_inplace_interm0_out_dfb = output_is_sharded && (per_core_M != out_block_h);

    uint32_t in0_block_h = out_block_h;
    uint32_t in1_block_w = out_block_w;
    uint32_t in0_num_blocks_y = per_core_M / out_block_h;
    uint32_t in1_num_blocks_x = per_core_N / out_block_w;
    uint32_t out_num_blocks_x = in1_num_blocks_x;
    uint32_t out_num_blocks_y = in0_num_blocks_y;

    uint32_t in0_block_tiles = in0_block_h * in0_block_w;
    uint32_t in0_dfb_tiles = in0_block_tiles;

    if (in0_B == 1 && in1_B > 1) {
        in0_dfb_tiles = per_core_M * num_blocks * in0_block_w;
    } else if (in0_is_sharded) {
        in0_dfb_tiles = num_blocks * per_core_M * in0_block_w * in0_B;
    } else if (in0_B * num_blocks > 1) {
        in0_dfb_tiles = in0_dfb_tiles * 2;  // double buffer
    }
    uint32_t in0_dfb_size = in0_dfb_tiles * in0_aligned_tile_size;

    const auto& a_shape_logical = operations::matmul::utilities::get_matmul_tensor_logical_shape(a, transpose_a);
    const auto in0_last_ktile_w = transpose_a ? 0 : a_shape_logical[-1] % in0_tile.get_width();
    const auto in0_last_ktile_h = transpose_a ? a_shape_logical[-1] % in0_tile.get_width() : 0;
    TT_FATAL(
        in0_last_ktile_w == 0 || in0_last_ktile_h == 0,
        "At most one of in0_last_ktile_w ({}) and in0_last_ktile_h ({}) can be non-zero",
        in0_last_ktile_w,
        in0_last_ktile_h);

    bool extract_shard_sub_blocks = false;
    uint32_t in0_shard_height_in_tiles = 0;
    uint32_t in0_shard_width_in_tiles = 0;
    if (in0_is_sharded) {
        in0_shard_height_in_tiles = in0_tensor.shard_spec()->shape[0] / in0_tile.get_height();
        in0_shard_width_in_tiles = in0_tensor.shard_spec()->shape[1] / in0_tile.get_width();
        // Do a real per-block copy (not point the in0 buffer at L1) when K needs splitting, or when there's more
        // than 1 row-block AND col-block: a row-block's data is needed twice, but advancing lands on the wrong one.
        if (in0_shard_width_in_tiles / in0_block_w > 1 || (in0_num_blocks_y > 1 && in1_num_blocks_x > 1)) {
            extract_shard_sub_blocks = true;
        }
    }
    uint32_t in2_dfb_tiles = in0_block_tiles;
    uint32_t in2_dfb_size = in2_dfb_tiles * in0_single_tile_size;

    uint32_t in1_block_tiles = out_block_w * in0_block_w;
    uint32_t in1_dfb_tiles = in1_block_tiles;
    if (in1_B * num_blocks > 1) {
        in1_dfb_tiles = in1_dfb_tiles * 2;  // double buffer
    }
    uint32_t in1_dfb_size = in1_dfb_tiles * in1_aligned_tile_size;

    uint32_t out_block_tiles = out_block_h * out_block_w;
    uint32_t out_shard_tiles = per_core_M * per_core_N;
    uint32_t out_dfb_tiles = out_block_tiles;  // No double buffer
    if (output_is_sharded) {
        out_dfb_tiles = out_shard_tiles;
    }
    uint32_t out_dfb_size = out_dfb_tiles * output_single_tile_size;
    uint32_t interm0_dfb_tiles = out_block_tiles;  // No double buffer
    uint32_t interm0_dfb_size = interm0_dfb_tiles * interm0_single_tile_size;

    uint32_t in3_block_tiles = out_block_w;
    uint32_t in3_dfb_tiles = in3_block_tiles;  // No double buffer
    uint32_t in3_dfb_size = in3_dfb_tiles * bias_aligned_tile_size;

    CoreCoord start_core = sub_device_start_core;

    // The matmul region is the rectangle of size `compute_with_storage_grid_size`
    // anchored at `start_core`. Callers must ensure this rectangle lies entirely
    // within the active sub-device's worker cores (validated upstream).
    CoreRangeSet matmul_core_rect(CoreRange(
        start_core,
        CoreCoord(
            start_core.x + compute_with_storage_grid_size.x - 1, start_core.y + compute_with_storage_grid_size.y - 1)));

    uint32_t num_blocks_y = ((M - 1) / per_core_M) + 1;
    uint32_t num_blocks_x = ((N - 1) / per_core_N) + 1;
    uint32_t num_blocks_total = num_blocks_y * num_blocks_x;
    uint32_t num_cores = num_blocks_total;

    TT_FATAL(
        num_blocks_x == 1,
        "mcast_in1 requires N ({}) to fit within one per_core_N block ({}); got num_blocks_x={}",
        N,
        per_core_N,
        num_blocks_x);
    TT_FATAL(
        ((N - 1) / out_block_w) + 1 == out_num_blocks_x,
        "mcast_in1 requires the logical N tail to be in the final internal W block; got N={}, per_core_N={}, "
        "out_block_w={}",
        N,
        per_core_N,
        out_block_w);
    TT_FATAL(
        num_blocks_y != 1 || ((M - 1) / out_block_h) + 1 == out_num_blocks_y,
        "a single-Y mcast_in1 sender requires the logical M tail to be in the final internal H block; got M={}, "
        "per_core_M={}, out_block_h={}",
        M,
        per_core_M,
        out_block_h);
    TT_FATAL(
        num_blocks_y != 1 || M % out_block_h == 0 || out_num_blocks_y == 1,
        "a single-Y mcast_in1 sender supports a partial final H block only with one internal H block; got M={}, "
        "per_core_M={}, out_block_h={}",
        M,
        per_core_M,
        out_block_h);

    constexpr bool row_major = true;
    CoreRangeSet all_cores =
        tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(start_core, num_cores, matmul_core_rect, row_major);
    CoreRange in1_mcast_receiver_cores_bounding_box = all_cores.bounding_box();
    uint32_t in1_mcast_receiver_num_cores = in1_mcast_receiver_cores_bounding_box.size();  // always mcast to full grid

    CoreRange in1_mcast_sender(start_core, start_core);
    CoreRangeSet in1_mcast_receivers;
    if (in1_mcast_receiver_num_cores > 1) {
        // Check against the actual rectangle width (instead of bare grid_size.x-1) so that
        // sub-devices anchored away from (0, 0) wrap correctly.
        auto receiver_start_core = compute_with_storage_grid_size.x > 1 ? CoreCoord{start_core.x + 1, start_core.y}
                                                                        : CoreCoord{start_core.x, start_core.y + 1};
        in1_mcast_receivers = tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(
            receiver_start_core, num_cores - 1, matmul_core_rect, row_major);
    }

    CoreCoord top_left_core = in1_mcast_receiver_cores_bounding_box.start_coord;
    CoreCoord bottom_right_core = in1_mcast_receiver_cores_bounding_box.end_coord;
    auto top_left_core_physical = device->worker_core_from_logical_core(top_left_core);
    auto bottom_right_core_physical = device->worker_core_from_logical_core(bottom_right_core);

    const auto& a_padded_shape = operations::matmul::utilities::get_matmul_tensor_padded_shape(a, transpose_a);
    const uint32_t M_per_batch = a_padded_shape[-2] / in0_tile.get_height();
    const auto [in0_tensor_stride_w, in0_tensor_stride_h] =
        operations::matmul::utilities::get_in0_transpose_strides(M, M_per_batch, transpose_a, K);
    const auto in0_tensor_next_block_stride = in0_block_w * in0_tensor_stride_w;
    const auto in0_tensor_next_h_dim_block_stride = in0_block_h * in0_tensor_stride_h;
    const auto in0_tensor_start_tile_id_stride = per_core_M * in0_tensor_stride_h;

    const auto in1_tensor_stride_w = transpose_b ? K : 1;
    const auto in1_tensor_stride_h = transpose_b ? 1 : N;
    const auto in1_tensor_next_block_stride = in0_block_w * in1_tensor_stride_h;
    const auto in1_tensor_next_w_dim_block_stride = in1_block_w * in1_tensor_stride_w;
    const auto in1_tensor_start_tile_id_stride = per_core_N * in1_tensor_stride_w;

    const bool reuse_in0_in_dfb = (in0_B == 1 && in1_B > 1) && !in0_is_sharded && !output_is_sharded && !bcast_batch &&
                                  !fused_activation.has_value();

    // create_program_artifacts always passes std::nullopt for the fused-op signaler, so the CCL
    // fused-op path is unreachable here. The Metal 2.0 kernel forks do not carry it: MatmulOpReceiver
    // and OpSignaler read positional runtime args from outside this op's directory, which named
    // arguments cannot supply. Fail loudly rather than silently building a program without it.
    TT_FATAL(!fuse_op, "matmul_multicore_reuse_mcast_1d: the Metal 2.0 path does not support fused CCL ops");

    // ------------------------------------------------------------------
    // Spec-scope resource names. Declared function-local rather than at file scope: the matmul
    // factory .cpp files share one unity-build target, so file-scope constants with these names
    // would collide as sibling factories are ported.
    // ------------------------------------------------------------------
    const KernelSpecName IN0_SENDER{"in0_sender"};
    const KernelSpecName IN1_SENDER_WRITER{"in1_sender_writer"};
    const KernelSpecName IN1_RECEIVER_WRITER{"in1_receiver_writer"};
    const KernelSpecName COMPUTE{"compute"};

    const DFBSpecName IN0_DFB{"in0"};
    const DFBSpecName IN1_DFB{"in1"};
    const DFBSpecName IN0_SHARDED_DFB{"in0_sharded"};
    const DFBSpecName OUT_DFB{"out"};
    const DFBSpecName INTERM0_DFB{"intermed0"};
    const DFBSpecName INTERM0_ALIAS_DFB{"intermed0_reload_alias"};
    const DFBSpecName BIAS_DFB{"bias"};
    const DFBSpecName IN0_TRANSPOSED_DFB{"in0_transposed"};

    const TensorParamName IN0{"in0"};
    const TensorParamName IN1{"in1"};
    const TensorParamName OUTPUT{"output"};
    const TensorParamName BIAS{"bias"};

    const SemaphoreSpecName SENDER_SEM{"in1_mcast_sender"};
    const SemaphoreSpecName RECEIVER_SEM{"in1_mcast_receiver"};

    // ------------------------------------------------------------------
    // Compute-kernel derived sizes (unchanged from the legacy factory)
    // ------------------------------------------------------------------
    uint32_t in0_num_subblocks = (out_block_h / out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;

    uint32_t in1_num_subblocks = (out_block_w / out_subblock_w);
    uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks;
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    // See create_program_mcast_in0_artifacts: the fused bias add reads the partials buffer via SrcA,
    // so when bias is present the reload copies through an alias view of the same SRAM that carries
    // UnpackToDest, while the bias add keeps reading the partials buffer. The alias reaches the
    // compute kernel as the MM_PARTIALS_RELOAD_ALIAS define plus its own binding.
    const bool bias_reload_alias =
        fp32_dest_acc_en && interm0_data_format == tt::DataFormat::Float32 && bias_tensor.has_value();

    // ------------------------------------------------------------------
    // Defines
    // ------------------------------------------------------------------
    std::map<std::string, std::string> mm_kernel_defines;
    std::map<std::string, std::string> mm_kernel_in0_sender_defines;
    std::map<std::string, std::string> mm_kernel_in1_sender_writer_defines;
    std::map<std::string, std::string> mm_kernel_in1_receiver_writer_defines;
    if (bias_tensor.has_value()) {
        mm_kernel_defines["FUSE_BIAS"] = "1";
        mm_kernel_in1_sender_writer_defines["FUSE_BIAS"] = "1";
        mm_kernel_in1_receiver_writer_defines["FUSE_BIAS"] = "1";
    }
    if (fused_activation.has_value()) {
        if (fused_activation.value().op_type == UnaryOpType::RELU) {
            mm_kernel_defines["PACK_RELU"] = "1";
        } else {
            mm_kernel_defines["SFPU_ACTIVATION"] = "1";
        }
    }
    if (packer_l1_acc_en) {
        mm_kernel_defines["PACKER_L1_ACC"] = "1";
    }
    if (fp32_dest_acc_en) {
        mm_kernel_defines["FP32_DEST_ACC_EN"] = "1";
    }
    if (in1_transpose_tile) {
        mm_kernel_defines["IN1_TRANSPOSE_TILE"] = "1";
    }
    // in0_transpose_tile selects which buffer the compute kernel matmuls from; it arrives as a
    // define because the in0_transposed buffer is only bound when the transpose is wanted.
    if (in0_transpose_tile) {
        mm_kernel_defines["IN0_TRANSPOSE_TILE"] = "1";
    }
    if (bias_reload_alias) {
        mm_kernel_defines["MM_PARTIALS_RELOAD_ALIAS"] = "1";
    }

    ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
        device->arch(), num_cores, mm_kernel_defines);
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(), num_cores, mm_kernel_defines, throttle_level);

    if (in0_is_sharded) {
        mm_kernel_in0_sender_defines["IN0_SHARDED"] = "1";
        // Whether the sender extracts sub-blocks out of the resident shard selects which buffer it
        // reads from, so it gates a binding and must be a define rather than a compile-time arg.
        if (extract_shard_sub_blocks) {
            mm_kernel_in0_sender_defines["EXTRACT_SHARD_SUB_BLOCKS"] = "1";
        }
    }
    if (output_is_sharded) {
        mm_kernel_in1_sender_writer_defines["OUT_SHARDED"] = "1";
        mm_kernel_in1_receiver_writer_defines["OUT_SHARDED"] = "1";
    }

    mm_kernel_in0_sender_defines["SKIP_MCAST"] = "1";

    if (in1_mcast_receiver_num_cores == 1) {
        mm_kernel_in1_sender_writer_defines["SKIP_MCAST"] = "1";
    }

    // in1 is the reader of weights/output writer, and we choose to make it use the optimized reader noc
    tt_metal::NOC in0_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    tt_metal::NOC in1_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());

    ////////////////////////////////////////////////////////////////////////////
    //                      Dataflow buffers
    ////////////////////////////////////////////////////////////////////////////
    const bool separate_out_and_interm0 = do_not_inplace_interm0_out_dfb ||
                                          (interm0_data_format != output_data_format) ||
                                          (untilize_out && (in1_num_subblocks > 1));
    const uint32_t interm0_total_size = separate_out_and_interm0 ? interm0_dfb_size : out_dfb_size;

    Group<DataflowBufferSpec> dataflow_buffers;

    Group<DFBSpecName> alias_group;
    if (!separate_out_and_interm0) {
        alias_group.push_back(OUT_DFB);
    }
    alias_group.push_back(INTERM0_DFB);
    if (bias_reload_alias) {
        alias_group.push_back(INTERM0_ALIAS_DFB);
    }
    const bool has_alias_group = alias_group.size() > 1;
    // Legacy put every shared index on ONE CBDescriptor whose single `tensor` field backed the whole
    // region, so when the partials share the output's region and the output is sharded, they are
    // backed by the output tensor too. The alias legality rule wants the same thing: either no member
    // of a group borrows, or all of them borrow from the same TensorParameter.
    const std::optional<TensorParamName> alias_group_borrow =
        (!separate_out_and_interm0 && output_is_sharded) ? std::optional<TensorParamName>(OUTPUT) : std::nullopt;
    auto alias_with_others = [&](const DFBSpecName& self) {
        Group<DFBSpecName> others;
        // A DFB that is not a member of the group aliases nothing. The output is exactly that case
        // when it has its own region: filtering only `self` would hand it the partials as aliases
        // while the partials never name it back, and the transitivity rule rejects a group whose
        // members disagree on their membership.
        const bool self_in_group = std::find(alias_group.begin(), alias_group.end(), self) != alias_group.end();
        if (has_alias_group && self_in_group) {
            for (const auto& name : alias_group) {
                if (name != self) {
                    others.push_back(name);
                }
            }
        }
        return others;
    };

    // in0. When in0 is sharded and the sender does NOT extract sub-blocks, the sender multicasts
    // straight out of the resident shard, so this buffer is borrowed onto the in0 tensor.
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = IN0_DFB,
        .entry_size = in0_aligned_tile_size,
        .num_entries = in0_dfb_size / in0_aligned_tile_size,
        .data_format_metadata = in0_data_format,
        .tile_format_metadata = in0_tile,
        .borrowed_from =
            (in0_is_sharded && !extract_shard_sub_blocks) ? std::optional<TensorParamName>(IN0) : std::nullopt,
    });

    // in0 sharded: only present when the sender extracts sub-blocks out of the resident shard.
    if (in0_is_sharded && extract_shard_sub_blocks) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = IN0_SHARDED_DFB,
            .entry_size = in0_single_tile_size,
            .num_entries = in2_dfb_size / in0_single_tile_size,
            .data_format_metadata = in0_data_format,
            .tile_format_metadata = in0_tile,
            .borrowed_from = IN0,
        });
    }

    // in1
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = IN1_DFB,
        .entry_size = in1_aligned_tile_size,
        .num_entries = in1_dfb_size / in1_aligned_tile_size,
        .data_format_metadata = in1_data_format,
        .tile_format_metadata = in1_tile,
    });

    // output
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = out_dfb_size / output_single_tile_size,
        .data_format_metadata = output_data_format,
        .tile_format_metadata = output_tile,
        .borrowed_from = output_is_sharded ? std::optional<TensorParamName>(OUTPUT) : std::nullopt,
        .advanced_options = {.alias_with = alias_with_others(OUT_DFB)},
    });

    // partials
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = INTERM0_DFB,
        .entry_size = interm0_single_tile_size,
        .num_entries = interm0_total_size / interm0_single_tile_size,
        .data_format_metadata = interm0_data_format,
        .tile_format_metadata = output_tile,
        .borrowed_from = alias_group_borrow,
        .advanced_options = {.alias_with = alias_with_others(INTERM0_DFB)},
    });

    if (bias_reload_alias) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = INTERM0_ALIAS_DFB,
            .entry_size = interm0_single_tile_size,
            .num_entries = interm0_total_size / interm0_single_tile_size,
            .data_format_metadata = interm0_data_format,
            .tile_format_metadata = output_tile,
            .borrowed_from = alias_group_borrow,
            .advanced_options = {.alias_with = alias_with_others(INTERM0_ALIAS_DFB)},
        });
    }

    // bias
    if (bias_tensor.has_value()) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = BIAS_DFB,
            .entry_size = bias_aligned_tile_size,
            .num_entries = in3_dfb_size / bias_aligned_tile_size,
            .data_format_metadata = bias_data_format,
            .tile_format_metadata = bias_tile,
        });
    }

    // in0 transposed
    if (in0_transpose_tile) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = IN0_TRANSPOSED_DFB,
            .entry_size = in0_aligned_tile_size,
            .num_entries = in0_dfb_size / in0_aligned_tile_size,
            .data_format_metadata = in0_data_format,
            .tile_format_metadata = in0_tile,
        });
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Semaphores
    ////////////////////////////////////////////////////////////////////////////
    Group<SemaphoreSpec> semaphores = {
        SemaphoreSpec{.unique_id = SENDER_SEM, .target_nodes = all_cores},
        SemaphoreSpec{.unique_id = RECEIVER_SEM, .target_nodes = all_cores},
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      Kernels
    ////////////////////////////////////////////////////////////////////////////
    const auto in0_sender_hw_config =
        DataMovementGen1Config{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = in0_noc};
    const auto in1_writer_hw_config =
        DataMovementGen1Config{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = in1_noc};

    Group<KernelSpec> kernels;

    // ---- in0 sender (runs on every core; the in0 multicast is skipped here) ----
    {
        KernelSpec in0_sender{
            .unique_id = IN0_SENDER,
            .source =
                "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
                "reader_bmm_tile_layout_in0_sender_padding_metal2.cpp",
            .compiler_options =
                {
                    .defines = KernelSpec::CompilerOptions::Defines(mm_kernel_in0_sender_defines),
                },
            .dfb_bindings =
                {
                    DFBBinding{
                        .dfb_spec_name = IN0_DFB,
                        .accessor_name = "in0",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },
                },
            .semaphore_bindings =
                {
                    SemaphoreBinding{.semaphore_spec_name = SENDER_SEM, .accessor_name = "in0_mcast_sender"},
                    SemaphoreBinding{.semaphore_spec_name = RECEIVER_SEM, .accessor_name = "in0_mcast_receiver"},
                },
            .compile_time_args =
                {
                    {"in0_tensor_stride_w", static_cast<uint32_t>(in0_tensor_stride_w)},
                    {"in0_tensor_stride_h", static_cast<uint32_t>(in0_tensor_stride_h)},
                    {"in0_tensor_next_inner_dim_block_stride", static_cast<uint32_t>(in0_tensor_next_block_stride)},
                    {"in0_tensor_next_h_dim_block_stride", static_cast<uint32_t>(in0_tensor_next_h_dim_block_stride)},
                    {"in0_block_w", in0_block_w},
                    {"in0_block_h", in0_block_h},
                    {"in0_block_num_tiles", in0_block_w * in0_block_h},
                    {"in0_last_ktile_w", static_cast<uint32_t>(in0_last_ktile_w)},
                    {"in0_last_ktile_h", static_cast<uint32_t>(in0_last_ktile_h)},
                    {"shard_width_in_tiles", in0_shard_width_in_tiles},
                    {"shard_height_in_tiles", in0_shard_height_in_tiles},
                    {"num_blocks_inner_dim", num_blocks},
                    {"num_blocks_w_dim", out_num_blocks_x},
                    {"num_blocks_h_dim", out_num_blocks_y},
                    {"in0_mcast_num_dests", 0u},
                    {"in0_mcast_num_cores", 0u},
                    {"MtKt", M * K},
                    {"in0_B", in0_B},
                    {"in1_B", in1_B},
                    {"in0_reuse_in_dfb", static_cast<uint32_t>(reuse_in0_in_dfb)},
                    {"batchB", 0u},
                    {"bcast_A", 1u},
                    {"get_batch_from_reader", 0u},
                    {"num_active", 0u},
                },
            .runtime_arg_schema =
                {
                    .runtime_arg_names =
                        {"in0_tensor_start_tile_id",
                         "in0_mcast_dest_noc_start_x",
                         "in0_mcast_dest_noc_start_y",
                         "in0_mcast_dest_noc_end_x",
                         "in0_mcast_dest_noc_end_y",
                         "last_block_h"},
                },
            .hw_config = in0_sender_hw_config,
        };
        // The interleaved path reads in0 through an accessor; when in0 is sharded the data is
        // already resident and reached through the borrowed buffer instead.
        if (!in0_is_sharded) {
            in0_sender.tensor_bindings = {
                TensorBinding{.tensor_parameter_name = IN0, .accessor_name = "in0"},
            };
        }
        if (in0_is_sharded && extract_shard_sub_blocks) {
            // The resident shard is read through a raw pointer, never through the FIFO, so the one
            // kernel that touches it carries both endpoints.
            in0_sender.dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = IN0_SHARDED_DFB,
                .accessor_name = "in0_sharded",
                .endpoint_type = DFBEndpointType::PRODUCER,
            });
            in0_sender.dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = IN0_SHARDED_DFB,
                .accessor_name = "in0_sharded",
                .endpoint_type = DFBEndpointType::CONSUMER,
            });
        }
        kernels.push_back(std::move(in0_sender));
    }

    // ---- in1 sender / output writer (start core only) ----------------------
    {
        KernelSpec in1_sender{
            .unique_id = IN1_SENDER_WRITER,
            .source =
                "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
                "reader_bmm_tile_layout_in1_sender_writer_padding_metal2.cpp",
            .compiler_options =
                {
                    .defines = KernelSpec::CompilerOptions::Defines(mm_kernel_in1_sender_writer_defines),
                },
            .dfb_bindings =
                {
                    DFBBinding{
                        .dfb_spec_name = IN1_DFB,
                        .accessor_name = "in1",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },
                    DFBBinding{
                        .dfb_spec_name = OUT_DFB,
                        .accessor_name = "out",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                },
            .semaphore_bindings =
                {
                    SemaphoreBinding{.semaphore_spec_name = SENDER_SEM, .accessor_name = "in1_mcast_sender"},
                    SemaphoreBinding{.semaphore_spec_name = RECEIVER_SEM, .accessor_name = "in1_mcast_receiver"},
                },
            .tensor_bindings =
                {
                    TensorBinding{.tensor_parameter_name = IN1, .accessor_name = "in1"},
                    TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "out"},
                },
            .compile_time_args =
                {
                    {"in1_tensor_stride_w", static_cast<uint32_t>(in1_tensor_stride_w)},
                    {"in1_tensor_stride_h", static_cast<uint32_t>(in1_tensor_stride_h)},
                    {"in1_tensor_next_block_stride", static_cast<uint32_t>(in1_tensor_next_block_stride)},
                    {"in1_tensor_next_w_dim_block_stride", static_cast<uint32_t>(in1_tensor_next_w_dim_block_stride)},
                    {"in1_block_w", in1_block_w},
                    {"in1_block_h", in0_block_w},
                    {"in1_block_num_tiles", in1_block_w * in0_block_w},
                    {"num_blocks_inner_dim", num_blocks},
                    {"num_blocks_w_dim", out_num_blocks_x},
                    {"num_blocks_h_dim", out_num_blocks_y},
                    {"in1_mcast_num_dests", num_cores - 1},
                    {"in1_mcast_num_cores", in1_mcast_receiver_num_cores - 1},
                    {"KtNt", K * N},
                    {"batch", reuse_in0_in_dfb ? in1_B : in0_B},
                    {"bcast_B", static_cast<uint32_t>(bcast_batch)},
                    {"batchB", 0u},
                    {"out_tensor_stride_w", 1u},
                    {"out_tensor_stride_h", N},
                    {"out_tensor_next_subblock_stride_w", out_subblock_w},
                    {"out_tensor_next_subblock_stride_h", out_subblock_h * N},
                    {"out_tensor_next_w_dim_block_stride", out_block_w},
                    {"out_tensor_next_h_dim_block_stride", out_block_h * N},
                    {"out_subblock_w", out_subblock_w},
                    {"out_subblock_h", out_subblock_h},
                    {"out_subblock_tile_count", out_subblock_w * out_subblock_h},
                    {"MtNt", M * N},
                    {"compact_output", 0u},
                    {"num_active", 0u},
                },
            .hw_config = in1_writer_hw_config,
        };

        Group<std::string> in1_sender_rta_names = {
            "in1_tensor_start_tile_id",
            "in1_mcast_dest_noc_start_x",
            "in1_mcast_dest_noc_start_y",
            "in1_mcast_dest_noc_end_x",
            "in1_mcast_dest_noc_end_y",
            "out_tensor_start_tile_id",
            "last_block_w",
            "out_num_nonzero_subblocks_h",
            "out_last_subblock_h",
            "padded_block_tiles_h_skip",
            "out_num_nonzero_subblocks_w",
            "out_last_num_nonzero_subblocks_w",
            "out_last_subblock_w",
            "padded_subblock_tiles_addr_skip",
            "padded_block_tiles_w_skip",
        };
        if (bias_tensor.has_value()) {
            in1_sender.dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = BIAS_DFB,
                .accessor_name = "bias",
                .endpoint_type = DFBEndpointType::PRODUCER,
            });
            in1_sender.tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = BIAS, .accessor_name = "bias"});
            in1_sender.compile_time_args.insert({"in3_tensor_stride_w", 1u});
            in1_sender_rta_names.push_back("in3_tensor_start_tile_id");
        }
        if (!output_is_sharded) {
            in1_sender_rta_names.push_back("last_num_blocks_w_dim");
        }
        in1_sender.runtime_arg_schema = {.runtime_arg_names = std::move(in1_sender_rta_names)};
        kernels.push_back(std::move(in1_sender));
    }

    // ---- in1 receiver / output writer -------------------------------------
    const bool has_in1_receiver_writer_kernel = in1_mcast_receivers.num_cores() > 0;
    if (has_in1_receiver_writer_kernel) {
        KernelSpec in1_receiver{
            .unique_id = IN1_RECEIVER_WRITER,
            .source =
                "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
                "reader_bmm_tile_layout_in1_receiver_writer_padding_metal2.cpp",
            .compiler_options =
                {
                    .defines = KernelSpec::CompilerOptions::Defines(mm_kernel_in1_receiver_writer_defines),
                },
            .dfb_bindings =
                {
                    DFBBinding{
                        .dfb_spec_name = IN1_DFB,
                        .accessor_name = "in1",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },
                    DFBBinding{
                        .dfb_spec_name = OUT_DFB,
                        .accessor_name = "out",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                },
            .semaphore_bindings =
                {
                    SemaphoreBinding{.semaphore_spec_name = SENDER_SEM, .accessor_name = "in1_mcast_sender"},
                    SemaphoreBinding{.semaphore_spec_name = RECEIVER_SEM, .accessor_name = "in1_mcast_receiver"},
                },
            .tensor_bindings =
                {
                    TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "out"},
                },
            .compile_time_args =
                {
                    {"in1_block_num_tiles", in1_block_w * in0_block_w},
                    {"num_blocks_inner_dim", num_blocks},
                    {"num_blocks_w_dim", out_num_blocks_x},
                    {"num_blocks_h_dim", out_num_blocks_y},
                    {"batch", reuse_in0_in_dfb ? in1_B : in0_B},
                    {"out_tensor_stride_w", 1u},
                    {"out_tensor_stride_h", N},
                    {"out_tensor_next_subblock_stride_w", out_subblock_w},
                    {"out_tensor_next_subblock_stride_h", out_subblock_h * N},
                    {"out_tensor_next_w_dim_block_stride", out_block_w},
                    {"out_tensor_next_h_dim_block_stride", out_block_h * N},
                    {"out_subblock_w", out_subblock_w},
                    {"out_subblock_h", out_subblock_h},
                    {"out_subblock_tile_count", out_subblock_w * out_subblock_h},
                    {"MtNt", M * N},
                },
            .hw_config = in1_writer_hw_config,
        };

        Group<std::string> in1_receiver_rta_names = {
            "in1_mcast_sender_noc_x",
            "in1_mcast_sender_noc_y",
            "out_tensor_start_tile_id",
            "out_num_nonzero_subblocks_h",
            "out_last_num_nonzero_subblocks_h",
            "out_last_subblock_h",
            "padded_block_tiles_h_skip",
            "out_num_nonzero_subblocks_w",
            "out_last_num_nonzero_subblocks_w",
            "out_last_subblock_w",
            "padded_subblock_tiles_addr_skip",
            "padded_block_tiles_w_skip",
        };
        if (bias_tensor.has_value()) {
            in1_receiver.dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = BIAS_DFB,
                .accessor_name = "bias",
                .endpoint_type = DFBEndpointType::PRODUCER,
            });
            in1_receiver.compile_time_args.insert({"in3_block_w", in1_block_w});
        }
        if (!output_is_sharded) {
            in1_receiver_rta_names.push_back("last_num_blocks_h_dim");
            in1_receiver_rta_names.push_back("last_num_blocks_w_dim");
        }
        in1_receiver.runtime_arg_schema = {.runtime_arg_names = std::move(in1_receiver_rta_names)};
        kernels.push_back(std::move(in1_receiver));
    }

    // ---- compute ----------------------------------------------------------
    {
        auto compute_hw = ttnn::to_compute_hardware_config(device->arch(), compute_kernel_config);
        // The legacy factory resolves dst_full_sync_en but never passes it to either descriptor
        // builder, so the descriptor default applied and this op has always ignored the knob. Pin the
        // legacy-default result rather than letting the TTNN helper hand the caller's value back.
        // Preserved deliberately, not a fix.
        double_buffer_dest(compute_hw) = true;

        // See create_program_mcast_in0_artifacts for why the fp32 partials reload needs UnpackToDest,
        // why the flag goes on the alias when bias is fused, and why every Float32 buffer this kernel
        // consumes needs an entry -- not just the partials.
        if (fp32_dest_acc_en) {
            const DFBSpecName marked = bias_reload_alias ? INTERM0_ALIAS_DFB : INTERM0_DFB;
            const bool mark = interm0_data_format == tt::DataFormat::Float32;
            auto add_if_float32 = [&](const DFBSpecName& name, tt::DataFormat fmt) {
                if (fmt != tt::DataFormat::Float32) {
                    return;
                }
                unpack_modes(compute_hw).emplace(
                    name,
                    (mark && name == marked) ? tt::tt_metal::UnpackMode::UnpackToDest
                                             : tt::tt_metal::UnpackMode::UnpackToSrc);
            };
            add_if_float32(IN0_DFB, in0_data_format);
            add_if_float32(IN1_DFB, in1_data_format);
            add_if_float32(INTERM0_DFB, interm0_data_format);
            if (bias_reload_alias) {
                add_if_float32(INTERM0_ALIAS_DFB, interm0_data_format);
            }
            if (bias_tensor.has_value()) {
                add_if_float32(BIAS_DFB, bias_data_format);
            }
            if (in0_transpose_tile) {
                add_if_float32(IN0_TRANSPOSED_DFB, in0_data_format);
            }
        }

        Group<DFBBinding> compute_dfb_bindings = {
            DFBBinding{
                .dfb_spec_name = IN0_DFB,
                .accessor_name = "in0",
                .endpoint_type = DFBEndpointType::CONSUMER,
            },
            DFBBinding{
                .dfb_spec_name = IN1_DFB,
                .accessor_name = "in1",
                .endpoint_type = DFBEndpointType::CONSUMER,
            },
            DFBBinding{
                .dfb_spec_name = OUT_DFB,
                .accessor_name = "out",
                .endpoint_type = DFBEndpointType::PRODUCER,
            },
            // The partials buffer is filled and re-read by this kernel alone.
            DFBBinding{
                .dfb_spec_name = INTERM0_DFB,
                .accessor_name = "intermed0",
                .endpoint_type = DFBEndpointType::PRODUCER,
            },
            DFBBinding{
                .dfb_spec_name = INTERM0_DFB,
                .accessor_name = "intermed0",
                .endpoint_type = DFBEndpointType::CONSUMER,
            },
        };
        if (bias_reload_alias) {
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = INTERM0_ALIAS_DFB,
                .accessor_name = "intermed0_reload_alias",
                .endpoint_type = DFBEndpointType::PRODUCER,
            });
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = INTERM0_ALIAS_DFB,
                .accessor_name = "intermed0_reload_alias",
                .endpoint_type = DFBEndpointType::CONSUMER,
            });
        }
        if (bias_tensor.has_value()) {
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = BIAS_DFB,
                .accessor_name = "bias",
                .endpoint_type = DFBEndpointType::CONSUMER,
            });
        }
        if (in0_transpose_tile) {
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = IN0_TRANSPOSED_DFB,
                .accessor_name = "in0_transposed",
                .endpoint_type = DFBEndpointType::PRODUCER,
            });
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = IN0_TRANSPOSED_DFB,
                .accessor_name = "in0_transposed",
                .endpoint_type = DFBEndpointType::CONSUMER,
            });
        }

        KernelSpec::CompileTimeArgs compute_cta = {
            {"in0_block_w", in0_block_w},
            {"in0_num_subblocks", in0_num_subblocks},
            {"in0_block_num_tiles", in0_block_num_tiles},
            {"in0_subblock_num_tiles", in0_subblock_num_tiles},
            {"in1_num_subblocks", in1_num_subblocks},
            {"in1_block_num_tiles", in1_block_num_tiles},
            {"in1_block_w", in1_per_core_w},
            {"num_blocks_inner_dim", num_blocks},
            {"num_blocks_w_dim", out_num_blocks_x},
            {"num_blocks_h_dim", out_num_blocks_y},
            {"out_subblock_h", out_subblock_h},
            {"out_subblock_w", out_subblock_w},
            {"out_subblock_num_tiles", out_subblock_num_tiles},
            {"batch", reuse_in0_in_dfb ? in1_B : in0_B},
            {"out_block_num_tiles", out_block_tiles},
            {"untilize_out", static_cast<uint32_t>(untilize_out)},
            {"get_batch_from_reader", 0u},
            {"bias_ntiles", in1_per_core_w},
        };
        if (bias_tensor.has_value()) {
            // true: row-0 broadcast ([N] / [...,1,N]); false: elementwise add_tiles.
            compute_cta.insert({"row_broadcast_bias", static_cast<uint32_t>(row_broadcast_bias ? 1u : 0u)});
        }
        if (fused_activation.has_value() && fused_activation.value().op_type != UnaryOpType::RELU) {
            using ttnn::operations::matmul::utilities::get_activation_params;
            const auto params = get_activation_params(fused_activation.value());
            compute_cta.insert({"activation_type", static_cast<uint32_t>(params.type)});
            compute_cta.insert({"activation_param0", params.param0});
            compute_cta.insert({"activation_param1", params.param1});
            compute_cta.insert({"activation_param2", params.param2});
        }

        kernels.push_back(KernelSpec{
            .unique_id = COMPUTE,
            .source = "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/"
                      "bmm_large_block_zm_fused_bias_activation_metal2.cpp",
            .compiler_options =
                {
                    .defines = KernelSpec::CompilerOptions::Defines(mm_kernel_defines),
                    // A legacy ComputeConfigDescriptor with no opt_level resolves to O3; Metal 2.0's
                    // CompilerOptions defaults to O2, so it has to be stated.
                    .opt_level = KernelBuildOptLevel::O3,
                },
            .dfb_bindings = std::move(compute_dfb_bindings),
            .compile_time_args = std::move(compute_cta),
            .hw_config = compute_hw,
        });
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Work units
    ////////////////////////////////////////////////////////////////////////////
    Group<WorkUnitSpec> work_units;
    work_units.push_back(WorkUnitSpec{
        .name = "in1_sender",
        .kernels = {IN0_SENDER, IN1_SENDER_WRITER, COMPUTE},
        .target_nodes = CoreRangeSet(in1_mcast_sender),
    });
    if (has_in1_receiver_writer_kernel) {
        work_units.push_back(WorkUnitSpec{
            .name = "in1_receivers",
            .kernels = {IN0_SENDER, IN1_RECEIVER_WRITER, COMPUTE},
            .target_nodes = in1_mcast_receivers,
        });
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Tensor parameters
    ////////////////////////////////////////////////////////////////////////////
    Group<TensorParameter> tensor_parameters = {
        TensorParameter{.unique_id = IN0, .spec = in0_tensor.tensor_spec()},
        TensorParameter{.unique_id = IN1, .spec = in1_tensor.tensor_spec()},
        TensorParameter{.unique_id = OUTPUT, .spec = out_tensor.tensor_spec()},
    };
    if (bias_tensor.has_value()) {
        tensor_parameters.push_back(TensorParameter{.unique_id = BIAS, .spec = bias_tensor->tensor_spec()});
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Runtime args (per-core loop)
    ////////////////////////////////////////////////////////////////////////////
    // Parameters for last row, col, or block
    uint32_t last_per_core_M = M % per_core_M == 0 ? per_core_M : M % per_core_M;
    uint32_t last_out_block_h = last_per_core_M % out_block_h == 0 ? out_block_h : last_per_core_M % out_block_h;
    uint32_t last_out_num_blocks_h = ((last_per_core_M - 1) / out_block_h) + 1;
    uint32_t last_block_num_nonzero_subblocks_h = ((last_out_block_h - 1) / out_subblock_h) + 1;
    uint32_t last_subblock_of_last_block_h =
        last_out_block_h % out_subblock_h == 0 ? out_subblock_h : last_out_block_h % out_subblock_h;
    uint32_t last_block_padded_block_tiles_h_skip =
        (out_block_h / out_subblock_h - last_block_num_nonzero_subblocks_h) * (out_block_w * out_subblock_h);

    // W-dim padding parameters for the last block in X. Mirrors the mcast_in0
    // factory. Without these, the receiver-writer emits full per_core_N-wide
    // writes for the last X block even when N is not divisible by per_core_N,
    // sending pages past the tensor's logical extent and producing OOB writes
    // past the L1 allocation on the far banks.
    uint32_t last_per_core_N = N % per_core_N == 0 ? per_core_N : N % per_core_N;
    uint32_t last_out_block_w = last_per_core_N % out_block_w == 0 ? out_block_w : last_per_core_N % out_block_w;
    uint32_t last_out_num_blocks_w = ((last_per_core_N - 1) / out_block_w) + 1;
    uint32_t last_block_num_nonzero_subblocks_w = ((last_out_block_w - 1) / out_subblock_w) + 1;
    uint32_t last_subblock_of_last_block_w =
        last_out_block_w % out_subblock_w == 0 ? out_subblock_w : last_out_block_w % out_subblock_w;
    uint32_t last_block_padded_subblock_tiles_addr_skip =
        output_single_tile_size * (out_subblock_w - last_subblock_of_last_block_w);
    uint32_t last_block_padded_block_tiles_w_skip =
        (out_subblock_w * out_subblock_h) * (out_block_w / out_subblock_w - last_block_num_nonzero_subblocks_w);

    CoreCoord start_core_noc = bottom_right_core_physical;
    CoreCoord end_core_noc = top_left_core_physical;
    if (in1_noc == tt::tt_metal::NOC::NOC_0) {
        std::swap(start_core_noc, end_core_noc);
    }

    KernelRunArgs in0_sender_run_args{.kernel = IN0_SENDER};
    KernelRunArgs in1_sender_run_args{.kernel = IN1_SENDER_WRITER};
    KernelRunArgs in1_receiver_run_args{.kernel = IN1_RECEIVER_WRITER};

    const auto& cores = corerange_to_cores(all_cores, std::nullopt, row_major);
    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& core = cores[i];
        uint32_t output_idx_x = i / num_blocks_y;
        uint32_t output_idx_y = i % num_blocks_y;

        // The sender can independently be the last block in either dimension.
        const bool last_y = (output_idx_y == num_blocks_y - 1);
        const bool last_x = (output_idx_x == num_blocks_x - 1);

        // in0 sender and in1 sender
        if (core == start_core) {
            AddRuntimeArgsForNode(
                in1_sender_run_args.runtime_arg_values,
                core,
                {{"in1_tensor_start_tile_id", static_cast<uint32_t>(in1_tensor_start_tile_id_stride * output_idx_x)},
                 {"in1_mcast_dest_noc_start_x", static_cast<uint32_t>(start_core_noc.x)},
                 {"in1_mcast_dest_noc_start_y", static_cast<uint32_t>(start_core_noc.y)},
                 {"in1_mcast_dest_noc_end_x", static_cast<uint32_t>(end_core_noc.x)},
                 {"in1_mcast_dest_noc_end_y", static_cast<uint32_t>(end_core_noc.y)},
                 {"out_tensor_start_tile_id", (output_idx_x * per_core_N) + (output_idx_y * per_core_M * N)},
                 // padding args (READER)
                 {"last_block_w", last_x ? last_out_block_w : out_block_w},
                 // padding args (WRITER)
                 {"out_num_nonzero_subblocks_h",
                  last_y ? last_block_num_nonzero_subblocks_h : out_block_h / out_subblock_h},
                 {"out_last_subblock_h", last_y ? last_subblock_of_last_block_h : out_subblock_h},
                 {"padded_block_tiles_h_skip", last_y ? last_block_padded_block_tiles_h_skip : 0u},
                 {"out_num_nonzero_subblocks_w", out_block_w / out_subblock_w},
                 {"out_last_num_nonzero_subblocks_w",
                  last_x ? last_block_num_nonzero_subblocks_w : out_block_w / out_subblock_w},
                 {"out_last_subblock_w", last_x ? last_subblock_of_last_block_w : out_subblock_w},
                 {"padded_subblock_tiles_addr_skip", last_x ? last_block_padded_subblock_tiles_addr_skip : 0u},
                 {"padded_block_tiles_w_skip", last_x ? last_block_padded_block_tiles_w_skip : 0u}});
            if (bias_tensor.has_value()) {
                in1_sender_run_args.runtime_arg_values["in3_tensor_start_tile_id"][core] = per_core_N * output_idx_x;
            }
            if (!output_is_sharded) {
                in1_sender_run_args.runtime_arg_values["last_num_blocks_w_dim"][core] =
                    last_x ? last_out_num_blocks_w : out_num_blocks_x;
            }
        }
        // in0 sender and in1 receiver
        else {
            // padding args (WRITER): H-dim tail depends on output_idx_y == num_blocks_y - 1;
            // W-dim tail depends on output_idx_x == num_blocks_x - 1. The two dimensions are
            // independent.
            AddRuntimeArgsForNode(
                in1_receiver_run_args.runtime_arg_values,
                core,
                {{"in1_mcast_sender_noc_x", static_cast<uint32_t>(top_left_core_physical.x)},
                 {"in1_mcast_sender_noc_y", static_cast<uint32_t>(top_left_core_physical.y)},
                 {"out_tensor_start_tile_id", (output_idx_x * per_core_N) + (output_idx_y * per_core_M * N)},
                 {"out_num_nonzero_subblocks_h", out_block_h / out_subblock_h},
                 {"out_last_num_nonzero_subblocks_h",
                  last_y ? last_block_num_nonzero_subblocks_h : out_block_h / out_subblock_h},
                 {"out_last_subblock_h", last_y ? last_subblock_of_last_block_h : out_subblock_h},
                 {"padded_block_tiles_h_skip", last_y ? last_block_padded_block_tiles_h_skip : 0u},
                 {"out_num_nonzero_subblocks_w", out_block_w / out_subblock_w},
                 {"out_last_num_nonzero_subblocks_w",
                  last_x ? last_block_num_nonzero_subblocks_w : out_block_w / out_subblock_w},
                 {"out_last_subblock_w", last_x ? last_subblock_of_last_block_w : out_subblock_w},
                 {"padded_subblock_tiles_addr_skip", last_x ? last_block_padded_subblock_tiles_addr_skip : 0u},
                 {"padded_block_tiles_w_skip", last_x ? last_block_padded_block_tiles_w_skip : 0u}});
            if (!output_is_sharded) {
                in1_receiver_run_args.runtime_arg_values["last_num_blocks_h_dim"][core] =
                    last_y ? last_out_num_blocks_h : out_num_blocks_y;
                in1_receiver_run_args.runtime_arg_values["last_num_blocks_w_dim"][core] =
                    last_x ? last_out_num_blocks_w : out_num_blocks_x;
            }
        }

        AddRuntimeArgsForNode(
            in0_sender_run_args.runtime_arg_values,
            core,
            {{"in0_tensor_start_tile_id", static_cast<uint32_t>(in0_tensor_start_tile_id_stride * output_idx_y)},
             {"in0_mcast_dest_noc_start_x", 0u},
             {"in0_mcast_dest_noc_start_y", 0u},
             {"in0_mcast_dest_noc_end_x", 0u},
             {"in0_mcast_dest_noc_end_y", 0u},
             {"last_block_h", per_core_M}});
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Assemble
    ////////////////////////////////////////////////////////////////////////////
    ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(in0_sender_run_args));
    run_args.kernel_run_args.push_back(std::move(in1_sender_run_args));
    if (has_in1_receiver_writer_kernel) {
        run_args.kernel_run_args.push_back(std::move(in1_receiver_run_args));
    }

    run_args.tensor_args = {
        {IN0, in0_tensor},
        {IN1, in1_tensor},
        {OUTPUT, out_tensor},
    };
    if (bias_tensor.has_value()) {
        run_args.tensor_args.emplace(BIAS, *bias_tensor);
    }

    ProgramSpec spec{
        .name = "matmul_multi_core_reuse_mcast_1d_in1",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .semaphores = std::move(semaphores),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = std::move(work_units),
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace reuse_mcast_1d_optimized_helpers

MatmulMultiCoreReuseMcast1DProgramFactory::shared_variables_t matmul_multi_core_reuse_mcast_1d_optimized_(
    tt_metal::Program& program,
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
    uint32_t out_block_h,
    uint32_t out_block_w,
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
    tt::DataFormat bias_data_format = tt::DataFormat::Bfp8_b;  // bias; doesn't matter if bias=nullptr
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

    tt_metal::IDevice* device = a.device();

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
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    ////////////////////////////////////////////////////////////////////////////
    //                      Matmul Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // NOTE: Pads matmul input dims to 512 x 512 multiples (ie. multiples of 16*32 x 16*32)
    // NOTE: Maximum number of tiles in output is 120 * 16^2 = 30,720 (eg. [1, 1, 5120, 6144])
    const auto in0_B = fuse_batch ? 1 : get_batch_size(ashape);
    const auto in1_B = fuse_batch ? 1 : get_batch_size(bshape);
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

    if (gather_in0) {
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
        return reuse_mcast_1d_optimized_helpers::process_gather_in0_program_and_create_override_variables(
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
    TT_FATAL(start_cb_index == tt::CBIndex::c_0, "mcast does not support a non-zero start cb index");

    ////////////////////////////////////////////////////////////////////////////
    //                      Sub-device start core
    ////////////////////////////////////////////////////////////////////////////
    // The 1D mcast matmul only supports rectangular sub-device worker grids, because the in0/in1
    // multicast targets a single bounding-box rectangle and the per-core index math assumes a
    // contiguous row-major rectangle of width `compute_with_storage_grid_size.x`.
    CoreCoord sub_device_start_core = {0, 0};
    if (sub_device_id.has_value()) {
        auto sd_worker_cores =
            device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sub_device_id.value());
        auto bbox = sd_worker_cores.bounding_box();
        TT_FATAL(
            sd_worker_cores.num_cores() == bbox.size(),
            "matmul_multicore_reuse_mcast_1d only supports rectangular sub-device worker grids. "
            "Got sub-device worker cores: {} (bounding box: {})",
            sd_worker_cores,
            bbox);
        TT_FATAL(
            bbox.start_coord.x + compute_with_storage_grid_size.x - 1 <= bbox.end_coord.x &&
                bbox.start_coord.y + compute_with_storage_grid_size.y - 1 <= bbox.end_coord.y,
            "matmul_multicore_reuse_mcast_1d compute_with_storage_grid_size {} anchored at sub-device start {} "
            "extends past the sub-device's worker bounding box {}",
            compute_with_storage_grid_size,
            bbox.start_coord,
            bbox);
        sub_device_start_core = bbox.start_coord;
    }

    if (mcast_in0) {
        return reuse_mcast_1d_optimized_helpers::process_mcast_in0_program_and_create_override_variables(
            program,
            a,
            device,
            math_fidelity,
            fp32_dest_acc_en,
            math_approx_mode,
            packer_l1_acc,
            compute_with_storage_grid_size,
            throttle_level,
            in0_B,
            Mt,
            Nt,
            Kt,
            bcast_batch,
            transpose_a,
            transpose_b,
            in0_block_w,
            out_subblock_h,
            out_subblock_w,
            out_block_h,
            out_block_w,
            per_core_M,
            per_core_N,
            fused_activation,
            in0_tensor,
            in1_tensor,
            global_cb,
            bias_mesh_tensor,
            output,
            in0_tile,
            in1_tile,
            bias.has_value() ? bias->tensor_spec().tile() : output_tile,
            output_tile,
            in0_data_format,
            in1_data_format,
            bias_data_format,
            output_data_format,
            a.memory_config().is_sharded(),
            in1_tensor.memory_config().is_sharded(),
            bias.has_value() ? bias->memory_config().is_sharded() : false,
            output.memory_config().is_sharded(),
            untilize_out,
            fused_op_signaler,
            operations::matmul::utilities::fused_matmul_bias_row_broadcastable(bias),
            sub_device_start_core);
    }
    return reuse_mcast_1d_optimized_helpers::process_mcast_in1_program_and_create_override_variables(
        program,
        a,
        device,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode,
        packer_l1_acc,
        compute_with_storage_grid_size,
        throttle_level,
        in0_B,
        in1_B,
        Mt,
        Nt,
        Kt,
        bcast_batch,
        transpose_a,
        transpose_b,
        in0_block_w,
        out_subblock_h,
        out_subblock_w,
        out_block_h,
        out_block_w,
        per_core_M,
        per_core_N,
        fused_activation,
        in0_tensor,
        in1_tensor,
        bias_mesh_tensor,
        output,
        in0_tile,
        in1_tile,
        bias.has_value() ? bias->tensor_spec().tile() : output_tile,
        output_tile,
        in0_data_format,
        in1_data_format,
        bias_data_format,
        output_data_format,
        a.memory_config().is_sharded(),
        output.memory_config().is_sharded(),
        untilize_out,
        operations::matmul::utilities::fused_matmul_bias_row_broadcastable(bias),
        sub_device_start_core);
}

void MatmulMultiCoreReuseMcast1DProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const shared_variables_t& shared_variables,
    const ttnn::prim::MatmulParams& operation_attributes,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    reuse_mcast_1d_optimized_helpers::override_program_parameters(
        shared_variables, operation_attributes.global_cb, program, tensor_args, tensor_return_value);
}

ttnn::device_operation::ProgramArtifacts MatmulMultiCoreReuseMcast1DProgramFactory::create_program_artifacts(
    const ttnn::prim::MatmulParams& operation_attributes,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    using namespace tt;
    using namespace operations::matmul::utilities;

    const auto& a = tensor_args.input_tensors.at(0);
    const auto& b = tensor_args.input_tensors.at(1);
    const auto& bias = tensor_args.optional_input_tensors.at(0);
    const auto& output = tensor_return_value.at(0).mesh_tensor();

    TT_FATAL(operation_attributes.bcast_batch.has_value(), "Error: bcast_batch field should have been populated");
    TT_FATAL(operation_attributes.program_config.has_value(), "Error: program_config field should have been populated");
    bool bcast_batch = operation_attributes.bcast_batch.value();
    bool transpose_a = operation_attributes.transpose_a;
    bool transpose_b = operation_attributes.transpose_b;

    auto program_config = std::get<operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>(
        operation_attributes.program_config.value());

    auto fuse_batch = program_config.fuse_batch;
    auto in0_block_w = program_config.in0_block_w;
    auto out_subblock_h = program_config.out_subblock_h;
    auto out_subblock_w = program_config.out_subblock_w;
    auto out_block_h = program_config.out_block_h;
    auto out_block_w = program_config.out_block_w;
    auto per_core_M = program_config.per_core_M;
    auto per_core_N = program_config.per_core_N;
    auto mcast_in0 = program_config.mcast_in0;
    auto gather_in0 = program_config.gather_in0;

    TT_FATAL(!gather_in0, "create_program_artifacts does not support gather_in0 mode");

    TT_FATAL(
        operation_attributes.compute_kernel_config.has_value(),
        "Error: compute_kernel_config field should have been populated");
    auto compute_kernel_config = operation_attributes.compute_kernel_config.value();
    auto untilize_out = operation_attributes.untilize_out;

    const auto& a_shape_padded = get_matmul_tensor_padded_shape(a, transpose_a);
    const auto& b_shape_padded = get_matmul_tensor_padded_shape(b, transpose_b);
    const auto in0_tile = get_matmul_tile(a, transpose_a);
    const auto in1_tile = get_matmul_tile(b, transpose_b);

    // cannot use the output tensor tile directly as that might be changed by user override
    const auto output_tile = tt::tt_metal::Tile({in0_tile.get_height(), in1_tile.get_width()});

    // Buffer dataformats
    const MeshTensor& in0_tensor = a.mesh_tensor();
    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(in0_tensor.dtype());
    const MeshTensor& in1_tensor = b.mesh_tensor();
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(in1_tensor.dtype());
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());

    ttsl::optional_reference<const MeshTensor> bias_mesh_tensor;
    tt::DataFormat bias_data_format = tt::DataFormat::Bfp8_b;
    if (bias.has_value()) {
        const auto& c = bias.value();
        bias_mesh_tensor = c.mesh_tensor();
        bias_data_format = tt_metal::datatype_to_dataformat_converter(c.dtype());
    }

    tt_metal::IDevice* device = &in0_tensor.mutable_device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    const auto in0_B = fuse_batch ? 1 : get_batch_size(a_shape_padded);
    const auto in1_B = fuse_batch ? 1 : get_batch_size(b_shape_padded);
    const auto Mt = get_M_dim(a_shape_padded, in0_tile, fuse_batch);
    const auto Kt = get_K_dim(a_shape_padded, in0_tile);
    const auto Nt = get_N_dim(b_shape_padded, in1_tile);

    // The 1D mcast matmul only supports rectangular sub-device worker grids, because the in0/in1
    // multicast targets a single bounding-box rectangle and the per-core index math assumes a
    // contiguous row-major rectangle of width `grid_size.x`.
    if (!program_config.allowed_worker_cores.has_value()) {
        log_warning(
            tt::LogOp,
            "MatmulMultiCoreReuseMcast1DProgramFactory::create_program_artifacts: program_config.allowed_worker_cores "
            "not populated; auto-populating from compute_with_storage_grid_size. Callers that bypass "
            "ttnn::prim::matmul() should invoke ttnn::operations::matmul::normalize_program_config() on the "
            "program config first. This will become a hard error in a future release.");
        program_config.allowed_worker_cores = CoreRangeSet(CoreRange(
            CoreCoord(0, 0),
            CoreCoord(
                program_config.compute_with_storage_grid_size.x - 1,
                program_config.compute_with_storage_grid_size.y - 1)));
    }
    auto grid_size = program_config.allowed_worker_cores.value().bounding_box().grid_size();

    // When a sub-device is present use its bounding-box start; otherwise fall
    // back to allowed_worker_cores start so non-(0,0) placements work correctly.
    CoreCoord sub_device_start_core = program_config.allowed_worker_cores.value().bounding_box().start_coord;
    if (operation_attributes.sub_device_id.has_value()) {
        auto sd_worker_cores = device->worker_cores(
            tt::tt_metal::HalProgrammableCoreType::TENSIX, operation_attributes.sub_device_id.value());
        auto bbox = sd_worker_cores.bounding_box();
        TT_FATAL(
            sd_worker_cores.num_cores() == bbox.size(),
            "matmul_multicore_reuse_mcast_1d only supports rectangular sub-device worker grids. "
            "Got sub-device worker cores: {} (bounding box: {})",
            sd_worker_cores,
            bbox);
        TT_FATAL(
            bbox.start_coord.x + grid_size.x - 1 <= bbox.end_coord.x &&
                bbox.start_coord.y + grid_size.y - 1 <= bbox.end_coord.y,
            "matmul_multicore_reuse_mcast_1d grid_size {} anchored at sub-device start {} "
            "extends past the sub-device's worker bounding box {}",
            grid_size,
            bbox.start_coord,
            bbox);
        sub_device_start_core = bbox.start_coord;
    }

    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler> fused_op_signaler = std::nullopt;

    if (mcast_in0) {
        return reuse_mcast_1d_optimized_helpers::create_program_mcast_in0_artifacts(
            a,
            device,
            fp32_dest_acc_en,
            packer_l1_acc,
            grid_size,
            ttnn::get_throttle_level(compute_kernel_config),
            compute_kernel_config,
            in0_B,
            in1_B,
            Mt,
            Nt,
            Kt,
            bcast_batch,
            transpose_a,
            transpose_b,
            in0_block_w,
            out_subblock_h,
            out_subblock_w,
            out_block_h,
            out_block_w,
            per_core_M,
            per_core_N,
            program_config.fused_activation,
            in0_tensor,
            in1_tensor,
            bias_mesh_tensor,
            output,
            in0_tile,
            in1_tile,
            bias.has_value() ? bias->tensor_spec().tile() : output_tile,
            output_tile,
            in0_data_format,
            in1_data_format,
            bias_data_format,
            output_data_format,
            in0_tensor.memory_config().is_sharded(),
            in1_tensor.memory_config().is_sharded(),
            bias.has_value() ? bias->memory_config().is_sharded() : false,
            output.memory_config().is_sharded(),
            untilize_out,
            fused_op_signaler,
            fused_matmul_bias_row_broadcastable(bias),
            sub_device_start_core,
            operation_attributes.prefetcher_pipes);
    }
    TT_FATAL(
        operation_attributes.prefetcher_pipes.empty(),
        "matmul over prefetcher_pipes is supported only for mcast_in0=true, not the mcast_in1 variant");
    return reuse_mcast_1d_optimized_helpers::create_program_mcast_in1_artifacts(
        a,
        device,
        fp32_dest_acc_en,
        packer_l1_acc,
        grid_size,
        ttnn::get_throttle_level(compute_kernel_config),
        compute_kernel_config,
        in0_B,
        in1_B,
        Mt,
        Nt,
        Kt,
        bcast_batch,
        transpose_a,
        transpose_b,
        in0_block_w,
        out_subblock_h,
        out_subblock_w,
        out_block_h,
        out_block_w,
        per_core_M,
        per_core_N,
        program_config.fused_activation,
        in0_tensor,
        in1_tensor,
        bias_mesh_tensor,
        output,
        in0_tile,
        in1_tile,
        bias.has_value() ? bias->tensor_spec().tile() : output_tile,
        output_tile,
        in0_data_format,
        in1_data_format,
        bias_data_format,
        output_data_format,
        in0_tensor.memory_config().is_sharded(),
        output.memory_config().is_sharded(),
        untilize_out,
        fused_matmul_bias_row_broadcastable(bias),
        sub_device_start_core);
}

MatmulMultiCoreReuseMcast1DProgramFactory::shared_variables_t matmul_multi_core_reuse_mcast_1d_optimized_helper(
    tt_metal::Program& program,
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
            "matmul_multi_core_reuse_mcast_1d_optimized_helper: program_config.allowed_worker_cores not populated; "
            "auto-populating from compute_with_storage_grid_size. Callers that bypass ttnn::prim::matmul() (e.g. "
            "CCL fused ops) should invoke ttnn::operations::matmul::normalize_program_config() on the program "
            "config first. This will become a hard error in a future release.");
        config.allowed_worker_cores = CoreRangeSet(CoreRange(
            CoreCoord(0, 0),
            CoreCoord(config.compute_with_storage_grid_size.x - 1, config.compute_with_storage_grid_size.y - 1)));
    }
    auto resolved_grid = config.allowed_worker_cores.value().bounding_box().grid_size();

    return matmul_multi_core_reuse_mcast_1d_optimized_(
        program,
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

MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory::cached_mesh_workload_t
MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory::create_mesh_workload(
    const ttnn::prim::MatmulParams& attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    for (const auto& mesh_coord_range : tensor_coords.ranges()) {
        for (const auto& mesh_coord : mesh_coord_range) {
            const ttnn::MeshCoordinateRange mesh_coord_range{mesh_coord, mesh_coord};
            auto pc = std::get<operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>(
                attributes.program_config.value());
            if (!pc.allowed_worker_cores.has_value()) {
                log_warning(
                    tt::LogOp,
                    "MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory::create_mesh_workload: "
                    "program_config.allowed_worker_cores not populated; auto-populating from "
                    "compute_with_storage_grid_size. Callers that bypass ttnn::prim::matmul() should invoke "
                    "ttnn::operations::matmul::normalize_program_config() on the program config first. This will "
                    "become a hard error in a future release.");
                pc.allowed_worker_cores = CoreRangeSet(CoreRange(
                    CoreCoord(0, 0),
                    CoreCoord(pc.compute_with_storage_grid_size.x - 1, pc.compute_with_storage_grid_size.y - 1)));
            }
            auto mesh_grid = pc.allowed_worker_cores.value().bounding_box().grid_size();
            DeviceComputeKernelConfig ckc = attributes.compute_kernel_config.value();
            tt_metal::Program program{};
            std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler> empty_signaler = std::nullopt;
            auto b_tensors =
                std::vector<Tensor>{tensor_args.input_tensors.begin() + 1, tensor_args.input_tensors.end()};
            auto shared_vars = matmul_multi_core_reuse_mcast_1d_optimized_(
                program,
                tensor_args.input_tensors.at(0),
                b_tensors,
                tensor_args.optional_input_tensors.at(0),
                tensor_return_value,
                attributes.bcast_batch.value_or(false),
                attributes.transpose_a,
                attributes.transpose_b,
                mesh_grid,
                ckc,
                ttnn::get_throttle_level(ckc),
                pc.in0_block_w,
                pc.out_subblock_h,
                pc.out_subblock_w,
                pc.out_block_h,
                pc.out_block_w,
                pc.per_core_M,
                pc.per_core_N,
                pc.fuse_batch,
                pc.fused_activation,
                pc.mcast_in0,
                pc.gather_in0,
                pc.hop_cores,
                attributes.untilize_out,
                empty_signaler,
                attributes.global_cb,
                pc.num_global_cb_receivers,
                pc.stream_in1,
                attributes.sub_device_id,
                tt::CBIndex::c_0,
                std::nullopt);
            shared_variables[mesh_coord_range] = std::move(shared_vars);
            workload.add_program(mesh_coord_range, std::move(program));
        }
    }
    return {std::move(workload), std::move(shared_variables)};
}

void MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const ttnn::prim::MatmulParams& attributes,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    for (auto& [mesh_coord_range, program] : cached_workload.workload.get_programs()) {
        MatmulMultiCoreReuseMcast1DProgramFactory::override_runtime_arguments(
            program,
            cached_workload.shared_variables.at(mesh_coord_range),
            attributes,
            tensor_args,
            tensor_return_value);
    }
}

ttnn::device_operation::CachedProgram<MatmulMultiCoreReuseMcast1DProgramFactory::shared_variables_t>
matmul_multi_core_reuse_mcast_1d_optimized_helper(
    tt_metal::Program& program,
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
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    MatmulMultiCoreReuseMcast1DProgramFactory::shared_variables_t shared_vars =
        matmul_multi_core_reuse_mcast_1d_optimized_helper(
            program,
            a,
            b_tensors,
            bias,
            output_tensors,
            broadcast_batch,
            compute_kernel_config,
            program_config,
            untilize_out,
            fused_op_signaler,
            global_cb,
            sub_device_id,
            tt::CBIndex::c_0,
            std::nullopt);

    return {std::move(program), std::move(shared_vars)};
}

}  // namespace ttnn::prim
