// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ProgramDescriptor form of the 1D mcast_in0 matmul builder, for CCL-fused ops built as a ProgramDescriptor /
// WorkloadDescriptor (all_gather_matmul_async). matmul's own 1D factory is on Metal 2.0 (#56836); the fused ops keep
// building the matmul next to their CCL kernels and exchange fused-op signals through positional runtime args
// (ttnn/operations/ccl/ccl_op_fusion.hpp), which the Metal 2.0 path does not express yet.
//
// The builder is matmul's own create_program_mcast_in0_descriptor as of 917da3f9949^ (removed by #56836), which
// already pushed the fused-op runtime args but never initialised the signaler. Updated for what changed in matmul's
// Program& builder since (Quasar mcast-rectangle normalisation), pointed at the ccl_fusion kernel copies, with the
// mcast semaphore ids allocated in the caller's descriptor and the signaler initialised. mcast_in1 / gather_in0 are
// rejected: neither works fused today (#50167). Delete this file once the CCL ops are on Metal 2.0.

#include "ttnn/operations/experimental/matmul/ccl_fusion/device/ccl_fusion_mcast_1d.hpp"
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

void create_program_mcast_in0_descriptor(
    tt::tt_metal::ProgramDescriptor& desc,
    const ttnn::Tensor& a,
    const tt_metal::distributed::MeshDevice& device,
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
    bool in1_is_sharded,
    bool bias_is_sharded,
    bool output_is_sharded,
    bool untilize_out,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& fused_op_signaler,
    bool row_broadcast_bias = true,
    CoreCoord sub_device_start_core = {0, 0}) {
    using tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids;

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

    // Tiles whose size is not a multiple of the DRAM alignment (e.g. bfp8 32x16 = 544B on
    // Blackhole's 64B alignment) are padded to it in DRAM. The interleaved reader copies tiles at
    // the padded stride, so the in0/in1/bias CBs must hold pages at the aligned stride and the
    // reader/unpacker walk tiles at the same stride. No-op when already aligned (all bf16 tiles,
    // 32-wide bfp8, Wormhole). Replaces the staging-CB workaround. Sharded CBs are backed by the
    // tensor buffer and keep their natural page size.
    const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();
    uint32_t in0_aligned_tile_size =
        in0_is_sharded ? in0_single_tile_size : tt::align(in0_single_tile_size, dram_alignment);
    uint32_t in1_aligned_tile_size =
        in1_is_sharded ? in1_single_tile_size : tt::align(in1_single_tile_size, dram_alignment);
    // Bias CB pages must be padded to the DRAM alignment so the reader's L1 write stride
    // matches the DRAM page stride (e.g. 64B on Blackhole for a 32B (1,16) bf16 bias tile).
    // Mirrors in0/in1 above. Sharded bias is backed by the L1 tensor buffer and keeps its
    // natural page size. No-op on Wormhole and for tiles already >= dram_alignment.
    uint32_t bias_aligned_tile_size =
        bias_is_sharded ? bias_single_tile_size : tt::align(bias_single_tile_size, dram_alignment);
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
    if (in0_B * num_blocks > 1) {
        in0_CB_tiles *= operations::matmul::utilities::MCAST_INPUT_BUFFERING_DEPTH;
    }
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
    if (in1_B * num_blocks > 1) {
        in1_CB_tiles *= operations::matmul::utilities::MCAST_INPUT_BUFFERING_DEPTH;
    }
    if (in1_is_sharded) {
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
                device.worker_core_from_logical_core({start_core_x + core_idx_x, start_core_y}).x);
        }
        for (uint32_t core_idx_y = 0; core_idx_y < in0_mcast_sender_cores_grid.y; ++core_idx_y) {
            in0_mcast_noc_y.push_back(
                device.worker_core_from_logical_core({start_core_x, start_core_y + core_idx_y}).y);
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

    // Mcast args — semaphore IDs assigned sequentially (0, 1)
    // The fused CCL op may already have placed semaphores in desc; take ids that are free on these cores.
    const uint32_t in0_mcast_sender_semaphore_id =
        ttnn::experimental::ccl::add_semaphore_descriptor(desc, CoreRangeSet(all_cores), INVALID);
    const uint32_t in0_mcast_receiver_semaphore_id =
        ttnn::experimental::ccl::add_semaphore_descriptor(desc, CoreRangeSet(all_cores), INVALID);

    CoreCoord top_left_core = in0_mcast_receiver_cores_bounding_box.start_coord;
    CoreCoord bottom_right_core = in0_mcast_receiver_cores_bounding_box.end_coord;
    auto top_left_core_physical = device.worker_core_from_logical_core(top_left_core);
    auto bottom_right_core_physical = device.worker_core_from_logical_core(bottom_right_core);

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
            (std::uint32_t)in0_B  // batch
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
            (std::uint32_t)in0_B,  // batch
            (std::uint32_t)in1_B,  // batch
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
        (std::uint32_t)in0_B,        // batch
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
        (std::uint32_t)in0_B,  // batch
        (std::uint32_t)false   // get_batch_from_reader
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
        device.arch(), num_cores, mm_kernel_defines);
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device.arch(), num_cores, mm_kernel_defines, throttle_level);

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

    // Intermediate CB read

    // Helper to convert std::map defines to KernelDescriptor::Defines (vector of pairs)
    auto map_to_defines = [](const std::map<std::string, std::string>& m) -> KernelDescriptor::Defines {
        KernelDescriptor::Defines result;
        result.reserve(m.size());
        for (const auto& [k, v] : m) {
            result.emplace_back(k, v);
        }
        return result;
    };

    // in1 is the reader of weights/output writer, and we choose to make it use the optimized reader noc
    tt_metal::NOC in0_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device.arch());
    tt_metal::NOC in1_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device.arch());

    if (fuse_op && fused_op_signaler->is_all_gather()) {
        // Create semaphores
        fused_op_signaler->init_fused_op(
            desc,
            &device,
            in0_mcast_sender_cores,
            in0_is_sharded ? ttnn::experimental::ccl::FusedOpSignalerMode::SINGLE
                           : ttnn::experimental::ccl::FusedOpSignalerMode::MULTI);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Build Kernel Descriptors
    ////////////////////////////////////////////////////////////////////////////
    KernelDescriptor in0_sender_kernel_desc;
    KernelDescriptor in0_no_work_in_receiver_kernel_desc;
    bool has_in0_no_work_in_receiver_kernel = false;
    KernelDescriptor in0_no_work_not_in_receiver_kernel_desc;
    bool has_in0_no_work_not_in_receiver_kernel = false;
    KernelDescriptor in0_receiver_kernel_desc;
    bool has_in0_receiver_kernel = false;
    KernelDescriptor in1_sender_writer_kernel_desc;
    KernelDescriptor compute_kernel_desc;

    in0_sender_kernel_desc.kernel_source =
        in0_is_sharded ? "ttnn/cpp/ttnn/operations/experimental/matmul/ccl_fusion/device/kernels/dataflow/"
                         "reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp"
                       : "ttnn/cpp/ttnn/operations/experimental/matmul/ccl_fusion/device/kernels/dataflow/"
                         "reader_bmm_tile_layout_in0_sender_padding.cpp";
    in0_sender_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    in0_sender_kernel_desc.core_ranges = in0_mcast_cores_with_work_and_in_receiver_grid;
    in0_sender_kernel_desc.compile_time_args = in0_sender_compile_time_args;
    in0_sender_kernel_desc.defines = map_to_defines(mm_kernel_in0_sender_writer_defines);
    if (in0_is_sharded) {
        in0_sender_kernel_desc.named_compile_time_args = {
            {"cb_in0", tt::CBIndex::c_0},
            {"cb_in0_sharded", tt::CBIndex::c_2},
            {"cb_l1_array", tt::CBIndex::c_6},
            {"cb_sparsity", tt::CBIndex::c_6},
            // Indexed/gather mode: sparse_matmul only (0 = disabled). Passed on both branches so the
            // arg does not depend on which reader was selected above; the block-sharded one ignores it.
            {"num_active", 0},
        };
    } else {
        in0_sender_kernel_desc.named_compile_time_args = {
            {"cb_in0", tt::CBIndex::c_0},
            {"cb_in0_sharded", tt::CBIndex::c_2},
            {"cb_sparsity", tt::CBIndex::c_6},
            {"num_active", 0},  // indexed/gather mode: sparse_matmul only (0 = disabled)
        };
    }
    in0_sender_kernel_desc.config =
        DataMovementConfigDescriptor{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = in0_noc};

    if (in0_is_sharded) {
        if (in0_mcast_cores_without_work_and_in_receiver_grid.num_cores() > 0) {
            has_in0_no_work_in_receiver_kernel = true;
            auto no_work_ct_args = in0_sender_compile_time_args;
            no_work_ct_args[0] = 0;  // core_has_output_block_work
            no_work_ct_args[1] = 1;  // core_in_in0_receiver_mcast_grid
            in0_no_work_in_receiver_kernel_desc.kernel_source =
                "ttnn/cpp/ttnn/operations/experimental/matmul/ccl_fusion/device/kernels/dataflow/"
                "reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp";
            in0_no_work_in_receiver_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
            in0_no_work_in_receiver_kernel_desc.core_ranges = in0_mcast_cores_without_work_and_in_receiver_grid;
            in0_no_work_in_receiver_kernel_desc.compile_time_args = no_work_ct_args;
            in0_no_work_in_receiver_kernel_desc.defines = map_to_defines(mm_kernel_in0_sender_writer_defines);
            in0_no_work_in_receiver_kernel_desc.named_compile_time_args = {
                {"cb_in0", tt::CBIndex::c_0},
                {"cb_in0_sharded", tt::CBIndex::c_2},
                {"cb_l1_array", tt::CBIndex::c_6},
            };
            in0_no_work_in_receiver_kernel_desc.config =
                DataMovementConfigDescriptor{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = in0_noc};
        }
        if (in0_mcast_cores_without_work_and_not_in_receiver_grid.num_cores() > 0) {
            has_in0_no_work_not_in_receiver_kernel = true;
            auto no_work_ct_args = in0_sender_compile_time_args;
            no_work_ct_args[0] = 0;  // core_has_output_block_work
            no_work_ct_args[1] = 0;  // core_in_in0_receiver_mcast_grid
            in0_no_work_not_in_receiver_kernel_desc.kernel_source =
                "ttnn/cpp/ttnn/operations/experimental/matmul/ccl_fusion/device/kernels/dataflow/"
                "reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp";
            in0_no_work_not_in_receiver_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
            in0_no_work_not_in_receiver_kernel_desc.core_ranges = in0_mcast_cores_without_work_and_not_in_receiver_grid;
            in0_no_work_not_in_receiver_kernel_desc.compile_time_args = no_work_ct_args;
            in0_no_work_not_in_receiver_kernel_desc.defines = map_to_defines(mm_kernel_in0_sender_writer_defines);
            in0_no_work_not_in_receiver_kernel_desc.named_compile_time_args = {
                {"cb_in0", tt::CBIndex::c_0},
                {"cb_in0_sharded", tt::CBIndex::c_2},
                {"cb_l1_array", tt::CBIndex::c_6},
            };
            in0_no_work_not_in_receiver_kernel_desc.config =
                DataMovementConfigDescriptor{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = in0_noc};
        }
    }

    if (!in0_is_sharded && in0_mcast_receivers.num_cores() > 0) {
        has_in0_receiver_kernel = true;
        in0_receiver_kernel_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/matmul/ccl_fusion/device/kernels/dataflow/"
            "reader_bmm_tile_layout_in0_receiver.cpp";
        in0_receiver_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        in0_receiver_kernel_desc.core_ranges = in0_mcast_receivers;
        in0_receiver_kernel_desc.compile_time_args = in0_receiver_compile_time_args;
        in0_receiver_kernel_desc.named_compile_time_args = {
            {"cb_in0", tt::CBIndex::c_0},
        };
        in0_receiver_kernel_desc.config =
            DataMovementConfigDescriptor{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = in0_noc};
    }

    in1_sender_writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/matmul/ccl_fusion/device/kernels/dataflow/"
        "reader_bmm_tile_layout_in1_sender_writer_padding.cpp";
    in1_sender_writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    in1_sender_writer_kernel_desc.core_ranges = all_cores_with_work;
    in1_sender_writer_kernel_desc.compile_time_args = in1_sender_writer_compile_time_args;
    in1_sender_writer_kernel_desc.defines = map_to_defines(mm_kernel_in1_sender_writer_defines);
    in1_sender_writer_kernel_desc.named_compile_time_args = {
        {"cb_in1", tt::CBIndex::c_1},
        {"cb_bias", tt::CBIndex::c_3},
        {"cb_out", tt::CBIndex::c_4},
        {"cb_sparsity", tt::CBIndex::c_7},
        {"num_active", 0},  // indexed/gather mode: sparse_matmul only (0 = disabled)
    };
    in1_sender_writer_kernel_desc.config =
        DataMovementConfigDescriptor{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = in1_noc};

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
        in0_B,                   // batch
        out_block_tiles,         // out_block_num_tiles

        untilize_out,  // untilize_out
        false,         // get_batch_from_reader
        in0_transpose_tile,
    };
    if (bias_tensor.has_value()) {
        compute_kernel_args.push_back(row_broadcast_bias ? 1u : 0u);
    }

    compute_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/matmul/ccl_fusion/device/kernels/compute/"
        "bmm_large_block_zm_fused_bias_activation.cpp";
    compute_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel_desc.core_ranges = all_cores_with_work;
    compute_kernel_desc.compile_time_args = compute_kernel_args;
    constexpr auto cb_intermed0 = tt::CBIndex::c_5;
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
    compute_kernel_desc.defines = map_to_defines(mm_kernel_defines);
    {
        KernelDescriptor::NamedCompileTimeArgs named_compile_args = {
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
            const auto params = get_activation_params(fused_activation.value());
            named_compile_args.push_back({"activation_type", static_cast<uint32_t>(params.type)});
            named_compile_args.push_back({"activation_param0", params.param0});
            named_compile_args.push_back({"activation_param1", params.param1});
            named_compile_args.push_back({"activation_param2", params.param2});
        }
        compute_kernel_desc.named_compile_time_args = std::move(named_compile_args);
    }
    // When accumulating in fp32 with the K reduction split across blocks, the intermediate
    // partials CB holds Float32 and is reloaded into DEST between blocks by copy_block_matmul_partials.
    // Unless the reload's CB view is marked UnpackToDestFp32, that reload is routed through SrcA and
    // rounded to TF32 (10 mantissa bits), so the fp32 partial loses precision on every block boundary
    // and accuracy degrades as the number of K-blocks grows. The flag is set on the alias when bias
    // forces a separate SrcA view of cb_intermed0 (see above), else on cb_intermed0 itself.
    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    if (fp32_dest_acc_en && interm0_data_format == tt::DataFormat::Float32) {
        const uint32_t cb_to_mark =
            bias_reload_alias ? static_cast<uint32_t>(cb_intermed0_alias) : static_cast<uint32_t>(cb_intermed0);
        unpack_to_dest_mode[cb_to_mark] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }
    compute_kernel_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
        .math_approx_mode = math_approx_mode};

    ////////////////////////////////////////////////////////////////////////////
    //                      Build CBDescriptors
    ////////////////////////////////////////////////////////////////////////////

    tt::tt_metal::TileDescriptor in0_tile_desc{in0_tile};
    tt::tt_metal::TileDescriptor in1_tile_desc{in1_tile};
    tt::tt_metal::TileDescriptor bias_tile_desc{bias_tile};
    tt::tt_metal::TileDescriptor output_tile_desc{output_tile};

    // CB 0: in0
    {
        CBDescriptor cb_desc;
        cb_desc.total_size = in0_CB_size;
        cb_desc.core_ranges = all_cores;
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_0,
            .data_format = in0_data_format,
            .page_size = in0_aligned_tile_size,
            .tile = in0_tile_desc});
        desc.cbs.push_back(std::move(cb_desc));
    }

    // CB 1: in1
    {
        CBDescriptor cb_desc;
        cb_desc.total_size = in1_CB_size;
        cb_desc.core_ranges = all_cores;
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_1,
            .data_format = in1_data_format,
            .page_size = in1_aligned_tile_size,
            .tile = in1_tile_desc});
        if (in1_is_sharded) {
            cb_desc.tensor = &in1_tensor;
        }
        desc.cbs.push_back(std::move(cb_desc));
    }

    // CB 2: in0 sharded
    if (in0_is_sharded) {
        {
            CBDescriptor cb_desc;
            cb_desc.total_size = in2_CB_size;
            cb_desc.core_ranges = all_cores;
            cb_desc.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = tt::CBIndex::c_2,
                .data_format = in0_data_format,
                .page_size = in0_single_tile_size,
                .tile = in0_tile_desc});
            cb_desc.tensor = &in0_tensor;
            desc.cbs.push_back(std::move(cb_desc));
        }

        // Local L1 to store temp vars
        {
            CBDescriptor cb_desc;
            cb_desc.total_size = 32 * 2;
            cb_desc.core_ranges = all_cores;
            cb_desc.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = tt::CBIndex::c_6, .data_format = tt::DataFormat::Float16_b, .page_size = 32 * 2});
            desc.cbs.push_back(std::move(cb_desc));
        }
    }

    // CB 4 and CB 5: output and intermediate
    if (do_not_inplace_interm0_out_CB || (interm0_data_format != output_data_format) ||
        (untilize_out && (in1_num_subblocks > 1))) {
        // Separate output and intermediate CBs
        // output
        {
            CBDescriptor cb_desc;
            cb_desc.total_size = out_CB_size;
            cb_desc.core_ranges = all_cores;
            cb_desc.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = tt::CBIndex::c_4,
                .data_format = output_data_format,
                .page_size = output_single_tile_size,
                .tile = output_tile_desc});
            if (output_is_sharded) {
                cb_desc.tensor = &out_tensor;
            }
            desc.cbs.push_back(std::move(cb_desc));
        }
        // interm0
        {
            CBDescriptor cb_desc;
            cb_desc.total_size = interm0_CB_size;
            cb_desc.core_ranges = CoreRangeSet({all_cores});
            cb_desc.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = tt::CBIndex::c_5,
                .data_format = interm0_data_format,
                .page_size = interm0_single_tile_size,
                .tile = output_tile_desc});
            // Alias over the same SRAM, marked UnpackToDestFp32, for the bias reload (see above).
            if (bias_reload_alias) {
                cb_desc.format_descriptors.push_back(CBFormatDescriptor{
                    .buffer_index = cb_intermed0_alias,
                    .data_format = interm0_data_format,
                    .page_size = interm0_single_tile_size,
                    .tile = output_tile_desc});
            }
            desc.cbs.push_back(std::move(cb_desc));
        }
    } else {
        // share buffer
        CBDescriptor cb_desc;
        cb_desc.total_size = out_CB_size;
        cb_desc.core_ranges = all_cores;
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_4,
            .data_format = output_data_format,
            .page_size = output_single_tile_size,
            .tile = output_tile_desc});
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_5,
            .data_format = interm0_data_format,
            .page_size = interm0_single_tile_size,
            .tile = output_tile_desc});
        // Alias over the same SRAM, marked UnpackToDestFp32, for the bias reload (see above).
        if (bias_reload_alias) {
            cb_desc.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = cb_intermed0_alias,
                .data_format = interm0_data_format,
                .page_size = interm0_single_tile_size,
                .tile = output_tile_desc});
        }
        if (output_is_sharded) {
            cb_desc.tensor = &out_tensor;
        }
        desc.cbs.push_back(std::move(cb_desc));
    }

    // CB for bias
    if (bias_tensor.has_value()) {
        CBDescriptor cb_desc;
        cb_desc.total_size = in3_CB_size;
        cb_desc.core_ranges = all_cores;
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_3,
            .data_format = bias_data_format,
            .page_size = bias_aligned_tile_size,
            .tile = bias_tile_desc});
        if (bias_is_sharded) {
            cb_desc.tensor = &*bias_tensor;
        }
        desc.cbs.push_back(std::move(cb_desc));
    }

    // Intermediate CB read

    // Transpose CB for input0
    if (in0_transpose_tile) {
        CBDescriptor cb_desc;
        cb_desc.total_size = in0_CB_size;
        cb_desc.core_ranges = all_cores;
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_10,
            .data_format = in0_data_format,
            .page_size = in0_aligned_tile_size,
            .tile = in0_tile_desc});
        desc.cbs.push_back(std::move(cb_desc));
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Runtime Args (per-core loop)
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
    // Quasar is single-NOC / non-torus: the mcast rectangle MUST stay ascending [min..max]. in0_noc =
    // preferred_noc_for_dram_write(arch), which is NOC_1 on Quasar, so the WH/BH NOC_1 swap above reverses it
    // to [max..min] -> NoC "multicast invalid range" and the in0 sender hangs (waypoint NMWW). Re-normalize to
    // ascending on Quasar.
    if (device.arch() == tt::ARCH::QUASAR) {
        const CoreCoord lo{std::min(start_core_noc.x, end_core_noc.x), std::min(start_core_noc.y, end_core_noc.y)};
        const CoreCoord hi{std::max(start_core_noc.x, end_core_noc.x), std::max(start_core_noc.y, end_core_noc.y)};
        start_core_noc = lo;
        end_core_noc = hi;
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
                in0_sender_kernel_desc.runtime_args.emplace_back(core, mm_in0_sender_args);
            } else if (i < in0_mcast_receiver_num_dests) {
                in0_no_work_in_receiver_kernel_desc.runtime_args.emplace_back(core, mm_in0_sender_args);
            } else {
                in0_no_work_not_in_receiver_kernel_desc.runtime_args.emplace_back(core, mm_in0_sender_args);
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

            {
                std::vector<std::variant<uint32_t, std::reference_wrapper<const MeshTensor>>> in0_sender_variant(
                    mm_in0_sender_args.begin(), mm_in0_sender_args.end());
                in0_sender_variant[0] = in0_tensor;
                in0_sender_kernel_desc.emplace_runtime_args(core, in0_sender_variant);
            }
        }
        // in0 receiver and in 1 sender
        else {
            std::vector<uint32_t> mm_in0_receiver_args = {
                // in0 mcast args
                (std::uint32_t)top_left_core_physical.x,  // in0_mcast_sender_noc_x
                (std::uint32_t)top_left_core_physical.y   // in0_mcast_sender_noc_y
            };
            in0_receiver_kernel_desc.runtime_args.emplace_back(core, mm_in0_receiver_args);
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

            // Bias base address placeholder; rebound to a tensor binding via in1_sender_variant[18] below.
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

            {
                std::vector<std::variant<uint32_t, std::reference_wrapper<const MeshTensor>>> in1_sender_variant(
                    mm_in1_sender_writer_args.begin(), mm_in1_sender_writer_args.end());
                in1_sender_variant[0] = in1_tensor;
                in1_sender_variant[7] = out_tensor;
                if (bias_tensor.has_value()) {
                    in1_sender_variant[18] = *bias_tensor;
                }
                in1_sender_writer_kernel_desc.emplace_runtime_args(core, in1_sender_variant);
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Push kernels to descriptor
    ////////////////////////////////////////////////////////////////////////////
    desc.kernels.push_back(std::move(in0_sender_kernel_desc));
    if (has_in0_no_work_in_receiver_kernel) {
        desc.kernels.push_back(std::move(in0_no_work_in_receiver_kernel_desc));
    }
    if (has_in0_no_work_not_in_receiver_kernel) {
        desc.kernels.push_back(std::move(in0_no_work_not_in_receiver_kernel_desc));
    }
    if (has_in0_receiver_kernel) {
        desc.kernels.push_back(std::move(in0_receiver_kernel_desc));
    }
    desc.kernels.push_back(std::move(in1_sender_writer_kernel_desc));
    desc.kernels.push_back(std::move(compute_kernel_desc));
}

}  // namespace reuse_mcast_1d_optimized_helpers

static void matmul_multi_core_reuse_mcast_1d_optimized_descriptor_(
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
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& fused_op_signaler) {
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

    TT_FATAL(
        !gather_in0,
        "CCL-fused 1D matmul: gather_in0 is not supported here (only the llama ring ops fuse with gather_in0)");
    TT_FATAL(
        mcast_in0,
        "CCL-fused 1D matmul: only mcast_in0 is supported; the mcast_in1 path never threaded the fused-op "
        "signaler through and crashed the all-gather side (#50167)");

    ////////////////////////////////////////////////////////////////////////////
    //                      Sub-device start core
    ////////////////////////////////////////////////////////////////////////////
    // The 1D mcast matmul only supports rectangular sub-device worker grids, because the in0/in1
    // multicast targets a single bounding-box rectangle and the per-core index math assumes a
    // contiguous row-major rectangle of width `compute_with_storage_grid_size.x`.
    CoreCoord sub_device_start_core = {0, 0};
    const std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt;
    if (sub_device_id.has_value()) {
        auto sd_worker_cores =
            device.worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sub_device_id.value());
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

    {
        reuse_mcast_1d_optimized_helpers::create_program_mcast_in0_descriptor(
            desc,
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
            in1_tensor.memory_config().is_sharded(),
            bias.has_value() ? bias->memory_config().is_sharded() : false,
            output.memory_config().is_sharded(),
            untilize_out,
            fused_op_signaler,
            operations::matmul::utilities::fused_matmul_bias_row_broadcastable(bias),
            sub_device_start_core);
    }
}

void matmul_multi_core_reuse_mcast_1d_optimized_helper(
    tt::tt_metal::ProgramDescriptor& desc,
    const Tensor& a,
    const std::vector<Tensor>& b_tensors,
    const std::optional<const Tensor>& bias,
    const std::vector<Tensor>& output_tensors,
    bool broadcast_batch,
    DeviceComputeKernelConfig compute_kernel_config,
    const operations::matmul::MatmulProgramConfig& program_config,
    bool untilize_out,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& fused_op_signaler) {
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

    matmul_multi_core_reuse_mcast_1d_optimized_descriptor_(
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
        fused_op_signaler);
}

}  // namespace ttnn::prim::ccl_fusion
