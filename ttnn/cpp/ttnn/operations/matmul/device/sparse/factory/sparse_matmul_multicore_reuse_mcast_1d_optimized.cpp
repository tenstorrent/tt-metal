// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"

#include <tt-metalium/program_descriptors.hpp>
#include "tt-metalium/work_split.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor SparseMatmulMultiCoreReuseMcast1DProgramFactory::create_descriptor(
    const ttnn::prim::SparseMatmulParams& operation_attributes,
    const ttnn::prim::SparseMatmulInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    using tt::tt_metal::CBDescriptor;
    using tt::tt_metal::CBFormatDescriptor;
    using tt::tt_metal::ComputeConfigDescriptor;
    using tt::tt_metal::DataMovementConfigDescriptor;
    using tt::tt_metal::KernelDescriptor;
    using tt::tt_metal::ProgramDescriptor;
    using tt::tt_metal::SemaphoreDescriptor;
    using tt::tt_metal::TileDescriptor;
    using namespace tt;
    using namespace operations::matmul::utilities;

    // from create_mesh-workload
    auto matmul_attributes = ttnn::prim::MatmulParams{
        operation_attributes.program_config,
        /*bcast_batch=*/std::nullopt,
        operation_attributes.output_mem_config,
        operation_attributes.output_dtype,
        operation_attributes.compute_kernel_config,
        /*untilize_out=*/false,
        operation_attributes.user_core_coord,
        /*user_fused_activation=*/std::nullopt,
        /*user_run_batched=*/false,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        operation_attributes.output_tile,
        operation_attributes.global_cb,
        operation_attributes.sub_device_id};

    auto chosen_program_config = operations::matmul::get_program_config(
        tensor_args.input_tensors.at(0),
        tensor_args.input_tensors.at(1),
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        /*bias_single_tile_size=*/0,
        matmul_attributes);
    operations::matmul::normalize_program_config(
        chosen_program_config, tensor_args.input_tensors.at(0).device()->compute_with_storage_grid_size());

    const auto& a = tensor_args.input_tensors.at(0);
    const auto& b = tensor_args.input_tensors.at(1);
    const auto& sparsity = tensor_args.input_tensors.at(2);
    const auto& output_tensor = tensor_return_value.at(0);
    auto program_config =
        std::get<operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>(chosen_program_config);
    auto compute_with_storage_grid_size = program_config.allowed_worker_cores.value().bounding_box().grid_size();
    auto in0_block_w = program_config.in0_block_w;
    auto out_subblock_h = program_config.out_subblock_h;
    auto out_subblock_w = program_config.out_subblock_w;
    auto out_block_h = program_config.out_block_h;
    auto out_block_w = program_config.out_block_w;
    auto per_core_M = program_config.per_core_M;
    auto per_core_N = program_config.per_core_N;
    auto mcast_in0 = program_config.mcast_in0;

    auto nnz = operation_attributes.nnz;
    auto is_input_a_sparse = operation_attributes.is_input_a_sparse;

    // Indexed/gather mode: an optional `indices` operand (optional_input_tensors[0]) holds the
    // compacted list of active sparse-group ids. When present, the reader/sender kernels iterate only
    // the num_active selected groups (bB = indices[i]) instead of scanning all batchB sparsity slots,
    // and the output group axis is compact (length num_active). This maps onto the existing
    // get_batch_from_reader=false semantics: every iterated batch is processed (none skipped), so
    // compute and the in0 receiver simply loop num_batch_compute = num_active.
    //
    // The mode is signalled to the kernels purely by the "num_active" named compile-time arg below
    // (0 = off), so no preprocessor defines and no compile-time arg layout changes are needed in the
    // shared reader kernels. Both readers are shared with the dense matmul factories, which pass
    // {"num_active", 0} for the same reason they already pass an unused "cb_sparsity".
    const bool use_indices = operation_attributes.use_indices && !tensor_args.optional_input_tensors.empty() &&
                             tensor_args.optional_input_tensors.at(0).has_value();
    uint32_t num_active = 0;
    if (use_indices) {
        num_active = tensor_args.optional_input_tensors.at(0)->logical_volume();
    }
    // In indexed mode the readers never broadcast per-slot validity (every iterated batch is valid),
    // so get_batch_from_reader is forced false regardless of nnz.
    const bool get_batch_from_reader = use_indices ? false : !nnz.has_value();

    const auto& ashape = get_matmul_tensor_padded_shape(a, /*transpose=*/false);
    const auto& bshape = get_matmul_tensor_padded_shape(b, /*transpose=*/false);
    const auto in0_tile = get_matmul_tile(a, /*transpose=*/false);
    const auto in1_tile = get_matmul_tile(b, /*transpose=*/false);
    // cannot use the output tensor tile directly as that might be changed by user override
    const auto output_tile = tt::tt_metal::Tile({in0_tile.get_height(), in1_tile.get_width()});

    // CB dataformats
    const auto in0_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    const auto in1_data_format = tt_metal::datatype_to_dataformat_converter(b.dtype());
    const auto output_data_format = tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());

    auto* const device = a.device();

    const auto in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    const auto in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    // Tiles whose size is not a multiple of the DRAM alignment (e.g. bfp8 32x16 = 544B on Blackhole's
    // 64B alignment) are padded to it in DRAM. in0/in1 are interleaved here, so the reader copies
    // tiles at the padded stride; the CBs must hold pages at the aligned stride and the unpacker
    // walks tiles at the same stride. No-op when already aligned. Replaces the staging-CB workaround.
    const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();
    const uint32_t in0_aligned_tile_size = tt::align(in0_single_tile_size, dram_alignment);
    const uint32_t in1_aligned_tile_size = tt::align(in1_single_tile_size, dram_alignment);
    const auto output_single_tile_size = output_tile.get_tile_size(output_data_format);

    auto* const in0_buffer = a.buffer();
    auto* const in1_buffer = b.buffer();
    auto* const sparsity_buffer = sparsity.buffer();
    auto* const out_buffer = output_tensor.buffer();
    // The in1 sender/writer's "sparsity" slot (accessor args, page size, sparsity_addr runtime arg and
    // the c_7 buffer) carries the active-group id list in indexed/gather mode -- that kernel never
    // reads the sparsity mask there, so reusing the slot avoids adding an operand to a kernel shared
    // with the dense matmul factories. The in0 sender ignores its own slot entirely in this mode, so
    // it keeps pointing at the real sparsity tensor.
    const Tensor& in1_sparsity_tensor = use_indices ? tensor_args.optional_input_tensors.at(0).value() : sparsity;
    auto* const in1_sparsity_buffer = in1_sparsity_tensor.buffer();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config.value());

    ////////////////////////////////////////////////////////////////////////////
    //                      Matmul Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto batchB = get_batch_size(bshape);

    // When input A and input B are sparse, the batch dims are same.
    // We pick batchB and set batchA to 1.
    // When input A is sparse but B is not, both in0 and in1 need to loop over the "additional"
    // batch dims in A that are not in B. So we divide by batchB and set that.
    // In the default case (only input B is sparse), we set batchA to the batch dims of A.
    uint32_t batchA;
    if (operation_attributes.is_input_a_sparse && operation_attributes.is_input_b_sparse) {
        batchA = 1;
    } else if (operation_attributes.is_input_a_sparse) {
        batchA = get_batch_size(ashape) / batchB;
    } else {
        batchA = get_batch_size(ashape);
    }

    const auto Mt = get_M_dim(ashape, in0_tile, /*fuse_batch=*/false);
    const auto Kt = get_K_dim(ashape, in0_tile);
    const auto Nt = get_N_dim(bshape, in1_tile);

    TT_FATAL(Kt % in0_block_w == 0, "Kt ({}) must be divisible by in0_block_w ({})", Kt, in0_block_w);

    // This should allocate a DRAM buffer on the device
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_available = num_cores_x * num_cores_y;

    // Calculate number of blocks along x and y; tensor dims are padded up to 512
    uint32_t num_blocks_y = ((Mt - 1) / per_core_M) + 1;
    uint32_t num_blocks_x = ((Nt - 1) / per_core_N) + 1;
    uint32_t num_blocks_total = num_blocks_y * num_blocks_x;

    TT_FATAL(
        num_blocks_total <= num_cores_available,
        "Number of blocks exceeds number of cores available: {} blocks > {} cores",
        num_blocks_total,
        num_cores_available);

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    // Only support mcast_in0 for now
    TT_FATAL(mcast_in0, "Only mcast_in0 is supported for sparse matmul");

    using tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids;

    uint32_t num_blocks = Kt / in0_block_w;
    // Only enable packer l1 accumulation when there are spills, otherwise
    // unnecessary overhead for reconfigs are added
    bool packer_l1_acc_en = packer_l1_acc && num_blocks > 1;

    // if fp32 enabled then we pack fp32 in l1, if not, then we pack fp16 in l1
    const auto interm0_data_format = packer_l1_acc_en
                                         ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                                         : (fp32_dest_acc_en ? tt::DataFormat::Float32 : output_data_format);
    // interm0 CB page size follows interm0_data_format, not the output dtype.
    const auto interm0_single_tile_size = output_tile.get_tile_size(interm0_data_format);

    uint32_t in0_block_h = out_block_h;
    uint32_t in1_block_w = out_block_w;
    uint32_t in0_num_blocks_y = per_core_M / out_block_h;
    uint32_t in1_num_blocks_x = per_core_N / out_block_w;
    uint32_t out_num_blocks_x = in1_num_blocks_x;
    uint32_t out_num_blocks_y = in0_num_blocks_y;

    uint32_t in0_block_tiles = in0_block_h * in0_block_w;
    uint32_t in0_CB_tiles = in0_block_tiles;
    if (batchA * batchB * num_blocks > 1) {
        in0_CB_tiles *= ttnn::operations::matmul::utilities::MCAST_INPUT_BUFFERING_DEPTH;
    }
    uint32_t in0_CB_size = in0_CB_tiles * in0_aligned_tile_size;

    uint32_t in1_block_tiles = out_block_w * in0_block_w;
    uint32_t in1_CB_tiles = in1_block_tiles;
    if (batchA * batchB * num_blocks > 1) {
        in1_CB_tiles *= ttnn::operations::matmul::utilities::MCAST_INPUT_BUFFERING_DEPTH;
    }

    uint32_t in1_CB_size = in1_CB_tiles * in1_aligned_tile_size;

    uint32_t out_block_tiles = out_block_h * out_block_w;
    uint32_t out_CB_tiles = out_block_tiles;  // No double buffer

    uint32_t out_CB_size = out_CB_tiles * output_single_tile_size;
    uint32_t interm0_CB_tiles = out_block_tiles;  // No double buffer
    uint32_t interm0_CB_size = interm0_CB_tiles * interm0_single_tile_size;

    CoreCoord start_core = {0, 0};

    // The matmul region is the rectangle of size `compute_with_storage_grid_size`
    // anchored at `start_core`. The sparse 1D matmul path does not yet anchor at a sub-device
    // start, but keeping the rectangle expression here keeps the API uniform with the dense 1D
    // path and is safe (matmul_core_rect == full compute grid when start_core == (0, 0)).
    CoreRangeSet matmul_core_rect(CoreRange(
        start_core,
        CoreCoord(
            start_core.x + compute_with_storage_grid_size.x - 1, start_core.y + compute_with_storage_grid_size.y - 1)));

    uint32_t num_cores_with_work = num_blocks_total;

    uint32_t in0_sender_num_cores = 1;
    uint32_t num_cores = num_cores_with_work;

    constexpr bool row_major = true;
    CoreRangeSet all_cores =
        num_cores_to_corerangeset_in_subcoregrids(start_core, num_cores, matmul_core_rect, row_major);

    CoreRangeSet in0_mcast_sender_cores =
        num_cores_to_corerangeset_in_subcoregrids(start_core, in0_sender_num_cores, matmul_core_rect, row_major);

    CoreRangeSet all_cores_with_work =
        num_cores_to_corerangeset_in_subcoregrids(start_core, num_cores_with_work, matmul_core_rect, row_major);
    CoreRange in0_mcast_receiver_cores_bounding_box = all_cores_with_work.bounding_box();
    uint32_t in0_mcast_receiver_num_cores = in0_mcast_receiver_cores_bounding_box.size();  // always mcast to full grid

    // There should not be any cores without work in the receiver grid. If a grid is
    // not rectangular, then there will be some cores without work in the receiver grid.
    // For example, if there are 12 blocks of work, it should be put into a 3x4 grid.
    // If its laid out in row major with 8 cores in first row and 4 cores in second row,
    // then there will be 4 cores without work in the receiver grid, causing a hang.
    // We check for this below and error out.
    TT_FATAL(
        num_cores_with_work == in0_mcast_receiver_num_cores,
        "num_cores_with_work ({}) must be equal to in0_mcast_receiver_num_cores ({}), please adjust the core grid to "
        "make it rectangular.",
        num_cores_with_work,
        in0_mcast_receiver_num_cores);

    CoreRangeSet in0_mcast_cores_with_work_and_in_receiver_grid;
    CoreRangeSet in0_mcast_cores_without_work_and_in_receiver_grid;
    CoreRangeSet in0_mcast_cores_without_work_and_not_in_receiver_grid;
    CoreRangeSet in0_mcast_receivers;
    std::vector<uint32_t> in0_mcast_noc_x;
    std::vector<uint32_t> in0_mcast_noc_y;

    in0_mcast_cores_with_work_and_in_receiver_grid = CoreRangeSet({CoreRange(start_core, start_core)});
    if (in0_mcast_receiver_num_cores > 1) {
        // Check against the actual rectangle width (instead of bare grid_size.x-1) so that
        // sub-devices anchored away from (0, 0) would wrap correctly.
        auto receiver_start_core = compute_with_storage_grid_size.x > 1 ? CoreCoord{start_core.x + 1, start_core.y}
                                                                        : CoreCoord{start_core.x, start_core.y + 1};
        in0_mcast_receivers =
            num_cores_to_corerangeset_in_subcoregrids(receiver_start_core, num_cores - 1, matmul_core_rect, row_major);
    }

    // Mcast args
    // The descriptor path takes semaphore ids as given, so these are hand-assigned. 0 and 1 are
    // exactly what CreateSemaphore returned on a fresh program, which keeps the ids baked into the
    // sender/receiver compile-time args below -- and hence the kernel ELFs -- unchanged. The
    // descriptors themselves are pushed alongside the CBs further down.
    constexpr std::uint32_t in0_mcast_sender_semaphore_id = 0;
    constexpr std::uint32_t in0_mcast_receiver_semaphore_id = 1;

    CoreCoord top_left_core = in0_mcast_receiver_cores_bounding_box.start_coord;
    CoreCoord bottom_right_core = in0_mcast_receiver_cores_bounding_box.end_coord;
    auto top_left_core_physical = device->worker_core_from_logical_core(top_left_core);
    auto bottom_right_core_physical = device->worker_core_from_logical_core(bottom_right_core);

    uint32_t num_batch_compute = use_indices ? num_active : nnz.value_or(sparsity.logical_volume());
    // Compact output packs only the `nnz` active batch pairs in scan order. Detect it exactly as the
    // device op (device/sparse/sparse_matmul_device_operation.cpp): [1, nnz, M, N]. Shape matching,
    // rather than volume matching, prevents a same-volume tensor with incompatible geometry from
    // selecting compact writer indexing.
    // (Orthogonal to indexed/gather mode, which is already compact by construction and rejects nnz:
    // the writer's skip path -- the only thing this flag guards -- is never reached there.)
    const bool compact_output =
        nnz.has_value() &&
        output_tensor.logical_shape() == ttnn::Shape{1U, nnz.value(), a.logical_shape()[-2], b.logical_shape()[-1]};

    uint32_t in0_num_subblocks = (out_block_h / out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
    const auto& a_shape_logical = get_matmul_tensor_logical_shape(a, /*transpose=*/false);
    const auto in0_last_ktile_w = a_shape_logical[-1] % in0_tile.get_width();

    // We don't support transpose for this program configuration. However, we retain the logic here
    // to keep the code consistent with the other program configurations.
    const auto transpose_a = false;
    const auto transpose_b = false;
    const auto in0_tensor_stride_w = transpose_a ? Mt : 1;
    const auto in0_tensor_stride_h = transpose_a ? 1 : Kt;
    const auto in0_tensor_next_block_stride = in0_block_w * in0_tensor_stride_w;
    const auto in0_tensor_next_h_dim_block_stride = in0_block_h * in0_tensor_stride_h;

    const auto in1_tensor_stride_w = transpose_b ? Kt : 1;
    const auto in1_tensor_stride_h = transpose_b ? 1 : Nt;
    const auto in1_tensor_next_block_stride = in0_block_w * in1_tensor_stride_h;
    const auto in1_tensor_next_w_dim_block_stride = in1_block_w * in1_tensor_stride_w;

    std::vector<uint32_t> in0_sender_compile_time_args;
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
        (std::uint32_t)0,  // in0_last_ktile_h (transpose not supported for sparse)

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
        (std::uint32_t)Mt * Kt,  // MtKt
        (std::uint32_t)batchA,   // batchA
        (std::uint32_t)batchA,   // batchA
        (std::uint32_t)false,    // reuse_in0_in_CB
        // sparsity args
        (std::uint32_t)batchB,                                  // batchB
        (std::uint32_t)sparsity.buffer()->aligned_page_size(),  // sparsity_pagesize
        (std::uint32_t)!is_input_a_sparse,                      // bcast_A
        (std::uint32_t)get_batch_from_reader,                   // get_batch_from_reader
        // fuse op args
        (std::uint32_t)false,  // fuse_op
    };
    tt::tt_metal::TensorAccessorArgs(*in0_buffer).append_to(in0_sender_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*sparsity_buffer).append_to(in0_sender_compile_time_args);
    // num_batch_compute (== nnz when supplied). The sender uses this to validate, on-device, that
    // count_nonzero(sparsity) matches the loop count baked into the receiver/compute kernels, failing
    // loudly instead of deadlocking. See https://github.com/tenstorrent/tt-metal/issues/45943.
    in0_sender_compile_time_args.push_back((std::uint32_t)num_batch_compute);

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
        (std::uint32_t)Kt * Nt,  // KtNt
        (std::uint32_t)batchA,   // batchA
        (std::uint32_t)true,     // bcast_B
        // sparsity args (in indexed/gather mode this slot carries the active-group id list)
        (std::uint32_t)batchB,                                    // batchB
        (std::uint32_t)in1_sparsity_buffer->aligned_page_size(),  // sparsity_pagesize

        // WRITER
        // out tensor args
        (std::uint32_t)1,                    // out_tensor_stride_w
        (std::uint32_t)Nt,                   // out_tensor_stride_h
        (std::uint32_t)out_subblock_w,       // out_tensor_next_subblock_stride_w
        (std::uint32_t)out_subblock_h * Nt,  // out_tensor_next_subblock_stride_h
        (std::uint32_t)out_block_w,          // out_tensor_next_w_dim_block_stride
        (std::uint32_t)out_block_h * Nt,     // out_tensor_next_h_dim_block_stride
        // out subblock args
        (std::uint32_t)out_subblock_w,                     // out_subblock_w
        (std::uint32_t)out_subblock_h,                     // out_subblock_h
        (std::uint32_t)(out_subblock_w * out_subblock_h),  // out_subblocks_w * out_subblocks_h
        // batch args
        (std::uint32_t)Mt * Nt,  // MtNt
        // bias args (placeholders)
        (std::uint32_t)0,  // in3_tensor_stride_w
        // fuse op args
        (std::uint32_t)false,  // fuse_op
        (std::uint32_t)false,  // fuse_op_reduce_scatter
        (std::uint32_t)compact_output,
    };

    // Append TensorAccessorArgs
    tt::tt_metal::TensorAccessorArgs(*in1_buffer).append_to(in1_sender_writer_compile_time_args);
    // Indexed/gather mode reuses this slot for the active-group id list (see in1_sparsity_buffer).
    tt::tt_metal::TensorAccessorArgs(*in1_sparsity_buffer).append_to(in1_sender_writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*out_buffer).append_to(in1_sender_writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs().append_to(in1_sender_writer_compile_time_args);  // placeholder for bias

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
        (std::uint32_t)num_batch_compute,      // batch
        (std::uint32_t)get_batch_from_reader,  // get_batch_from_reader
    };

    std::map<std::string, std::string> mm_kernel_defines;
    std::map<std::string, std::string> mm_kernel_in0_sender_writer_defines;
    std::map<std::string, std::string> mm_kernel_in1_sender_writer_defines;

    mm_kernel_defines["FUSE_ACTIVATION"] = "0";
    if (packer_l1_acc_en) {
        mm_kernel_defines["PACKER_L1_ACC"] = "1";
    }
    if (fp32_dest_acc_en) {
        mm_kernel_defines["FP32_DEST_ACC_EN"] = "1";
    }

    ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
        device->arch(), num_cores, mm_kernel_defines);
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(),
        num_cores,
        mm_kernel_defines,
        ttnn::get_throttle_level(operation_attributes.compute_kernel_config));

    mm_kernel_in1_sender_writer_defines["SKIP_MCAST"] = "1";

    // in1 is the reader of weights/output writer, and we choose to make it use the optimized reader noc
    tt_metal::NOC in0_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    tt_metal::NOC in1_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());

    // Helper to convert std::map defines to KernelDescriptor::Defines (vector of pairs). The map
    // iterates in sorted key order, so the resulting vector -- and the descriptor hash over it -- is
    // deterministic across builds.
    auto map_to_defines = [](const std::map<std::string, std::string>& m) -> KernelDescriptor::Defines {
        KernelDescriptor::Defines result;
        result.reserve(m.size());
        for (const auto& [k, v] : m) {
            result.emplace_back(k, v);
        }
        return result;
    };

    KernelDescriptor in0_sender_kernel_desc;
    in0_sender_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp";
    in0_sender_kernel_desc.core_ranges = in0_mcast_sender_cores;
    in0_sender_kernel_desc.compile_time_args = in0_sender_compile_time_args;
    in0_sender_kernel_desc.defines = map_to_defines(mm_kernel_in0_sender_writer_defines);
    in0_sender_kernel_desc.named_compile_time_args = {
        {"cb_in0", tt::CBIndex::c_0},
        {"cb_in0_sharded", tt::CBIndex::c_2},
        {"cb_sparsity", tt::CBIndex::c_6},
        {"num_active", num_active},  // indexed/gather mode loop count (0 = not indexed)
    };
    in0_sender_kernel_desc.config =
        DataMovementConfigDescriptor{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = in0_noc};

    KernelDescriptor in0_receiver_kernel_desc;
    in0_receiver_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_receiver.cpp";
    in0_receiver_kernel_desc.core_ranges = in0_mcast_receivers;
    in0_receiver_kernel_desc.compile_time_args = in0_receiver_compile_time_args;
    in0_receiver_kernel_desc.named_compile_time_args = {
        {"cb_in0", tt::CBIndex::c_0},
    };
    in0_receiver_kernel_desc.config =
        DataMovementConfigDescriptor{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = in0_noc};

    KernelDescriptor in1_sender_writer_kernel_desc;
    in1_sender_writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
        "reader_bmm_tile_layout_in1_sender_writer_padding.cpp";
    in1_sender_writer_kernel_desc.core_ranges = all_cores_with_work;
    in1_sender_writer_kernel_desc.compile_time_args = in1_sender_writer_compile_time_args;
    in1_sender_writer_kernel_desc.defines = map_to_defines(mm_kernel_in1_sender_writer_defines);
    in1_sender_writer_kernel_desc.named_compile_time_args = {
        {"cb_in1", tt::CBIndex::c_1},
        {"cb_bias", tt::CBIndex::c_3},
        {"cb_out", tt::CBIndex::c_4},
        {"cb_sparsity", tt::CBIndex::c_7},
        {"num_active", num_active},  // indexed/gather mode loop count (0 = not indexed)
    };
    in1_sender_writer_kernel_desc.config =
        DataMovementConfigDescriptor{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = in1_noc};

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
        num_batch_compute,       // batch_nnz
        out_block_tiles,         // out_block_num_tiles

        false,                  // untilize_out
        get_batch_from_reader,  // get_batch_from_reader
        false,                  // in0_transpose_tile
    };

    // Create compute kernel
    // bool fp32_dest_acc_en = false;
    // Gelu currently has better accuracy when run in approx mode
    // bool math_approx_mode = false;
    KernelDescriptor compute_kernel_desc;
    compute_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp";
    compute_kernel_desc.core_ranges = all_cores_with_work;
    compute_kernel_desc.compile_time_args = compute_kernel_args;
    compute_kernel_desc.defines = map_to_defines(mm_kernel_defines);
    compute_kernel_desc.named_compile_time_args = {
        {"cb_in0", tt::CBIndex::c_0},
        {"cb_in1", tt::CBIndex::c_1},
        {"cb_bias", tt::CBIndex::c_3},
        {"cb_out", tt::CBIndex::c_4},
        {"cb_intermed0", tt::CBIndex::c_5},
        {"cb_in0_transposed", tt::CBIndex::c_10},
    };
    // unpack_to_dest_mode and opt_level are left at their descriptor defaults to preserve
    // behaviour: an empty unpack_to_dest_mode matches the legacy ComputeConfig default, and the
    // default opt_level applies O2 for data movement and O3 for compute, as the legacy configs did.
    compute_kernel_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode};
    ////////////////////////////////////////////////////////////////////////////
    //                      Descriptor Assembly
    ////////////////////////////////////////////////////////////////////////////
    ProgramDescriptor desc;

    const TileDescriptor in0_tile_desc{in0_tile};
    const TileDescriptor in1_tile_desc{in1_tile};
    const TileDescriptor output_tile_desc{output_tile};

    // CB push order is preserved from the legacy factory (c_0, c_1, c_6, c_7, [c_5], c_4): CB
    // descriptors are consumed positionally, so keeping the order identical keeps the resulting
    // program structurally identical. in0/in1/sparsity/output are all interleaved here, so every CB
    // below is a plain L1 allocation described by size and format alone; the framework's CB-side
    // patching applies to the .buffer/.tensor fields, which belong to tensor-backed CBs.
    uint32_t src0_cb_index = tt::CBIndex::c_0;
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
    log_debug(
        LogOp,
        "CB {} :: PS = {}, NP = {}, TOTAL = {}",
        src0_cb_index,
        in0_single_tile_size,
        in0_CB_size / in0_single_tile_size,
        in0_CB_size);

    uint32_t src1_cb_index = tt::CBIndex::c_1;
    {
        CBDescriptor cb_desc;
        cb_desc.total_size = in1_CB_size;
        cb_desc.core_ranges = all_cores;
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_1,
            .data_format = in1_data_format,
            .page_size = in1_aligned_tile_size,
            .tile = in1_tile_desc});
        desc.cbs.push_back(std::move(cb_desc));
    }
    log_debug(
        LogOp,
        "CB {} :: PS = {}, NP = {}, TOTAL = {}",
        src1_cb_index,
        in1_single_tile_size,
        in1_CB_size / in1_single_tile_size,
        in1_CB_size);

    const uint32_t sparsity_cb_size = static_cast<uint32_t>(sparsity.buffer()->aligned_page_size());
    {
        CBDescriptor cb_desc;
        cb_desc.total_size = sparsity_cb_size;
        cb_desc.core_ranges = all_cores;
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_6,
            .data_format = tt::tt_metal::datatype_to_dataformat_converter(sparsity.dtype()),
            .page_size = sparsity_cb_size});
        desc.cbs.push_back(std::move(cb_desc));
    }

    // c_7 is the in1 sender/writer's slot; in indexed/gather mode it holds the active-group id list
    // instead of a sparsity page, so it is sized and typed from whichever tensor that kernel reads.
    const uint32_t in1_sparsity_cb_size = static_cast<uint32_t>(in1_sparsity_buffer->aligned_page_size());
    {
        CBDescriptor cb_desc;
        cb_desc.total_size = in1_sparsity_cb_size;
        cb_desc.core_ranges = all_cores;
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_7,
            .data_format = tt::tt_metal::datatype_to_dataformat_converter(in1_sparsity_tensor.dtype()),
            .page_size = in1_sparsity_cb_size});
        desc.cbs.push_back(std::move(cb_desc));
    }

    uint32_t output_cb_index = tt::CBIndex::c_4;
    uint32_t interm0_cb_index = tt::CBIndex::c_5;
    if (interm0_data_format != output_data_format) {
        // interm0
        {
            CBDescriptor cb_desc;
            cb_desc.total_size = interm0_CB_size;
            cb_desc.core_ranges = all_cores;
            cb_desc.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = tt::CBIndex::c_5,
                .data_format = interm0_data_format,
                .page_size = interm0_single_tile_size,
                .tile = output_tile_desc});
            desc.cbs.push_back(std::move(cb_desc));
        }
        log_debug(
            LogOp,
            "CB {} :: PS = {}, NP = {}, TOTAL = {}",
            interm0_cb_index,
            interm0_single_tile_size,
            interm0_CB_size / interm0_single_tile_size,
            interm0_CB_size);

        // output
        CBDescriptor cb_desc;
        cb_desc.total_size = out_CB_size;
        cb_desc.core_ranges = all_cores;
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_4,
            .data_format = output_data_format,
            .page_size = output_single_tile_size,
            .tile = output_tile_desc});
        desc.cbs.push_back(std::move(cb_desc));
    } else {
        // share buffer: c_4 and c_5 alias one L1 allocation, expressed as two format descriptors
        // on a single CBDescriptor.
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
        desc.cbs.push_back(std::move(cb_desc));
    }
    log_debug(
        LogOp,
        "CB {} :: PS = {}, NP = {}, TOTAL = {}",
        output_cb_index,
        output_single_tile_size,
        out_CB_size / output_single_tile_size,
        out_CB_size);

    desc.semaphores.push_back(
        SemaphoreDescriptor{.id = in0_mcast_sender_semaphore_id, .core_ranges = all_cores, .initial_value = INVALID});
    desc.semaphores.push_back(
        SemaphoreDescriptor{.id = in0_mcast_receiver_semaphore_id, .core_ranges = all_cores, .initial_value = INVALID});

    // Parameters for last row, col, or block, no need to re-calc h-dim since there's no split on height
    uint32_t last_per_core_N = Nt % per_core_N == 0 ? per_core_N : Nt % per_core_N;
    uint32_t last_out_block_w = last_per_core_N % out_block_w == 0 ? out_block_w : last_per_core_N % out_block_w;
    uint32_t last_out_num_blocks_w = ((last_per_core_N - 1) / out_block_w) + 1;
    uint32_t last_block_num_nonzero_subblocks_w = ((last_out_block_w - 1) / out_subblock_w) + 1;
    uint32_t last_subblock_of_last_block_w =
        last_out_block_w % out_subblock_w == 0 ? out_subblock_w : last_out_block_w % out_subblock_w;
    uint32_t last_block_padded_subblock_tiles_addr_skip =
        output_single_tile_size * (out_subblock_w - last_subblock_of_last_block_w);
    uint32_t last_block_padded_block_tiles_w_skip =
        (out_subblock_w * out_subblock_h) * (out_block_w / out_subblock_w - last_block_num_nonzero_subblocks_w);

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

        // in0 sender and in1 sender
        if (core == start_core) {
            std::vector<uint32_t> mm_in0_sender_args = {
                // in0 tensor args
                (std::uint32_t)in0_buffer->address(),
                (std::uint32_t)Kt * per_core_M * output_idx_y,  // in0_tensor_start_tile_id
                // in0 mcast args
                (std::uint32_t)start_core_noc.x,  // in0_mcast_dest_noc_start_x
                (std::uint32_t)start_core_noc.y,  // in0_mcast_dest_noc_start_y
                (std::uint32_t)end_core_noc.x,    // in0_mcast_dest_noc_end_x
                (std::uint32_t)end_core_noc.y,    // in0_mcast_dest_noc_end_y

                // padding args
                (std::uint32_t)out_block_h,  // last_block_h
                // sparsity args
                (std::uint32_t)sparsity_buffer->address()  // sparsity_addr
            };

            // in0 and sparsity are declared as Buffer* bindings so the framework patches their
            // addresses in place on a cache hit. Every other slot here is derived from the hashed
            // shapes/attributes, so a hit guarantees it is already correct.
            std::vector<std::variant<std::uint32_t, tt::tt_metal::Buffer*>> in0_args(
                mm_in0_sender_args.begin(), mm_in0_sender_args.end());
            in0_args[0] = in0_buffer;
            in0_args[7] = sparsity_buffer;
            in0_sender_kernel_desc.emplace_runtime_args(core, in0_args);
        }
        // in0 receiver and in 1 sender
        else {
            std::vector<uint32_t> mm_in0_receiver_args = {
                // in0 mcast args
                (std::uint32_t)top_left_core_physical.x,  // in0_mcast_sender_noc_x
                (std::uint32_t)top_left_core_physical.y   // in0_mcast_sender_noc_y
            };
            // The receiver's args are both NoC coordinates, fixed for a given core grid, so these
            // go in as plain values.
            in0_receiver_kernel_desc.runtime_args.emplace_back(core, mm_in0_receiver_args);
        }
        if (i < num_cores_with_work) {
            std::vector<uint32_t> mm_in1_sender_writer_args = {
                // READER
                // in1 tensor args
                (std::uint32_t)in1_buffer->address(),
                (std::uint32_t)per_core_N * output_idx_x,  // in1_tensor_start_tile_id
                // in1 mcast args
                (std::uint32_t)0,  // in1_mcast_dest_noc_start_x
                (std::uint32_t)0,  // in1_mcast_dest_noc_start_y
                (std::uint32_t)0,  // in1_mcast_dest_noc_end_x
                (std::uint32_t)0,  // in1_mcast_dest_noc_end_y

                // sparsity args (the active-group id list in indexed/gather mode)
                (std::uint32_t)in1_sparsity_buffer->address(),  // sparsity_addr

                // WRITER
                // out tensor args
                (std::uint32_t)out_buffer->address(),
                ((std::uint32_t)output_idx_x * per_core_N) +
                    (output_idx_y * per_core_M * Nt)  // out_tensor_start_tile_id
            };

            if (output_idx_x == num_blocks_x - 1) {
                // padding args (READER)
                mm_in1_sender_writer_args.push_back(last_out_block_w);

                // padding args (WRITER)
                mm_in1_sender_writer_args.push_back(out_block_h / out_subblock_h);
                mm_in1_sender_writer_args.push_back(out_subblock_h);
                mm_in1_sender_writer_args.push_back(0);
                mm_in1_sender_writer_args.push_back(out_block_w / out_subblock_w);  // out_num_nonzero_subblocks_w
                mm_in1_sender_writer_args.push_back(last_block_num_nonzero_subblocks_w);
                mm_in1_sender_writer_args.push_back(last_subblock_of_last_block_w);
                mm_in1_sender_writer_args.push_back(last_block_padded_subblock_tiles_addr_skip);
                mm_in1_sender_writer_args.push_back(last_block_padded_block_tiles_w_skip);
            } else {
                // padding args (READER)
                mm_in1_sender_writer_args.push_back(out_block_w);

                // padding args (WRITER)
                mm_in1_sender_writer_args.push_back(out_block_h / out_subblock_h);
                mm_in1_sender_writer_args.push_back(out_subblock_h);
                mm_in1_sender_writer_args.push_back(0);
                mm_in1_sender_writer_args.push_back(out_block_w / out_subblock_w);  // out_num_nonzero_subblocks_w
                mm_in1_sender_writer_args.push_back(out_block_w / out_subblock_w);
                mm_in1_sender_writer_args.push_back(out_subblock_w);
                mm_in1_sender_writer_args.push_back(0);
                mm_in1_sender_writer_args.push_back(0);
            }

            mm_in1_sender_writer_args.push_back(0);
            mm_in1_sender_writer_args.push_back(0);

            if (output_idx_x == num_blocks_x - 1) {
                mm_in1_sender_writer_args.push_back(last_out_num_blocks_w);
            } else {
                mm_in1_sender_writer_args.push_back(out_num_blocks_x);
            }

            mm_in1_sender_writer_args.push_back(0);
            mm_in1_sender_writer_args.push_back(0);
            mm_in1_sender_writer_args.push_back(0);
            mm_in1_sender_writer_args.push_back(0);
            mm_in1_sender_writer_args.push_back(0);

            std::vector<std::variant<std::uint32_t, tt::tt_metal::Buffer*>> in1_args(
                mm_in1_sender_writer_args.begin(), mm_in1_sender_writer_args.end());
            in1_args[0] = in1_buffer;
            in1_args[6] = in1_sparsity_buffer;
            in1_args[7] = out_buffer;
            in1_sender_writer_kernel_desc.emplace_runtime_args(core, in1_args);
        }
    }

    // Kernel push order defines each kernel's handle (its index in desc.kernels). The in0 receiver
    // is conditional, so indices after it shift on the single-core geometry -- fine here because
    // buffer bindings are resolved positionally at cache-miss time, but anything that later hard-codes
    // a kernel index (e.g. a hand-written override) must account for it.
    desc.kernels.push_back(std::move(in0_sender_kernel_desc));
    if (in0_mcast_receivers.num_cores() > 0) {
        desc.kernels.push_back(std::move(in0_receiver_kernel_desc));
    }
    desc.kernels.push_back(std::move(in1_sender_writer_kernel_desc));
    desc.kernels.push_back(std::move(compute_kernel_desc));

    return desc;
}

}  // namespace ttnn::prim
