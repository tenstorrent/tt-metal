// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/tile_sparse/factory/tile_sparse_matmul_program_factory.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config.hpp"

#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "tt-metalium/work_split.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"
#include "tt-metalium/buffer.hpp"
#include <tt-metalium/tt_metal.hpp>

namespace ttnn::prim {

TileSparseMatmulProgramFactory::cached_program_t TileSparseMatmulProgramFactory::create(
    const TileSparseMatmulParams& operation_attributes,
    const TileSparseMatmulInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    tt::tt_metal::Program program{};
    using namespace tt;
    using namespace operations::matmul::utilities;

    // Convert to standard matmul attributes for program config selection
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

    const auto& a = tensor_args.input_tensors.at(0);
    const auto& b = tensor_args.input_tensors.at(1);
    const auto& output_tensor = tensor_return_value.at(0);

    // Get tile-sparse program config (currently using 1D multicast)
    // Handle case where a different config type is returned
    operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig program_config;
    if (auto* config_1d =
            std::get_if<operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>(&chosen_program_config)) {
        program_config = *config_1d;
        // Tile-sparse matmul requires mcast_in0; override if necessary
        program_config.mcast_in0 = true;
    } else {
        // Fallback: create a simple 1D config for tile-sparse matmul
        auto* device = a.device();
        auto grid_size = device->compute_with_storage_grid_size();
        program_config = operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig{
            .compute_with_storage_grid_size = grid_size,
            .in0_block_w = 1,
            .out_subblock_h = 1,
            .out_subblock_w = 1,
            .out_block_h = 1,
            .out_block_w = 1,
            .per_core_M = 1,
            .per_core_N = 1,
            .mcast_in0 = true,
        };
    }
    auto compute_with_storage_grid_size = program_config.compute_with_storage_grid_size;
    auto in0_block_w = program_config.in0_block_w;
    auto out_subblock_h = program_config.out_subblock_h;
    auto out_subblock_w = program_config.out_subblock_w;
    auto out_block_h = program_config.out_block_h;
    auto out_block_w = program_config.out_block_w;
    auto per_core_M = program_config.per_core_M;
    auto per_core_N = program_config.per_core_N;
    auto mcast_in0 = program_config.mcast_in0;

    // Get sparsity information and create mask buffer
    uint32_t nnz_tiles = 0;
    const TileSparsityMask* active_mask = nullptr;
    tt::tt_metal::Buffer* sparsity_mask_buffer = nullptr;

    if (operation_attributes.input_a_sparsity_mask.has_value()) {
        active_mask = &operation_attributes.input_a_sparsity_mask.value();
        nnz_tiles = active_mask->nnz_tiles;
    } else if (operation_attributes.input_b_sparsity_mask.has_value()) {
        active_mask = &operation_attributes.input_b_sparsity_mask.value();
        nnz_tiles = active_mask->nnz_tiles;
    }

    // Get device from input tensor for buffer allocation
    auto* const device_for_buffer = a.device();

    // Note: For MVP, we skip the tile indices buffer creation since the actual
    // tile-skipping kernel logic is not yet implemented. The sparsity mask info
    // is stored in the shared_variables for future use when tile-sparse kernels
    // are added.
    // TODO: Implement tile_indices buffer creation and transfer when adding
    // custom tile-sparse dataflow kernels.
    std::shared_ptr<tt::tt_metal::Buffer> tile_indices_buffer;
    (void)device_for_buffer;  // Suppress unused variable warning for now

    const auto& ashape = get_matmul_tensor_padded_shape(a, /*transpose=*/false);
    const auto& bshape = get_matmul_tensor_padded_shape(b, /*transpose=*/false);
    const auto in0_tile = get_matmul_tile(a, /*transpose=*/false);
    const auto in1_tile = get_matmul_tile(b, /*transpose=*/false);
    const auto output_tile = tt::tt_metal::Tile({in0_tile.get_height(), in1_tile.get_width()});

    // CB dataformats
    const auto in0_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    const auto in1_data_format = tt_metal::datatype_to_dataformat_converter(b.dtype());
    const auto output_data_format = tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());

    auto* const device = a.device();

    const auto in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    const auto in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    const auto output_single_tile_size = output_tile.get_tile_size(output_data_format);
    const auto interm0_single_tile_size = output_tile.get_tile_size(output_data_format);

    auto* const in0_buffer = a.buffer();
    auto* const in1_buffer = b.buffer();
    auto* const out_buffer = output_tensor.buffer();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config.value());

    // Matmul parameters setup
    const auto batchA = get_batch_size(ashape);
    const auto batchB = get_batch_size(bshape);

    const auto Mt = get_M_dim(ashape, in0_tile, /*fuse_batch=*/false);
    const auto Kt = get_K_dim(ashape, in0_tile);
    const auto Nt = get_N_dim(bshape, in1_tile);

    // mcast_in0 requires num_blocks_y = 1: the single sender broadcasts in0 for its
    // output row to ALL receivers. If num_blocks_y > 1, receivers at other rows would
    // get the wrong in0 data. Fix: set per_core_M = Mt so num_blocks_y = Mt/Mt = 1.
    per_core_M = Mt;
    if (per_core_M % out_block_h != 0) {
        out_block_h = out_subblock_h;
    }

    TT_FATAL(Kt % in0_block_w == 0, "Kt ({}) must be divisible by in0_block_w ({})", Kt, in0_block_w);
    TT_FATAL(mcast_in0, "Only mcast_in0 is supported for tile-sparse matmul");

    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_available = num_cores_x * num_cores_y;

    uint32_t num_blocks_y = ((Mt - 1) / per_core_M) + 1;
    uint32_t num_blocks_x = ((Nt - 1) / per_core_N) + 1;
    uint32_t num_blocks_total = num_blocks_y * num_blocks_x;

    TT_FATAL(
        num_blocks_total <= num_cores_available,
        "Number of blocks exceeds number of cores available: {} blocks > {} cores",
        num_blocks_total,
        num_cores_available);

    using tt::tt_metal::num_cores_to_corerangeset;

    uint32_t num_blocks = Kt / in0_block_w;

    // ---- Tile-sparse K-skip path ----
    // When a B-sparsity mask is present and num_blocks <= 32, use kernels that
    // skip inactive K-blocks entirely (no DRAM read, no CB push, no compute).
    bool use_sparse_kernels = (operation_attributes.input_b_sparsity_mask.has_value() && num_blocks <= 32u);

    // Per-core K-active bitmasks: bit k = 1 means K-block k should be read.
    // Dense default: all bits set (0xFFFFFFFF = all active).
    uint32_t global_k_active_mask = 0xFFFFFFFFu;
    std::vector<uint32_t> per_core_k_masks(num_blocks_total, 0xFFFFFFFFu);

    if (use_sparse_kernels) {
        global_k_active_mask = 0u;
        std::fill(per_core_k_masks.begin(), per_core_k_masks.end(), 0u);

        const auto& mask_b = operation_attributes.input_b_sparsity_mask.value();
        // B mask shape: [Kt, Nt]. Flat index = k_tile * Nt + n_tile.
        for (uint32_t tile_idx : mask_b.tile_indices) {
            uint32_t k_tile = tile_idx / Nt;
            uint32_t n_tile = tile_idx % Nt;
            uint32_t k_block = k_tile / in0_block_w;
            uint32_t core_n = n_tile / per_core_N;  // core index along N axis

            if (k_block < 32u && core_n < (uint32_t)num_blocks_total) {
                per_core_k_masks[core_n] |= (1u << k_block);
                global_k_active_mask |= (1u << k_block);
            }
        }

        // If all K-blocks are inactive (B is entirely zero), fall back to dense path.
        // The sparse compute kernel produces no cb_out tiles when k_active_mask=0,
        // which would stall the writer's output section.
        if (global_k_active_mask == 0u) {
            use_sparse_kernels = false;
            global_k_active_mask = 0xFFFFFFFFu;
            std::fill(per_core_k_masks.begin(), per_core_k_masks.end(), 0xFFFFFFFFu);
        }
    }
    bool packer_l1_acc_en = packer_l1_acc && num_blocks > 1;

    const auto interm0_data_format = packer_l1_acc_en
                                         ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                                         : (fp32_dest_acc_en ? tt::DataFormat::Float32 : output_data_format);

    uint32_t in0_block_h = out_block_h;
    uint32_t in1_block_w = out_block_w;
    uint32_t in0_num_blocks_y = per_core_M / out_block_h;
    uint32_t in1_num_blocks_x = per_core_N / out_block_w;

    uint32_t in0_block_tiles = in0_block_h * in0_block_w;
    uint32_t in0_CB_tiles = in0_block_tiles;
    if (batchA * batchB * num_blocks > 1) {
        in0_CB_tiles *= ttnn::operations::matmul::utilities::MCAST_INPUT_BUFFERING_DEPTH;
    }
    uint32_t in0_CB_size = in0_CB_tiles * in0_single_tile_size;

    uint32_t in1_block_tiles = out_block_w * in0_block_w;
    uint32_t in1_CB_tiles = in1_block_tiles;
    if (batchA * batchB * num_blocks > 1) {
        in1_CB_tiles *= ttnn::operations::matmul::utilities::MCAST_INPUT_BUFFERING_DEPTH;
    }
    uint32_t in1_CB_size = in1_CB_tiles * in1_single_tile_size;

    uint32_t out_block_tiles = out_block_h * out_block_w;
    uint32_t out_CB_tiles = out_block_tiles;
    uint32_t out_CB_size = out_CB_tiles * output_single_tile_size;
    uint32_t interm0_CB_tiles = out_block_tiles;
    uint32_t interm0_CB_size = interm0_CB_tiles * interm0_single_tile_size;

    CoreCoord start_core = {0, 0};
    uint32_t num_cores_with_work = num_blocks_total;
    uint32_t num_cores = num_cores_with_work;

    constexpr bool row_major = true;
    CoreRangeSet all_cores =
        num_cores_to_corerangeset(start_core, num_cores, compute_with_storage_grid_size, row_major);

    CoreRangeSet all_cores_with_work =
        num_cores_to_corerangeset(num_cores_with_work, compute_with_storage_grid_size, row_major);
    CoreRange mcast_cores_bounding_box = all_cores_with_work.bounding_box();

    // In mcast_in0 protocol, only core 0 is the sender; all others are receivers.
    uint32_t in0_sender_num_cores = 1;
    CoreRangeSet in0_mcast_sender_cores =
        num_cores_to_corerangeset(in0_sender_num_cores, compute_with_storage_grid_size, row_major);
    CoreRangeSet in0_mcast_receivers;
    if (num_cores > 1) {
        auto receiver_start_core = start_core.x != (compute_with_storage_grid_size.x - 1)
                                       ? CoreCoord{start_core.x + 1, start_core.y}
                                       : CoreCoord{start_core.x, start_core.y + 1};
        in0_mcast_receivers =
            num_cores_to_corerangeset(receiver_start_core, num_cores - 1, compute_with_storage_grid_size, row_major);
    }

    // Semaphores for multicast
    auto mcast_sender_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto mcast_receiver_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);

    CoreCoord top_left_core = mcast_cores_bounding_box.start_coord;
    CoreCoord bottom_right_core = mcast_cores_bounding_box.end_coord;
    auto top_left_core_physical = device->worker_core_from_logical_core(top_left_core);
    auto bottom_right_core_physical = device->worker_core_from_logical_core(bottom_right_core);

    uint32_t in0_num_subblocks = (out_block_h / out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;

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

    uint32_t out_num_blocks_x = in1_num_blocks_x;
    uint32_t out_num_blocks_y = in0_num_blocks_y;

    // Compile time args for reader kernel (must match reader_bmm_tile_layout_in0_sender_padding.cpp)
    std::vector<uint32_t> reader_compile_time_args = {
        // in0 tensor args (indices 0-3)
        (std::uint32_t)in0_tensor_stride_w,
        (std::uint32_t)in0_tensor_stride_h,
        (std::uint32_t)in0_tensor_next_block_stride,
        (std::uint32_t)in0_tensor_next_h_dim_block_stride,
        // in0 block args (indices 4-7)
        (std::uint32_t)in0_block_w,
        (std::uint32_t)in0_block_h,
        (std::uint32_t)in0_block_num_tiles,
        (std::uint32_t)0,  // in0_last_ktile_w (no padding for now)
        // shard args (indices 8-10)
        (std::uint32_t)0,  // extract_shard_sub_blocks = false
        (std::uint32_t)0,  // shard_width_in_tiles
        (std::uint32_t)0,  // shard_height_in_tiles
        // in0/in1 common args (indices 11-13)
        (std::uint32_t)num_blocks,
        (std::uint32_t)out_num_blocks_x,
        (std::uint32_t)out_num_blocks_y,
        // mcast args (indices 14-17)
        (std::uint32_t)mcast_sender_semaphore_id,
        (std::uint32_t)mcast_receiver_semaphore_id,
        (std::uint32_t)num_cores - 1,  // in0_mcast_num_dests
        (std::uint32_t)num_cores - 1,  // in0_mcast_num_cores
        // batch args (indices 18-20)
        (std::uint32_t)Mt * Kt,
        (std::uint32_t)batchA * batchB,  // batch (combined)
        (std::uint32_t)0,                // batchB = 0 to disable MoE sparsity path
        // sparsity args (indices 21-24)
        (std::uint32_t)0,  // sparsity_pagesize (not used for tile-sparse)
        (std::uint32_t)0,  // bcast_A = false
        (std::uint32_t)0,  // get_batch_from_reader = false
        (std::uint32_t)0,  // fuse_op = false
    };
    tt::tt_metal::TensorAccessorArgs(*in0_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs().append_to(reader_compile_time_args);  // sparsity placeholder (not used)

    // Compile time args for writer kernel (must match reader_bmm_tile_layout_in1_sender_writer_padding.cpp)
    std::vector<uint32_t> writer_compile_time_args = {
        // READER section
        // in1 tensor args (indices 0-3)
        (std::uint32_t)in1_tensor_stride_w,
        (std::uint32_t)in1_tensor_stride_h,
        (std::uint32_t)in1_tensor_next_block_stride,
        (std::uint32_t)in1_tensor_next_w_dim_block_stride,
        // in1 block args (indices 4-6)
        (std::uint32_t)in1_block_w,
        (std::uint32_t)in0_block_w,                // in1_block_h = in0_block_w (K dimension block)
        (std::uint32_t)in1_block_w * in0_block_w,  // in1_block_num_tiles
        // common args (indices 7-9)
        (std::uint32_t)num_blocks,
        (std::uint32_t)out_num_blocks_x,
        (std::uint32_t)out_num_blocks_y,
        // mcast args (indices 10-13) - not used when SKIP_MCAST is defined
        (std::uint32_t)0,  // mcast_sender_semaphore (placeholder)
        (std::uint32_t)0,  // mcast_receiver_semaphore (placeholder)
        (std::uint32_t)0,  // in1_mcast_num_dests
        (std::uint32_t)0,  // in1_mcast_num_cores
        // batch args (indices 14-16)
        (std::uint32_t)Kt * Nt,          // KtNt
        (std::uint32_t)batchA * batchB,  // batch (combined)
        (std::uint32_t)0,                // bcast_B = false
        // sparsity args (indices 17-18)
        (std::uint32_t)0,  // batchB = 0 to disable MoE sparsity path
        (std::uint32_t)0,  // sparsity_pagesize (not used for tile-sparse)
        // WRITER section
        // out tensor args (indices 19-24)
        (std::uint32_t)1,                    // out_tensor_stride_w
        (std::uint32_t)Nt,                   // out_tensor_stride_h
        (std::uint32_t)out_subblock_w,       // out_tensor_next_subblock_stride_w
        (std::uint32_t)out_subblock_h * Nt,  // out_tensor_next_subblock_stride_h
        (std::uint32_t)out_block_w,          // out_tensor_next_w_dim_block_stride
        (std::uint32_t)out_block_h * Nt,     // out_tensor_next_h_dim_block_stride
        // out subblock args (indices 25-27)
        (std::uint32_t)out_subblock_w,
        (std::uint32_t)out_subblock_h,
        (std::uint32_t)(out_subblock_w * out_subblock_h),  // out_subblock_tile_count
        // batch args (index 28)
        (std::uint32_t)Mt * Nt,  // MtNt
        // bias args (index 29) - placeholder, no bias
        (std::uint32_t)0,
        // fuse_op args (indices 30-31)
        (std::uint32_t)0,  // fuse_op_all_gather = false
        (std::uint32_t)0,  // fuse_op_reduce_scatter = false
    };
    // TensorAccessorArgs: in1 (starts at index 32), sparsity (placeholder), out
    tt::tt_metal::TensorAccessorArgs(*in1_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs().append_to(writer_compile_time_args);  // sparsity placeholder
    tt::tt_metal::TensorAccessorArgs(*out_buffer).append_to(writer_compile_time_args);

    std::map<std::string, std::string> mm_kernel_defines;
    if (packer_l1_acc_en) {
        mm_kernel_defines["PACKER_L1_ACC"] = "1";
    }
    if (fp32_dest_acc_en) {
        mm_kernel_defines["FP32_DEST_ACC_EN"] = "1";
    }
    mm_kernel_defines["FUSE_ACTIVATION"] = "0";

    ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
        device->arch(), num_cores, mm_kernel_defines);
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(device->arch(), num_cores, mm_kernel_defines);

    // NOC assignment
    tt_metal::NOC in0_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    tt_metal::NOC in1_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());

    // Create reader kernel (sender: core 0 only)
    tt::tt_metal::KernelHandle reader_kernel_id;
    if (use_sparse_kernels) {
        // Sparse path: tile-skip sender zero-fills CB for inactive K-blocks.
        std::vector<uint32_t> sparse_sender_compile_time_args = {
            (uint32_t)in0_tensor_stride_w,                 // 0: in0_stride_w
            (uint32_t)in0_tensor_stride_h,                 // 1: in0_stride_h = Kt
            (uint32_t)in0_tensor_next_block_stride,        // 2: in0_k_stride = in0_block_w
            (uint32_t)in0_tensor_next_h_dim_block_stride,  // 3: in0_m_stride = out_block_h*Kt
            (uint32_t)in0_block_w,                         // 4: in0_block_w
            (uint32_t)in0_block_h,                         // 5: in0_block_h
            (uint32_t)in0_block_num_tiles,                 // 6: in0_block_num_tiles
            (uint32_t)num_blocks,                          // 7: num_k_blocks
            (uint32_t)in0_num_blocks_y,                    // 8: num_m_blocks
            (uint32_t)mcast_sender_semaphore_id,           // 9
            (uint32_t)mcast_receiver_semaphore_id,         // 10
            (uint32_t)(num_cores - 1),                     // 11: mcast_num_dests
            (uint32_t)(num_cores - 1),                     // 12: mcast_num_cores
        };
        tt::tt_metal::TensorAccessorArgs(*in0_buffer).append_to(sparse_sender_compile_time_args);
        reader_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/matmul/device/tile_sparse/kernels/dataflow/reader_tile_sparse_in0_sender.cpp",
            in0_mcast_sender_cores,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = in0_noc,
                .compile_args = sparse_sender_compile_time_args,
                .named_compile_args = {
                    {"cb_in0", tt::CBIndex::c_0},
                }});
    } else {
        reader_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp",
            in0_mcast_sender_cores,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = in0_noc,
                .compile_args = reader_compile_time_args,
                .named_compile_args = {
                    {"cb_in0", tt::CBIndex::c_0},
                    {"cb_in0_sharded", tt::CBIndex::c_2},
                    {"cb_sparsity", tt::CBIndex::c_6},
                    {"cb_in0_intermediate", tt::CBIndex::c_8},
                }});
    }

    // Create receiver kernel (cores 1+ only)
    tt::tt_metal::KernelHandle receiver_kernel_id = 0;
    if (in0_mcast_receivers.num_cores() > 0) {
        uint32_t in0_block_num_tiles_local = out_subblock_h * in0_block_w * in0_num_subblocks;
        if (use_sparse_kernels) {
            // Sparse receiver: skips inactive K-blocks (no CB push, no semaphore exchange)
            std::vector<uint32_t> sparse_receiver_compile_time_args = {
                in0_block_num_tiles_local,    // 0: in0_block_num_tiles
                num_blocks,                   // 1: num_k_blocks
                in0_num_blocks_y,             // 2: num_m_blocks
                mcast_sender_semaphore_id,    // 3: sender_semaphore_id
                mcast_receiver_semaphore_id,  // 4: receiver_semaphore_id
            };
            receiver_kernel_id = tt_metal::CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/matmul/device/tile_sparse/kernels/dataflow/"
                "reader_tile_sparse_in0_receiver.cpp",
                in0_mcast_receivers,
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = in0_noc,
                    .compile_args = sparse_receiver_compile_time_args,
                    .named_compile_args = {
                        {"cb_in0", tt::CBIndex::c_0},
                    }});
        } else {
            std::vector<uint32_t> in0_receiver_compile_time_args = {
                in0_block_num_tiles_local,    // in0_block_num_tiles
                num_blocks,                   // num_blocks
                out_num_blocks_x,             // out_num_blocks_x
                out_num_blocks_y,             // out_num_blocks_y
                mcast_sender_semaphore_id,    // in0_mcast_sender_semaphore_id
                mcast_receiver_semaphore_id,  // in0_mcast_receiver_semaphore_id
                batchA * batchB,              // batch
                (uint32_t)0,                  // get_batch_from_reader = false
            };
            receiver_kernel_id = tt_metal::CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_receiver.cpp",
                in0_mcast_receivers,
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = in0_noc,
                    .compile_args = in0_receiver_compile_time_args,
                    .named_compile_args = {
                        {"cb_in0", tt::CBIndex::c_0},
                    }});
        }
    }

    // Create writer kernel
    tt::tt_metal::KernelHandle writer_kernel_id;
    if (use_sparse_kernels) {
        // Sparse path: combined B-reader + output-writer with K-block bitmask.
        std::vector<uint32_t> sparse_writer_compile_time_args = {
            (uint32_t)in1_tensor_stride_w,                 // 0: in1_stride_w
            (uint32_t)in1_tensor_stride_h,                 // 1: in1_stride_h = Nt
            (uint32_t)in1_tensor_next_block_stride,        // 2: in1_k_stride = in0_block_w*Nt
            (uint32_t)in1_tensor_next_w_dim_block_stride,  // 3: in1_n_stride = out_block_w
            (uint32_t)in1_block_w,                         // 4: in1_block_w
            (uint32_t)in0_block_w,                         // 5: in1_block_h = in0_block_w
            (uint32_t)(in1_block_w * in0_block_w),         // 6: in1_block_num_tiles
            (uint32_t)num_blocks,                          // 7: num_k_blocks
            (uint32_t)in1_num_blocks_x,                    // 8: num_n_blocks
            (uint32_t)in0_num_blocks_y,                    // 9: num_m_blocks
            (uint32_t)1,                                   // 10: out_stride_w
            (uint32_t)Nt,                                  // 11: out_stride_h
            (uint32_t)out_subblock_w,                      // 12: out_sb_stride_w
            (uint32_t)(out_subblock_h * Nt),               // 13: out_sb_stride_h
            (uint32_t)out_block_w,                         // 14: out_blk_stride_w
            (uint32_t)(out_block_h * Nt),                  // 15: out_blk_stride_h
            (uint32_t)out_subblock_w,                      // 16: out_subblock_w
            (uint32_t)out_subblock_h,                      // 17: out_subblock_h
            (uint32_t)(out_subblock_w * out_subblock_h),   // 18: out_sb_tiles
            (uint32_t)(Mt * Nt),                           // 19: MtNt
        };
        tt::tt_metal::TensorAccessorArgs(*in1_buffer).append_to(sparse_writer_compile_time_args);
        tt::tt_metal::TensorAccessorArgs(*out_buffer).append_to(sparse_writer_compile_time_args);
        writer_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/matmul/device/tile_sparse/kernels/dataflow/reader_tile_sparse_in1_writer.cpp",
            all_cores_with_work,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = in1_noc,
                .compile_args = sparse_writer_compile_time_args,
                .named_compile_args = {
                    {"cb_in1", tt::CBIndex::c_1},
                    {"cb_out", tt::CBIndex::c_4},
                }});
    } else {
        writer_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
            "reader_bmm_tile_layout_in1_sender_writer_padding.cpp",
            all_cores_with_work,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = in1_noc,
                .compile_args = writer_compile_time_args,
                .defines = {{"SKIP_MCAST", "1"}},
                .named_compile_args = {
                    {"cb_in1", tt::CBIndex::c_1},
                    {"cb_bias", tt::CBIndex::c_3},
                    {"cb_out", tt::CBIndex::c_4},
                    {"cb_sparsity", tt::CBIndex::c_7},
                    {"cb_in1_intermediate", tt::CBIndex::c_9},
                }});
    }

    // Compute kernel compile time args
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;
    uint32_t in1_num_subblocks = (out_block_w / out_subblock_w);
    uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks;
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;
    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    std::vector<uint32_t> compute_kernel_args = {
        in0_block_w,
        in0_num_subblocks,
        in0_block_num_tiles,
        in0_subblock_num_tiles,
        in1_num_subblocks,
        in1_block_num_tiles,
        in1_per_core_w,
        num_blocks,
        out_num_blocks_x,
        out_num_blocks_y,
        out_subblock_h,
        out_subblock_w,
        out_subblock_num_tiles,
        batchA * batchB,
        out_block_tiles,
        false,  // untilize_out
        false,  // get_batch_from_reader
        false,  // in0_transpose_tile
    };

    tt::tt_metal::KernelHandle compute_kernel_id;
    if (use_sparse_kernels) {
        // Sparse compute: skips cb_wait/matmul/cb_pop for inactive K-blocks.
        // Uses software spill/reload (mm_partials) instead of PACKER_L1_ACC.
        compute_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/matmul/device/tile_sparse/kernels/compute/compute_tile_sparse_matmul.cpp",
            all_cores_with_work,
            tt_metal::ComputeConfig{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .math_approx_mode = math_approx_mode,
                .compile_args = compute_kernel_args,
                .defines = mm_kernel_defines,
                .named_compile_args = {
                    {"cb_in0", tt::CBIndex::c_0},
                    {"cb_in1", tt::CBIndex::c_1},
                    {"cb_out", tt::CBIndex::c_4},
                    {"cb_intermed0", tt::CBIndex::c_5},
                }});
    } else {
        compute_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp",
            all_cores_with_work,
            tt_metal::ComputeConfig{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .math_approx_mode = math_approx_mode,
                .compile_args = compute_kernel_args,
                .defines = mm_kernel_defines,
                .named_compile_args = {
                    {"cb_in0", tt::CBIndex::c_0},
                    {"cb_in1", tt::CBIndex::c_1},
                    {"cb_bias", tt::CBIndex::c_3},
                    {"cb_out", tt::CBIndex::c_4},
                    {"cb_intermed0", tt::CBIndex::c_5},
                    {"cb_in0_intermediate", tt::CBIndex::c_8},
                    {"cb_in1_intermediate", tt::CBIndex::c_9},
                    {"cb_in0_transposed", tt::CBIndex::c_10},
                }});
    }
    // Create circular buffers
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    tt_metal::CircularBufferConfig src0_cb_config =
        tt_metal::CircularBufferConfig(in0_CB_size, {{src0_cb_index, in0_data_format}})
            .set_page_size(src0_cb_index, in0_single_tile_size)
            .set_tile_dims(src0_cb_index, in0_tile);
    tt_metal::CreateCircularBuffer(program, all_cores, src0_cb_config);

    uint32_t src1_cb_index = tt::CBIndex::c_1;
    tt_metal::CircularBufferConfig src1_cb_config =
        tt_metal::CircularBufferConfig(in1_CB_size, {{src1_cb_index, in1_data_format}})
            .set_page_size(src1_cb_index, in1_single_tile_size)
            .set_tile_dims(src1_cb_index, in1_tile);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, src1_cb_config);

    uint32_t output_cb_index = tt::CBIndex::c_4;
    uint32_t interm0_cb_index = tt::CBIndex::c_5;

    tt_metal::CircularBufferConfig output_cb_config =
        tt_metal::CircularBufferConfig(0, {{output_cb_index, output_data_format}});
    tt_metal::CircularBufferConfig interm0_cb_config =
        tt_metal::CircularBufferConfig(0, {{interm0_cb_index, interm0_data_format}});

    if (interm0_data_format != output_data_format) {
        output_cb_config = tt_metal::CircularBufferConfig(out_CB_size, {{output_cb_index, output_data_format}})
                               .set_page_size(output_cb_index, output_single_tile_size)
                               .set_tile_dims(output_cb_index, output_tile);
        interm0_cb_config = tt_metal::CircularBufferConfig(interm0_CB_size, {{interm0_cb_index, interm0_data_format}})
                                .set_page_size(interm0_cb_index, interm0_single_tile_size)
                                .set_tile_dims(interm0_cb_index, output_tile);
        tt_metal::CreateCircularBuffer(program, CoreRangeSet({all_cores}), interm0_cb_config);
    } else {
        std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
            {output_cb_index, output_data_format}, {interm0_cb_index, interm0_data_format}};
        output_cb_config = tt_metal::CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
                               .set_page_size(output_cb_index, output_single_tile_size)
                               .set_page_size(interm0_cb_index, interm0_single_tile_size)
                               .set_tile_dims(output_cb_index, output_tile)
                               .set_tile_dims(interm0_cb_index, output_tile);
    }

    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);

    // Set runtime arguments
    const auto& cores = corerange_to_cores(all_cores, std::nullopt, row_major);
    CoreCoord start_core_noc = top_left_core_physical;
    CoreCoord end_core_noc = bottom_right_core_physical;
    if (in0_noc == tt::tt_metal::NOC::NOC_1) {
        std::swap(start_core_noc, end_core_noc);
    }

    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& core = cores[i];
        uint32_t output_idx_x = i % num_blocks_x;
        uint32_t output_idx_y = i / num_blocks_x;

        uint32_t sparsity_addr = (sparsity_mask_buffer != nullptr) ? sparsity_mask_buffer->address() : 0;

        if (core == start_core) {
            // Sender: reads in0 from DRAM and multicasts to receivers
            std::vector<uint32_t> reader_runtime_args;
            if (use_sparse_kernels) {
                reader_runtime_args = {
                    (std::uint32_t)in0_buffer->address(),
                    (std::uint32_t)Kt * per_core_M * output_idx_y,
                    (std::uint32_t)start_core_noc.x,
                    (std::uint32_t)start_core_noc.y,
                    (std::uint32_t)end_core_noc.x,
                    (std::uint32_t)end_core_noc.y,
                    (std::uint32_t)global_k_active_mask,  // sparse: K-block bitmask
                };
            } else {
                reader_runtime_args = {
                    (std::uint32_t)in0_buffer->address(),
                    (std::uint32_t)Kt * per_core_M * output_idx_y,
                    (std::uint32_t)start_core_noc.x,
                    (std::uint32_t)start_core_noc.y,
                    (std::uint32_t)end_core_noc.x,
                    (std::uint32_t)end_core_noc.y,
                    (std::uint32_t)out_block_h,
                    sparsity_addr,
                };
            }
            tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
        } else if (receiver_kernel_id != 0) {
            // Receiver: waits for mcast from sender
            std::vector<uint32_t> receiver_runtime_args = {
                (std::uint32_t)top_left_core_physical.x,  // sender_noc_x
                (std::uint32_t)top_left_core_physical.y,  // sender_noc_y
            };
            if (use_sparse_kernels) {
                receiver_runtime_args.push_back((std::uint32_t)global_k_active_mask);  // k_active_mask
            }
            tt_metal::SetRuntimeArgs(program, receiver_kernel_id, core, receiver_runtime_args);
        }

        // Compute runtime args: sparse kernel needs the K-active bitmask
        if (use_sparse_kernels && i < num_cores_with_work) {
            tt_metal::SetRuntimeArgs(program, compute_kernel_id, core, {global_k_active_mask});
        }

        // Writer runtime args
        if (i < num_cores_with_work) {
            std::vector<uint32_t> writer_runtime_args;
            if (use_sparse_kernels) {
                writer_runtime_args = {
                    (std::uint32_t)in1_buffer->address(),
                    (std::uint32_t)per_core_N * output_idx_x,
                    (std::uint32_t)0,  // mcast placeholders (no in1 mcast)
                    (std::uint32_t)0,
                    (std::uint32_t)0,
                    (std::uint32_t)0,
                    (std::uint32_t)global_k_active_mask,  // K-block bitmask (global: matches sender/compute)
                    (std::uint32_t)out_buffer->address(),
                    (std::uint32_t)(output_idx_x * per_core_N) + (output_idx_y * per_core_M * Nt),
                    (std::uint32_t)(out_block_h / out_subblock_h),  // out_sh_h
                    (std::uint32_t)out_subblock_h,                  // out_lsh_h
                    (std::uint32_t)0,                               // pad_h
                    (std::uint32_t)(out_block_w / out_subblock_w),  // out_sh_w
                    (std::uint32_t)(out_block_w / out_subblock_w),  // out_lsh_w
                    (std::uint32_t)out_subblock_w,                  // out_lsw
                    (std::uint32_t)0,                               // pad_sb
                    (std::uint32_t)0,                               // pad_w
                };
            } else {
                writer_runtime_args = {
                    (std::uint32_t)in1_buffer->address(),
                    (std::uint32_t)per_core_N * output_idx_x,
                    (std::uint32_t)0,  // in1_mcast placeholders
                    (std::uint32_t)0,
                    (std::uint32_t)0,
                    (std::uint32_t)0,
                    sparsity_addr,  // Tile indices buffer address (0 if dense)
                    (std::uint32_t)out_buffer->address(),
                    ((std::uint32_t)output_idx_x * per_core_N) + (output_idx_y * per_core_M * Nt),
                    // padding args
                    (std::uint32_t)out_block_w,
                    (std::uint32_t)out_block_h / out_subblock_h,
                    (std::uint32_t)out_subblock_h,
                    (std::uint32_t)0,
                    (std::uint32_t)out_block_w / out_subblock_w,
                    (std::uint32_t)out_block_w / out_subblock_w,
                    (std::uint32_t)out_subblock_w,
                    (std::uint32_t)0,
                    (std::uint32_t)0,
                    (std::uint32_t)0,
                    (std::uint32_t)0,
                    (std::uint32_t)out_num_blocks_x,
                    (std::uint32_t)0,
                    (std::uint32_t)0,
                    (std::uint32_t)0,
                    (std::uint32_t)0,
                    (std::uint32_t)0,
                };
            }
            tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
        }
    }

    std::vector<tt::tt_metal::KernelHandle> kernel_handles = {reader_kernel_id, writer_kernel_id};
    if (receiver_kernel_id != 0) {
        kernel_handles.push_back(receiver_kernel_id);
    }
    auto shared_vars = TileSparseMatmulProgramFactory::shared_variables_t{
        kernel_handles, {cb_src1, cb_output}, start_core, cores, num_cores_with_work, nnz_tiles, tile_indices_buffer};

    return {std::move(program), std::move(shared_vars)};
}

void TileSparseMatmulProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const TileSparseMatmulParams& /*operation_attributes*/,
    const TileSparseMatmulInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    auto& program = cached_program.program;
    auto& shared_vars = cached_program.shared_variables;

    auto* src_buffer_a = tensor_args.input_tensors.at(0).buffer();
    auto* src_buffer_b = tensor_args.input_tensors.at(1).buffer();
    auto* dst_buffer = tensor_return_value.at(0).buffer();

    // Update reader runtime args
    auto& reader_runtime_args = GetRuntimeArgs(program, shared_vars.kernels.at(0), shared_vars.start_core);
    reader_runtime_args[0] = src_buffer_a->address();

    auto& writer_runtime_args_by_core = GetRuntimeArgs(program, shared_vars.kernels.at(1));

    for (uint32_t i = 0; i < shared_vars.num_cores_with_work; ++i) {
        const auto& core = shared_vars.cores[i];
        auto& writer_runtime_args = writer_runtime_args_by_core[core.x][core.y];
        writer_runtime_args[0] = src_buffer_b->address();
        writer_runtime_args[7] = dst_buffer->address();
    }
}

// Mesh workload implementation
TileSparseMatmulMeshWorkloadFactory::cached_mesh_workload_t TileSparseMatmulMeshWorkloadFactory::create_mesh_workload(
    const TileSparseMatmulParams& attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const TileSparseMatmulInputs& tensor_args,
    std::vector<Tensor>& output) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    for (const auto& mesh_coord_range : tensor_coords.ranges()) {
        for (const auto& mesh_coord : mesh_coord_range) {
            const ttnn::MeshCoordinateRange single_coord_range{mesh_coord, mesh_coord};
            auto single_device_program = TileSparseMatmulProgramFactory::create(attributes, tensor_args, output);
            shared_variables[single_coord_range] = single_device_program.shared_variables;
            workload.add_program(single_coord_range, std::move(single_device_program.program));
        }
    }
    return {std::move(workload), std::move(shared_variables)};
}

void TileSparseMatmulMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const TileSparseMatmulParams& attributes,
    const TileSparseMatmulInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    for (auto& [mesh_coord_range, program] : cached_workload.workload.get_programs()) {
        auto cached_program_proxy = TileSparseMatmulProgramFactory::cached_program_t::proxy(
            program, cached_workload.shared_variables.at(mesh_coord_range));
        TileSparseMatmulProgramFactory::override_runtime_arguments(
            cached_program_proxy, attributes, tensor_args, tensor_return_value);
    }
}

}  // namespace ttnn::prim
