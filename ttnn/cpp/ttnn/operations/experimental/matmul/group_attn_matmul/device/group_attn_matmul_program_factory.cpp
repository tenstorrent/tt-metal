// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "group_attn_matmul_device_operation.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/work_split.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::matmul {

using namespace tt::constants;
using namespace tt;

operation::ProgramWithCallbacks multi_core_group_attn_matmul(const Tensor &a, const Tensor &b, Tensor& output, std::optional<const uint32_t> num_tokens, std::optional<const bool> transpose_hw, const uint32_t out_subblock_w, CoreCoord compute_with_storage_grid_size, const bool row_major, ttnn::DeviceComputeKernelConfig compute_kernel_config) {

    tt::tt_metal::Program program{};

    const auto& ashape = a.get_legacy_shape(), bshape = b.get_legacy_shape();

    // This should allocate a DRAM buffer on the device
    tt::tt_metal::Device *device = a.device();

    MathFidelity math_fidelity;
    bool math_approx_mode;
    bool fp32_dest_acc_en;
    bool packer_l1_acc;

    std::visit([&](auto&& compute_kernel_config) {
        using T = std::decay_t<decltype(compute_kernel_config)>;
        if constexpr (std::is_same_v<T, GrayskullComputeKernelConfig>) {
            TT_ASSERT(device->arch() == ARCH::GRAYSKULL, "kernel config is not for graykull");
            math_fidelity = compute_kernel_config.math_fidelity;
            math_approx_mode = compute_kernel_config.math_approx_mode;
            fp32_dest_acc_en = false;
            packer_l1_acc = false;
        } else if constexpr (std::is_same_v<T, WormholeComputeKernelConfig>) {
            TT_ASSERT(ttnn::device::is_wormhole_or_blackhole(device->arch()), "kernel config is not for wormhole_b0 or blackhole");
            math_fidelity = compute_kernel_config.math_fidelity;
            math_approx_mode = compute_kernel_config.math_approx_mode;
            fp32_dest_acc_en = compute_kernel_config.fp32_dest_acc_en;
            packer_l1_acc = compute_kernel_config.packer_l1_acc;
        } else {
            TT_FATAL("arch not supported");
        }

    }, compute_kernel_config);

    tt::DataFormat in0_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    tt::DataFormat in1_data_format = tt::tt_metal::datatype_to_dataformat_converter(b.get_dtype());
    tt::DataFormat interm_data_format = fp32_dest_acc_en and in0_data_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t in0_single_tile_size = tt::tt_metal::detail::TileSize(in0_data_format);
    uint32_t in1_single_tile_size = tt::tt_metal::detail::TileSize(in1_data_format);
    uint32_t interm_single_tile_size = tt::tt_metal::detail::TileSize(interm_data_format);
    uint32_t output_single_tile_size = tt::tt_metal::detail::TileSize(output_data_format);

    if (in0_data_format == tt::DataFormat::Float32 or in1_data_format == tt::DataFormat::Float32 or output_data_format == tt::DataFormat::Float32) {
        TT_ASSERT(fp32_dest_acc_en == true, "when inputs/output are in fp32 format, fp32_dest_acc_en must be set");
    }

    tt::tt_metal::Buffer *src0_buffer = a.buffer();
    tt::tt_metal::Buffer *src1_buffer = b.buffer();
    tt::tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    // Load kernels on all device cores, because we use cached program for input shapes with changing shapes
    CoreCoord device_compute_with_storage_grid = device->compute_with_storage_grid_size();
    auto all_device_cores = CoreRange({0, 0}, {device_compute_with_storage_grid.x - 1, device_compute_with_storage_grid.y - 1});

    // See set_runtime_args for how input shapes are used; these are the variables needed for setting up kernels and CBs
    const bool transpose_hw_bool = transpose_hw.value_or(false);
    const uint32_t num_tokens_val = num_tokens.value_or(0); // should not be nullopt if transpose_hw=true
    uint32_t KV_HEADS = bshape[1]; // bshape[0] is user batch
    uint32_t Mt = ashape[2]/TILE_HEIGHT;
    uint32_t Kt = ashape[3]/TILE_WIDTH;
    uint32_t Nt = transpose_hw_bool ? num_tokens_val/TILE_HEIGHT : bshape[3]/TILE_WIDTH;
    uint32_t MtNt = Mt * Nt;

    // Matmul blocking parameters; these determine how large CBs are; actual value doesn't matter here since CBs are resized in callback...
    const uint32_t out_block_w = out_subblock_w;
    const uint32_t in0_block_w = Kt;
    const uint32_t in1_block_num_tiles = KV_HEADS * in0_block_w * out_subblock_w;

    // out_subblock_w and other hardcoded params are known at compile time
    constexpr uint32_t out_subblock_h = 1; // TODO: Only support per_core_Mt = 1 (mcasting assumes batch=32 for now anyways)
    constexpr uint32_t in1_num_subblocks = 1; // out_block_w / out_subblock_w
    const uint32_t out_subblock_num_tiles = out_subblock_h * out_block_w;
    const uint32_t intermediate_num_tiles = out_subblock_num_tiles;
    const uint32_t in1_per_core_w = in1_num_subblocks * out_block_w;
    const uint32_t in1_block_w_tile_bytes = out_subblock_w * in1_single_tile_size;
    uint32_t ONE_ROW_BFLOAT16_BYTES = fp32_dest_acc_en and in0_data_format == tt::DataFormat::Float32 ? 128 : 64;
    const uint32_t bfloat16_row_bytes = ONE_ROW_BFLOAT16_BYTES * out_block_w; // TODO: Generalize

    log_debug("in0_block_w: {}", in0_block_w);
    log_debug("out_subblock_h: {}", out_subblock_h);
    log_debug("out_subblock_w: {}", out_subblock_w);
    log_debug("math_fidelity: {}", math_fidelity);
    log_debug("math_approx_mode: {}", math_approx_mode);
    log_debug("fp32_dest_acc_en: {}", fp32_dest_acc_en);
    log_debug("packer_l1_acc: {}", packer_l1_acc);
    log_debug("in0_data_format: {}", in0_data_format);
    log_debug("in1_data_format: {}", in1_data_format);
    log_debug("interm_data_format: {}", interm_data_format);
    log_debug("output_data_format: {}", output_data_format);


    // Mcast args
    auto in1_mcast_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_device_cores, INVALID);
    auto in1_mcast_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_device_cores, INVALID);

    // Only first 32 of cores mcast KV heads to match num_rows_in_one_tile in reader kernel, so these coordinates are static if we cache on compute_with_storage_grid_size
    // TODO: If this is not the case, then we should set reader_runtime_args to max possible size and update sender noc coordinates based on input
    CoreCoord mcast_sender_grid = ((CoreRangeSet) num_cores_to_corerange_set(TILE_HEIGHT, compute_with_storage_grid_size, row_major)).bounding_box().grid_size();
    std::vector<uint32_t> in1_mcast_sender_noc_x(mcast_sender_grid.x);
    std::vector<uint32_t> in1_mcast_sender_noc_y(mcast_sender_grid.y);
    for(uint32_t core_idx_x = 0; core_idx_x < mcast_sender_grid.x; ++core_idx_x) {
        in1_mcast_sender_noc_x[core_idx_x] = device->worker_core_from_logical_core({core_idx_x, 0}).x;
    }
    for(uint32_t core_idx_y = 0; core_idx_y < mcast_sender_grid.y; ++core_idx_y) {
        in1_mcast_sender_noc_y[core_idx_y] = device->worker_core_from_logical_core({0, core_idx_y}).y;
    }

    // Set up CBs
    const bool in0_is_sharded = a.is_sharded();
    const bool in1_is_sharded = b.is_sharded();
    const bool output_is_sharded = output.is_sharded();

    // CB for in0 (ie. q_heads)
    uint32_t src0_cb_index = tt::CB::c_in0;
    CBHandle cb_src0;
    if (in0_is_sharded) {
        uint32_t cb0_num_input_tiles = a.shard_spec().value().numel() / TILE_HW; // Should be full MtKt and C should be 1
        tt::tt_metal::CircularBufferConfig src0_cb_config = tt::tt_metal::CircularBufferConfig(cb0_num_input_tiles * in0_single_tile_size, {{src0_cb_index, in0_data_format}})
		    .set_page_size(src0_cb_index, in0_single_tile_size).set_globally_allocated_address(*src0_buffer);
        cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_device_cores, src0_cb_config);
    } else {
        uint32_t cb0_num_input_tiles = in0_block_w; // TODO: Generalize; double buffer and add blocking along inner dim if we have Mt > 1
        tt::tt_metal::CircularBufferConfig src0_cb_config = tt::tt_metal::CircularBufferConfig(cb0_num_input_tiles * in0_single_tile_size, {{src0_cb_index, in0_data_format}})
		    .set_page_size(src0_cb_index, in0_single_tile_size);
        cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_device_cores, src0_cb_config);
    }

    // CB for interleaved/sharded KV heads for mcasting; mcasts to same CB
    // Then, push all KV_HEADS to compute and compute chooses which head to use for matmul
    uint32_t src1_cb_index = tt::CB::c_in1;
    uint32_t cb1_num_input_tiles = 2 * in1_block_num_tiles;
    tt::tt_metal::CircularBufferConfig cb_src1_config = tt::tt_metal::CircularBufferConfig(cb1_num_input_tiles * in1_single_tile_size, {{src1_cb_index, in1_data_format}})
		.set_page_size(src1_cb_index, in1_single_tile_size);
    auto cb_src1 = tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_src1_config);

    // CB for sharded KV heads
    CBHandle cb_src2 = 0;  // unused if KV heads is interleaved
    if (in1_is_sharded) {
        uint32_t src2_cb_index = tt::CB::c_in2;
        uint32_t cb2_num_input_tiles = b.shard_spec().value().numel() / TILE_HW; // Should be full CKtNt and batch must be 32
        tt::tt_metal::CircularBufferConfig cb_src2_config = tt::tt_metal::CircularBufferConfig(cb2_num_input_tiles * in1_single_tile_size, {{src2_cb_index, in1_data_format}})
		    .set_page_size(src2_cb_index, in1_single_tile_size).set_globally_allocated_address(*src1_buffer);
        cb_src2 = tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_src2_config);
    }

    // Intermediate CBs for handling untilizing, copying rows, and tilizing to output CB
    uint32_t interm_cb_num_tiles = 2 * intermediate_num_tiles; // TODO: Generalize; double buffering should help when we are not reader bound
    uint32_t cb_intermed0_index = tt::CB::c_intermed0;
    tt::tt_metal::CircularBufferConfig cb_interm0_config = tt::tt_metal::CircularBufferConfig(interm_cb_num_tiles * interm_single_tile_size, {{cb_intermed0_index, interm_data_format}})
		.set_page_size(cb_intermed0_index, interm_single_tile_size);
    auto cb_interm0 = tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_interm0_config);

    uint32_t cb_intermed1_index = tt::CB::c_intermed1;
    tt::tt_metal::CircularBufferConfig cb_interm1_config = tt::tt_metal::CircularBufferConfig(MtNt * interm_single_tile_size, {{cb_intermed1_index, interm_data_format}})
		.set_page_size(cb_intermed1_index, interm_single_tile_size);
    auto cb_interm1 = tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_interm1_config);

    // CB for output (if sharded, full num tiles per core)
    uint32_t output_cb_index = tt::CB::c_out0; // output operands start at index 16
    CBHandle cb_output;
    if (output_is_sharded) {
        uint32_t num_output_tiles = output.shard_spec().value().numel() / TILE_HW; // Should be full MtNt and C should be 1
        tt::tt_metal::CircularBufferConfig cb_output_config = tt::tt_metal::CircularBufferConfig(num_output_tiles * output_single_tile_size, {{output_cb_index, output_data_format}})
		    .set_page_size(output_cb_index, output_single_tile_size).set_globally_allocated_address(*dst_buffer);
        cb_output = tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_output_config);
    } else {
        uint32_t num_output_tiles = MtNt; // TODO: Should be MtNt if Mt > 1? Or, produce one Nt at a time and double buffer?
        tt::tt_metal::CircularBufferConfig cb_output_config = tt::tt_metal::CircularBufferConfig(num_output_tiles * output_single_tile_size, {{output_cb_index, output_data_format}})
		    .set_page_size(output_cb_index, output_single_tile_size);
        cb_output = tt::tt_metal::CreateCircularBuffer(program, all_device_cores, cb_output_config);
    }

    const uint32_t src0_is_dram = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    const uint32_t src1_is_dram = src1_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        (uint32_t) src0_is_dram,
        (uint32_t) src1_is_dram,
        (uint32_t) transpose_hw_bool,
        (uint32_t) row_major,
        out_subblock_w,
    };

    const uint32_t dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (uint32_t) src0_is_dram,
        (uint32_t) dst_is_dram,
        (uint32_t) output_cb_index,
        out_subblock_w,
        intermediate_num_tiles,
    };

    std::map<string, string> reader_kernel_defines;
    std::map<string, string> writer_kernel_defines;
    if (in0_is_sharded) {
        writer_kernel_defines["IN0_SHARDED"] = "1";
    }
    if (in1_is_sharded) {
        reader_kernel_defines["IN1_SHARDED"] = "1";
    }
    if (output_is_sharded) {
        writer_kernel_defines["OUT_SHARDED"] = "1";
    }

    tt::tt_metal::NOC reader_noc = tt::tt_metal::detail::GetPreferredNOCForDRAMRead(device->arch()); // Default is NOC_1
    const bool reader_noc_is_NOC_0 = reader_noc == tt::tt_metal::NOC::NOC_0;
    tt::tt_metal::NOC writer_noc = reader_noc_is_NOC_0 ? tt::tt_metal::NOC::NOC_1 : tt::tt_metal::NOC::NOC_0;
    auto reader_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/device/kernels/dataflow/reader_mcast_transformer_group_attn_matmul.cpp",
        all_device_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = reader_noc,
            .compile_args = reader_compile_time_args,
            .defines = reader_kernel_defines,
        }
    );

    auto writer_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/device/kernels/dataflow/writer_transformer_group_attn_matmul.cpp",
        all_device_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = writer_noc,
            .compile_args = writer_compile_time_args,
            .defines = writer_kernel_defines,
        }
    );

    vector<uint32_t> compute_args = {
        (uint32_t) transpose_hw_bool, // transpose_hw for matmul_init
        out_subblock_w,
        out_subblock_num_tiles,
        intermediate_num_tiles,
    }; // bmm compute kernel the B, Mt, Nt are just 3 for loops that technically act as 1 large loop, so only set Nt for simplicity

    auto compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/device/kernels/compute/transformer_group_attn_matmul.cpp",
        all_device_cores,
        tt::tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = compute_args}
    );

    auto set_runtime_args = [
            num_tokens,
            transpose_hw,
            row_major,
            compute_with_storage_grid_size,
            device_compute_with_storage_grid,
            reader_id,
            writer_id,
            compute_kernel_id,
            cb_src0,
            cb_src1,
            cb_src2,
            cb_interm1,
            cb_output,
            in0_single_tile_size,
            in1_single_tile_size,
            interm_single_tile_size,
            output_single_tile_size,
            in0_is_sharded,
            in1_is_sharded,
            output_is_sharded,
            reader_noc_is_NOC_0,
            in1_mcast_sender_semaphore_id,
            in1_mcast_receiver_semaphore_id,
            in1_mcast_sender_noc_x,
            in1_mcast_sender_noc_y,

            // Params determined by out_subblock_w
            out_subblock_w,
            out_subblock_h,
            in1_num_subblocks,
            out_block_w,
            in1_per_core_w,
            in1_block_w_tile_bytes,
            ONE_ROW_BFLOAT16_BYTES,
            bfloat16_row_bytes
        ]
    (
        Program& program,
        const Tensor& a,
        const Tensor& b,
        const Tensor& output
    ) {
        tt::tt_metal::Buffer *src0_buffer = a.buffer();
        tt::tt_metal::Buffer *src1_buffer = b.buffer();
        tt::tt_metal::Buffer *dst_buffer = output.buffer();

        const auto& ashape = a.get_legacy_shape(), bshape = b.get_legacy_shape();

        tt::tt_metal::Device *device = a.device();

        // A block of work is one MtNt
        uint32_t Q_HEADS = ashape[1]; // ashape[0] is q_len (always 1) and ashape[1] is Q num_heads; only parallelize on this
        // Must always have at least 32 cores active since there are always 32 mcast cores for KV_HEADS
        // TODO: Currently, we always mcast to at least 32 cores even when Q_HEADS < 32; we can optimize if we pass in proper mcast receiver grid based on Q_HEADS
        // TODO: If batch > 32 (ie. 64), each core handles all batches; only supported for interleaved KV_heads
        // TODO: For sharded KV_heads, user batch must be 32 due to how we shard
        // TODO: To generalize to allow parallelizing/sharding across generic batch for KV_heads, we need to track sender cores across batch-number of rows instead of 32
        // TODO: Only support one block of work (ie. 1 Q head per core) because each core assumes only one KV_heads to use
        uint32_t num_active_cores = std::max(Q_HEADS, TILE_HEIGHT);
        auto [num_cores, all_cores, core_group_1, core_group_2, num_output_blocks_per_core_group_1, num_output_blocks_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, num_active_cores, row_major);
        // num_cores should be same as num_active_cores
        TT_FATAL(num_output_blocks_per_core_group_1 == 1 and num_output_blocks_per_core_group_2 == 0, "Group attention matmul only supports one q_heads per core. Increase compute grid size to at least have as many cores as q_heads!");

        // C = torch.matmul(A.transpose(0, 2) * B).transpose(0, 2)
        // MN = MK*KN
        // Note, in1 K may not be the same as in0 K. We will read up to in0 K from in1 K for matmul.
        const bool transpose_hw_bool = transpose_hw.value_or(false);
        const uint32_t num_tokens_val = num_tokens.value_or(0); // should not be nullopt if transpose_hw=true

        uint32_t KV_HEADS = bshape[1]; // bshape[0] is user batch
        uint32_t Mt = ashape[2]/TILE_HEIGHT;
        uint32_t Kt = ashape[3]/TILE_WIDTH;
        // For transpose_hw=true, in1_Kt is same as in0_Kt but on bshape[3]
        // For transpose_hw=false, in1_Kt is on bshape[2] but represents the max cache length to read from (ie. may not equal in0_Kt)
        uint32_t in1_Kt = transpose_hw_bool ? Kt : bshape[2]/TILE_HEIGHT;
        uint32_t Nt = transpose_hw_bool ? num_tokens_val/TILE_HEIGHT : bshape[3]/TILE_WIDTH;
        uint32_t MtKt = Mt * Kt;
        uint32_t MtNt = Mt * Nt;
        // For transpose_hw=true, in1_Kt is max cache length
        // For transpose_hw=false, bshape[2] is max cache length
        uint32_t in1_KtNt = transpose_hw_bool ? bshape[2]/TILE_HEIGHT * in1_Kt : in1_Kt * Nt;
        uint32_t in1_CKtNt = KV_HEADS * in1_KtNt;

        // Matmul params that is runtime dependent
        uint32_t in0_block_w = Kt;
        uint32_t in0_subblock_num_tiles = in0_block_w * out_subblock_h;
        uint32_t in0_block_num_tiles = in0_subblock_num_tiles;
        uint32_t in1_block_num_tiles_per_kv_heads = in0_block_w * out_subblock_w;
        uint32_t in1_block_num_tiles = KV_HEADS * in1_block_num_tiles_per_kv_heads;
        uint32_t in1_num_blocks = (Nt - 1) / out_block_w + 1; // Rounds up to include nearest out_block_w; "padding" is handled internally

        uint32_t Nt_bytes = Nt * in1_single_tile_size;
        uint32_t out_last_subblock_w = Nt % out_block_w == 0 ? out_block_w : Nt % out_block_w;
        uint32_t in1_last_block_w_tile_read_bytes = out_last_subblock_w * in1_single_tile_size;
        uint32_t in1_last_block_addr_skip = (out_subblock_w - out_last_subblock_w) * in1_single_tile_size;

        uint32_t bfloat16_Nt_bytes = ONE_ROW_BFLOAT16_BYTES * Nt;
        uint32_t bfloat16_last_row_bytes_read = ONE_ROW_BFLOAT16_BYTES * out_last_subblock_w;

        // Mcast receiver args
        // NOTE: If Q_HEADS < 32, have all cores in mcast grid participate in semaphore syncing because we mcast KV_HEADS from/to the same CB
        // Otherwise, data corruption if sender is in mcast grid and it starts populating its mcast CB as other senders are sending
        CoreRangeSet mcast_receiver_cores = num_cores_to_corerange_set(Q_HEADS, compute_with_storage_grid_size, row_major);
        CoreRange mcast_receiver_cores_bounding_box = mcast_receiver_cores.bounding_box();
        uint32_t mcast_num_dests = mcast_receiver_cores.num_cores(); // same as num_active_cores if Q_HEADS >= 32; also, always same as Q_HEADS
        uint32_t mcast_num_cores = mcast_receiver_cores_bounding_box.size();
        CoreCoord top_left_core = mcast_receiver_cores_bounding_box.start_coord;
        CoreCoord bottom_right_core = mcast_receiver_cores_bounding_box.end_coord;
        CoreCoord top_left_core_physical = device->worker_core_from_logical_core(top_left_core);
        CoreCoord bottom_right_core_physical = device->worker_core_from_logical_core(bottom_right_core);

        // Default reader runtime args
        std::vector<uint32_t> reader_runtime_args = {
            0, // 0: has_work_for_mcast_kv_heads
            0, // 1: has_work_for_q_heads
            src1_buffer->address(),
            Mt,
            Nt,
            KV_HEADS,
            in1_CKtNt,
            in1_CKtNt * TILE_HEIGHT, // in1 stride; skips 32 * KtNt in bshape[0] for one block of MtNt
            0, // 8: blocks of work
            0, // in1_start_id; always start at 0 for each block of work and let kernels handle id tracking; for sharded, this isn't used

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

            // mcast args
            (uint32_t) (reader_noc_is_NOC_0 ? top_left_core_physical.x : bottom_right_core_physical.x), // in1_mcast_dest_noc_start_x
            (uint32_t) (reader_noc_is_NOC_0 ? top_left_core_physical.y : bottom_right_core_physical.y), // in1_mcast_dest_noc_start_y
            (uint32_t) (reader_noc_is_NOC_0 ? bottom_right_core_physical.x : top_left_core_physical.x), // in1_mcast_dest_noc_end_x
            (uint32_t) (reader_noc_is_NOC_0 ? bottom_right_core_physical.y : top_left_core_physical.y), // in1_mcast_dest_noc_end_y
            0, // 24: in1_mcast_num_dests
            0, // 25: in1_mcast_num_cores
            mcast_num_cores, // mcast grid size; in1_mcast_num_cores may change depending on if sender is part of the receiver grid or not
            in1_mcast_sender_semaphore_id,
            in1_mcast_receiver_semaphore_id,
            in1_block_num_tiles * in1_single_tile_size, // in1_mcast_sender_size_bytes
            0, // 30: in1_mcast_sender_id
            (uint32_t) in1_mcast_sender_noc_x.size(), // in1_mcast_sender_num_x
            (uint32_t) in1_mcast_sender_noc_y.size(), // in1_mcast_sender_num_y
        };
        // TODO: Length of these variables should be static in length since we hard-code 32 mcast sender cores and cache on compute_with_storage_grid_size
        reader_runtime_args.insert(reader_runtime_args.end(), in1_mcast_sender_noc_x.begin(), in1_mcast_sender_noc_x.end());
        reader_runtime_args.insert(reader_runtime_args.end(), in1_mcast_sender_noc_y.begin(), in1_mcast_sender_noc_y.end());

        // Default writer runtime args
        std::vector<uint32_t> writer_runtime_args = {
            0, // 0: has_work_for_q_heads
            src0_buffer->address(),
            dst_buffer->address(),
            Mt,
            Kt,
            Nt,
            MtKt,
            0, // 7: blocks of work
            0, // 8: in0_start_id
            0, // 9: out_start_id

            in0_block_w,
            in1_num_subblocks,
            in1_num_blocks,
            MtNt, // out_num_tiles

            bfloat16_row_bytes,
            bfloat16_Nt_bytes,
            bfloat16_last_row_bytes_read,
        };

        // Default compute runtime args
        std::vector<uint32_t> compute_runtime_args = {
            0, // 0: has_work_for_q_heads
            0, // 1: batch,
            Mt,
            0, // 3: num_kv_heads_skip
            0, // 4: num_kv_heads_remaining

            in0_block_w,
            out_subblock_h,
            in1_num_subblocks,
            in1_num_blocks,
            in0_block_num_tiles,
            in1_block_num_tiles,
            MtNt, // out_num_tiles

            in0_subblock_num_tiles,
            in1_per_core_w,
        };

        CoreRange all_cores_bounding_box = all_cores.bounding_box();
        std::vector<CoreCoord> cores = grid_to_cores_with_noop(
            all_cores_bounding_box.end_coord.x,
            all_cores_bounding_box.end_coord.y,
            device_compute_with_storage_grid.x,
            device_compute_with_storage_grid.y,
            row_major
        );
        uint32_t g1_numcores = core_group_1.num_cores();
        uint32_t g2_numcores = core_group_2.num_cores();

        std::vector<std::vector<uint32_t>> all_reader_runtime_args = { cores.size(), reader_runtime_args };
        std::vector<std::vector<uint32_t>> all_writer_runtime_args = { cores.size(), writer_runtime_args };
        std::vector<std::vector<uint32_t>> all_compute_runtime_args = { cores.size(), compute_runtime_args };

        // Set runtime args
        uint32_t num_output_blocks_per_core;
        for (uint32_t i = 0, num_blocks_written = 0; i < num_cores; i++){
            const CoreCoord &core = cores.at(i);

            if (i < g1_numcores) {
                num_output_blocks_per_core = num_output_blocks_per_core_group_1;
            } else {
                num_output_blocks_per_core = num_output_blocks_per_core_group_2;
            }

            uint32_t kv_heads_id = i / (Q_HEADS / KV_HEADS);
            // Runtime method of turning off kernels/code blocks
            // Needed because some cores only have partial readers for reading/mcasting kv_heads
            uint32_t has_work_for_q_heads = i < Q_HEADS; // compute and writer only does work if this is true; reader will only receive kv_heads if this is true
            uint32_t has_work_for_mcast_kv_heads = i < num_active_cores; // reader only does any work if this is true
            uint32_t mcast_num_cores_for_core = mcast_num_cores - (uint32_t) (i < mcast_num_cores); // if sender is not part of mcast grid, send to full grid

            // Update core dependent runtime args
            all_reader_runtime_args[i][0] = has_work_for_mcast_kv_heads;
            all_reader_runtime_args[i][1] = has_work_for_q_heads;
            all_reader_runtime_args[i][8] = num_output_blocks_per_core;
            // If Q_HEADS < 32, have all cores participate in receiving; std::min is needed for cases where mcast grid is > num_active_cores and non-active cores are turned off and don't receive
            all_reader_runtime_args[i][24] = Q_HEADS < TILE_HEIGHT ? std::min(mcast_num_cores_for_core, num_active_cores - 1) : mcast_num_dests - 1;
            all_reader_runtime_args[i][25] = mcast_num_cores_for_core;
            all_reader_runtime_args[i][30] = i;

            all_writer_runtime_args[i][0] = has_work_for_q_heads;
            all_writer_runtime_args[i][7] = num_output_blocks_per_core;
            all_writer_runtime_args[i][8] = num_blocks_written * MtKt;
            all_writer_runtime_args[i][9] = num_blocks_written * MtNt;

            all_compute_runtime_args[i][0] = has_work_for_q_heads;
            all_compute_runtime_args[i][1] = num_output_blocks_per_core;
            all_compute_runtime_args[i][3] = kv_heads_id * in1_block_num_tiles_per_kv_heads;
            all_compute_runtime_args[i][4] = (KV_HEADS - kv_heads_id) * in1_block_num_tiles_per_kv_heads;

            num_blocks_written += num_output_blocks_per_core;
        }

        SetRuntimeArgs(program, reader_id, cores, all_reader_runtime_args);
        SetRuntimeArgs(program, writer_id, cores, all_writer_runtime_args);
        SetRuntimeArgs(program, compute_kernel_id, cores, all_compute_runtime_args);

        // Update dynamic CBs (which is most of them)
        if (in0_is_sharded) {
            uint32_t cb0_num_input_tiles = a.shard_spec().value().numel() / TILE_HW; // Should be full MtKt and C should be 1
            UpdateDynamicCircularBufferAddress(program, cb_src0, *src0_buffer);
            UpdateCircularBufferTotalSize(program, cb_src0, cb0_num_input_tiles * in0_single_tile_size);
        } else {
            uint32_t cb0_num_input_tiles = in0_block_w; // TODO: Generalize; double buffer and add blocking along ineer dim if we have Mt > 1
            UpdateCircularBufferTotalSize(program, cb_src0, cb0_num_input_tiles * in0_single_tile_size);
        }

        uint32_t cb1_num_input_tiles = 2 * in1_block_num_tiles;
        UpdateCircularBufferTotalSize(program, cb_src1, cb1_num_input_tiles * in1_single_tile_size);

        if (in1_is_sharded) {
            uint32_t cb2_num_input_tiles = b.shard_spec().value().numel() / TILE_HW; // Should be full CKtNt and batch must be 32
            UpdateDynamicCircularBufferAddress(program, cb_src2, *src1_buffer);
            UpdateCircularBufferTotalSize(program, cb_src2, cb2_num_input_tiles * in1_single_tile_size);
        }

        UpdateCircularBufferTotalSize(program, cb_interm1, MtNt * interm_single_tile_size);

        if (output_is_sharded) {
            uint32_t num_output_tiles = output.shard_spec().value().numel() / TILE_HW; // Should be full MtNt and C should be 1
            UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);
            UpdateCircularBufferTotalSize(program, cb_output, num_output_tiles * output_single_tile_size);
        } else {
            uint32_t num_output_tiles = MtNt; // TODO: Should be MtNt if Mt > 1? Or, produce one Nt at a time and double buffer?
            UpdateCircularBufferTotalSize(program, cb_output, num_output_tiles * output_single_tile_size);
        }
    };

    set_runtime_args(program, a, b, output);

    auto override_runtime_arguments_callback = [
            set_runtime_args
        ]
    (
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {
        const auto& output_tensor = output_tensors.at(0);

        set_runtime_args(program, input_tensors.at(0), input_tensors.at(1), output_tensor);
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // ttnn::operations::experimental::transformer
