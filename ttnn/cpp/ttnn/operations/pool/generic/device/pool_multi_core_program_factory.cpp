// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pool_op.hpp"
#include "tt-metalium/circular_buffer.hpp"
#include "tt-metalium/circular_buffer_types.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"  // for reduce_op_utils
#include <tt-metalium/math.hpp>

/**
 * Generic pool implementation that uses the new sliding window infrastructure.
 */

namespace ttnn::operations::pool {

Pool2D::MultiCore::cached_program_t pool2d_multi_core_sharded_with_halo_v2_impl_new(
    Program& program,
    const Tensor& input,
    const Tensor& reader_indices,
    Tensor& output,
    Pool2DType pool_type,
    uint32_t in_n,
    uint32_t in_h,
    uint32_t in_w,
    uint32_t out_h,
    uint32_t out_w,
    uint32_t kernel_size_h,
    uint32_t kernel_size_w,
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t pad_h,
    uint32_t pad_w,
    uint32_t ceil_pad_w,
    uint32_t dilation_h,
    uint32_t dilation_w,
    uint32_t num_shards_c,
    const MemoryConfig& out_mem_config,
    uint32_t nblocks) {
    TT_FATAL(pool_type == Pool2DType::MAX_POOL2D, "Currently only support max pool2d (#12151)");

    // This should allocate a DRAM buffer on the device
    IDevice* device = input.device();
    tt::tt_metal::Buffer* src_dram_buffer = input.buffer();
    auto reader_indices_buffer = reader_indices.mesh_buffer();
    tt::tt_metal::Buffer* dst_dram_buffer = output.buffer();

    const auto input_shape = input.get_padded_shape();
    const auto output_shape = output.get_padded_shape();

    tt::DataFormat in_df = datatype_to_dataformat_converter(input.get_dtype());
    tt::DataFormat out_df = datatype_to_dataformat_converter(output.get_dtype());
    uint32_t in_nbytes = datum_size(in_df);
    uint32_t out_nbytes = datum_size(out_df);

    uint32_t in_nbytes_c = input_shape[3] / num_shards_c * in_nbytes;     // row of input (channels)
    uint32_t out_nbytes_c = output_shape[3] / num_shards_c * out_nbytes;  // row of output (channels)

    tt::DataFormat indices_df =
        tt::DataFormat::RawUInt16;  // datatype_to_dataformat_converter(reader_indices.get_dtype());
    uint32_t indices_nbytes = datum_size(indices_df);

    uint32_t kernel_size_hw = kernel_size_w * kernel_size_h;  // number of valid rows, to read
    uint32_t kernel_size_hw_padded = tt::round_up(kernel_size_hw, tt::constants::TILE_HEIGHT);
    uint32_t in_ntiles_hw = (uint32_t)std::ceil((float)kernel_size_hw_padded / tt::constants::TILE_HEIGHT);
    uint32_t in_ntiles_c = (uint32_t)std::ceil((float)input_shape[3] / num_shards_c / tt::constants::TILE_WIDTH);
    uint32_t out_ntiles_c = (uint32_t)std::ceil((float)output_shape[3] / num_shards_c / tt::constants::TILE_WIDTH);

    uint32_t max_rows_for_reduction = 16;
    // TODO #14588: temporarily disabling 32 row reductions due to issues in large kernels
    /* uint32_t max_rows_for_reduction = tt::constants::TILE_HEIGHT;
    // For GRAYSKULL, make reduction for 16 rows at a time.
    if (device->arch() == tt::ARCH::GRAYSKULL)
        max_rows_for_reduction /= 2; */

    // Hardware can do reduction of 8 tiles at a time.
    // CB sizes can be restricted to this in case input channels are more than 256 to perform reduction iteratively.
    constexpr uint32_t MAX_TILES_PER_REDUCTION = 8;
    const bool is_large_kernel = kernel_size_hw > max_rows_for_reduction;
    const bool is_wide_reduction = in_ntiles_c > MAX_TILES_PER_REDUCTION;

    TT_FATAL(nblocks == 1, "Multiple blocks not yet supported");

    uint32_t tile_w = tt::constants::TILE_WIDTH;
    if (input_shape[3] < tt::constants::TILE_WIDTH) {
        TT_FATAL(input_shape[3] == 16, "Error");
        tile_w = tt::constants::FACE_WIDTH;
    }
    uint32_t out_w_loop_count = std::ceil((float)out_w / nblocks);

    // distributing out_hw across the grid
    auto grid_size = device->compute_with_storage_grid_size();
    auto all_cores = input.shard_spec().value().grid;
    uint32_t ncores = all_cores.num_cores();
    auto core_range = all_cores;
    auto core_range_cliff = CoreRangeSet();
    uint32_t in_nhw_per_core = input.shard_spec()->shape[0];
    uint32_t in_nhw_per_core_cliff = 0;
    uint32_t out_nhw_per_core = output.shard_spec()->shape[0];

    uint32_t ncores_w = grid_size.x;

    // TODO: support generic nblocks
    TT_FATAL(
        out_nhw_per_core % nblocks == 0,
        "number of sticks per core ({}) should be divisible by nblocks ({})",
        out_nhw_per_core,
        nblocks);

    // CBs
    uint32_t multi_buffering_factor = 2;

    uint32_t split_reader = 1;

    // scalar CB as coefficient of reduce
    using tt::tt_metal::CBHandle;
    using tt::tt_metal::CircularBuffer;
    using tt::tt_metal::CircularBufferConfig;
    uint32_t in_scalar_cb_id = tt::CBIndex::c_4;
    uint32_t in_scalar_cb_pagesize = tile_size(in_df);
    uint32_t in_scalar_cb_npages = 1;
    CircularBufferConfig in_scalar_cb_config =
        CircularBufferConfig(in_scalar_cb_npages * in_scalar_cb_pagesize, {{in_scalar_cb_id, in_df}})
            .set_page_size(in_scalar_cb_id, in_scalar_cb_pagesize);
    auto in_scalar_cb = tt::tt_metal::CreateCircularBuffer(program, all_cores, in_scalar_cb_config);
    log_debug(tt::LogOp, "CB {} :: PS = {}, NP = {}", in_scalar_cb_id, in_scalar_cb_pagesize, in_scalar_cb_npages);

    // incoming data is the input cb instead of raw l1/dram addr
    // this input shard has halo and padding inserted.
    auto raw_in_cb_id = tt::CBIndex::c_2;
    uint32_t raw_in_cb_npages = input.shard_spec().value().shape[0];
    uint32_t raw_in_cb_pagesize = in_nbytes_c;
    CircularBufferConfig raw_in_cb_config =
        CircularBufferConfig(raw_in_cb_npages * raw_in_cb_pagesize, {{raw_in_cb_id, in_df}})
            .set_page_size(raw_in_cb_id, raw_in_cb_pagesize)
            .set_globally_allocated_address(*input.buffer());
    auto raw_in_cb = tt::tt_metal::CreateCircularBuffer(program, all_cores, raw_in_cb_config);
    log_debug(tt::LogOp, "CB {} :: PS = {}, NP = {}", raw_in_cb_id, raw_in_cb_pagesize, raw_in_cb_npages);

    // reader indices
    auto in_reader_indices_cb_id = tt::CBIndex::c_3;
    uint32_t in_reader_indices_cb_pagesize =
        tt::round_up(out_nhw_per_core * indices_nbytes, 4);  // pagesize needs to be multiple of 4
    uint32_t in_reader_indices_cb_npages = 1;
    log_debug(
        tt::LogOp,
        "CB {} :: PS = {}, NP = {}",
        in_reader_indices_cb_id,
        in_reader_indices_cb_pagesize,
        in_reader_indices_cb_npages);
    CircularBufferConfig in_reader_indices_cb_config =
        CircularBufferConfig(
            in_reader_indices_cb_npages * in_reader_indices_cb_pagesize, {{in_reader_indices_cb_id, indices_df}})
            .set_page_size(in_reader_indices_cb_id, in_reader_indices_cb_pagesize)
            .set_globally_allocated_address(*reader_indices_buffer->get_device_buffer());
    auto in_reader_indices_cb = tt::tt_metal::CreateCircularBuffer(program, all_cores, in_reader_indices_cb_config);

    uint32_t in_cb_sz = 0;
    uint32_t in_nblocks_c = 1;
    if (is_wide_reduction) {
        in_cb_sz = MAX_TILES_PER_REDUCTION * tt::constants::TILE_HW;
        in_nblocks_c = std::ceil((float)in_ntiles_c / MAX_TILES_PER_REDUCTION);
    } else {
        in_cb_sz = in_ntiles_c * tt::constants::TILE_HW;
    }
    // reader output == input to tilize
    uint32_t in_cb_id_0 = tt::CBIndex::c_0;  // input rows for "multiple (out_nelems)" output pixels
    uint32_t in_cb_id_1 = tt::CBIndex::c_1;  // input rows for "multiple (out_nelems)" output pixels
    uint32_t in_cb_page_padded = tt::round_up(
        in_cb_sz,
        tt::constants::TILE_HW);  // NOTE: ceil to tile size since triscs work with tilesize instead of pagesize
    uint32_t in_cb_pagesize = in_nbytes * in_cb_page_padded;
    uint32_t in_cb_npages = multi_buffering_factor * nblocks;

    CircularBufferConfig in_cb_config_0 = CircularBufferConfig(in_cb_npages * in_cb_pagesize, {{in_cb_id_0, in_df}})
                                              .set_page_size(in_cb_id_0, in_cb_pagesize);
    auto in_cb_0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, in_cb_config_0);
    log_debug(tt::LogOp, "CB {} :: PS = {}, NP = {}", in_cb_id_0, in_cb_pagesize, in_cb_npages);

    if (split_reader) {
        CircularBufferConfig in_cb_config_1 = CircularBufferConfig(in_cb_npages * in_cb_pagesize, {{in_cb_id_1, in_df}})
                                                  .set_page_size(in_cb_id_1, in_cb_pagesize);
        auto in_cb_1 = tt::tt_metal::CreateCircularBuffer(program, all_cores, in_cb_config_1);
        log_debug(tt::LogOp, "CB {} :: PS = {}, NP = {}", in_cb_id_1, in_cb_pagesize, in_cb_npages);
    }

    // output of tilize == input to reduce
    uint32_t in_tiled_cb_id = tt::CBIndex::c_24;  // tiled input
    uint32_t in_tiled_cb_pagesize = tile_size(in_df);
    uint32_t in_tiled_cb_npages = in_ntiles_c * in_ntiles_hw * nblocks;
    CircularBufferConfig in_tiled_cb_config =
        CircularBufferConfig(in_tiled_cb_npages * in_tiled_cb_pagesize, {{in_tiled_cb_id, in_df}})
            .set_page_size(in_tiled_cb_id, in_tiled_cb_pagesize);
    auto in_tiled_cb = tt::tt_metal::CreateCircularBuffer(program, all_cores, in_tiled_cb_config);
    log_debug(tt::LogOp, "CB {} :: PS = {}, NP = {}", in_tiled_cb_id, in_tiled_cb_pagesize, in_tiled_cb_npages);

    // output of reduce == writer to write
    uint32_t out_cb_id = tt::CBIndex::c_16;  // output rows in RM
    // after reduction
    uint32_t out_cb_pagesize = std::min(tt::constants::TILE_WIDTH, output.shard_spec().value().shape[1]) *
                               out_nbytes;  // there is just one row of channels after each reduction (or 1 block
                                            // of c if its greater than 8 tiles)
    uint32_t out_cb_npages = output.shard_spec().value().shape[0] * in_ntiles_c;

    CircularBufferConfig cb_out_config = CircularBufferConfig(out_cb_npages * out_cb_pagesize, {{out_cb_id, out_df}})
                                             .set_page_size(out_cb_id, out_cb_pagesize)
                                             .set_globally_allocated_address(*output.buffer());
    ;
    auto cb_out = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_out_config);
    log_debug(tt::LogOp, "CB {} :: PS = {}, NP = {}", out_cb_id, out_cb_pagesize, out_cb_npages);

    if (is_large_kernel) {
        uint32_t max_pool_partials_cb_id = tt::CBIndex::c_25;  // max_pool partials
        uint32_t max_pool_partials_cb_pagesize = out_cb_pagesize;
        uint32_t max_pool_partials_cb_npages = nblocks;
        CircularBufferConfig max_pool_partials_cb_config =
            CircularBufferConfig(
                max_pool_partials_cb_npages * max_pool_partials_cb_pagesize, {{max_pool_partials_cb_id, out_df}})
                .set_page_size(max_pool_partials_cb_id, max_pool_partials_cb_pagesize);
        auto max_pool_partials_cb = tt::tt_metal::CreateCircularBuffer(program, all_cores, max_pool_partials_cb_config);
        log_debug(
            tt::LogOp,
            "CB {} :: PS = {}, NP = {}",
            max_pool_partials_cb_id,
            max_pool_partials_cb_pagesize,
            max_pool_partials_cb_npages);
    }
    TT_FATAL(output.memory_config().is_sharded(), "Output memory config needs to be sharded");

#if 1
    {  // debug
        log_debug(tt::LogOp, "raw_in_cb :: PS = {}, NP = {}", raw_in_cb_pagesize, raw_in_cb_npages);
        log_debug(tt::LogOp, "in_cb :: PS = {}, NP = {}", in_cb_pagesize, in_cb_npages);
        log_debug(
            tt::LogOp,
            "in_reader_indices_cb :: PS = {}, NP = {}",
            in_reader_indices_cb_pagesize,
            in_reader_indices_cb_npages);
        log_debug(tt::LogOp, "in_scalar_cb :: PS = {}, NP = {}", in_scalar_cb_pagesize, in_scalar_cb_npages);
        log_debug(tt::LogOp, "in_tiled_cb :: PS = {}, NP = {}", in_tiled_cb_pagesize, in_tiled_cb_npages);
        log_debug(tt::LogOp, "out_cb :: PS = {}, NP = {}", out_cb_pagesize, out_cb_npages);
        log_debug(tt::LogOp, "in_addr: {}", src_dram_buffer->address());
        log_debug(tt::LogOp, "in_reader_indices_addr: {}", reader_indices_buffer->address());
        log_debug(tt::LogOp, "out_addr: {}", dst_dram_buffer->address());
        log_debug(tt::LogOp, "kernel_size_h: {}", kernel_size_h);
        log_debug(tt::LogOp, "kernel_size_w: {}", kernel_size_w);
        log_debug(tt::LogOp, "kernel_size_hw: {}", kernel_size_hw);
        log_debug(tt::LogOp, "kernel_size_hw_padded: {}", kernel_size_hw_padded);
        log_debug(tt::LogOp, "stride_h: {}", stride_h);
        log_debug(tt::LogOp, "stride_w: {}", stride_w);
        log_debug(tt::LogOp, "pad_h: {}", pad_h);
        log_debug(tt::LogOp, "pad_w: {}", pad_w);
        log_debug(tt::LogOp, "out_h: {}", out_h);
        log_debug(tt::LogOp, "out_w: {}", out_w);
        log_debug(tt::LogOp, "out_w_loop_count: {}", out_w_loop_count);
        log_debug(tt::LogOp, "out_c: {}", output_shape[3]);
        log_debug(tt::LogOp, "out_nbytes_c: {}", out_nbytes_c);
        log_debug(tt::LogOp, "in_h: {}", in_h);
        log_debug(tt::LogOp, "in_w: {}", in_w);
        log_debug(tt::LogOp, "in_c: {}", input_shape[3]);
        log_debug(tt::LogOp, "in_ntiles_c: {}", in_ntiles_c);
        log_debug(tt::LogOp, "in_nblocks_c: {}", in_nblocks_c);
        log_debug(tt::LogOp, "in_nbytes_c: {}", in_nbytes_c);
        log_debug(tt::LogOp, "out_ntiles_c: {}", out_ntiles_c);
        log_debug(tt::LogOp, "nblocks: {}", nblocks);
        log_debug(tt::LogOp, "ncores: {}", ncores);
        log_debug(tt::LogOp, "in_nhw_per_core: {}", in_nhw_per_core);
        log_debug(tt::LogOp, "out_nhw_per_core: {}", out_nhw_per_core);
        log_debug(tt::LogOp, "split_reader: {}", split_reader);
        log_debug(tt::LogOp, "multi_buffering_factor: {}", multi_buffering_factor);
        log_debug(tt::LogOp, "is_wide_reduction: {}", is_wide_reduction);
        log_debug(tt::LogOp, "is_large_kernel: {}", is_large_kernel);
        log_debug(tt::LogOp, "is_in_sharded: {}", input.memory_config().is_sharded());
        log_debug(tt::LogOp, "is_out_sharded: {}", output.memory_config().is_sharded());
    }
#endif

    /**
     * Reader Kernel: input rows -> input cb
     */
    float one = 1.;
    uint32_t bf16_one_u32 = *reinterpret_cast<uint32_t*>(&one);
    std::vector<uint32_t> reader0_ct_args = {
        out_nhw_per_core,
        kernel_size_h,
        kernel_size_w,
        pad_w,
        in_nbytes_c,
        in_w,
        in_cb_page_padded * in_cb_npages / tile_w,
        input_shape[3] / num_shards_c,
        nblocks,
        split_reader,  // enable split reader
        0,             // split reader id
        bf16_one_u32,
        in_nblocks_c,
        in_cb_sz,
        max_rows_for_reduction,
        ceil_pad_w};

    std::vector<uint32_t> reader1_ct_args = {
        out_nhw_per_core,
        kernel_size_h,
        kernel_size_w,
        pad_w,
        in_nbytes_c,
        in_w,
        in_cb_page_padded * in_cb_npages / tile_w,
        input_shape[3] / num_shards_c,
        nblocks,
        split_reader,  // enable split reader
        1,             // split reader id
        bf16_one_u32,
        in_nblocks_c,
        in_cb_sz,
        max_rows_for_reduction,
        ceil_pad_w};

    std::string reader_kernel_fname;
    if (is_large_kernel) {
        reader_kernel_fname =
            "ttnn/cpp/ttnn/operations/pool/generic/device/kernels/dataflow/"
            "reader_max_pool_2d_multi_core_sharded_with_halo_large_kernel_v2.cpp";
    } else if (is_wide_reduction) {
        reader_kernel_fname =
            "ttnn/cpp/ttnn/operations/pool/generic/device/kernels/dataflow/"
            "reader_max_pool_2d_multi_core_sharded_with_halo_wide.cpp";
    } else {
        reader_kernel_fname =
            "ttnn/cpp/ttnn/operations/pool/generic/device/kernels/dataflow/"
            "reader_max_pool_2d_multi_core_sharded_with_halo_v2.cpp";
    }

    auto reader0_config = tt::tt_metal::DataMovementConfig{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
        .noc = tt::tt_metal::NOC::RISCV_0_default,
        .compile_args = reader0_ct_args};
    auto reader0_kernel = CreateKernel(program, reader_kernel_fname, all_cores, reader0_config);

    auto reader1_config = tt::tt_metal::DataMovementConfig{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
        .noc = tt::tt_metal::NOC::RISCV_1_default,
        .compile_args = reader1_ct_args};
    auto reader1_kernel = split_reader ? CreateKernel(program, reader_kernel_fname, all_cores, reader1_config) : 0;

    /**
     * Compute Kernel: input cb -> tilize_block -> input tiles -> reduce_h max -> output tiles -> untilize_block ->
     * output cb
     */
    std::vector<uint32_t> compute_ct_args = {
        in_ntiles_hw,
        in_ntiles_c,
        in_ntiles_hw * in_ntiles_c,
        kernel_size_hw,
        out_h,
        out_w,
        tt::div_up(output_shape[2], tt::constants::TILE_HEIGHT),
        tt::div_up(output_shape[3], num_shards_c * tt::constants::TILE_WIDTH),
        nblocks,
        out_w_loop_count,
        1,
        out_nhw_per_core,
        split_reader,                // enable split reader
        out_nhw_per_core / nblocks,  // loop count with blocks
        input_shape[3] / num_shards_c,
        in_nblocks_c,
        max_rows_for_reduction};

    auto reduce_op = tt::tt_metal::ReduceOpMath::MAX;
    auto reduce_dim = tt::tt_metal::ReduceOpDim::H;
    auto compute_config = tt::tt_metal::ComputeConfig{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = false,
        .math_approx_mode = false,
        .compile_args = compute_ct_args,
        .defines = reduce_op_utils::get_defines(reduce_op, reduce_dim)};
    std::string compute_kernel_fname;
    if (is_large_kernel) {
        compute_kernel_fname =
            "ttnn/cpp/ttnn/operations/pool/generic/device/kernels/compute/max_pool_multi_core_large_kernel.cpp";
    } else {
        // both regular and wide reductions
        compute_kernel_fname = "ttnn/cpp/ttnn/operations/pool/generic/device/kernels/compute/max_pool_multi_core.cpp";
    }

    auto compute_kernel = CreateKernel(program, compute_kernel_fname, core_range, compute_config);

    // Capture reader_indices_buffer to cache this with the program
    return {
        std::move(program),
        {.reader0_kernel = reader0_kernel,
         .reader1_kernel = reader1_kernel,
         .raw_in_cb = raw_in_cb,
         .cb_out = cb_out,
         .ncores = ncores,
         .ncores_w = ncores_w,
         .reader_indices_buffer = reader_indices_buffer}};
}

Pool2D::MultiCore::cached_program_t Pool2D::MultiCore::create(
    const operation_attributes_t& op_attr, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
    const auto& input = tensor_args.input_tensor_;
    const auto& sliding_window_config = op_attr.sliding_window_config_;
    const auto& pool_type = op_attr.pool_type_;
    const auto& out_mem_config = op_attr.memory_config_;

    tt::tt_metal::Program program{};

    auto parallel_config = sliding_window::ParallelConfig{
        .grid = input.shard_spec().value().grid,
        .shard_scheme = input.memory_config().memory_layout,
        .shard_orientation = input.shard_spec().value().orientation,
    };

    auto output_shape = sliding_window_config.get_output_shape();
    uint32_t out_h = output_shape[1];
    uint32_t out_w = output_shape[2];

    bool is_block_sharded = input.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED;

    auto pad_metadata = sliding_window::generate_pad_metadata(sliding_window_config);
    auto op_trace_metadata = sliding_window::generate_op_trace_metadata(sliding_window_config);
    auto shard_boundaries = sliding_window::generate_shard_boundaries(sliding_window_config, op_trace_metadata);
    auto top_left_indices =
        sliding_window::generate_sliding_window_op_config(op_trace_metadata, shard_boundaries, false, true);
    auto reader_indices =
        sliding_window::construct_on_host_config_tensor(top_left_indices, sliding_window_config, parallel_config);
    log_debug(tt::LogOp, "reader_indices shape: {}", reader_indices.logical_shape());
    auto reader_indices_on_device =
        sliding_window::move_config_tensor_to_device(reader_indices, parallel_config, is_block_sharded, input.device());

    auto in_n = sliding_window_config.batch_size;
    auto in_h = sliding_window_config.input_hw.first;
    auto in_w = sliding_window_config.input_hw.second;
    auto kernel_size_h = sliding_window_config.window_hw.first;
    auto kernel_size_w = sliding_window_config.window_hw.second;
    auto stride_h = sliding_window_config.stride_hw.first;
    auto stride_w = sliding_window_config.stride_hw.second;
    auto pad_h = sliding_window_config.pad_hw.first;
    auto pad_w = sliding_window_config.pad_hw.second;
    auto ceil_pad_w = sliding_window_config.get_ceil_pad_w();
    auto dilation_h = sliding_window_config.dilation_hw.first;
    auto dilation_w = sliding_window_config.dilation_hw.second;
    auto num_shards_c = sliding_window_config.num_cores_c;

    return pool2d_multi_core_sharded_with_halo_v2_impl_new(
        program,
        input,
        reader_indices_on_device,
        output_tensor,
        pool_type,
        in_n,
        in_h,
        in_w,
        out_h,
        out_w,
        kernel_size_h,
        kernel_size_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        ceil_pad_w,
        dilation_h,
        dilation_w,
        num_shards_c,
        out_mem_config,
        1);
}

void Pool2D::MultiCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    auto& program = cached_program.program;
    auto& reader0_kernel = cached_program.shared_variables.reader0_kernel;
    auto& reader1_kernel = cached_program.shared_variables.reader1_kernel;
    auto& raw_in_cb = cached_program.shared_variables.raw_in_cb;
    auto& cb_out = cached_program.shared_variables.cb_out;
    auto& ncores = cached_program.shared_variables.ncores;
    auto& ncores_w = cached_program.shared_variables.ncores_w;

    const auto& input_tensor = tensor_args.input_tensor_;

    auto src_buffer = input_tensor.buffer();
    bool input_sharded = input_tensor.is_sharded();

    auto dst_buffer = output_tensor.buffer();
    bool out_sharded = output_tensor.is_sharded();

    if (input_sharded) {
        UpdateDynamicCircularBufferAddress(program, raw_in_cb, *src_buffer);
    }
    if (out_sharded) {
        UpdateDynamicCircularBufferAddress(program, cb_out, *dst_buffer);
    }
}

}  // namespace ttnn::operations::pool
