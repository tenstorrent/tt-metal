// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "ttnn/operations/normalization/layernorm/device/layernorm_device_operation.hpp"
#include <tt-metalium/circular_buffer_config.hpp>
#include "ttnn/operations/normalization/layernorm/device/layernorm_common.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_device_operation_types.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/math.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/normalization/layernorm/device/sharded_layernorm_factory_helpers.hpp"

#include <optional>
#include <bit>

using uint32_t = std::uint32_t;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor LayerNormShardedProgramFactory::create_descriptor(
    const LayerNormParams& operation_attributes,
    const LayerNormInputs& tensor_args,
    Tensor& tensor_return_value,
    const std::optional<CoreRangeSet>& core_range_set) {
    using namespace sharded_layernorm_helpers;

    // For sharded layernorm, core ranges are derived from tensor shard spec.
    // If core_range_set is provided, validate that shard spec cores are within it.
    const auto& input_shard_spec = tensor_args.input.shard_spec();
    TT_FATAL(input_shard_spec.has_value(), "Sharded layernorm requires input tensor to have a shard spec");

    if (core_range_set.has_value()) {
        const auto& shard_grid = input_shard_spec.value().grid;
        // Verify that all cores in the shard spec are within the provided core_range_set
        for (const auto& shard_core_range : shard_grid.ranges()) {
            for (auto x = shard_core_range.start_coord.x; x <= shard_core_range.end_coord.x; ++x) {
                for (auto y = shard_core_range.start_coord.y; y <= shard_core_range.end_coord.y; ++y) {
                    CoreCoord core = {x, y};
                    TT_FATAL(
                        core_range_set.value().contains(core),
                        "Sharded tensor shard spec core ({}, {}) is not within the provided core_range_set. "
                        "The sharded tensor must lie entirely within the input core range.",
                        x,
                        y);
                }
            }
        }
    }

    // Extract from operation_attributes and tensor_args
    const auto& a = tensor_args.input;
    const auto& b = tensor_args.residual_input_tensor;
    const auto& gamma = tensor_args.weight;
    const auto& beta = tensor_args.bias;
    const auto& stats = tensor_args.stats;
    auto& output = tensor_return_value;
    bool rms_norm = operation_attributes.norm_type == LayerNormType::RMSNORM;
    bool is_pre_all_gather = operation_attributes.distributed_norm_stage == DistributedLayerNormStage::PRE_ALL_GATHER;
    bool is_post_all_gather = operation_attributes.distributed_norm_stage == DistributedLayerNormStage::POST_ALL_GATHER;
    float eps = operation_attributes.eps;
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config;

    // Extract program config
    CoreCoord compute_with_storage_grid_size;
    uint32_t subblock_wt = 0;
    uint32_t block_ht = 0;
    uint32_t block_wt = 0;
    bool legacy_reduction = false;
    bool legacy_rsqrt = false;
    bool use_welford = false;
    std::visit(
        [&](const auto& program_config) {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (std::is_same_v<ProgramConfigType, LayerNormShardedMultiCoreProgramConfig>) {
                compute_with_storage_grid_size = program_config.compute_with_storage_grid_size;
                subblock_wt = program_config.subblock_w;
                block_ht = program_config.block_h;
                block_wt = program_config.block_w;
                legacy_reduction = program_config.legacy_reduction;
                legacy_rsqrt = program_config.legacy_rsqrt;
                use_welford = program_config.use_welford;
            }
        },
        operation_attributes.program_config);

    const uint32_t tile_width = a.tensor_spec().tile().get_width();

    uint32_t block_wt_resharded = output.shard_spec().value().shape[1] / tile_width;
    bool skip_write_back = output.shard_spec().value() == a.shard_spec().value();

    ////////////////////////////////////////////////////////////////////////////
    //                            Device Setup
    ////////////////////////////////////////////////////////////////////////////
    IDevice* device = a.device();

    // convert data format
    tt::DataFormat in_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    assert_subblock_compute_config_compatible(dst_full_sync_en, fp32_dest_acc_en, subblock_wt);

    auto
        [out_data_format,
         cb_data_format,
         gamma_cb_data_format,
         beta_cb_data_format,
         stats_cb_data_format,
         reciprocal_cb_data_format] = get_cb_data_formats(output, gamma, beta, stats, fp32_dest_acc_en);

    // tile sizes
    uint32_t in_single_tile_size = tt::tile_size(in_data_format);
    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    uint32_t out_single_tile_size = tt::tile_size(out_data_format);
    uint32_t gamma_single_tile_size = tt::tile_size(gamma_cb_data_format);
    uint32_t beta_single_tile_size = tt::tile_size(beta_cb_data_format);
    uint32_t stats_single_tile_size = tt::tile_size(stats_cb_data_format);
    uint32_t bfloat16_tile_size = tt::tile_size(tt::DataFormat::Float16_b);

    // tensor shape
    const auto& shape = a.padded_shape();
    uint32_t K = shape[-1];
    uint32_t Kt = K / tile_width;
    uint32_t block_w = block_wt * tile_width;
    // Logical (un-padded) width. Welford normalizes over the true element count N, so a
    // non-tile-aligned width must exclude the tile padding columns from both the running count
    // and the final 1/N divisor rather than folding them into the mean and variance.
    const uint32_t logical_K = a.logical_shape()[-1];

    // Compute grid and worker distribution using helper structs
    auto grid = GridParams::compute(a, block_ht, device->compute_with_storage_grid_size());
    auto workers = WorkerDistribution::compute(grid, block_ht);
    auto core_ranges = CoreRanges::compute(grid, workers);

    // Get all storage cores
    ShardSpec output_shard_spec = output.shard_spec().value();
    bool output_row_wise = output_shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    CoreRangeSet all_storage_cores = output_shard_spec.grid;
    CoreRangeSet all_worker_and_storage_cores = all_storage_cores.merge(a.shard_spec().value().grid);
    std::vector<uint32_t> storage_core_noc_x;
    std::vector<uint32_t> storage_core_noc_y;
    std::vector<CoreCoord> storage_core_coords =
        corerange_to_cores(all_storage_cores, all_storage_cores.num_cores(), output_row_wise);
    for (auto core : storage_core_coords) {
        storage_core_noc_x.push_back((std::uint32_t)device->worker_core_from_logical_core(core).x);
        storage_core_noc_y.push_back((std::uint32_t)device->worker_core_from_logical_core(core).y);
    }

    // get sharded addr
    auto gamma_dram_addr = gamma.has_value() ? gamma.value().buffer()->address() : 0;
    auto beta_dram_addr = beta.has_value() ? beta.value().buffer()->address() : 0;

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t pre_all_gather_stats_block_tiles = rms_norm ? 1 : 2;
    uint32_t post_all_gather_stats_block_tiles = 1;
    uint32_t num_distributed_devices = 1;
    if (is_post_all_gather && stats.has_value()) {
        post_all_gather_stats_block_tiles = stats.value().padded_shape()[-1] / tile_width;
        num_distributed_devices = post_all_gather_stats_block_tiles / pre_all_gather_stats_block_tiles;
    }

    // Reciprocal LUT for Welford
    std::optional<Tensor> recip_tensor = std::nullopt;
    uint32_t reciprocal_CB_size_bytes = 0;
    if (use_welford) {
        TT_FATAL(tensor_args.recip_tensor.has_value(), "Reciprocal tensor not provided for Welford layernorm");
        recip_tensor = tensor_args.recip_tensor;
        reciprocal_CB_size_bytes = recip_tensor->buffer()->aligned_size_per_bank();
    }

    // Compute CB sizes using helper
    CBSizeParams cb_size_params{
        .block_ht = block_ht,
        .block_wt = block_wt,
        .block_wt_resharded = block_wt_resharded,
        .Kt = Kt,
        .in_single_tile_size = in_single_tile_size,
        .single_tile_size = single_tile_size,
        .out_single_tile_size = out_single_tile_size,
        .gamma_single_tile_size = gamma_single_tile_size,
        .beta_single_tile_size = beta_single_tile_size,
        .stats_single_tile_size = stats_single_tile_size,
        .bfloat16_tile_size = bfloat16_tile_size,
        .reciprocal_CB_size_bytes = reciprocal_CB_size_bytes,
        .num_rows_per_all_to_all_worker = workers.num_rows_per_all_to_all_worker,
        .num_blocks_first_stage = workers.num_blocks_first_stage,
        .num_blocks_second_stage = workers.num_blocks_second_stage,
        .pre_all_gather_stats_block_tiles = pre_all_gather_stats_block_tiles,
        .post_all_gather_stats_block_tiles = post_all_gather_stats_block_tiles,
        .is_pre_all_gather = is_pre_all_gather,
        .is_post_all_gather = is_post_all_gather,
        .use_two_stage_reduce = grid.use_two_stage_reduce,
        .use_welford = use_welford,
        .skip_write_back = skip_write_back,
        .rms_norm = rms_norm};
    auto cb_sizes = cb_size_params.compute();

    // Build ProgramDescriptor
    ProgramDescriptor program_descriptor;

    // Create semaphore descriptors
    uint32_t reduce_sender_semaphore_id = 0;
    uint32_t reduce_receiver_semaphore_id = 1;
    uint32_t reduce_second_stage_semaphore_id = 2;

    program_descriptor.semaphores.push_back(SemaphoreDescriptor{
        .id = reduce_sender_semaphore_id,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = core_ranges.all_cores,
        .initial_value = 0});
    program_descriptor.semaphores.push_back(SemaphoreDescriptor{
        .id = reduce_receiver_semaphore_id,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = core_ranges.all_cores,
        .initial_value = 0});
    program_descriptor.semaphores.push_back(SemaphoreDescriptor{
        .id = reduce_second_stage_semaphore_id,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = core_ranges.all_cores,
        .initial_value = 0});

    // Get kernel defines using helper
    auto kernel_defines = KernelDefines::build(
        b.has_value(),
        gamma.has_value(),
        beta.has_value(),
        rms_norm,
        use_welford,
        skip_write_back,
        operation_attributes.fused_activation,
        tensor_return_value.dtype());

    // Get kernel paths using helper
    bool use_row_major_kernel = (gamma.has_value() && gamma.value().layout() == Layout::ROW_MAJOR) ||
                                (beta.has_value() && beta.value().layout() == Layout::ROW_MAJOR);
    auto kernel_paths = KernelPaths::get(is_pre_all_gather, is_post_all_gather, use_row_major_kernel, use_welford);

    // NOC selection
    tt::tt_metal::NOC reader_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());
    tt::tt_metal::NOC writer_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    if (is_post_all_gather && !skip_write_back) {
        reader_noc = NOC::NOC_0;
        writer_noc = NOC::NOC_1;
    }

    // Build compile-time args using helper
    CompileTimeArgsContext ct_ctx{
        .reduce_receiver_semaphore_id = reduce_receiver_semaphore_id,
        .reduce_sender_semaphore_id = reduce_sender_semaphore_id,
        .reduce_second_stage_semaphore_id = reduce_second_stage_semaphore_id,
        .grid = &grid,
        .workers = &workers,
        .core_ranges = &core_ranges,
        .block_ht = block_ht,
        .block_wt = block_wt,
        .subblock_wt = subblock_wt,
        .single_tile_size = single_tile_size,
        .out_single_tile_size = out_single_tile_size,
        .block_wt_resharded = block_wt_resharded,
        .K = K,
        .logical_K = logical_K,
        .rms_norm = rms_norm,
        .use_welford = use_welford,
        .has_gamma = gamma.has_value(),
        .has_beta = beta.has_value(),
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .legacy_reduction = legacy_reduction,
        .legacy_rsqrt = legacy_rsqrt,
        .gamma_cb_data_format = gamma_cb_data_format,
        .beta_cb_data_format = beta_cb_data_format,
        .gamma_buffer = gamma.has_value() ? gamma.value().buffer() : nullptr,
        .beta_buffer = beta.has_value() ? beta.value().buffer() : nullptr,
        .gamma_is_row_major = gamma.has_value() && gamma.value().layout() == Layout::ROW_MAJOR,
        .beta_is_row_major = beta.has_value() && beta.value().layout() == Layout::ROW_MAJOR,
        .gamma_stick_size = gamma.has_value() && gamma.value().layout() == Layout::ROW_MAJOR
                                ? gamma.value().padded_shape()[-1] * gamma.value().element_size()
                                : 0,
        .beta_stick_size = beta.has_value() && beta.value().layout() == Layout::ROW_MAJOR
                               ? beta.value().padded_shape()[-1] * beta.value().element_size()
                               : 0,
        .eps = eps,
        .per_core_recip_lut_size = block_w,
        .tile_width = tile_width};
    auto compile_time_args = CompileTimeArgs::build(ct_ctx);

    // Pack eps for later use
    uint32_t eps_u = std::bit_cast<uint32_t>(eps);

    // Build runtime args using helper
    const auto& cores = corerange_to_cores(core_ranges.all_cores, core_ranges.all_cores.num_cores(), grid.row_wise);

    uint32_t last_core_width_index =
        grid.mcast_1d ? (cores.size() - 1) : (grid.row_wise ? (grid.grid_size.x - 1) : (grid.grid_size.y - 1));

    // A column mask is needed only when a reduced tile contains padding, i.e. the last tile of the
    // logical width is partially valid (logical width not a multiple of the tile width). Whole padding
    // tiles past the logical width are excluded from the reduction by the per-core reduce-tile count
    // instead.
    const bool col_mask_needed = (logical_K % tile_width != 0);

    // Compute packed values for writer.
    // The reduction scaler (winv) is applied per core; when the reduction is split across cores
    // (num_blocks > 1) the cross-core global reduce then averages across the num_blocks blocks
    // (cinv = 1/num_blocks). The net per-element divide is winv*cinv and must equal 1/logical_K (the
    // reduction only ever sums the logical columns; padding is kept out of the sum by the masking
    // described below). With num_blocks == 1 there is no cross-core average, so winv = 1/logical_K.
    // With num_blocks > 1, winv = num_blocks/logical_K cancels the average;
    // this equals the per-core 1/block_w when the blocks tile the logical width exactly, and
    // stays correct when they do not (e.g. 96 over two 64-wide blocks: divide by the logical 96, not
    // the physical 128).
    float winv = (grid.num_blocks == 1) ? (1.0f / logical_K) : (static_cast<float>(grid.num_blocks) / logical_K);
    float cinv = is_post_all_gather ? (1.0f / num_distributed_devices) : (1.0f / grid.num_blocks);
    auto bfloat_cinv = bfloat16(cinv);
    auto bfloat_cinv_one = bfloat16(1.0f);
    auto bfloat_winv = bfloat16(winv);

    // Build mcast NOC coordinates
    std::vector<uint32_t> mcast_noc_x, mcast_noc_y;
    mcast_noc_x.reserve(grid.grid_size.x);
    mcast_noc_y.reserve(grid.grid_size.y);
    CoreCoord core_start_offset = grid.grid_offset.value_or(CoreCoord{0, 0});
    for (uint32_t x = core_start_offset.x; x < grid.grid_size.x + core_start_offset.x; ++x) {
        mcast_noc_x.push_back(device->worker_core_from_logical_core({x, core_start_offset.y}).x);
    }
    for (uint32_t y = core_start_offset.y; y < grid.grid_size.y + core_start_offset.y; ++y) {
        mcast_noc_y.push_back(device->worker_core_from_logical_core({core_start_offset.x, y}).y);
    }

    // A non-tile-aligned width split across multiple cores is supported on every path. The non-Welford
    // path masks each core's final-tile padding columns with its per-core column mask (CB 19). Welford
    // has no column mask, so each core is instead told its real (logical) column count (welford_reduce_w)
    // and reduces exactly those columns; full block_w on the cores before the last, the remaining logical
    // columns (ending in a partial tile) on the final real core; and the cross-core combine weights the
    // final block by its true width (last_block_w).
    // Legacy (non-Welford) path: zero the padding columns of a non-tile-aligned width's final tile so
    // they do not enter the statistics (E[x] and variance for layernorm, the mean of squares for
    // RMSNorm), except the post-all-gather stage, which reduces gathered stats rather than the input.
    // The mask is CB 19 at every masking site, generated on-device in the writer (generate_mask_w<T>)
    // keyed off each core's width position, so it carries the correct validity whether the width lives
    // on one core or is split across many. CB 14 (E[x] scratch) additionally feeds the non-distributed
    // LayerNorm E[x] site so cb_in stays intact for the (x - E[x]) pass.
    const bool do_col_mask = col_mask_needed && !use_welford && !is_post_all_gather;
    const bool do_legacy_layernorm_col_mask = do_col_mask && !rms_norm && !is_pre_all_gather;

    RuntimeArgsContext rt_ctx{
        .grid = grid,
        .workers = workers,
        .core_ranges = core_ranges,
        .mcast_noc_x = std::move(mcast_noc_x),
        .mcast_noc_y = std::move(mcast_noc_y),
        .packed_cinv_value = pack_two_bfloat16_into_uint32({bfloat_cinv, bfloat_cinv}),
        .packed_cinv_value_one = pack_two_bfloat16_into_uint32({bfloat_cinv_one, bfloat_cinv_one}),
        .packed_winv_value = pack_two_bfloat16_into_uint32({bfloat_winv, bfloat_winv}),
        .eps_u = eps_u,
        .gamma_dram_addr = gamma_dram_addr,
        .beta_dram_addr = beta_dram_addr,
        .single_tile_size = single_tile_size,
        .out_single_tile_size = out_single_tile_size,
        .block_wt = block_wt,
        .block_wt_resharded = block_wt_resharded,
        .Kt = Kt,
        .logical_K = logical_K,
        .last_core_width_index = last_core_width_index,
        .is_post_all_gather = is_post_all_gather,
        .num_distributed_devices = num_distributed_devices,
        .reader_noc = reader_noc,
        .storage_core_noc_x = std::move(storage_core_noc_x),
        .storage_core_noc_y = std::move(storage_core_noc_y),
        .num_storage_cores = (uint32_t)all_storage_cores.num_cores()};

    auto runtime_args = RuntimeArgsResult::build(cores, rt_ctx, device);

    ////////////////////////////////////////////////////////////////////////////
    //                      Build Kernel Descriptors
    ////////////////////////////////////////////////////////////////////////////
    KernelConfig kernel_config;
    kernel_config.reader_sender_path = kernel_paths.reader_sender;
    kernel_config.reader_receiver_path = kernel_paths.reader_receiver;
    kernel_config.writer_path = kernel_paths.writer;
    kernel_config.compute_path = kernel_paths.compute;
    kernel_config.reader_sender_ct_args = std::move(compile_time_args.reader_sender);
    kernel_config.reader_receiver_all_to_all_ct_args = std::move(compile_time_args.reader_receiver_all_to_all);
    kernel_config.reader_receiver_ct_args = std::move(compile_time_args.reader_receiver);
    kernel_config.writer_sender_ct_args = std::move(compile_time_args.writer_sender);
    kernel_config.writer_receiver_ct_args = std::move(compile_time_args.writer_receiver);
    kernel_config.compute_all_to_all_ct_args = std::move(compile_time_args.compute_all_to_all);
    kernel_config.compute_not_all_to_all_ct_args = std::move(compile_time_args.compute_not_all_to_all);
    kernel_config.reader_sender_defines = kernel_defines.reader;
    kernel_config.reader_receiver_defines = kernel_defines.reader;
    kernel_config.writer_defines = std::move(kernel_defines.writer);
    kernel_config.compute_defines = std::move(kernel_defines.compute);
    kernel_config.reader_sender_rt_args = std::move(runtime_args.reader_sender);
    kernel_config.reader_receiver_all_to_all_rt_args = std::move(runtime_args.reader_receiver_all_to_all);
    kernel_config.reader_receiver_rt_args = std::move(runtime_args.reader_receiver);
    kernel_config.writer_sender_rt_args = std::move(runtime_args.writer_sender);
    kernel_config.writer_receiver_rt_args = std::move(runtime_args.writer_receiver);
    kernel_config.compute_all_to_all_rt_args = std::move(runtime_args.compute_all_to_all);
    kernel_config.compute_not_all_to_all_rt_args = std::move(runtime_args.compute_not_all_to_all);
    kernel_config.reader_noc = reader_noc;
    kernel_config.writer_noc = writer_noc;
    kernel_config.math_fidelity = math_fidelity;
    kernel_config.fp32_dest_acc_en = fp32_dest_acc_en;
    kernel_config.dst_full_sync_en = dst_full_sync_en;
    kernel_config.math_approx_mode = math_approx_mode;
    // Enable the welford-fp32 alias only when the SrcA-routed transpose_tile would
    // otherwise truncate Float32 input to TF32. Restricting to !rms_norm because
    // RMSNorm doesn't use Welford in this kernel path.
    kernel_config.welford_fp32_alias =
        use_welford && !rms_norm && in_data_format == tt::DataFormat::Float32 && fp32_dest_acc_en;
    // Writer named compile-time args (block_w in tiles and the Welford flag).
    kernel_config.block_wt = block_wt;
    kernel_config.use_welford = use_welford;
    if (do_col_mask) {
        // The writer generates the CB 19 mask on-device with generate_mask_w; compute applies it at
        // every masking site. Pass the logical width so the writer knows where the padding columns begin.
        kernel_config.compute_defines.emplace_back("DO_COL_MASK", "1");
        kernel_config.writer_defines.emplace_back("DO_COL_MASK", "1");
        kernel_config.logical_K = logical_K;
    }
    kernel_config.gamma_buffer = gamma.has_value() ? gamma.value().buffer() : nullptr;
    kernel_config.beta_buffer = beta.has_value() ? beta.value().buffer() : nullptr;

    add_kernel_descriptors(program_descriptor, core_ranges, workers, grid, std::move(kernel_config));

    ////////////////////////////////////////////////////////////////////////////
    //                      Build CB Descriptors
    ////////////////////////////////////////////////////////////////////////////
    // When resharding (is_post_all_gather && !skip_write_back):
    //   CB 16 is intermediate (no buffer), CB 17 is the final resharded output
    // Otherwise:
    //   CB 16 is the output, CB 17 is not used
    Buffer* output_buffer = nullptr;
    Buffer* output_reshard_buffer = nullptr;
    if (is_post_all_gather && !skip_write_back) {
        // Resharding case: CB 17 needs the output buffer
        output_reshard_buffer = output.buffer();
    } else {
        // Normal case: CB 16 needs the output buffer
        output_buffer = output.buffer();
    }

    CBConfig cb_config;
    cb_config.in0_CB_size = cb_sizes.in0_CB_size;
    cb_config.in1_CB_size = cb_sizes.in1_CB_size;
    cb_config.in2_CB_size = cb_sizes.in2_CB_size;
    cb_config.in3_CB_size = cb_sizes.in3_CB_size;
    cb_config.in5_CB_size = cb_sizes.in5_CB_size;
    cb_config.in6_CB_size = cb_sizes.in6_CB_size;
    cb_config.x_CB_size = cb_sizes.x_CB_size;
    cb_config.xmm_CB_size = cb_sizes.xmm_CB_size;
    cb_config.ex_partial_CB_size = cb_sizes.ex_partial_CB_size;
    cb_config.ex_CB_size = cb_sizes.ex_CB_size;
    cb_config.ex_external_CB_size = cb_sizes.ex_external_CB_size;
    cb_config.ex_global_CB_size = cb_sizes.ex_global_CB_size;
    cb_config.ex2pe_CB_size = cb_sizes.ex2pe_CB_size;
    cb_config.out_CB_size = cb_sizes.out_CB_size;
    cb_config.out_reshard_CB_size = cb_sizes.out_reshard_CB_size;
    cb_config.stats_cb_size = cb_sizes.stats_cb_size;
    cb_config.stats_reduced_cb_size = cb_sizes.stats_reduced_cb_size;
    cb_config.reciprocal_CB_size_bytes = reciprocal_CB_size_bytes;
    cb_config.in_data_format = in_data_format;
    cb_config.cb_data_format = cb_data_format;
    cb_config.out_data_format = out_data_format;
    cb_config.gamma_cb_data_format = gamma_cb_data_format;
    cb_config.beta_cb_data_format = beta_cb_data_format;
    cb_config.stats_cb_data_format = stats_cb_data_format;
    cb_config.reciprocal_cb_data_format = reciprocal_cb_data_format;
    cb_config.in_single_tile_size = in_single_tile_size;
    cb_config.single_tile_size = single_tile_size;
    cb_config.out_single_tile_size = out_single_tile_size;
    cb_config.gamma_single_tile_size = gamma_single_tile_size;
    cb_config.beta_single_tile_size = beta_single_tile_size;
    cb_config.stats_single_tile_size = stats_single_tile_size;
    cb_config.bfloat16_tile_size = bfloat16_tile_size;
    cb_config.a_buffer = a.buffer();
    cb_config.b_buffer = b.has_value() ? b.value().buffer() : nullptr;
    cb_config.gamma_buffer = gamma.has_value() ? gamma.value().buffer() : nullptr;
    cb_config.beta_buffer = beta.has_value() ? beta.value().buffer() : nullptr;
    cb_config.stats_buffer = stats.has_value() ? stats.value().buffer() : nullptr;
    cb_config.recip_buffer = recip_tensor.has_value() ? recip_tensor.value().buffer() : nullptr;
    cb_config.output_buffer = output_buffer;
    cb_config.output_reshard_buffer = output_reshard_buffer;
    cb_config.has_b = b.has_value();
    cb_config.has_gamma = gamma.has_value();
    cb_config.has_beta = beta.has_value();
    cb_config.rms_norm = rms_norm;
    cb_config.use_welford = use_welford;
    cb_config.is_pre_all_gather = is_pre_all_gather;
    cb_config.is_post_all_gather = is_post_all_gather;
    cb_config.skip_write_back = skip_write_back;
    // CB 19 is the writer-generated column mask; size it to block_wt tiles (one tile-row). The mask holds
    // only 1.0 or 0.0 in bfloat16.
    cb_config.do_col_mask = do_col_mask;
    cb_config.col_mask_gen_CB_size_bytes = block_wt * bfloat16_tile_size;
    cb_config.do_legacy_layernorm_col_mask = do_legacy_layernorm_col_mask;
    // Enable the welford-fp32 alias only when the SrcA-routed transpose_tile would
    // otherwise truncate Float32 input to TF32. Restricting to !rms_norm because
    // RMSNorm doesn't use Welford in this kernel path.
    cb_config.welford_fp32_alias =
        use_welford && !rms_norm && in_data_format == tt::DataFormat::Float32 && fp32_dest_acc_en;

    add_cb_descriptors(program_descriptor, core_ranges, all_worker_and_storage_cores, cb_config);

    return program_descriptor;
}

}  // namespace ttnn::prim
