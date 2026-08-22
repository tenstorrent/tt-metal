// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "ttnn/operations/normalization/layernorm/device/layernorm_device_operation.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_common.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_device_operation_types.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "ttnn/operations/normalization/layernorm/device/sharded_layernorm_factory_helpers.hpp"

#include <optional>
#include <bit>
#include <cstdlib>
#include <cstdint>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace m2 = tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts LayerNormShardedProgramFactory::create_program_artifacts(
    const LayerNormParams& operation_attributes,
    const LayerNormInputs& tensor_args,
    Tensor& tensor_return_value,
    const std::optional<CoreRangeSet>& core_range_set) {
    using namespace sharded_layernorm_helpers;

    // For sharded layernorm, core ranges are derived from tensor shard spec.
    // If core_range_set is provided, validate that every core this program will touch is within it.
    const auto& input_shard_spec = tensor_args.input.shard_spec();
    TT_FATAL(input_shard_spec.has_value(), "Sharded layernorm requires input tensor to have a shard spec");

    if (core_range_set.has_value()) {
        const auto& shard_grid = input_shard_spec.value().grid;
        // Multicast destinations span the whole bounding box of the shard grid, so a non-rectangular
        // shard grid also places idle kernels, buffers and semaphores on the holes inside that box.
        // Validate the bounding box rather than just the active shard cores, otherwise the program
        // would run on cores the caller excluded and can collide with other programs.
        const CoreRange mcast_bbox = shard_grid.bounding_box();
        for (auto x = mcast_bbox.start_coord.x; x <= mcast_bbox.end_coord.x; ++x) {
            for (auto y = mcast_bbox.start_coord.y; y <= mcast_bbox.end_coord.y; ++y) {
                CoreCoord core = {x, y};
                if (core_range_set.value().contains(core)) {
                    continue;
                }
                TT_FATAL(
                    !shard_grid.contains(core),
                    "Sharded tensor shard spec core ({}, {}) is not within the provided core_range_set. "
                    "The sharded tensor must lie entirely within the input core range.",
                    x,
                    y);
                TT_THROW(
                    "Core ({}, {}) is a hole in the non-rectangular shard grid {} and is not within the provided "
                    "core_range_set. Sharded layernorm multicasts over the bounding box {} of the shard grid, so "
                    "every core in that bounding box must be included in core_range_set.",
                    x,
                    y,
                    shard_grid.str(),
                    mcast_bbox.str());
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
    std::uint32_t subblock_wt = 0;
    std::uint32_t block_ht = 0;
    std::uint32_t block_wt = 0;
    bool legacy_reduction = false;
    bool legacy_rsqrt = false;
    bool requested_use_welford = false;
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
                requested_use_welford = program_config.use_welford;
            }
        },
        operation_attributes.program_config);

    const std::uint32_t tile_width = a.tensor_spec().tile().get_width();

    std::uint32_t block_wt_resharded = output.shard_spec().value().shape[1] / tile_width;
    bool skip_write_back = output.shard_spec().value() == a.shard_spec().value();
    // The write-back reads runtime arguments that only the post-all-gather stage supplies, so the
    // build that compiles it is the build where those arguments exist.
    const bool writes_back = is_post_all_gather && !skip_write_back;

    ////////////////////////////////////////////////////////////////////////////
    //                            Device Setup
    ////////////////////////////////////////////////////////////////////////////
    IDevice* device = a.device();

    // Blackhole's centred tile-reduction kernel is substantially faster and showed no numerical
    // disadvantage versus the transpose/SFPU two-pass kernel for non-distributed sharded LayerNorm.
    // Keep the latter on other architectures until they have equivalent numerical and performance validation.
    const bool use_welford =
        requested_use_welford && !(device->arch() == tt::ARCH::BLACKHOLE && !is_pre_all_gather && !is_post_all_gather);

    // convert data format
    tt::DataFormat in_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    assert_subblock_compute_config_compatible(dst_full_sync_en, fp32_dest_acc_en, subblock_wt);

    auto
        [out_data_format,
         dfb_data_format,
         gamma_dfb_data_format,
         beta_dfb_data_format,
         stats_dfb_data_format,
         reciprocal_dfb_data_format] = get_dfb_data_formats(output, gamma, beta, stats, fp32_dest_acc_en);

    // tile sizes
    uint32_t in_single_tile_size = tt::tile_size(in_data_format);
    uint32_t single_tile_size = tt::tile_size(dfb_data_format);
    uint32_t out_single_tile_size = tt::tile_size(out_data_format);
    uint32_t gamma_single_tile_size = tt::tile_size(gamma_dfb_data_format);
    uint32_t beta_single_tile_size = tt::tile_size(beta_dfb_data_format);
    uint32_t stats_single_tile_size = tt::tile_size(stats_dfb_data_format);
    uint32_t bfloat16_tile_size = tt::tile_size(tt::DataFormat::Float16_b);

    // tensor shape
    const auto& shape = a.padded_shape();
    std::uint32_t K = shape[-1];
    std::uint32_t Kt = K / tile_width;
    std::uint32_t block_w = block_wt * tile_width;
    // Logical (un-padded) width. Welford normalizes over the true element count N, so a
    // non-tile-aligned width must exclude the tile padding columns from both the running count
    // and the final 1/N divisor rather than folding them into the mean and variance.
    const std::uint32_t logical_K = a.logical_shape()[-1];

    // Compute grid and worker distribution using helper structs
    auto grid = GridParams::compute(a, block_ht, device->compute_with_storage_grid_size());
    auto workers = WorkerDistribution::compute(grid, block_ht);
    auto core_ranges = CoreRanges::compute(grid, workers);

    // Get all storage cores
    ShardSpec output_shard_spec = output.shard_spec().value();
    bool output_row_wise = output_shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    CoreRangeSet all_storage_cores = output_shard_spec.grid;
    std::vector<uint32_t> storage_core_noc_x;
    std::vector<uint32_t> storage_core_noc_y;
    std::vector<CoreCoord> storage_core_coords =
        corerange_to_cores(all_storage_cores, all_storage_cores.num_cores(), output_row_wise);
    for (auto core : storage_core_coords) {
        storage_core_noc_x.push_back((std::uint32_t)device->worker_core_from_logical_core(core).x);
        storage_core_noc_y.push_back((std::uint32_t)device->worker_core_from_logical_core(core).y);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    std::uint32_t pre_all_gather_stats_block_tiles = rms_norm ? 1 : 2;
    std::uint32_t post_all_gather_stats_block_tiles = 1;
    std::uint32_t num_distributed_devices = 1;
    if (is_post_all_gather && stats.has_value()) {
        post_all_gather_stats_block_tiles = stats.value().padded_shape()[-1] / tile_width;
        num_distributed_devices = post_all_gather_stats_block_tiles / pre_all_gather_stats_block_tiles;
    }

    // Reciprocal LUT for Welford
    std::optional<Tensor> recip_tensor = std::nullopt;
    uint32_t reciprocal_dfb_size_bytes = 0;
    if (use_welford) {
        TT_FATAL(tensor_args.recip_tensor.has_value(), "Reciprocal tensor not provided for Welford layernorm");
    }
    if (use_welford) {
        recip_tensor = tensor_args.recip_tensor;
        reciprocal_dfb_size_bytes = recip_tensor->buffer()->aligned_size_per_bank();
    }

    // Compute buffer sizes using helper
    DFBSizeParams dfb_size_params{
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
        .reciprocal_dfb_size_bytes = reciprocal_dfb_size_bytes,
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
    auto dfb_sizes = dfb_size_params.compute();

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

    // Pack eps for later use
    std::uint32_t eps_u = std::bit_cast<std::uint32_t>(eps);

    // Enumerate the shard grid as given, not core_ranges.all_cores: that is merge_ranges()'d, and merging
    // can re-partition a non-rectangular grid into different rectangles whose traversal order no longer
    // matches the tensor's shard order. Per-core index drives the gamma/beta offset, so a mismatch feeds
    // cores the wrong weight slice.
    const auto& shard_grid = grid.shard_spec.grid;
    const auto& cores = corerange_to_cores(shard_grid, shard_grid.num_cores(), grid.row_wise);

    std::uint32_t last_core_width_index =
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
    std::vector<std::uint32_t> mcast_noc_x, mcast_noc_y;
    mcast_noc_x.reserve(grid.grid_size.x);
    mcast_noc_y.reserve(grid.grid_size.y);
    CoreCoord core_start_offset = grid.grid_offset.value_or(CoreCoord{0, 0});
    for (std::uint32_t x = core_start_offset.x; x < grid.grid_size.x + core_start_offset.x; ++x) {
        mcast_noc_x.push_back(device->worker_core_from_logical_core({x, core_start_offset.y}).x);
    }
    for (std::uint32_t y = core_start_offset.y; y < grid.grid_size.y + core_start_offset.y; ++y) {
        mcast_noc_y.push_back(device->worker_core_from_logical_core({core_start_offset.x, y}).y);
    }

    // A non-tile-aligned width split across multiple cores is supported on every path. The non-Welford
    // path masks each core's final-tile padding columns with its per-core column mask. Welford
    // has no column mask, so each core is instead told its real (logical) column count (welford_reduce_w)
    // and reduces exactly those columns; full block_w on the cores before the last, the remaining logical
    // columns (ending in a partial tile) on the final real core; and the cross-core combine weights the
    // final block by its true width (last_block_w).
    // Legacy (non-Welford) path: zero the padding columns of a non-tile-aligned width's final tile so
    // they do not enter the statistics (E[x] and variance for layernorm, the mean of squares for
    // RMSNorm), except the post-all-gather stage, which reduces gathered stats rather than the input.
    // The mask is generated on-device in the writer (generate_mask_w<T>) keyed off each core's width
    // position, so it carries the correct validity whether the width lives on one core or is split
    // across many. A separate scratch buffer additionally feeds the non-distributed LayerNorm E[x] site
    // so the input buffer stays intact for the (x - E[x]) pass.
    const bool do_col_mask = col_mask_needed && !use_welford && !is_post_all_gather;
    const bool do_legacy_layernorm_col_mask = do_col_mask && !rms_norm && !is_pre_all_gather;

    ////////////////////////////////////////////////////////////////////////////
    //                      Spec configuration
    ////////////////////////////////////////////////////////////////////////////
    SpecConfig config{
        .reader_sender_path = kernel_paths.reader_sender,
        .reader_receiver_path = kernel_paths.reader_receiver,
        .writer_path = kernel_paths.writer,
        .compute_path = kernel_paths.compute,
        .is_pre_all_gather = is_pre_all_gather,
        .is_post_all_gather = is_post_all_gather,
        .rms_norm = rms_norm,
        .use_welford = use_welford,
        .has_b = b.has_value(),
        .has_gamma = gamma.has_value(),
        .has_beta = beta.has_value(),
        .skip_write_back = skip_write_back,
        .writes_back = writes_back,
        .do_col_mask = do_col_mask,
        .do_legacy_layernorm_col_mask = do_legacy_layernorm_col_mask,
        // Enable the welford-fp32 alias only when the SrcA-routed transpose_tile would
        // otherwise truncate Float32 input to TF32. Restricting to !rms_norm because
        // RMSNorm doesn't use Welford in this kernel path.
        .welford_fp32_alias = use_welford && !rms_norm && in_data_format == tt::DataFormat::Float32 && fp32_dest_acc_en,
        .sizes = dfb_sizes,
        .reciprocal_dfb_size_bytes = reciprocal_dfb_size_bytes,
        // The column mask is one tile-row wide: block_wt tiles, holding only 1.0 or 0.0 in bfloat16.
        .col_mask_gen_dfb_size_bytes = block_wt * bfloat16_tile_size,
        .in_data_format = in_data_format,
        .dfb_data_format = dfb_data_format,
        .out_data_format = out_data_format,
        .gamma_dfb_data_format = gamma_dfb_data_format,
        .beta_dfb_data_format = beta_dfb_data_format,
        .stats_dfb_data_format = stats_dfb_data_format,
        .reciprocal_dfb_data_format = reciprocal_dfb_data_format,
        .in_single_tile_size = in_single_tile_size,
        .single_tile_size = single_tile_size,
        .out_single_tile_size = out_single_tile_size,
        .gamma_single_tile_size = gamma_single_tile_size,
        .beta_single_tile_size = beta_single_tile_size,
        .stats_single_tile_size = stats_single_tile_size,
        .bfloat16_tile_size = bfloat16_tile_size,
        .block_ht = block_ht,
        .block_wt = block_wt,
        .subblock_wt = subblock_wt,
        .block_wt_resharded = block_wt_resharded,
        .K = K,
        .logical_K = logical_K,
        .tile_width = tile_width,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .legacy_reduction = legacy_reduction,
        .legacy_rsqrt = legacy_rsqrt,
        .eps = eps,
        .per_core_recip_lut_size = block_w,
        .reader_noc = reader_noc,
        .writer_noc = writer_noc,
        .compute_hw = to_compute_hardware_config(device->arch(), compute_kernel_config),
    };
    if (operation_attributes.fused_activation.has_value()) {
        const auto& act = operation_attributes.fused_activation.value();
        // The inner tile loop variable in the sharded compute kernels is "w" (dst register index).
        // Using "i" would refer to the outer block_h loop and apply the activation to the
        // wrong dst register.
        auto act_defines = ttnn::operations::unary::utils::get_defines(
            act.op_type, act.params, "ACTIVATION", "w", tensor_return_value.dtype());
        for (auto& [key, val] : act_defines) {
            config.activation_defines.emplace(key, val);
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Program spec and run args
    ////////////////////////////////////////////////////////////////////////////
    m2::ProgramSpec spec{.name = "layernorm_sharded"};
    add_dataflow_buffer_specs(spec, config);
    add_tensor_parameter_specs(spec, config, a, b, gamma, beta, stats, recip_tensor, output);

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
        .single_tile_size = single_tile_size,
        .out_single_tile_size = out_single_tile_size,
        .block_wt = block_wt,
        .block_wt_resharded = block_wt_resharded,
        .Kt = Kt,
        .logical_K = logical_K,
        .last_core_width_index = last_core_width_index,
        .is_post_all_gather = is_post_all_gather,
        .writes_back = writes_back,
        .num_distributed_devices = num_distributed_devices,
        .reader_noc = reader_noc,
        .storage_core_noc_x = std::move(storage_core_noc_x),
        .storage_core_noc_y = std::move(storage_core_noc_y),
        .num_storage_cores = (std::uint32_t)all_storage_cores.num_cores()};

    // The write-back segment block's length is measured per node while the run args are built, and
    // the kernel specs declare it, so the run args come first.
    auto [run_args, writer_num_varargs] =
        build_run_args(cores, rt_ctx, config, device, a, b, gamma, beta, stats, recip_tensor, output);
    add_kernel_and_work_unit_specs(spec, core_ranges, workers, grid, config, writer_num_varargs);

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim
