// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "ttnn/operations/normalization/layernorm/device/layernorm_op_multi_core_sharded.hpp"
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

LayerNormShardedProgramFactory::cached_program_t LayerNormShardedProgramFactory::create(
    const LayerNormParams& operation_attributes, const LayerNormInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace sharded_layernorm_helpers;

    // Get program descriptor and create program
    ProgramDescriptor program_descriptor = create_descriptor(operation_attributes, tensor_args, tensor_return_value);
    Program program{program_descriptor};

    // Extract needed values
    const auto& a = tensor_args.input;
    bool is_pre_all_gather = operation_attributes.distributed_norm_stage == DistributedLayerNormStage::PRE_ALL_GATHER;

    // Extract block_ht from program config
    uint32_t block_ht = 0;
    std::visit(
        [&](const auto& program_config) {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (std::is_same_v<ProgramConfigType, LayerNormShardedMultiCoreProgramConfig>) {
                block_ht = program_config.block_h;
            }
        },
        operation_attributes.program_config);

    // Compute grid and worker distribution
    auto grid = GridParams::compute(a, block_ht, a.device()->compute_with_storage_grid_size());
    auto workers = WorkerDistribution::compute(grid, block_ht);
    auto core_ranges = CoreRanges::compute(grid, workers);

    // Determine if reader_receiver_all_to_all kernel is present
    // This must match add_kernel_descriptors(): grid.use_mcast && !core_ranges.all_to_all_workers_except_sender.empty()
    bool has_reader_receiver_all_to_all = grid.use_mcast && !core_ranges.all_to_all_workers_except_sender.empty();

    // Calculate kernel handle indices based on conditional kernel ordering:
    // 0: reader_sender (always)
    // 1: reader_receiver_all_to_all (if has_reader_receiver_all_to_all)
    // next: reader_receiver (if num_none_all_to_all_workers > 0)
    // next: writer_sender (always)
    // next: writer_receiver (if num_none_all_to_all_workers > 0)
    KernelHandle kernel_idx = 1;  // Start after reader_sender (index 0)
    if (has_reader_receiver_all_to_all) {
        kernel_idx++;
    }
    if (workers.num_none_all_to_all_workers > 0) {
        kernel_idx++;
    }
    KernelHandle writer_mcast_sender_kernels_id = kernel_idx++;
    KernelHandle writer_mcast_receiver_kernels_id = 0;
    if (workers.num_none_all_to_all_workers > 0) {
        writer_mcast_receiver_kernels_id = kernel_idx++;
    }

    // Build cores vector and writer_kernel_ids
    // Must use the same core set as in create_descriptor() to match kernel assignments
    auto all_cores_vec = corerange_to_cores(core_ranges.all_cores, core_ranges.all_cores.num_cores(), grid.row_wise);
    std::vector<CoreCoord> cores;
    std::vector<KernelHandle> writer_kernel_ids;
    cores.reserve(all_cores_vec.size());
    writer_kernel_ids.reserve(all_cores_vec.size());

    for (uint32_t i = 0; i < all_cores_vec.size(); ++i) {
        const CoreCoord& core = all_cores_vec[i];
        cores.push_back(core);
        // Determine if this core is an all-to-all worker by checking if it's in the all_to_all_cores range
        bool is_all_to_all_worker = core_ranges.all_to_all_cores.contains(core);
        writer_kernel_ids.push_back(
            is_all_to_all_worker ? writer_mcast_sender_kernels_id : writer_mcast_receiver_kernels_id);
    }

    // Determine if resharding is being used (CB 17 has the output buffer)
    // Resharding happens when: is_post_all_gather AND output shard spec differs from input
    bool uses_reshard = !is_pre_all_gather && !(tensor_return_value.shard_spec().value() == a.shard_spec().value());

    // Find CB handles by buffer index
    CBHandle cb_in0 = 0, cb_in1 = 0, cb_stats = 0, cb_add_out = 0, cb_output = 0, cb_output_reshard = 0;
    for (const auto& cb : program.circular_buffers()) {
        const auto& indices = cb->buffer_indices();
        if (indices.contains(tt::CBIndex::c_0)) {
            cb_in0 = cb->id();
        }
        if (indices.contains(tt::CBIndex::c_1)) {
            cb_in1 = cb->id();
        }
        if (indices.contains(tt::CBIndex::c_7)) {
            cb_stats = cb->id();
        }
        if (indices.contains(tt::CBIndex::c_14)) {
            cb_add_out = cb->id();
        }
        if (indices.contains(tt::CBIndex::c_16)) {
            cb_output = cb->id();
        }
        if (indices.contains(tt::CBIndex::c_17)) {
            cb_output_reshard = cb->id();
        }
    }

    return cached_program_t{
        std::move(program),
        shared_variables_t{
            .writer_kernel_ids = writer_kernel_ids,
            .writer_mcast_sender_kernels_id = writer_mcast_sender_kernels_id,
            .writer_mcast_receiver_kernels_id = writer_mcast_receiver_kernels_id,
            .num_none_all_to_all_workers = workers.num_none_all_to_all_workers,
            .is_pre_all_gather = is_pre_all_gather,
            .uses_reshard = uses_reshard,
            .cb_in0 = cb_in0,
            .cb_in1 = cb_in1,
            .cb_stats = cb_stats,
            .cb_add_out = cb_add_out,
            .cb_output = cb_output,
            .cb_output_reshard = cb_output_reshard,
            .cores = cores}};
}

void LayerNormShardedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const LayerNormParams& /*operation_attributes*/,
    const LayerNormInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto* const src_buffer_a = tensor_args.input.buffer();
    const auto& b_tensor = tensor_args.residual_input_tensor;
    const auto& gamma_tensor = tensor_args.weight;
    const auto& beta_tensor = tensor_args.bias;
    const auto& stats_tensor = tensor_args.stats;
    auto* const dst_buffer = tensor_return_value.buffer();

    const auto& capture = cached_program.shared_variables;
    auto& program = cached_program.program;

    UpdateDynamicCircularBufferAddress(program, capture.cb_in0, *src_buffer_a);

    if (b_tensor.has_value()) {
        UpdateDynamicCircularBufferAddress(program, capture.cb_in1, *b_tensor.value().buffer());
        if (capture.is_pre_all_gather) {
            UpdateDynamicCircularBufferAddress(program, capture.cb_add_out, *src_buffer_a);
        }
    }
    if (stats_tensor.has_value()) {
        UpdateDynamicCircularBufferAddress(program, capture.cb_stats, *stats_tensor.value().buffer());
    }

    // Update the correct output CB based on whether resharding is used
    // When resharding: CB 17 has the output buffer; When not resharding: CB 16 has the output buffer
    if (capture.uses_reshard) {
        UpdateDynamicCircularBufferAddress(program, capture.cb_output_reshard, *dst_buffer);
    } else {
        UpdateDynamicCircularBufferAddress(program, capture.cb_output, *dst_buffer);
    }

    auto& writer_sender_args_by_core = GetRuntimeArgs(program, capture.writer_mcast_sender_kernels_id);
    auto& writer_receiver_args_by_core = capture.num_none_all_to_all_workers > 0
                                             ? GetRuntimeArgs(program, capture.writer_mcast_receiver_kernels_id)
                                             : writer_sender_args_by_core;

    const auto gamma_address = gamma_tensor.has_value() ? gamma_tensor.value().buffer()->address() : 0;
    const auto beta_address = beta_tensor.has_value() ? beta_tensor.value().buffer()->address() : 0;

    for (uint32_t i = 0; i < capture.cores.size(); ++i) {
        const CoreCoord& core = capture.cores[i];
        const auto writer_kernel_id = capture.writer_kernel_ids.at(i);

        if (writer_kernel_id == capture.writer_mcast_sender_kernels_id) {
            auto& runtime_args = writer_sender_args_by_core[core.x][core.y];
            runtime_args[3] = gamma_address;
            runtime_args[4] = beta_address;

        } else if (writer_kernel_id == capture.writer_mcast_receiver_kernels_id) {
            auto& runtime_args = writer_receiver_args_by_core[core.x][core.y];
            runtime_args[3] = gamma_address;
            runtime_args[4] = beta_address;
        }
    }
}

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

    uint32_t block_wt_resharded = output.shard_spec().value().shape[1] / TILE_WIDTH;
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

    auto [out_data_format, cb_data_format, gamma_cb_data_format, beta_cb_data_format, reciprocal_cb_data_format] =
        get_cb_data_formats(output, gamma, beta, fp32_dest_acc_en);

    // tile sizes
    uint32_t in_single_tile_size = tt::tile_size(in_data_format);
    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    uint32_t out_single_tile_size = tt::tile_size(out_data_format);
    uint32_t gamma_single_tile_size = tt::tile_size(gamma_cb_data_format);
    uint32_t beta_single_tile_size = tt::tile_size(beta_cb_data_format);
    uint32_t bfloat16_tile_size = tt::tile_size(tt::DataFormat::Float16_b);

    // tensor shape
    const auto& shape = a.padded_shape();
    uint32_t K = shape[-1];
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t block_w = block_wt * TILE_WIDTH;

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
        post_all_gather_stats_block_tiles = stats.value().padded_shape()[-1] / TILE_WIDTH;
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
        .skip_write_back = skip_write_back};
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
        b.has_value(), gamma.has_value(), beta.has_value(), rms_norm, use_welford, skip_write_back);

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
        .per_core_recip_lut_size = block_w};
    auto compile_time_args = CompileTimeArgs::build(ct_ctx);

    // Pack eps for later use
    union {
        float f;
        uint32_t u;
    } e{};
    e.f = eps;

    // Build runtime args using helper
    const auto& cores = corerange_to_cores(core_ranges.all_cores, core_ranges.all_cores.num_cores(), grid.row_wise);

    // Compute packed values for writer
    float winv = 1.0f / block_w;
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

    uint32_t last_core_width_index =
        grid.mcast_1d ? cores.size() - 1 : (grid.row_wise ? grid.grid_size.x - 1 : grid.grid_size.y - 1);

    RuntimeArgsContext rt_ctx{
        .grid = grid,
        .workers = workers,
        .core_ranges = core_ranges,
        .mcast_noc_x = std::move(mcast_noc_x),
        .mcast_noc_y = std::move(mcast_noc_y),
        .packed_cinv_value = pack_two_bfloat16_into_uint32({bfloat_cinv, bfloat_cinv}),
        .packed_cinv_value_one = pack_two_bfloat16_into_uint32({bfloat_cinv_one, bfloat_cinv_one}),
        .packed_winv_value = pack_two_bfloat16_into_uint32({bfloat_winv, bfloat_winv}),
        .eps_u = e.u,
        .gamma_dram_addr = gamma_dram_addr,
        .beta_dram_addr = beta_dram_addr,
        .single_tile_size = single_tile_size,
        .out_single_tile_size = out_single_tile_size,
        .block_wt = block_wt,
        .block_wt_resharded = block_wt_resharded,
        .Kt = Kt,
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
    kernel_config.math_approx_mode = math_approx_mode;

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
    cb_config.reciprocal_cb_data_format = reciprocal_cb_data_format;
    cb_config.in_single_tile_size = in_single_tile_size;
    cb_config.single_tile_size = single_tile_size;
    cb_config.out_single_tile_size = out_single_tile_size;
    cb_config.gamma_single_tile_size = gamma_single_tile_size;
    cb_config.beta_single_tile_size = beta_single_tile_size;
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

    add_cb_descriptors(program_descriptor, core_ranges, all_worker_and_storage_cores, cb_config);

    return program_descriptor;
}

}  // namespace ttnn::prim
