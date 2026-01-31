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

    // Determine if reader_receiver_all_to_all kernel is present
    bool has_reader_receiver_all_to_all =
        grid.use_mcast &&
        (workers.num_rows_per_all_to_all_worker_last != workers.num_rows_per_all_to_all_worker ||
         workers.num_none_all_to_all_workers > 0) &&
        workers.num_cores_all_to_all > 1;

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
    auto all_cores_vec = corerange_to_cores(grid.shard_spec.grid, grid.shard_spec.num_cores(), grid.row_wise);
    std::vector<CoreCoord> cores;
    std::vector<KernelHandle> writer_kernel_ids;
    cores.reserve(all_cores_vec.size());
    writer_kernel_ids.reserve(all_cores_vec.size());

    for (uint32_t i = 0; i < all_cores_vec.size(); ++i) {
        cores.push_back(all_cores_vec[i]);
        bool is_all_to_all_worker = (i < workers.num_cores_all_to_all);
        writer_kernel_ids.push_back(
            is_all_to_all_worker ? writer_mcast_sender_kernels_id : writer_mcast_receiver_kernels_id);
    }

    // Find CB handles by buffer index
    CBHandle cb_in0 = 0, cb_in1 = 0, cb_stats = 0, cb_add_out = 0, cb_output = 0;
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
    }

    return cached_program_t{
        std::move(program),
        shared_variables_t{
            .writer_kernel_ids = writer_kernel_ids,
            .writer_mcast_sender_kernels_id = writer_mcast_sender_kernels_id,
            .writer_mcast_receiver_kernels_id = writer_mcast_receiver_kernels_id,
            .num_none_all_to_all_workers = workers.num_none_all_to_all_workers,
            .is_pre_all_gather = is_pre_all_gather,
            .cb_in0 = cb_in0,
            .cb_in1 = cb_in1,
            .cb_stats = cb_stats,
            .cb_add_out = cb_add_out,
            .cb_output = cb_output,
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

    UpdateDynamicCircularBufferAddress(program, capture.cb_output, *dst_buffer);

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
    auto core_ranges = compute_core_ranges(grid, workers);

    uint32_t num_subblocks_w = block_wt / subblock_wt;

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
    // b, gamma, beta addr
    auto gamma_dram_addr = gamma.has_value() ? gamma.value().buffer()->address() : 0;
    auto beta_dram_addr = beta.has_value() ? beta.value().buffer()->address() : 0;

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t in0_block_tiles = block_wt * block_ht;
    // pre_all_gather_stats_block_tiles
    uint32_t pre_all_gather_stats_block_tiles = rms_norm ? 1 : 2;
    // post_all_gather_stats_block_tiles
    uint32_t post_all_gather_stats_block_tiles = 1;
    uint32_t num_distributed_devices = 1;
    if (is_post_all_gather && stats.has_value()) {
        post_all_gather_stats_block_tiles = stats.value().padded_shape()[-1] / TILE_WIDTH;
        num_distributed_devices = post_all_gather_stats_block_tiles / pre_all_gather_stats_block_tiles;
    }

    uint32_t in0_CB_tiles = in0_block_tiles;
    uint32_t in0_CB_size = in0_CB_tiles * in_single_tile_size;
    // block size for in1 (tensor b)
    uint32_t in1_CB_size = in0_CB_size;
    // in2 - scaler
    uint32_t in2_CB_size = bfloat16_tile_size;
    // in3 - eps
    uint32_t in3_CB_size = bfloat16_tile_size;
    // gamma
    uint32_t in5_CB_size = in0_block_tiles * gamma_single_tile_size / block_ht;
    // beta
    uint32_t in6_CB_size = in0_block_tiles * beta_single_tile_size / block_ht;
    // itermediate buffers change later
    uint32_t x_CB_size = in0_block_tiles * single_tile_size;
    uint32_t xmm_CB_size = in0_block_tiles * single_tile_size;
    uint32_t ex_partial_CB_size = in0_block_tiles * single_tile_size / block_wt;
    uint32_t ex_external_CB_size = tt::div_up(Kt, block_wt) * single_tile_size;
    if (is_pre_all_gather || is_post_all_gather) {
        ex_partial_CB_size = ex_partial_CB_size * pre_all_gather_stats_block_tiles;
    }
    uint32_t ex_CB_size = ex_partial_CB_size;
    uint32_t ex_global_CB_size = ex_partial_CB_size;
    uint32_t ex2pe_CB_size = workers.num_rows_per_all_to_all_worker * single_tile_size;
    uint32_t stats_cb_size = 0;
    uint32_t stats_reduced_cb_size = 0;
    if (is_post_all_gather) {
        stats_cb_size = post_all_gather_stats_block_tiles * single_tile_size;
        stats_reduced_cb_size = pre_all_gather_stats_block_tiles * single_tile_size;
    }
    // output buffer size
    uint32_t out_CB_size;
    if (is_pre_all_gather) {
        out_CB_size = pre_all_gather_stats_block_tiles * out_single_tile_size;
    } else {
        out_CB_size = in0_block_tiles * out_single_tile_size;
    }
    uint32_t out_reshard_CB_size = out_CB_size;
    if (is_post_all_gather && !skip_write_back) {
        out_reshard_CB_size = block_wt_resharded * block_ht * out_single_tile_size;
    }

    // Update CB sizes based on configuration
    if (grid.use_two_stage_reduce) {
        ex_external_CB_size = (workers.num_blocks_first_stage + workers.num_blocks_second_stage - 1) * single_tile_size;
    }
    if (is_pre_all_gather) {
        ex_external_CB_size = ex_external_CB_size * pre_all_gather_stats_block_tiles;
    }

    if (use_welford) {
        // Welford calculates 1 mean tile and 1 var tile per height tile
        ex_external_CB_size *= 2;
        ex_partial_CB_size *= 2;
        ex_CB_size *= 2;
        ex_global_CB_size *= 2;
    }

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

    // reader defines
    KernelDescriptor::Defines reader_mcast_sender_defines;
    KernelDescriptor::Defines reader_mcast_receiver_defines;
    if (b) {
        reader_mcast_sender_defines.emplace_back("FUSE_PRE_ADD", "1");
        reader_mcast_receiver_defines.emplace_back("FUSE_PRE_ADD", "1");
    }
    if (gamma.has_value()) {
        reader_mcast_sender_defines.emplace_back("FUSE_GAMMA", "1");
        reader_mcast_receiver_defines.emplace_back("FUSE_GAMMA", "1");
    }
    if (beta.has_value()) {
        reader_mcast_sender_defines.emplace_back("FUSE_BETA", "1");
        reader_mcast_receiver_defines.emplace_back("FUSE_BETA", "1");
    }

    // reader compile time args
    std::vector<uint32_t> reader_mcast_sender_compile_time_args = {
        (std::uint32_t)reduce_receiver_semaphore_id,
        (std::uint32_t)reduce_sender_semaphore_id,
        (std::uint32_t)grid.num_blocks,
        (std::uint32_t)block_ht,
        (std::uint32_t)block_ht * single_tile_size,
        (std::uint32_t)workers.num_cores_all_to_all_first_stage,
        (std::uint32_t)workers.num_rows_per_all_to_all_worker,
        (std::uint32_t)workers.num_rows_per_all_to_all_worker * single_tile_size,
        (std::uint32_t)workers.num_rows_per_all_to_all_worker_last,
        (std::uint32_t)workers.num_rows_per_all_to_all_worker_last * single_tile_size,
        (std::uint32_t)grid.row_wise,
        (std::uint32_t)core_ranges.num_cores_x_mcast,
        (std::uint32_t)core_ranges.num_cores_y_mcast,
        (std::uint32_t)grid.use_two_stage_reduce,
        (std::uint32_t)workers.num_blocks_first_stage,
        (std::uint32_t)workers.num_blocks_second_stage,
        (std::uint32_t)reduce_second_stage_semaphore_id,
        (std::uint32_t)rms_norm,
        (std::uint32_t)use_welford};
    std::vector<uint32_t> reader_mcast_receiver_all_to_all_compile_time_args = {
        (std::uint32_t)reduce_receiver_semaphore_id,
        (std::uint32_t)reduce_sender_semaphore_id,
        (std::uint32_t)grid.num_blocks,
        (std::uint32_t)block_ht,
        (std::uint32_t)1,
        (std::uint32_t)workers.num_cores_all_to_all_first_stage,
        (std::uint32_t)workers.num_rows_per_all_to_all_worker,
        (std::uint32_t)workers.num_rows_per_all_to_all_worker_last,
        (std::uint32_t)grid.row_wise,
        (std::uint32_t)core_ranges.num_cores_x_mcast,
        (std::uint32_t)core_ranges.num_cores_y_mcast,
        (std::uint32_t)grid.use_two_stage_reduce,
        (std::uint32_t)workers.num_blocks_first_stage,
        (std::uint32_t)workers.num_blocks_second_stage,
        (std::uint32_t)reduce_second_stage_semaphore_id,
        (std::uint32_t)rms_norm,
        (std::uint32_t)use_welford};
    std::vector<uint32_t> reader_mcast_receiver_compile_time_args = {
        (std::uint32_t)reduce_receiver_semaphore_id,
        (std::uint32_t)reduce_sender_semaphore_id,
        (std::uint32_t)grid.num_blocks,
        (std::uint32_t)block_ht,
        (std::uint32_t)0,
        (std::uint32_t)workers.num_cores_all_to_all_first_stage,
        (std::uint32_t)workers.num_rows_per_all_to_all_worker,
        (std::uint32_t)workers.num_rows_per_all_to_all_worker_last,
        (std::uint32_t)grid.row_wise,
        (std::uint32_t)1,
        (std::uint32_t)1,
        (std::uint32_t)0,
        (std::uint32_t)0,
        (std::uint32_t)0,
        (std::uint32_t)reduce_second_stage_semaphore_id,
        (std::uint32_t)rms_norm,
        (std::uint32_t)use_welford};

    tt::tt_metal::NOC reader_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());
    tt::tt_metal::NOC writer_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());

    if (is_post_all_gather && !skip_write_back) {
        reader_noc = NOC::NOC_0;
        writer_noc = NOC::NOC_1;
    }

    // reader kernel paths
    std::string sender_reader_kernel_file =
        "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/"
        "reader_mcast_sender_unary_sharded_ln.cpp";
    std::string receiver_reader_kernel_file =
        "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/"
        "reader_mcast_receiver_unary_sharded_ln.cpp";

    if (is_pre_all_gather) {
        sender_reader_kernel_file =
            "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/"
            "reader_mcast_sender_unary_sharded_ln_pre_allgather.cpp";
        receiver_reader_kernel_file =
            "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/"
            "reader_mcast_receiver_unary_sharded_ln_pre_allgather.cpp";
    } else if (is_post_all_gather) {
        sender_reader_kernel_file =
            "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/"
            "reader_mcast_sender_unary_sharded_ln_post_allgather.cpp";
        receiver_reader_kernel_file =
            "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/"
            "reader_mcast_receiver_unary_sharded_ln_post_allgather.cpp";
    }

    // writer defines
    KernelDescriptor::Defines writer_defines;
    if (rms_norm) {
        writer_defines.emplace_back("RMSNORM", "1");
    }
    if (skip_write_back) {
        writer_defines.emplace_back("SKIP_WRITE_BACK", "1");
    }

    // writer compile time args
    std::vector<uint32_t> writer_mcast_sender_compile_time_args = {
        1,  // is_all_to_all_worker
        (std::uint32_t)gamma.has_value(),
        (std::uint32_t)beta.has_value(),
        (std::uint32_t)block_wt,
        (std::uint32_t)use_welford};
    tt::tt_metal::TensorAccessorArgs(gamma ? gamma->buffer() : nullptr)
        .append_to(writer_mcast_sender_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(beta ? beta->buffer() : nullptr).append_to(writer_mcast_sender_compile_time_args);

    std::vector<uint32_t> writer_mcast_receiver_compile_time_args = {
        0,  // is_all_to_all_worker
        (std::uint32_t)gamma.has_value(),
        (std::uint32_t)beta.has_value(),
        (std::uint32_t)block_wt,
        (std::uint32_t)use_welford};
    tt::tt_metal::TensorAccessorArgs(gamma ? gamma->buffer() : nullptr)
        .append_to(writer_mcast_receiver_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(beta ? beta->buffer() : nullptr)
        .append_to(writer_mcast_receiver_compile_time_args);

    if (gamma.has_value() and gamma.value().layout() == Layout::ROW_MAJOR) {
        auto gamma_stick_size = gamma.value().padded_shape()[-1] * gamma.value().element_size();
        writer_mcast_sender_compile_time_args.push_back(gamma_stick_size);
        writer_mcast_receiver_compile_time_args.push_back(gamma_stick_size);
    } else if (beta.has_value() and beta.value().layout() == Layout::ROW_MAJOR) {
        auto beta_stick_size = beta.value().padded_shape()[-1] * beta.value().element_size();
        writer_mcast_sender_compile_time_args.push_back(beta_stick_size);
        writer_mcast_receiver_compile_time_args.push_back(beta_stick_size);
    }
    writer_mcast_sender_compile_time_args.push_back(gamma_cb_data_format == tt::DataFormat::Float32);
    writer_mcast_sender_compile_time_args.push_back(beta_cb_data_format == tt::DataFormat::Float32);
    writer_mcast_receiver_compile_time_args.push_back(gamma_cb_data_format == tt::DataFormat::Float32);
    writer_mcast_receiver_compile_time_args.push_back(beta_cb_data_format == tt::DataFormat::Float32);

    // write back compile time args
    writer_mcast_sender_compile_time_args.push_back(block_wt * out_single_tile_size);
    writer_mcast_sender_compile_time_args.push_back(block_wt_resharded * out_single_tile_size);
    writer_mcast_sender_compile_time_args.push_back(block_ht);

    writer_mcast_receiver_compile_time_args.push_back(block_wt * out_single_tile_size);
    writer_mcast_receiver_compile_time_args.push_back(block_wt_resharded * out_single_tile_size);
    writer_mcast_receiver_compile_time_args.push_back(block_ht);
    writer_mcast_receiver_compile_time_args.push_back(use_welford);

    // writer kernel path
    bool use_row_major_kernel = (gamma.has_value() and gamma.value().layout() == Layout::ROW_MAJOR) or
                                (beta.has_value() and beta.value().layout() == Layout::ROW_MAJOR);
    std::string writer_kernel;
    if (is_pre_all_gather) {
        writer_kernel =
            "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/"
            "writer_unary_sharded_ln_pre_all_gather.cpp";
    } else {
        writer_kernel = use_row_major_kernel ? "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/"
                                               "dataflow/writer_unary_sharded_ln_rm_gb.cpp"
                                             : "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/"
                                               "dataflow/writer_unary_sharded_ln.cpp";
    }

    // compute defines
    KernelDescriptor::Defines compute_defines;
    if (b) {
        compute_defines.emplace_back("FUSE_PRE_ADD", "1");
    }
    if (rms_norm && !use_welford) {
        compute_defines.emplace_back("RMSNORM", "1");
    }

    // Reciprocal LUT
    uint32_t per_core_recip_lut_size = block_w;
    std::optional<Tensor> recip_tensor = std::nullopt;
    uint32_t reciprocal_CB_size_bytes = 0;
    if (use_welford) {
        TT_FATAL(tensor_args.recip_tensor.has_value(), "Reciprocal tensor not provided for Welford layernorm");
        recip_tensor = tensor_args.recip_tensor;
        reciprocal_CB_size_bytes = recip_tensor->buffer()->aligned_size_per_bank();
    }

    // compute kernel compile time args
    bool float32_reduction = fp32_dest_acc_en && !legacy_reduction;
    std::vector<uint32_t> all_to_all_except_top_compute_compile_time_args = {
        0,
        gamma.has_value(),
        beta.has_value(),
        workers.num_blocks_first_stage,
        block_ht,
        block_wt,
        subblock_wt,
        num_subblocks_w,
        1,
        block_ht * block_wt,
        fp32_dest_acc_en,
        float32_reduction,
        legacy_rsqrt,
        workers.num_blocks_second_stage};
    std::vector<uint32_t> not_all_to_all_compute_compile_time_args = {
        0,
        gamma.has_value(),
        beta.has_value(),
        workers.num_blocks_first_stage,
        block_ht,
        block_wt,
        subblock_wt,
        num_subblocks_w,
        0,
        block_ht * block_wt,
        fp32_dest_acc_en,
        float32_reduction,
        legacy_rsqrt,
        workers.num_blocks_second_stage};

    constexpr uint32_t tile_width = tt::constants::TILE_WIDTH;
    uint32_t last_tile_W = K - ((K - tile_width) / tile_width) * tile_width;
    union {
        float f;
        uint32_t u;
    } e{};
    e.f = eps;
    if (use_welford) {
        all_to_all_except_top_compute_compile_time_args.push_back(tile_width);
        all_to_all_except_top_compute_compile_time_args.push_back(last_tile_W);
        all_to_all_except_top_compute_compile_time_args.push_back(K);
        all_to_all_except_top_compute_compile_time_args.push_back(e.u);
        all_to_all_except_top_compute_compile_time_args.push_back(per_core_recip_lut_size);
        not_all_to_all_compute_compile_time_args.push_back(tile_width);
        not_all_to_all_compute_compile_time_args.push_back(last_tile_W);
        not_all_to_all_compute_compile_time_args.push_back(K);
        not_all_to_all_compute_compile_time_args.push_back(e.u);
        not_all_to_all_compute_compile_time_args.push_back(per_core_recip_lut_size);
    }

    // compute kernel path
    std::string compute_kernel_file;
    if (is_pre_all_gather) {
        compute_kernel_file =
            "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/"
            "layernorm_sharded_pre_allgather.cpp";
    } else if (is_post_all_gather) {
        compute_kernel_file =
            "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/"
            "layernorm_sharded_post_allgather.cpp";
    } else {
        compute_kernel_file =
            use_welford
                ? "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/"
                  "layernorm_sharded_welford.cpp"
                : "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_sharded.cpp";
    }

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

    auto runtime_args = build_all_runtime_args(cores, rt_ctx, device);

    ////////////////////////////////////////////////////////////////////////////
    //                      Build Kernel Descriptors
    ////////////////////////////////////////////////////////////////////////////
    KernelConfig kernel_config;
    kernel_config.reader_sender_path = sender_reader_kernel_file;
    kernel_config.reader_receiver_path = receiver_reader_kernel_file;
    kernel_config.writer_path = writer_kernel;
    kernel_config.compute_path = compute_kernel_file;
    kernel_config.reader_sender_ct_args = std::move(reader_mcast_sender_compile_time_args);
    kernel_config.reader_receiver_all_to_all_ct_args = std::move(reader_mcast_receiver_all_to_all_compile_time_args);
    kernel_config.reader_receiver_ct_args = std::move(reader_mcast_receiver_compile_time_args);
    kernel_config.writer_sender_ct_args = std::move(writer_mcast_sender_compile_time_args);
    kernel_config.writer_receiver_ct_args = std::move(writer_mcast_receiver_compile_time_args);
    kernel_config.compute_all_to_all_ct_args = std::move(all_to_all_except_top_compute_compile_time_args);
    kernel_config.compute_not_all_to_all_ct_args = std::move(not_all_to_all_compute_compile_time_args);
    kernel_config.reader_sender_defines = std::move(reader_mcast_sender_defines);
    kernel_config.reader_receiver_defines = std::move(reader_mcast_receiver_defines);
    kernel_config.writer_defines = std::move(writer_defines);
    kernel_config.compute_defines = std::move(compute_defines);
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
    Buffer* output_buffer = nullptr;
    if (!is_post_all_gather || skip_write_back) {
        output_buffer = output.buffer();
    }

    CBConfig cb_config;
    cb_config.in0_CB_size = in0_CB_size;
    cb_config.in1_CB_size = in1_CB_size;
    cb_config.in2_CB_size = in2_CB_size;
    cb_config.in3_CB_size = in3_CB_size;
    cb_config.in5_CB_size = in5_CB_size;
    cb_config.in6_CB_size = in6_CB_size;
    cb_config.x_CB_size = x_CB_size;
    cb_config.xmm_CB_size = xmm_CB_size;
    cb_config.ex_partial_CB_size = ex_partial_CB_size;
    cb_config.ex_CB_size = ex_CB_size;
    cb_config.ex_external_CB_size = ex_external_CB_size;
    cb_config.ex_global_CB_size = ex_global_CB_size;
    cb_config.ex2pe_CB_size = ex2pe_CB_size;
    cb_config.out_CB_size = out_CB_size;
    cb_config.out_reshard_CB_size = out_reshard_CB_size;
    cb_config.stats_cb_size = stats_cb_size;
    cb_config.stats_reduced_cb_size = stats_reduced_cb_size;
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
