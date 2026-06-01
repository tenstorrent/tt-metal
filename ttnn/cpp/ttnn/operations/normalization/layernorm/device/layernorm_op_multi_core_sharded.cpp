// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "ttnn/operations/normalization/layernorm/device/layernorm_device_operation.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_common.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_device_operation_types.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/normalization/layernorm/device/sharded_layernorm_factory_helpers.hpp"

#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_params.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>

#include <optional>
#include <bit>

using uint32_t = std::uint32_t;
using namespace tt::constants;
using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental::metal2_host_api;

namespace ttnn::prim {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE_SH {

using namespace sharded_layernorm_helpers;

// Kernel / DFB / TensorParameter / Semaphore unique_ids.
constexpr const char* K_READER_SENDER = "reader_sender";
constexpr const char* K_READER_RCV_A2A = "reader_receiver_all_to_all";
constexpr const char* K_READER_RCV = "reader_receiver";
constexpr const char* K_WRITER_SENDER = "writer_sender";
constexpr const char* K_WRITER_RCV = "writer_receiver";
constexpr const char* K_COMPUTE_A2A = "compute_all_to_all";
constexpr const char* K_COMPUTE_NOT_A2A = "compute_not_all_to_all";

constexpr const char* WU_SENDER = "wu_sender";
constexpr const char* WU_A2A_RCV = "wu_a2a_rcv";
constexpr const char* WU_NOT_A2A = "wu_not_a2a";
constexpr const char* WU_ALL_TO_ALL = "wu_all_to_all";

// DFB names (mapped to legacy CB indices in the old code)
constexpr const char* DFB_IN0 = "cb_in0";                      // c_0  — sharded input
constexpr const char* DFB_INB = "cb_inb";                      // c_1  — sharded residual
constexpr const char* DFB_SCALER = "cb_scaler";                // c_2
constexpr const char* DFB_EPS = "cb_eps";                      // c_3
constexpr const char* DFB_SCALER_GLOBAL = "cb_scaler_global";  // c_4
constexpr const char* DFB_GAMMA = "cb_gamma";                  // c_5
constexpr const char* DFB_BETA = "cb_beta";                    // c_6
constexpr const char* DFB_STATS = "cb_stats";                  // c_7
constexpr const char* DFB_EX_PARTIAL = "cb_ex_partial";        // c_8
constexpr const char* DFB_EX = "cb_ex";                        // c_9
constexpr const char* DFB_EX_EXTERNAL = "cb_ex_external";      // c_10
constexpr const char* DFB_EX_PARTIAL2 = "cb_ex_partial2";      // c_11
constexpr const char* DFB_EX2 = "cb_ex2";                      // c_12
constexpr const char* DFB_EX_EXTERNAL2 = "cb_ex_external2";    // c_13
constexpr const char* DFB_IN0_PRE = "cb_in0_pre";              // c_14 — alias for in0 in pre-allgather
constexpr const char* DFB_EX_GLOBAL = "cb_ex_global";          // c_15
constexpr const char* DFB_OUT = "cb_out";                      // c_16
constexpr const char* DFB_OUT_RESHARD = "cb_out_resharded";    // c_17
constexpr const char* DFB_XMM = "cb_xmm";                      // c_18
constexpr const char* DFB_VAR = "cb_var";                      // c_19
constexpr const char* DFB_EX2PE = "cb_ex2pe";                  // c_20
constexpr const char* DFB_STATS_REDUCED = "cb_stats_reduced";  // c_21
constexpr const char* DFB_TRANSPOSE = "cb_transpose";          // c_22
constexpr const char* DFB_X = "cb_x";                          // c_24
constexpr const char* DFB_RECIPROCALS = "cb_reciprocals";      // c_25

constexpr const char* TP_INPUT_A = "input_a";
constexpr const char* TP_RESIDUAL_B = "residual_b";
constexpr const char* TP_GAMMA = "gamma";
constexpr const char* TP_BETA = "beta";
constexpr const char* TP_OUTPUT = "output";
constexpr const char* TP_STATS = "stats";
constexpr const char* TP_RECIP = "recip";

constexpr const char* SEM_REDUCE_SENDER = "reduce_sender";
constexpr const char* SEM_REDUCE_RECEIVER = "reduce_receiver";
constexpr const char* SEM_REDUCE_SECOND_STAGE = "reduce_second_stage";

m2::KernelSpec::DFBBinding ConsumerDFB(const char* dfb_name, const char* accessor_name) {
    return m2::KernelSpec::DFBBinding{
        .dfb_spec_name = dfb_name,
        .local_accessor_name = accessor_name,
        .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER};
}
m2::KernelSpec::DFBBinding ProducerDFB(const char* dfb_name, const char* accessor_name) {
    return m2::KernelSpec::DFBBinding{
        .dfb_spec_name = dfb_name,
        .local_accessor_name = accessor_name,
        .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER};
}

m2::DataflowBufferSpec MakeDFB(
    const char* unique_id,
    uint32_t entry_size,
    uint32_t num_entries,
    tt::DataFormat data_format,
    std::optional<m2::TensorParameterName> borrowed_from = std::nullopt) {
    m2::DataflowBufferSpec dfb{
        .unique_id = unique_id,
        .entry_size = entry_size,
        .num_entries = num_entries,
        .data_format_metadata = data_format};
    if (borrowed_from.has_value()) {
        dfb.borrowed_from = *borrowed_from;
    }
    return dfb;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE_SH
}  // namespace

ttnn::device_operation::ProgramArtifacts LayerNormShardedProgramFactory::create_program_spec(
    const LayerNormParams& operation_attributes, const LayerNormInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace CMAKE_UNIQUE_NAMESPACE_SH;
    using namespace sharded_layernorm_helpers;

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

    TT_FATAL(a.shard_spec().has_value(), "Sharded layernorm requires input tensor to have a shard spec");

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

    IDevice* device = a.device();
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

    uint32_t in_single_tile_size = tt::tile_size(in_data_format);
    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    uint32_t out_single_tile_size = tt::tile_size(out_data_format);
    uint32_t gamma_single_tile_size = tt::tile_size(gamma_cb_data_format);
    uint32_t beta_single_tile_size = tt::tile_size(beta_cb_data_format);
    uint32_t stats_single_tile_size = tt::tile_size(stats_cb_data_format);
    uint32_t bfloat16_tile_size = tt::tile_size(tt::DataFormat::Float16_b);

    const auto& shape = a.padded_shape();
    uint32_t K = shape[-1];
    uint32_t Kt = K / tile_width;
    uint32_t block_w = block_wt * tile_width;

    auto grid = GridParams::compute(a, block_ht, device->compute_with_storage_grid_size());
    auto workers = WorkerDistribution::compute(grid, block_ht);
    auto core_ranges = CoreRanges::compute(grid, workers);

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

    auto gamma_dram_addr = gamma.has_value() ? gamma.value().buffer()->address() : 0;
    auto beta_dram_addr = beta.has_value() ? beta.value().buffer()->address() : 0;

    uint32_t pre_all_gather_stats_block_tiles = rms_norm ? 1 : 2;
    uint32_t post_all_gather_stats_block_tiles = 1;
    uint32_t num_distributed_devices = 1;
    if (is_post_all_gather && stats.has_value()) {
        post_all_gather_stats_block_tiles = stats.value().padded_shape()[-1] / tile_width;
        num_distributed_devices = post_all_gather_stats_block_tiles / pre_all_gather_stats_block_tiles;
    }

    uint32_t reciprocal_CB_size_bytes = 0;
    if (use_welford) {
        TT_FATAL(tensor_args.recip_tensor.has_value(), "Reciprocal tensor not provided for Welford layernorm");
        reciprocal_CB_size_bytes = tensor_args.recip_tensor->buffer()->aligned_size_per_bank();
    }

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

    // NOC selection
    tt::tt_metal::NOC reader_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());
    tt::tt_metal::NOC writer_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    if (is_post_all_gather && !skip_write_back) {
        reader_noc = NOC::NOC_0;
        writer_noc = NOC::NOC_1;
    }

    bool use_row_major_kernel = (gamma.has_value() && gamma.value().layout() == Layout::ROW_MAJOR) ||
                                (beta.has_value() && beta.value().layout() == Layout::ROW_MAJOR);
    auto kernel_paths = KernelPaths::get(is_pre_all_gather, is_post_all_gather, use_row_major_kernel, use_welford);
    auto kernel_defines = KernelDefines::build(
        b.has_value(),
        gamma.has_value(),
        beta.has_value(),
        rms_norm,
        use_welford,
        skip_write_back,
        operation_attributes.fused_activation,
        tensor_return_value.dtype());

    // Whether the receiver-side all-to-all reader kernel exists at all
    const bool has_reader_receiver_all_to_all = grid.use_mcast && !core_ranges.all_to_all_workers_except_sender.empty();
    const bool has_not_all_to_all_workers = workers.num_none_all_to_all_workers > 0;
    const bool float32_reduction = fp32_dest_acc_en && !legacy_reduction;
    const uint32_t num_subblocks_w = block_wt / subblock_wt;

    ////////////////////////////////////////////////////////////////////////////
    // Buffer-backed DFBs
    ////////////////////////////////////////////////////////////////////////////
    std::vector<m2::DataflowBufferSpec> dfbs;

    // CB 0: in0 sharded → DFB borrowed from input tensor
    dfbs.push_back(MakeDFB(
        DFB_IN0,
        in_single_tile_size,
        cb_sizes.in0_CB_size / in_single_tile_size,
        in_data_format,
        m2::TensorParameterName{TP_INPUT_A}));

    // CB 1: in1 sharded (if has_b)
    if (b.has_value()) {
        dfbs.push_back(MakeDFB(
            DFB_INB,
            in_single_tile_size,
            cb_sizes.in1_CB_size / in_single_tile_size,
            in_data_format,
            m2::TensorParameterName{TP_RESIDUAL_B}));
        if (is_pre_all_gather) {
            // CB 14: alias for in0 in pre-allgather mode
            dfbs.push_back(MakeDFB(
                DFB_IN0_PRE,
                in_single_tile_size,
                cb_sizes.in1_CB_size / in_single_tile_size,
                in_data_format,
                m2::TensorParameterName{TP_INPUT_A}));
        }
    }

    // CB 5: gamma (if gamma)
    if (gamma.has_value()) {
        dfbs.push_back(MakeDFB(
            DFB_GAMMA, gamma_single_tile_size, cb_sizes.in5_CB_size / gamma_single_tile_size, gamma_cb_data_format));
    }
    // CB 6: beta (if beta)
    if (beta.has_value()) {
        dfbs.push_back(MakeDFB(
            DFB_BETA, beta_single_tile_size, cb_sizes.in6_CB_size / beta_single_tile_size, beta_cb_data_format));
    }

    // CB 24: x
    dfbs.push_back(MakeDFB(DFB_X, single_tile_size, cb_sizes.x_CB_size / single_tile_size, cb_data_format));
    // CB 18: xmm
    dfbs.push_back(MakeDFB(DFB_XMM, single_tile_size, cb_sizes.xmm_CB_size / single_tile_size, cb_data_format));

    // ex_partial, ex, ex_external (if not rms_norm)
    if (!rms_norm) {
        dfbs.push_back(
            MakeDFB(DFB_EX_PARTIAL, single_tile_size, cb_sizes.ex_partial_CB_size / single_tile_size, cb_data_format));
        dfbs.push_back(MakeDFB(DFB_EX, single_tile_size, cb_sizes.ex_CB_size / single_tile_size, cb_data_format));
        dfbs.push_back(MakeDFB(
            DFB_EX_EXTERNAL, single_tile_size, cb_sizes.ex_external_CB_size / single_tile_size, cb_data_format));
    }

    if (!use_welford) {
        // CB 2: in2 scaler
        dfbs.push_back(MakeDFB(
            DFB_SCALER, bfloat16_tile_size, cb_sizes.in2_CB_size / bfloat16_tile_size, tt::DataFormat::Float16_b));
        // CB 3: in3 eps
        dfbs.push_back(
            MakeDFB(DFB_EPS, bfloat16_tile_size, cb_sizes.in3_CB_size / bfloat16_tile_size, tt::DataFormat::Float16_b));
        // CB 4: in4 scaler-c
        tt::DataFormat scaler_global_format =
            cb_data_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
        uint32_t scaler_global_tile_size = tt::tile_size(scaler_global_format);
        dfbs.push_back(MakeDFB(DFB_SCALER_GLOBAL, scaler_global_tile_size, 1, scaler_global_format));
        // CB 11: ex_partial2
        dfbs.push_back(
            MakeDFB(DFB_EX_PARTIAL2, single_tile_size, cb_sizes.ex_partial_CB_size / single_tile_size, cb_data_format));
        // CB 12: ex2
        dfbs.push_back(MakeDFB(DFB_EX2, single_tile_size, cb_sizes.ex_CB_size / single_tile_size, cb_data_format));
        // CB 13: ex_external2
        dfbs.push_back(MakeDFB(
            DFB_EX_EXTERNAL2, single_tile_size, cb_sizes.ex_external_CB_size / single_tile_size, cb_data_format));
        // CB 20: ex2pe
        dfbs.push_back(MakeDFB(DFB_EX2PE, single_tile_size, cb_sizes.ex2pe_CB_size / single_tile_size, cb_data_format));
    }

    // CB 15: ex_global
    dfbs.push_back(
        MakeDFB(DFB_EX_GLOBAL, single_tile_size, cb_sizes.ex_global_CB_size / single_tile_size, cb_data_format));

    if (use_welford) {
        // CB 22: transpose intermediate
        dfbs.push_back(
            MakeDFB(DFB_TRANSPOSE, single_tile_size, cb_sizes.ex_global_CB_size / single_tile_size, cb_data_format));
        // CB 25: Reciprocal LUT — borrowed-memory DFB
        dfbs.push_back(MakeDFB(
            DFB_RECIPROCALS,
            reciprocal_CB_size_bytes,
            1,
            reciprocal_cb_data_format,
            m2::TensorParameterName{TP_RECIP}));
    }

    if (is_post_all_gather) {
        // CB 7: cb_stats — borrowed-memory DFB
        dfbs.push_back(MakeDFB(
            DFB_STATS,
            stats_single_tile_size,
            cb_sizes.stats_cb_size / stats_single_tile_size,
            stats_cb_data_format,
            m2::TensorParameterName{TP_STATS}));
        // CB 21: cb_stats_reduced
        dfbs.push_back(MakeDFB(
            DFB_STATS_REDUCED, single_tile_size, cb_sizes.stats_reduced_cb_size / single_tile_size, cb_data_format));
        // CB 19: cb_var
        dfbs.push_back(
            MakeDFB(DFB_VAR, single_tile_size, cb_sizes.ex_global_CB_size / single_tile_size, cb_data_format));
    }

    // CB 16: output
    {
        // Normal/pre paths: output buffer backs CB 16.
        // Post-allgather + !skip_write_back: CB 16 is intermediate (no buffer), CB 17 holds the output.
        const bool out_borrowed = !(is_post_all_gather && !skip_write_back);
        dfbs.push_back(MakeDFB(
            DFB_OUT,
            out_single_tile_size,
            cb_sizes.out_CB_size / out_single_tile_size,
            out_data_format,
            out_borrowed ? std::optional<m2::TensorParameterName>{TP_OUTPUT} : std::nullopt));
    }
    if (is_post_all_gather && !skip_write_back) {
        // CB 17: output reshard — borrowed-memory DFB
        dfbs.push_back(MakeDFB(
            DFB_OUT_RESHARD,
            out_single_tile_size,
            cb_sizes.out_reshard_CB_size / out_single_tile_size,
            out_data_format,
            m2::TensorParameterName{TP_OUTPUT}));
    }

    ////////////////////////////////////////////////////////////////////////////
    // TensorParameters
    ////////////////////////////////////////////////////////////////////////////
    std::vector<m2::TensorParameter> tensor_parameters;
    tensor_parameters.push_back({.unique_id = TP_INPUT_A, .spec = a.tensor_spec()});
    tensor_parameters.push_back({.unique_id = TP_OUTPUT, .spec = output.tensor_spec()});
    if (b.has_value()) {
        tensor_parameters.push_back({.unique_id = TP_RESIDUAL_B, .spec = b.value().tensor_spec()});
    }
    if (gamma.has_value()) {
        tensor_parameters.push_back({.unique_id = TP_GAMMA, .spec = gamma.value().tensor_spec()});
    }
    if (beta.has_value()) {
        tensor_parameters.push_back({.unique_id = TP_BETA, .spec = beta.value().tensor_spec()});
    }
    if (is_post_all_gather && stats.has_value()) {
        tensor_parameters.push_back({.unique_id = TP_STATS, .spec = stats.value().tensor_spec()});
    }
    if (use_welford) {
        tensor_parameters.push_back({.unique_id = TP_RECIP, .spec = tensor_args.recip_tensor->tensor_spec()});
    }

    ////////////////////////////////////////////////////////////////////////////
    // SemaphoreSpecs (always all 3; placement = all_cores)
    ////////////////////////////////////////////////////////////////////////////
    std::vector<m2::SemaphoreSpec> semaphores = {
        m2::SemaphoreSpec{.unique_id = SEM_REDUCE_SENDER, .target_nodes = core_ranges.all_cores},
        m2::SemaphoreSpec{.unique_id = SEM_REDUCE_RECEIVER, .target_nodes = core_ranges.all_cores},
        m2::SemaphoreSpec{.unique_id = SEM_REDUCE_SECOND_STAGE, .target_nodes = core_ranges.all_cores},
    };

    ////////////////////////////////////////////////////////////////////////////
    // Reader common CTA bindings (mapped from legacy positional CTAs)
    ////////////////////////////////////////////////////////////////////////////
    // The reader kernels share many of the same named CTAs across sender / receiver variants.
    // Semaphore IDs that were CTAs in legacy are replaced by SemaphoreBinding.
    m2::KernelSpec::CompileTimeArgBindings reader_common_ctas = {
        {"num_blocks", grid.num_blocks},
        {"block_h", block_ht},
        {"block_h_size_bytes", block_ht * single_tile_size},
        {"num_all_to_all_workers_first_stage", workers.num_cores_all_to_all_first_stage},
        {"num_tiles_per_worker", workers.num_rows_per_all_to_all_worker},
        {"num_tiles_per_worker_bytes", workers.num_rows_per_all_to_all_worker * single_tile_size},
        {"num_tiles_per_worker_last", workers.num_rows_per_all_to_all_worker_last},
        {"num_tiles_per_worker_last_bytes", workers.num_rows_per_all_to_all_worker_last * single_tile_size},
        {"row_major", static_cast<uint32_t>(grid.row_wise)},
        {"num_x", core_ranges.num_cores_x_mcast},
        {"num_y", core_ranges.num_cores_y_mcast},
        {"use_two_stage_reduce", static_cast<uint32_t>(grid.use_two_stage_reduce)},
        {"num_blocks_first_stage", workers.num_blocks_first_stage},
        {"num_blocks_second_stage", workers.num_blocks_second_stage},
        {"rms_norm", static_cast<uint32_t>(rms_norm)},
        {"use_welford", static_cast<uint32_t>(use_welford)},
    };

    auto bind_reader_dfbs = [&](m2::KernelSpec& k) {
        // Reader produces input/scaler/eps and the partial reduction CBs that downstream
        // compute / sender consumes. Many of these are also consumed by the reader itself
        // during the all-to-all phase. We unconditionally bind all DFBs that the reader
        // source might reference and let the kernel's `#ifdef` guards skip the unused ones.
        k.dfb_bindings.push_back(ProducerDFB(DFB_IN0, "cb_in0"));
        if (b.has_value()) {
            k.dfb_bindings.push_back(ProducerDFB(DFB_INB, "cb_inb"));
            if (is_pre_all_gather) {
                k.dfb_bindings.push_back(ProducerDFB(DFB_IN0_PRE, "cb_in0_pre"));
            }
        }
        if (!use_welford) {
            k.dfb_bindings.push_back(ProducerDFB(DFB_SCALER, "cb_scaler"));
            k.dfb_bindings.push_back(ProducerDFB(DFB_EPS, "cb_eps"));
            k.dfb_bindings.push_back(ProducerDFB(DFB_SCALER_GLOBAL, "cb_scaler_global"));
            k.dfb_bindings.push_back(ConsumerDFB(DFB_EX_PARTIAL2, "cb_ex_partial2"));
            k.dfb_bindings.push_back(ProducerDFB(DFB_EX2, "cb_ex2"));
            k.dfb_bindings.push_back(ProducerDFB(DFB_EX_EXTERNAL2, "cb_ex_external2"));
        }
        if (!rms_norm) {
            k.dfb_bindings.push_back(ConsumerDFB(DFB_EX_PARTIAL, "cb_ex_partial"));
            k.dfb_bindings.push_back(ProducerDFB(DFB_EX, "cb_ex"));
            k.dfb_bindings.push_back(ProducerDFB(DFB_EX_EXTERNAL, "cb_ex_external"));
        }
        k.dfb_bindings.push_back(ProducerDFB(DFB_EX_GLOBAL, "cb_ex_global"));
    };

    auto bind_reader_semaphores = [&](m2::KernelSpec& k) {
        k.semaphore_bindings = {
            {.semaphore_spec_name = SEM_REDUCE_RECEIVER, .accessor_name = "reduce_receiver"},
            {.semaphore_spec_name = SEM_REDUCE_SENDER, .accessor_name = "reduce_sender"},
            {.semaphore_spec_name = SEM_REDUCE_SECOND_STAGE, .accessor_name = "reduce_second_stage"},
        };
    };

    // Ghost TensorBindings for borrowed-memory-backed TensorParameters. The validator
    // requires every TensorParameter to have ≥1 TensorBinding, but for borrowed-memory
    // DFBs the kernel doesn't actually call `TensorAccessor(ta::name)` — the data is
    // consumed via the borrowed CB. Bind them on reader_sender (a DM kernel; compute
    // kernels don't support TensorBindings).
    auto bind_borrowed_tensors = [&](m2::KernelSpec& k) {
        k.tensor_bindings.push_back({.tensor_parameter_name = TP_INPUT_A, .accessor_name = "input_a"});
        k.tensor_bindings.push_back({.tensor_parameter_name = TP_OUTPUT, .accessor_name = "output"});
        if (b.has_value()) {
            k.tensor_bindings.push_back({.tensor_parameter_name = TP_RESIDUAL_B, .accessor_name = "residual_b"});
        }
        if (is_post_all_gather && stats.has_value()) {
            k.tensor_bindings.push_back({.tensor_parameter_name = TP_STATS, .accessor_name = "stats"});
        }
        if (use_welford) {
            k.tensor_bindings.push_back({.tensor_parameter_name = TP_RECIP, .accessor_name = "recip"});
        }
    };

    ////////////////////////////////////////////////////////////////////////////
    // Reader sender KernelSpec
    ////////////////////////////////////////////////////////////////////////////
    m2::KernelSpec reader_sender;
    reader_sender.unique_id = K_READER_SENDER;
    reader_sender.source = std::filesystem::path{kernel_paths.reader_sender};
    reader_sender.config_spec = m2::DataMovementConfiguration{
        .gen1_data_movement_config =
            m2::DataMovementConfiguration::Gen1DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = reader_noc, .noc_mode = NOC_MODE::DM_DEDICATED_NOC},
        .gen2_data_movement_config = m2::DataMovementConfiguration::Gen2DataMovementConfig{}};
    reader_sender.compile_time_arg_bindings = reader_common_ctas;
    reader_sender.compile_time_arg_bindings.push_back({"is_all_to_all_worker", 1u});
    bind_reader_dfbs(reader_sender);
    bind_reader_semaphores(reader_sender);
    bind_borrowed_tensors(reader_sender);
    for (auto& d : kernel_defines.reader) {
        reader_sender.compiler_options.defines.push_back(d);
    }
    reader_sender.runtime_arguments_schema.named_runtime_args = {
        "mcast_start_x", "mcast_start_y", "mcast_end_x", "mcast_end_y", "start_x", "start_y"};
    // Mcast noc lists are passed as varargs (variable count per node)
    reader_sender.advanced_options.num_runtime_varargs = core_ranges.num_cores_x_mcast + core_ranges.num_cores_y_mcast;

    ////////////////////////////////////////////////////////////////////////////
    // Reader receiver (all-to-all) KernelSpec — only when grid uses mcast
    ////////////////////////////////////////////////////////////////////////////
    m2::KernelSpec reader_rcv_a2a;
    if (has_reader_receiver_all_to_all) {
        reader_rcv_a2a.unique_id = K_READER_RCV_A2A;
        reader_rcv_a2a.source = std::filesystem::path{kernel_paths.reader_receiver};
        reader_rcv_a2a.config_spec = m2::DataMovementConfiguration{
            .gen1_data_movement_config =
                m2::DataMovementConfiguration::Gen1DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = reader_noc,
                    .noc_mode = NOC_MODE::DM_DEDICATED_NOC},
            .gen2_data_movement_config = m2::DataMovementConfiguration::Gen2DataMovementConfig{}};
        reader_rcv_a2a.compile_time_arg_bindings = reader_common_ctas;
        reader_rcv_a2a.compile_time_arg_bindings.push_back({"is_all_to_all_worker", 1u});
        bind_reader_dfbs(reader_rcv_a2a);
        bind_reader_semaphores(reader_rcv_a2a);
        for (auto& d : kernel_defines.reader) {
            reader_rcv_a2a.compiler_options.defines.push_back(d);
        }
        reader_rcv_a2a.runtime_arguments_schema.named_runtime_args = {
            "is_last_all_to_all_worker", "all_to_all_offset_bytes", "is_second_stage_reader", "start_x", "start_y"};
        reader_rcv_a2a.advanced_options.num_runtime_varargs =
            core_ranges.num_cores_x_mcast + core_ranges.num_cores_y_mcast;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Reader receiver (not all-to-all) KernelSpec
    ////////////////////////////////////////////////////////////////////////////
    m2::KernelSpec reader_rcv;
    if (has_not_all_to_all_workers) {
        reader_rcv.unique_id = K_READER_RCV;
        reader_rcv.source = std::filesystem::path{kernel_paths.reader_receiver};
        reader_rcv.config_spec = m2::DataMovementConfiguration{
            .gen1_data_movement_config =
                m2::DataMovementConfiguration::Gen1DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = reader_noc,
                    .noc_mode = NOC_MODE::DM_DEDICATED_NOC},
            .gen2_data_movement_config = m2::DataMovementConfiguration::Gen2DataMovementConfig{}};
        reader_rcv.compile_time_arg_bindings = reader_common_ctas;
        reader_rcv.compile_time_arg_bindings.push_back({"is_all_to_all_worker", 0u});
        bind_reader_dfbs(reader_rcv);
        bind_reader_semaphores(reader_rcv);
        for (auto& d : kernel_defines.reader) {
            reader_rcv.compiler_options.defines.push_back(d);
        }
        // Receiver-not-all-to-all expects fewer args; legacy uses 5 positional + 2 noc coords.
        reader_rcv.runtime_arguments_schema.named_runtime_args = {
            "is_last_all_to_all_worker",
            "all_to_all_offset_bytes",
            "is_second_stage_reader",
            "noc_pad_0",
            "noc_pad_1",
            "noc_x",
            "noc_y"};
    }

    ////////////////////////////////////////////////////////////////////////////
    // Writer CTAs (shared by sender + receiver)
    ////////////////////////////////////////////////////////////////////////////
    auto build_writer_ctas = [&](bool is_a2a) {
        m2::KernelSpec::CompileTimeArgBindings ctas = {
            {"is_all_to_all_worker", static_cast<uint32_t>(is_a2a)},
            {"do_gamma", static_cast<uint32_t>(gamma.has_value())},
            {"do_beta", static_cast<uint32_t>(beta.has_value())},
            {"block_wt", block_wt},
            {"use_welford", static_cast<uint32_t>(use_welford)},
        };
        if (gamma.has_value() && gamma.value().layout() == Layout::ROW_MAJOR) {
            uint32_t gamma_stick = gamma.value().padded_shape()[-1] * gamma.value().element_size();
            ctas.push_back({"gamma_or_beta_stick", gamma_stick});
        } else if (beta.has_value() && beta.value().layout() == Layout::ROW_MAJOR) {
            uint32_t beta_stick = beta.value().padded_shape()[-1] * beta.value().element_size();
            ctas.push_back({"gamma_or_beta_stick", beta_stick});
        }
        ctas.push_back({"gamma_f32", static_cast<uint32_t>(gamma_cb_data_format == tt::DataFormat::Float32)});
        ctas.push_back({"beta_f32", static_cast<uint32_t>(beta_cb_data_format == tt::DataFormat::Float32)});
        ctas.push_back({"block_wt_bytes", block_wt * out_single_tile_size});
        ctas.push_back({"block_wt_resharded_bytes", block_wt_resharded * out_single_tile_size});
        ctas.push_back({"block_ht", block_ht});
        return ctas;
    };

    auto bind_writer_dfbs = [&](m2::KernelSpec& k) {
        if (gamma.has_value()) {
            k.dfb_bindings.push_back(ProducerDFB(DFB_GAMMA, "cb_gamma"));
        }
        if (beta.has_value()) {
            k.dfb_bindings.push_back(ProducerDFB(DFB_BETA, "cb_beta"));
        }
        k.dfb_bindings.push_back(ConsumerDFB(DFB_OUT, "cb_out"));
        if (is_post_all_gather && !skip_write_back) {
            k.dfb_bindings.push_back(ProducerDFB(DFB_OUT_RESHARD, "cb_out_resharded"));
        }
        if (!use_welford) {
            k.dfb_bindings.push_back(ProducerDFB(DFB_SCALER, "cb_in_2"));
            k.dfb_bindings.push_back(ProducerDFB(DFB_EPS, "cb_eps"));
            k.dfb_bindings.push_back(ProducerDFB(DFB_SCALER_GLOBAL, "cb_in_4"));
        }
    };

    auto bind_writer_tensors = [&](m2::KernelSpec& k) {
        if (gamma.has_value()) {
            k.tensor_bindings.push_back({.tensor_parameter_name = TP_GAMMA, .accessor_name = "gamma"});
        }
        if (beta.has_value()) {
            k.tensor_bindings.push_back({.tensor_parameter_name = TP_BETA, .accessor_name = "beta"});
        }
    };

    m2::KernelSpec writer_sender;
    writer_sender.unique_id = K_WRITER_SENDER;
    writer_sender.source = std::filesystem::path{kernel_paths.writer};
    writer_sender.config_spec = m2::DataMovementConfiguration{
        .gen1_data_movement_config =
            m2::DataMovementConfiguration::Gen1DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1, .noc = writer_noc, .noc_mode = NOC_MODE::DM_DEDICATED_NOC},
        .gen2_data_movement_config = m2::DataMovementConfiguration::Gen2DataMovementConfig{}};
    writer_sender.compile_time_arg_bindings = build_writer_ctas(true);
    bind_writer_dfbs(writer_sender);
    bind_writer_tensors(writer_sender);
    for (auto& d : kernel_defines.writer) {
        writer_sender.compiler_options.defines.push_back(d);
    }
    // Writer RTAs: scaler values + addresses + write-back. Use named for the common ones,
    // and varargs for the variable write-back segment list (per-core, multi-segment).
    writer_sender.runtime_arguments_schema.named_runtime_args = {
        "packed_cinv", "packed_winv", "eps_u", "gamma_addr", "beta_addr", "gamma_tile_start", "beta_tile_start"};
    writer_sender.advanced_options.num_runtime_varargs = 64;  // upper bound; legacy was variable

    m2::KernelSpec writer_rcv;
    if (has_not_all_to_all_workers) {
        writer_rcv.unique_id = K_WRITER_RCV;
        writer_rcv.source = std::filesystem::path{kernel_paths.writer};
        writer_rcv.config_spec = m2::DataMovementConfiguration{
            .gen1_data_movement_config =
                m2::DataMovementConfiguration::Gen1DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_1,
                    .noc = writer_noc,
                    .noc_mode = NOC_MODE::DM_DEDICATED_NOC},
            .gen2_data_movement_config = m2::DataMovementConfiguration::Gen2DataMovementConfig{}};
        writer_rcv.compile_time_arg_bindings = build_writer_ctas(false);
        bind_writer_dfbs(writer_rcv);
        bind_writer_tensors(writer_rcv);
        for (auto& d : kernel_defines.writer) {
            writer_rcv.compiler_options.defines.push_back(d);
        }
        writer_rcv.runtime_arguments_schema.named_runtime_args = {
            "packed_cinv", "packed_winv", "eps_u", "gamma_addr", "beta_addr", "gamma_tile_start", "beta_tile_start"};
        writer_rcv.advanced_options.num_runtime_varargs = 64;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Compute CTAs (sender vs receiver variants — preserves multiplicity)
    ////////////////////////////////////////////////////////////////////////////
    auto build_compute_ctas = [&](bool is_a2a) {
        m2::KernelSpec::CompileTimeArgBindings ctas = {
            {"unused0", 0u},
            {"do_gamma", static_cast<uint32_t>(gamma.has_value())},
            {"do_beta", static_cast<uint32_t>(beta.has_value())},
            {"num_blocks_first_stage", workers.num_blocks_first_stage},
            {"block_ht", block_ht},
            {"block_wt", block_wt},
            {"subblock_wt", subblock_wt},
            {"num_subblocks_w", num_subblocks_w},
            {"is_all_to_all_worker", static_cast<uint32_t>(is_a2a)},
            {"block_ht_block_wt", block_ht * block_wt},
            {"fp32_dest_acc_en", static_cast<uint32_t>(fp32_dest_acc_en)},
            {"float32_reduction", static_cast<uint32_t>(float32_reduction)},
            {"legacy_rsqrt", static_cast<uint32_t>(legacy_rsqrt)},
            {"num_blocks_second_stage", workers.num_blocks_second_stage},
        };
        if (use_welford) {
            uint32_t last_tile_W = K - (((K - tile_width) / tile_width) * tile_width);
            auto eps_u32 = std::bit_cast<uint32_t>(eps);
            ctas.push_back({"tile_width", tile_width});
            ctas.push_back({"last_tile_W", last_tile_W});
            ctas.push_back({"K_dim", K});
            ctas.push_back({"eps_u32", eps_u32});
            ctas.push_back({"per_core_recip_lut_size", block_w});
        }
        return ctas;
    };

    auto bind_compute_dfbs = [&](m2::KernelSpec& k) {
        // Compute consumes inputs (cb_in0, cb_inb, cb_gamma, cb_beta) and produces cb_out,
        // plus produces+consumes the intermediate reduction CBs (self-loop pattern).
        k.dfb_bindings.push_back(ConsumerDFB(DFB_IN0, "cb_in0"));
        if (b.has_value()) {
            k.dfb_bindings.push_back(ConsumerDFB(DFB_INB, "cb_inb"));
        }
        if (gamma.has_value()) {
            k.dfb_bindings.push_back(ConsumerDFB(DFB_GAMMA, "cb_gamma"));
        }
        if (beta.has_value()) {
            k.dfb_bindings.push_back(ConsumerDFB(DFB_BETA, "cb_beta"));
        }
        if (!use_welford) {
            k.dfb_bindings.push_back(ConsumerDFB(DFB_SCALER, "cb_scaler"));
            k.dfb_bindings.push_back(ConsumerDFB(DFB_EPS, "cb_eps"));
            k.dfb_bindings.push_back(ConsumerDFB(DFB_SCALER_GLOBAL, "cb_scaler_global"));
            // Self-loop on cb_ex_partial2 and cb_ex2 — compute writes them then reads them back
            k.dfb_bindings.push_back(ProducerDFB(DFB_EX_PARTIAL2, "cb_ex_partial2"));
            k.dfb_bindings.push_back(ConsumerDFB(DFB_EX_PARTIAL2, "cb_ex_partial2"));
            k.advanced_options.dfb_compute_self_loop_scopes.push_back(
                {.dfb_spec_name = DFB_EX_PARTIAL2, .scope = m2::DFBComputeSelfLoopScope::Scope::INTRA});
            k.dfb_bindings.push_back(ConsumerDFB(DFB_EX2, "cb_ex2"));
            k.dfb_bindings.push_back(ConsumerDFB(DFB_EX_EXTERNAL2, "cb_ex_external2"));
            k.dfb_bindings.push_back(ConsumerDFB(DFB_EX2PE, "cb_ex2pe"));
        }
        if (!rms_norm) {
            k.dfb_bindings.push_back(ProducerDFB(DFB_EX_PARTIAL, "cb_ex_partial"));
            k.dfb_bindings.push_back(ConsumerDFB(DFB_EX_PARTIAL, "cb_ex_partial"));
            k.advanced_options.dfb_compute_self_loop_scopes.push_back(
                {.dfb_spec_name = DFB_EX_PARTIAL, .scope = m2::DFBComputeSelfLoopScope::Scope::INTRA});
            k.dfb_bindings.push_back(ConsumerDFB(DFB_EX, "cb_ex"));
            k.dfb_bindings.push_back(ConsumerDFB(DFB_EX_EXTERNAL, "cb_ex_external"));
        }
        // cb_x: self-loop
        k.dfb_bindings.push_back(ProducerDFB(DFB_X, "cb_x"));
        k.dfb_bindings.push_back(ConsumerDFB(DFB_X, "cb_x"));
        k.advanced_options.dfb_compute_self_loop_scopes.push_back(
            {.dfb_spec_name = DFB_X, .scope = m2::DFBComputeSelfLoopScope::Scope::INTRA});
        // cb_xmm: self-loop
        k.dfb_bindings.push_back(ProducerDFB(DFB_XMM, "cb_xmm"));
        k.dfb_bindings.push_back(ConsumerDFB(DFB_XMM, "cb_xmm"));
        k.advanced_options.dfb_compute_self_loop_scopes.push_back(
            {.dfb_spec_name = DFB_XMM, .scope = m2::DFBComputeSelfLoopScope::Scope::INTRA});
        // cb_ex_global: consumer
        k.dfb_bindings.push_back(ConsumerDFB(DFB_EX_GLOBAL, "cb_ex_global"));
        // Output
        k.dfb_bindings.push_back(ProducerDFB(DFB_OUT, "cb_out"));
        if (use_welford) {
            // Borrowed-memory DFB backed by the recip tensor — no kernel produces, but
            // the spec validator requires a PRODUCER binding. Declare a ghost PRODUCER
            // alongside the real CONSUMER. The kernel never calls reserve_back/push_back.
            k.dfb_bindings.push_back(ProducerDFB(DFB_RECIPROCALS, "cb_reciprocals"));
            k.dfb_bindings.push_back(ConsumerDFB(DFB_RECIPROCALS, "cb_reciprocals"));
            k.dfb_bindings.push_back(ProducerDFB(DFB_TRANSPOSE, "cb_transpose"));
            k.dfb_bindings.push_back(ConsumerDFB(DFB_TRANSPOSE, "cb_transpose"));
            k.advanced_options.dfb_compute_self_loop_scopes.push_back(
                {.dfb_spec_name = DFB_TRANSPOSE, .scope = m2::DFBComputeSelfLoopScope::Scope::INTRA});
        }
        if (is_post_all_gather) {
            // cb_stats is borrowed-memory (backed by stats tensor) — no kernel produces
            // tiles into it. Add a ghost PRODUCER for validator + the real CONSUMER.
            k.dfb_bindings.push_back(ProducerDFB(DFB_STATS, "cb_stats"));
            k.dfb_bindings.push_back(ConsumerDFB(DFB_STATS, "cb_stats"));
            // cb_stats_reduced: compute is producer; reader_sender_post_allgather is the
            // consumer (it mcasts the reduced stats). Adding both endpoints on compute as
            // a self-loop fallback would create over-binding; for now, declare it as
            // self-loop on compute + plus the reader_sender binding handles consumption.
            k.dfb_bindings.push_back(ProducerDFB(DFB_STATS_REDUCED, "cb_stats_reduced"));
            k.dfb_bindings.push_back(ConsumerDFB(DFB_STATS_REDUCED, "cb_stats_reduced"));
            k.advanced_options.dfb_compute_self_loop_scopes.push_back(
                {.dfb_spec_name = DFB_STATS_REDUCED, .scope = m2::DFBComputeSelfLoopScope::Scope::INTRA});
            // cb_var: compute self-loop (variance written and read back within compute).
            k.dfb_bindings.push_back(ProducerDFB(DFB_VAR, "cb_var"));
            k.dfb_bindings.push_back(ConsumerDFB(DFB_VAR, "cb_var"));
            k.advanced_options.dfb_compute_self_loop_scopes.push_back(
                {.dfb_spec_name = DFB_VAR, .scope = m2::DFBComputeSelfLoopScope::Scope::INTRA});
        }
    };

    // Build the compute ComputeConfiguration shared across both KernelSpecs. When
    // fp32_dest_acc_en is true, the validator requires an explicit unpack_to_dest_mode
    // entry for every FP32 DFB the compute kernel consumes.
    auto build_compute_config = [&]() {
        m2::ComputeConfiguration cfg{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .math_approx_mode = math_approx_mode};
        if (fp32_dest_acc_en) {
            auto add_default = [&](const char* dfb_name) {
                cfg.unpack_to_dest_mode.push_back({dfb_name, tt::tt_metal::UnpackToDestMode::Default});
            };
            // cb_data_format == Float32 when fp32_dest_acc_en, so all intermediate DFBs
            // using cb_data_format need entries. Mirror the host DFB-declaration guards.
            add_default(DFB_X);
            add_default(DFB_XMM);
            if (!rms_norm) {
                add_default(DFB_EX_PARTIAL);
                add_default(DFB_EX);
                add_default(DFB_EX_EXTERNAL);
            }
            if (!use_welford) {
                add_default(DFB_EX_PARTIAL2);
                add_default(DFB_EX2);
                add_default(DFB_EX_EXTERNAL2);
                add_default(DFB_EX2PE);
            }
            add_default(DFB_EX_GLOBAL);
            if (use_welford) {
                add_default(DFB_TRANSPOSE);
                add_default(DFB_RECIPROCALS);  // Float32 always (Welford LUT)
            }
            if (is_post_all_gather) {
                add_default(DFB_STATS_REDUCED);
                add_default(DFB_VAR);
                // cb_stats data format follows the stats tensor; only add if FP32.
                if (stats_cb_data_format == tt::DataFormat::Float32) {
                    add_default(DFB_STATS);
                }
            }
            // Inputs / weights: only when their dtype is FP32.
            if (in_data_format == tt::DataFormat::Float32) {
                add_default(DFB_IN0);
            }
            if (b.has_value() && in_data_format == tt::DataFormat::Float32) {
                add_default(DFB_INB);
            }
            if (gamma.has_value() && gamma_cb_data_format == tt::DataFormat::Float32) {
                add_default(DFB_GAMMA);
            }
            if (beta.has_value() && beta_cb_data_format == tt::DataFormat::Float32) {
                add_default(DFB_BETA);
            }
        }
        return cfg;
    };

    m2::KernelSpec compute_a2a;
    compute_a2a.unique_id = K_COMPUTE_A2A;
    compute_a2a.source = std::filesystem::path{kernel_paths.compute};
    compute_a2a.config_spec = build_compute_config();
    compute_a2a.compile_time_arg_bindings = build_compute_ctas(true);
    bind_compute_dfbs(compute_a2a);
    for (auto& d : kernel_defines.compute) {
        compute_a2a.compiler_options.defines.push_back(d);
    }
    compute_a2a.runtime_arguments_schema.named_runtime_args = {"num_reduce_tiles_per_block_h"};
    compute_a2a.advanced_options.num_runtime_varargs = 4;  // is_a2a + variable suffix

    m2::KernelSpec compute_not_a2a;
    if (has_not_all_to_all_workers) {
        compute_not_a2a.unique_id = K_COMPUTE_NOT_A2A;
        compute_not_a2a.source = std::filesystem::path{kernel_paths.compute};
        compute_not_a2a.config_spec = build_compute_config();
        compute_not_a2a.compile_time_arg_bindings = build_compute_ctas(false);
        bind_compute_dfbs(compute_not_a2a);
        for (auto& d : kernel_defines.compute) {
            compute_not_a2a.compiler_options.defines.push_back(d);
        }
        compute_not_a2a.runtime_arguments_schema.named_runtime_args = {"num_reduce_tiles_per_block_h"};
    }

    ////////////////////////////////////////////////////////////////////////////
    // WorkUnitSpecs
    ////////////////////////////////////////////////////////////////////////////
    // The sender / all_to_all_workers_except_sender / not_all_to_all_workers core groups
    // each correspond to a distinct kernel set. We organize work units per (kernel set,
    // core range).
    //
    // - Sender cores (1×1 typical): reader_sender + writer_sender + compute_all_to_all
    // - All-to-all-except-sender: reader_receiver_all_to_all + writer_sender + compute_all_to_all
    // - Not-all-to-all: reader_receiver + writer_receiver + compute_not_all_to_all
    //
    // Note: writer_sender's core range is `all_to_all_cores` which is sender + all_to_all_except_sender.
    std::vector<m2::WorkUnitSpec> work_units;

    // Sender work unit
    work_units.push_back({
        .unique_id = WU_SENDER,
        .kernels = {K_READER_SENDER, K_WRITER_SENDER, K_COMPUTE_A2A},
        .target_nodes = m2::NodeRangeSet(core_ranges.sender_cores),
    });

    if (has_reader_receiver_all_to_all) {
        work_units.push_back({
            .unique_id = WU_A2A_RCV,
            .kernels = {K_READER_RCV_A2A, K_WRITER_SENDER, K_COMPUTE_A2A},
            .target_nodes = core_ranges.all_to_all_workers_except_sender,
        });
    }
    if (has_not_all_to_all_workers) {
        work_units.push_back({
            .unique_id = WU_NOT_A2A,
            .kernels = {K_READER_RCV, K_WRITER_RCV, K_COMPUTE_NOT_A2A},
            .target_nodes = core_ranges.not_all_to_all_workers,
        });
    }

    ////////////////////////////////////////////////////////////////////////////
    // Assemble ProgramSpec
    ////////////////////////////////////////////////////////////////////////////
    std::vector<m2::KernelSpec> all_kernels;
    all_kernels.push_back(std::move(reader_sender));
    if (has_reader_receiver_all_to_all) {
        all_kernels.push_back(std::move(reader_rcv_a2a));
    }
    if (has_not_all_to_all_workers) {
        all_kernels.push_back(std::move(reader_rcv));
    }
    all_kernels.push_back(std::move(writer_sender));
    if (has_not_all_to_all_workers) {
        all_kernels.push_back(std::move(writer_rcv));
    }
    all_kernels.push_back(std::move(compute_a2a));
    if (has_not_all_to_all_workers) {
        all_kernels.push_back(std::move(compute_not_a2a));
    }

    m2::ProgramSpec spec{
        .program_id = "layernorm_sharded",
        .kernels = std::move(all_kernels),
        .dataflow_buffers = std::move(dfbs),
        .semaphores = std::move(semaphores),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = std::move(work_units),
    };

    ////////////////////////////////////////////////////////////////////////////
    // ProgramRunParams: per-core RTAs and tensor args
    ////////////////////////////////////////////////////////////////////////////
    m2::ProgramRunParams run_params;

    // Packed values for writer
    float winv = 1.0f / block_w;
    float cinv = is_post_all_gather ? (1.0f / num_distributed_devices) : (1.0f / grid.num_blocks);
    auto bfloat_cinv = bfloat16(cinv);
    auto bfloat_cinv_one = bfloat16(1.0f);
    auto bfloat_winv = bfloat16(winv);
    uint32_t packed_cinv_value = pack_two_bfloat16_into_uint32({bfloat_cinv, bfloat_cinv});
    uint32_t packed_cinv_value_one = pack_two_bfloat16_into_uint32({bfloat_cinv_one, bfloat_cinv_one});
    uint32_t packed_winv_value = pack_two_bfloat16_into_uint32({bfloat_winv, bfloat_winv});
    uint32_t eps_u = std::bit_cast<uint32_t>(eps);

    // Build mcast NOC coordinate lists
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

    uint32_t last_core_width_index = grid.mcast_1d ? (uint32_t)(core_ranges.all_cores.num_cores() - 1)
                                                   : (grid.row_wise ? grid.grid_size.x - 1 : grid.grid_size.y - 1);

    m2::ProgramRunParams::KernelRunParams reader_sender_rp{.kernel_spec_name = K_READER_SENDER};
    m2::ProgramRunParams::KernelRunParams reader_rcv_a2a_rp{.kernel_spec_name = K_READER_RCV_A2A};
    m2::ProgramRunParams::KernelRunParams reader_rcv_rp{.kernel_spec_name = K_READER_RCV};
    m2::ProgramRunParams::KernelRunParams writer_sender_rp{.kernel_spec_name = K_WRITER_SENDER};
    m2::ProgramRunParams::KernelRunParams writer_rcv_rp{.kernel_spec_name = K_WRITER_RCV};
    m2::ProgramRunParams::KernelRunParams compute_a2a_rp{.kernel_spec_name = K_COMPUTE_A2A};
    m2::ProgramRunParams::KernelRunParams compute_not_a2a_rp{.kernel_spec_name = K_COMPUTE_NOT_A2A};

    const auto& cores = corerange_to_cores(core_ranges.all_cores, core_ranges.all_cores.num_cores(), grid.row_wise);
    uint32_t current_storage_core = 0;
    uint32_t current_storage_core_offset = 0;
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const auto& core = cores[i];
        auto idx =
            PerCoreIndices::compute(i, core, grid, workers, block_wt, Kt, last_core_width_index, single_tile_size);
        bool is_a2a = idx.is_all_to_all(grid, workers);
        bool is_sender = (idx.width_index == 0);
        m2::NodeCoord node{core.x, core.y};

        // Compute runtime args
        {
            std::unordered_map<std::string, uint32_t> args = {
                {"num_reduce_tiles_per_block_h", idx.num_reduce_tiles_per_block_h},
            };
            // Vararg payload (legacy compute_args tail)
            std::vector<uint32_t> varargs;
            if (is_a2a) {
                uint32_t num_rows;
                if (grid.use_two_stage_reduce) {
                    num_rows = idx.width_index_two_stage == workers.num_cores_all_to_all_first_stage - 1
                                   ? workers.num_rows_per_all_to_all_worker_last
                                   : workers.num_rows_per_all_to_all_worker;
                } else {
                    num_rows = idx.width_index == workers.num_cores_all_to_all - 1
                                   ? workers.num_rows_per_all_to_all_worker_last
                                   : workers.num_rows_per_all_to_all_worker;
                }
                varargs.push_back(num_rows);
                varargs.push_back((uint32_t)grid.use_two_stage_reduce);
                bool is_second_stage_reader =
                    grid.use_two_stage_reduce && idx.width_index < workers.num_cores_all_to_all_first_stage;
                varargs.push_back((uint32_t)is_second_stage_reader);
                if (is_post_all_gather) {
                    varargs.push_back(num_distributed_devices);
                }
                compute_a2a_rp.named_runtime_args.push_back({.node = node, .args = std::move(args)});
                compute_a2a_rp.advanced_options.runtime_varargs.push_back({.node = node, .args = std::move(varargs)});
            } else {
                compute_not_a2a_rp.named_runtime_args.push_back({.node = node, .args = std::move(args)});
            }
        }

        // Reader runtime args
        if (is_sender) {
            // Mcast range
            CoreCoord mcast_start, mcast_end;
            if (grid.mcast_1d) {
                CoreCoord top_left = {(std::size_t)core_ranges.start_core.x, (std::size_t)core_ranges.start_core.y};
                CoreCoord bottom_right = {
                    (std::size_t)core_ranges.start_core.x + grid.grid_size.x - 1,
                    (std::size_t)core_ranges.start_core.y + grid.grid_size.y - 1};
                mcast_start = device->worker_core_from_logical_core(top_left);
                mcast_end = device->worker_core_from_logical_core(bottom_right);
            } else {
                if (grid.row_wise) {
                    CoreCoord left_plus_one = {(std::size_t)core_ranges.start_core.x + 1, (std::size_t)core.y};
                    CoreCoord right = {
                        (std::size_t)core_ranges.start_core.x + grid.grid_size.x - 1, (std::size_t)core.y};
                    mcast_start = device->worker_core_from_logical_core(left_plus_one);
                    mcast_end = device->worker_core_from_logical_core(right);
                } else {
                    CoreCoord top_plus_one = {(std::size_t)core.x, (std::size_t)core_ranges.start_core.y + 1};
                    CoreCoord bottom = {
                        (std::size_t)core.x, (std::size_t)core_ranges.start_core.y + grid.grid_size.y - 1};
                    mcast_start = device->worker_core_from_logical_core(top_plus_one);
                    mcast_end = device->worker_core_from_logical_core(bottom);
                }
            }
            if (reader_noc == NOC::NOC_1) {
                std::swap(mcast_start, mcast_end);
            }

            std::unordered_map<std::string, uint32_t> args = {
                {"mcast_start_x", static_cast<uint32_t>(mcast_start.x)},
                {"mcast_start_y", static_cast<uint32_t>(mcast_start.y)},
                {"mcast_end_x", static_cast<uint32_t>(mcast_end.x)},
                {"mcast_end_y", static_cast<uint32_t>(mcast_end.y)},
            };
            std::vector<uint32_t> varargs;
            if (grid.mcast_1d) {
                args["start_x"] = static_cast<uint32_t>(core.x - core_ranges.start_core.x);
                args["start_y"] = static_cast<uint32_t>(core.y - core_ranges.start_core.y);
                varargs.insert(varargs.end(), mcast_noc_x.begin(), mcast_noc_x.end());
                varargs.insert(varargs.end(), mcast_noc_y.begin(), mcast_noc_y.end());
            } else {
                if (grid.row_wise) {
                    args["start_x"] = static_cast<uint32_t>(core.x - core_ranges.start_core.x);
                    args["start_y"] = 0;
                    varargs.insert(varargs.end(), mcast_noc_x.begin(), mcast_noc_x.end());
                    varargs.push_back(mcast_noc_y[idx.height_index]);
                } else {
                    args["start_x"] = 0;
                    args["start_y"] = static_cast<uint32_t>(core.y - core_ranges.start_core.y);
                    varargs.push_back(mcast_noc_x[idx.height_index]);
                    varargs.insert(varargs.end(), mcast_noc_y.begin(), mcast_noc_y.end());
                }
            }
            reader_sender_rp.named_runtime_args.push_back({.node = node, .args = std::move(args)});
            reader_sender_rp.advanced_options.runtime_varargs.push_back({.node = node, .args = std::move(varargs)});
        } else if (is_a2a) {
            bool is_last_a2a;
            if (grid.use_two_stage_reduce) {
                is_last_a2a = idx.width_index_two_stage == workers.num_cores_all_to_all_first_stage - 1;
            } else {
                is_last_a2a = idx.width_index == workers.num_cores_all_to_all - 1;
            }
            bool is_second_stage_reader =
                grid.use_two_stage_reduce && idx.width_index < workers.num_cores_all_to_all_first_stage;

            std::unordered_map<std::string, uint32_t> args = {
                {"is_last_all_to_all_worker", static_cast<uint32_t>(is_last_a2a)},
                {"all_to_all_offset_bytes", idx.all_to_all_worker_tile_offset_bytes},
                {"is_second_stage_reader", static_cast<uint32_t>(is_second_stage_reader)},
            };
            std::vector<uint32_t> varargs;
            if (grid.mcast_1d) {
                args["start_x"] = static_cast<uint32_t>(core.x - core_ranges.start_core.x);
                args["start_y"] = static_cast<uint32_t>(core.y - core_ranges.start_core.y);
                varargs.insert(varargs.end(), mcast_noc_x.begin(), mcast_noc_x.end());
                varargs.insert(varargs.end(), mcast_noc_y.begin(), mcast_noc_y.end());
            } else {
                if (grid.row_wise) {
                    args["start_x"] = static_cast<uint32_t>(core.x - core_ranges.start_core.x);
                    args["start_y"] = 0;
                    varargs.insert(varargs.end(), mcast_noc_x.begin(), mcast_noc_x.end());
                    varargs.push_back(mcast_noc_y[idx.height_index]);
                } else {
                    args["start_x"] = 0;
                    args["start_y"] = static_cast<uint32_t>(core.y - core_ranges.start_core.y);
                    varargs.push_back(mcast_noc_x[idx.height_index]);
                    varargs.insert(varargs.end(), mcast_noc_y.begin(), mcast_noc_y.end());
                }
            }
            reader_rcv_a2a_rp.named_runtime_args.push_back({.node = node, .args = std::move(args)});
            reader_rcv_a2a_rp.advanced_options.runtime_varargs.push_back({.node = node, .args = std::move(varargs)});
        } else {
            // Non-all-to-all receiver
            std::unordered_map<std::string, uint32_t> args = {
                {"is_last_all_to_all_worker", 0u},
                {"all_to_all_offset_bytes", idx.all_to_all_worker_tile_offset_bytes},
                {"is_second_stage_reader", 0u},
                {"noc_pad_0", 0u},
                {"noc_pad_1", 0u},
            };
            if (grid.mcast_1d) {
                args["noc_x"] = mcast_noc_x[0];
                args["noc_y"] = mcast_noc_y[0];
            } else if (grid.row_wise) {
                args["noc_x"] = mcast_noc_x[0];
                args["noc_y"] = mcast_noc_y[idx.height_index];
            } else {
                args["noc_x"] = mcast_noc_x[idx.height_index];
                args["noc_y"] = mcast_noc_y[0];
            }
            reader_rcv_rp.named_runtime_args.push_back({.node = node, .args = std::move(args)});
        }

        // Write-back varargs (post-allgather only, multi-segment)
        std::vector<uint32_t> write_back_varargs;
        if (is_post_all_gather) {
            write_back_varargs.push_back(current_storage_core_offset * out_single_tile_size);
            uint32_t num_segments = 0;
            uint32_t worker_offset = 0;
            // Build segment list
            std::vector<uint32_t> segments;
            while (worker_offset < block_wt) {
                uint32_t tiles_available = block_wt_resharded - current_storage_core_offset;
                uint32_t tiles_left = block_wt - worker_offset;
                uint32_t tiles_to_write = std::min(tiles_left, tiles_available);
                ++num_segments;
                segments.push_back(tiles_to_write * out_single_tile_size);
                segments.push_back(storage_core_noc_x[current_storage_core]);
                segments.push_back(storage_core_noc_y[current_storage_core]);
                worker_offset += tiles_to_write;
                current_storage_core_offset += tiles_to_write;
                if (current_storage_core_offset >= block_wt_resharded) {
                    ++current_storage_core;
                    current_storage_core_offset = 0;
                    TT_FATAL(
                        current_storage_core <= (uint32_t)all_storage_cores.num_cores(),
                        "current_storage_core {} is exceeding number of storage cores {}",
                        current_storage_core,
                        all_storage_cores.num_cores());
                }
            }
            // Prepend num_segments
            write_back_varargs.insert(write_back_varargs.begin() + 1, num_segments);
            write_back_varargs.insert(write_back_varargs.end(), segments.begin(), segments.end());
        }

        // Writer runtime args
        {
            uint32_t packed_cinv_eff = packed_cinv_value;
            if (is_a2a && grid.use_two_stage_reduce && idx.width_index >= workers.num_cores_all_to_all_first_stage) {
                packed_cinv_eff = packed_cinv_value_one;
            }
            std::unordered_map<std::string, uint32_t> args = {
                {"packed_cinv", packed_cinv_eff},
                {"packed_winv", packed_winv_value},
                {"eps_u", eps_u},
                {"gamma_addr", static_cast<uint32_t>(gamma_dram_addr)},
                {"beta_addr", static_cast<uint32_t>(beta_dram_addr)},
                {"gamma_tile_start", idx.gamma_tile_start_id},
                {"beta_tile_start", idx.beta_tile_start_id},
            };
            if (is_a2a) {
                writer_sender_rp.named_runtime_args.push_back({.node = node, .args = std::move(args)});
                writer_sender_rp.advanced_options.runtime_varargs.push_back({.node = node, .args = write_back_varargs});
            } else {
                writer_rcv_rp.named_runtime_args.push_back({.node = node, .args = std::move(args)});
                writer_rcv_rp.advanced_options.runtime_varargs.push_back({.node = node, .args = write_back_varargs});
            }
        }
    }

    run_params.kernel_run_params.push_back(std::move(reader_sender_rp));
    if (has_reader_receiver_all_to_all) {
        run_params.kernel_run_params.push_back(std::move(reader_rcv_a2a_rp));
    }
    if (has_not_all_to_all_workers) {
        run_params.kernel_run_params.push_back(std::move(reader_rcv_rp));
    }
    run_params.kernel_run_params.push_back(std::move(writer_sender_rp));
    if (has_not_all_to_all_workers) {
        run_params.kernel_run_params.push_back(std::move(writer_rcv_rp));
    }
    run_params.kernel_run_params.push_back(std::move(compute_a2a_rp));
    if (has_not_all_to_all_workers) {
        run_params.kernel_run_params.push_back(std::move(compute_not_a2a_rp));
    }

    // TensorArgs
    run_params.tensor_args.push_back({.tensor_parameter_name = TP_INPUT_A, .tensor = a.mesh_tensor()});
    run_params.tensor_args.push_back({.tensor_parameter_name = TP_OUTPUT, .tensor = output.mesh_tensor()});
    if (b.has_value()) {
        run_params.tensor_args.push_back({.tensor_parameter_name = TP_RESIDUAL_B, .tensor = b.value().mesh_tensor()});
    }
    if (gamma.has_value()) {
        run_params.tensor_args.push_back({.tensor_parameter_name = TP_GAMMA, .tensor = gamma.value().mesh_tensor()});
    }
    if (beta.has_value()) {
        run_params.tensor_args.push_back({.tensor_parameter_name = TP_BETA, .tensor = beta.value().mesh_tensor()});
    }
    if (is_post_all_gather && stats.has_value()) {
        run_params.tensor_args.push_back({.tensor_parameter_name = TP_STATS, .tensor = stats.value().mesh_tensor()});
    }
    if (use_welford) {
        run_params.tensor_args.push_back(
            {.tensor_parameter_name = TP_RECIP, .tensor = tensor_args.recip_tensor->mesh_tensor()});
    }

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_params),
    };
}

}  // namespace ttnn::prim
