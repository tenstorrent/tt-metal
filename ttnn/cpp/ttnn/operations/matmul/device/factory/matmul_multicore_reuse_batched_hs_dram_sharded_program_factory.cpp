// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_batched_hs_dram_sharded_program_factory.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <utility>

#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/matmul/shared_with_host/activation_type.hpp"

#include "ttnn/metal2_artifacts.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt;

using ttnn::operations::unary::UnaryOpType;
using ttnn::operations::unary::UnaryWithParam;

namespace m2 = tt::tt_metal::experimental;

namespace ttnn::prim {
namespace reuse_batched_hs_dram_sharded_optimized_helpers {

using dram_sharded_helpers::get_max_page_size_and_num_pages;
using dram_sharded_helpers::get_optimal_dram_bank_to_reader_assignment;

// Batch-sharded DRAM matmul
// For batched matmul: [1, B, M, K] x [1, B, K, N] = [1, B, M, N]
// Sharded by batch dimension - each worker handles B/num_workers complete matmuls
// Metal 2.0 ProgramSpec variant: translates the same logic as the legacy create_descriptor into
// a ProgramSpec + ProgramRunArgs.
//
// Notes on the structural translation:
//  - Local CBs (c_0 in0, c_1 in1, c_3 bias, c_4 out, c_5 intermed0) become same-node DFBs. When
//    interm0_data_format == output_data_format the legacy uses ONE CBDescriptor with two
//    format_descriptors (c_4 + c_5 over one L1 region); that is expressed as two DFBs sharing
//    backing memory via advanced_options.alias_with.
//  - The legacy borrowed CBs c_2 (sharded in0 on input storage cores, backed by in0_tensor) and
//    c_6 (output reshard on output storage cores, backed by out_tensor) are NOT bound by any
//    kernel — they were the pre-Metal-2.0 way to reserve the io tensors' L1 on the storage cores.
//    In Metal 2.0 the io tensors' own buffer reservations cover that, and the kernels obtain the
//    shard base addresses via the typed `a`/`out` tensor bindings (Case-2 bridge). So c_2/c_6
//    drop, subsumed by the TensorParameters.
//  - Raw addresses (in0 shard L1, in1/bias DRAM bank base, out shard L1) flowed through RTAs in
//    the legacy data-movement kernels; they now flow through the typed tensor channel and the
//    kernels pull the base via TensorAccessor::get_bank_base_address(), keeping the explicit NoC
//    walks unchanged (Case-2 portable).
static ttnn::device_operation::ProgramArtifacts create_program_batch_sharded_spec(
    tt::tt_metal::IDevice* device,
    const CoreRangeSet& input_all_storage_cores,
    const CoreRangeSet& output_all_storage_cores,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode,
    bool packer_l1_acc,
    bool dst_full_sync_en,
    ttnn::operations::compute_throttle_utils::ThrottleLevel throttle_level,
    uint32_t B,
    uint32_t /* M */,
    uint32_t K,
    uint32_t /* N */,
    uint32_t in0_block_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    std::optional<UnaryWithParam> fused_activation,
    const ttnn::Tensor& in0_tensor,
    const ttnn::Tensor& in1_tensor,
    const std::optional<const ttnn::Tensor>& bias_tensor,
    const ttnn::Tensor& out_tensor,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const tt::tt_metal::Tile& bias_tile,
    const tt::tt_metal::Tile& output_tile,
    tt::DataFormat in0_data_format,
    tt::DataFormat in1_data_format,
    tt::DataFormat bias_data_format,
    tt::DataFormat output_data_format,
    bool untilize_out,
    bool skip_compute,
    bool skip_write_back) {
    tt_metal::NOC in1_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());

    std::vector<CoreCoord> all_worker_cores_ordered;
    CoreRangeSet all_worker_cores;
    get_optimal_dram_bank_to_reader_assignment(device, all_worker_cores_ordered, all_worker_cores, in1_noc);

    // Input / output storage core ordering
    std::vector<CoreCoord> input_storage_cores_ordered =
        corerange_to_cores(input_all_storage_cores, std::nullopt, true);
    std::vector<CoreCoord> output_storage_cores_ordered =
        corerange_to_cores(output_all_storage_cores, std::nullopt, true);

    uint32_t num_workers = all_worker_cores_ordered.size();
    TT_FATAL(
        input_storage_cores_ordered.size() == num_workers,
        "Input storage cores ({}) must match number of workers/DRAM banks ({})",
        input_storage_cores_ordered.size(),
        num_workers);
    TT_FATAL(
        output_storage_cores_ordered.size() == num_workers,
        "Output storage cores ({}) must match number of workers/DRAM banks ({})",
        output_storage_cores_ordered.size(),
        num_workers);
    for (uint32_t i = 0; i < num_workers; ++i) {
        TT_FATAL(
            input_storage_cores_ordered[i] == all_worker_cores_ordered[i],
            "Input storage core ordering mismatch at index {}",
            i);
        TT_FATAL(
            output_storage_cores_ordered[i] == all_worker_cores_ordered[i],
            "Output storage core ordering mismatch at index {}",
            i);
    }

    // NOC coordinate vectors for storage cores
    std::vector<uint32_t> input_storage_noc_x, input_storage_noc_y;
    std::vector<uint32_t> output_storage_noc_x, output_storage_noc_y;
    for (const auto& core : input_storage_cores_ordered) {
        auto phys_core = device->worker_core_from_logical_core(core);
        input_storage_noc_x.push_back(phys_core.x);
        input_storage_noc_y.push_back(phys_core.y);
    }
    for (const auto& core : output_storage_cores_ordered) {
        auto phys_core = device->worker_core_from_logical_core(core);
        output_storage_noc_x.push_back(phys_core.x);
        output_storage_noc_y.push_back(phys_core.y);
    }

    // Bounding box of all cores (workers + storage)
    std::set<CoreRange> all_cores_set;
    for (const auto& core : all_worker_cores_ordered) {
        all_cores_set.insert(CoreRange(core));
    }
    for (const auto& core : input_storage_cores_ordered) {
        all_cores_set.insert(CoreRange(core));
    }
    for (const auto& core : output_storage_cores_ordered) {
        all_cores_set.insert(CoreRange(core));
    }
    CoreRangeSet all_cores(all_cores_set);
    CoreRange bounding_box = all_cores.bounding_box();
    CoreRangeSet all_cores_in_rect_grid({bounding_box});

    uint32_t num_cores = num_workers;
    uint32_t num_dram_banks = device->num_dram_channels();
    uint32_t batches_per_core = (B + num_cores - 1) / num_cores;

    TT_FATAL(
        num_cores <= num_dram_banks,
        "Number of worker cores ({}) cannot exceed number of DRAM banks ({})",
        num_cores,
        num_dram_banks);

    // Subblock parameters
    auto subblock_hw = operations::matmul::bmm_op_utils::get_matmul_subblock_params(
        per_core_M, per_core_N, false, false, fp32_dest_acc_en);
    auto out_subblock_h = std::get<0>(subblock_hw);
    auto out_subblock_w = std::get<1>(subblock_hw);

    uint32_t num_blocks = K / in0_block_w;
    bool packer_l1_acc_en = packer_l1_acc && num_blocks > 1;

    tt::DataFormat interm0_data_format = packer_l1_acc_en
                                             ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                                             : (fp32_dest_acc_en ? tt::DataFormat::Float32 : output_data_format);

    // Tile sizes
    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    uint32_t bias_single_tile_size = bias_tile.get_tile_size(bias_data_format);
    uint32_t output_single_tile_size = output_tile.get_tile_size(output_data_format);
    uint32_t interm0_single_tile_size = output_tile.get_tile_size(interm0_data_format);

    // CB / DFB sizes (in tiles; entry_size carries the per-tile byte size)
    uint32_t in0_block_tiles = per_core_M * in0_block_w;
    uint32_t in0_CB_tiles = in0_block_tiles * 2;

    uint32_t in1_block_tiles = in0_block_w * per_core_N;
    uint32_t in1_CB_tiles = in1_block_tiles * 3;

    uint32_t out_block_tiles = per_core_M * per_core_N;
    uint32_t interm0_CB_tiles = out_block_tiles;

    uint32_t in0_shard_tiles = in0_tensor.shard_spec()->shape[0] / in0_tile.get_tile_shape()[0] *
                               in0_tensor.shard_spec()->shape[1] / in0_tile.get_tile_shape()[1];
    uint32_t in2_CB_size = in0_shard_tiles * in0_single_tile_size;

    uint32_t in3_block_tiles = per_core_N;

    uint32_t out_shard_tiles = out_tensor.shard_spec()->shape[0] / output_tile.get_tile_shape()[0] *
                               out_tensor.shard_spec()->shape[1] / output_tile.get_tile_shape()[1];
    uint32_t out_reshard_CB_tiles = out_shard_tiles;
    uint32_t out_reshard_CB_size = out_shard_tiles * output_single_tile_size;

    // Page sizes for DRAM reads
    uint32_t in1_buffer_page_size, in1_buffer_num_pages;
    get_max_page_size_and_num_pages(
        device, in1_block_tiles, in1_single_tile_size, in1_buffer_page_size, in1_buffer_num_pages);

    uint32_t bias_buffer_page_size, bias_buffer_num_pages;
    get_max_page_size_and_num_pages(
        device, in3_block_tiles, bias_single_tile_size, bias_buffer_page_size, bias_buffer_num_pages);

    // Tensor stride calculations
    uint32_t in0_batch_stride_bytes = per_core_M * K * in0_single_tile_size;
    uint32_t in1_batch_stride_bytes = K * per_core_N * in1_single_tile_size;
    uint32_t out_batch_stride_bytes = per_core_M * per_core_N * output_single_tile_size;

    ////////////////////////////////////////////////////////////////////////////
    //                      ProgramSpec (immutable)
    ////////////////////////////////////////////////////////////////////////////
    m2::ProgramSpec spec;
    spec.name = "matmul_multi_core_reuse_batched_hs_dram_sharded";

    const bool has_bias = bias_tensor.has_value();

    // ---- DataflowBufferSpecs (one per legacy local CBDescriptor) ----
    // c_0: in0 activations (local FIFO on all cores in the bounding box).
    spec.dataflow_buffers.push_back(m2::DataflowBufferSpec{
        .unique_id = m2::DFBSpecName{"in0"},
        .entry_size = in0_single_tile_size,
        .num_entries = in0_CB_tiles,
        .data_format_metadata = in0_data_format,
        .tile_format_metadata = in0_tile});
    // c_1: in1 weights (local FIFO).
    spec.dataflow_buffers.push_back(m2::DataflowBufferSpec{
        .unique_id = m2::DFBSpecName{"in1"},
        .entry_size = in1_single_tile_size,
        .num_entries = in1_CB_tiles,
        .data_format_metadata = in1_data_format,
        .tile_format_metadata = in1_tile});
    // c_3: bias (local FIFO, only when fused).
    if (has_bias) {
        spec.dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"bias"},
            .entry_size = bias_single_tile_size,
            .num_entries = in3_block_tiles,
            .data_format_metadata = bias_data_format,
            .tile_format_metadata = bias_tile});
    }
    // c_4 (out) and c_5 (intermed0). Legacy keeps them as a single CB (one CBDescriptor, two
    // format_descriptors) when the formats match; that aliasing is expressed as two DFBs sharing
    // backing memory via advanced_options.alias_with. The out DFB sizing matches the legacy
    // out_reshard_CB_size; intermed0 uses interm0 tiles. (Both share one L1 region in the aliased
    // case; their entry_size*num_entries must be equal, which holds because the reshard tile count
    // equals out_block_tiles for this batch-sharded layout.)
    if (interm0_data_format != output_data_format) {
        spec.dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"out"},
            .entry_size = output_single_tile_size,
            .num_entries = out_reshard_CB_tiles,
            .data_format_metadata = output_data_format,
            .tile_format_metadata = output_tile});
        spec.dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"intermed0"},
            .entry_size = interm0_single_tile_size,
            .num_entries = interm0_CB_tiles,
            .data_format_metadata = interm0_data_format,
            .tile_format_metadata = output_tile});
    } else {
        m2::DataflowBufferSpec out_dfb{
            .unique_id = m2::DFBSpecName{"out"},
            .entry_size = output_single_tile_size,
            .num_entries = out_reshard_CB_tiles,
            .data_format_metadata = output_data_format,
            .tile_format_metadata = output_tile};
        out_dfb.advanced_options.alias_with = {m2::DFBSpecName{"intermed0"}};
        // Aliased: out and intermed0 share one L1 region (legacy single CBDescriptor sized to
        // out_reshard_CB_size). Alias legality requires entry_size*num_entries equal; here
        // interm0_data_format == output_data_format so the tile sizes match and intermed0 uses
        // the same out_reshard_CB_tiles count, matching legacy's shared allocation size.
        m2::DataflowBufferSpec interm_dfb{
            .unique_id = m2::DFBSpecName{"intermed0"},
            .entry_size = interm0_single_tile_size,
            .num_entries = out_reshard_CB_tiles,
            .data_format_metadata = interm0_data_format,
            .tile_format_metadata = output_tile};
        interm_dfb.advanced_options.alias_with = {m2::DFBSpecName{"out"}};
        spec.dataflow_buffers.push_back(std::move(out_dfb));
        spec.dataflow_buffers.push_back(std::move(interm_dfb));
    }

    // ---- TensorParameters (one per distinct accessed tensor) ----
    // a / out provide the storage-core shard L1 base addresses (Case-2 bridge); b / bias provide
    // the DRAM-bank base addresses. The legacy borrowed CBs c_2/c_6 are subsumed by `a`/`out`.
    spec.tensor_parameters.push_back(
        m2::TensorParameter{.unique_id = m2::TensorParamName{"a"}, .spec = in0_tensor.tensor_spec()});
    spec.tensor_parameters.push_back(
        m2::TensorParameter{.unique_id = m2::TensorParamName{"b"}, .spec = in1_tensor.tensor_spec()});
    if (has_bias) {
        spec.tensor_parameters.push_back(
            m2::TensorParameter{.unique_id = m2::TensorParamName{"bias"}, .spec = bias_tensor->tensor_spec()});
    }
    spec.tensor_parameters.push_back(
        m2::TensorParameter{.unique_id = m2::TensorParamName{"out"}, .spec = out_tensor.tensor_spec()});

    ////////////////////////////////////////////////////////////////////////////
    //                      Kernel defines
    ////////////////////////////////////////////////////////////////////////////
    std::map<std::string, std::string> mm_kernel_defines;
    std::map<std::string, std::string> reader_defines;
    std::map<std::string, std::string> writer_defines;

    if (has_bias) {
        mm_kernel_defines["FUSE_BIAS"] = "1";
        writer_defines["FUSE_BIAS"] = "1";
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
    if (skip_compute) {
        mm_kernel_defines["SKIP_COMPUTE"] = "1";
    }
    if (skip_write_back) {
        writer_defines["SKIP_WRITE_BACK"] = "1";
    }
    mm_kernel_defines["MATMUL_DRAM_SHARDED"] = "1";

    ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
        device->arch(), num_cores, mm_kernel_defines);
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(), num_cores, mm_kernel_defines, throttle_level);

    writer_defines["OUT_SHARDED"] = "1";

    // Helper to convert std::map defines to a Metal 2.0 Defines table.
    auto map_to_defines = [](const std::map<std::string, std::string>& m) -> m2::KernelSpec::CompilerOptions::Defines {
        m2::KernelSpec::CompilerOptions::Defines result;
        for (const auto& [k, v] : m) {
            result.insert({k, v});
        }
        return result;
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      Subblock / compute parameters
    ////////////////////////////////////////////////////////////////////////////
    uint32_t in0_num_subblocks = per_core_M / out_subblock_h;
    uint32_t in1_num_subblocks = per_core_N / out_subblock_w;
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;
    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    ////////////////////////////////////////////////////////////////////////////
    //                      Build KernelSpecs
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::NOC in0_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());

    // in0 reader kernel
    m2::KernelSpec in0_reader{
        .unique_id = m2::KernelSpecName{"in0_reader"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
                                        "reader_bmm_tile_layout_in0_sender_dram_sharded_height.cpp"},
        .compiler_options = {.defines = map_to_defines(reader_defines)},
        .compile_time_args =
            {{"in0_block_num_tiles", in0_block_tiles},
             {"in0_block_size_bytes", in0_block_tiles * in0_single_tile_size},
             {"num_blocks", num_blocks},
             {"num_batches_per_core", batches_per_core},
             {"in0_tensor_stride_batch_bytes", in0_batch_stride_bytes},
             {"in0_shard_size_bytes", in2_CB_size}},
        .runtime_arg_schema = {.runtime_arg_names = {"worker_core_type", "input_storage_noc_x", "input_storage_noc_y"}},
        .hw_config =
            m2::DataMovementHardwareConfig{
                .gen1_config =
                    m2::DataMovementHardwareConfig::Gen1Config{
                        .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = in0_noc}},
    };
    in0_reader.dfb_bindings.push_back(m2::DFBBinding{
        .dfb_spec_name = m2::DFBSpecName{"in0"},
        .accessor_name = "cb_in0",
        .endpoint_type = m2::DFBEndpointType::PRODUCER});
    in0_reader.tensor_bindings.push_back(
        m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"a"}, .accessor_name = "a"});

    // in1 reader / output writer kernel
    m2::KernelSpec::CompilerOptions::Defines writer_defines_table = map_to_defines(writer_defines);
    m2::KernelSpec in1_writer{
        .unique_id = m2::KernelSpecName{"in1_writer"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
                                        "reader_bmm_tile_layout_in1_sender_dram_sharded_height.cpp"},
        .compiler_options = {.defines = writer_defines_table},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"is_worker_core", "dram_bank_id", "vc", "output_storage_noc_x", "output_storage_noc_y"}},
        .hw_config =
            m2::DataMovementHardwareConfig{
                .gen1_config =
                    m2::DataMovementHardwareConfig::Gen1Config{
                        .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = in1_noc}},
    };
    {
        m2::KernelSpec::CompileTimeArgs writer_cta = {
            {"in1_page_size", in1_buffer_page_size},
            {"in1_num_pages", in1_buffer_num_pages},
            {"in1_block_w", per_core_N},
            {"in1_block_num_tiles", in1_block_tiles},
            {"num_blocks", num_blocks},
            {"out_block_num_tiles", out_block_tiles},
            {"num_batches_per_core", batches_per_core},
            {"in1_tensor_stride_batch_bytes", in1_batch_stride_bytes},
            {"out_tensor_stride_batch_bytes", out_batch_stride_bytes},
            {"out_shard_size_bytes", out_reshard_CB_size}};
        if (has_bias) {
            writer_cta.insert({"in3_page_size", bias_buffer_page_size});
            writer_cta.insert({"in3_num_pages", bias_buffer_num_pages});
            writer_cta.insert({"in3_block_tiles", in3_block_tiles});
        }
        in1_writer.compile_time_args = std::move(writer_cta);
    }
    in1_writer.dfb_bindings.push_back(m2::DFBBinding{
        .dfb_spec_name = m2::DFBSpecName{"in1"},
        .accessor_name = "cb_in1",
        .endpoint_type = m2::DFBEndpointType::PRODUCER});
    in1_writer.dfb_bindings.push_back(m2::DFBBinding{
        .dfb_spec_name = m2::DFBSpecName{"out"},
        .accessor_name = "cb_out",
        .endpoint_type = m2::DFBEndpointType::CONSUMER});
    if (has_bias) {
        in1_writer.dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{"bias"},
            .accessor_name = "cb_bias",
            .endpoint_type = m2::DFBEndpointType::PRODUCER});
    }
    in1_writer.tensor_bindings.push_back(
        m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"b"}, .accessor_name = "b"});
    if (has_bias) {
        in1_writer.tensor_bindings.push_back(
            m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"bias"}, .accessor_name = "bias"});
    }
    in1_writer.tensor_bindings.push_back(
        m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"out"}, .accessor_name = "out"});

    // compute kernel (forked Metal 2.0 source). CTA layout matches the legacy positional emission
    // order; num_blocks_w_dim and num_blocks_h_dim are both 1, batch is batches_per_core, and the
    // in0-transpose / get-batch-from-reader paths are off for this batch-sharded factory.
    const char* COMPUTE_SRC =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_m2.cpp";
    m2::KernelSpec::CompileTimeArgs compute_cta = {
        {"in0_block_w", in0_block_w},
        {"in0_num_subblocks", in0_num_subblocks},
        {"in0_block_num_tiles", in0_block_tiles},
        {"in0_subblock_num_tiles", in0_subblock_num_tiles},
        {"in1_num_subblocks", in1_num_subblocks},
        {"in1_block_num_tiles", in1_block_tiles},
        {"in1_block_w", per_core_N},
        {"num_blocks_inner_dim", num_blocks},
        {"num_blocks_w_dim", 1},
        {"num_blocks_h_dim", 1},
        {"out_subblock_h", out_subblock_h},
        {"out_subblock_w", out_subblock_w},
        {"out_subblock_num_tiles", out_subblock_num_tiles},
        {"batch", batches_per_core},
        {"out_block_num_tiles", out_block_tiles},
        {"untilize_out", untilize_out ? 1u : 0u},
        {"get_batch_from_reader", 0u},
        {"in0_transpose_tile", 0u},
        // bias_ntiles / last_subblock_w_valid are named CTAs read by the compute kernel; this
        // factory does not pad per_core_N beyond per_core_N (last subblock always fully valid),
        // so last_subblock_w_valid == out_subblock_w (kernel takes its full-width path).
        {"bias_ntiles", per_core_N},
        {"last_subblock_w_valid", out_subblock_w}};
    if (has_bias) {
        // row_broadcast_bias: DRAM sharded always uses row broadcast.
        compute_cta.insert({"row_broadcast_bias", 1u});
    }
    if (fused_activation.has_value() && fused_activation.value().op_type != UnaryOpType::RELU) {
        using ttnn::operations::matmul::utilities::get_activation_params;
        const auto params = get_activation_params(fused_activation.value());
        compute_cta.insert({"activation_type", static_cast<uint32_t>(params.type)});
        compute_cta.insert({"activation_param0", params.param0});
        compute_cta.insert({"activation_param1", params.param1});
        compute_cta.insert({"activation_param2", params.param2});
    }

    m2::KernelSpec compute{
        .unique_id = m2::KernelSpecName{"compute"},
        .source = std::filesystem::path{COMPUTE_SRC},
        .compiler_options = {.defines = map_to_defines(mm_kernel_defines)},
        .compile_time_args = std::move(compute_cta),
        .runtime_arg_schema = {.runtime_arg_names = {"is_worker_core"}},
        .hw_config =
            m2::ComputeHardwareConfig{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .dst_full_sync_en = dst_full_sync_en,
                .math_approx_mode = math_approx_mode},
    };
    compute.dfb_bindings.push_back(m2::DFBBinding{
        .dfb_spec_name = m2::DFBSpecName{"in0"},
        .accessor_name = "cb_in0",
        .endpoint_type = m2::DFBEndpointType::CONSUMER});
    compute.dfb_bindings.push_back(m2::DFBBinding{
        .dfb_spec_name = m2::DFBSpecName{"in1"},
        .accessor_name = "cb_in1",
        .endpoint_type = m2::DFBEndpointType::CONSUMER});
    compute.dfb_bindings.push_back(m2::DFBBinding{
        .dfb_spec_name = m2::DFBSpecName{"out"},
        .accessor_name = "cb_out",
        .endpoint_type = m2::DFBEndpointType::PRODUCER});
    // intermed0: PRODUCER + CONSUMER on compute (the kernel spills partials and reloads them — a
    // real self-loop).
    compute.dfb_bindings.push_back(m2::DFBBinding{
        .dfb_spec_name = m2::DFBSpecName{"intermed0"},
        .accessor_name = "cb_intermed0",
        .endpoint_type = m2::DFBEndpointType::PRODUCER});
    compute.dfb_bindings.push_back(m2::DFBBinding{
        .dfb_spec_name = m2::DFBSpecName{"intermed0"},
        .accessor_name = "cb_intermed0",
        .endpoint_type = m2::DFBEndpointType::CONSUMER});
    if (has_bias) {
        compute.dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{"bias"},
            .accessor_name = "cb_bias",
            .endpoint_type = m2::DFBEndpointType::CONSUMER});
    }

    // ---- Kernels + WorkUnitSpec ----
    // All three kernels run on the full bounding box (workers + idle storage cores). Idle cores
    // supply only worker_core_type/is_worker_core == 0 and return early. The local DFBs require
    // their producer & consumer KernelSpecs to share the same WorkUnitSpec, so there is one
    // WorkUnitSpec hosting in0_reader + in1_writer + compute on all_cores_in_rect_grid.
    spec.kernels = {in0_reader, in1_writer, compute};
    spec.work_units = std::vector<m2::WorkUnitSpec>{m2::WorkUnitSpec{
        .name = "wu",
        .kernels = {m2::KernelSpecName{"in0_reader"}, m2::KernelSpecName{"in1_writer"}, m2::KernelSpecName{"compute"}},
        .target_nodes = all_cores_in_rect_grid}};

    ////////////////////////////////////////////////////////////////////////////
    //                      ProgramRunArgs (mutable)
    ////////////////////////////////////////////////////////////////////////////
    std::vector<CoreCoord> all_cores_in_rect_grid_vec = corerange_to_cores(all_cores_in_rect_grid);
    std::set<CoreCoord> worker_cores_set(all_worker_cores_ordered.begin(), all_worker_cores_ordered.end());

    m2::ProgramRunArgs run;
    m2::KernelRunArgs in0_reader_run{.kernel = m2::KernelSpecName{"in0_reader"}};
    m2::KernelRunArgs in1_writer_run{.kernel = m2::KernelSpecName{"in1_writer"}};
    m2::KernelRunArgs compute_run{.kernel = m2::KernelSpecName{"compute"}};

    // Idle cores in the bounding box: worker_core_type / is_worker_core == 0, dummy values for the
    // remaining named args (the kernels return early before reading them).
    for (const auto& core : all_cores_in_rect_grid_vec) {
        if (!worker_cores_set.contains(core)) {
            in0_reader_run.runtime_arg_values.push_back(
                {core, {{"worker_core_type", 0u}, {"input_storage_noc_x", 0u}, {"input_storage_noc_y", 0u}}});
            in1_writer_run.runtime_arg_values.push_back(
                {core,
                 {{"is_worker_core", 0u},
                  {"dram_bank_id", 0u},
                  {"vc", 0u},
                  {"output_storage_noc_x", 0u},
                  {"output_storage_noc_y", 0u}}});
            compute_run.runtime_arg_values.push_back({core, {{"is_worker_core", 0u}}});
        }
    }

    // Worker cores
    std::vector<uint32_t> bank_ids;
    for (uint32_t worker_idx = 0; worker_idx < all_worker_cores_ordered.size(); ++worker_idx) {
        auto core = all_worker_cores_ordered[worker_idx];

        uint32_t bank_id = worker_idx;
        uint32_t vc = bank_id & 0x3;
        bank_ids.push_back(bank_id);
        for (uint32_t j = 0; j < worker_idx; ++j) {
            auto core_prev = all_worker_cores_ordered[j];
            if (core_prev.y == core.y && ((bank_id & 0x3) == (bank_ids[j] & 0x3))) {
                vc = (vc + 1) & 0x3;
                break;
            }
        }

        in0_reader_run.runtime_arg_values.push_back(
            {core,
             {{"worker_core_type", 1u},
              {"input_storage_noc_x", input_storage_noc_x[worker_idx]},
              {"input_storage_noc_y", input_storage_noc_y[worker_idx]}}});
        in1_writer_run.runtime_arg_values.push_back(
            {core,
             {{"is_worker_core", 1u},
              {"dram_bank_id", bank_id},
              {"vc", vc},
              {"output_storage_noc_x", output_storage_noc_x[worker_idx]},
              {"output_storage_noc_y", output_storage_noc_y[worker_idx]}}});
        compute_run.runtime_arg_values.push_back({core, {{"is_worker_core", 1u}}});
    }

    run.kernel_run_args = {in0_reader_run, in1_writer_run, compute_run};
    run.tensor_args.insert({m2::TensorParamName{"a"}, in0_tensor.mesh_tensor()});
    run.tensor_args.insert({m2::TensorParamName{"b"}, in1_tensor.mesh_tensor()});
    if (has_bias) {
        run.tensor_args.insert({m2::TensorParamName{"bias"}, bias_tensor->mesh_tensor()});
    }
    run.tensor_args.insert({m2::TensorParamName{"out"}, out_tensor.mesh_tensor()});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run)};
}

}  // namespace reuse_batched_hs_dram_sharded_optimized_helpers

ttnn::device_operation::ProgramArtifacts MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory::create_program_spec(
    const ttnn::prim::MatmulParams& operation_attributes,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    const auto& input_tensors = tensor_args.input_tensors;
    const auto& optional_input_tensors = tensor_args.optional_input_tensors;
    auto& output_tensors = tensor_return_value;

    const auto& a = input_tensors.at(0);
    const auto& b = input_tensors.at(1);
    std::optional<const ttnn::Tensor> bias =
        optional_input_tensors.empty() ? std::nullopt : optional_input_tensors.at(0);
    const auto& output = output_tensors.at(0);
    const auto& a_mesh = a.mesh_tensor();
    const auto& b_mesh = b.mesh_tensor();
    [[maybe_unused]] const auto& out_mesh = output.mesh_tensor();
    const auto& ashape = a_mesh.padded_shape();
    const auto& bshape = b_mesh.padded_shape();
    auto in0_tile = a.tensor_spec().tile();
    auto in1_tile = b.tensor_spec().tile();
    auto in0_tile_shape = in0_tile.get_tile_shape();
    auto in1_tile_shape = in1_tile.get_tile_shape();
    auto output_tile = tt::tt_metal::Tile({in0_tile.get_tile_shape()[0], in1_tile.get_tile_shape()[1]});

    // CB dataformats
    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(b.dtype());
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());

    tt::DataFormat bias_data_format = tt::DataFormat::Bfp8_b;
    if (bias.has_value()) {
        const auto& c = bias.value().mesh_tensor();
        TT_FATAL(&a_mesh.device() == &c.device(), "Operands to matmul need to be on the same device!");
        bias_data_format = tt_metal::datatype_to_dataformat_converter(bias.value().dtype());
    }

    tt::tt_metal::IDevice* device = &a_mesh.mutable_device();

    TT_FATAL(
        a.shard_spec().has_value() && output.shard_spec().has_value(), "Both input A and output must have shard specs");
    CoreRangeSet input_all_cores_storage = a.shard_spec().value().grid;
    CoreRangeSet output_all_cores_storage = output.shard_spec().value().grid;

    uint32_t in0_single_tile_size = in0_tile.get_tile_size(tt_metal::datatype_to_dataformat_converter(a.dtype()));
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(tt_metal::datatype_to_dataformat_converter(b.dtype()));

    TT_FATAL(
        a_mesh.mesh_buffer().device_local_size() % in0_single_tile_size == 0,
        "Input A buffer size ({}) must be divisible by single tile size ({})",
        a_mesh.mesh_buffer().device_local_size(),
        in0_single_tile_size);
    TT_FATAL(
        b_mesh.mesh_buffer().device_local_size() % in1_single_tile_size == 0,
        "Input B buffer size ({}) must be divisible by single tile size ({})",
        b_mesh.mesh_buffer().device_local_size(),
        in1_single_tile_size);

    TT_FATAL(
        ashape[-1] == bshape[-2],
        "Dimension K (A.shape[-1] = {}, B.shape[-2] = {}) must match for matmul",
        ashape[-1],
        bshape[-2]);
    TT_FATAL(ashape[-2] % in0_tile_shape[0] == 0, "A.shape[-2] must be divisible by tile shape[0]");
    TT_FATAL(ashape[-1] % in0_tile_shape[1] == 0, "A.shape[-1] must be divisible by tile shape[1]");
    TT_FATAL(bshape[-2] % in1_tile_shape[0] == 0, "B.shape[-2] must be divisible by tile shape[0]");
    TT_FATAL(bshape[-1] % in1_tile_shape[1] == 0, "B.shape[-1] must be divisible by tile shape[1]");

    const auto& compute_kernel_config = operation_attributes.compute_kernel_config.value();
    const auto& program_config =
        std::get<operations::matmul::MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig>(
            operation_attributes.program_config.value());
    const auto& in0_block_w = program_config.in0_block_w;
    const auto& per_core_M = program_config.per_core_M;
    const auto& per_core_N = program_config.per_core_N;
    const auto& fused_activation = program_config.fused_activation;
    const auto& untilize_out = operation_attributes.untilize_out;

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    uint32_t B = ashape[1];
    uint32_t M = ashape[-2] / in0_tile_shape[0];
    uint32_t K = ashape[-1] / in0_tile_shape[1];
    uint32_t N = bshape[-1] / in1_tile_shape[1];

    TT_FATAL(per_core_M == M, "For batch sharding, per_core_M ({}) must equal M ({})", per_core_M, M);
    TT_FATAL(per_core_N == N, "For batch sharding, per_core_N ({}) must equal N ({})", per_core_N, N);
    TT_FATAL(K % in0_block_w == 0, "K ({}) must be divisible by in0_block_w ({})", K, in0_block_w);

    return reuse_batched_hs_dram_sharded_optimized_helpers::create_program_batch_sharded_spec(
        device,
        input_all_cores_storage,
        output_all_cores_storage,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode,
        packer_l1_acc,
        dst_full_sync_en,
        ttnn::get_throttle_level(operation_attributes.compute_kernel_config),
        B,
        M,
        K,
        N,
        in0_block_w,
        per_core_M,
        per_core_N,
        fused_activation,
        a,
        b,
        bias,
        output,
        in0_tile,
        in1_tile,
        bias.has_value() ? bias->tensor_spec().tile() : output_tile,
        output_tile,
        in0_data_format,
        in1_data_format,
        bias_data_format,
        output_data_format,
        untilize_out,
        false,   // skip_compute
        false);  // skip_write_back
}

}  // namespace ttnn::prim
