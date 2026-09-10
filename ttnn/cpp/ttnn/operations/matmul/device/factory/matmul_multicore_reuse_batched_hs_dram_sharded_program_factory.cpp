// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_batched_hs_dram_sharded_program_factory.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <utility>

#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/matmul/shared_with_host/activation_type.hpp"

using namespace tt;

using ttnn::operations::unary::UnaryOpType;
using ttnn::operations::unary::UnaryWithParam;

using tt::tt_metal::KernelBuildOptLevel;
using tt::tt_metal::UnpackMode;
using tt::tt_metal::experimental::AddRuntimeArgsForNode;
using tt::tt_metal::experimental::ComputeHardwareConfig;
using tt::tt_metal::experimental::DataflowBufferSpec;
using tt::tt_metal::experimental::DataMovementGen1Config;
using tt::tt_metal::experimental::DFBBinding;
using tt::tt_metal::experimental::DFBEndpointType;
using tt::tt_metal::experimental::DFBSpecName;
using tt::tt_metal::experimental::Group;
using tt::tt_metal::experimental::KernelRunArgs;
using tt::tt_metal::experimental::KernelSpec;
using tt::tt_metal::experimental::KernelSpecName;
using tt::tt_metal::experimental::ProgramRunArgs;
using tt::tt_metal::experimental::ProgramSpec;
using tt::tt_metal::experimental::TensorBinding;
using tt::tt_metal::experimental::TensorParameter;
using tt::tt_metal::experimental::TensorParamName;
using tt::tt_metal::experimental::unpack_modes;
using tt::tt_metal::experimental::WorkUnitSpec;

namespace ttnn::prim {
namespace reuse_batched_hs_dram_sharded_optimized_helpers {

using dram_sharded_helpers::get_max_page_size_and_num_pages;
using dram_sharded_helpers::get_optimal_dram_bank_to_reader_assignment;

// Batch-sharded DRAM matmul
// For batched matmul: [1, B, M, K] x [1, B, K, N] = [1, B, M, N]
// Sharded by batch dimension - each worker handles B/num_workers complete matmuls
static ttnn::device_operation::ProgramArtifacts create_program_batch_sharded_spec(
    tt::tt_metal::IDevice* device,
    const CoreRangeSet& input_all_storage_cores,
    const CoreRangeSet& output_all_storage_cores,
    ComputeHardwareConfig compute_hw,
    bool fp32_dest_acc_en,
    bool packer_l1_acc,
    ttnn::operations::compute_throttle_utils::ThrottleLevel throttle_level,
    uint32_t B,
    uint32_t /* M */,
    uint32_t K,
    uint32_t /* N */,
    uint32_t in0_block_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    std::optional<UnaryWithParam> fused_activation,
    const tt_metal::MeshTensor& in0_tensor,
    const tt_metal::MeshTensor& in1_tensor,
    ttsl::optional_reference<const tt_metal::MeshTensor> bias_tensor,
    const tt_metal::MeshTensor& out_tensor,
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

    // Dataflow buffer entry counts
    uint32_t in0_block_tiles = per_core_M * in0_block_w;
    uint32_t in0_num_entries = in0_block_tiles * 2;

    uint32_t in1_block_tiles = in0_block_w * per_core_N;
    uint32_t in1_num_entries = in1_block_tiles * 3;

    uint32_t out_block_tiles = per_core_M * per_core_N;
    uint32_t interm0_num_entries = out_block_tiles;

    uint32_t in0_shard_tiles = in0_tensor.shard_spec()->shape[0] / in0_tile.get_tile_shape()[0] *
                               in0_tensor.shard_spec()->shape[1] / in0_tile.get_tile_shape()[1];
    uint32_t in0_shard_size_bytes = in0_shard_tiles * in0_single_tile_size;

    uint32_t in3_block_tiles = per_core_N;

    uint32_t out_shard_tiles = out_tensor.shard_spec()->shape[0] / output_tile.get_tile_shape()[0] *
                               out_tensor.shard_spec()->shape[1] / output_tile.get_tile_shape()[1];
    uint32_t out_num_entries = out_shard_tiles;
    uint32_t out_shard_size_bytes = out_shard_tiles * output_single_tile_size;

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

    const KernelSpecName IN0_READER{"in0_reader"};
    const KernelSpecName IN1_WRITER{"in1_writer"};
    const KernelSpecName COMPUTE{"compute"};

    const DFBSpecName IN0_DFB{"in0"};
    const DFBSpecName IN1_DFB{"in1"};
    const DFBSpecName BIAS_DFB{"bias"};
    const DFBSpecName OUT_DFB{"out"};
    const DFBSpecName INTERMED0_DFB{"intermed0"};

    const TensorParamName IN0{"in0"};
    const TensorParamName IN1{"in1"};
    const TensorParamName BIAS{"bias"};
    const TensorParamName OUTPUT{"output"};

    ////////////////////////////////////////////////////////////////////////////
    //                      Build DataflowBufferSpecs
    ////////////////////////////////////////////////////////////////////////////

    // Legacy carried two further borrowed buffers here - a view of in0's L1 shard on the input
    // storage cores and one of the output's on the output storage cores. Neither had a single
    // endpoint: no kernel bound either index and no named argument carried it, because the kernels
    // reach that memory by explicit NOC address from a tensor binding instead. Metal 2.0 cannot
    // express a buffer with no producer and no consumer, and a buffer nothing touches has no
    // behaviour, so they are dropped. The sibling DRAM-sharded factory still uses that idiom for
    // real - there the buffer is how the kernel obtains the shard's base address - which is where
    // these two came from.

    // in0 arrives from the input storage core one block at a time; double buffered.
    DataflowBufferSpec in0_dfb_spec{
        .unique_id = IN0_DFB,
        .entry_size = in0_single_tile_size,
        .num_entries = in0_num_entries,
        .data_format_metadata = in0_data_format,
        .tile_format_metadata = in0_tile,
    };
    DataflowBufferSpec in1_dfb_spec{
        .unique_id = IN1_DFB,
        .entry_size = in1_single_tile_size,
        .num_entries = in1_num_entries,
        .data_format_metadata = in1_data_format,
        .tile_format_metadata = in1_tile,
    };
    DataflowBufferSpec out_dfb_spec{
        .unique_id = OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = out_num_entries,
        .data_format_metadata = output_data_format,
        .tile_format_metadata = output_tile,
    };
    // The partials buffer. When its format matches the output's, legacy placed both buffer indices
    // on a single descriptor - two logically distinct buffers over one L1 region, sized for the
    // output - so the two become an alias pair at that size. Otherwise they are independent, and
    // the partials buffer is sized for the compute block rather than the output shard.
    const bool share_out_interm_buffer = interm0_data_format == output_data_format;
    DataflowBufferSpec intermed0_dfb_spec{
        .unique_id = INTERMED0_DFB,
        .entry_size = interm0_single_tile_size,
        .num_entries = share_out_interm_buffer ? out_num_entries : interm0_num_entries,
        .data_format_metadata = interm0_data_format,
        .tile_format_metadata = output_tile,
    };
    if (share_out_interm_buffer) {
        out_dfb_spec.advanced_options.alias_with = {INTERMED0_DFB};
        intermed0_dfb_spec.advanced_options.alias_with = {OUT_DFB};
    }

    Group<DataflowBufferSpec> dataflow_buffers;
    dataflow_buffers.reserve(bias_tensor.has_value() ? 5 : 4);
    dataflow_buffers.push_back(std::move(in0_dfb_spec));
    dataflow_buffers.push_back(std::move(in1_dfb_spec));
    dataflow_buffers.push_back(std::move(out_dfb_spec));
    dataflow_buffers.push_back(std::move(intermed0_dfb_spec));
    if (bias_tensor.has_value()) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = BIAS_DFB,
            .entry_size = bias_single_tile_size,
            .num_entries = in3_block_tiles,
            .data_format_metadata = bias_data_format,
            .tile_format_metadata = bias_tile,
        });
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Kernel defines
    ////////////////////////////////////////////////////////////////////////////
    std::map<std::string, std::string> mm_kernel_defines;
    std::map<std::string, std::string> reader_defines;
    std::map<std::string, std::string> writer_defines;

    if (bias_tensor.has_value()) {
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
    // No IN0_TRANSPOSE_TILE: this factory never transposes in0 tiles, so the compute kernel's
    // in0_transposed buffer is left unbound and its name never enters lookup. No
    // MM_PARTIALS_RELOAD_ALIAS either: the partials reload copies through the partials buffer
    // itself here, as it did before the port.

    ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
        device->arch(), num_cores, mm_kernel_defines);
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(), num_cores, mm_kernel_defines, throttle_level);

    tt_metal::NOC in0_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());

    writer_defines["OUT_SHARDED"] = "1";

    ////////////////////////////////////////////////////////////////////////////
    //                      Runtime Args (per-core loops)
    ////////////////////////////////////////////////////////////////////////////
    std::vector<CoreCoord> all_cores_in_rect_grid_vec = corerange_to_cores(all_cores_in_rect_grid);
    std::set<CoreCoord> worker_cores_set(all_worker_cores_ordered.begin(), all_worker_cores_ordered.end());

    KernelRunArgs in0_run_args{.kernel = IN0_READER};
    KernelRunArgs in1_run_args{.kernel = IN1_WRITER};
    KernelRunArgs compute_run_args{.kernel = COMPUTE};

    std::vector<uint32_t> bank_ids;
    bank_ids.reserve(all_worker_cores_ordered.size());

    // Idle cores in the bounding box. Legacy emitted a one-element argument list here; every name
    // in a kernel's schema needs a value on every node it runs on, so the rest are zero-filled.
    // Both data movement kernels return on the worker test before reading any of them.
    for (const auto& core : all_cores_in_rect_grid_vec) {
        bool is_worker = worker_cores_set.contains(core);

        if (!is_worker) {
            AddRuntimeArgsForNode(
                in0_run_args.runtime_arg_values,
                core,
                {{"worker_core_type", 0u}, {"input_storage_noc_x", 0u}, {"input_storage_noc_y", 0u}});

            AddRuntimeArgsForNode(
                in1_run_args.runtime_arg_values,
                core,
                {{"is_worker_core", 0u},
                 {"dram_bank_id", 0u},
                 {"vc", 0u},
                 {"output_storage_noc_x", 0u},
                 {"output_storage_noc_y", 0u}});

            AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, core, {{"is_worker_core", 0u}});
        }
    }

    // Worker cores
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

        // in0 reader runtime args. The in0 shard's base address arrives as a tensor binding.
        AddRuntimeArgsForNode(
            in0_run_args.runtime_arg_values,
            core,
            {{"worker_core_type", 1u},
             {"input_storage_noc_x", input_storage_noc_x[worker_idx]},
             {"input_storage_noc_y", input_storage_noc_y[worker_idx]}});

        // in1 reader / output writer runtime args. The in1, bias and output base addresses arrive
        // as tensor bindings.
        AddRuntimeArgsForNode(
            in1_run_args.runtime_arg_values,
            core,
            {{"is_worker_core", 1u},
             {"dram_bank_id", bank_id},
             {"vc", vc},
             {"output_storage_noc_x", output_storage_noc_x[worker_idx]},
             {"output_storage_noc_y", output_storage_noc_y[worker_idx]}});

        AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, core, {{"is_worker_core", 1u}});
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Build KernelSpecs
    ////////////////////////////////////////////////////////////////////////////

    // in0 reader kernel
    KernelSpec in0_reader{
        .unique_id = IN0_READER,
        .source =
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
            "reader_bmm_tile_layout_in0_sender_dram_sharded_height.cpp",
        .compiler_options = {.defines = KernelSpec::CompilerOptions::Defines(reader_defines)},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = IN0_DFB,
                    .accessor_name = "in0",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = IN0,
                    .accessor_name = "in0",
                },
            },
        .compile_time_args =
            {
                {"in0_block_num_tiles", in0_block_tiles},
                {"in0_block_size_bytes", in0_block_tiles * in0_single_tile_size},
                {"num_blocks", num_blocks},
                {"num_batches_per_core", batches_per_core},
                {"in0_tensor_stride_batch_bytes", in0_batch_stride_bytes},
                {"in0_shard_size_bytes", in0_shard_size_bytes},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"worker_core_type", "input_storage_noc_x", "input_storage_noc_y"},
            },
        .hw_config = DataMovementGen1Config{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = in0_noc},
    };

    // in1 reader / output writer kernel
    KernelSpec in1_writer{
        .unique_id = IN1_WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
            "reader_bmm_tile_layout_in1_sender_dram_sharded_height.cpp",
        .compiler_options = {.defines = KernelSpec::CompilerOptions::Defines(writer_defines)},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = IN1_DFB,
                    .accessor_name = "in1",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = OUT_DFB,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = IN1,
                    .accessor_name = "in1",
                },
                TensorBinding{
                    .tensor_parameter_name = OUTPUT,
                    .accessor_name = "output",
                },
            },
        .compile_time_args =
            {
                {"in1_page_size", in1_buffer_page_size},
                {"in1_num_pages", in1_buffer_num_pages},
                {"in1_block_w", per_core_N},
                {"in1_block_num_tiles", in1_block_tiles},
                {"num_blocks", num_blocks},
                {"out_block_num_tiles", out_block_tiles},
                {"num_batches_per_core", batches_per_core},
                {"in1_tensor_stride_batch_bytes", in1_batch_stride_bytes},
                {"out_tensor_stride_batch_bytes", out_batch_stride_bytes},
                {"out_shard_size_bytes", out_shard_size_bytes},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {"is_worker_core", "dram_bank_id", "vc", "output_storage_noc_x", "output_storage_noc_y"},
            },
        .hw_config = DataMovementGen1Config{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = in1_noc},
    };
    if (bias_tensor.has_value()) {
        in1_writer.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = BIAS_DFB,
            .accessor_name = "bias",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        in1_writer.tensor_bindings.push_back(TensorBinding{
            .tensor_parameter_name = BIAS,
            .accessor_name = "bias",
        });
        in1_writer.compile_time_args.insert({"in3_page_size", bias_buffer_page_size});
        in1_writer.compile_time_args.insert({"in3_num_pages", bias_buffer_num_pages});
        in1_writer.compile_time_args.insert({"in3_block_tiles", in3_block_tiles});
    }

    // compute kernel
    uint32_t in0_num_subblocks = per_core_M / out_subblock_h;
    uint32_t in1_num_subblocks = per_core_N / out_subblock_w;
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;
    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    // UnpackToSrc is the framework default and the legacy compute config left unpack_to_dest_mode
    // empty, so these entries change nothing - they are stated because an explicit mode is
    // required for any 32-bit-format buffer once the Dest register holds 32-bit elements, which
    // fp32_dest_acc_en both sets and forces on the partials format.
    unpack_modes(compute_hw) = {
        {IN0_DFB, UnpackMode::UnpackToSrc},
        {IN1_DFB, UnpackMode::UnpackToSrc},
        {INTERMED0_DFB, UnpackMode::UnpackToSrc},
    };
    if (bias_tensor.has_value()) {
        unpack_modes(compute_hw).insert({BIAS_DFB, UnpackMode::UnpackToSrc});
    }

    KernelSpec compute{
        .unique_id = COMPUTE,
        .source =
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/"
            "bmm_large_block_zm_fused_bias_activation_metal2.cpp",
        .compiler_options =
            {
                .defines = KernelSpec::CompilerOptions::Defines(mm_kernel_defines),
                // Explicit, not a copied default: Metal 2.0's type-agnostic CompilerOptions
                // defaults to O2, while the legacy compute config this replaces defaulted compute
                // kernels to O3. Left unset, the kernel would quietly drop a level. The two data
                // movement specs carry no opt_level for the same reason - their legacy default was
                // already O2.
                .opt_level = KernelBuildOptLevel::O3,
            },
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = IN0_DFB,
                    .accessor_name = "in0",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = IN1_DFB,
                    .accessor_name = "in1",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = OUT_DFB,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                // The partials buffer is compute's alone: it packs into it and reads it back for
                // accumulation and the bias add, so it holds both endpoints.
                DFBBinding{
                    .dfb_spec_name = INTERMED0_DFB,
                    .accessor_name = "intermed0",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = INTERMED0_DFB,
                    .accessor_name = "intermed0",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .compile_time_args =
            {
                {"in0_block_w", in0_block_w},
                {"in0_num_subblocks", in0_num_subblocks},
                {"in0_block_num_tiles", in0_block_tiles},
                {"in0_subblock_num_tiles", in0_subblock_num_tiles},
                {"in1_num_subblocks", in1_num_subblocks},
                {"in1_block_num_tiles", in1_block_tiles},
                {"in1_block_w", per_core_N},
                {"num_blocks_inner_dim", num_blocks},
                {"num_blocks_w_dim", 1u},
                {"num_blocks_h_dim", 1u},
                {"out_subblock_h", out_subblock_h},
                {"out_subblock_w", out_subblock_w},
                {"out_subblock_num_tiles", out_subblock_num_tiles},
                {"batch", batches_per_core},
                {"out_block_num_tiles", out_block_tiles},
                {"untilize_out", untilize_out ? 1u : 0u},
                {"get_batch_from_reader", 0u},
                // This factory does not pad per_core_N_compute beyond per_core_N_in1_sender, so the
                // last subblock is always fully valid. Pass out_subblock_w so the compute kernel
                // takes its original full-width path (last_subblock_padded == false).
                {"last_subblock_w_valid", out_subblock_w},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"is_worker_core"},
            },
        .hw_config = std::move(compute_hw),
    };
    if (bias_tensor.has_value()) {
        compute.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = BIAS_DFB,
            .accessor_name = "bias",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        compute.compile_time_args.insert({"bias_ntiles", per_core_N});
        // row_broadcast_bias: DRAM sharded always uses row broadcast
        compute.compile_time_args.insert({"row_broadcast_bias", 1u});
    }
    if (fused_activation.has_value() && fused_activation.value().op_type != UnaryOpType::RELU) {
        using ttnn::operations::matmul::utilities::get_activation_params;
        const auto params = get_activation_params(fused_activation.value());
        compute.compile_time_args.insert({"activation_type", static_cast<uint32_t>(params.type)});
        compute.compile_time_args.insert({"activation_param0", params.param0});
        compute.compile_time_args.insert({"activation_param1", params.param1});
        compute.compile_time_args.insert({"activation_param2", params.param2});
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Assemble
    ////////////////////////////////////////////////////////////////////////////

    Group<KernelSpec> kernels;
    kernels.reserve(3);
    kernels.push_back(std::move(in0_reader));
    kernels.push_back(std::move(in1_writer));
    kernels.push_back(std::move(compute));

    Group<TensorParameter> tensor_parameters;
    tensor_parameters.reserve(bias_tensor.has_value() ? 4 : 3);
    tensor_parameters.push_back(TensorParameter{.unique_id = IN0, .spec = in0_tensor.tensor_spec()});
    tensor_parameters.push_back(TensorParameter{.unique_id = IN1, .spec = in1_tensor.tensor_spec()});
    tensor_parameters.push_back(TensorParameter{.unique_id = OUTPUT, .spec = out_tensor.tensor_spec()});
    if (bias_tensor.has_value()) {
        tensor_parameters.push_back(TensorParameter{.unique_id = BIAS, .spec = bias_tensor->tensor_spec()});
    }

    ProgramSpec spec{
        .name = "matmul_multi_core_reuse_batched_hs_dram_sharded",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units =
            {
                WorkUnitSpec{
                    .name = "all_cores_in_rect_grid",
                    .kernels = {IN0_READER, IN1_WRITER, COMPUTE},
                    .target_nodes = all_cores_in_rect_grid,
                },
            },
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args.reserve(3);
    run_args.kernel_run_args.push_back(std::move(in0_run_args));
    run_args.kernel_run_args.push_back(std::move(in1_run_args));
    run_args.kernel_run_args.push_back(std::move(compute_run_args));
    run_args.tensor_args = {
        {IN0, in0_tensor},
        {IN1, in1_tensor},
        {OUTPUT, out_tensor},
    };
    if (bias_tensor.has_value()) {
        run_args.tensor_args.insert({BIAS, *bias_tensor});
    }

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace reuse_batched_hs_dram_sharded_optimized_helpers

ttnn::device_operation::ProgramArtifacts
MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory::create_program_artifacts(
    const ttnn::prim::MatmulParams& operation_attributes,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    const auto& input_tensors = tensor_args.input_tensors;
    const auto& optional_input_tensors = tensor_args.optional_input_tensors;
    const auto& output_tensors = tensor_return_value;

    const auto& a = input_tensors.at(0).mesh_tensor();
    const auto& b = input_tensors.at(1).mesh_tensor();
    auto bias = ttnn::as_optional_mesh_tensor(optional_input_tensors.at(0));
    const auto& output = output_tensors.at(0).mesh_tensor();
    const auto& ashape = a.padded_shape();
    const auto& bshape = b.padded_shape();
    auto in0_tile = a.tensor_spec().tile();
    auto in1_tile = b.tensor_spec().tile();
    auto in0_tile_shape = in0_tile.get_tile_shape();
    auto in1_tile_shape = in1_tile.get_tile_shape();
    auto output_tile = tt::tt_metal::Tile({in0_tile.get_tile_shape()[0], in1_tile.get_tile_shape()[1]});

    // Dataflow buffer dataformats
    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(b.dtype());
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());

    tt::DataFormat bias_data_format = tt::DataFormat::Bfp8_b;
    if (bias.has_value()) {
        const auto& c = bias.value();
        TT_FATAL(&a.device() == &c.device(), "Operands to matmul need to be on the same device!");
        bias_data_format = tt_metal::datatype_to_dataformat_converter(c.dtype());
    }

    tt::tt_metal::IDevice* device = &a.mutable_device();

    TT_FATAL(
        a.shard_spec().has_value() && output.shard_spec().has_value(), "Both input A and output must have shard specs");
    CoreRangeSet input_all_cores_storage = a.shard_spec().value().grid;
    CoreRangeSet output_all_cores_storage = output.shard_spec().value().grid;

    uint32_t in0_single_tile_size = in0_tile.get_tile_size(tt_metal::datatype_to_dataformat_converter(a.dtype()));
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(tt_metal::datatype_to_dataformat_converter(b.dtype()));

    TT_FATAL(
        a.mesh_buffer().device_local_size() % in0_single_tile_size == 0,
        "Input A buffer size ({}) must be divisible by single tile size ({})",
        a.mesh_buffer().device_local_size(),
        in0_single_tile_size);
    TT_FATAL(
        b.mesh_buffer().device_local_size() % in1_single_tile_size == 0,
        "Input B buffer size ({}) must be divisible by single tile size ({})",
        b.mesh_buffer().device_local_size(),
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

    [[maybe_unused]] auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
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
        ttnn::to_compute_hardware_config(device->arch(), compute_kernel_config),
        fp32_dest_acc_en,
        packer_l1_acc,
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
