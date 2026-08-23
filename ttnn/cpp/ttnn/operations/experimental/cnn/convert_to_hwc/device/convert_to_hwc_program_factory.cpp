// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "convert_to_hwc_program_factory.hpp"

#include "tt-metalium/tt_backend_api_types.hpp"
#include <tt-metalium/hal.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_common.hpp"

#include "gather.hpp"

#include <algorithm>

namespace ttnn::experimental::prim {

using tt::constants::TILE_HEIGHT;
using tt::constants::TILE_WIDTH;
using tt::tt_metal::BufferType;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::KernelBuildOptLevel;
using tt::tt_metal::MathFidelity;
using tt::tt_metal::Precision;
using tt::tt_metal::experimental::ComputeGen1Config;
using tt::tt_metal::experimental::DataflowBufferSpec;
using tt::tt_metal::experimental::DFBBinding;
using tt::tt_metal::experimental::DFBEndpointType;
using tt::tt_metal::experimental::DFBSpecName;
using tt::tt_metal::experimental::KernelRunArgs;
using tt::tt_metal::experimental::KernelSpec;
using tt::tt_metal::experimental::KernelSpecName;
using tt::tt_metal::experimental::ProgramRunArgs;
using tt::tt_metal::experimental::ProgramSpec;
using tt::tt_metal::experimental::TensorBinding;
using tt::tt_metal::experimental::TensorParameter;
using tt::tt_metal::experimental::TensorParamName;
using tt::tt_metal::experimental::WorkUnitSpec;

namespace {

// Per-core tiling and addressing parameters used by the writers and compute kernels
struct BlockTilingParams {
    uint32_t total_tiles_per_block;
    uint32_t tiles_per_block_writer0;
    uint32_t tiles_per_block_writer1;
    uint32_t output_addr_stride;
};

struct GroupingResult {
    std::vector<std::vector<BlockedTransferGroup>> per_core_groups;
    uint32_t num_blocks;
};

inline BlockTilingParams compute_block_tiling_params(const ConvertToHwcConfig& config, uint32_t block_size_width) {
    const uint32_t total_tiles_per_block = tt::div_up(block_size_width, TILE_HEIGHT);
    const uint32_t tiles_per_block_writer0 = tt::div_up(total_tiles_per_block, 2);
    const uint32_t tiles_per_block_writer1 = total_tiles_per_block - tiles_per_block_writer0;
    const uint32_t output_stride_sticks = TILE_WIDTH;
    // Inter-writer L1 output stride (bytes) between consecutive tiles written by a single writer.
    // If a block is only one tile tall, only one writer is active, so no stride is needed.
    const uint32_t output_addr_stride =
        (block_size_width != TILE_HEIGHT) ? output_stride_sticks * config.output_shard_width * config.element_size_bytes
                                          : 0;
    return {total_tiles_per_block, tiles_per_block_writer0, tiles_per_block_writer1, output_addr_stride};
}

// Select an appropriate block size that evenly divides gather_l1_output_shard_width
// Tries to find a block size >= 1024 that is a multiple of 32 and divides evenly
inline uint32_t select_block_size(uint32_t gather_l1_output_shard_width) {
    const uint32_t min_block_size_width = 1024;
    uint32_t block_size_width = gather_l1_output_shard_width;
    for (uint32_t candidate = min_block_size_width; candidate <= gather_l1_output_shard_width; candidate += 32) {
        if (gather_l1_output_shard_width % candidate == 0) {
            block_size_width = candidate;
            break;
        }
    }
    return block_size_width;
}

// Generate gather transfers, group them into output column blocks, and coalesce contiguous copies.
GroupingResult group_and_coalesce_transfers(
    const ConvertToHwcConfig& config,
    const std::vector<CoreCoord>& in_cores,
    uint32_t effective_hw_for_gather,
    uint32_t block_size_width) {
    // Use the actual output shard width for transfer generation (determines which output core)
    // block_size_width is only used for grouping transfers into blocks
    const auto gather_transfers = precompute_gather_transfers(
        config.batch_size,
        config.input_channels,
        effective_hw_for_gather,
        in_cores,
        config.output_cores,
        config.gather_l1_output_shard_width);

    const auto blocked_result = group_transfers_by_output_column_blocks(
        gather_transfers,
        config.batch_size,
        config.input_channels,
        effective_hw_for_gather,
        in_cores,
        config.output_cores.size(),
        /*element_size_bytes=*/config.element_size_bytes,
        /*block_size=*/block_size_width,
        /*output_shard_width=*/config.gather_l1_output_shard_width);

    auto blocked_gather_transfers = blocked_result.blocked_transfers;
    auto per_core_blocked_gather_transfers =
        split_by_destination_core(blocked_gather_transfers, config.output_cores.size());

    // Verify all cores have the same number of blocks
    // This is critical because the compute kernel expects total_num_blocks blocks from each core
    const uint32_t expected_blocks_per_core = blocked_result.num_logical_blocks;
    for (size_t core_idx = 0; core_idx < per_core_blocked_gather_transfers.size(); core_idx++) {
        uint32_t core_blocks = static_cast<uint32_t>(per_core_blocked_gather_transfers[core_idx].size());
        TT_FATAL(
            core_blocks == expected_blocks_per_core,
            "Core {} has {} blocks but expected {} blocks per core. "
            "All cores must have the same number of blocks for the compute kernel to work correctly.",
            core_idx,
            core_blocks,
            expected_blocks_per_core);
    }

    // Coalesce contiguous transfers for each core
    for (auto& core_transfers : per_core_blocked_gather_transfers) {
        core_transfers = coalesce_contiguous_transfers(core_transfers);
    }
    return {std::move(per_core_blocked_gather_transfers), blocked_result.num_logical_blocks};
}

// Serialize grouped transfers per destination core with the provided source-address mapping.
inline std::vector<std::vector<uint32_t>> serialize_transfers_per_core(
    const std::vector<std::vector<BlockedTransferGroup>>& per_core_groups,
    const std::vector<CoreCoord>& in_cores,
    const std::function<CoreCoord(const CoreCoord&)>& logical_to_addr_id) {
    std::vector<std::vector<uint32_t>> per_core_serialized;
    per_core_serialized.resize(per_core_groups.size());
    for (size_t core_idx = 0; core_idx < per_core_groups.size(); core_idx++) {
        per_core_serialized[core_idx] =
            serialize_blocked_transfer_groups(per_core_groups[core_idx], in_cores, logical_to_addr_id);
    }
    return per_core_serialized;
}

}  // namespace

// Effective HW used by gather: always the padded capacity per input core
uint32_t calculate_effective_hw_for_sharding(
    uint32_t /*hw_total*/, uint32_t /*batch_size*/, uint32_t padded_shard_width, uint32_t num_cores) {
    // Covers both even sharding (exact fit) and uneven sharding (B=1; padded)
    return num_cores * padded_shard_width;
}

ConvertToHwcConfig ConvertToHwcConfig::create_from_tensors(const Tensor& input, const Tensor& output) {
    ConvertToHwcConfig config;

    // Input tensor properties
    config.batch_size = input.logical_shape()[1];
    config.input_channels = input.logical_shape()[2];
    config.hw_total = input.logical_shape()[3];
    config.input_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    config.element_size_bytes = tt::datum_size(config.input_format);

    // DRAM/L1 configuration
    config.is_input_in_dram = input.buffer()->core_type() == tt::CoreType::DRAM;
    config.remote_buffer_type = input.buffer()->buffer_type();

    // Shard specifications
    config.output_shard_height = output.shard_spec()->shape[0];
    config.output_shard_width = output.shard_spec()->shape[1];
    config.l1_input_shard_height = config.is_input_in_dram ? input.logical_shape()[-2] : input.shard_spec()->shape[0];
    // Use input's padded sharded width (WIDTH_SHARDED) for both DRAM and L1 inputs
    config.l1_input_shard_width = input.shard_spec()->shape[1];

    // Core information
    // Kernels run on output cores; data sources are the input cores
    config.output_core_grid = output.shard_spec()->grid;
    config.output_cores = corerange_to_cores(
        config.output_core_grid,
        std::nullopt,
        output.shard_spec()->orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);

    // Always derive input core locations from the input tensor's shard grid
    config.l1_input_core_grid = input.shard_spec()->grid;
    config.l1_input_cores = corerange_to_cores(
        config.l1_input_core_grid,
        std::nullopt,
        input.shard_spec()->orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);
    config.dram_input_cores = corerange_to_cores(
        input.shard_spec()->grid,
        std::nullopt,
        input.shard_spec()->orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);

    // Gather output shard specifications (for the intermediate gather result)
    // The gather operation transforms from [B, C, HW] to [C, B, HW] layout
    // So the gather output has height=C and width=B*HW_effective/num_output_cores
    config.gather_l1_output_shard_height = config.input_channels;

    // Set per-destination-core gather width to the padded B*HW per output core (output shard height)
    config.gather_l1_output_shard_width = config.output_shard_height;

    // Alignment requirements
    config.alignment_elements = compute_alignment_requirement_in_elements(output);

    log_debug(
        tt::LogType::LogOp,
        "convert_to_hwc config: B={}, C={}, HW={}, input_in_dram={}, in_cores={}, out_cores={}, out_shard=[{}x{}], "
        "gather_width={}",
        config.batch_size,
        config.input_channels,
        config.hw_total,
        config.is_input_in_dram,
        config.l1_input_cores.size(),
        config.output_cores.size(),
        config.output_shard_height,
        config.output_shard_width,
        config.gather_l1_output_shard_width);

    return config;
}

void ConvertToHwcConfig::validate() const {
    TT_FATAL(alignment_elements != 0, "Number of alignment elements cannot be 0");
    TT_FATAL(
        output_shard_width % alignment_elements == 0,
        "Output shard width {} must be multiple of {} to satisfy alignment constraints",
        output_shard_width,
        alignment_elements);
    TT_FATAL(output_shard_height % 32 == 0, "Shard height {} must be multiple of tile width (32)", output_shard_height);
    TT_FATAL(!output_cores.empty(), "No output cores available for processing");

    // Check for uneven sharding and validate B=1 requirement
    uint32_t input_num_cores = l1_input_cores.size();
    // Uneven sharding occurs when the last core has fewer logical elements than the shard width
    uint32_t total_padded_elements = input_num_cores * l1_input_shard_width;
    bool is_uneven_sharding = hw_total < total_padded_elements;
    if (is_uneven_sharding) {
        TT_FATAL(
            batch_size == 1,
            "Uneven sharding (HW={} < total_padded_capacity={}) is only supported for batch_size=1, got batch_size={}",
            hw_total,
            total_padded_elements,
            batch_size);
    }
}

uint32_t compute_alignment_requirement_in_elements(const Tensor& input_tensor) {
    const uint32_t element_size_bytes = input_tensor.element_size();
    const uint32_t l1_alignment_bytes = tt::tt_metal::hal::get_l1_alignment();
    return l1_alignment_bytes / element_size_bytes;
}

}  // namespace ttnn::experimental::prim

namespace ttnn::experimental::prim {

ttnn::device_operation::ProgramArtifacts ConvertToHWCProgramFactory::create_program_artifacts(
    const ConvertToHwcParams& /*operation_attributes*/,
    const ConvertToHwcInputs& tensor_args,
    Tensor& tensor_return_value) {
    const auto& a = tensor_args.input;
    auto& output = tensor_return_value;

    // Create configuration from input tensors
    auto config = ConvertToHwcConfig::create_from_tensors(a, output);
    config.validate();

    // Select input cores based on source memory (DRAM vs L1)
    const auto& in_cores = config.is_input_in_dram ? config.dram_input_cores : config.l1_input_cores;

    // Effective HW for gather transfers (padded capacity per input core)
    uint32_t effective_hw_for_gather = calculate_effective_hw_for_sharding(
        config.hw_total, config.batch_size, config.l1_input_shard_width, static_cast<uint32_t>(in_cores.size()));

    // Use smaller block size to reduce L1 consumption
    // Find a block size that evenly divides gather_l1_output_shard_width
    // This reduces the CB_IN_BATCH buffer size significantly
    const auto block_width = select_block_size(config.gather_l1_output_shard_width);

    auto grouping = group_and_coalesce_transfers(config, in_cores, effective_hw_for_gather, block_width);
    const uint32_t num_blocks = grouping.num_blocks;

    // Source-address mapping for serialization:
    // - L1 input: logical core -> worker core (x,y)
    // - DRAM input: x := bank_id, y := 0
    std::function<CoreCoord(const CoreCoord&)> logical_to_addr_id;
    if (config.is_input_in_dram) {
        std::map<std::pair<int, int>, uint32_t> bank_id_by_core;
        for (const auto& c : config.dram_input_cores) {
            auto bank_ids = a.device()->allocator()->get_bank_ids_from_logical_core(config.remote_buffer_type, c);
            uint32_t bank_id = bank_ids.empty() ? 0 : bank_ids[0];
            bank_id_by_core[{c.x, c.y}] = bank_id;
        }
        logical_to_addr_id = [bank_id_by_core = std::move(bank_id_by_core)](const CoreCoord& logical_core) {
            auto it = bank_id_by_core.find({logical_core.x, logical_core.y});
            uint32_t bank_id = (it == bank_id_by_core.end()) ? 0 : it->second;
            return CoreCoord(bank_id, 0);
        };
    } else {
        logical_to_addr_id = [&a](const CoreCoord& logical_core) {
            return a.device()->worker_core_from_logical_core(logical_core);
        };
    }

    // Serialize blocked transfer groups for each core
    auto per_core_serialized_transfers =
        serialize_transfers_per_core(grouping.per_core_groups, in_cores, logical_to_addr_id);

    // Compute per-core tiling/state based on the chosen block width
    const BlockTilingParams tiling = compute_block_tiling_params(config, block_width);

    const tt::DataFormat intermediary_format = tt::DataFormat::Float16_b;
    const uint32_t intermediary_tile_size = tt::tile_size(intermediary_format);

    const TensorParamName INPUT{"input"};
    const TensorParamName DISCARDED_INPUT_TOKEN{"discarded_input_token"};
    const TensorParamName OUTPUT{"output"};
    const DFBSpecName INPUT_DFB{"input"};
    const DFBSpecName BATCH_DFB{"batch"};
    const DFBSpecName TILED_DFB{"tiled"};
    const DFBSpecName TRANSPOSE_0_DFB{"transpose0"};
    const DFBSpecName TRANSPOSE_1_DFB{"transpose1"};
    const DFBSpecName OUTPUT_DFB{"output"};
    const KernelSpecName READER_WRITER{"reader_writer"};
    const KernelSpecName SECONDARY_WRITER{"secondary_writer"};
    const KernelSpecName COMPUTE{"compute"};

    const TensorParameter input_param{.unique_id = INPUT, .spec = a.tensor_spec()};
    const TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    // CB_IN: raw-address view of the input shard. The L1 path borrows tensor storage; the
    // DRAM path keeps only a minimal slot-compatible allocation because its constexpr-selected
    // reader uses TensorAccessor instead.
    const uint32_t input_page_bytes = config.l1_input_shard_width * config.element_size_bytes;
    const uint32_t input_dfb_bytes = config.l1_input_shard_height * input_page_bytes;
    const uint32_t input_packed_bytes = static_cast<uint32_t>(a.tensor_spec().compute_packed_buffer_size_bytes());
    // The kernels use the borrowed input DFB only as a raw L1 base-address source. For the
    // single-core uneven case, the physical shard allocation includes row padding that is not
    // represented by TensorSpec's packed logical size. Describe only the logical bytes in that
    // case so Metal 2.0 can validate the borrow; raw offsets still address the physical padding.
    const bool input_dfb_fits_packed_tensor = input_dfb_bytes <= input_packed_bytes;
    const uint32_t input_raw_view_bytes = input_dfb_fits_packed_tensor ? input_dfb_bytes : input_packed_bytes;
    const DataflowBufferSpec input_dfb{
        .unique_id = INPUT_DFB,
        .entry_size = input_raw_view_bytes,
        .num_entries = 1,
        .data_format_metadata = config.input_format,
        .borrowed_from = INPUT,
    };
    // CB_IN_BATCH: [C x block_size_width] staging for gathered sticks
    const DataflowBufferSpec batch_dfb{
        .unique_id = BATCH_DFB,
        .entry_size = block_width * config.element_size_bytes,
        .num_entries = config.gather_l1_output_shard_height,
        .data_format_metadata = config.input_format,
    };
    // CB_IN_TILED: intermediate tiles
    const DataflowBufferSpec tiled_dfb{
        .unique_id = TILED_DFB,
        .entry_size = intermediary_tile_size,
        .num_entries = tt::div_up(block_width, TILE_WIDTH),
        .data_format_metadata = intermediary_format,
    };
    const DataflowBufferSpec transpose_0_dfb{
        .unique_id = TRANSPOSE_0_DFB,
        .entry_size = intermediary_tile_size,
        .num_entries = tt::div_up(block_width, TILE_WIDTH),
        .data_format_metadata = intermediary_format,
    };
    const DataflowBufferSpec transpose_1_dfb{
        .unique_id = TRANSPOSE_1_DFB,
        .entry_size = intermediary_tile_size,
        .num_entries = tt::div_up(block_width, TILE_WIDTH),
        .data_format_metadata = intermediary_format,
    };
    // CB_OUT: output shard per core
    const uint32_t output_page_bytes = config.output_shard_width * config.element_size_bytes;
    const uint32_t output_packed_bytes = static_cast<uint32_t>(output.tensor_spec().compute_packed_buffer_size_bytes());
    // Likewise, output is raw-addressed and never uses FIFO accounting. Clamp the advertised
    // entry count to the TensorSpec's packed size for a single-core uneven shard; the backing L1
    // allocation still has the padded shard capacity written by the two DM kernels.
    const uint32_t output_raw_view_bytes =
        std::min(config.output_shard_height * output_page_bytes, output_packed_bytes);
    const DataflowBufferSpec output_dfb{
        .unique_id = OUTPUT_DFB,
        .entry_size = output_raw_view_bytes,
        .num_entries = 1,
        .data_format_metadata = config.input_format,
        .borrowed_from = OUTPUT,
    };

    KernelSpec reader_writer{
        .unique_id = READER_WRITER,
        .source = "ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_hwc/device/kernels/writer_convert_to_hwc.cpp",
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = INPUT_DFB,
                 .accessor_name = "input",
                 .endpoint_type = DFBEndpointType::PRODUCER,
                 .allow_unbound_for_constexpr_discard = config.is_input_in_dram},
             DFBBinding{
                 .dfb_spec_name = INPUT_DFB,
                 .accessor_name = "input",
                 .endpoint_type = DFBEndpointType::CONSUMER,
                 .allow_unbound_for_constexpr_discard = config.is_input_in_dram},
             DFBBinding{
                 .dfb_spec_name = BATCH_DFB, .accessor_name = "batch", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = TRANSPOSE_0_DFB,
                 .accessor_name = "transpose",
                 .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = OUTPUT_DFB, .accessor_name = "output", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{
            .tensor_parameter_name = config.is_input_in_dram ? INPUT : DISCARDED_INPUT_TOKEN,
            .accessor_name = "input",
            .allow_unbound_for_constexpr_discard = !config.is_input_in_dram}},
        .compile_time_args =
            {{"num_output_channels_padded", config.output_shard_width},
             {"num_full_tiles", tiling.tiles_per_block_writer0},
             {"total_tiles_per_block", tiling.total_tiles_per_block},
             {"initial_write_stick_offset", 0},
             {"element_size_bytes", config.element_size_bytes},
             {"is_input_in_dram", static_cast<uint32_t>(config.is_input_in_dram)},
             {"input_block_size_sticks_per_core", config.gather_l1_output_shard_height},
             {"l1_write_output_addr_stride", tiling.output_addr_stride}},
        .hw_config = ttnn::create_reader_datamovement_config(a.device()->arch()),
    };
    const KernelSpec secondary_writer{
        .unique_id = SECONDARY_WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_hwc/device/kernels/"
            "writer_convert_to_hwc_secondary.cpp",
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = TRANSPOSE_1_DFB,
                 .accessor_name = "transpose",
                 .endpoint_type = DFBEndpointType::CONSUMER},
             // Gen1 compatibility shim: this second DM writes disjoint addresses in the borrowed
             // output shard, but the primary writer owns the single producer endpoint. Binding the
             // secondary as CONSUMER preserves the legacy per-RISC plain-CB view without another
             // buffer, copy, or FIFO operation.
             DFBBinding{
                 .dfb_spec_name = OUTPUT_DFB, .accessor_name = "output", .endpoint_type = DFBEndpointType::CONSUMER}},
        .compile_time_args =
            {{"num_output_channels_padded", config.output_shard_width},
             {"num_full_tiles", tiling.tiles_per_block_writer1},
             {"total_tiles_per_block", tiling.total_tiles_per_block},
             {"initial_write_stick_offset", TILE_WIDTH},
             {"element_size_bytes", config.element_size_bytes},
             {"input_num_blocks", num_blocks},
             {"l1_write_output_addr_stride", tiling.output_addr_stride}},
        .hw_config = ttnn::create_writer_datamovement_config(a.device()->arch()),
    };
    const KernelSpec compute{
        .unique_id = COMPUTE,
        .source = "ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_hwc/device/kernels/convert_to_hwc.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = BATCH_DFB, .accessor_name = "batch", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = TILED_DFB, .accessor_name = "tiled", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = TILED_DFB, .accessor_name = "tiled", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = TRANSPOSE_0_DFB,
                 .accessor_name = "transpose0",
                 .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = TRANSPOSE_1_DFB,
                 .accessor_name = "transpose1",
                 .endpoint_type = DFBEndpointType::PRODUCER}},
        .compile_time_args =
            {{"total_tiles_per_block", tiling.total_tiles_per_block},
             {"total_sticks_per_block", config.gather_l1_output_shard_height},
             {"total_num_blocks", num_blocks}},
        .hw_config =
            ComputeGen1Config{
                .fpu_math_fidelity = MathFidelity::HiFi4,
                .sfpu_precision_mode = Precision::Precise,
            },
    };

    KernelRunArgs reader_writer_run{.kernel = READER_WRITER};
    uint32_t max_runtime_varargs = 0;
    for (const auto& transfer_args : per_core_serialized_transfers) {
        max_runtime_varargs = std::max(max_runtime_varargs, static_cast<uint32_t>(transfer_args.size()));
    }
    reader_writer.advanced_options.num_runtime_varargs = max_runtime_varargs;
    for (uint32_t core_idx = 0; core_idx < config.output_cores.size(); ++core_idx) {
        const auto core = config.output_cores[core_idx];
        auto& transfer_args = per_core_serialized_transfers.at(core_idx);
        transfer_args.resize(max_runtime_varargs, 0);
        reader_writer_run.advanced_options.runtime_varargs.emplace(core, std::move(transfer_args));
    }

    ProgramSpec spec{
        .name = "convert_to_hwc",
        .kernels = {reader_writer, secondary_writer, compute},
        // The L1 path preserves the legacy physical CB-slot order: input=c0, batch=c1,
        // tiled=c2, transpose0=c3, transpose1=c4, output=c5. The DRAM binary constexpr-discards
        // input and omits that unused allocation; named bindings keep the remaining slots coherent.
        .dataflow_buffers =
            [&] {
                std::vector<DataflowBufferSpec> buffers;
                buffers.reserve(config.is_input_in_dram ? 5 : 6);
                if (!config.is_input_in_dram) {
                    buffers.push_back(input_dfb);
                }
                buffers.insert(buffers.end(), {batch_dfb, tiled_dfb, transpose_0_dfb, transpose_1_dfb, output_dfb});
                return buffers;
            }(),
        .tensor_parameters = {input_param, output_param},
        .work_units = {WorkUnitSpec{
            .name = "convert_to_hwc",
            .kernels = {READER_WRITER, SECONDARY_WRITER, COMPUTE},
            .target_nodes = config.output_core_grid,
        }},
    };
    ProgramRunArgs run_args{
        .kernel_run_args =
            {std::move(reader_writer_run), KernelRunArgs{.kernel = SECONDARY_WRITER}, KernelRunArgs{.kernel = COMPUTE}},
        .tensor_args = {{INPUT, a.mesh_tensor()}, {OUTPUT, output.mesh_tensor()}},
    };
    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
