// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_expert_rows.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "moe_compute_device_operation.hpp"
#include "moe_expert_rows_plan.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

namespace ttnn::experimental::prim {

using namespace expert_rows_host;

namespace {

using ttnn::experimental::ccl::add_semaphore_descriptor;

// The reader adds the deal times of a group's jobs (at most its ctl entries) in u32: one deal unit is the smallest
// power of two of cycles that keeps the sum below this.
constexpr double kDealLoadLimit = 4.0e9;
// Blackhole and Wormhole run the same kernels: their plans differ only in numbers (readers per bank, packets per read
// block) that plan_expert_rows derives from the device.
constexpr const char* kReaderKernel = "rows_reader.cpp";
constexpr const char* kWriterKernel = "rows_writer.cpp";
constexpr const char* kComputeKernel = "rows_compute.cpp";
// The expert rows program's kernels, in order
constexpr uint32_t kRowsReaderKernelIdx = 0;
constexpr uint32_t kRowsWriterKernelIdx = 1;
constexpr uint32_t kRowsComputeKernelIdx = 2;

// mapping rows of the devices whose tokens this device receives (moe_compute tilize_reader: the devices of the
// dispatch axis through this device, in axis order)
std::vector<uint32_t> source_rows(
    const ttnn::MeshDevice& mesh_device, const ttnn::MeshCoordinate& coord, uint32_t axis) {
    const auto& view = mesh_device.get_view();
    const uint32_t rows = view.num_rows(), cols = view.num_cols();
    std::vector<uint32_t> out;
    if (axis == 1) {
        for (uint32_t j = 0; j < cols; ++j) {
            out.push_back(coord[0] * cols + j);
        }
    } else {
        for (uint32_t i = 0; i < rows; ++i) {
            out.push_back(i * cols + coord[1]);
        }
    }
    return out;
}

// Reader runtime args 5..6, which override_runtime_arguments patches: the layer's W0/W1 and W2 (the bank address of
// the layer's first expert), offsets into the weights that move with layer_id.
constexpr uint32_t kReaderLayerWeightArgs = 5;
// The bank address of `layer` of prepared weights whose (layer, expert) holds tiles_per_expert tiles per bank.
uint32_t layer_address(const ttnn::Tensor& weights, uint32_t layer, uint32_t local_experts, uint32_t tiles_per_expert) {
    const uint64_t address = uint64_t{weights.buffer()->address()} +
                             uint64_t{layer} * local_experts * tiles_per_expert * tt::tile_size(tt::DataFormat::Bfp4_b);
    TT_FATAL(
        address <= std::numeric_limits<uint32_t>::max(),
        "moe_compute expert rows: layer {} of the weights at {} B in a bank is past 32 bits",
        layer,
        address);
    return static_cast<uint32_t>(address);
}

std::array<uint32_t, 2> layer_weight_args(
    const MoEExpertRowsParams& args, const MoEExpertRowsInputs& tensor_args, const PreparedLayout& layout) {
    const uint32_t local_experts = args.shape.local_experts;
    return {
        layer_address(tensor_args.w0_w1_tensor, args.layer_id, local_experts, layout.w0_w1_piece_tiles),
        layer_address(tensor_args.w2_tensor, args.layer_id, local_experts, layout.w2_slice_tiles)};
}

// The tiles of one shard per (layer, expert) of prepared weights: their last three dims.
uint32_t prepared_tiles_per_expert(const ttnn::Shape& shape) {
    return shape[-3] * (shape[-2] / tt::constants::TILE_HEIGHT) * (shape[-1] / tt::constants::TILE_WIDTH);
}

// `count` consecutive semaphores on `cores`, all 0; returns the first id.
uint32_t create_semaphores(tt::tt_metal::ProgramDescriptor& desc, const CoreRangeSet& cores, uint32_t count) {
    const uint32_t first = add_semaphore_descriptor(desc, cores, 0);
    for (uint32_t i = 1; i < count; ++i) {
        add_semaphore_descriptor(desc, cores, 0);
    }
    return first;
}

}  // namespace

//-----------------------------------------------------------------------------------------------------------------
// The expert rows program
//-----------------------------------------------------------------------------------------------------------------
MoEExpertRowsDeviceOperation::program_factory_t MoEExpertRowsDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return MoEExpertRowsFactory{};
}

ttsl::hash::hash_t MoEExpertRowsDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // the layer only offsets the weight addresses (runtime arguments): one program serves every layer
    MoEExpertRowsParams keyed = args;
    keyed.layer_id = 0;
    return ttsl::hash::hash_objects_with_default_seed(keyed, tensor_args);
}

void MoEExpertRowsDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void MoEExpertRowsDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_activation_limit(args.activation_type, args.activation_limit);
    const auto& shape = args.shape;
    const auto refusal = shape_refusal(shape);
    TT_FATAL(!refusal.has_value(), "moe_compute expert rows: {}", refusal.value_or(""));
    const auto& w0_w1 = tensor_args.w0_w1_tensor;
    const auto& w2 = tensor_args.w2_tensor;
    TT_FATAL(
        w0_w1.dtype() == DataType::BFLOAT4_B && w2.dtype() == DataType::BFLOAT4_B,
        "moe_compute expert rows: weights must be bfloat4_b, got W0/W1 {} and W2 {}",
        w0_w1.dtype(),
        w2.dtype());
    const uint32_t indices_page = aligned_page(tensor_args.expert_indices_tensor);
    const uint32_t scores_page = aligned_page(tensor_args.expert_scores_tensor);
    TT_FATAL(
        indices_page == shape.index_page_bytes && scores_page == shape.index_page_bytes,
        "moe_compute expert rows: indices and scores must share one page size of {} B, got {} and {} B",
        shape.index_page_bytes,
        indices_page,
        scores_page);
    const uint32_t mapping_page = aligned_page(tensor_args.expert_mapping_tensor);
    TT_FATAL(
        mapping_page == shape.mapping_page_bytes,
        "moe_compute expert rows: the expert mapping's pages are {} B, expected {}",
        mapping_page,
        shape.mapping_page_bytes);
    const auto w0_w1_banks = shard_banks_of(w0_w1);
    const auto w2_banks = shard_banks_of(w2);
    TT_FATAL(
        w0_w1_banks == w2_banks,
        "moe_compute expert rows: W0/W1 and W2 shards must sit on the same banks, got {} and {}",
        w0_w1_banks,
        w2_banks);
    TT_FATAL(
        args.layer_id < w0_w1.logical_shape()[1],
        "moe_compute expert rows: layer_id {} is out of range for {} layers of weights",
        args.layer_id,
        w0_w1.logical_shape()[1]);
}

MoEExpertRowsDeviceOperation::spec_return_value_t MoEExpertRowsDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& /*tensor_args*/) {
    const auto& shape = args.shape;
    const uint32_t rows = std::min(shape.tokens * shape.top_k, shape.local_experts * shape.tokens);
    const auto rows_spec = tt::tt_metal::TensorSpec(
        ttnn::Shape({rows, shape.hidden_size}),
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), ttnn::DRAM_MEMORY_CONFIG));
    const auto table_spec = tt::tt_metal::TensorSpec(
        ttnn::Shape({1, expert_rows::table_layout(shape.tokens, shape.top_k, shape.local_experts).bytes / kBf16Bytes}),
        tt::tt_metal::TensorLayout(
            DataType::UINT16, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), ttnn::DRAM_MEMORY_CONFIG));
    return {rows_spec, table_spec};
}

MoEExpertRowsDeviceOperation::tensor_return_value_t MoEExpertRowsDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto specs = compute_output_specs(args, tensor_args);
    auto* device = tensor_args.input_tensor.device();
    return {create_device_tensor(specs[0], device), create_device_tensor(specs[1], device)};
}

namespace {

// The expert rows program on the device at mesh_coordinate.
tt::tt_metal::ProgramDescriptor build_expert_rows_program_descriptor(
    const MoEExpertRowsParams& args,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const MoEExpertRowsInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    tt::tt_metal::ProgramDescriptor desc;
    auto* mesh_device = tensor_args.input_tensor.device();
    const auto& shape = args.shape;
    const uint32_t hidden_tiles = shape.hidden_tiles();
    const uint32_t intermediate_tiles = shape.intermediate_tiles();
    const auto shard_banks = shard_banks_of(tensor_args.w0_w1_tensor);
    const uint32_t num_shards = shard_banks.size();
    const auto result = plan_expert_rows(
        mesh_device, shape, shard_banks, args.cb_budget, args.row_tiles, partials_format(args.fp32_dest_acc_en));
    TT_FATAL(
        std::holds_alternative<ExpertRowsPlan>(result), "moe_compute expert rows: {}", std::get<std::string>(result));
    TT_FATAL(
        expert_rows_arch(mesh_device->arch()),
        "moe_compute expert rows: the expert rows program runs on Blackhole and Wormhole, not {}",
        mesh_device->arch());
    const ExpertRowsPlan& plan = std::get<ExpertRowsPlan>(result);
    const auto& routing = plan.routing;
    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    const GridCoords virtual_coords = grid_coords(mesh_device, grid);
    const auto& view = mesh_device->get_view();
    const uint32_t device_id = mesh_coordinate[0] * view.num_cols() + mesh_coordinate[1];
    const auto sources = source_rows(*mesh_device, mesh_coordinate, args.cluster_axis);
    TT_FATAL(
        sources.size() == shape.num_sources,
        "moe_compute expert rows: {} sources, expected {}",
        sources.size(),
        shape.num_sources);

    const ttnn::Tensor& rows_tensor = tensor_return_value[0];
    const ttnn::Tensor& table_tensor = tensor_return_value[1];

    const uint32_t weight_tile_bytes = tt::tile_size(tt::DataFormat::Bfp4_b);
    const PreparedLayout layout = prepared_layout(hidden_tiles, intermediate_tiles, shape.has_bias, num_shards);
    const auto shards = shard_layouts(hidden_tiles, intermediate_tiles, shape.has_bias, num_shards);
    const uint32_t w0_w1_tiles = prepared_tiles_per_expert(tensor_args.w0_w1_tensor.logical_shape());
    const uint32_t w2_tiles = prepared_tiles_per_expert(tensor_args.w2_tensor.logical_shape());
    TT_FATAL(
        w0_w1_tiles == layout.w0_w1_piece_tiles && w2_tiles == layout.w2_slice_tiles,
        "moe_compute expert rows: weights do not have the prepared layout of hidden size {} / intermediate size {} on "
        "{} banks: W0/W1 {} / W2 {} tiles per (layer, expert) in a bank, expected {} / {}",
        shape.hidden_size,
        shape.intermediate_size,
        num_shards,
        w0_w1_tiles,
        w2_tiles,
        layout.w0_w1_piece_tiles,
        layout.w2_slice_tiles);

    std::vector<CoreCoord> all_cores;
    all_cores.reserve(plan.cores.size());
    for (const auto& member : plan.cores) {
        all_cores.push_back(member.core);
    }
    const CoreRangeSet all_set = CoreRangeSet(ttsl::Span<const CoreCoord>(all_cores));
    const std::vector<uint32_t> all_mask = core_mask(all_cores, grid);

    // circular buffers: x chunk slots, row tiles of activated and intermediate tiles per job; with several chunks per
    // job FP32 (or 16-bit DEST) partials
    enum ExpertRowsCb : uint32_t {
        cb_x,
        cb_w,
        cb_activated,
        cb_intermediate,
        cb_rows,
        cb_ctl,
        cb_routing,
        cb_scratch,
        cb_ones,
        cb_partials,
    };
    const CbSizes& cbs = plan.cbs;
    create_cb(desc, all_set, cb_x, cbs.x_chunks, tt::DataFormat::Float16_b, kTileBytesBf16);
    create_cb(desc, all_set, cb_w, cbs.weights, tt::DataFormat::Bfp4_b, weight_tile_bytes);
    create_cb(desc, all_set, cb_activated, cbs.activated, tt::DataFormat::Float16_b, kTileBytesBf16);
    create_cb(desc, all_set, cb_intermediate, cbs.intermediate, tt::DataFormat::Float16_b, kTileBytesBf16);
    create_cb(desc, all_set, cb_rows, cbs.rows, tt::DataFormat::Float16_b, kTileBytesBf16);
    create_cb(desc, all_set, cb_ctl, cbs.ctl, tt::DataFormat::UInt32, plan.ctl_bytes);
    create_cb(desc, all_set, cb_routing, cbs.routing, tt::DataFormat::UInt32, routing.size);
    create_cb(desc, all_set, cb_scratch, cbs.scratch, tt::DataFormat::UInt32, cbs.scratch);
    if (cbs.ones > 0) {
        create_cb(desc, all_set, cb_ones, cbs.ones, tt::DataFormat::Float16_b, kTileBytesBf16);
    }
    tt::tt_metal::ComputeConfigDescriptor::UnpackToDestModes unpack_to_dest(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    if (cbs.partials > 0) {
        // partials go back into DEST exactly: FP32 tiles unpacked straight to DEST, 16-bit tiles through SrcA
        const tt::DataFormat format = partials_format(args.fp32_dest_acc_en);
        create_cb(desc, all_set, cb_partials, cbs.partials, format, tt::tile_size(format));
        if (args.fp32_dest_acc_en) {
            unpack_to_dest[cb_partials] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
        }
    }

    // semaphores: routing (entries at the root, table ready), intermediate slots, x slots; with credits also those that
    // hand a slot back (every group core freed it)
    const uint32_t sem_entries = add_semaphore_descriptor(desc, all_set, 0);
    const uint32_t sem_table = add_semaphore_descriptor(desc, all_set, 0);
    const uint32_t sem_intermediate = create_semaphores(desc, all_set, plan.intermediate_slots);
    const uint32_t sem_x = create_semaphores(desc, all_set, plan.x_slots);
    const uint32_t sem_intermediate_free =
        plan.row_tiles > 1 ? create_semaphores(desc, all_set, plan.intermediate_slots) : 0;
    const uint32_t sem_x_free = plan.row_tiles > 1 ? create_semaphores(desc, all_set, plan.x_slots) : 0;

    // several row tiles per job: the root deals the jobs to the groups by the cost model's predicted job time
    // (rows_reader.cpp), in cycles scaled down by a power of two while a group's most jobs (its ctl list) could
    // overflow its 32-bit load
    const JobCycles cycles = job_cycles(shape, plan, num_shards, mesh_device->arch(), args.math_fidelity);
    const double job_most = std::max({cycles.stream, plan.row_tiles * cycles.row_tile, cycles.single});
    const uint32_t ctl_words = plan.ctl_bytes / kWordBytes;
    double deal_unit = 1.0;
    while (job_most * ctl_words / deal_unit >= kDealLoadLimit) {
        deal_unit *= 2.0;
    }
    auto deal_units = [deal_unit](double cycles) { return static_cast<uint32_t>(std::ceil(cycles / deal_unit)); };
    tt::tt_metal::KernelDescriptor::NamedCompileTimeArgs reader_ct = {
        {"cb_w", cb_w},
        {"cb_routing", cb_routing},
        {"cb_ctl", cb_ctl},
        {"block_tiles", plan.block_tiles},
        {"weight_tile_bytes", weight_tile_bytes},
        {"block_slots", plan.block_slots},
        {"block_packets", plan.block_packets},
        {"blocks_in_flight", plan.blocks_in_flight},
        {"tokens", shape.tokens},
        {"local_experts", shape.local_experts},
        {"top_k", shape.top_k},
        {"global_experts", shape.global_experts},
        {"index_page_bytes", shape.index_page_bytes},
        {"mapping_page_bytes", shape.mapping_page_bytes},
        {"tokens_per_source", shape.tokens / shape.num_sources},
        {"expert_groups", plan.expert_groups},
        {"job_rows", expert_rows::tile_rows * plan.row_tiles},
        {"row_tiles", plan.row_tiles},
        {"chunk_tiles", plan.chunk_tiles},
        {"chunk_blocks", plan.chunk_blocks},
        {"w2_chunk_rows", plan.w2_chunk_rows},
        {"hidden_tiles", hidden_tiles},
        {"intermediate_tiles", intermediate_tiles},
        {"has_bias", shape.has_bias ? 1u : 0u},
        {"ctl_words", ctl_words},
        {"deal_stream", deal_units(cycles.stream)},
        {"deal_row_tile", deal_units(cycles.row_tile)},
        {"deal_single", deal_units(cycles.single)},
        {"grid_w", grid.x},
        {"grid_h", grid.y},
        {"mask_words", static_cast<uint32_t>(all_mask.size())},
        {"sem_entries", sem_entries},
        {"sem_table", sem_table},
        {"routing_own_mapping", routing.own_mapping},
        {"routing_source_mappings", routing.source_mappings},
        {"routing_token_ids", routing.token_ids},
        {"routing_token_scores", routing.token_scores},
        {"routing_ctl", routing.ctl},
        {"routing_active_experts", routing.active_experts},
        {"routing_expert_rows", routing.expert_rows},
        {"routing_expert_first_rows", routing.expert_first_rows},
        {"routing_table", routing.table},
        {"routing_jobs", routing.jobs},
        {"routing_entry_areas", routing.entry_areas},
        {"w0_w1_group_tiles", layout.w0_w1_group_tiles},
        {"w0_w1_piece_tiles", layout.w0_w1_piece_tiles},
        {"w2_group_tiles", layout.w2_group_tiles},
        {"w2_slice_tiles", layout.w2_slice_tiles},
    };
    std::vector<uint32_t> reader_ct_pos;
    tt::tt_metal::TensorAccessorArgs(*tensor_args.expert_indices_tensor.buffer()).append_to(reader_ct_pos);
    tt::tt_metal::TensorAccessorArgs(*tensor_args.expert_scores_tensor.buffer()).append_to(reader_ct_pos);
    tt::tt_metal::TensorAccessorArgs(*tensor_args.expert_mapping_tensor.buffer()).append_to(reader_ct_pos);
    tt::tt_metal::TensorAccessorArgs(*table_tensor.buffer()).append_to(reader_ct_pos);

    tt::tt_metal::KernelDescriptor::NamedCompileTimeArgs writer_ct = {
        {"cb_activated", cb_activated},
        {"cb_intermediate", cb_intermediate},
        {"cb_routing", cb_routing},
        {"cb_x", cb_x},
        {"cb_scratch", cb_scratch},
        {"cb_rows", cb_rows},
        {"group_cores", plan.group_cores},
        {"intermediate_tiles", intermediate_tiles},
        {"hidden_tiles", hidden_tiles},
        {"intermediate_slots", plan.intermediate_slots},
        {"x_slots", plan.x_slots},
        {"activated_tiles", plan.activated_tiles},
        {"routing_ctl", routing.ctl},
        {"routing_row_tokens", routing.row_tokens},
        {"routing_jobs", routing.jobs},
        {"grid_w", grid.x},
        {"grid_h", grid.y},
        {"mask_words", static_cast<uint32_t>(all_mask.size())},
        {"row_bytes", shape.hidden_size * kBf16Bytes},
        {"expert_groups", plan.expert_groups},
        {"sem_intermediate", sem_intermediate},
        {"sem_x", sem_x},
        {"sem_intermediate_free", sem_intermediate_free},
        {"sem_x_free", sem_x_free},
        {"row_tiles", plan.row_tiles},
        {"chunk_tiles", plan.chunk_tiles},
        {"chunk_piece_max", plan.chunk_piece_max},
    };
    std::vector<uint32_t> writer_ct_pos;
    tt::tt_metal::TensorAccessorArgs(*tensor_args.input_tensor.buffer()).append_to(writer_ct_pos);
    tt::tt_metal::TensorAccessorArgs(*rows_tensor.buffer()).append_to(writer_ct_pos);

    tt::tt_metal::KernelDescriptor::NamedCompileTimeArgs compute_ct = {
        {"block_tiles", plan.block_tiles},
        {"cb_w", cb_w},
        {"hidden_tiles", hidden_tiles},
        {"intermediate_tiles", intermediate_tiles},
        {"ring_cores", num_shards},
        {"has_bias", shape.has_bias ? 1u : 0u},
        {"cb_x", cb_x},
        {"cb_activated", cb_activated},
        {"cb_intermediate", cb_intermediate},
        {"cb_rows", cb_rows},
        {"cb_ctl", cb_ctl},
        {"cb_ones", cb_ones},
        {"activated_tiles", plan.activated_tiles},
        {"activation_function", static_cast<uint32_t>(args.activation_type)},
        {"activation_limit_bits", std::bit_cast<uint32_t>(args.activation_limit.value_or(0.0f))},
        {"row_tiles", plan.row_tiles},
        {"chunk_tiles", plan.chunk_tiles},
        {"chunk_blocks", plan.chunk_blocks},
        {"w2_chunk_rows", plan.w2_chunk_rows},
        {"cb_partials", cb_partials},
    };

    // readers on NOC 0, writers on NOC 1: one kernel each over the whole core set, one binary per program written to
    // the cores' rectangles
    desc.kernels.push_back(tt::tt_metal::KernelDescriptor{
        .kernel_source = std::string(kKernelDir) + kReaderKernel,
        .core_ranges = all_set,
        .compile_time_args = std::move(reader_ct_pos),
        .named_compile_time_args = std::move(reader_ct),
        .config =
            tt::tt_metal::DataMovementConfigDescriptor{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::NOC_0,
            },
    });
    desc.kernels.push_back(tt::tt_metal::KernelDescriptor{
        .kernel_source = std::string(kKernelDir) + kWriterKernel,
        .core_ranges = all_set,
        .compile_time_args = std::move(writer_ct_pos),
        .named_compile_time_args = std::move(writer_ct),
        .config =
            tt::tt_metal::DataMovementConfigDescriptor{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt::tt_metal::NOC::NOC_1,
            },
    });
    desc.kernels.push_back(tt::tt_metal::KernelDescriptor{
        .kernel_source = std::string(kKernelDir) + kComputeKernel,
        .core_ranges = all_set,
        .named_compile_time_args = std::move(compute_ct),
        .config =
            tt::tt_metal::ComputeConfigDescriptor{
                .math_fidelity = args.math_fidelity,
                .fp32_dest_acc_en = args.fp32_dest_acc_en,
                .dst_full_sync_en = false,
                .unpack_to_dest_mode = std::move(unpack_to_dest),
                .math_approx_mode = true,
            },
    });

    // runtime args: per core, plus common args shared by every core of a kernel
    std::vector<uint32_t> reader_common = {static_cast<uint32_t>(sources.size())};
    reader_common.insert(reader_common.end(), sources.begin(), sources.end());
    reader_common.insert(reader_common.end(), virtual_coords.xs.begin(), virtual_coords.xs.end());
    reader_common.insert(reader_common.end(), virtual_coords.ys.begin(), virtual_coords.ys.end());
    reader_common.insert(reader_common.end(), all_mask.begin(), all_mask.end());
    reader_common.insert(reader_common.end(), shard_banks.begin(), shard_banks.end());
    std::vector<uint32_t> writer_common = virtual_coords.xs;
    writer_common.insert(writer_common.end(), virtual_coords.ys.begin(), virtual_coords.ys.end());
    desc.kernels[kRowsReaderKernelIdx].common_runtime_args = std::move(reader_common);
    desc.kernels[kRowsWriterKernelIdx].common_runtime_args = std::move(writer_common);
    const CoreCoord root = mesh_device->worker_core_from_logical_core(plan.cores.front().core);
    const uint32_t program_cores = plan.cores.size();
    const auto layer_weights = layer_weight_args(args, tensor_args, layout);
    for (uint32_t core_index = 0; core_index < program_cores; ++core_index) {
        const ExpertRowsCore& member = plan.cores[core_index];
        std::vector<CoreCoord> group_cores;
        for (const auto& other : plan.cores) {
            if (other.expert_group == member.expert_group) {
                group_cores.push_back(other.core);
            }
        }
        tt::tt_metal::KernelDescriptor::RTArgList reader_args;
        reader_args.push_back(member.bank);
        reader_args.push_back(tensor_args.expert_indices_tensor.buffer());
        reader_args.push_back(tensor_args.expert_scores_tensor.buffer());
        reader_args.push_back(tensor_args.expert_mapping_tensor.buffer());
        reader_args.push_back(table_tensor.buffer());
        reader_args.append(
            {layer_weights[0],
             layer_weights[1],
             shards[member.ring_position].w0_w1_first_tile + member.first_column_group * layout.w0_w1_group_tiles,
             member.column_groups,
             member.first_output_group,
             member.output_groups,
             member.expert_group,
             core_index,
             program_cores,
             static_cast<uint32_t>(root.x),
             static_cast<uint32_t>(root.y),
             device_id,
             member.half_column ? 1u : 0u,
             member.half_output ? 1u : 0u});
        desc.kernels[kRowsReaderKernelIdx].emplace_runtime_args(member.core, reader_args);

        tt::tt_metal::KernelDescriptor::RTArgList writer_args;
        writer_args.push_back(tensor_args.input_tensor.buffer());
        writer_args.push_back(rows_tensor.buffer());
        writer_args.append(
            {member.activated_columns,
             member.first_column,
             member.output_groups,
             member.first_output_tile,
             member.output_tiles,
             member.expert_group,
             core_index % plan.group_cores});
        writer_args.append(core_mask(group_cores, grid));
        desc.kernels[kRowsWriterKernelIdx].emplace_runtime_args(member.core, writer_args);

        desc.kernels[kRowsComputeKernelIdx].emplace_runtime_args(
            member.core,
            {member.column_groups,
             member.output_groups,
             member.ring_position,
             member.half_column ? 1u : 0u,
             member.half_output ? 1u : 0u});
    }
    return desc;
}

}  // namespace

tt::tt_metal::WorkloadDescriptor MoEExpertRowsFactory::create_workload_descriptor(
    const MoEExpertRowsParams& args,
    const MoEExpertRowsInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor workload;
    for (const auto& coord : tensor_coords.coords()) {
        workload.programs.push_back(
            {ttnn::MeshCoordinateRange(coord),
             build_expert_rows_program_descriptor(args, coord, tensor_args, tensor_return_value)});
    }
    return workload;
}

void MoEExpertRowsFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const MoEExpertRowsParams& args,
    const MoEExpertRowsInputs& tensor_args,
    std::vector<ttnn::Tensor>& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& /*coord*/) {
    // Buffer addresses are bindings. The layer's weight addresses are offsets that layer_id moves.
    const auto& shape = args.shape;
    const uint32_t num_shards = shard_banks_of(tensor_args.w0_w1_tensor).size();
    const PreparedLayout layout =
        prepared_layout(shape.hidden_tiles(), shape.intermediate_tiles(), shape.has_bias, num_shards);
    const auto layer_weights = layer_weight_args(args, tensor_args, layout);
    auto& reader_args_by_core = tt::tt_metal::GetRuntimeArgs(program, kRowsReaderKernelIdx);
    for (auto& reader_args_column : reader_args_by_core) {
        for (auto& reader_args : reader_args_column) {
            if (reader_args.size() == 0) {
                continue;
            }
            for (uint32_t weight = 0; weight < layer_weights.size(); ++weight) {
                reader_args[kReaderLayerWeightArgs + weight] = layer_weights[weight];
            }
        }
    }
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<ttnn::Tensor> moe_expert_rows(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& expert_indices_tensor,
    const ttnn::Tensor& expert_scores_tensor,
    const ttnn::Tensor& expert_mapping_tensor,
    const ttnn::Tensor& w0_w1_tensor,
    const ttnn::Tensor& w2_tensor,
    const ttnn::experimental::prim::MoEExpertRowsParams& params) {
    using OperationType = ttnn::experimental::prim::MoEExpertRowsDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        params,
        OperationType::tensor_args_t{
            .input_tensor = input_tensor,
            .expert_indices_tensor = expert_indices_tensor,
            .expert_scores_tensor = expert_scores_tensor,
            .expert_mapping_tensor = expert_mapping_tensor,
            .w0_w1_tensor = w0_w1_tensor,
            .w2_tensor = w2_tensor});
}

}  // namespace ttnn::prim