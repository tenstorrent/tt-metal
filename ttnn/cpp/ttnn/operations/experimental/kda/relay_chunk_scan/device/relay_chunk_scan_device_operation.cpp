// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "relay_chunk_scan_device_operation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <span>

#include <tt-metalium/data_formats/tile_constants.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"
#include "ttnn/operations/experimental/kda/kda_performance_model.hpp"

namespace ttnn::experimental::prim {

RelayChunkScanOperation::program_factory_t RelayChunkScanOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return RelayChunkScanProgramFactory{};
}

// A cache hit matches every attribute and tensor spec the miss validated: only the buffers can differ.
void RelayChunkScanOperation::validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t& in) {
    constexpr std::string_view operation_name = "relay_chunk_scan";
    const std::initializer_list<std::pair<const Tensor*, std::string_view>> tensors = {
        {&in.packed, "packed"},
        {&in.terms, "terms"},
        {&in.initial_state, "initial_state"},
        {&in.actual_start, "actual_start"},
        {&in.landing, "landing"},
        {&in.gate_workspace, "gate_workspace"},
        {&in.final_e, "final_e"},
        {&in.final_b, "final_b"},
        {&in.gate_input, "gate_input"},
        {in.gate_local ? &*in.gate_local : nullptr, "gate_local"},
        {&in.gate_weight, "gate_weight"},
        {&in.norm_weight, "norm_weight"},
        {in.slab_index ? &*in.slab_index : nullptr, "slab_index"}};
    kda_factory_detail::check_allocated_on_device(in.packed, tensors, operation_name);
    kda_factory_detail::check_distinct_buffers(tensors, operation_name);
    // actual_end may share a buffer with actual_start, so it is only checked for allocation and device.
    kda_factory_detail::check_allocated_on_device(
        in.packed, {{in.actual_end ? &*in.actual_end : nullptr, "actual_end"}}, operation_name);
}

void RelayChunkScanOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    constexpr std::string_view operation_name = "relay_chunk_scan";
    using tt::tt_metal::CoreCoord;
    TT_FATAL(
        attrs.local_rows > 0 && attrs.local_rows % tt::constants::TILE_HEIGHT == 0,
        "{}: local_rows must be positive and 32-aligned",
        operation_name);
    kda_factory_detail::check_actual_start(in.packed, in.actual_start, operation_name);
    kda_factory_detail::check_allocated_device_tensor(in.packed, operation_name, "packed");
    kda_factory_detail::check_layout(in.packed, tt::tt_metal::Layout::TILE, operation_name, "packed");
    kda_factory_detail::check_dtype(in.packed, tt::tt_metal::DataType::BFLOAT16, operation_name, "packed");
    kda_factory_detail::check_interleaved(in.packed, operation_name, "packed");
    kda_factory_detail::check_allocated_device_tensor(in.initial_state, operation_name, "initial_state");
    kda_factory_detail::check_layout(in.initial_state, tt::tt_metal::Layout::TILE, operation_name, "initial_state");
    kda_factory_detail::check_dtype(in.initial_state, tt::tt_metal::DataType::FLOAT32, operation_name, "initial_state");
    kda_factory_detail::check_same_device(in.packed, in.initial_state, operation_name, "initial_state");
    const auto& state = in.initial_state.logical_shape();
    if (in.slab_index) {
        TT_FATAL(state.rank() == 4 && state[2] == state[3], "{}: slab must be [entries, H, K, K]", operation_name);
        kda_factory_detail::check_slab_index(
            in.packed,
            in.initial_state,
            *in.slab_index,
            attrs.slab_index_scale,
            attrs.slab_index_offset,
            operation_name);
    } else {
        kda_factory_detail::check_interleaved(in.initial_state, operation_name, "initial_state");
        TT_FATAL(state.rank() == 3 && state[1] == state[2], "{}: initial_state must be [B*H, K, K]", operation_name);
    }
    const auto [heads, key_dim] = relay_state_geometry(in);
    if (in.slab_index && attrs.sequence_parallel_axis < in.packed.device()->shape().dims()) {
        const uint32_t sp_size = in.packed.device()->shape()[attrs.sequence_parallel_axis];
        TT_FATAL(
            heads % sp_size == 0 && state[1] * sp_size == heads,
            "{}: a slab entry holds this SP rank's {} of the {} heads (SP-sharded), got {}",
            operation_name,
            heads / sp_size,
            heads,
            state[1]);
    }
    TT_FATAL(
        heads > 0 && key_dim > 0 && key_dim % tt::constants::TILE_WIDTH == 0,
        "{}: B*H must be positive and K positive and tile aligned",
        operation_name);
    const uint32_t key_tiles = key_dim / tt::constants::TILE_WIDTH;
    // The reader indexes packed as B*H x K tile rows of the complement form [E | B], E = A - I.
    TT_FATAL(
        in.packed.padded_shape()[-1] == 2 * key_dim && in.packed.physical_volume() == heads * key_dim * 2 * key_dim,
        "{}: packed must be [B*H, K, 2K] rows of [E | B]",
        operation_name);

    const auto* mesh = in.packed.device();
    TT_FATAL(attrs.sequence_parallel_axis < mesh->shape().dims(), "{}: invalid sequence_parallel_axis", operation_name);
    TT_FATAL(mesh->shape()[attrs.sequence_parallel_axis] >= 2, "{}: needs at least two SP ranks", operation_name);
    // Each head's correction cores, then the relay's sender per link and the final carry's second, then the relay
    // cores.
    const uint32_t senders = 2 * attrs.num_links;
    const uint32_t chunks = attrs.local_rows / tt::constants::TILE_HEIGHT;
    TT_FATAL(attrs.num_links > 0, "{}: needs at least one link", operation_name);
    const uint32_t relay_cores = relay_chunk_scan_relay_cores(heads, attrs.num_links);
    TT_FATAL(
        attrs.cores.size() > senders + relay_cores && (attrs.cores.size() - senders - relay_cores) % heads == 0,
        "{}: needs the same number of cores per head, then two senders per link, then {} relay cores",
        operation_name,
        relay_cores);
    const uint32_t cores_per_head = relay_chunk_scan_cores_per_head(attrs.cores.size(), heads, attrs.num_links);
    TT_FATAL(
        chunks % cores_per_head == 0,
        "{}: a head's {} cores must split its {} chunks evenly",
        operation_name,
        cores_per_head,
        chunks);
    TT_FATAL(
        (heads + attrs.num_links - 1) / attrs.num_links <= relay_chunk_scan_max_heads_per_link,
        "{}: {} heads over {} links put {} on a sender, which has semaphores for {}: use more links",
        operation_name,
        heads,
        attrs.num_links,
        (heads + attrs.num_links - 1) / attrs.num_links,
        relay_chunk_scan_max_heads_per_link);
    TT_FATAL(
        std::set<CoreCoord>(attrs.cores.begin(), attrs.cores.end()).size() == attrs.cores.size(),
        "{}: the cores must be distinct",
        operation_name);
    const auto grid = mesh->compute_with_storage_grid_size();
    for (const auto& core : attrs.cores) {
        TT_FATAL(
            core.x < grid.x && core.y < grid.y,
            "{}: core {} is outside the {}x{} compute grid",
            operation_name,
            core,
            grid.x,
            grid.y);
    }
    // Each correction kernel finds its head and share from its place in the grid (kernels/correction_core.hpp): head
    // h's cores are a column, x = h / heads_per_column, from row first_row + (h % heads_per_column) * cores_per_head.
    const uint32_t first_row = attrs.cores[0].y;
    const uint32_t heads_per_column = relay_chunk_scan_heads_per_column(grid.y, first_row, cores_per_head);
    TT_FATAL(heads_per_column > 0, "{}: a head's {} cores do not fit a grid column", operation_name, cores_per_head);
    for (uint32_t i = 0; i < heads * cores_per_head; ++i) {
        const uint32_t h = i / cores_per_head;
        const CoreCoord expected(
            h / heads_per_column, first_row + h % heads_per_column * cores_per_head + i % cores_per_head);
        TT_FATAL(
            attrs.cores[i] == expected,
            "{}: head {}'s correction cores must be the column of the compute grid at {}; core {} is {}",
            operation_name,
            h,
            expected,
            i,
            attrs.cores[i]);
    }
    TT_FATAL(
        std::ranges::all_of(attrs.semaphore_addresses, [](uint32_t address) { return address != 0; }),
        "{}: needs {} global semaphores",
        operation_name,
        relay_semaphores);
    TT_FATAL(
        tt::tt_fabric::get_tt_fabric_max_payload_size_bytes() >= tt::constants::TILE_HW * sizeof(float),
        "{}: the relay sends whole FP32 tiles",
        operation_name);
    TT_FATAL(
        relay_chunk_scan_sender_slots(key_tiles, mesh->arch()) > 0,
        "{}: a tile column of K = {} tiles needs {} packet headers, the pool holds {} per RISC",
        operation_name,
        key_tiles,
        relay_chunk_scan_column_packets(key_tiles),
        fabric_packet_headers_per_risc(mesh->arch()));

    const std::vector<CoreCoord> relay_core_list(attrs.cores.end() - relay_cores, attrs.cores.end());
    kda_factory_detail::check_allocated_device_tensor(in.landing, operation_name, "landing");
    kda_factory_detail::check_layout(in.landing, tt::tt_metal::Layout::TILE, operation_name, "landing");
    kda_factory_detail::check_dtype(in.landing, tt::tt_metal::DataType::FLOAT32, operation_name, "landing");
    kda_factory_detail::check_same_device(in.packed, in.landing, operation_name, "landing");
    const auto& memory = in.landing.memory_config();
    TT_FATAL(
        memory.buffer_type() == tt::tt_metal::BufferType::L1 &&
            memory.memory_layout() == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        "{}: landing must be height-sharded in L1",
        operation_name);
    const auto& shard = *memory.shard_spec();
    constexpr uint32_t landing_states = relay_landing_slots * relay_chunk_scan_heads_per_relay_core;
    TT_FATAL(
        in.landing.physical_volume() == landing_states * relay_cores * key_dim * key_dim &&
            shard.shape[0] == landing_states * key_dim && shard.shape[1] == key_dim &&
            kda_factory_detail::same_cores(shard.grid, relay_core_list),
        "{}: landing must hold {} {} x {} states per relay core",
        operation_name,
        landing_states,
        key_dim,
        key_dim);
    // The output gate's workspace, on the correction cores.
    const std::vector<CoreCoord> gate_cores(attrs.cores.begin(), attrs.cores.begin() + heads * cores_per_head);
    kda_factory_detail::check_allocated_device_tensor(in.gate_workspace, operation_name, "gate_workspace");
    kda_factory_detail::check_same_device(in.packed, in.gate_workspace, operation_name, "gate_workspace");
    kda_factory_detail::check_layout(in.gate_workspace, tt::tt_metal::Layout::TILE, operation_name, "gate_workspace");
    kda_factory_detail::check_dtype(
        in.gate_workspace, tt::tt_metal::DataType::FLOAT32, operation_name, "gate_workspace");
    const auto& workspace = in.gate_workspace.memory_config();
    TT_FATAL(
        workspace.buffer_type() == tt::tt_metal::BufferType::L1 &&
            workspace.memory_layout() == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED &&
            kda_factory_detail::same_cores(workspace.shard_spec()->grid, gate_cores),
        "{}: gate_workspace must be height-sharded in L1 over the correction cores",
        operation_name);
    kda_factory_detail::check_distinct_buffers(
        {{&in.packed, "packed"},
         {&in.terms, "terms"},
         {&in.initial_state, "initial_state"},
         {&in.actual_start, "actual_start"},
         {&in.landing, "landing"},
         {&in.gate_workspace, "gate_workspace"},
         {&in.final_e, "final_e"},
         {&in.final_b, "final_b"},
         {&in.gate_input, "gate_input"},
         {in.gate_local ? &*in.gate_local : nullptr, "gate_local"},
         {&in.gate_weight, "gate_weight"},
         {&in.norm_weight, "norm_weight"},
         {in.slab_index ? &*in.slab_index : nullptr, "slab_index"}},
        operation_name);
    for (const auto& [tensor, name] : {std::pair{&in.final_e, "final_e"}, std::pair{&in.final_b, "final_b"}}) {
        kda_factory_detail::check_allocated_device_tensor(*tensor, operation_name, name);
        kda_factory_detail::check_layout(*tensor, tt::tt_metal::Layout::TILE, operation_name, name);
        kda_factory_detail::check_dtype(*tensor, tt::tt_metal::DataType::FLOAT32, operation_name, name);
        kda_factory_detail::check_interleaved(*tensor, operation_name, name);
        kda_factory_detail::check_same_device(in.packed, *tensor, operation_name, name);
        TT_FATAL(
            tensor->logical_shape() == tt::tt_metal::Shape({heads, key_dim, key_dim}),
            "{}: {} must be [B*H, K, K]",
            operation_name,
            name);
    }
    if (in.actual_end) {
        kda_factory_detail::check_actual_start(in.packed, *in.actual_end, operation_name);
    }
    // A sender reuses the relay's packet headers for the carry, a write per tile of a column and the arrival
    // increment, and reads the next column into a second stage slot.
    const uint32_t slots = relay_chunk_scan_sender_slots(key_tiles, mesh->arch());
    TT_FATAL(
        slots * relay_chunk_scan_column_packets(key_tiles) >= key_tiles + 1 && slots >= 2,
        "{}: the final carry needs {} packet headers and two stage slots per sender",
        operation_name,
        key_tiles + 1);
    // The chunk terms: in place in each core's shard, or streamed from interleaved memory.
    const std::span<const CoreCoord> correction_cores(attrs.cores.data(), heads * cores_per_head);
    kda_factory_detail::check_allocated_device_tensor(in.terms, operation_name, "terms");
    kda_factory_detail::check_layout(in.terms, tt::tt_metal::Layout::TILE, operation_name, "terms");
    kda_factory_detail::check_dtype(in.terms, tt::tt_metal::DataType::FLOAT32, operation_name, "terms");
    kda_factory_detail::check_same_device(in.packed, in.terms, operation_name, "terms");
    const uint32_t core_rows = chunks / cores_per_head * tt::constants::TILE_HEIGHT;
    const auto& terms_memory = in.terms.memory_config();
    if (terms_memory.is_sharded()) {
        TT_FATAL(
            in.terms.logical_shape() == Shape({heads * cores_per_head, 2, core_rows, key_dim}) &&
                terms_memory.buffer_type() == tt::tt_metal::BufferType::L1 &&
                terms_memory.memory_layout() == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED &&
                terms_memory.shard_spec()->shape[0] == 2 * core_rows && terms_memory.shard_spec()->shape[1] == key_dim,
            "{}: sharded terms must be [B*H * {}, 2, {}, K], height-sharded in L1 one core's rows per shard",
            operation_name,
            cores_per_head,
            core_rows);
        // Core i reads shard i in place.
        TT_FATAL(
            std::ranges::equal(kda_factory_detail::shard_cores(in.terms), correction_cores),
            "{}: terms must be sharded over the heads' cores in their order",
            operation_name);
    } else {
        TT_FATAL(
            in.terms.logical_shape() == Shape({heads, 2, chunks * tt::constants::TILE_HEIGHT, key_dim}),
            "{}: interleaved terms must be [B*H, 2, {}, K]",
            operation_name,
            chunks * tt::constants::TILE_HEIGHT);
    }
    // The gated output norm and its gate x @ W.
    TT_FATAL(
        attrs.num_heads > 0 && heads % attrs.num_heads == 0,
        "{}: {} heads are not a whole number of batches of {}",
        operation_name,
        heads,
        attrs.num_heads);
    std::vector<std::pair<const Tensor*, const char*>> norm_tensors = {
        {&in.gate_input, "gate_input"}, {&in.gate_weight, "gate_weight"}, {&in.norm_weight, "norm_weight"}};
    if (in.gate_local) {
        norm_tensors.emplace_back(&*in.gate_local, "gate_local");
    }
    for (const auto& [tensor, name] : norm_tensors) {
        kda_factory_detail::check_allocated_device_tensor(*tensor, operation_name, name);
        kda_factory_detail::check_layout(*tensor, tt::tt_metal::Layout::TILE, operation_name, name);
        if (tensor != &in.gate_weight) {
            kda_factory_detail::check_dtype(*tensor, tt::tt_metal::DataType::BFLOAT16, operation_name, name);
        } else {
            TT_FATAL(
                tensor->dtype() == tt::tt_metal::DataType::BFLOAT16 ||
                    tensor->dtype() == tt::tt_metal::DataType::BFLOAT8_B ||
                    tensor->dtype() == tt::tt_metal::DataType::BFLOAT4_B,
                "{}: gate_weight must be BFLOAT16, BFLOAT8_B or BFLOAT4_B, got {}",
                operation_name,
                tensor->dtype());
        }
        kda_factory_detail::check_interleaved(*tensor, operation_name, name);
        kda_factory_detail::check_same_device(in.packed, *tensor, operation_name, name);
    }
    const auto& input_shape = in.gate_input.logical_shape();
    TT_FATAL(
        input_shape.rank() == 3 && input_shape[0] == heads / attrs.num_heads && input_shape[1] == attrs.local_rows &&
            input_shape[2] % (2 * tt::constants::TILE_WIDTH) == 0,
        "{}: gate_input {} must be [{}, {}, X] with X a multiple of 2 tiles",
        operation_name,
        input_shape,
        heads / attrs.num_heads,
        attrs.local_rows);
    TT_FATAL(
        in.gate_weight.logical_shape() == Shape({input_shape[2], attrs.num_heads * key_dim}),
        "{}: gate_weight {} must be [X, H*K] = [{}, {}]",
        operation_name,
        in.gate_weight.logical_shape(),
        input_shape[2],
        attrs.num_heads * key_dim);
    if (in.gate_local) {
        const uint32_t tp_size = mesh->shape().dims() == 2 ? mesh->shape()[1 - attrs.sequence_parallel_axis] : 1;
        TT_FATAL(
            tp_size > 1 && input_shape[2] % (tp_size * tt::constants::TILE_WIDTH) == 0 &&
                in.gate_local->logical_shape() == Shape({input_shape[0], input_shape[1], input_shape[2] / tp_size}),
            "{}: gate_local {} must be gate_input's TP slice [{}, {}, X / {}] (TP > 1)",
            operation_name,
            in.gate_local->logical_shape(),
            input_shape[0],
            input_shape[1],
            tp_size);
    }
    TT_FATAL(
        in.norm_weight.logical_shape().rank() == 1 && in.norm_weight.logical_shape()[0] == key_dim,
        "{}: norm_weight must be [K] with K = {}",
        operation_name,
        key_dim);
    TT_FATAL(
        std::isfinite(attrs.norm_epsilon) && attrs.norm_epsilon > 0.0f,
        "{}: norm_epsilon must be finite and positive",
        operation_name);
    TT_FATAL(
        attrs.output_dtype == tt::tt_metal::DataType::FLOAT32 || attrs.output_dtype == tt::tt_metal::DataType::BFLOAT16,
        "{}: output_dtype must be FLOAT32 or BFLOAT16",
        operation_name);
    kda_factory_detail::check_output_interleaved(attrs.output_mem_config, operation_name);
    kda_factory_detail::check_compute_config(attrs.compute_kernel_config, operation_name);
    kda_factory_detail::check_fp32_dest_acc(
        attrs.compute_kernel_config, operation_name, "the hop adds the FP32 entry state and B exactly in DST");
    TT_FATAL(
        attrs.compute_kernel_config.math_fidelity == tt::tt_metal::MathFidelity::HiFi3 ||
            attrs.compute_kernel_config.math_fidelity == tt::tt_metal::MathFidelity::HiFi4,
        "{}: needs HiFi3 or above: below it the norm's FPU products truncate 1 / rms and the squares",
        operation_name);
    // Compute forms a K-tile column of the next state in one DST pass.
    const uint32_t dst_tiles = ttnn::get_dest_reg_count(attrs.compute_kernel_config);
    TT_FATAL(
        key_tiles <= dst_tiles,
        "{}: K = {} tiles exceeds the {} DST tiles a column is formed in",
        operation_name,
        key_tiles,
        dst_tiles);
}

RelayChunkScanOperation::spec_return_value_t RelayChunkScanOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    const auto spec =
        [](const ttnn::Shape& shape, tt::tt_metal::DataType dtype, const tt::tt_metal::MemoryConfig& memory) {
            return tt::tt_metal::TensorSpec(
                shape, tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), memory));
        };
    // The normalized outputs [B, T, H*V], and the final carry: with a slab, written into the slab's entry, in place.
    const auto [heads, key_dim] = relay_state_geometry(in);
    // Output specs are derived before validation runs.
    TT_FATAL(
        attrs.num_heads > 0 && heads % attrs.num_heads == 0,
        "relay_chunk_scan: num_heads {} must be positive and divide B*H = {}",
        attrs.num_heads,
        heads);
    const uint32_t value_dim = in.initial_state.logical_shape()[-1];
    return {
        spec(
            Shape({heads / attrs.num_heads, attrs.local_rows, attrs.num_heads * value_dim}),
            attrs.output_dtype,
            attrs.output_mem_config),
        in.slab_index
            ? in.initial_state.tensor_spec()
            : spec(Shape({heads, key_dim, value_dim}), tt::tt_metal::DataType::FLOAT32, ttnn::DRAM_MEMORY_CONFIG)};
}

RelayChunkScanOperation::tensor_return_value_t RelayChunkScanOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    const auto specs = compute_output_specs(attrs, in);
    tensor_return_value_t outputs;
    outputs.push_back(create_device_tensor(specs[0], in.packed.device()));
    // With a slab, the final carry is written into the slab's entry, in place.
    outputs.push_back(in.slab_index ? in.initial_state : create_device_tensor(specs[1], in.packed.device()));
    return outputs;
}

tt::tt_metal::operation::OpPerformanceModelGeneral<RelayChunkScanOperation::tensor_return_value_t>
RelayChunkScanOperation::create_op_performance_model(
    const operation_attributes_t& attrs, const tensor_args_t& in, tensor_return_value_t& outputs) {
    using namespace kda_performance_model;
    const auto geometry = relay_state_geometry(in);
    const double heads = geometry.heads;
    const double key_dim = geometry.key_dim;
    const double rows = attrs.local_rows;
    // Per head: one hop S_out = S_in + E S_in + B (E = A - I), then each chunk row's F0 + (FI - F0) @ seed, and the
    // head's output gate x @ W over its rows.
    const double gate_k = in.gate_input.logical_shape()[-1];
    const KdaFpuWork work{
        .fpu_matrix_flops = heads * 2.0 * ((key_dim + rows) * key_dim * key_dim + rows * gate_k * key_dim),
        .fpu_add_ops = heads * (key_dim + rows) * key_dim,
    };
    const std::array<const Tensor*, 5> inputs = {
        &in.packed, &in.terms, &in.initial_state, &in.gate_input, &in.gate_weight};
    return make_profiler_model(work, inputs, outputs, attrs.compute_kernel_config.math_fidelity);
}

std::tuple<Tensor, Tensor> relay_chunk_scan(
    const Tensor& packed,
    const Tensor& terms,
    const Tensor& initial_state,
    const Tensor& final_e,
    const Tensor& final_b,
    const Tensor& gate_input,
    const std::optional<Tensor>& gate_local,
    const Tensor& gate_weight,
    const Tensor& norm_weight,
    const Tensor& actual_start,
    const std::optional<Tensor>& actual_end,
    const Tensor& landing,
    const Tensor& gate_workspace,
    uint32_t num_heads,
    tt::tt_metal::MathFidelity gate_math_fidelity,
    float norm_epsilon,
    tt::tt_metal::DataType output_dtype,
    const tt::tt_metal::MemoryConfig& memory_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    const std::array<uint32_t, relay_semaphores>& semaphore_addresses,
    uint32_t sequence_parallel_axis,
    uint32_t local_rows,
    uint32_t num_links,
    const std::vector<tt::tt_metal::CoreCoord>& cores,
    const std::optional<Tensor>& slab_index,
    uint32_t slab_index_scale,
    uint32_t slab_index_offset) {
    auto outputs = ttnn::device_operation::launch<RelayChunkScanOperation>(
        RelayChunkScanParams{
            .sequence_parallel_axis = sequence_parallel_axis,
            .local_rows = local_rows,
            .num_links = num_links,
            .cores = cores,
            .num_heads = num_heads,
            .gate_math_fidelity = gate_math_fidelity,
            .norm_epsilon = norm_epsilon,
            .output_dtype = output_dtype,
            .output_mem_config = memory_config,
            .compute_kernel_config = compute_kernel_config,
            .semaphore_addresses = semaphore_addresses,
            .slab_index_scale = slab_index_scale,
            .slab_index_offset = slab_index_offset},
        RelayChunkScanInputs{
            .packed = packed,
            .terms = terms,
            .initial_state = initial_state,
            .actual_start = actual_start,
            .landing = landing,
            .gate_workspace = gate_workspace,
            .final_e = final_e,
            .final_b = final_b,
            .actual_end = actual_end,
            .gate_input = gate_input,
            .gate_local = gate_local,
            .gate_weight = gate_weight,
            .norm_weight = norm_weight,
            .slab_index = slab_index});
    return {outputs[0], outputs[1]};
}

}  // namespace ttnn::experimental::prim
