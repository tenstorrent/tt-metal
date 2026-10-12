// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/qkv_causal_conv1d_silu_program_factory.hpp"

#include <algorithm>
#include <limits>
#include <optional>
#include <vector>

#include <internal/fabric.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/semaphore_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/experimental/kda/factory/chronology_binding.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"
#include "ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/qkv_causal_conv1d_silu_layout.hpp"

namespace ttnn::experimental::prim {

namespace {

using namespace tt::tt_metal::experimental;

constexpr uint32_t no_fabric_connection = 0xFF;
const TensorParamName predecessor_rows_tensor{"predecessor_rows"};
const TensorParamName final_history_tensor{"final_history"};
const SemaphoreSpecName predecessor_ready_semaphore{"predecessor_ready"};
const SemaphoreSpecName staged_semaphore{"staged"};

// Fabric connections of one rank's exchange core along the SP axis, and how its sends use them.
struct ExchangeRoute {
    std::vector<tt::tt_fabric::FabricNodeId> neighbors;
    uint32_t forward_connection = no_fabric_connection;
    uint32_t backward_connection = no_fabric_connection;
    // A multicast from this rank covers every other rank exactly once.
    uint32_t forward_range = 0;
    uint32_t backward_range = 0;
    uint32_t successor_connection = no_fabric_connection;
    uint32_t successor_hops = 0;
    uint32_t successor_chip = 0;
    uint32_t successor_mesh = 0;
};

ExchangeRoute exchange_route(const ttnn::MeshDevice& device, const ttnn::MeshCoordinate& coordinate, uint32_t axis) {
    const uint32_t sp_size = device.shape()[axis];
    const uint32_t rank = coordinate[axis];
    const bool ring = ttnn::ccl::is_axis_wrap_wired(device, axis);
    const auto at_rank = [&](uint32_t other) {
        auto result = coordinate;
        result[axis] = other;
        return device.get_fabric_node_id(result);
    };
    const auto successor = at_rank((rank + 1) % sp_size);
    ExchangeRoute route{.successor_chip = successor.chip_id, .successor_mesh = *successor.mesh_id};
    if (ring || rank + 1 < sp_size) {
        route.forward_connection = route.neighbors.size();
        route.neighbors.push_back(at_rank((rank + 1) % sp_size));
    }
    if (ring || rank > 0) {
        route.backward_connection = route.neighbors.size();
        route.neighbors.push_back(at_rank((rank + sp_size - 1) % sp_size));
    }
    route.forward_range = ring ? sp_size / 2 : sp_size - 1 - rank;
    route.backward_range = ring ? sp_size - 1 - sp_size / 2 : rank;
    // The successor is the forward neighbor, except for a line's last rank, which reaches rank 0 backward.
    const bool successor_forward = ring || rank + 1 < sp_size;
    route.successor_connection = successor_forward ? route.forward_connection : route.backward_connection;
    route.successor_hops = successor_forward ? 1 : sp_size - 1;
    return route;
}

// Add the fabric history exchange to every rank's program: one kernel on exchange_core that sends the rows the
// convolution cores staged, releases them once the predecessor's rows have landed, and distributes the carry.
void add_history_exchange(
    ttnn::device_operation::MeshWorkloadArtifacts& workload,
    const QkvCausalConv1dSiluParams& attrs,
    const QkvCausalConv1dSiluInputs& in,
    const std::vector<CoreCoord>& convolution_cores,
    const CoreCoord& exchange_core,
    uint32_t channel_tiles) {
    const auto& device = in.input.mesh_tensor().device();
    const uint32_t axis = attrs.sequence_parallel_axis;
    const uint32_t sp_size = device.shape()[axis];
    TT_FATAL(
        sp_size == 1 || ttnn::ccl::is_axis_straight(device, axis),
        "qkv_causal_conv1d_silu: the fabric history exchange needs a straight SP axis");
    const uint32_t row_bytes = channel_tiles * tt::constants::TILE_WIDTH * sizeof(uint16_t);
    // Any multiple of 64 bytes keeps the row chunks aligned when nothing is sent.
    const uint32_t max_payload =
        sp_size > 1 ? static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes()) : 64;

    std::vector<uint32_t> convolution_core_xy;
    convolution_core_xy.reserve(convolution_cores.size());
    for (const auto& core : convolution_cores) {
        const auto xy = device.worker_core_from_logical_core(core);
        convolution_core_xy.push_back(static_cast<uint32_t>(xy.x) | (static_cast<uint32_t>(xy.y) << 16));
    }

    const KernelSpecName exchange_kernel{"exchange"};
    const ScratchpadSpecName exchange_scratch{"exchange_scratch"};
    const SemaphoreSpecName teardown[2] = {SemaphoreSpecName{"teardown0"}, SemaphoreSpecName{"teardown1"}};
    const SemaphoreSpecName buffer_index[2] = {SemaphoreSpecName{"buffer_index0"}, SemaphoreSpecName{"buffer_index1"}};
    const CoreRangeSet exchange_nodes{CoreRange(exchange_core)};
    const std::vector<std::string> semaphore_args = {"entry_semaphore", "predecessor_semaphore", "carry_semaphore"};

    for (auto& program : workload.programs) {
        const auto coordinate = program.range.start_coord();
        // A single rank exchanges nothing and may run without a fabric: no connections, no cores to release.
        const auto route = sp_size > 1 ? exchange_route(device, coordinate, axis) : ExchangeRoute{};
        std::vector<uint32_t> varargs;
        if (sp_size > 1) {
            TT_FATAL(
                route.successor_connection != no_fabric_connection,
                "qkv_causal_conv1d_silu: no fabric route to the successor");
            const std::vector<uint32_t> unbound_semaphores(route.neighbors.size(), 0);
            varargs = tt::tt_metal::internal::compute_fabric_connection_rt_args(
                device.get_fabric_node_id(coordinate), route.neighbors, {0}, unbound_semaphores, unbound_semaphores);
            varargs.insert(varargs.end(), convolution_core_xy.begin(), convolution_core_xy.end());
        }

        KernelSpec kernel{
            .unique_id = exchange_kernel,
            .source =
                "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/kernels/dataflow/"
                "history_exchange_qkv_causal_conv1d_silu.cpp",
            .scratchpad_bindings = {{exchange_scratch, "exchange_scratch"}},
            .tensor_bindings =
                {
                    TensorBinding{TensorParamName{"history"}, "history"},
                    TensorBinding{predecessor_rows_tensor, "predecessor_rows"},
                    TensorBinding{final_history_tensor, "final_history"},
                    TensorBinding{TensorParamName{"actual_start"}, "actual_start"},
                },
            .compile_time_args =
                {{"channel_tiles", channel_tiles},
                 {"max_payload", max_payload},
                 {"named_runtime_args", static_cast<uint32_t>(semaphore_args.size())},
                 {"num_connections", static_cast<uint32_t>(route.neighbors.size())},
                 {"forward_connection", route.forward_connection},
                 {"forward_range", route.forward_range},
                 {"backward_connection", route.backward_connection},
                 {"backward_range", route.backward_range},
                 {"successor_connection", route.successor_connection},
                 {"successor_hops", route.successor_hops},
                 {"successor_chip", route.successor_chip},
                 {"successor_mesh", route.successor_mesh},
                 {"num_convolution_cores", static_cast<uint32_t>(convolution_cores.size())},
                 {"sp_rank", coordinate[axis]},
                 {"sp_size", sp_size},
                 {"local_rows", attrs.sequence},
                 {"has_actual_end", uint32_t(in.actual_end.has_value())}},
            .runtime_arg_schema = {.runtime_arg_names = {semaphore_args.begin(), semaphore_args.end()}},
            .hw_config = ttnn::create_writer_datamovement_config(),
        };
        kernel.semaphore_bindings = {
            SemaphoreBinding{staged_semaphore, "staged"},
            SemaphoreBinding{predecessor_ready_semaphore, "predecessor_ready"},
            SemaphoreBinding{teardown[0], "teardown0"},
            SemaphoreBinding{buffer_index[0], "buffer_index0"},
            SemaphoreBinding{teardown[1], "teardown1"},
            SemaphoreBinding{buffer_index[1], "buffer_index1"},
        };
        if (sp_size > 1) {
            for (const auto& [name, value] : tt::tt_metal::internal::get_fabric_kernel_defines()) {
                kernel.compiler_options.defines.insert({name, value});
            }
        }
        kernel.advanced_options.num_runtime_varargs = static_cast<uint32_t>(varargs.size());
        if (in.actual_end) {
            kernel.tensor_bindings.push_back({TensorParamName{"actual_end"}, "actual_end"});
        }
        auto& spec = program.spec;
        spec.kernels.push_back(std::move(kernel));
        for (uint32_t i = 0; i < 2; ++i) {
            spec.semaphores.push_back(SemaphoreSpec{.unique_id = teardown[i], .target_nodes = exchange_nodes});
            spec.semaphores.push_back(SemaphoreSpec{.unique_id = buffer_index[i], .target_nodes = exchange_nodes});
        }
        spec.scratchpads.push_back(
            {.unique_id = exchange_scratch, .size_per_node = 128 + 2 * qkv_conv::history_rows * row_bytes});
        spec.work_units.push_back(
            WorkUnitSpec{.name = "exchange", .kernels = {exchange_kernel}, .target_nodes = exchange_nodes});

        KernelRunArgs exchange_args{.kernel = exchange_kernel};
        for (uint32_t i = 0; i < semaphore_args.size(); ++i) {
            AddRuntimeArgsForNode(
                exchange_args.runtime_arg_values,
                exchange_core,
                {{semaphore_args[i], attrs.exchange_semaphore_addresses[i]}});
        }
        if (!varargs.empty()) {
            exchange_args.advanced_options.runtime_varargs[NodeCoord(exchange_core)] = std::move(varargs);
        }
        program.run_params.kernel_run_args.push_back(std::move(exchange_args));
    }
}

}  // namespace

ttnn::device_operation::MeshWorkloadArtifacts QkvCausalConv1dSiluProgramFactory::create_mesh_workload_artifacts(
    const QkvCausalConv1dSiluParams& attrs,
    const QkvCausalConv1dSiluInputs& in,
    std::vector<Tensor>& outputs,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    const auto& input = in.input.mesh_tensor();
    const auto& device = input.device();
    const bool exchange = in.exchange_buffer.has_value();
    const auto& predecessor_rows = exchange ? in.exchange_buffer->mesh_tensor() : in.predecessor_carry->mesh_tensor();

    const uint32_t Mt = attrs.sequence / tt::constants::TILE_HEIGHT;
    const uint32_t Qt = attrs.q_width / tt::constants::TILE_WIDTH;
    const uint32_t Kt = attrs.k_width / tt::constants::TILE_WIDTH;
    const uint32_t Vt = attrs.v_width / tt::constants::TILE_WIDTH;
    const uint32_t Ct = Qt + Kt + Vt;
    const uint32_t block_ct = attrs.channel_chunk_size / tt::constants::TILE_WIDTH;
    const uint32_t num_blocks = Ct / block_ct;
    const uint32_t group = qkv_causal_conv1d_silu_group(block_ct);
    const uint32_t input_wt = input.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    const uint32_t input_ct_offset = attrs.input_column_offset / tt::constants::TILE_WIDTH;
    // The fabric history exchange runs on the grid's last core, which the convolution then leaves free.
    const auto grid = device.compute_with_storage_grid_size();
    const CoreCoord exchange_core{grid.x - 1, grid.y - 1};
    auto dist = kda_factory_detail::distribute_prep(
        grid, Mt * num_blocks, exchange ? grid.x * grid.y - 1 : std::numeric_limits<uint32_t>::max());
    const auto& cores = dist.core_set;
    // Every convolution core stages a share of the channel tiles of the rows the exchange core sends.
    const uint32_t max_staged_ct = exchange ? tt::div_up(Ct, static_cast<uint32_t>(dist.cores.size())) : 0;
    const auto exchange_core_xy = device.worker_core_from_logical_core(exchange_core);

    const KernelSpecName reader_name{"reader"};
    const KernelSpecName writer_name{"writer"};
    const KernelSpecName compute_name{"compute"};
    const DFBSpecName act_name{"act_tile"};
    const DFBSpecName weights_name{"weights"};
    const DFBSpecName taps_name{"taps"};
    const DFBSpecName eye_name{"eye"};
    const DFBSpecName output_name{"output"};
    const ScratchpadSpecName scratch_name{"act_window"};
    const ScratchpadSpecName staging_scratch_name{"staging"};

    const auto data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const uint32_t tile_size = tt::tile_size(data_format);
    auto make_dfb = [data_format, tile_size](const DFBSpecName& name, uint32_t tiles) {
        return DataflowBufferSpec{
            .unique_id = name, .entry_size = tile_size, .num_entries = tiles, .data_format_metadata = data_format};
    };
    Group<DataflowBufferSpec> dfbs = {
        make_dfb(act_name, 2 * qkv_causal_conv1d_silu_tap_count * group),
        make_dfb(weights_name, qkv_causal_conv1d_silu_tap_count * block_ct),
        make_dfb(taps_name, qkv_causal_conv1d_silu_tap_count * block_ct),
        make_dfb(eye_name, 1),
        make_dfb(output_name, 2 * group),
    };

    const TensorParamName input_tensor{"input"};
    const TensorParamName history_tensor{"history"};
    const TensorParamName tap_tensors[4] = {
        TensorParamName{"tap0"}, TensorParamName{"tap1"}, TensorParamName{"tap2"}, TensorParamName{"tap3"}};
    const TensorParamName q_tensor{"q"};
    const TensorParamName k_tensor{"k"};
    const TensorParamName v_tensor{"v"};
    const TensorParamName actual_start_tensor{"actual_start"};

    KernelSpec reader{
        .unique_id = reader_name,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/kernels/dataflow/"
            "reader_qkv_causal_conv1d_silu.cpp",
        .dfb_bindings =
            {
                DFBBinding{act_name, "act_tile", DFBEndpointType::PRODUCER},
                DFBBinding{taps_name, "taps", DFBEndpointType::PRODUCER},
                DFBBinding{eye_name, "eye", DFBEndpointType::PRODUCER},
            },
        .semaphore_bindings = {SemaphoreBinding{predecessor_ready_semaphore, "predecessor_ready"}},
        .scratchpad_bindings = {{scratch_name, "act_window"}},
        .tensor_bindings =
            {
                TensorBinding{input_tensor, "input"},
                TensorBinding{history_tensor, "history"},
                TensorBinding{predecessor_rows_tensor, "predecessor_rows"},
                TensorBinding{tap_tensors[0], "tap0"},
                TensorBinding{tap_tensors[1], "tap1"},
                TensorBinding{tap_tensors[2], "tap2"},
                TensorBinding{tap_tensors[3], "tap3"},
                TensorBinding{actual_start_tensor, "actual_start"},
            },
        .compile_time_args =
            {{"block_ct", block_ct},
             {"Mt", Mt},
             {"group", group},
             {"input_wt", input_wt},
             {"input_ct_offset", input_ct_offset},
             {"exchange", uint32_t(exchange)}},
        .runtime_arg_schema = {.runtime_arg_names = {"wi_start", "wi_count"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };
    KernelSpec writer{
        .unique_id = writer_name,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/kernels/dataflow/"
            "writer_qkv_causal_conv1d_silu.cpp",
        .dfb_bindings = {DFBBinding{output_name, "output", DFBEndpointType::CONSUMER}},
        .semaphore_bindings = {SemaphoreBinding{staged_semaphore, "staged"}},
        .scratchpad_bindings = {{staging_scratch_name, "staging"}},
        .tensor_bindings =
            {
                TensorBinding{q_tensor, "q"},
                TensorBinding{k_tensor, "k"},
                TensorBinding{v_tensor, "v"},
                TensorBinding{input_tensor, "input"},
                TensorBinding{predecessor_rows_tensor, "predecessor_rows"},
                TensorBinding{actual_start_tensor, "actual_start"},
            },
        .compile_time_args =
            {{"Qt", Qt},
             {"Kt", Kt},
             {"Vt", Vt},
             {"block_ct", block_ct},
             {"Mt", Mt},
             {"group", group},
             {"exchange", uint32_t(exchange)},
             {"input_wt", input_wt},
             {"input_ct_offset", input_ct_offset},
             {"exchange_core_x", static_cast<uint32_t>(exchange_core_xy.x)},
             {"exchange_core_y", static_cast<uint32_t>(exchange_core_xy.y)},
             {"max_staged_ct", max_staged_ct}},
        .runtime_arg_schema = {.runtime_arg_names = {"wi_start", "wi_count", "staged_ct_start", "staged_ct_count"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };
    KernelSpec compute{
        .unique_id = compute_name,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/kernels/compute/"
            "qkv_causal_conv1d_silu.cpp",
        .compiler_options = {.opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                DFBBinding{act_name, "act_tile", DFBEndpointType::CONSUMER},
                DFBBinding{taps_name, "taps", DFBEndpointType::CONSUMER},
                DFBBinding{eye_name, "eye", DFBEndpointType::CONSUMER},
                DFBBinding{weights_name, "weights", DFBEndpointType::PRODUCER},
                DFBBinding{weights_name, "weights", DFBEndpointType::CONSUMER},
                DFBBinding{output_name, "output", DFBEndpointType::PRODUCER},
            },
        .compile_time_args =
            {{"block_ct", block_ct}, {"Mt", Mt}, {"group", group}, {"defer_first_rows", uint32_t(exchange)}},
        .runtime_arg_schema = {.runtime_arg_names = {"wi_start", "wi_count"}},
        .hw_config = ttnn::to_compute_hardware_config(attrs.compute_kernel_config),
    };

    KernelRunArgs reader_run_args{.kernel = reader_name};
    KernelRunArgs writer_run_args{.kernel = writer_name};
    KernelRunArgs compute_run_args{.kernel = compute_name};
    for (uint32_t i = 0; i < dist.cores.size(); ++i) {
        const auto& core = dist.cores[i];
        for (auto* args : {&reader_run_args, &writer_run_args, &compute_run_args}) {
            AddRuntimeArgsForNode(
                args->runtime_arg_values, core, {{"wi_start", dist.wi_start[i]}, {"wi_count", dist.wi_count[i]}});
        }
        const uint32_t staged_begin = std::min(Ct, i * max_staged_ct);
        const uint32_t staged_end = std::min(Ct, staged_begin + max_staged_ct);
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"staged_ct_start", staged_begin}, {"staged_ct_count", staged_end - staged_begin}});
    }

    Group<TensorParameter> tensor_parameters = {
        TensorParameter{.unique_id = input_tensor, .spec = input.tensor_spec()},
        TensorParameter{.unique_id = history_tensor, .spec = in.history.mesh_tensor().tensor_spec()},
        TensorParameter{.unique_id = predecessor_rows_tensor, .spec = predecessor_rows.tensor_spec()},
        TensorParameter{.unique_id = tap_tensors[0], .spec = in.tap0.mesh_tensor().tensor_spec()},
        TensorParameter{.unique_id = tap_tensors[1], .spec = in.tap1.mesh_tensor().tensor_spec()},
        TensorParameter{.unique_id = tap_tensors[2], .spec = in.tap2.mesh_tensor().tensor_spec()},
        TensorParameter{.unique_id = tap_tensors[3], .spec = in.tap3.mesh_tensor().tensor_spec()},
        TensorParameter{.unique_id = q_tensor, .spec = outputs[0].mesh_tensor().tensor_spec()},
        TensorParameter{.unique_id = k_tensor, .spec = outputs[1].mesh_tensor().tensor_spec()},
        TensorParameter{.unique_id = v_tensor, .spec = outputs[2].mesh_tensor().tensor_spec()},
        TensorParameter{.unique_id = actual_start_tensor, .spec = in.actual_start.mesh_tensor().tensor_spec()},
    };

    ProgramRunArgs run_args;
    run_args.tensor_args = {
        {input_tensor, input},
        {history_tensor, in.history.mesh_tensor()},
        {predecessor_rows_tensor, predecessor_rows},
        {tap_tensors[0], in.tap0.mesh_tensor()},
        {tap_tensors[1], in.tap1.mesh_tensor()},
        {tap_tensors[2], in.tap2.mesh_tensor()},
        {tap_tensors[3], in.tap3.mesh_tensor()},
        {q_tensor, outputs[0].mesh_tensor()},
        {k_tensor, outputs[1].mesh_tensor()},
        {v_tensor, outputs[2].mesh_tensor()},
        {actual_start_tensor, in.actual_start.mesh_tensor()},
    };
    if (exchange) {
        const auto& final_history = outputs[3].mesh_tensor();
        tensor_parameters.push_back({.unique_id = final_history_tensor, .spec = final_history.tensor_spec()});
        run_args.tensor_args.emplace(final_history_tensor, final_history);
    }

    // Every convolution core waits on predecessor_ready before reading exchanged rows; the exchange core raises it.
    CoreRangeSet semaphore_nodes = cores;
    if (exchange) {
        semaphore_nodes = semaphore_nodes.merge(CoreRangeSet(CoreRange(exchange_core)));
    }
    ProgramSpec spec{
        .name = "qkv_causal_conv1d_silu",
        .kernels = {},
        .dataflow_buffers = std::move(dfbs),
        .semaphores =
            {
                SemaphoreSpec{.unique_id = predecessor_ready_semaphore, .target_nodes = semaphore_nodes},
                SemaphoreSpec{.unique_id = staged_semaphore, .target_nodes = semaphore_nodes},
            },
        .scratchpads =
            {{.unique_id = scratch_name,
              .size_per_node = static_cast<uint32_t>(qkv_causal_conv1d_silu_scratch_bytes(block_ct, tile_size))},
             {.unique_id = staging_scratch_name,
              .size_per_node = 128 + qkv_conv::staged_rows_scratch_bytes(2 * qkv_conv::history_rows, max_staged_ct)}},
        .tensor_parameters = std::move(tensor_parameters),
        .work_units =
            {
                WorkUnitSpec{
                    .name = "main",
                    .kernels = {reader_name, writer_name, compute_name},
                    .target_nodes = cores,
                },
            },
    };
    kda_factory_detail::bind_actual_end(spec, run_args, in.actual_end, writer);
    spec.kernels = {std::move(reader), std::move(writer), std::move(compute)};

    run_args.kernel_run_args.reserve(3);
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.kernel_run_args.push_back(std::move(compute_run_args));

    auto workload = kda_factory_detail::chronology_workload(
        ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)},
        tensor_coords,
        device,
        attrs.sequence_parallel_axis,
        attrs.sequence,
        reader_name);
    // The writer stages the exchanged rows, so it needs the same chronology as the reader.
    for (auto& program : workload.programs) {
        auto& reader_args = program.spec.kernels[0].compile_time_args;
        auto& writer_args = program.spec.kernels[1].compile_time_args;
        for (const char* name : {"sp_rank", "sp_size", "local_rows"}) {
            writer_args.insert({name, reader_args.find(name)->second});
        }
    }
    if (exchange) {
        add_history_exchange(workload, attrs, in, dist.cores, exchange_core, Ct);
    }
    return workload;
}

}  // namespace ttnn::experimental::prim
