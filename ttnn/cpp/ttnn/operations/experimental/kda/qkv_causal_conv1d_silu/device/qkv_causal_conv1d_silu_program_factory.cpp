// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/qkv_causal_conv1d_silu_program_factory.hpp"

#include "ttnn/operations/experimental/kda/factory/chronology_binding.hpp"
#include <algorithm>
#include <vector>

#include <internal/fabric.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/semaphore_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"
#include "ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/hostdevcommon/qkv_causal_conv1d_silu_layout.hpp"

// The shared layout header spells out the tile geometry so it stays kernel-includable; keep it in step with the host.
static_assert(qkv_conv::tile_height == tt::constants::TILE_HEIGHT);
static_assert(qkv_conv::tile_width == tt::constants::TILE_WIDTH);
static_assert(qkv_conv::face_height == tt::constants::FACE_HEIGHT);
static_assert(qkv_conv::face_width == tt::constants::FACE_WIDTH);

namespace ttnn::experimental::prim {

namespace {

using namespace tt::tt_metal::experimental;
using qkv_conv::no_fabric_connection;

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
    // The successor is the forward neighbor, except for a line's last rank, which reaches rank 0 backward.
    const bool successor_forward = ring || rank + 1 < sp_size;
    if (successor_forward) {
        route.forward_connection = route.neighbors.size();
        route.neighbors.push_back(successor);
    }
    if (ring || rank > 0) {
        route.backward_connection = route.neighbors.size();
        route.neighbors.push_back(at_rank((rank + sp_size - 1) % sp_size));
    }
    route.forward_range = ring ? sp_size / 2 : sp_size - 1 - rank;
    route.backward_range = ring ? sp_size - 1 - sp_size / 2 : rank;
    route.successor_connection = successor_forward ? route.forward_connection : route.backward_connection;
    route.successor_hops = successor_forward ? 1 : sp_size - 1;
    return route;
}

// The exchange core's packets go to the ETH cores above the first Tensix row, so its kernel uses the NoC that routes
// toward lower rows (NOC_1) rather than inheriting whichever the writer placement picks.
tt::tt_metal::experimental::DataMovementHardwareConfig exchange_datamovement_config() {
    auto config = ttnn::create_writer_datamovement_config();
    config.config_1xx->noc = tt::tt_metal::NOC::NOC_1;
    return config;
}

// Add the fabric history exchange to every rank's program: one kernel on exchange_core that sends the rows the
// convolution cores staged, releases them once the predecessor's rows have landed, and distributes the carry.
void add_history_exchange(
    ttnn::device_operation::MeshWorkloadArtifacts& workload,
    const QkvCausalConv1dSiluParams& attrs,
    const QkvCausalConv1dSiluInputs& in,
    const TensorParamName& exchange_buffer,
    const Tensor& final_history,
    const std::vector<CoreCoord>& convolution_cores,
    const CoreCoord& exchange_core,
    uint32_t channel_tiles) {
    const auto& device = in.input.mesh_tensor().device();
    const uint32_t axis = attrs.sequence_parallel_axis;
    const uint32_t sp_size = device.shape()[axis];
    // A 2D multicast is routed by the connection's ETH direction, so every hop must leave in the same direction.
    // 1D fabric follows its routing plane, which may turn (the QuietBox ring).
    TT_FATAL(
        sp_size == 1 || !tt::tt_fabric::is_2d_fabric_config(tt::tt_fabric::GetFabricConfig()) ||
            ttnn::ccl::is_axis_straight(device, axis),
        "qkv_causal_conv1d_silu: the 2D fabric history exchange needs a straight SP axis");
    const uint32_t row_bytes = channel_tiles * qkv_conv::tile_row_bytes;
    // A single rank sends nothing; any aligned value satisfies the kernel's static_assert.
    const uint32_t max_payload = sp_size > 1
                                     ? static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes())
                                     : qkv_conv::read_alignment;

    std::vector<uint32_t> convolution_core_xy;
    convolution_core_xy.reserve(2 * convolution_cores.size());
    for (const auto& core : convolution_cores) {
        const auto xy = device.worker_core_from_logical_core(core);
        convolution_core_xy.push_back(xy.x);
        convolution_core_xy.push_back(xy.y);
    }

    const TensorParamName final_history_tensor{"final_history"};
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
            .semaphore_bindings =
                {
                    SemaphoreBinding{staged_semaphore, "staged"},
                    SemaphoreBinding{predecessor_ready_semaphore, "predecessor_ready"},
                    SemaphoreBinding{teardown[0], "teardown0"},
                    SemaphoreBinding{buffer_index[0], "buffer_index0"},
                    SemaphoreBinding{teardown[1], "teardown1"},
                    SemaphoreBinding{buffer_index[1], "buffer_index1"},
                },
            .scratchpad_bindings = {{exchange_scratch, "exchange_scratch"}},
            .tensor_bindings =
                {
                    TensorBinding{TensorParamName{"history"}, "history"},
                    TensorBinding{exchange_buffer, "exchange_buffer"},
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
            .hw_config = exchange_datamovement_config(),
        };
        if (sp_size > 1) {
            // The fabric API-type defines; lowering adds only the control plane's routing and header-size ones.
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
        spec.tensor_parameters.push_back(
            {.unique_id = final_history_tensor, .spec = final_history.mesh_tensor().tensor_spec()});
        program.run_params.tensor_args.emplace(final_history_tensor, final_history.mesh_tensor());
        for (uint32_t i = 0; i < 2; ++i) {
            spec.semaphores.push_back(SemaphoreSpec{.unique_id = teardown[i], .target_nodes = exchange_nodes});
            spec.semaphores.push_back(SemaphoreSpec{.unique_id = buffer_index[i], .target_nodes = exchange_nodes});
        }
        // An aligned slot for the chronology scalars, alignment slack, then the outgoing and carry rows.
        spec.scratchpads.push_back(
            {.unique_id = exchange_scratch,
             .size_per_node = 2 * qkv_conv::read_alignment + 2 * qkv_conv::history_rows * row_bytes});
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
    const auto& history = in.history.mesh_tensor();
    const auto& tap0 = in.tap0.mesh_tensor();
    const auto& tap1 = in.tap1.mesh_tensor();
    const auto& tap2 = in.tap2.mesh_tensor();
    const auto& tap3 = in.tap3.mesh_tensor();
    const auto& q = outputs[0].mesh_tensor();
    const auto& k = outputs[1].mesh_tensor();
    const auto& v = outputs[2].mesh_tensor();
    const auto& device = input.device();
    const bool exchange = in.exchange_buffer.has_value();
    const auto& predecessor = exchange ? in.exchange_buffer->mesh_tensor() : in.predecessor_carry->mesh_tensor();

    const uint32_t Mt = attrs.sequence / tt::constants::TILE_HEIGHT;
    const uint32_t Qt = attrs.q_width / tt::constants::TILE_WIDTH;
    const uint32_t Kt = attrs.k_width / tt::constants::TILE_WIDTH;
    const uint32_t Vt = attrs.v_width / tt::constants::TILE_WIDTH;
    const uint32_t Ct = Qt + Kt + Vt;
    const uint32_t block_ct = attrs.channel_chunk_size / tt::constants::TILE_WIDTH;
    const uint32_t num_blocks = Ct / block_ct;
    const uint32_t dest_limit = qkv_conv::dest_tile_limit(attrs.compute_kernel_config.fp32_dest_acc_en);
    const uint32_t group = qkv_conv::dest_group(block_ct, dest_limit);
    const uint32_t input_wt = input.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    const uint32_t input_ct_offset = attrs.input_column_offset / tt::constants::TILE_WIDTH;
    const auto grid = device.compute_with_storage_grid_size();
    const CoreCoord exchange_core = qkv_causal_conv1d_silu_exchange_core(grid);
    auto dist = qkv_causal_conv1d_silu_distribution(grid, Mt * num_blocks, exchange);
    const auto& cores = dist.core_set;
    const uint32_t max_staged_ct =
        exchange ? qkv_causal_conv1d_silu_staged_ct(Ct, static_cast<uint32_t>(dist.cores.size())) : 0;
    const auto exchange_core_xy = device.worker_core_from_logical_core(exchange_core);

    const tt::tt_metal::experimental::KernelSpecName reader_kernel_name{"reader"};
    const tt::tt_metal::experimental::KernelSpecName writer_kernel_name{"writer"};
    const tt::tt_metal::experimental::KernelSpecName compute_kernel_name{"compute"};

    const tt::tt_metal::experimental::ScratchpadSpecName act_window_name{"act_window"};
    const tt::tt_metal::experimental::ScratchpadSpecName staging_name{"staging"};
    const tt::tt_metal::experimental::DFBSpecName act_tile_dfb_name{"act_tile"};
    const tt::tt_metal::experimental::DFBSpecName taps_dfb_name{"taps"};
    const tt::tt_metal::experimental::DFBSpecName eye_dfb_name{"eye"};
    const tt::tt_metal::experimental::DFBSpecName weights_dfb_name{"weights"};
    const tt::tt_metal::experimental::DFBSpecName output_dfb_name{"output"};

    const tt::tt_metal::experimental::TensorParamName input_tensor_name{"input"};
    const tt::tt_metal::experimental::TensorParamName history_tensor_name{"history"};
    const tt::tt_metal::experimental::TensorParamName tap0_tensor_name{"tap0"};
    const tt::tt_metal::experimental::TensorParamName tap1_tensor_name{"tap1"};
    const tt::tt_metal::experimental::TensorParamName tap2_tensor_name{"tap2"};
    const tt::tt_metal::experimental::TensorParamName tap3_tensor_name{"tap3"};
    const tt::tt_metal::experimental::TensorParamName q_tensor_name{"q"};
    const tt::tt_metal::experimental::TensorParamName k_tensor_name{"k"};
    const tt::tt_metal::experimental::TensorParamName v_tensor_name{"v"};

    const auto input_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const uint32_t tile_size = tt::tile_size(input_data_format);
    auto make_dfb = [input_data_format, tile_size](
                        const tt::tt_metal::experimental::DFBSpecName& name, uint32_t tiles) {
        return tt::tt_metal::experimental::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = tile_size,
            .num_entries = tiles,
            .data_format_metadata = input_data_format,
        };
    };

    // The reader stages each channel tile's face rows, then builds each shifted tap tile from them with local
    // copies instead of rereading DRAM once per tap. The staging layout assumes BF16 tiles;
    // validate_on_program_cache_miss restricts input and history to BF16.
    TT_FATAL(
        input.dtype() == tt::tt_metal::DataType::BFLOAT16 && history.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "qkv_causal_conv1d_silu: activation window staging requires BF16 input and history");
    // validate_on_program_cache_miss checks qkv_causal_conv1d_silu_l1_bytes against the free L1 per core.
    const uint32_t window_bytes = qkv_conv::ReaderScratch{block_ct}.bytes();

    const auto dfb_tiles = qkv_causal_conv1d_silu_dfb_tiles(block_ct, dest_limit);
    tt::tt_metal::experimental::Group<tt::tt_metal::experimental::DataflowBufferSpec> dfbs = {
        make_dfb(act_tile_dfb_name, dfb_tiles.act_tile),
        make_dfb(taps_dfb_name, dfb_tiles.taps),
        make_dfb(eye_dfb_name, dfb_tiles.eye),
        make_dfb(weights_dfb_name, dfb_tiles.weights),
        make_dfb(output_dfb_name, dfb_tiles.output),
    };

    tt::tt_metal::experimental::KernelSpec reader{
        .unique_id = reader_kernel_name,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/kernels/dataflow/"
            "reader_qkv_causal_conv1d_silu.cpp",
        .dfb_bindings =
            {
                tt::tt_metal::experimental::DFBBinding{
                    act_tile_dfb_name, "act_tile", tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
                tt::tt_metal::experimental::DFBBinding{
                    taps_dfb_name, "taps", tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
                tt::tt_metal::experimental::DFBBinding{
                    eye_dfb_name, "eye", tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
            },
        .semaphore_bindings = {tt::tt_metal::experimental::SemaphoreBinding{
            predecessor_ready_semaphore, "predecessor_ready"}},
        .scratchpad_bindings = {{act_window_name, "act_window"}},
        .tensor_bindings =
            {
                tt::tt_metal::experimental::TensorBinding{input_tensor_name, "input"},
                tt::tt_metal::experimental::TensorBinding{history_tensor_name, "history"},
                tt::tt_metal::experimental::TensorBinding{tap0_tensor_name, "tap0"},
                tt::tt_metal::experimental::TensorBinding{tap1_tensor_name, "tap1"},
                tt::tt_metal::experimental::TensorBinding{tap2_tensor_name, "tap2"},
                tt::tt_metal::experimental::TensorBinding{tap3_tensor_name, "tap3"},
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

    const tt::tt_metal::experimental::TensorParamName actual_start_name{"actual_start"};
    const tt::tt_metal::experimental::TensorParamName predecessor_name{"predecessor_carry"};
    reader.tensor_bindings.push_back({actual_start_name, "actual_start"});
    reader.tensor_bindings.push_back({predecessor_name, "predecessor_carry"});

    tt::tt_metal::experimental::KernelSpec writer{
        .unique_id = writer_kernel_name,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/kernels/dataflow/"
            "writer_qkv_causal_conv1d_silu.cpp",
        .dfb_bindings = {tt::tt_metal::experimental::DFBBinding{
            output_dfb_name, "output", tt::tt_metal::experimental::DFBEndpointType::CONSUMER}},
        .semaphore_bindings = {tt::tt_metal::experimental::SemaphoreBinding{staged_semaphore, "staged"}},
        .scratchpad_bindings = {{staging_name, "staging"}},
        .tensor_bindings =
            {
                tt::tt_metal::experimental::TensorBinding{q_tensor_name, "q"},
                tt::tt_metal::experimental::TensorBinding{k_tensor_name, "k"},
                tt::tt_metal::experimental::TensorBinding{v_tensor_name, "v"},
                tt::tt_metal::experimental::TensorBinding{input_tensor_name, "input"},
                tt::tt_metal::experimental::TensorBinding{actual_start_name, "actual_start"},
                tt::tt_metal::experimental::TensorBinding{predecessor_name, "exchange_buffer"},
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

    tt::tt_metal::experimental::KernelSpec compute{
        .unique_id = compute_kernel_name,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/kernels/compute/"
            "qkv_causal_conv1d_silu.cpp",
        .compiler_options = {.opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                tt::tt_metal::experimental::DFBBinding{
                    act_tile_dfb_name, "act_tile", tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
                tt::tt_metal::experimental::DFBBinding{
                    taps_dfb_name, "taps", tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
                tt::tt_metal::experimental::DFBBinding{
                    eye_dfb_name, "eye", tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
                tt::tt_metal::experimental::DFBBinding{
                    weights_dfb_name, "weights", tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
                tt::tt_metal::experimental::DFBBinding{
                    weights_dfb_name, "weights", tt::tt_metal::experimental::DFBEndpointType::CONSUMER},
                tt::tt_metal::experimental::DFBBinding{
                    output_dfb_name, "output", tt::tt_metal::experimental::DFBEndpointType::PRODUCER},
            },
        .compile_time_args =
            {{"block_ct", block_ct},
             {"Mt", Mt},
             {"group", group},
             {"dest_limit", dest_limit},
             {"defer_first_rows", uint32_t(exchange)}},
        .runtime_arg_schema = {.runtime_arg_names = {"wi_start", "wi_count"}},
        .hw_config = ttnn::to_compute_hardware_config(attrs.compute_kernel_config),
    };

    tt::tt_metal::experimental::KernelRunArgs reader_run_args{.kernel = reader_kernel_name};
    tt::tt_metal::experimental::KernelRunArgs writer_run_args{.kernel = writer_kernel_name};
    tt::tt_metal::experimental::KernelRunArgs compute_run_args{.kernel = compute_kernel_name};
    for (uint32_t i = 0; i < dist.cores.size(); ++i) {
        const auto& core = dist.cores[i];
        tt::tt_metal::experimental::AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values, core, {{"wi_start", dist.wi_start[i]}, {"wi_count", dist.wi_count[i]}});
        tt::tt_metal::experimental::AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values, core, {{"wi_start", dist.wi_start[i]}, {"wi_count", dist.wi_count[i]}});
        tt::tt_metal::experimental::AddRuntimeArgsForNode(
            compute_run_args.runtime_arg_values,
            core,
            {{"wi_start", dist.wi_start[i]}, {"wi_count", dist.wi_count[i]}});
        const uint32_t staged_begin = std::min(Ct, i * max_staged_ct);
        const uint32_t staged_end = std::min(Ct, staged_begin + max_staged_ct);
        tt::tt_metal::experimental::AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"staged_ct_start", staged_begin}, {"staged_ct_count", staged_end - staged_begin}});
    }

    tt::tt_metal::experimental::Group<tt::tt_metal::experimental::TensorParameter> tensor_parameters = {
        tt::tt_metal::experimental::TensorParameter{.unique_id = input_tensor_name, .spec = input.tensor_spec()},
        tt::tt_metal::experimental::TensorParameter{.unique_id = history_tensor_name, .spec = history.tensor_spec()},
        tt::tt_metal::experimental::TensorParameter{.unique_id = tap0_tensor_name, .spec = tap0.tensor_spec()},
        tt::tt_metal::experimental::TensorParameter{.unique_id = tap1_tensor_name, .spec = tap1.tensor_spec()},
        tt::tt_metal::experimental::TensorParameter{.unique_id = tap2_tensor_name, .spec = tap2.tensor_spec()},
        tt::tt_metal::experimental::TensorParameter{.unique_id = tap3_tensor_name, .spec = tap3.tensor_spec()},
        tt::tt_metal::experimental::TensorParameter{.unique_id = q_tensor_name, .spec = q.tensor_spec()},
        tt::tt_metal::experimental::TensorParameter{.unique_id = k_tensor_name, .spec = k.tensor_spec()},
        tt::tt_metal::experimental::TensorParameter{.unique_id = v_tensor_name, .spec = v.tensor_spec()},
    };

    tensor_parameters.push_back({.unique_id = actual_start_name, .spec = in.actual_start.mesh_tensor().tensor_spec()});
    tensor_parameters.push_back({.unique_id = predecessor_name, .spec = predecessor.tensor_spec()});

    // The exchange core releases every convolution core on predecessor_ready and waits for their staged increments, so
    // both semaphores live on all of them.
    CoreRangeSet semaphore_cores = cores;
    if (exchange) {
        semaphore_cores = semaphore_cores.merge(CoreRangeSet(CoreRange(exchange_core)));
    }

    tt::tt_metal::experimental::ProgramSpec spec{
        .name = "qkv_causal_conv1d_silu",
        .dataflow_buffers = std::move(dfbs),
        .semaphores =
            {
                tt::tt_metal::experimental::SemaphoreSpec{
                    .unique_id = predecessor_ready_semaphore, .target_nodes = semaphore_cores},
                tt::tt_metal::experimental::SemaphoreSpec{
                    .unique_id = staged_semaphore, .target_nodes = semaphore_cores},
            },
        .scratchpads =
            {{.unique_id = act_window_name, .size_per_node = window_bytes},
             {.unique_id = staging_name, .size_per_node = qkv_conv::StagingScratch{max_staged_ct}.bytes()}},
        .tensor_parameters = std::move(tensor_parameters),
        .work_units =
            {
                tt::tt_metal::experimental::WorkUnitSpec{
                    .name = "main",
                    .kernels = {reader_kernel_name, writer_kernel_name, compute_kernel_name},
                    .target_nodes = cores,
                },
            },
    };

    tt::tt_metal::experimental::ProgramRunArgs run_args;
    run_args.kernel_run_args.reserve(3);
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.kernel_run_args.push_back(std::move(compute_run_args));
    run_args.tensor_args = {
        {input_tensor_name, input},
        {history_tensor_name, history},
        {tap0_tensor_name, tap0},
        {tap1_tensor_name, tap1},
        {tap2_tensor_name, tap2},
        {tap3_tensor_name, tap3},
        {q_tensor_name, q},
        {k_tensor_name, k},
        {v_tensor_name, v},
    };

    run_args.tensor_args.emplace(actual_start_name, in.actual_start.mesh_tensor());
    run_args.tensor_args.emplace(predecessor_name, predecessor);
    kda_factory_detail::bind_actual_end(spec, run_args, in.actual_end, writer);
    spec.kernels = {std::move(reader), std::move(writer), std::move(compute)};

    auto workload = kda_factory_detail::chronology_workload(
        ttnn::device_operation::ProgramArtifacts{
            .spec = std::move(spec),
            .run_params = std::move(run_args),
        },
        tensor_coords,
        device,
        attrs.sequence_parallel_axis,
        attrs.sequence,
        // The writer stages the exchanged rows, so it needs the same chronology as the reader.
        {reader_kernel_name, writer_kernel_name});
    if (exchange) {
        add_history_exchange(workload, attrs, in, predecessor_name, outputs[3], dist.cores, exchange_core, Ct);
    }
    return workload;
}

}  // namespace ttnn::experimental::prim
