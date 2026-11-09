// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// agmm_tuned: the TP all-gather of in0 runs inside the matmul's program, beside it.
//
// The matmul is the 2D multicast factory's (MatmulMultiCoreReuseMcast2DProgramFactory) on its program config's grid,
// over the gathered in0. Two of its kernels change, by define: the in0 sender (IN0_TP_GATHER) runs the K blocks slice
// by slice, this rank's slice first and read from its own shard, and starts another rank's slice once that slice's
// counter on it holds its rows; the in1 sender (IN1_K_SLICE_ORDER) reads the weight's K rows in the same order. So the
// matmul starts at once and waits only if a slice arrives later than the matmul reaches it.
//
// The gather runs on the worker column right of the matmul grid: per link, one sender per ETH port the peers' routes
// leave on (the two ring directions), each sending this rank's shard rows to the peers routed through its port, up to
// three tiles a packet; each row's K block ends with a packet fused with an increment of the destination's slice
// counter on the in0 sender of the row.

#include "agmm_tuned_device_operation.hpp"

#include <algorithm>
#include <vector>

#include <internal/fabric.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/scratchpad_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/semaphore_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/tuned/kernels/agmm_tuned_table.hpp"
#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"

namespace ttnn::experimental::prim {

namespace {

using namespace tt::tt_metal::experimental;

const KernelSpecName in0_sender_kernel{"in0_sender"};
const KernelSpecName in1_sender_kernel{"in1_sender_writer"};
const KernelSpecName gather_kernel{"gather_sender"};
const TensorParamName in0_local_tensor{"in0_local"};
const TensorParamName gather_table_tensor{"gather_table"};
// The table row the in0 and in1 senders read (one each: they share the grid's corner core).
const ScratchpadSpecName gather_row_in0_scratchpad{"gather_row_in0"};
const ScratchpadSpecName gather_row_in1_scratchpad{"gather_row_in1"};
const SemaphoreSpecName teardown_semaphore{"gather_teardown"};
const SemaphoreSpecName buffer_index_semaphore{"gather_buffer_index"};
const ScratchpadSpecName stage_scratchpad{"gather_stage"};

constexpr uint32_t max_dests = 2;

// One sender of a rank's gather: a link, and the peers whose route over that link leaves on one ETH port, nearest
// first.
struct Sender {
    uint32_t link = 0;
    CoreCoord port;
    std::vector<ttnn::MeshCoordinate> dests;
};

uint32_t ring_distance(uint32_t from, uint32_t to, uint32_t ranks) {
    const uint32_t ahead = (to + ranks - from) % ranks;
    return std::min(ahead, ranks - ahead);
}

// The senders of the rank at `coordinate`, link by link; sender k runs on row k of the gather column.
std::vector<Sender> senders_of(
    const ttnn::MeshDevice& mesh, const ttnn::MeshCoordinate& coordinate, uint32_t axis, uint32_t links) {
    const uint32_t ranks = mesh.shape()[axis];
    const auto source = mesh.get_fabric_node_id(coordinate);
    std::vector<Sender> senders;
    for (uint32_t link = 0; link < links; ++link) {
        const size_t first = senders.size();
        for (uint32_t ahead = 1; ahead < ranks; ++ahead) {
            auto peer = coordinate;
            peer[axis] = (coordinate[axis] + ahead) % ranks;
            const auto port = tt::tt_fabric::get_forwarding_eth_core(mesh, source, mesh.get_fabric_node_id(peer), link);
            auto it =
                std::find_if(senders.begin() + first, senders.end(), [&](const Sender& s) { return s.port == port; });
            if (it == senders.end()) {
                senders.push_back(Sender{.link = link, .port = port});
                it = senders.end() - 1;
            }
            it->dests.push_back(peer);
        }
        for (auto it = senders.begin() + first; it != senders.end(); ++it) {
            std::stable_sort(it->dests.begin(), it->dests.end(), [&](const auto& a, const auto& b) {
                return ring_distance(coordinate[axis], a[axis], ranks) <
                       ring_distance(coordinate[axis], b[axis], ranks);
            });
            TT_FATAL(
                it->dests.size() <= max_dests,
                "agmm_tuned: a sender reaches {} ranks, at most {} supported",
                it->dests.size(),
                max_dests);
        }
    }
    return senders;
}

DataMovementHardwareConfig gather_datamovement_config() {
    auto config = ttnn::create_writer_datamovement_config();
    config.config_1xx->noc = tt::tt_metal::NOC::NOC_1;
    return config;
}

}  // namespace

std::vector<uint32_t> table_rows_of(
    const ttnn::MeshDevice& mesh,
    const ttnn::MeshCoordinate& coordinate,
    uint32_t axis,
    uint32_t num_links,
    uint32_t gather_column,
    const std::vector<uint32_t>& counter_addresses) {
    namespace table = agmm_tuned_table;
    const uint32_t ranks = mesh.shape()[axis];
    const uint32_t rank = coordinate[axis];
    const auto senders = senders_of(mesh, coordinate, axis, num_links);
    std::vector<uint32_t> rows((1 + senders.size()) * table::kTableRowWords, 0);
    rows[table::kRank] = rank;
    const auto source = mesh.get_fabric_node_id(coordinate);
    for (uint32_t k = 0; k < senders.size(); ++k) {
        const auto& sender = senders[k];
        uint32_t* row = rows.data() + (1 + k) * table::kTableRowWords;
        row[table::kNumDests] = static_cast<uint32_t>(sender.dests.size());
        for (uint32_t d = 0; d < sender.dests.size(); ++d) {
            const auto& dest = sender.dests[d];
            const auto node = mesh.get_fabric_node_id(dest);
            // On the destination this rank is `behind` ranks back: its counter for this rank's slice.
            const uint32_t behind = (dest[axis] + ranks - rank) % ranks;
            // The destination's sender that writes into this rank over the same link: it waits for this rank.
            const auto dest_senders = senders_of(mesh, dest, axis, num_links);
            uint32_t partner = dest_senders.size();
            for (uint32_t j = 0; j < dest_senders.size(); ++j) {
                const auto& candidate = dest_senders[j];
                if (candidate.link == sender.link &&
                    std::find(candidate.dests.begin(), candidate.dests.end(), coordinate) != candidate.dests.end()) {
                    partner = j;
                }
            }
            TT_FATAL(partner < dest_senders.size(), "agmm_tuned: no sender writes back on the link");
            const auto partner_xy = mesh.worker_core_from_logical_core(CoreCoord{gather_column, partner});
            row[table::dest_word(d, table::kDestChip)] = node.chip_id;
            row[table::dest_word(d, table::kDestMesh)] = *node.mesh_id;
            row[table::dest_word(d, table::kDestCounter)] = counter_addresses.at(behind - 1);
            row[table::dest_word(d, table::kDestPartner)] = partner_xy.x | (partner_xy.y << 16);
            row[table::dest_word(d, table::kDestHops)] = ring_distance(rank, dest[axis], ranks);
        }
        if (!sender.dests.empty()) {
            const auto connection = tt::tt_metal::internal::compute_fabric_connection_rt_args(
                source, {mesh.get_fabric_node_id(sender.dests.front())}, {sender.link}, {0}, {0});
            TT_FATAL(
                connection.size() <= table::kMaxConnectionWords,
                "agmm_tuned: {} fabric connection words exceed the table row",
                connection.size());
            row[table::kNumConnectionWords] = static_cast<uint32_t>(connection.size());
            std::copy(connection.begin(), connection.end(), row + table::kConnectionWords);
        }
    }
    return rows;
}

ttnn::device_operation::MeshWorkloadArtifacts AgmmTunedProgramFactory::create_mesh_workload_artifacts(
    const AgmmTunedParams& attrs,
    const AgmmTunedInputs& in,
    std::vector<Tensor>& outputs,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    namespace table = agmm_tuned_table;
    const auto& mesh = in.input.mesh_tensor().device();
    const uint32_t axis = attrs.cluster_axis;
    const uint32_t ranks = mesh.shape()[axis];
    const auto& config = attrs.program_config;
    const uint32_t tile_bytes = in.input.tensor_spec().compute_page_size_bytes();
    const uint32_t slice_tiles = in.input.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    const uint32_t full_row_tiles = in.gathered.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    const uint32_t row_blocks = config.compute_with_storage_grid_size.y;
    const uint32_t gather_column = config.compute_with_storage_grid_size.x;
    const uint32_t row_bytes = 4 * table::kTableRowWords;

    // One program serves every device: what differs between them (the rank, the senders' destinations and fabric
    // connections) is in the per-device gather table. Its senders must therefore sit alike on every device: as many,
    // on the same links, with connection arguments of one length (the program's placeholders).
    const auto first = *tensor_coords.coords().begin();
    const auto layout = senders_of(mesh, first, axis, attrs.num_links);
    uint32_t connection_words = 0;
    for (const auto& coordinate : tensor_coords.coords()) {
        const auto senders = senders_of(mesh, coordinate, axis, attrs.num_links);
        TT_FATAL(senders.size() == layout.size(), "agmm_tuned: the devices' gather senders differ");
        const auto rows =
            table_rows_of(mesh, coordinate, axis, attrs.num_links, gather_column, attrs.counter_addresses);
        for (uint32_t k = 0; k < senders.size(); ++k) {
            TT_FATAL(
                senders[k].link == layout[k].link, "agmm_tuned: the devices' gather senders differ");
            const uint32_t words = rows[(1 + k) * table::kTableRowWords + table::kNumConnectionWords];
            TT_FATAL(
                connection_words == 0 || words == connection_words,
                "agmm_tuned: the devices' fabric connection arguments differ in length");
            connection_words = words;
        }
    }
    TT_FATAL(
        layout.size() <= mesh.compute_with_storage_grid_size().y,
        "agmm_tuned: {} gather senders do not fit in a worker column",
        layout.size());
    TT_FATAL(
        in.gather_table.padded_shape()[-2] >= 1 + layout.size() &&
            in.gather_table.padded_shape()[-1] == table::kTableRowWords,
        "agmm_tuned: gather_table must be [1, 1, >= {}, {}] (agmm_tuned_gather_table_rows)",
        1 + layout.size(),
        table::kTableRowWords);

    // The matmul over the gathered in0, as ttnn.linear builds it.
    ttnn::prim::MatmulParams matmul{
        .program_config = config,
        .bcast_batch = true,
        .output_mem_config = attrs.output_mem_config,
        .output_dtype = attrs.output_dtype,
        .compute_kernel_config = attrs.compute_kernel_config,
    };
    ttnn::prim::MatmulInputs matmul_inputs{
        .input_tensors = {in.gathered, in.weight},
        .optional_input_tensors = {std::nullopt},
        .optional_output_tensors = {}};
    auto artifacts =
        ttnn::prim::MatmulMultiCoreReuseMcast2DProgramFactory::create_program_artifacts(matmul, matmul_inputs, outputs);
    auto& spec = artifacts.spec;
    auto& run = artifacts.run_params;

    // The matmul's in0 and in1 senders read the rank from the table and take the slices in its order.
    for (auto& kernel : spec.kernels) {
        if (kernel.unique_id != in0_sender_kernel && kernel.unique_id != in1_sender_kernel) {
            continue;
        }
        kernel.compile_time_args.insert({"gather_slices", ranks});
        kernel.compile_time_args.insert({"gather_blocks_per_slice", slice_tiles / config.in0_block_w});
        kernel.tensor_bindings.push_back(KernelSpec::TensorBinding{gather_table_tensor, "gather_table"});
        kernel.scratchpad_bindings.push_back(
            {kernel.unique_id == in0_sender_kernel ? gather_row_in0_scratchpad : gather_row_in1_scratchpad,
             "gather_row"});
        if (kernel.unique_id == in0_sender_kernel) {
            kernel.compiler_options.defines.insert({"IN0_TP_GATHER", "1"});
            kernel.compile_time_args.insert({"gather_slice_tiles", slice_tiles});
            // The gather counts, K block by K block, each tile row of a block landed.
            kernel.compile_time_args.insert({"gather_block_arrivals", config.per_core_M});
            kernel.tensor_bindings.push_back(KernelSpec::TensorBinding{in0_local_tensor, "in0_local"});
            kernel.runtime_arg_schema.common_runtime_array_args.push_back({"gather_counters", ranks - 1});
        } else {
            kernel.compiler_options.defines.insert({"IN1_K_SLICE_ORDER", "1"});
        }
    }
    for (auto& kernel_run : run.kernel_run_args) {
        if (kernel_run.kernel == in0_sender_kernel) {
            // The counters are global semaphores: one address on every device.
            kernel_run.common_runtime_array_arg_values.insert({"gather_counters", attrs.counter_addresses});
        }
    }
    spec.scratchpads.push_back({.unique_id = gather_row_in0_scratchpad, .size_per_node = row_bytes});
    spec.scratchpads.push_back({.unique_id = gather_row_in1_scratchpad, .size_per_node = row_bytes});
    // The matmul was built over this op's inputs: bind them (one parameter per tensor; the gather shares them).
    for (const auto& [name, tensor] :
         {std::pair{TensorParamName{"in0"}, &in.gathered}, std::pair{TensorParamName{"in1"}, &in.weight}}) {
        run.tensor_args.erase(name);
        run.tensor_args.emplace(name, std::cref(tensor->mesh_tensor()));
    }
    for (const auto& [name, tensor] :
         {std::pair{in0_local_tensor, &in.input}, std::pair{gather_table_tensor, &in.gather_table}}) {
        spec.tensor_parameters.push_back({.unique_id = name, .spec = tensor->mesh_tensor().tensor_spec()});
        run.tensor_args.emplace(name, std::cref(tensor->mesh_tensor()));
    }

    // The gather's senders, in the column right of the matmul grid.
    std::vector<uint32_t> in0_sender_xy;
    for (uint32_t row = 0; row < row_blocks; ++row) {
        const auto xy = mesh.worker_core_from_logical_core(CoreCoord{0, row});
        in0_sender_xy.push_back(xy.x | (xy.y << 16));
    }
    const std::vector<std::string> named_args = {
        "first_row_block", "row_block_step", "ready_semaphore", "sender_index"};
    KernelSpec gather{
        .unique_id = gather_kernel,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/tuned/kernels/"
            "agmm_tuned_gather_sender.cpp",
        .semaphore_bindings =
            {
                KernelSpec::SemaphoreBinding{teardown_semaphore, "teardown"},
                KernelSpec::SemaphoreBinding{buffer_index_semaphore, "buffer_index"},
            },
        .scratchpad_bindings = {{stage_scratchpad, "stage"}},
        .tensor_bindings =
            {
                KernelSpec::TensorBinding{in0_local_tensor, "input"},
                KernelSpec::TensorBinding{TensorParamName{"in0"}, "gathered"},
                KernelSpec::TensorBinding{gather_table_tensor, "gather_table"},
            },
        .compile_time_args =
            {{"tile_bytes", tile_bytes},
             {"slice_tiles", slice_tiles},
             {"full_row_tiles", full_row_tiles},
             {"row_block_tiles", config.per_core_M},
             {"num_row_blocks", row_blocks},
             {"block_tiles", config.in0_block_w},
             {"num_named_runtime_args", static_cast<uint32_t>(named_args.size())}},
        .runtime_arg_schema = {.runtime_arg_names = {named_args.begin(), named_args.end()}},
        .hw_config = gather_datamovement_config(),
    };
    for (const auto& [name, value] : tt::tt_metal::internal::get_fabric_kernel_defines()) {
        gather.compiler_options.defines.insert({name, value});
    }
    // Varargs: the fabric connection's placeholders (the kernel writes the device's own from its table), then the
    // row blocks' in0 senders (the same cores on every device).
    std::vector<uint32_t> varargs(connection_words, 0);
    varargs.insert(varargs.end(), in0_sender_xy.begin(), in0_sender_xy.end());
    gather.advanced_options.num_runtime_varargs = static_cast<uint32_t>(varargs.size());
    ProgramRunArgs::KernelRunArgs gather_run{.kernel = gather_kernel};
    std::vector<CoreRange> sender_ranges;
    for (uint32_t k = 0; k < layout.size(); ++k) {
        const CoreCoord core{gather_column, k};
        sender_ranges.emplace_back(core);
        for (const auto& [name, value] : std::vector<std::pair<std::string, uint32_t>>{
                 {"first_row_block", layout[k].link},
                 {"row_block_step", attrs.num_links},
                 {"ready_semaphore", attrs.ready_semaphore_address},
                 {"sender_index", k}}) {
            gather_run.runtime_arg_values[name][core] = value;
        }
        gather_run.advanced_options.runtime_varargs[NodeCoord(core)] = varargs;
    }
    const CoreRangeSet sender_nodes(sender_ranges);
    spec.kernels.push_back(std::move(gather));
    spec.semaphores.push_back(SemaphoreSpec{.unique_id = teardown_semaphore, .target_nodes = sender_nodes});
    spec.semaphores.push_back(SemaphoreSpec{.unique_id = buffer_index_semaphore, .target_nodes = sender_nodes});
    // The device's two table rows, then stage_units (4) of one row's K block each.
    spec.scratchpads.push_back(
        {.unique_id = stage_scratchpad, .size_per_node = 2 * row_bytes + 4 * config.in0_block_w * tile_bytes});
    spec.work_units.push_back(WorkUnitSpec{.name = "gather", .kernels = {gather_kernel}, .target_nodes = sender_nodes});
    run.kernel_run_args.push_back(std::move(gather_run));

    ttnn::device_operation::MeshWorkloadArtifacts workload;
    for (const auto& range : tensor_coords.ranges()) {
        workload.programs.push_back({.range = range, .spec = spec, .run_params = run});
    }
    return workload;
}

std::vector<std::vector<uint32_t>> agmm_tuned_gather_table_rows(
    const ttnn::MeshDevice& mesh,
    uint32_t cluster_axis,
    uint32_t num_links,
    uint32_t gather_column,
    const std::vector<uint32_t>& counter_addresses) {
    std::vector<std::vector<uint32_t>> tables;
    const auto shape = mesh.shape();
    for (const auto& coordinate : ttnn::MeshCoordinateRange(shape)) {
        tables.push_back(table_rows_of(mesh, coordinate, cluster_axis, num_links, gather_column, counter_addresses));
    }
    return tables;
}

}  // namespace ttnn::experimental::prim
