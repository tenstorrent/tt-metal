// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar-native all-gather program factory (Metal 2.0 ProgramSpec).
//
// Page model. The input is TILE and sharded over N clusters, so input page p (tile id in the full tensor)
// lives in shard s(p); shard s is rank s's slice. The output is [N, ...] height-sharded over the same
// clusters with one full-tensor shard per rank, so output page r * P + p (P = pages of the input) is tile p
// of rank r's copy. Every transfer is therefore "input page p -> output page r * P + p", and the sharding-
// aware TensorAccessors turn those page ids into NoC addresses on whichever cluster owns them. The gather
// dim never appears on the device: it is the input's sharding dim.
//
// Packets. A packet is tiles_per_packet consecutive shard-local tiles, contiguous at both ends (see
// all_gather_common.hpp), so it moves as ONE NoC transaction. Packets are owned round-robin by the K*T global
// threads of a rank, the same enumeration on every rank.
//
// DIRECT: one kernel of T (<= 6) threads on each rank's home cluster reads owner shards straight into the
// local output shard; no DFB, no credits, every read in flight until a single barrier.
// RING: the reader (T threads, PRODUCER) stages packets in a one-packet-per-entry DFB that the writer
// (T threads, CONSUMER) sends on. The (worker, thread) that sends a packet is mirrored by the (worker, thread)
// that receives it, and one ready semaphore per thread carries its credits.
//
// Transport seam. The writer's remote write + ready credit are NoC writes + Semaphore::up today; a fabric
// transport is the same two calls with packet headers once Quasar has fabric endpoints.

#include "all_gather_device_operation_types.hpp"
#include "all_gather_program_factory.hpp"

#include <algorithm>
#include <string>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/work_split.hpp>

#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/semaphore_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::prim::qsr {

using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

constexpr const char* kDirectKernel =
    "ttnn/cpp/ttnn/operations/experimental/quasar/ccl/all_gather/device/kernels/dataflow/all_gather_direct.cpp";
constexpr const char* kReaderKernel =
    "ttnn/cpp/ttnn/operations/experimental/quasar/ccl/all_gather/device/kernels/dataflow/all_gather_reader.cpp";
constexpr const char* kWriterKernel =
    "ttnn/cpp/ttnn/operations/experimental/quasar/ccl/all_gather/device/kernels/dataflow/all_gather_writer.cpp";

constexpr uint32_t kMaxRingThreads = 4;  // one ready semaphore per thread: ready0..ready3
const char* const kReadySemaphoreNames[kMaxRingThreads] = {"ready0", "ready1", "ready2", "ready3"};

// Worker clusters of each rank: the shard's home cluster first, then helpers taken row-major from the
// compute grid, skipping every home cluster.
std::vector<std::vector<CoreCoord>> assign_worker_cores(
    const std::vector<CoreCoord>& home_cores, uint32_t workers_per_rank, const CoreCoord& grid) {
    std::vector<std::vector<CoreCoord>> workers(home_cores.size());
    std::vector<CoreCoord> helpers;
    if (workers_per_rank > 1) {
        for (uint32_t y = 0; y < grid.y; y++) {
            for (uint32_t x = 0; x < grid.x; x++) {
                const CoreCoord c{x, y};
                if (std::find(home_cores.begin(), home_cores.end(), c) == home_cores.end()) {
                    helpers.push_back(c);
                }
            }
        }
        TT_FATAL(
            helpers.size() >= home_cores.size() * (workers_per_rank - 1),
            "quasar.all_gather: {} ranks x {} helper clusters do not fit beside the {} shard clusters on a {}x{} grid",
            home_cores.size(),
            workers_per_rank - 1,
            home_cores.size(),
            grid.x,
            grid.y);
    }
    size_t next_helper = 0;
    for (size_t r = 0; r < home_cores.size(); r++) {
        workers[r].push_back(home_cores[r]);
        for (uint32_t w = 1; w < workers_per_rank; w++) {
            workers[r].push_back(helpers[next_helper++]);
        }
    }
    return workers;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

ttnn::device_operation::ProgramArtifacts AllGatherProgramFactory::create_program_artifacts(
    const AllGatherOperationAttributes& args, const AllGatherTensorArgs& tensor_args, Tensor& output) {
    using namespace CMAKE_UNIQUE_NAMESPACE;

    const Tensor& input = tensor_args.input_tensor;
    const auto& shard = input.shard_spec().value();
    const bool width_sharded = input.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED;
    const uint32_t num_ranks = shard.grid.num_cores();
    const uint32_t T = args.reader_threads;
    const uint32_t K = args.num_workers_per_rank;
    const uint32_t TP = args.tiles_per_packet;
    const bool ring = args.topology == AllGatherTopology::RING;

    // Tile geometry: shard rows/cols in tiles, full tensor cols in tiles.
    const auto& padded = input.padded_shape();
    const uint32_t tensor_cols_t = padded[-1] / tt::constants::TILE_WIDTH;
    const uint32_t shard_rows_t = shard.shape[0] / tt::constants::TILE_HEIGHT;
    const uint32_t shard_cols_t = shard.shape[1] / tt::constants::TILE_WIDTH;
    const uint32_t pages_per_slice = shard_rows_t * shard_cols_t;
    const uint32_t pages_total = pages_per_slice * num_ranks;
    TT_FATAL(
        pages_total == input.buffer()->num_pages(),
        "quasar.all_gather: page model mismatch: {} shard pages x {} ranks vs {} buffer pages",
        pages_per_slice,
        num_ranks,
        input.buffer()->num_pages());

    const uint32_t page_size = input.buffer()->page_size();
    const uint32_t entry_size = tt::align(page_size, hal::get_l1_alignment());

    // Placement.
    const tt::tt_metal::distributed::MeshDevice* device = input.device();
    const CoreCoord grid = device->compute_with_storage_grid_size();
    const std::vector<CoreCoord> home_cores =
        corerange_to_cores(shard.grid, std::nullopt, shard.orientation == ShardOrientation::ROW_MAJOR);
    TT_FATAL(
        home_cores.size() == num_ranks,
        "quasar.all_gather: shard grid enumerates {} cores, expected {}",
        home_cores.size(),
        num_ranks);
    const auto workers = assign_worker_cores(home_cores, K, grid);
    std::vector<CoreRange> all_worker_ranges;
    for (const auto& rank_workers : workers) {
        for (const auto& c : rank_workers) {
            all_worker_ranges.emplace_back(c);
        }
    }
    const CoreRangeSet all_workers(all_worker_ranges);

    //
    // -------- ProgramSpec --------
    //
    const m2::DFBSpecName PKT{"all_gather_pkt"};
    const m2::KernelSpecName DIRECT{"all_gather_direct"};
    const m2::KernelSpecName READER{"all_gather_reader"};
    const m2::KernelSpecName WRITER{"all_gather_writer"};
    const m2::TensorParamName INPUT{"input"};
    const m2::TensorParamName OUTPUT{"output"};

    m2::ProgramSpec spec;
    spec.name = ring ? "quasar_all_gather_ring" : "quasar_all_gather_direct";

    spec.tensor_parameters.push_back(m2::TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()});
    spec.tensor_parameters.push_back(m2::TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()});

    const m2::KernelSpec::CompileTimeArgs ctas = {
        {"num_ranks", num_ranks},
        {"num_workers", K},
        {"tiles_per_packet", TP},
        {"chunks_per_sync", args.chunks_per_sync},
        {"pages_total", pages_total},
        {"pages_per_slice", pages_per_slice},
        {"shard_rows_t", shard_rows_t},
        {"shard_cols_t", shard_cols_t},
        {"tensor_cols_t", tensor_cols_t},
        {"width_sharded", static_cast<uint32_t>(width_sharded)},
    };
    const m2::Group<std::string> rta_names = {"rank", "worker", "next_noc_x", "next_noc_y"};
    m2::Group<m2::KernelSpecName> kernels;

    if (!ring) {
        // DIRECT: one kernel of T threads per home cluster, reading owner shards straight into the local
        // output shard. No DFB, no semaphores.
        spec.kernels.push_back(m2::KernelSpec{
            .unique_id = DIRECT,
            .source = kDirectKernel,
            .num_threads = T,
            .compiler_options = {.opt_level = m2::KernelSpec::CompilerOptions::OptLevel::O3},
            .tensor_bindings =
                {m2::TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"},
                 m2::TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
            .compile_time_args = ctas,
            .runtime_arg_schema = {.runtime_arg_names = rta_names},
            .hw_config = ttnn::create_reader_datamovement_config(/*disable_dfb_implicit_sync_for_all=*/true),
        });
        kernels = {DIRECT};
    } else {
        // Packet ring: one entry per packet (tiles_per_packet contiguous pages), ring_depth entries per thread,
        // strided through the ring by max(producers, consumers) = T.
        constexpr uint32_t ring_depth = 4;
        spec.dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = PKT,
            .entry_size = TP * entry_size,
            .num_entries = ring_depth * T,
        });

        // One ready semaphore per thread on every worker cluster; thread t of the previous rank's matching
        // worker credits ready<t> here.
        m2::Group<m2::KernelSpec::SemaphoreBinding> sems;
        for (uint32_t t = 0; t < T; t++) {
            const m2::SemaphoreSpecName name{std::string(kReadySemaphoreNames[t])};
            spec.semaphores.push_back(m2::SemaphoreSpec{.unique_id = name, .target_nodes = all_workers});
            sems.push_back({.semaphore_spec_name = name, .accessor_name = kReadySemaphoreNames[t]});
        }

        // The sem::ready<t> tokens exist only when bound, and a discarded if-constexpr branch would still
        // have to name them, so the thread count reaches the kernels as a define.
        m2::KernelSpec::CompilerOptions::Defines defines;
        defines.emplace("ALL_GATHER_RING", "1");
        defines.emplace("ALL_GATHER_THREADS", std::to_string(T));

        spec.kernels.push_back(m2::KernelSpec{
            .unique_id = READER,
            .source = kReaderKernel,
            .num_threads = T,
            .compiler_options = {.defines = defines, .opt_level = m2::KernelSpec::CompilerOptions::OptLevel::O3},
            .dfb_bindings = {m2::DFBBinding{
                .dfb_spec_name = PKT, .accessor_name = "pkt", .endpoint_type = m2::DFBEndpointType::PRODUCER}},
            .semaphore_bindings = sems,
            .tensor_bindings =
                {m2::TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"},
                 m2::TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
            .compile_time_args = ctas,
            .runtime_arg_schema = {.runtime_arg_names = rta_names},
            .hw_config = ttnn::create_reader_datamovement_config(/*disable_dfb_implicit_sync_for_all=*/true),
        });
        spec.kernels.push_back(m2::KernelSpec{
            .unique_id = WRITER,
            .source = kWriterKernel,
            .num_threads = T,
            .compiler_options = {.defines = defines, .opt_level = m2::KernelSpec::CompilerOptions::OptLevel::O3},
            .dfb_bindings = {m2::DFBBinding{
                .dfb_spec_name = PKT, .accessor_name = "pkt", .endpoint_type = m2::DFBEndpointType::CONSUMER}},
            .semaphore_bindings = sems,
            .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
            .compile_time_args = ctas,
            .runtime_arg_schema = {.runtime_arg_names = rta_names},
            .hw_config = ttnn::create_writer_datamovement_config(/*disable_dfb_implicit_sync_for_all=*/true),
        });
        kernels = {READER, WRITER};
    }

    spec.work_units.push_back(m2::WorkUnitSpec{
        .name = "all_gather_workers",
        .kernels = kernels,
        .target_nodes = all_workers,
    });

    //
    // -------- ProgramRunArgs --------
    //
    m2::ProgramRunArgs run_args;
    for (const auto& kernel : kernels) {
        m2::KernelRunArgs kernel_run_args{.kernel = kernel};
        for (uint32_t r = 0; r < num_ranks; r++) {
            const uint32_t next_rank = (r + 1) % num_ranks;
            for (uint32_t w = 0; w < K; w++) {
                const CoreCoord core = workers[r][w];
                const CoreCoord next_noc = device->worker_core_from_logical_core(workers[next_rank][w]);
                m2::AddRuntimeArgsForNode(
                    kernel_run_args.runtime_arg_values,
                    core,
                    {
                        {"rank", r},
                        {"worker", w},
                        {"next_noc_x", static_cast<uint32_t>(next_noc.x)},
                        {"next_noc_y", static_cast<uint32_t>(next_noc.y)},
                    });
            }
        }
        run_args.kernel_run_args.push_back(std::move(kernel_run_args));
    }
    run_args.tensor_args.emplace(INPUT, input.mesh_tensor());
    run_args.tensor_args.emplace(OUTPUT, output.mesh_tensor());

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim::qsr
