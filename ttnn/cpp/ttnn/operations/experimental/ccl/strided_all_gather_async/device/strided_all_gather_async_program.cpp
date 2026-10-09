// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
///
#include <tt-metalium/allocator.hpp>
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/operations/experimental/ccl/strided_all_gather_async/device/strided_all_gather_async_op.hpp"
#include "ttnn/operations/experimental/ccl/llama_common.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/math.hpp>
#include "ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"

#include "ttnn/operations/ccl/common/uops/command_lowering.hpp"

#include "ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "ttnn/operations/ccl/common/host/command_backend_runtime_args_overrider.hpp"
#include <sstream>
#include <type_traits>
#include <ranges>
#include <optional>
#include <tuple>
#include <cstdlib>

using namespace tt::constants;

namespace ttnn::experimental::prim {

namespace detail {

uint32_t strided_all_gather_async_core_count_per_link(
    uint32_t num_workers_per_direction,
    uint32_t num_directions_per_link,
    uint32_t num_mux_cores_per_direction_per_link) {
    return (num_workers_per_direction + num_mux_cores_per_direction_per_link) * num_directions_per_link;
}

uint32_t strided_default_workers(
    const MeshDevice& mesh_device,
    ttnn::ccl::Topology topology,
    uint32_t output_data_size_bytes,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t num_directions_per_link,
    uint32_t num_mux_cores_per_direction_per_link) {
    auto d_id = mesh_device.get_sub_device_ids().at(0);
    auto core_range_set = mesh_device.worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, d_id);
    uint32_t num_cores = core_range_set.num_cores();
    // Above 4 workers we start getting performance drops, so we limit to 4 workers or less, depending on the number of
    // available cores This was determined by the sweep
    // tests/ttnn/multidevice_perf_tests/sweep_all_gather_hyperparameters_T3K.py
    ttsl::SmallVector<uint32_t> candidate_worker_counts;
    // if per link data moved is greater than 0.25 MB, we search greedily for 4 workers, otherwise we search greedily
    // for 2 workers. for ring, half the data is moved per link, so we divide by 2
    double data_moved_per_link_bytes = double(output_data_size_bytes) * (ring_size - 1) / ring_size / num_links /
                                       (topology == ttnn::ccl::Topology::Ring ? 2 : 1);
    if (data_moved_per_link_bytes > double(0.25 * 1024 * 1024)) {
        candidate_worker_counts = {4, 2, 1};
    } else {
        candidate_worker_counts = {2, 1};
    }
    for (auto worker_count : candidate_worker_counts) {
        uint32_t core_count =
            num_links * strided_all_gather_async_core_count_per_link(
                            worker_count, num_directions_per_link, num_mux_cores_per_direction_per_link);
        if (num_cores >= core_count) {
            log_trace(
                tt::LogOp,
                "data_moved_per_link_bytes: {} and worker_count: {}",
                data_moved_per_link_bytes,
                worker_count);
            return worker_count;
        }
    }
    TT_THROW(
        "Not enough cores available on the subdevice or device for the requested match the number of links {}",
        num_links);
}

}  // namespace detail

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

// Kernel push order and runtime-arg layout of the pipeline appended by
// strided_all_gather_async_minimal_default_helper. strided_all_gather_async_patch_runtime_args writes at
// these positions, so they must track the helper in lockstep. Starting at the helper's first kernel index:
//   [reader, writer] per worker pair, pair = (link * kNumDirectionsPerLink + dir) * num_workers + worker,
//   then one matmul-signal aggregator per direction (fused op, when the aggregators are in use),
//   then the fabric mux kernels, whose count differs per device.
namespace strided_all_gather_async_layout {
constexpr uint32_t kNumDirectionsPerLink = 2;
constexpr uint32_t kNumMuxCoresPerDirectionPerLink = 1;

// Reader runtime args: [0] input address, [1] output address, ..., [9] out-ready semaphore address.
constexpr uint32_t kReaderInputAddrArg = 0;
constexpr uint32_t kReaderOutputAddrArg = 1;
constexpr uint32_t kReaderSemaphoreArg = 9;
// Writer runtime args: [0] output address, ..., [11] out-ready semaphore address.
constexpr uint32_t kWriterOutputAddrArg = 0;
constexpr uint32_t kWriterSemaphoreArg = 11;
// Fused-op writers end with [writer_signals_mm, aggregator noc x, aggregator noc y, aggregator semaphore].
constexpr uint32_t kWriterAggregatorTailSize = 4;
constexpr uint32_t kWriterAggregatorTailSignalsOffset = 0;
constexpr uint32_t kWriterAggregatorTailSemaphoreOffset = 3;
// Aggregator runtime args: 6 header words, ring_size k-block counts, then one semaphore address per AG worker.
constexpr uint32_t kAggregatorHeaderArgs = 6;
// semaphore[dir] is the out-ready semaphore of direction dir; the per-worker aggregator semaphores follow,
// direction-major: semaphore[kAggregatorSemaphoreBase + dir * num_ag_workers + global_worker_id].
constexpr uint32_t kAggregatorSemaphoreBase = kNumDirectionsPerLink;

uint32_t worker_pair_index(uint32_t link, uint32_t dir, uint32_t worker, uint32_t num_workers_per_direction) {
    return (((link * kNumDirectionsPerLink) + dir) * num_workers_per_direction) + worker;
}
uint32_t reader_kernel_index(uint32_t first_kernel_index, uint32_t pair) { return first_kernel_index + (2 * pair); }
uint32_t writer_kernel_index(uint32_t first_kernel_index, uint32_t pair) { return first_kernel_index + (2 * pair) + 1; }
uint32_t aggregator_kernel_index(
    uint32_t first_kernel_index, uint32_t num_links, uint32_t num_workers_per_direction, uint32_t dir) {
    return first_kernel_index + (2 * num_links * kNumDirectionsPerLink * num_workers_per_direction) + dir;
}
}  // namespace strided_all_gather_async_layout

// Workers per direction per link. Shared by the descriptor build and the cache-hit patch so both see the
// same kernel layout.
uint32_t strided_all_gather_async_num_workers_per_direction(
    const MeshDevice& mesh_device,
    ttnn::ccl::Topology topology,
    uint32_t output_data_size_bytes,
    uint32_t num_links,
    uint32_t ring_size,
    std::optional<uint32_t> num_workers_per_direction_opt) {
    namespace layout = strided_all_gather_async_layout;
    return num_workers_per_direction_opt.value_or(detail::strided_default_workers(
        mesh_device,
        topology,
        output_data_size_bytes,
        num_links,
        ring_size,
        layout::kNumDirectionsPerLink,
        layout::kNumMuxCoresPerDirectionPerLink));
}

constexpr const char* strided_all_gather_reader_kernel_path =
    "ttnn/cpp/ttnn/operations/experimental/ccl/strided_all_gather_async/device/kernels/minimal_default_reader.cpp";
constexpr const char* strided_all_gather_writer_kernel_path =
    "ttnn/cpp/ttnn/operations/experimental/ccl/strided_all_gather_async/device/kernels/minimal_default_writer.cpp";
constexpr const char* strided_all_gather_aggregator_kernel_path =
    "ttnn/cpp/ttnn/operations/experimental/ccl/strided_all_gather_async/device/kernels/"
    "minimal_default_mm_signal_aggregator.cpp";

// Single-core WORKER semaphore with the lowest id free on `core`.
uint32_t add_core_semaphore(tt::tt_metal::ProgramDescriptor& desc, const CoreCoord& core) {
    const auto semaphore_id = desc.find_available_semaphore_id(core, tt::CoreType::WORKER);
    TT_FATAL(semaphore_id.has_value(), "strided_all_gather_async: no free semaphore id on worker core {}", core);
    desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
        .id = semaphore_id.value(),
        .core_type = tt::CoreType::WORKER,
        .core_ranges = CoreRangeSet(CoreRange(core)),
        .initial_value = 0,
    });
    return semaphore_id.value();
}

template <typename F>
void for_each_strided_all_gather_core_runtime_args(
    tt::tt_metal::Program& program, tt::tt_metal::KernelHandle kernel_index, F&& patch) {
    auto& runtime_args_by_core = tt::tt_metal::GetRuntimeArgs(program, kernel_index);
    for (auto& runtime_args_column : runtime_args_by_core) {
        for (auto& runtime_args : runtime_args_column) {
            if (runtime_args.size() > 0) {
                patch(runtime_args);
            }
        }
    }
}

// Whether the fused writers signal the matmul through the aggregators. The builder makes this choice once for every
// writer and records it in the writer tail; re-deriving it would re-run the worker-core selection (and its warning)
// on every cache hit.
bool writer_signals_mm_flag(tt::tt_metal::Program& program, tt::tt_metal::KernelHandle writer_kernel_index) {
    namespace layout = strided_all_gather_async_layout;
    std::optional<bool> signals_mm;
    for_each_strided_all_gather_core_runtime_args(program, writer_kernel_index, [&](auto& writer_args) {
        TT_FATAL(
            writer_args.size() >= layout::kWriterAggregatorTailSize,
            "strided_all_gather_async fused writer is missing its aggregator tail");
        const size_t tail = writer_args.size() - layout::kWriterAggregatorTailSize;
        signals_mm = writer_args[tail + layout::kWriterAggregatorTailSignalsOffset] != 0;
    });
    TT_FATAL(
        signals_mm.has_value(), "strided_all_gather_async writer kernel {} has no runtime args", writer_kernel_index);
    return signals_mm.value();
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

tt::tt_metal::ProgramDescriptor StridedAllGatherAsyncProgramFactory::create_descriptor(
    const StridedAllGatherAsyncParams& attributes,
    const StridedAllGatherAsyncInputs& tensor_args,
    Tensor& output_tensor,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_FATAL(
        mesh_dispatch_coordinate.has_value(),
        "strided_all_gather_async builds one program per mesh coordinate; no coordinate was given");
    const auto& mesh_coordinate = mesh_dispatch_coordinate.value();

    uint32_t device_index = ttnn::ccl::get_linearized_index_from_physical_coord(
        tensor_args.input_tensor, mesh_coordinate, attributes.cluster_axis);

    std::optional<MeshCoordinate> forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.input_tensor, mesh_coordinate, 1, attributes.topology, attributes.cluster_axis);

    std::optional<MeshCoordinate> backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.input_tensor, mesh_coordinate, -1, attributes.topology, attributes.cluster_axis);
    TT_FATAL(forward_coord.has_value() || backward_coord.has_value(), "DEBUG: forward_coord or backward_coord is null");

    tt::tt_metal::ProgramDescriptor desc;
    std::optional<ttnn::experimental::ccl::StridedAllGatherFusedOpSignaler> empty_fused_op_signaler;
    strided_all_gather_async_minimal_default_helper(
        desc,
        tensor_args.input_tensor,
        mesh_coordinate,
        forward_coord,
        backward_coord,
        output_tensor,
        attributes.dim,
        attributes.num_links,
        attributes.ring_size,
        device_index,
        attributes.topology,
        attributes.semaphore,
        empty_fused_op_signaler,
        false,
        attributes.num_workers_per_link,
        attributes.num_buffers_per_channel,
        attributes.mm_cores_y,
        attributes.mm_block_ht,
        attributes.mm_block_wt,
        CoreCoord(0, 0));
    return desc;
}

void StridedAllGatherAsyncProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const StridedAllGatherAsyncParams& attributes,
    const StridedAllGatherAsyncInputs& tensor_args,
    Tensor& output_tensor,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    strided_all_gather_async_patch_runtime_args(
        program,
        /*first_kernel_index=*/0,
        attributes,
        tensor_args.input_tensor,
        output_tensor,
        /*fused=*/false);
}

void strided_all_gather_async_patch_runtime_args(
    tt::tt_metal::Program& program,
    uint32_t first_kernel_index,
    const StridedAllGatherAsyncParams& attributes,
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    bool fused) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    namespace layout = strided_all_gather_async_layout;

    // The semaphore addresses are excluded from the program hash (see compute_program_hash of the device
    // operations), so every one of them is rewritten here along with the tensor addresses.
    auto* mesh_device = input_tensor.device();
    TT_FATAL(mesh_device != nullptr, "strided_all_gather_async input tensor has no mesh device");
    const auto* input_buffer = input_tensor.buffer();
    const auto* output_buffer = output_tensor.buffer();
    TT_FATAL(input_buffer != nullptr, "strided_all_gather_async input tensor buffer is null");
    TT_FATAL(output_buffer != nullptr, "strided_all_gather_async output tensor buffer is null");
    const uint32_t input_address = input_buffer->address();
    const uint32_t output_address = output_buffer->address();

    const uint32_t num_links = attributes.num_links;
    const uint32_t num_workers_per_direction = strided_all_gather_async_num_workers_per_direction(
        *mesh_device,
        attributes.topology,
        output_buffer->size(),
        num_links,
        attributes.ring_size,
        attributes.num_workers_per_link);
    const uint32_t num_ag_workers = num_links * num_workers_per_direction;
    const auto& semaphore = attributes.semaphore;
    const auto aggregator_semaphore_address = [&](uint32_t dir, uint32_t global_worker_id) {
        return static_cast<uint32_t>(
            semaphore.at(layout::kAggregatorSemaphoreBase + (dir * num_ag_workers) + global_worker_id).address());
    };

    const bool aggregators_in_use =
        fused && writer_signals_mm_flag(program, layout::writer_kernel_index(first_kernel_index, 0));
    for (uint32_t link = 0; link < num_links; link++) {
        for (uint32_t dir = 0; dir < layout::kNumDirectionsPerLink; dir++) {
            const auto out_ready_semaphore_address = static_cast<uint32_t>(semaphore.at(dir).address());
            for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                const uint32_t pair = layout::worker_pair_index(link, dir, worker, num_workers_per_direction);
                const uint32_t global_worker_id = (link * num_workers_per_direction) + worker;

                for_each_strided_all_gather_core_runtime_args(
                    program, layout::reader_kernel_index(first_kernel_index, pair), [&](auto& reader_args) {
                        reader_args[layout::kReaderInputAddrArg] = input_address;
                        reader_args[layout::kReaderOutputAddrArg] = output_address;
                        reader_args[layout::kReaderSemaphoreArg] = out_ready_semaphore_address;
                    });
                for_each_strided_all_gather_core_runtime_args(
                    program, layout::writer_kernel_index(first_kernel_index, pair), [&](auto& writer_args) {
                        writer_args[layout::kWriterOutputAddrArg] = output_address;
                        writer_args[layout::kWriterSemaphoreArg] = out_ready_semaphore_address;
                        if (aggregators_in_use) {
                            const size_t tail = writer_args.size() - layout::kWriterAggregatorTailSize;
                            writer_args[tail + layout::kWriterAggregatorTailSemaphoreOffset] =
                                aggregator_semaphore_address(dir, global_worker_id);
                        }
                    });
            }
        }
    }

    if (aggregators_in_use) {
        for (uint32_t dir = 0; dir < layout::kNumDirectionsPerLink; dir++) {
            for_each_strided_all_gather_core_runtime_args(
                program,
                layout::aggregator_kernel_index(first_kernel_index, num_links, num_workers_per_direction, dir),
                [&](auto& aggregator_args) {
                    for (uint32_t w = 0; w < num_ag_workers; w++) {
                        aggregator_args[layout::kAggregatorHeaderArgs + attributes.ring_size + w] =
                            aggregator_semaphore_address(dir, w);
                    }
                });
        }
    }
}

void strided_all_gather_async_minimal_default_helper(
    tt::tt_metal::ProgramDescriptor& desc,
    const Tensor& input_tensor,
    const MeshCoordinate& sender_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    Tensor& output_tensor,
    const uint32_t /*dim*/,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    std::optional<ttnn::experimental::ccl::StridedAllGatherFusedOpSignaler>& fused_op_signaler,
    bool read_local_slice_from_input,
    std::optional<uint32_t> num_workers_per_direction_opt,
    std::optional<uint32_t> num_buffers_per_channel,
    std::optional<uint32_t> mm_cores_y,
    std::optional<uint32_t> mm_block_ht,
    std::optional<uint32_t> mm_block_wt,
    const CoreCoord core_grid_offset,
    const MMSignalAggregatorMode mm_signal_aggregator_mode) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    namespace layout = strided_all_gather_async_layout;
    using tt::tt_metal::CBDescriptor;
    using tt::tt_metal::CBFormatDescriptor;
    using tt::tt_metal::DataMovementConfigDescriptor;
    using tt::tt_metal::KernelDescriptor;
    using tt::tt_metal::ReaderConfigDescriptor;
    using tt::tt_metal::WriterConfigDescriptor;

    const uint32_t first_kernel_index = desc.kernels.size();

    // Tensor Info
    auto* input_buffer = input_tensor.buffer();
    auto* output_buffer = output_tensor.buffer();
    TT_FATAL(input_buffer != nullptr, "strided_all_gather_async input tensor buffer is null");
    TT_FATAL(output_buffer != nullptr, "strided_all_gather_async output tensor buffer is null");
    const auto input_tensor_num_pages = input_buffer->num_pages();
    const auto& input_tensor_shape = input_tensor.padded_shape();
    const auto& output_tensor_shape = output_tensor.padded_shape();
    auto* mesh_device = input_tensor.device();
    TT_FATAL(mesh_device != nullptr, "Mesh device not found");

    // op hyperparams
    uint32_t num_directions_per_link = layout::kNumDirectionsPerLink;
    uint32_t num_mux_cores_per_direction_per_link = layout::kNumMuxCoresPerDirectionPerLink;
    // Get worker cores
    // 2 senders (reader + writer) per direction (forward, reverse_order) per link
    uint32_t output_data_size_bytes = output_buffer->size();
    uint32_t num_workers_per_direction = strided_all_gather_async_num_workers_per_direction(
        *mesh_device, topology, output_data_size_bytes, num_links, ring_size, num_workers_per_direction_opt);
    uint32_t num_cores_per_link = detail::strided_all_gather_async_core_count_per_link(
        num_workers_per_direction, num_directions_per_link, num_mux_cores_per_direction_per_link);

    log_trace(tt::LogOp, "DEBUG: num_workers_per_direction: {}", num_workers_per_direction);
    uint32_t num_buffers_full_size_channels = num_buffers_per_channel.value_or(1);

    /* All gather fusion */
    bool fuse_op = fused_op_signaler.has_value();

    // Option W: the AG writer signals the remote device's matmul through per-direction aggregator cores, which cost
    // num_directions_per_link worker cores on top of the mux/workers. Without them the reader signals the matmul.
    const uint32_t num_mux_worker_cores = num_links * num_cores_per_link;
    bool writer_signals_mm = false;
    std::optional<ttnn::ccl::WorkerCoreSelection> aggregator_core_selection;
    if (fuse_op) {
        // Ask for the aggregator-inclusive set and check what comes back: core_grid_offset can shift the trailing
        // cores off the worker grid, where kernels cannot be placed.
        auto selection = ttnn::ccl::try_choose_worker_cores(
            1, num_mux_worker_cores + num_directions_per_link, mesh_device, std::nullopt, core_grid_offset);
        const bool aggregators_fit = selection.all_placeable();
        switch (mm_signal_aggregator_mode) {
            case MMSignalAggregatorMode::On:
                TT_FATAL(
                    aggregators_fit,
                    "strided AG: matmul-signal aggregators need {} worker cores ({} mux/worker + {} aggregator), but "
                    "at core grid offset {} the last {} of them fall outside the worker grid",
                    num_mux_worker_cores + num_directions_per_link,
                    num_mux_worker_cores,
                    num_directions_per_link,
                    core_grid_offset.str(),
                    selection.unplaceable_cores.size());
                writer_signals_mm = true;
                break;
            case MMSignalAggregatorMode::Off: break;
            case MMSignalAggregatorMode::Auto:
                writer_signals_mm = aggregators_fit;
                if (!aggregators_fit) {
                    log_warning(
                        tt::LogOp,
                        "strided AG: matmul-signal aggregators need {} worker cores ({} mux/worker + {} aggregator), "
                        "but at core grid offset {} the last {} of them fall outside the worker grid; falling back to "
                        "reader-signaled matmul. Pass mm_signal_aggregator_mode=On to require the aggregators instead.",
                        num_mux_worker_cores + num_directions_per_link,
                        num_mux_worker_cores,
                        num_directions_per_link,
                        core_grid_offset.str(),
                        selection.unplaceable_cores.size());
                }
                break;
        }
        if (writer_signals_mm) {
            aggregator_core_selection = std::move(selection);
        }
    }
    const uint32_t num_ag_workers = num_links * num_workers_per_direction;

    // Need a separate signaler for the sender workers, to handle the first tensor slice that is locally available
    std::optional<ttnn::experimental::ccl::StridedAllGatherFusedOpSignaler> fused_op_signaler_sender_workers;
    std::optional<ttnn::experimental::ccl::StridedAllGatherFusedOpSignaler> fused_op_signaler_forward;
    std::optional<ttnn::experimental::ccl::StridedAllGatherFusedOpSignaler> fused_op_signaler_backward;
    if (fuse_op) {
        fused_op_signaler_sender_workers = fused_op_signaler.value();
        fused_op_signaler_forward = fused_op_signaler.value();
        fused_op_signaler_backward = fused_op_signaler.value();
    }

    // Get OP Config, topology config
    uint32_t page_size = input_buffer->page_size();
    auto [num_targets_forward, num_targets_backward] =
        ttnn::ccl::get_forward_backward_line_mcast_distance(ring_size, ring_index, topology, false);
    auto [unicast_forward_args, unicast_backward_args] = ttnn::ccl::get_forward_backward_line_unicast_configuration(
        sender_device_coord, forward_coord, backward_coord, mesh_device);

    // Option W: carve `num_directions_per_link` extra trailing cores as per-direction matmul-signal aggregators
    const uint32_t num_agg_cores = writer_signals_mm ? num_directions_per_link : 0;
    const uint32_t total_worker_cores = num_mux_worker_cores + num_agg_cores;
    CoreRangeSet all_core_range;
    std::vector<CoreCoord> all_cores;
    if (aggregator_core_selection.has_value()) {
        all_core_range = std::move(aggregator_core_selection->core_range_set);
        all_cores = std::move(aggregator_core_selection->cores);
    } else {
        std::tie(all_core_range, all_cores) =
            ttnn::ccl::choose_worker_cores(1, total_worker_cores, mesh_device, std::nullopt, core_grid_offset);
    }
    std::set<CoreRange> sender_worker_core_ranges;
    std::set<CoreRange> sender_forward_core_ranges;
    std::set<CoreRange> sender_backward_core_ranges;
    std::set<CoreRange> mux_forward_core_ranges;
    std::set<CoreRange> mux_backward_core_ranges;
    std::vector<CoreCoord> sender_forward_cores;
    sender_forward_cores.reserve(num_links * num_workers_per_direction);
    std::vector<CoreCoord> sender_backward_cores;
    sender_backward_cores.reserve(num_links * num_workers_per_direction);
    uint32_t core_id = 0;
    for (uint32_t link = 0; link < num_links; link++) {
        for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
            const auto& mux_core = all_cores[core_id++];
            if (dir) {
                mux_forward_core_ranges.insert(CoreRange(mux_core));
            } else {
                mux_backward_core_ranges.insert(CoreRange(mux_core));
            }
            for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                const auto& worker_core = all_cores[core_id++];
                if (dir) {
                    sender_forward_cores.push_back(worker_core);
                    sender_forward_core_ranges.insert(CoreRange(worker_core));
                } else {
                    sender_backward_cores.push_back(worker_core);
                    sender_backward_core_ranges.insert(CoreRange(worker_core));
                }
                sender_worker_core_ranges.insert(CoreRange(worker_core));
            }
        }
    }
    CoreRangeSet sender_worker_core_range_set = CoreRangeSet(sender_worker_core_ranges);
    CoreRangeSet sender_forward_core_range_set = CoreRangeSet(sender_forward_core_ranges);
    CoreRangeSet sender_backward_core_range_set = CoreRangeSet(sender_backward_core_ranges);
    CoreRangeSet mux_forward_core_range_set = CoreRangeSet(mux_forward_core_ranges);
    CoreRangeSet mux_backward_core_range_set = CoreRangeSet(mux_backward_core_ranges);

    // Option W: per-direction matmul-signal aggregator cores (the trailing cores from choose_worker_cores)
    std::vector<CoreCoord> agg_core_logical(num_directions_per_link);
    std::vector<CoreCoord> agg_core_virtual(num_directions_per_link);
    // Holds L1 ADDRESSES (not semaphore ids): these are cross-device fabric atomic-inc targets
    std::vector<std::vector<uint32_t>> agg_per_worker_sem_ids(num_directions_per_link);
    if (writer_signals_mm) {
        for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
            CoreCoord agg_logical = all_cores[(num_links * num_cores_per_link) + dir];
            agg_core_logical[dir] = agg_logical;
            agg_core_virtual[dir] = mesh_device->worker_core_from_logical_core(agg_logical);
            for (uint32_t w = 0; w < num_ag_workers; w++) {
                agg_per_worker_sem_ids[dir].push_back(static_cast<uint32_t>(
                    semaphore.at(layout::kAggregatorSemaphoreBase + dir * num_ag_workers + w)
                        .address()));  // smuggled-rta-ok: caller-supplied GlobalSemaphore, excluded from hash,
                                       // re-applied by strided_all_gather_async_patch_runtime_args
            }
        }
    }

    // L1 Scratch CB Creation
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    uint32_t l1_scratch_cb_page_size_bytes = page_size;

    // scatter-write packs this many tiles (distinct dest noc addresses) per fabric packet
    uint32_t max_target_noc_addresses_per_packet = 4;

    // for bfloat8_b, tile_num_per_link=6, we would need to send 2 packages, but they can be of size 3 instead of 4
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t num_tiles_to_write_per_packet = std::min(max_target_noc_addresses_per_packet, num_pages_per_packet);
    log_info(
        tt::LogOp,
        "strided AG: num_tiles_to_write_per_packet={} (cap={}, pages_per_packet={})",
        num_tiles_to_write_per_packet,
        max_target_noc_addresses_per_packet,
        num_pages_per_packet);
    uint32_t cb_num_pages = 3 * num_tiles_to_write_per_packet;  // triple buffering
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    // CBs for transferring data between sender_reader and sender_writer
    uint32_t sender_cb_index = tt::CB::c_in0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_num_pages * l1_scratch_cb_page_size_bytes,
        .core_ranges = sender_worker_core_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(sender_cb_index),
            .data_format = df,
            .page_size = l1_scratch_cb_page_size_bytes,
        }}},
    });

    uint32_t batch_head_size = input_tensor_shape[0] * input_tensor_shape[1];

    uint32_t single_batch_head_num_pages = input_tensor_num_pages / batch_head_size;
    TT_FATAL(!(input_tensor_shape[3] % TILE_WIDTH), "Input tensor width must be a multiple of TILE_WIDTH");
    TT_FATAL(!(output_tensor_shape[3] % TILE_WIDTH), "Output tensor width must be a multiple of TILE_WIDTH");
    uint32_t TILE_WIDTH = 32;

    uint32_t input_tensor_Wt = input_tensor_shape[3] / TILE_WIDTH;
    uint32_t input_tensor_Ht = input_tensor_shape[2] / TILE_WIDTH;

    uint32_t output_tensor_Wt = output_tensor_shape[3] / TILE_WIDTH;
    uint32_t output_tensor_Ht = output_tensor_shape[2] / TILE_WIDTH;

    uint32_t mm_cores_y_val = mm_cores_y.value_or(0);
    uint32_t mm_block_ht_val = mm_block_ht.value_or(0);
    uint32_t mm_block_wt_val = mm_block_wt.value_or(0);

    std::map<std::string, std::string> reader_compute_defines;
    std::map<std::string, std::string> writer_compute_defines;
    std::map<std::string, std::string> agg_defines;

    // Streaming matmul signal: deliver each chunk's M-rows as IN0_SUB_CHUNKS row-bands, one aggregator inc per band
    const char* in0_sub_chunks_env = std::getenv("IN0_SUB_CHUNKS");
    const std::string in0_sub_chunks_str = (in0_sub_chunks_env != nullptr) ? in0_sub_chunks_env : "1";
    // Only the aggregator emits one matmul signal per band; the reader-signaled path emits one per chunk, which the
    // matmul's per-band waits would outrun.
    TT_FATAL(
        !fuse_op || writer_signals_mm || in0_sub_chunks_str == "1",
        "strided AG: IN0_SUB_CHUNKS={} requires the matmul-signal aggregators, which are not in use",
        in0_sub_chunks_str);
    reader_compute_defines["IN0_SUB_CHUNKS"] = in0_sub_chunks_str;
    writer_compute_defines["IN0_SUB_CHUNKS"] = in0_sub_chunks_str;
    agg_defines["IN0_SUB_CHUNKS"] = in0_sub_chunks_str;

    // The worker->fabric path goes through Mux V2 (dual-RISC forwarder+manager)
    writer_compute_defines["USE_MUX_V2"] = "1";

    // KERNEL CREATION
    /* All gather fusion */
    std::vector<std::vector<uint32_t>> device_chunk_widths(ring_size);
    std::vector<uint32_t> device_k_block_counts(ring_size, 0);
    uint32_t padded_K_tiles = tt::round_up(output_tensor_Wt, mm_block_wt_val);
    uint32_t K_blocks = padded_K_tiles / mm_block_wt_val;

    uint32_t curr_device = 0;
    uint32_t curr_device_end = input_tensor_Wt - 1;
    uint32_t device_max_chunks = 0;
    for (uint32_t k_block_iter = 0; k_block_iter < K_blocks; k_block_iter++) {
        uint32_t curr_k_block_start = k_block_iter * mm_block_wt_val;
        uint32_t curr_k_block_end = ((k_block_iter + 1) * mm_block_wt_val) - 1;
        if (curr_k_block_end < curr_device_end) {
            device_k_block_counts[curr_device]++;
            device_chunk_widths[curr_device].push_back(curr_k_block_end - curr_k_block_start + 1);
        } else if (curr_k_block_end == curr_device_end) {
            device_k_block_counts[curr_device]++;
            device_chunk_widths[curr_device].push_back(curr_k_block_end - curr_k_block_start + 1);
            curr_device++;
            curr_device_end = (curr_device + 1) * input_tensor_Wt - 1;
        } else if (curr_k_block_end > curr_device_end) {
            device_k_block_counts[curr_device]++;
            device_chunk_widths[curr_device].push_back(curr_device_end - curr_k_block_start + 1);
            if (curr_device + 1 < ring_size) {
                device_k_block_counts[curr_device + 1]++;
                device_chunk_widths[curr_device + 1].push_back(curr_k_block_end - curr_device_end);
            }
            curr_device++;
            curr_device_end = (curr_device + 1) * input_tensor_Wt - 1;
        }
    }
    for (uint32_t d = 0; d < ring_size; d++) {
        device_max_chunks = std::max(device_max_chunks, (uint32_t)device_chunk_widths[d].size());
    }

    if (fuse_op) {
        fused_op_signaler_forward->init_all_gather(
            desc, mesh_device, sender_forward_core_range_set, sender_forward_cores);
        fused_op_signaler_backward->init_all_gather(
            desc, mesh_device, sender_backward_core_range_set, sender_backward_cores);
        fused_op_signaler_sender_workers->init_all_gather(
            desc, mesh_device, sender_forward_core_range_set, sender_forward_cores);
    }

    const uint32_t l1_unreserved_base_address =
        mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const size_t mux_base_l1_address = l1_unreserved_base_address;
    // The mux stays below the floor of the L1_SMALL region, where carried semaphores live (#56769).
    const size_t mux_l1_small_floor_address = ttnn::ccl::l1_small_floor_address(*mesh_device);
    // V2 places one logical channel per worker
    const tt::tt_fabric::FabricMuxV2Config mux_v2_config(
        static_cast<uint8_t>(num_workers_per_direction),
        static_cast<uint8_t>(num_buffers_full_size_channels),
        tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes(),
        mux_base_l1_address,
        mux_l1_small_floor_address);

    const auto mux_core_offset_of = [&](uint32_t link, uint32_t dir) {
        return (link * num_cores_per_link) + (dir * (num_mux_cores_per_direction_per_link + num_workers_per_direction));
    };
    const auto mux_connection_valid_of = [&](uint32_t dir) {
        return (dir && backward_coord.has_value()) || (!dir && forward_coord.has_value());
    };

    for (uint32_t link = 0; link < num_links; link++) {
        for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
            uint32_t mux_core_offset = mux_core_offset_of(link, dir);
            CoreCoord mux_logical_core = all_cores[mux_core_offset];
            CoreCoord mux_virtual_core = mesh_device->worker_core_from_logical_core(mux_logical_core);
            const bool mux_connection_valid = mux_connection_valid_of(dir);

            for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                CoreCoord core = all_cores[mux_core_offset + num_mux_cores_per_direction_per_link + worker];
                CoreCoord virtual_core = mesh_device->worker_core_from_logical_core(core);
                CoreCoord supplemental_core = all_cores
                    [(link * num_cores_per_link) +
                     ((1 - dir) * (num_mux_cores_per_direction_per_link + num_workers_per_direction)) +
                     num_mux_cores_per_direction_per_link + worker];
                CoreCoord opposite_core_coord = mesh_device->worker_core_from_logical_core(supplemental_core);
                const CoreRangeSet worker_core_range{CoreRange(core)};
                const uint32_t pair = layout::worker_pair_index(link, dir, worker, num_workers_per_direction);

                uint32_t global_worker_id = (link * num_workers_per_direction) + worker;
                uint32_t global_worker_count = num_links * num_workers_per_direction;
                uint32_t base_pages_per_worker = single_batch_head_num_pages / global_worker_count;
                uint32_t remainder = single_batch_head_num_pages % global_worker_count;
                uint32_t tiles_per_core = base_pages_per_worker + ((global_worker_id < remainder) ? 1 : 0);
                const auto out_ready_semaphore_address = static_cast<uint32_t>(
                    semaphore.at(dir).address());  // smuggled-rta-ok: caller-supplied GlobalSemaphore, excluded
                                                   // from hash, re-applied by override_runtime_arguments

                // Reader
                KernelDescriptor reader_desc;
                reader_desc.kernel_source = strided_all_gather_reader_kernel_path;
                reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
                reader_desc.core_ranges = worker_core_range;
                reader_desc.compile_time_args = {
                    ring_index,                       // my_chip_id
                    sender_cb_index,                  // cb_forward_id
                    num_tiles_to_write_per_packet,    // num_tiles_to_write_per_packet
                    page_size,                        // tensor0_page_size
                    num_targets_forward,              // num_slices_forward_direction
                    num_targets_backward,             // num_slices_backward_direction
                    static_cast<uint32_t>(topology),  // topology
                    dir,                              // direction
                    fuse_op,                          // fused op
                    global_worker_count,
                    global_worker_id,
                };
                tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_desc.compile_time_args);
                tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(reader_desc.compile_time_args);
                reader_desc.named_compile_time_args = {{"cb_forward_id", sender_cb_index}};
                reader_desc.defines = {reader_compute_defines.begin(), reader_compute_defines.end()};
                reader_desc.config = ReaderConfigDescriptor{};

                KernelDescriptor::RTArgList reader_rt_args;
                reader_rt_args.push_back(input_buffer);   // input_tensor_address
                reader_rt_args.push_back(output_buffer);  // output_tensor_address
                std::vector<uint32_t> reader_rt_args_tail = {
                    input_tensor_Wt,              // width in tiles of the output shard
                    input_tensor_Ht,              // height in tiles of the output shard
                    output_tensor_Wt,             // width in tiles of entire output
                    batch_head_size,              // product of the first two dims
                    global_worker_id,             //
                    tiles_per_core,               //
                    ring_size,                    // ring_size
                    out_ready_semaphore_address,  // out_ready_semaphore_forward
                    mm_block_wt_val,
                    mm_block_ht_val,
                    mm_cores_y_val};
                reader_rt_args_tail.push_back(device_max_chunks);
                for (uint32_t d = 0; d < ring_size; d++) {
                    reader_rt_args_tail.push_back(device_k_block_counts[d]);
                    reader_rt_args_tail.push_back(device_chunk_widths[d].size());
                    for (unsigned int width : device_chunk_widths[d]) {
                        reader_rt_args_tail.push_back(width);
                    }
                }
                if (fuse_op) {
                    if (dir) {
                        fused_op_signaler_forward->push_all_gather_fused_op_rt_args(
                            reader_rt_args_tail,
                            num_workers_per_direction * num_links,
                            worker + (link * num_workers_per_direction),
                            1);
                    } else {
                        fused_op_signaler_backward->push_all_gather_fused_op_rt_args(
                            reader_rt_args_tail,
                            num_workers_per_direction * num_links,
                            worker + (link * num_workers_per_direction),
                            0);
                    }
                    // When the writer signals the matmul directly over fabric
                    reader_rt_args_tail.push_back(static_cast<uint32_t>(writer_signals_mm ? 1 : 0));
                }
                reader_rt_args.append(reader_rt_args_tail);
                reader_desc.emplace_runtime_args(core, reader_rt_args);
                TT_FATAL(
                    reader_desc.runtime_args.back().second.at(layout::kReaderSemaphoreArg) ==
                        out_ready_semaphore_address,
                    "strided_all_gather_async reader: kReaderSemaphoreArg ({}) no longer points at the out-ready "
                    "semaphore address the cache-hit patch rewrites",
                    layout::kReaderSemaphoreArg);

                TT_FATAL(
                    desc.kernels.size() == layout::reader_kernel_index(first_kernel_index, pair),
                    "strided_all_gather_async reader of worker pair {} must sit at kernel index {}",
                    pair,
                    layout::reader_kernel_index(first_kernel_index, pair));
                desc.kernels.push_back(std::move(reader_desc));

                // Writer
                KernelDescriptor writer_desc;
                writer_desc.kernel_source = strided_all_gather_writer_kernel_path;
                writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
                writer_desc.core_ranges = worker_core_range;
                writer_desc.compile_time_args = {
                    ring_index,                       // my_chip_id
                    sender_cb_index,                  // cb_forward_id
                    num_tiles_to_write_per_packet,    // num_tiles_to_write_per_packet
                    page_size,                        // tensor0_page_size
                    num_targets_forward,              // num_targets_forward_direction
                    num_targets_backward,             // num_targets_backward_direction
                    fuse_op,                          // fused op
                    static_cast<uint32_t>(topology),  // topology
                    dir,                              // direction
                    global_worker_count,
                    global_worker_id,
                };
                auto& sender_writer_compile_args = writer_desc.compile_time_args;
                // Mux V2 is runtime-arg driven and needs no mux compile-time args; the 12 slots after
                // is_termination_master stay reserved so the unicast args and TensorAccessorArgs keep their indices.
                sender_writer_compile_args.push_back(worker == 0);
                sender_writer_compile_args.insert(sender_writer_compile_args.end(), 12, 0);
                if (dir) {
                    sender_writer_compile_args.insert(
                        sender_writer_compile_args.end(), unicast_backward_args.begin(), unicast_backward_args.end());
                } else {
                    sender_writer_compile_args.insert(
                        sender_writer_compile_args.end(), unicast_forward_args.begin(), unicast_forward_args.end());
                }
                tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(sender_writer_compile_args);
                writer_desc.named_compile_time_args = {{"cb_forward_id", sender_cb_index}};
                writer_desc.defines = {writer_compute_defines.begin(), writer_compute_defines.end()};
                writer_desc.config = WriterConfigDescriptor{};

                KernelDescriptor::RTArgList writer_rt_args;
                writer_rt_args.push_back(output_buffer);  // output_tensor_address
                std::vector<uint32_t> writer_rt_args_tail = {
                    input_tensor_Wt,              // width in tiles of the input shard
                    input_tensor_Ht,              // height in tiles of the input shard
                    output_tensor_Wt,             // width in tiles of entire output
                    output_tensor_Ht,             // height in tiles of entire output
                    batch_head_size,              // product of the first two dims
                    global_worker_id,             //
                    tiles_per_core,               //
                    virtual_core.x,               // out_ready_sem_noc0_x
                    virtual_core.y,               // out_ready_sem_noc0_y
                    ring_size,                    // ring_size
                    out_ready_semaphore_address,  // out_ready_semaphore_forward
                    opposite_core_coord.x,
                    opposite_core_coord.y,
                    mm_block_wt_val,
                    mm_block_ht_val,
                    mm_cores_y_val,
                    read_local_slice_from_input};
                writer_rt_args_tail.push_back(device_max_chunks);
                for (uint32_t d = 0; d < ring_size; d++) {
                    writer_rt_args_tail.push_back(device_k_block_counts[d]);
                    writer_rt_args_tail.push_back(device_chunk_widths[d].size());
                    for (unsigned int width : device_chunk_widths[d]) {
                        writer_rt_args_tail.push_back(width);
                    }
                }
                // Layout: [mux_connection_valid][11 client-connection args]
                writer_rt_args_tail.push_back(static_cast<uint32_t>(mux_connection_valid ? 1 : 0));
                const uint32_t flow_control_sem_id = add_core_semaphore(desc, core);
                const uint32_t teardown_sem_id = add_core_semaphore(desc, core);
                mux_v2_config.append_client_connection_rt_args(
                    mux_virtual_core,
                    static_cast<uint8_t>(worker),
                    {flow_control_sem_id, teardown_sem_id},
                    writer_rt_args_tail);
                if (fuse_op) {
                    // Local self-signal path (op_signaler_sender): targets the single 'self' semaphore
                    const uint32_t self_sem_index =
                        fused_op_signaler_sender_workers->fused_op_receiver_signal_semaphores.size() - 1;
                    fused_op_signaler_sender_workers->push_all_gather_fused_op_rt_args(
                        writer_rt_args_tail,
                        num_workers_per_direction * num_links,
                        worker + (link * num_workers_per_direction),
                        self_sem_index);
                    // Option W: this worker signals its own per-worker semaphore on the remote direction's
                    writer_rt_args_tail.push_back(static_cast<uint32_t>(writer_signals_mm ? 1 : 0));
                    writer_rt_args_tail.push_back(
                        static_cast<uint32_t>(writer_signals_mm ? agg_core_virtual[dir].x : 0));
                    writer_rt_args_tail.push_back(
                        static_cast<uint32_t>(writer_signals_mm ? agg_core_virtual[dir].y : 0));
                    writer_rt_args_tail.push_back(
                        static_cast<uint32_t>(writer_signals_mm ? agg_per_worker_sem_ids[dir][global_worker_id] : 0));
                }
                writer_rt_args.append(writer_rt_args_tail);
                writer_desc.emplace_runtime_args(core, writer_rt_args);
                TT_FATAL(
                    writer_desc.runtime_args.back().second.at(layout::kWriterSemaphoreArg) ==
                        out_ready_semaphore_address,
                    "strided_all_gather_async writer: kWriterSemaphoreArg ({}) no longer points at the out-ready "
                    "semaphore address the cache-hit patch rewrites",
                    layout::kWriterSemaphoreArg);

                TT_FATAL(
                    desc.kernels.size() == layout::writer_kernel_index(first_kernel_index, pair),
                    "strided_all_gather_async writer of worker pair {} must sit at kernel index {}",
                    pair,
                    layout::writer_kernel_index(first_kernel_index, pair));
                desc.kernels.push_back(std::move(writer_desc));
            }
        }
    }

    // Option W: create one matmul-signal aggregator kernel per direction
    if (writer_signals_mm) {
        const uint32_t num_mm_cores = fused_op_signaler.value().num_fused_op_cores_to_signal;
        for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
            KernelDescriptor agg_desc;
            agg_desc.kernel_source = strided_all_gather_aggregator_kernel_path;
            agg_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
            agg_desc.core_ranges = CoreRangeSet(CoreRange(agg_core_logical[dir]));
            agg_desc.compile_time_args = {
                ring_index,
                num_targets_forward,
                num_targets_backward,
                static_cast<uint32_t>(topology),
                dir,
                num_ag_workers,
                num_mm_cores,
            };
            agg_desc.defines = {agg_defines.begin(), agg_defines.end()};
            agg_desc.config = DataMovementConfigDescriptor{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
            };

            std::vector<uint32_t> agg_rt_args = {
                ring_size,
                batch_head_size,
                input_tensor_Ht,
                mm_cores_y_val,
                mm_block_ht_val,
                fused_op_signaler.value().fused_op_receiver_signal_semaphores[dir],  // mm direction sem id
            };
            TT_FATAL(
                agg_rt_args.size() == layout::kAggregatorHeaderArgs,
                "strided_all_gather_async aggregator header must hold {} args",
                layout::kAggregatorHeaderArgs);
            for (uint32_t d = 0; d < ring_size; d++) {
                agg_rt_args.push_back(device_k_block_counts[d]);
            }
            for (uint32_t w = 0; w < num_ag_workers; w++) {
                agg_rt_args.push_back(agg_per_worker_sem_ids[dir][w]);
            }
            for (const auto& mm_core : fused_op_signaler.value().fused_op_receiver_cores_noc) {
                agg_rt_args.push_back(static_cast<uint32_t>(mm_core.x));
                agg_rt_args.push_back(static_cast<uint32_t>(mm_core.y));
            }
            agg_desc.runtime_args.emplace_back(agg_core_logical[dir], std::move(agg_rt_args));

            const uint32_t agg_kernel_index =
                layout::aggregator_kernel_index(first_kernel_index, num_links, num_workers_per_direction, dir);
            TT_FATAL(
                desc.kernels.size() == agg_kernel_index,
                "strided_all_gather_async aggregator of direction {} must sit at kernel index {}",
                dir,
                agg_kernel_index);
            desc.kernels.push_back(std::move(agg_desc));
        }
    }

    // Fabric mux kernels go last: their count differs per device (a missing neighbour has no mux connection),
    // which would otherwise shift the worker kernel indices the cache-hit patch relies on.
    for (uint32_t link = 0; link < num_links; link++) {
        for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
            if (!mux_connection_valid_of(dir)) {
                continue;
            }
            CoreCoord mux_logical_core = all_cores[mux_core_offset_of(link, dir)];
            const auto src_node_id = mesh_device->get_fabric_node_id(sender_device_coord);
            const auto dst_node_id =
                mesh_device->get_fabric_node_id(dir ? backward_coord.value() : forward_coord.value());
            // Creates both the forwarder (RISCV_0) and manager
            tt::tt_fabric::add_fabric_mux_v2_to_program(
                desc,
                mux_v2_config,
                mux_logical_core,
                src_node_id,
                dst_node_id,
                link,
                tt::tt_metal::NOC::RISCV_0_default);
        }
    }
}

}  // namespace ttnn::experimental::prim
