// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "width_sharded_all_reduce_program_factory.hpp"

#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tile.hpp>

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {
namespace {

CoreRangeSet cores_to_corerangeset(const std::vector<CoreCoord>& cores) {
    std::vector<CoreRange> core_ranges;
    core_ranges.reserve(cores.size());
    for (const auto& core : cores) {
        core_ranges.push_back(CoreRange(core));
    }
    return CoreRangeSet(core_ranges);
}

std::tuple<CoreRangeSet, std::vector<CoreCoord>> choose_sender_cores(
    size_t num_links, const CoreRangeSet& available_cores) {
    CoreRangeSet sender_worker_core_range;
    for (const auto& cr : available_cores.ranges()) {
        auto start = cr.start_coord;
        auto end = cr.end_coord;
        for (size_t y = start.y; y <= end.y; y++) {
            for (size_t x = start.x; x <= end.x; x++) {
                sender_worker_core_range =
                    sender_worker_core_range.merge(CoreRangeSet(CoreRange(CoreCoord(x, y), CoreCoord(x, y))));
                if (sender_worker_core_range.num_cores() == num_links) {
                    return {sender_worker_core_range, corerange_to_cores(sender_worker_core_range, std::nullopt, true)};
                }
            }
        }
    }
    TT_FATAL(
        sender_worker_core_range.num_cores() >= num_links,
        "width_sharded_all_reduce needs {} worker cores outside the tensor grid, but only {} are free",
        num_links,
        sender_worker_core_range.num_cores());
    return {sender_worker_core_range, corerange_to_cores(sender_worker_core_range, std::nullopt, true)};
}

tt::tt_metal::Tile rm_face_tile() { return tt::tt_metal::Tile({1, TILE_WIDTH}, false); }

std::shared_ptr<Tensor> make_scratch_tensor(const Tensor& input, uint32_t ring_size) {
    const auto& shard = input.shard_spec().value();
    auto logical = input.logical_shape();
    logical[-2] *= ring_size;
    const ShardSpec scratch_shard{shard.grid, {shard.shape[0] * ring_size, shard.shape[1]}, shard.orientation};
    const MemoryConfig scratch_memcfg{TensorMemoryLayout::WIDTH_SHARDED, BufferType::L1, scratch_shard};
    return std::make_shared<Tensor>(create_device_tensor(
        tt::tt_metal::TensorSpec(logical, TensorLayout(input.dtype(), PageConfig(Layout::ROW_MAJOR), scratch_memcfg)),
        input.device()));
}

}  // namespace

WidthShardedAllReduceMeshWorkloadFactory::cached_mesh_workload_t
WidthShardedAllReduceMeshWorkloadFactory::create_mesh_workload(
    const WidthShardedAllReduceParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const WidthShardedAllReduceInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto* mesh_device = tensor_args.input.device();
    auto subdevice_id = operation_attributes.sub_device_id.value_or(mesh_device->get_sub_device_ids().at(0));
    const auto available_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, subdevice_id);
    ttsl::SmallVector<tt::tt_metal::SubDeviceId> subdevices = {subdevice_id};

    auto scratch = make_scratch_tensor(tensor_args.input, operation_attributes.ring_size);
    auto out_ready_semaphore = ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0);
    tt::tt_metal::distributed::Synchronize(*mesh_device, std::nullopt, subdevices);

    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program =
            create_at(operation_attributes, coord, tensor_args, tensor_return_value, scratch, out_ready_semaphore);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

WidthShardedAllReduceMeshWorkloadFactory::cached_program_t WidthShardedAllReduceMeshWorkloadFactory::create_at(
    const WidthShardedAllReduceParams& operation_attributes,
    const ttnn::MeshCoordinate& coord,
    const WidthShardedAllReduceInputs& tensor_args,
    Tensor& output_tensor,
    const std::shared_ptr<Tensor>& scratch,
    const GlobalSemaphore& semaphore) {
    const auto& input_tensor = tensor_args.input;
    const uint32_t device_index =
        ttnn::ccl::get_linearized_index_from_physical_coord(input_tensor, coord, operation_attributes.cluster_axis);
    std::optional<MeshCoordinate> forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor, coord, 1, operation_attributes.topology, operation_attributes.cluster_axis);
    std::optional<MeshCoordinate> backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor, coord, -1, operation_attributes.topology, operation_attributes.cluster_axis);

    const auto& num_links = operation_attributes.num_links;
    const auto& ring_size = operation_attributes.ring_size;
    const auto& topology = operation_attributes.topology;
    const auto& sub_device_id = operation_attributes.sub_device_id;

    Program program{};
    auto* mesh_device = input_tensor.device();

    auto [num_targets_forward, num_targets_backward] =
        ttnn::ccl::get_forward_backward_line_mcast_distance(ring_size, device_index, topology, true);
    auto [forward_args, backward_args] = ttnn::ccl::get_forward_backward_line_mcast_configuration(
        coord, forward_coord, backward_coord, num_targets_forward, num_targets_backward, mesh_device);

    const auto input_tensor_cores = input_tensor.memory_config().shard_spec()->grid;
    const auto input_tensor_shard_shape = input_tensor.memory_config().shard_spec()->shape;
    const auto face = rm_face_tile();
    const auto df = datatype_to_dataformat_converter(input_tensor.dtype());
    const uint32_t face_bytes = face.get_tile_size(df);
    const uint32_t input_tensor_shard_num_pages =
        input_tensor_shard_shape[0] * input_tensor_shard_shape[1] / TILE_WIDTH;
    const auto num_input_cores = input_tensor_cores.num_cores();

    const auto output_tensor_cores = output_tensor.memory_config().shard_spec()->grid;
    const auto output_tensor_shard_shape = output_tensor.memory_config().shard_spec()->shape;
    const auto output_tensor_shard_num_pages = output_tensor_shard_shape[0] * output_tensor_shard_shape[1] / TILE_WIDTH;
    const auto num_output_cores = output_tensor_cores.num_cores();
    const auto output_tensor_num_pages = num_output_cores * output_tensor_shard_num_pages;

    auto sub_device_cores = mesh_device->worker_cores(
        tt::tt_metal::HalProgrammableCoreType::TENSIX, sub_device_id.value_or(mesh_device->get_sub_device_ids().at(0)));
    std::vector<CoreRange> output_cores;
    for (const auto& cr : sub_device_cores.ranges()) {
        const auto intersection = output_tensor_cores.intersection(cr);
        if (!intersection.empty()) {
            output_cores.push_back(intersection.bounding_box());
        }
    }
    CoreRangeSet output_cores_all(output_cores);
    auto available_cores = sub_device_cores.subtract(output_cores_all);
    auto [sender_worker_core_range, sender_worker_cores] = choose_sender_cores(num_links, available_cores);
    auto output_cores_unused = output_cores_all.subtract(output_tensor_cores);
    auto all_cores = output_cores_all.merge(sender_worker_core_range);

    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    uint32_t num_pages_per_packet = std::max<uint32_t>(1, packet_size_bytes / face_bytes);
    uint32_t cb_num_pages = tt::div_up(output_tensor_cores.num_cores(), num_links) * output_tensor_shard_num_pages;
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * face_bytes, {{src0_cb_index, df}})
            .set_page_size(src0_cb_index, face_bytes)
            .set_tile_dims(src0_cb_index, face);
    tt::tt_metal::CreateCircularBuffer(program, sender_worker_core_range, cb_src0_config);

    const auto reserved_packet_header_CB_index = tt::CBIndex::c_3;
    static constexpr auto num_packet_headers_storable = 8;
    auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    tt::tt_metal::CircularBufferConfig cb_reserved_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * 2,
            {{reserved_packet_header_CB_index, tt::DataFormat::RawUInt32}})
            .set_page_size(reserved_packet_header_CB_index, packet_header_size_bytes);
    tt::tt_metal::CreateCircularBuffer(program, sender_worker_core_range, cb_reserved_packet_header_config);

    auto input_cores_vec = corerange_to_cores(input_tensor_cores, std::nullopt, true);
    auto output_cores_vec = corerange_to_cores(output_tensor_cores, std::nullopt, true);

    std::vector<CoreRangeSet> output_corerangeset_per_link;
    output_corerangeset_per_link.reserve(num_links);
    std::vector<uint32_t> num_output_cores_in_link(num_links, 0);
    uint32_t output_cores_per_link = tt::div_up(output_tensor_cores.num_cores(), num_links);
    uint32_t num_assigned_cores = 0;
    for (uint32_t link = 0; link < num_links; link++) {
        uint32_t num_cores_this_link = std::min(output_cores_per_link, num_output_cores - num_assigned_cores);
        output_corerangeset_per_link.emplace_back(
            cores_to_corerangeset(std::vector<CoreCoord>(
                                      output_cores_vec.begin() + num_assigned_cores,
                                      output_cores_vec.begin() + num_assigned_cores + num_cores_this_link))
                .merge_ranges());
        num_output_cores_in_link[link] = num_cores_this_link;
        num_assigned_cores += num_cores_this_link;
    }

    std::vector<uint32_t> output_tensor_pages_in_link;
    output_tensor_pages_in_link.reserve(num_links);
    uint32_t num_assigned_pages = 0;
    for (uint32_t link = 0; link < num_links; link++) {
        uint32_t num_output_pages_per_link = output_tensor_shard_num_pages * num_output_cores_in_link[link];
        uint32_t num_pages_this_link =
            std::min(num_output_pages_per_link, output_tensor_num_pages - num_assigned_pages);
        output_tensor_pages_in_link.push_back(num_pages_this_link);
        num_assigned_pages += num_pages_this_link;
    }

    std::vector<std::pair<uint32_t, uint32_t>> input_cores_idx_per_link(num_links, {0, 0});
    std::vector<uint32_t> input_tensor_tile_offset_per_link;
    input_tensor_tile_offset_per_link.reserve(num_links);
    uint32_t start_core_idx = 0;
    uint32_t num_pages_overflow = 0;
    for (uint32_t link = 0; link < num_links; link++) {
        uint32_t num_pages_this_link = output_tensor_pages_in_link[link];
        uint32_t input_tensor_tile_offset =
            (input_tensor_shard_num_pages - num_pages_overflow) % input_tensor_shard_num_pages;
        input_tensor_tile_offset_per_link.push_back(input_tensor_tile_offset);
        uint32_t end_core_idx = std::min(
            start_core_idx + tt::div_up(num_pages_this_link + input_tensor_tile_offset, input_tensor_shard_num_pages),
            num_input_cores);
        uint32_t num_pages_allocated =
            ((end_core_idx - start_core_idx) * input_tensor_shard_num_pages) - input_tensor_tile_offset;
        num_pages_overflow = num_pages_allocated - num_pages_this_link;
        input_cores_idx_per_link[link] = {start_core_idx, end_core_idx};
        start_core_idx = num_pages_overflow > 0 ? end_core_idx - 1 : end_core_idx;
    }

    std::vector<uint32_t> reduction_semaphore_ids;
    reduction_semaphore_ids.reserve(num_links);
    for (uint32_t link = 0; link < num_links; link++) {
        reduction_semaphore_ids.push_back(tt::tt_metal::CreateSemaphore(program, all_cores, 0));
    }

    uint32_t reduction_CB_tiles = output_tensor_shard_num_pages * ring_size;
    uint32_t reduction_cb_index = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig reduction_cb_config =
        tt::tt_metal::CircularBufferConfig(reduction_CB_tiles * face_bytes, {{reduction_cb_index, df}})
            .set_page_size(reduction_cb_index, face_bytes)
            .set_tile_dims(reduction_cb_index, face)
            .set_globally_allocated_address(*scratch->buffer());
    auto cb_reduction = tt::tt_metal::CreateCircularBuffer(program, all_cores, reduction_cb_config);

    uint32_t out_cb_index = tt::CBIndex::c_2;
    tt::tt_metal::CircularBufferConfig out_cb_config =
        tt::tt_metal::CircularBufferConfig(output_tensor_shard_num_pages * face_bytes, {{out_cb_index, df}})
            .set_page_size(out_cb_index, face_bytes)
            .set_tile_dims(out_cb_index, face)
            .set_globally_allocated_address(*output_tensor.buffer());
    auto cb_out = tt::tt_metal::CreateCircularBuffer(program, output_tensor_cores, out_cb_config);

    auto reduction_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/dataflow/reduction_receiver.cpp",
        output_cores_all,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::NOC_1,
            .compile_args = {reduction_cb_index, reduction_CB_tiles}});
    constexpr bool has_work = true;
    if (!output_cores_unused.empty()) {
        tt::tt_metal::SetRuntimeArgs(program, reduction_reader_kernel_id, output_cores_unused, {!has_work, 0, 0, 0});
    }

    auto reduction_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek/width_sharded_all_reduce/device/kernels/compute/"
        "reduction_rm_face.cpp",
        output_cores_all,
        tt::tt_metal::ComputeConfig{.compile_args = {reduction_cb_index, out_cb_index}});
    tt::tt_metal::SetRuntimeArgs(
        program, reduction_kernel_id, output_tensor_cores, {1, ring_size, output_tensor_shard_num_pages});
    if (!output_cores_unused.empty()) {
        tt::tt_metal::SetRuntimeArgs(program, reduction_kernel_id, output_cores_unused, {!has_work, 0, 0});
    }

    auto worker_sender_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/dataflow/worker_reader.cpp",
        sender_worker_core_range,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::NOC_1,
            .compile_args = {device_index, src0_cb_index, face_bytes}});

    std::vector<uint32_t> writer_compile_args = {
        device_index,
        reserved_packet_header_CB_index,
        num_packet_headers_storable,
        src0_cb_index,
        num_pages_per_packet,
        face_bytes,
        num_targets_forward,
        num_targets_backward,
    };
    writer_compile_args.insert(writer_compile_args.end(), forward_args.begin(), forward_args.end());
    writer_compile_args.insert(writer_compile_args.end(), backward_args.begin(), backward_args.end());
    auto worker_sender_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/dataflow/worker_writer.cpp",
        sender_worker_core_range,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::NOC_0,
            .compile_args = writer_compile_args});

    for (uint32_t link = 0; link < num_links; link++) {
        CoreCoord core = sender_worker_cores[link];
        CoreCoord drain_sync_core = mesh_device->worker_core_from_logical_core(core);
        uint32_t worker_num_tiles_to_read = output_tensor_pages_in_link[link];
        uint32_t input_first_core_tile_start_offset = input_tensor_tile_offset_per_link[link];

        std::vector<uint32_t> input_tensor_cores_x;
        std::vector<uint32_t> input_tensor_cores_y;
        std::vector<uint32_t> output_tensor_cores_x;
        std::vector<uint32_t> output_tensor_cores_y;
        for (uint32_t i = input_cores_idx_per_link[link].first; i < input_cores_idx_per_link[link].second; i++) {
            auto this_core = mesh_device->worker_core_from_logical_core(input_cores_vec[i]);
            input_tensor_cores_x.push_back(this_core.x);
            input_tensor_cores_y.push_back(this_core.y);
        }
        for (uint32_t i = output_cores_per_link * link;
             i < output_cores_per_link * link + num_output_cores_in_link[link];
             i++) {
            auto this_core = mesh_device->worker_core_from_logical_core(output_cores_vec[i]);
            output_tensor_cores_x.push_back(this_core.x);
            output_tensor_cores_y.push_back(this_core.y);
        }

        std::vector<uint32_t> reader_rt_args = {
            input_tensor.buffer()->address(),
            input_tensor_shard_num_pages,
            worker_num_tiles_to_read,
            input_first_core_tile_start_offset,
            static_cast<uint32_t>(input_tensor_cores_x.size()),
        };
        reader_rt_args.insert(reader_rt_args.end(), input_tensor_cores_x.begin(), input_tensor_cores_x.end());
        reader_rt_args.insert(reader_rt_args.end(), input_tensor_cores_y.begin(), input_tensor_cores_y.end());
        tt::tt_metal::SetRuntimeArgs(program, worker_sender_reader_kernel_id, {core}, reader_rt_args);

        std::vector<uint32_t> mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y;
        uint32_t num_mcast_cores = 0;
        for (const auto& range : output_corerangeset_per_link[link].ranges()) {
            auto start_core = mesh_device->worker_core_from_logical_core(range.start_coord);
            auto end_core = mesh_device->worker_core_from_logical_core(range.end_coord);
            num_mcast_cores += (end_core.x - start_core.x + 1) * (end_core.y - start_core.y + 1);
            bool mcast_range_contains_self =
                start_core.x <= core.x && core.x <= end_core.x && start_core.y <= core.y && core.y <= end_core.y;
            if (mcast_range_contains_self) {
                num_mcast_cores -= 1;
            }
            mcast_start_x.push_back(start_core.x);
            mcast_start_y.push_back(start_core.y);
            mcast_end_x.push_back(end_core.x);
            mcast_end_y.push_back(end_core.y);
        }

        std::vector<uint32_t> writer_rt_args = {
            reduction_cb_index,
            semaphore.address(),
            output_tensor_shard_num_pages,
            worker_num_tiles_to_read,
            0,
            static_cast<uint32_t>(output_tensor_cores_x.size()),
            num_mcast_cores,
            drain_sync_core.x,
            drain_sync_core.y,
            ring_size,
            reduction_semaphore_ids[link],
            static_cast<uint32_t>(mcast_start_x.size()),
            link,
        };
        writer_rt_args.insert(writer_rt_args.end(), output_tensor_cores_x.begin(), output_tensor_cores_x.end());
        writer_rt_args.insert(writer_rt_args.end(), output_tensor_cores_y.begin(), output_tensor_cores_y.end());
        writer_rt_args.insert(writer_rt_args.end(), mcast_start_x.begin(), mcast_start_x.end());
        writer_rt_args.insert(writer_rt_args.end(), mcast_start_y.begin(), mcast_start_y.end());
        writer_rt_args.insert(writer_rt_args.end(), mcast_end_x.begin(), mcast_end_x.end());
        writer_rt_args.insert(writer_rt_args.end(), mcast_end_y.begin(), mcast_end_y.end());

        writer_rt_args.push_back(forward_coord.has_value());
        if (forward_coord.has_value()) {
            tt::tt_fabric::append_fabric_connection_rt_args(
                mesh_device->get_fabric_node_id(coord),
                mesh_device->get_fabric_node_id(forward_coord.value()),
                link,
                program,
                core,
                writer_rt_args);
        }
        writer_rt_args.push_back(backward_coord.has_value());
        if (backward_coord.has_value()) {
            tt::tt_fabric::append_fabric_connection_rt_args(
                mesh_device->get_fabric_node_id(coord),
                mesh_device->get_fabric_node_id(backward_coord.value()),
                link,
                program,
                core,
                writer_rt_args);
        }
        tt::tt_metal::SetRuntimeArgs(program, worker_sender_writer_kernel_id, {core}, writer_rt_args);

        std::vector<uint32_t> reduction_reader_rt_args = {
            has_work, reduction_semaphore_ids[link], semaphore.address(), ring_size};
        tt::tt_metal::SetRuntimeArgs(
            program, reduction_reader_kernel_id, output_corerangeset_per_link[link], reduction_reader_rt_args);
    }

    return {
        std::move(program),
        shared_variables_t{
            .worker_sender_reader_kernel_id = worker_sender_reader_kernel_id,
            .worker_sender_writer_kernel_id = worker_sender_writer_kernel_id,
            .reduction_reader_kernel_id = reduction_reader_kernel_id,
            .sender_worker_cores = sender_worker_cores,
            .output_tensor_cores = output_tensor_cores,
            .cb_out = cb_out,
            .cb_reduction = cb_reduction,
            .scratch = scratch,
            .out_ready_semaphore = semaphore,
        }};
}

void WidthShardedAllReduceMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const WidthShardedAllReduceParams&,
    const WidthShardedAllReduceInputs& tensor_args,
    Tensor& output_tensor) {
    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);
        auto& worker_reader_sender_runtime_args_by_core =
            GetRuntimeArgs(program, shared_vars.worker_sender_reader_kernel_id);
        auto& worker_writer_sender_runtime_args_by_core =
            GetRuntimeArgs(program, shared_vars.worker_sender_writer_kernel_id);
        auto& reduction_reader_runtime_args_by_core = GetRuntimeArgs(program, shared_vars.reduction_reader_kernel_id);
        for (const auto& core : shared_vars.sender_worker_cores) {
            worker_reader_sender_runtime_args_by_core[core.x][core.y][0] = tensor_args.input.buffer()->address();
            worker_writer_sender_runtime_args_by_core[core.x][core.y][1] = shared_vars.out_ready_semaphore.address();
        }
        UpdateDynamicCircularBufferAddress(program, shared_vars.cb_out, *output_tensor.buffer());
        UpdateDynamicCircularBufferAddress(program, shared_vars.cb_reduction, *shared_vars.scratch->buffer());
        for (const auto& cr : shared_vars.output_tensor_cores.ranges()) {
            for (const auto& core : corerange_to_cores(cr, std::nullopt, true)) {
                reduction_reader_runtime_args_by_core[core.x][core.y][2] = shared_vars.out_ready_semaphore.address();
            }
        }
    }
}

}  // namespace ttnn::prim
