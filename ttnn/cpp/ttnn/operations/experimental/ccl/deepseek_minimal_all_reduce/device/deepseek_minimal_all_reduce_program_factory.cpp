// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/experimental/ccl/deepseek_minimal_all_reduce/device/deepseek_minimal_all_reduce_device_operation.hpp"
#include "ttnn/operations/experimental/ccl/deepseek_minimal_all_reduce/device/deepseek_minimal_all_reduce_program_factory.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"

#include "ttnn/operations/ccl/common/uops/command_lowering.hpp"
#include "ttnn/global_semaphore.hpp"

#include "ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "ttnn/operations/ccl/common/host/command_backend_runtime_args_overrider.hpp"
#include <sstream>
#include <type_traits>
#include <ranges>
#include <optional>

#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"

#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::experimental::ccl::deepseek_minimal_all_reduce::program {

DeepseekMinimalAllReduceProgramFactory::cached_mesh_workload_t
DeepseekMinimalAllReduceProgramFactory::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    Tensor& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    auto* mesh_device = tensor_args.input_tensor.device();
    const auto available_cores = mesh_device->worker_cores(
        tt::tt_metal::HalProgrammableCoreType::TENSIX, mesh_device->get_sub_device_ids().at(0));

    auto semaphore1 = ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0);
    auto semaphore2 = ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0);
    log_debug(tt::LogOp, "Semaphores allocated and waiting for all devices to be ready");
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, {});
    log_debug(tt::LogOp, "All devices are ready, starting program execution");

    // Get or create intermediate tensor
    TT_FATAL(
        tensor_args.intermediate_tensor.has_value(),
        "Intermediate tensor must be provided for deepseek_minimal_all_reduce");
    const auto& intermediate_tensor = tensor_args.intermediate_tensor.value();

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = DeepseekMinimalAllReduceProgramFactory::create_at(
            operation_attributes, coord, tensor_args, tensor_return_value, intermediate_tensor, semaphore1, semaphore2);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }

    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

DeepseekMinimalAllReduceProgramFactory::cached_program_t DeepseekMinimalAllReduceProgramFactory::create_at(
    const operation_attributes_t& operation_attributes,
    const MeshCoordinate& self_coord,
    const tensor_args_t& tensor_args,
    Tensor& output_tensor,
    const Tensor& intermediate_tensor,
    const tt::tt_metal::GlobalSemaphore& semaphore1,
    const tt::tt_metal::GlobalSemaphore& semaphore2) {
    const auto& input_tensor = tensor_args.input_tensor;
    tt::tt_metal::Program program{};

    // uint32_t num_links = operation_attributes.num_links;
    uint32_t ring_size = operation_attributes.ring_size;
    uint32_t ring_index = ::ttnn::ccl::get_linearized_index_from_physical_coord(
        input_tensor, self_coord, operation_attributes.cluster_axis);
    auto topology = operation_attributes.topology;
    auto cluster_axis = operation_attributes.cluster_axis;

    std::optional<MeshCoordinate> forward_coord =
        ::ttnn::ccl::get_physical_neighbor_from_physical_coord(input_tensor, self_coord, 1, topology, cluster_axis);

    std::optional<MeshCoordinate> backward_coord =
        ::ttnn::ccl::get_physical_neighbor_from_physical_coord(input_tensor, self_coord, -1, topology, cluster_axis);
    TT_FATAL(forward_coord.has_value() || backward_coord.has_value(), "DEBUG: forward_coord or backward_coord is null");

    auto* mesh_device = input_tensor.device();
    [[maybe_unused]] bool is_first_chip = ring_index == 0;
    [[maybe_unused]] bool is_last_chip = ring_index == ring_size - 1;
    log_trace(
        tt::LogOp,
        "DEBUG: device coord: {}, is_first_chip: {}, is_last_chip: {}",
        self_coord,
        is_first_chip,
        is_last_chip);

    // Fatal if input is not tilized with tiny tiles (1, 32)
    bool tilized = input_tensor.layout() == ttnn::TILE_LAYOUT;
    const auto tile_width = input_tensor.tensor_spec().tile().get_width();
    const auto tile_height = input_tensor.tensor_spec().tile().get_height();
    TT_FATAL(
        tilized,
        "deepseek_minimal_all_reduce op currently only supports TILE_LAYOUT input tensors. Got layout: {}",
        input_tensor.layout());
    TT_FATAL(
        tile_width == 32 && tile_height == 1,
        "deepseek_minimal_all_reduce op currently only supports TILE_LAYOUT input tensors with tile size (1, 32). Got "
        "tile size: "
        "({}, {})",
        tile_height,
        tile_width);

    // Extract shard grid from input tensor
    TT_FATAL(input_tensor.is_sharded(), "Input tensor must be sharded");
    const auto& shard_spec = input_tensor.shard_spec().value();
    const auto& shard_grid = shard_spec.grid;

    // Get all cores from the shard grid
    std::vector<CoreCoord> cores;
    for (const auto& core_range : shard_grid.ranges()) {
        auto c = corerange_to_cores(core_range, std::nullopt);
        cores.insert(cores.end(), c.begin(), c.end());
    }
    // data should be on a single core: Fatal if not
    TT_FATAL(cores.size() == 1, "Input tensor must be sharded to a single core");

    // need two worker cores for 2 links
    // For 2-device all-reduce:
    // Link 0 : Device 0 sends → Device 1 receives
    // Link 1 : Device 1 sends → Device 0 receives

    // Divide compute across 28 cores
    constexpr uint32_t num_compute_cores = 28;

    CoreCoord data_core = cores[0];
    std::vector<CoreCoord> compute_cores;
    compute_cores.push_back(data_core);

    // Determine sender_core first so we can exclude it from compute cores
    CoreCoord sender_core;
    if (data_core.y > 0) {
        sender_core = CoreCoord{data_core.x, data_core.y - 1};
    } else {
        sender_core = CoreCoord{data_core.x, data_core.y + 1};
    }

    // pick non-data compute cores to form a contiguous
    // rectangular bounding box that does not include data_core or sender_core.
    constexpr uint32_t grid_width = 7;
    constexpr uint32_t grid_height = 7;

    uint32_t data_row = data_core.y;
    uint32_t sender_row = sender_core.y;
    for (uint32_t row = 0; row < grid_height && compute_cores.size() < num_compute_cores; ++row) {
        if (row == data_row || row == sender_row) {
            continue;
        }
        for (uint32_t col = 0; col < grid_width && compute_cores.size() < num_compute_cores; ++col) {
            compute_cores.push_back(CoreCoord{col, row});
        }
    }

    std::vector<CoreCoord> all_worker_cores;
    all_worker_cores.push_back(sender_core);
    for (const auto& cc : compute_cores) {
        all_worker_cores.push_back(cc);
    }
    auto all_worker_core_range = CoreRangeSet(all_worker_cores);
    auto compute_core_range = CoreRangeSet(compute_cores);

    // Tensor Info - get num_pages early for CB config
    const auto input_tensor_num_pages = input_tensor.buffer()->num_pages();

    // Calculate tile distribution across compute cores
    const uint32_t tiles_per_core = (input_tensor_num_pages + num_compute_cores - 1) / num_compute_cores;

    const auto tiny_tile = tt::tt_metal::Tile({1, 32});

    // L1 Scratch CB Creation
    const uint32_t input_page_size_bytes = input_tensor.buffer()->aligned_page_size();
    const uint32_t l1_alignment = tt::tt_metal::hal::get_l1_alignment();
    const size_t packet_size_bytes = input_page_size_bytes * input_tensor_num_pages;

    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    // Each compute core needs CBs for its tile subset
    // CB sizes are based on tiles_per_core, not full tensor
    const size_t compute_cb_size_bytes = tiles_per_core * input_page_size_bytes;

    constexpr auto compute_cb_in1 = tt::CBIndex::c_1;  // remote data (intermediate tensor)
    constexpr auto compute_cb_in2 = tt::CBIndex::c_2;  // local data (input tensor)
    constexpr auto compute_cb_out = tt::CBIndex::c_3;  // output

    // Data core (compute_cores[0]) uses CB-backed-by-tensor for all three CBs
    tt::tt_metal::CircularBufferConfig compute_cb_in1_config_data_core =
        tt::tt_metal::CircularBufferConfig(compute_cb_size_bytes, {{compute_cb_in1, df}})
            .set_page_size(compute_cb_in1, input_page_size_bytes)
            .set_tile_dims(compute_cb_in1, tiny_tile)
            .set_globally_allocated_address(*intermediate_tensor.buffer());
    auto compute_cb_in1_handle =
        CreateCircularBuffer(program, CoreRangeSet(CoreRange(data_core)), compute_cb_in1_config_data_core);

    tt::tt_metal::CircularBufferConfig compute_cb_in2_config_data_core =
        tt::tt_metal::CircularBufferConfig(compute_cb_size_bytes, {{compute_cb_in2, df}})
            .set_page_size(compute_cb_in2, input_page_size_bytes)
            .set_tile_dims(compute_cb_in2, tiny_tile)
            .set_globally_allocated_address(*input_tensor.buffer());
    auto compute_cb_in2_handle =
        CreateCircularBuffer(program, CoreRangeSet(CoreRange(data_core)), compute_cb_in2_config_data_core);

    tt::tt_metal::CircularBufferConfig compute_cb_out_config_data_core =
        tt::tt_metal::CircularBufferConfig(compute_cb_size_bytes, {{compute_cb_out, df}})
            .set_page_size(compute_cb_out, input_page_size_bytes)
            .set_tile_dims(compute_cb_out, tiny_tile)
            .set_globally_allocated_address(*output_tensor.buffer());
    auto compute_cb_out_handle =
        CreateCircularBuffer(program, CoreRangeSet(CoreRange(data_core)), compute_cb_out_config_data_core);

    std::vector<CoreCoord> non_data_compute_cores(compute_cores.begin() + 1, compute_cores.end());
    if (!non_data_compute_cores.empty()) {
        auto non_data_core_range = CoreRangeSet(non_data_compute_cores);

        tt::tt_metal::CircularBufferConfig compute_cb_in1_config =
            tt::tt_metal::CircularBufferConfig(compute_cb_size_bytes, {{compute_cb_in1, df}})
                .set_page_size(compute_cb_in1, input_page_size_bytes)
                .set_tile_dims(compute_cb_in1, tiny_tile);
        CreateCircularBuffer(program, non_data_core_range, compute_cb_in1_config);

        tt::tt_metal::CircularBufferConfig compute_cb_in2_config =
            tt::tt_metal::CircularBufferConfig(compute_cb_size_bytes, {{compute_cb_in2, df}})
                .set_page_size(compute_cb_in2, input_page_size_bytes)
                .set_tile_dims(compute_cb_in2, tiny_tile);
        CreateCircularBuffer(program, non_data_core_range, compute_cb_in2_config);

        tt::tt_metal::CircularBufferConfig compute_cb_out_config =
            tt::tt_metal::CircularBufferConfig(compute_cb_size_bytes, {{compute_cb_out, df}})
                .set_page_size(compute_cb_out, input_page_size_bytes)
                .set_tile_dims(compute_cb_out, tiny_tile);
        CreateCircularBuffer(program, non_data_core_range, compute_cb_out_config);
    }

    constexpr auto packet_header_cb_id = tt::CBIndex::c_4;
    constexpr auto buffering_factor = 2;
    constexpr auto num_packet_headers_storable = 2;
    const auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    tt::tt_metal::CircularBufferConfig cb_header_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * buffering_factor,
            {{packet_header_cb_id, tt::DataFormat::RawUInt32}})
            .set_page_size(packet_header_cb_id, packet_header_size_bytes)
            .set_tile_dims(packet_header_cb_id, tiny_tile);
    std::vector<CoreCoord> fabric_cores = {sender_core, data_core};
    CreateCircularBuffer(program, CoreRangeSet(fabric_cores), cb_header_config);

    // Packet CB for sender
    constexpr auto packet_cb_id = tt::CBIndex::c_5;
    tt::tt_metal::CircularBufferConfig cb_packet_config =
        tt::tt_metal::CircularBufferConfig(packet_size_bytes, {{packet_cb_id, df}})
            .set_page_size(packet_cb_id, input_page_size_bytes)
            .set_tile_dims(packet_cb_id, tiny_tile);
    CreateCircularBuffer(program, CoreRangeSet(CoreRange(sender_core)), cb_packet_config);

    // Create L1 semaphore on ALL compute cores (data_core + non-data cores) for synchronization
    // receiver_reader on data_core will set locally and mcast to signal all non-data cores when data is ready
    auto compute_sync_semaphore_id = tt::tt_metal::CreateSemaphore(program, CoreRangeSet(compute_cores), 0);

    // Physical core coordinates
    auto data_core_physical = mesh_device->worker_core_from_logical_core(data_core);
    auto data_noc_x = data_core_physical.x;
    auto data_noc_y = data_core_physical.y;

    auto sender_physical = mesh_device->worker_core_from_logical_core(sender_core);

    // Remote cores have same layout (symmetric)
    auto remote_sender_noc_x = sender_physical.x;
    auto remote_sender_noc_y = sender_physical.y;
    auto remote_receiver_noc_x = data_core_physical.x;
    auto remote_receiver_noc_y = data_core_physical.y;

    // Calculate mcast range for non-data compute cores (for signaling)
    // Find the bounding box of non-data compute cores in physical coordinates
    uint32_t mcast_start_x = UINT32_MAX, mcast_start_y = UINT32_MAX;
    uint32_t mcast_end_x = 0, mcast_end_y = 0;
    for (const auto& core : non_data_compute_cores) {
        auto phys = mesh_device->worker_core_from_logical_core(core);
        mcast_start_x = std::min(mcast_start_x, static_cast<uint32_t>(phys.x));
        mcast_start_y = std::min(mcast_start_y, static_cast<uint32_t>(phys.y));
        mcast_end_x = std::max(mcast_end_x, static_cast<uint32_t>(phys.x));
        mcast_end_y = std::max(mcast_end_y, static_cast<uint32_t>(phys.y));
    }
    uint32_t mcast_num_dests = non_data_compute_cores.size();

    // KERNEL CREATION
    // Sender kernels run on sender_core, receiver kernels run on receiver_core
    std::vector<uint32_t> sender_reader_compile_args = {
        packet_cb_id,            // cb0_id
        input_tensor_num_pages,  // num_tiles
        input_page_size_bytes,   // tensor0_page_size
        data_noc_x,
        data_noc_y,
    };

    std::vector<uint32_t> sender_writer_compile_args = {
        packet_header_cb_id,
        packet_cb_id,
        l1_alignment,
        input_tensor_num_pages,
        input_page_size_bytes,
        packet_size_bytes,
        data_noc_x,  // data core for writing payload
        data_noc_y,
        remote_receiver_noc_x,  // remote receiver core for semaphore
        remote_receiver_noc_y,
    };

    std::vector<uint32_t> receiver_reader_compile_args = {
        packet_header_cb_id,
        compute_cb_in1,
        l1_alignment,
        compute_cb_in2,
        tiles_per_core,
        input_page_size_bytes,
        packet_size_bytes,
        data_noc_x,  // data core for reading local tensor
        data_noc_y,
        remote_sender_noc_x,  // remote sender core for semaphore
        remote_sender_noc_y,
        mcast_start_x,  // mcast range for signaling compute cores
        mcast_start_y,
        mcast_end_x,
        mcast_end_y,
        mcast_num_dests,
    };

    // Create sender kernels on sender_core
    auto worker_sender_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_minimal_all_reduce/device/kernels/sender_reader.cpp",
        {sender_core},
        tt::tt_metal::ReaderDataMovementConfig(sender_reader_compile_args));

    auto worker_sender_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_minimal_all_reduce/device/kernels/sender_writer.cpp",
        {sender_core},
        tt::tt_metal::WriterDataMovementConfig(sender_writer_compile_args));

    // Create receiver reader kernel on data_core
    auto worker_receiver_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_minimal_all_reduce/device/kernels/receiver_reader.cpp",
        {data_core},
        tt::tt_metal::ReaderDataMovementConfig(receiver_reader_compile_args));

    auto compute_kernel_configuration = ttnn::init_device_compute_kernel_config(
        input_tensor.device()->arch(), std::nullopt, MathFidelity::HiFi4, true, false, false);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(input_tensor.device()->arch(), compute_kernel_configuration);

    // Create compute kernels for all compute cores
    // Each core processes a subset of tiles
    std::vector<tt::tt_metal::KernelHandle> compute_reader_kernel_ids;
    std::vector<tt::tt_metal::KernelHandle> compute_writer_kernel_ids;

    for (uint32_t core_idx = 0; core_idx < num_compute_cores; ++core_idx) {
        const auto& compute_core = compute_cores[core_idx];
        uint32_t tile_offset = core_idx * tiles_per_core;
        uint32_t num_tiles_this_core = std::min(tiles_per_core, input_tensor_num_pages - tile_offset);

        if (num_tiles_this_core == 0) {
            break;
        }

        std::vector<uint32_t> compute_ct_args = {
            compute_cb_in1, compute_cb_in2, compute_cb_out, 1, num_tiles_this_core};

        tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_minimal_all_reduce/device/kernels/reduction.cpp",
            {compute_core},
            tt::tt_metal::ComputeConfig{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = true,
                .math_approx_mode = math_approx_mode,
                .compile_args = compute_ct_args,
            });

        if (core_idx == 0) {
            // Data core: receiver_reader handles data movement, no separate reader/writer needed
            continue;
        }

        std::vector<uint32_t> compute_reader_compile_args = {
            compute_cb_in1,       // cb_in1
            compute_cb_in2,       // cb_in2
            num_tiles_this_core,  // num_tiles
            tile_offset,          // tile_offset
            input_page_size_bytes,
            data_noc_x,
            data_noc_y,
        };

        auto compute_reader_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_minimal_all_reduce/device/kernels/compute_reader.cpp",
            {compute_core},
            tt::tt_metal::ReaderDataMovementConfig(compute_reader_compile_args));
        compute_reader_kernel_ids.push_back(compute_reader_kernel_id);

        std::vector<uint32_t> compute_writer_compile_args = {
            compute_cb_out,       // cb_out
            num_tiles_this_core,  // num_tiles
            tile_offset,          // tile_offset
            input_page_size_bytes,
            data_noc_x,
            data_noc_y,
        };

        auto compute_writer_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_minimal_all_reduce/device/kernels/compute_writer.cpp",
            {compute_core},
            tt::tt_metal::WriterDataMovementConfig(compute_writer_compile_args));
        compute_writer_kernel_ids.push_back(compute_writer_kernel_id);
    }

    // Kernel Runtime Args
    // For 2-device all-reduce:
    // - Device 0 sends forward (link 0) using semaphore1, Device 1 receives
    // - Device 1 sends backward (link 1) using semaphore2, Device 0 receives
    //
    // Sender writes to intermediate_tensor on the receiver's device
    // Receiver reads from intermediate_tensor after semaphore signals data arrived

    const auto self_fabric_node_id = mesh_device->get_fabric_node_id(self_coord);

    const uint32_t sender_link = is_first_chip ? 0 : 1;
    const uint32_t receiver_link = is_first_chip ? 1 : 0;

    const auto& sender_semaphore = is_first_chip ? semaphore1 : semaphore2;
    const auto& receiver_semaphore = is_first_chip ? semaphore2 : semaphore1;

    TT_FATAL(
        (is_first_chip && forward_coord.has_value()) || (!is_first_chip && backward_coord.has_value()),
        "Missing expected neighbor coordinate");
    const auto neighbor_coord = is_first_chip ? forward_coord.value() : backward_coord.value();
    const auto neighbor_fabric_node_id = mesh_device->get_fabric_node_id(neighbor_coord);

    std::vector<uint32_t> sender_reader_rt_args = {
        input_tensor.buffer()->address(),  // tensor_address0
    };
    tt::tt_metal::SetRuntimeArgs(program, worker_sender_reader_kernel_id, {sender_core}, sender_reader_rt_args);

    // Sender writer sends data to receiver's intermediate buffer
    std::vector<uint32_t> sender_writer_rt_args = {
        intermediate_tensor.buffer()->address(),
        sender_semaphore.address(),
    };
    append_routing_plane_connection_manager_rt_args(
        self_fabric_node_id,
        {neighbor_fabric_node_id},
        {sender_link},
        program,
        worker_sender_writer_kernel_id,
        sender_core,
        sender_writer_rt_args);
    tt::tt_metal::SetRuntimeArgs(program, worker_sender_writer_kernel_id, {sender_core}, sender_writer_rt_args);

    // === RECEIVER RUNTIME ARGS ===
    std::vector<uint32_t> receiver_reader_rt_args = {
        input_tensor.buffer()->address(),
        intermediate_tensor.buffer()->address(),
        receiver_semaphore.address(),
        compute_sync_semaphore_id,
    };
    append_routing_plane_connection_manager_rt_args(
        self_fabric_node_id,
        {neighbor_fabric_node_id},
        {receiver_link},
        program,
        worker_receiver_reader_kernel_id,
        data_core,
        receiver_reader_rt_args);
    tt::tt_metal::SetRuntimeArgs(program, worker_receiver_reader_kernel_id, {data_core}, receiver_reader_rt_args);
    for (size_t i = 0; i < compute_reader_kernel_ids.size(); ++i) {
        const auto& compute_core = compute_cores[i + 1];  // Skip data_core (index 0)

        std::vector<uint32_t> compute_reader_rt_args = {
            input_tensor.buffer()->address(),
            intermediate_tensor.buffer()->address(),
            compute_sync_semaphore_id,
        };
        tt::tt_metal::SetRuntimeArgs(program, compute_reader_kernel_ids[i], {compute_core}, compute_reader_rt_args);

        std::vector<uint32_t> compute_writer_rt_args = {
            output_tensor.buffer()->address(),
        };
        tt::tt_metal::SetRuntimeArgs(program, compute_writer_kernel_ids[i], {compute_core}, compute_writer_rt_args);
    }

    shared_variables_t shared_variables{
        .sender_worker_cores = {sender_core},
        .receiver_worker_cores = {data_core},
        .compute_worker_cores = compute_cores,
        .worker_sender_reader_kernel_id = worker_sender_reader_kernel_id,
        .worker_sender_writer_kernel_id = worker_sender_writer_kernel_id,
        .worker_receiver_reader_kernel_id = worker_receiver_reader_kernel_id,
        .compute_reader_kernel_ids = std::move(compute_reader_kernel_ids),
        .compute_writer_kernel_ids = std::move(compute_writer_kernel_ids),
        .compute_cb_in1_handle = compute_cb_in1_handle,
        .compute_cb_in2_handle = compute_cb_in2_handle,
        .compute_cb_out_handle = compute_cb_out_handle,
        .semaphore1 = semaphore1,
        .semaphore2 = semaphore2,
        .ring_index = ring_index,
    };
    return {std::move(program), std::move(shared_variables)};
}

void DeepseekMinimalAllReduceProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    Tensor& tensor_return_value) {
    const auto& input = tensor_args.input_tensor;
    const auto& output = tensor_return_value;

    TT_FATAL(
        tensor_args.intermediate_tensor.has_value(),
        "Intermediate tensor must be provided for deepseek_minimal_all_reduce");
    const auto& intermediate = tensor_args.intermediate_tensor.value();

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);

        log_trace(tt::LogOp, "DEBUG: semaphore1: {}", shared_vars.semaphore1.address());
        log_trace(tt::LogOp, "DEBUG: semaphore2: {}", shared_vars.semaphore2.address());

        bool is_first_chip = shared_vars.ring_index == 0;
        const auto& sender_semaphore = is_first_chip ? shared_vars.semaphore1 : shared_vars.semaphore2;
        const auto& receiver_semaphore = is_first_chip ? shared_vars.semaphore2 : shared_vars.semaphore1;

        // Update CB-backed-by-tensor addresses
        UpdateDynamicCircularBufferAddress(program, shared_vars.compute_cb_in1_handle, *intermediate.buffer());
        UpdateDynamicCircularBufferAddress(program, shared_vars.compute_cb_in2_handle, *input.buffer());
        UpdateDynamicCircularBufferAddress(program, shared_vars.compute_cb_out_handle, *output.buffer());

        for (const auto& core : shared_vars.sender_worker_cores) {
            // Sender reader
            auto& reader_args = GetRuntimeArgs(program, shared_vars.worker_sender_reader_kernel_id)[core.x][core.y];
            reader_args[0] = input.buffer()->address();

            // Sender writer
            auto& writer_args = GetRuntimeArgs(program, shared_vars.worker_sender_writer_kernel_id)[core.x][core.y];
            writer_args[0] = intermediate.buffer()->address();
            writer_args[1] = sender_semaphore.address();
        }

        for (const auto& core : shared_vars.receiver_worker_cores) {
            // Receiver reader
            auto& reader_args = GetRuntimeArgs(program, shared_vars.worker_receiver_reader_kernel_id)[core.x][core.y];
            reader_args[0] = input.buffer()->address();
            reader_args[1] = intermediate.buffer()->address();
            reader_args[2] = receiver_semaphore.address();
        }

        // Update compute reader/writer runtime args for non-data cores
        for (size_t i = 0; i < shared_vars.compute_reader_kernel_ids.size(); ++i) {
            const auto& core = shared_vars.compute_worker_cores[i + 1];
            auto& reader_args = GetRuntimeArgs(program, shared_vars.compute_reader_kernel_ids[i])[core.x][core.y];
            reader_args[0] = input.buffer()->address();
            reader_args[1] = intermediate.buffer()->address();

            auto& writer_args = GetRuntimeArgs(program, shared_vars.compute_writer_kernel_ids[i])[core.x][core.y];
            writer_args[0] = output.buffer()->address();
        }
    }
}
}  // namespace ttnn::operations::experimental::ccl::deepseek_minimal_all_reduce::program
