// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "combine_device_operation.hpp"
#include <algorithm>
#include <array>
#include <utility>
#include <limits>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <ttnn/global_semaphore.hpp>
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::combine {

namespace detail {

uint32_t get_num_pages(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->num_pages(); }

uint32_t get_page_size(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->page_size(); }

uint32_t get_aligned_page_size(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->aligned_page_size(); }

uint32_t get_num_rows(const ttnn::Tensor& tensor) {
    auto logical_volume = tensor.logical_shape().volume();
    auto hidden_size = tensor.logical_shape()[-1];
    TT_FATAL(logical_volume % hidden_size == 0, "Logical volume must be divisible by hidden size");
    return logical_volume / hidden_size;
}

void create_tensor_cb(
    tt::tt_metal::Program& program,
    const CoreRangeSet& core_range_set,
    const ttnn::Tensor& tensor,
    uint32_t buffering_factor,
    tt::CBIndex cb_id,
    const std::string& tensor_name = "tensor") {
    auto page_size = get_page_size(tensor);
    auto num_pages = get_num_pages(tensor);
    auto aligned_page_size = get_aligned_page_size(tensor);
    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor.dtype());

    uint32_t cb_size = buffering_factor * aligned_page_size;

    log_debug(
        tt::LogOp,
        "{} shape: {}, pages: {}, page_size: {}, aligned_page_size: {} buffering_factor: {} cb_id: {} cb_size: {} "
        "cb_dtype: {}",
        tensor_name,
        tensor.logical_shape(),
        num_pages,
        page_size,
        aligned_page_size,
        buffering_factor,
        cb_id,
        cb_size,
        data_format);

    tt::tt_metal::CircularBufferConfig cb_config =
        tt::tt_metal::CircularBufferConfig(cb_size, {{cb_id, data_format}}).set_page_size(cb_id, aligned_page_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range_set, cb_config);
}

}  // namespace detail

CombineProgramFactory::cached_mesh_workload_t CombineProgramFactory::create_mesh_workload(
    const CombineParams& operation_attributes,
    const MeshCoordinateRangeSet& tensor_coords,
    const CombineInputs& tensor_args,
    ttnn::Tensor& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, CombineSharedVariables> shared_variables;

    auto* mesh_device = tensor_args.dispatched_buffer.device();

    auto init_barrier_semaphore =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, operation_attributes.worker_core_range_set, 0);
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, {});

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(
            operation_attributes, coord, tensor_args, tensor_return_value, tensor_coords, init_barrier_semaphore);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(coord, std::move(cached_program.shared_variables));
    }
    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

ttnn::device_operation::CachedProgram<CombineSharedVariables> CombineProgramFactory::create_at(
    const CombineParams& operation_attributes,
    const MeshCoordinate& mesh_coordinate,
    const CombineInputs& tensor_args,
    ttnn::Tensor& tensor_return_value,
    const MeshCoordinateRangeSet& tensor_coords,
    const GlobalSemaphore& init_semaphore) {
    tt::tt_metal::Program program{};

    const auto& dispatched_buffer = tensor_args.dispatched_buffer;
    const auto& dispatched_metadata = tensor_args.dispatched_metadata;
    const auto& expert_token_counts = tensor_args.expert_token_counts;
    const auto& output_tensor = tensor_return_value;

    auto* mesh_device = dispatched_buffer.device();
    auto worker_core_range_set = operation_attributes.worker_core_range_set;

    const auto& mesh_view = mesh_device->get_view();
    auto src_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
    uint32_t src_mesh_id = *src_fabric_node_id.mesh_id;
    uint32_t src_chip_id = (uint32_t)src_fabric_node_id.chip_id;
    uint32_t linearized_mesh_coord = ccl::common::get_linearized_index(mesh_coordinate, mesh_view);
    uint32_t mesh_rows = mesh_view.num_rows();
    uint32_t mesh_cols = mesh_view.num_cols();

    auto num_links = operation_attributes.num_links;
    auto topology = operation_attributes.topology;

    log_debug(
        tt::LogOp,
        "Creating prefill combine program for mesh coordinate: ({}, {}) with mesh id: {} "
        "chip id: {} linearized: {} mesh shape: ({}, {}) topology: {} num_links: {}",
        mesh_coordinate[0],
        mesh_coordinate[1],
        src_mesh_id,
        src_chip_id,
        linearized_mesh_coord,
        mesh_rows,
        mesh_cols,
        topology,
        num_links);

    auto fabric_max_packet_size = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
    auto l1_alignment = tt::tt_metal::hal::get_l1_alignment();

    const auto [neighbors, directions] =
        ccl::common::get_neighbors(mesh_view, mesh_coordinate, topology, operation_attributes.axis);

    auto dispatched_shape = dispatched_buffer.logical_shape();
    auto hidden_size = dispatched_shape[-1];
    auto max_dispatched_tokens_per_expert = dispatched_shape[-2];

    auto subdevice_cores = corerange_to_cores(worker_core_range_set);
    uint32_t effective_num_links = std::min(num_links, 4u);
    TT_FATAL(
        subdevice_cores.size() >= effective_num_links,
        "Not enough cores {} for {} links",
        subdevice_cores.size(),
        effective_num_links);

    uint32_t num_cores = effective_num_links;
    uint32_t experts_per_core_range = tt::div_up(operation_attributes.experts_per_chip, num_cores);

    auto sender_core_grid = tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(
        subdevice_cores.at(0), num_cores, worker_core_range_set, true);
    std::vector<CoreCoord> sender_cores = corerange_to_cores(sender_core_grid);

    log_debug(
        tt::LogOp,
        "Combine program: hidden_size: {} num_cores: {} experts_per_core_range: {} cores: {}",
        hidden_size,
        num_cores,
        experts_per_core_range,
        sender_cores);

    auto zero_init_semaphore_id = tt::tt_metal::CreateSemaphore(program, sender_core_grid, 0);
    auto zero_init_barrier_semaphore_id = tt::tt_metal::CreateSemaphore(program, sender_core_grid, 0);

    constexpr uint32_t read_batch_size = 8;

    // c_0: dispatched_buffer scratch (reader-only, batched DRAM reads)
    detail::create_tensor_cb(
        program,
        sender_core_grid,
        dispatched_buffer,
        /*buffering_factor=*/read_batch_size,
        /*cb_id=*/tt::CBIndex::c_0,
        "dispatched_buffer_scratch");
    // c_1: dispatched_metadata scratch (reader-only, batched DRAM reads)
    detail::create_tensor_cb(
        program,
        sender_core_grid,
        dispatched_metadata,
        /*buffering_factor=*/read_batch_size,
        /*cb_id=*/tt::CBIndex::c_1,
        "dispatched_metadata_scratch");
    // c_2: expert_token_counts (reader-only, full tensor)
    detail::create_tensor_cb(
        program,
        sender_core_grid,
        expert_token_counts,
        /*buffering_factor=*/detail::get_num_pages(expert_token_counts),
        /*cb_id=*/tt::CBIndex::c_2,
        "expert_token_counts");

    // c_3: route_info (reader->writer, 4 x uint32_t per entry)
    {
        uint32_t route_info_page_size = l1_alignment;
        constexpr uint32_t route_info_buffering = 16;
        tt::tt_metal::CircularBufferConfig route_info_cb_config =
            tt::tt_metal::CircularBufferConfig(
                route_info_buffering * route_info_page_size, {{tt::CBIndex::c_3, tt::DataFormat::UInt8}})
                .set_page_size(tt::CBIndex::c_3, route_info_page_size);
        tt::tt_metal::CreateCircularBuffer(program, sender_core_grid, route_info_cb_config);
    }

    // c_4: output_for_writer (reader->writer, output pages for fabric sends)
    detail::create_tensor_cb(
        program,
        sender_core_grid,
        dispatched_buffer,
        /*buffering_factor=*/16,
        /*cb_id=*/tt::CBIndex::c_4,
        "output_for_writer");

    // c_5: packet header CB for fabric sends (writer-only)
    if (num_links > 0) {
        constexpr uint32_t num_packet_headers = 2;
        auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
        uint32_t packet_header_cb_size = num_packet_headers * packet_header_size_bytes;

        tt::tt_metal::CircularBufferConfig packet_header_cb_config =
            tt::tt_metal::CircularBufferConfig(packet_header_cb_size, {{tt::CBIndex::c_5, tt::DataFormat::UInt8}})
                .set_page_size(tt::CBIndex::c_5, packet_header_size_bytes);
        tt::tt_metal::CreateCircularBuffer(program, sender_core_grid, packet_header_cb_config);
    }

    std::vector<uint32_t> dest_mesh_id, dest_chip_id;
    for (const auto& coord : tensor_coords.coords()) {
        auto dest_fabric_node_id = mesh_device->get_fabric_node_id(coord);
        dest_mesh_id.push_back(*dest_fabric_node_id.mesh_id);
        dest_chip_id.push_back((uint32_t)dest_fabric_node_id.chip_id);
    }

    // Compile-time args shared by reader and writer
    std::vector<uint32_t> compile_time_args = {
        // CB IDs (6)
        static_cast<uint32_t>(tt::CBIndex::c_0),  // cb_dispatched_buffer_id
        static_cast<uint32_t>(tt::CBIndex::c_1),  // cb_dispatched_metadata_id
        static_cast<uint32_t>(tt::CBIndex::c_2),  // cb_experts_tok_counter_id
        static_cast<uint32_t>(tt::CBIndex::c_3),  // cb_route_info_id
        static_cast<uint32_t>(tt::CBIndex::c_4),  // cb_output_for_writer_id
        static_cast<uint32_t>(tt::CBIndex::c_5),  // cb_packet_header_id

        // Page counts (4)
        detail::get_num_pages(dispatched_buffer),
        detail::get_num_pages(dispatched_metadata),
        detail::get_num_pages(expert_token_counts),
        detail::get_num_pages(output_tensor),

        // Page sizes (4)
        detail::get_page_size(dispatched_buffer),
        detail::get_page_size(dispatched_metadata),
        detail::get_page_size(expert_token_counts),
        detail::get_page_size(output_tensor),

        // Operation parameters (5)
        operation_attributes.dispatch_group_size,
        operation_attributes.experts_per_chip,
        operation_attributes.num_experts_per_tok,
        operation_attributes.seq_len_per_chip,
        (uint32_t)max_dispatched_tokens_per_expert,

        // Hidden dimension
        (uint32_t)hidden_size,

        // Aligned page sizes (4)
        detail::get_aligned_page_size(dispatched_buffer),
        detail::get_aligned_page_size(dispatched_metadata),
        detail::get_aligned_page_size(expert_token_counts),
        detail::get_aligned_page_size(output_tensor),

        // Mesh information (5)
        src_mesh_id,
        src_chip_id,
        mesh_rows,
        mesh_cols,
        linearized_mesh_coord,

        // Fabric configuration (4)
        (uint32_t)fabric_max_packet_size,
        l1_alignment,
        static_cast<uint32_t>(num_links),
        static_cast<uint32_t>(topology),
    };

    // Append TensorAccessorArgs for all 4 tensors
    tt::tt_metal::TensorAccessorArgs(dispatched_buffer.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(dispatched_metadata.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(expert_token_counts.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(compile_time_args);

    // Both reader and writer get fabric defines so the reader can compute routes
    std::map<std::string, std::string> fabric_defines;
    if (num_links > 0) {
        fabric_defines["DEST_CHIP_ID"] = ccl::common::stringify(dest_chip_id);
        fabric_defines["DEST_MESH_ID"] = ccl::common::stringify(dest_mesh_id);
        fabric_defines["DIRECTIONS"] = ccl::common::stringify(directions);
    }
    if (operation_attributes.axis.has_value()) {
        fabric_defines["AXIS"] = std::to_string(operation_attributes.axis.value());
    }

    std::map<std::string, std::string> reader_defines = fabric_defines;
    reader_defines["INIT_ZEROS"] = operation_attributes.init_zeros ? "1" : "0";

    const bool init_zeros = operation_attributes.init_zeros;
    tt::tt_metal::KernelHandle zero_init_kernel_id = 0;
    std::vector<CoreCoord> zero_init_cores_vec;
    uint32_t zi_done_semaphore_id = 0;
    uint32_t num_zero_init_cores = 0;
    uint32_t total_zero_init_cores = 0;
    uint32_t pages_per_core = 0;
    uint32_t remainder_pages = 0;

    std::map<std::string, std::string> writer_defines = fabric_defines;

    if (init_zeros) {
        uint32_t noc_max_burst_size;
        const auto arch = mesh_device->arch();
        if (arch == tt::ARCH::BLACKHOLE) {
            noc_max_burst_size = 16384;
        } else if (arch == tt::ARCH::WORMHOLE_B0) {
            noc_max_burst_size = 8192;
        } else {
            TT_THROW("Unsupported architecture for zero-init: {}", arch);
        }

        tt::tt_metal::CircularBufferConfig zi_inline_cb_config =
            tt::tt_metal::CircularBufferConfig(noc_max_burst_size, {{tt::CBIndex::c_7, tt::DataFormat::UInt8}})
                .set_page_size(tt::CBIndex::c_7, noc_max_burst_size);
        tt::tt_metal::CreateCircularBuffer(program, sender_core_grid, zi_inline_cb_config);

        // Find idle worker cores in the same row as sender cores
        uint32_t sender_row_y = sender_cores[0].y;
        std::set<CoreCoord> sender_core_set(sender_cores.begin(), sender_cores.end());
        std::vector<CoreCoord> idle_row_cores;
        for (const auto& core : subdevice_cores) {
            if (core.y == sender_row_y && !sender_core_set.contains(core)) {
                idle_row_cores.push_back(core);
            }
        }

        num_zero_init_cores = idle_row_cores.size();
        TT_FATAL(
            num_zero_init_cores > 0,
            "No idle cores found in sender row {} for zero-init; subdevice must have more than {} cores per row",
            sender_row_y,
            num_cores);
        total_zero_init_cores = num_cores + num_zero_init_cores;

        uint32_t total_output_pages = detail::get_num_pages(output_tensor);
        pages_per_core = total_output_pages / total_zero_init_cores;
        remainder_pages = total_output_pages % total_zero_init_cores;

        std::set<CoreRange> idle_ranges;
        for (const auto& core : idle_row_cores) {
            idle_ranges.insert(CoreRange(core));
        }
        CoreRangeSet idle_core_grid(idle_ranges);

        tt::tt_metal::CircularBufferConfig zi_idle_cb_config =
            tt::tt_metal::CircularBufferConfig(noc_max_burst_size, {{tt::CBIndex::c_6, tt::DataFormat::UInt8}})
                .set_page_size(tt::CBIndex::c_6, noc_max_burst_size);
        tt::tt_metal::CreateCircularBuffer(program, idle_core_grid, zi_idle_cb_config);

        zi_done_semaphore_id = tt::tt_metal::CreateSemaphore(program, worker_core_range_set, 0);

        uint32_t output_aligned_page_size = detail::get_aligned_page_size(output_tensor);
        std::vector<uint32_t> zi_compile_time_args = {
            output_aligned_page_size,
            num_cores,
            static_cast<uint32_t>(tt::CBIndex::c_6),
        };
        tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(zi_compile_time_args);

        zero_init_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/"
            "zero_init_writer.cpp",
            idle_core_grid,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::detail::preferred_noc_for_dram_write(mesh_device->arch()),
                .compile_args = zi_compile_time_args});

        zero_init_cores_vec = idle_row_cores;
    }

    // Reader gets its own compile-time args: shared base + zero-init args appended at the end
    std::vector<uint32_t> reader_compile_time_args = compile_time_args;
    if (init_zeros) {
        reader_compile_time_args.push_back(static_cast<uint32_t>(tt::CBIndex::c_7));  // zi_cb_id
        reader_compile_time_args.push_back(num_zero_init_cores);                      // num_idle_cores
    }

    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/reader_combine.cpp",
        sender_core_grid,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::detail::preferred_noc_for_dram_read(mesh_device->arch()),
            .compile_args = reader_compile_time_args,
            .defines = reader_defines});

    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/writer_combine.cpp",
        sender_core_grid,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::detail::preferred_noc_for_dram_write(mesh_device->arch()),
            .compile_args = compile_time_args,
            .defines = writer_defines});

    // Pre-compute NOC coordinates for all sender cores (for inter-core barrier signaling)
    std::vector<std::pair<uint32_t, uint32_t>> sender_noc_coords;
    for (const auto& sc : sender_cores) {
        auto noc_coord = mesh_device->virtual_core_from_logical_core(sc, tt::CoreType::WORKER);
        sender_noc_coords.emplace_back(noc_coord.x, noc_coord.y);
    }

    // Set runtime args for hybrid idle row cores
    if (init_zeros) {
        for (uint32_t idle_idx = 0; idle_idx < num_zero_init_cores; idle_idx++) {
            uint32_t row_idx = num_cores + idle_idx;
            uint32_t page_start = (row_idx * pages_per_core) + std::min(row_idx, remainder_pages);
            uint32_t page_end = page_start + pages_per_core + (row_idx < remainder_pages ? 1 : 0);

            std::vector<uint32_t> zi_runtime_args = {
                output_tensor.buffer()->address(),
                page_start,
                page_end,
                zi_done_semaphore_id,
            };
            for (const auto& [noc_x, noc_y] : sender_noc_coords) {
                zi_runtime_args.push_back(noc_x);
                zi_runtime_args.push_back(noc_y);
            }

            tt::tt_metal::SetRuntimeArgs(program, zero_init_kernel_id, zero_init_cores_vec[idle_idx], zi_runtime_args);
        }
    }

    uint32_t core_idx = 0;
    for (const auto& sender_core : sender_cores) {
        uint32_t expert_start = core_idx * experts_per_core_range;
        uint32_t expert_end = std::min((core_idx + 1) * experts_per_core_range, operation_attributes.experts_per_chip);

        std::vector<uint32_t> reader_runtime_args = {
            dispatched_buffer.buffer()->address(),
            dispatched_metadata.buffer()->address(),
            expert_token_counts.buffer()->address(),
            output_tensor.buffer()->address(),
            zero_init_semaphore_id,
            zero_init_barrier_semaphore_id,
            num_cores,
            expert_start,
            expert_end,
        };
        if (init_zeros) {
            uint32_t sender_page_start = (core_idx * pages_per_core) + std::min(core_idx, remainder_pages);
            uint32_t sender_page_end = sender_page_start + pages_per_core + (core_idx < remainder_pages ? 1 : 0);
            reader_runtime_args.push_back(sender_page_start);
            reader_runtime_args.push_back(sender_page_end);
            reader_runtime_args.push_back(zi_done_semaphore_id);
        }

        std::vector<uint32_t> writer_runtime_args = {
            dispatched_buffer.buffer()->address(),
            dispatched_metadata.buffer()->address(),
            expert_token_counts.buffer()->address(),
            output_tensor.buffer()->address(),
            zero_init_semaphore_id,
            (uint32_t)init_semaphore.address(),
            zero_init_barrier_semaphore_id,
            num_cores,
            expert_start,
            expert_end,
        };

        // Append NOC coordinates of all cores for inter-core barrier signaling
        for (const auto& [noc_x, noc_y] : sender_noc_coords) {
            writer_runtime_args.push_back(noc_x);
            writer_runtime_args.push_back(noc_y);
        }

        if (num_links > 0) {
            uint32_t core_link = core_idx % num_links;
            for (const auto& neighbor_coordinate : neighbors) {
                if (neighbor_coordinate[0] == mesh_coordinate[0] && neighbor_coordinate[1] == mesh_coordinate[1]) {
                    continue;
                }

                log_debug(
                    tt::LogOp,
                    "Combine connection: ({}, {}) -> ({}, {}) core {} link {} experts [{}, {})",
                    mesh_coordinate[0],
                    mesh_coordinate[1],
                    neighbor_coordinate[0],
                    neighbor_coordinate[1],
                    sender_core,
                    core_link,
                    expert_start,
                    expert_end);

                tt::tt_fabric::append_fabric_connection_rt_args(
                    src_fabric_node_id,
                    mesh_device->get_fabric_node_id(neighbor_coordinate),
                    core_link,
                    program,
                    sender_core,
                    writer_runtime_args);
            }
        }

        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, sender_core, reader_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, sender_core, writer_runtime_args);
        core_idx++;
    }

    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .zero_init_kernel_id = zero_init_kernel_id,
         .cores = sender_cores,
         .zero_init_cores = zero_init_cores_vec,
         .init_semaphore = init_semaphore,
         .zero_init_semaphore_id = zero_init_semaphore_id,
         .zero_init_barrier_semaphore_id = zero_init_barrier_semaphore_id}};
}

void CombineProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const CombineParams& /*operation_attributes*/,
    const CombineInputs& tensor_args,
    ttnn::Tensor& tensor_return_value) {
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& shared_variables = cached_workload.shared_variables.at(range);

        for (const auto& core : shared_variables.cores) {
            auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, shared_variables.reader_kernel_id, core);
            auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, shared_variables.writer_kernel_id, core);

            reader_runtime_args.at(0) = tensor_args.dispatched_buffer.buffer()->address();
            reader_runtime_args.at(1) = tensor_args.dispatched_metadata.buffer()->address();
            reader_runtime_args.at(2) = tensor_args.expert_token_counts.buffer()->address();
            reader_runtime_args.at(3) = tensor_return_value.buffer()->address();

            writer_runtime_args.at(0) = tensor_args.dispatched_buffer.buffer()->address();
            writer_runtime_args.at(1) = tensor_args.dispatched_metadata.buffer()->address();
            writer_runtime_args.at(2) = tensor_args.expert_token_counts.buffer()->address();
            writer_runtime_args.at(3) = tensor_return_value.buffer()->address();
            writer_runtime_args.at(5) = (uint32_t)shared_variables.init_semaphore.address();
        }

        for (const auto& core : shared_variables.zero_init_cores) {
            auto& zi_runtime_args = tt::tt_metal::GetRuntimeArgs(program, shared_variables.zero_init_kernel_id, core);
            zi_runtime_args.at(0) = tensor_return_value.buffer()->address();
        }
    }
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine
