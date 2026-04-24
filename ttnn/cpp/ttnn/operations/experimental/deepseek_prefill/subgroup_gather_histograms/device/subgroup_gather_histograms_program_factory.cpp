// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "subgroup_gather_histograms_device_operation.hpp"

#include <algorithm>
#include <array>
#include <map>
#include <string>
#include <utility>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <ttnn/global_semaphore.hpp>

#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::subgroup_gather_histograms {

namespace detail {

uint32_t get_num_pages(const ttnn::Tensor& t) { return (uint32_t)t.buffer()->num_pages(); }
uint32_t get_page_size(const ttnn::Tensor& t) { return (uint32_t)t.buffer()->page_size(); }
uint32_t get_aligned_page_size(const ttnn::Tensor& t) { return (uint32_t)t.buffer()->aligned_page_size(); }

}  // namespace detail

SubgroupGatherHistogramsProgramFactory::cached_mesh_workload_t
SubgroupGatherHistogramsProgramFactory::create_mesh_workload(
    const SubgroupGatherHistogramsParams& operation_attributes,
    const ttnn::distributed::MeshCoordinateRangeSet& tensor_coords,
    const SubgroupGatherHistogramsInputs& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, SubgroupGatherHistogramsSharedVariables> shared_variables;

    auto* mesh_device = tensor_args.input_tensor.device();

    const auto subgroups = ccl::common::split_into_subgroups(
        tensor_coords, operation_attributes.cluster_axis, operation_attributes.num_dispatch_subgroups);

    for (const auto& subgroup_range : subgroups) {
        auto init_semaphore = ttnn::global_semaphore::create_global_semaphore(
            mesh_device, operation_attributes.worker_core_range_set, 0, tt::tt_metal::BufferType::L1);
        tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, {});

        for (const auto& coord : subgroup_range) {
            auto cached_program = create_at(
                operation_attributes, coord, tensor_args, tensor_return_value, subgroup_range, init_semaphore);
            workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
            shared_variables.emplace(coord, std::move(cached_program.shared_variables));
        }
    }

    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

ttnn::device_operation::CachedProgram<SubgroupGatherHistogramsSharedVariables>
SubgroupGatherHistogramsProgramFactory::create_at(
    const SubgroupGatherHistogramsParams& operation_attributes,
    const ttnn::distributed::MeshCoordinate& mesh_coordinate,
    const SubgroupGatherHistogramsInputs& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const ttnn::distributed::MeshCoordinateRange& subgroup_range,
    const GlobalSemaphore& init_semaphore) {
    tt::tt_metal::Program program{};

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& output_tensor = tensor_return_value;
    auto* mesh_device = input_tensor.device();

    auto src_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
    const uint32_t src_mesh_id = *src_fabric_node_id.mesh_id;
    const uint32_t src_chip_id = (uint32_t)src_fabric_node_id.chip_id;

    const auto subgroup_shape = subgroup_range.shape();
    const uint32_t mesh_rows = subgroup_shape[0];
    const uint32_t mesh_cols = subgroup_shape.dims() > 1 ? subgroup_shape[1] : 1;
    const uint32_t subgroup_num_devices = mesh_rows * mesh_cols;

    uint32_t linearized_mesh_coord = 0;
    {
        const auto& start = subgroup_range.start_coord();
        const uint32_t local_row = mesh_coordinate[0] - start[0];
        const uint32_t local_col = mesh_coordinate.dims() > 1 ? (mesh_coordinate[1] - start[1]) : 0;
        linearized_mesh_coord = local_row * mesh_cols + local_col;
    }

    auto worker_core_range_set = operation_attributes.worker_core_range_set;
    auto subdevice_cores = corerange_to_cores(worker_core_range_set);
    TT_FATAL(!subdevice_cores.empty(), "No worker cores available on subdevice");

    // Single worker core (payload is small)
    const uint32_t num_cores = 1;
    auto sender_core_grid = tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(
        subdevice_cores.at(0), num_cores, worker_core_range_set, true);
    std::vector<CoreCoord> sender_cores = corerange_to_cores(sender_core_grid);
    const auto& sender_core = sender_cores.front();

    const uint32_t input_page_size = detail::get_page_size(input_tensor);
    const uint32_t aligned_input_page_size = detail::get_aligned_page_size(input_tensor);
    const uint32_t output_aligned_page_size = detail::get_aligned_page_size(output_tensor);
    const auto l1_alignment = tt::tt_metal::hal::get_l1_alignment();

    // N_ROWS = number of input pages this chip contributes (product of all leading dims).
    // 1D case: (1, W) → N_ROWS=1. 2D case after composite's axis-1 pre-pass: (mesh_cols, 1, W)
    // or (mesh_cols, W) → N_ROWS=mesh_cols.
    const auto& input_shape = input_tensor.logical_shape();
    uint32_t n_rows = 1;
    for (size_t i = 0; i + 1 < input_shape.size(); ++i) {
        n_rows *= input_shape[i];
    }

    // Input scratch CB — holds all N_ROWS pages of this chip's contribution.
    tt::tt_metal::CircularBufferConfig input_cb_config =
        tt::tt_metal::CircularBufferConfig(
            n_rows * aligned_input_page_size,
            {{tt::CBIndex::c_0, tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype())}})
            .set_page_size(tt::CBIndex::c_0, aligned_input_page_size);
    tt::tt_metal::CreateCircularBuffer(program, sender_core_grid, input_cb_config);

    // Packet header CB (2 headers: one for payload writes, one for semaphores).
    auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    tt::tt_metal::CircularBufferConfig packet_header_cb_config =
        tt::tt_metal::CircularBufferConfig(2 * packet_header_size_bytes, {{tt::CBIndex::c_1, tt::DataFormat::UInt8}})
            .set_page_size(tt::CBIndex::c_1, packet_header_size_bytes);
    tt::tt_metal::CreateCircularBuffer(program, sender_core_grid, packet_header_cb_config);

    // Fan-out restricted to axis-0 (same-column) peers; axis-1 replication is handled upstream
    // by the composite via ttnn::all_gather when the subgroup is 2D.
    const auto [neighbors, directions] = ccl::common::get_neighbors_in_range(
        subgroup_range, mesh_coordinate, operation_attributes.topology, operation_attributes.cluster_axis);

    std::vector<uint32_t> dest_mesh_id, dest_chip_id;
    for (const auto& coord : subgroup_range) {
        auto id = mesh_device->get_fabric_node_id(coord);
        dest_mesh_id.push_back(*id.mesh_id);
        dest_chip_id.push_back((uint32_t)id.chip_id);
    }

    auto fabric_max_packet_size = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();

    std::vector<uint32_t> compile_time_args = {
        /*cb_input_id*/ static_cast<uint32_t>(tt::CBIndex::c_0),
        /*cb_packet_header_id*/ static_cast<uint32_t>(tt::CBIndex::c_1),
        /*W*/ input_tensor.logical_shape()[-1],
        /*N_ROWS*/ n_rows,
        /*input_page_size*/ input_page_size,
        /*aligned_input_page_size*/ aligned_input_page_size,
        /*aligned_output_page_size*/ output_aligned_page_size,
        /*src_mesh_id*/ src_mesh_id,
        /*src_chip_id*/ src_chip_id,
        /*mesh_rows*/ mesh_rows,
        /*mesh_cols*/ mesh_cols,
        /*linearized_mesh_coord*/ linearized_mesh_coord,
        /*subgroup_num_devices*/ subgroup_num_devices,
        /*fabric_max_packet_size*/ (uint32_t)fabric_max_packet_size,
        /*l1_alignment*/ l1_alignment,
        /*num_links*/ operation_attributes.num_links,
        /*topology*/ static_cast<uint32_t>(operation_attributes.topology),
    };
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(compile_time_args);

    std::map<std::string, std::string> defines;
    defines["DEST_CHIP_ID"] = ccl::common::stringify(dest_chip_id);
    defines["DEST_MESH_ID"] = ccl::common::stringify(dest_mesh_id);
    defines["DIRECTIONS"] = ccl::common::stringify(directions);

    tt::tt_metal::KernelHandle kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/subgroup_gather_histograms/device/kernels/"
        "fabric_gather_histogram_worker.cpp",
        sender_core_grid,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::detail::preferred_noc_for_dram_write(mesh_device->arch()),
            .compile_args = compile_time_args,
            .defines = defines});

    std::vector<uint32_t> runtime_args = {
        input_tensor.buffer()->address(),
        output_tensor.buffer()->address(),
        (uint32_t)init_semaphore.address(),
    };

    if (operation_attributes.num_links > 0) {
        uint32_t core_link = 0;  // single worker core
        for (const auto& neighbor_coordinate : neighbors) {
            if (neighbor_coordinate == mesh_coordinate) {
                continue;
            }
            tt::tt_fabric::append_fabric_connection_rt_args(
                src_fabric_node_id,
                mesh_device->get_fabric_node_id(neighbor_coordinate),
                core_link,
                program,
                sender_core,
                runtime_args);
        }
    }

    tt::tt_metal::SetRuntimeArgs(program, kernel_id, sender_core, runtime_args);

    return {std::move(program), {/*kernel_id=*/kernel_id, /*cores=*/sender_cores, /*init_semaphore=*/init_semaphore}};
}

void SubgroupGatherHistogramsProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const SubgroupGatherHistogramsParams&,
    const SubgroupGatherHistogramsInputs& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& shared = cached_workload.shared_variables.at(range);
        for (const auto& core : shared.cores) {
            auto& rt = tt::tt_metal::GetRuntimeArgs(program, shared.kernel_id, core);
            rt.at(0) = tensor_args.input_tensor.buffer()->address();
            rt.at(1) = tensor_return_value.buffer()->address();
            rt.at(2) = (uint32_t)shared.init_semaphore.address();
        }
    }
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::subgroup_gather_histograms
