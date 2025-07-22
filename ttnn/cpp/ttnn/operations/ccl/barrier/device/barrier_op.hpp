// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn {

struct Barrier {
    const MemoryConfig output_mem_config;
    const ttnn::ccl::Topology topology;
    std::vector<IDevice*> devices;

    // Required functions to all tensor op functions
    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::MeshWorkloadWithCallbacks create_mesh_workload(
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const std::vector<Tensor>& input_tensors,
        std::vector<Tensor>& output_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program_at(
        const ttnn::MeshCoordinate& mesh_coord,
        const std::vector<Tensor>& input_tensors,
        std::vector<Tensor>& output_tensors) const;
};

namespace ccl::barrier::detail {

// Template for the barrier_with_workers function
// Found in device/host/barrier_full_worker_grid.cpp
tt::tt_metal::operation::ProgramWithCallbacks barrier_with_workers(
    const Tensor& input_tensors,
    const Tensor& output_tensors,
    bool is_starting_core,
    uint32_t ring_size,
    uint32_t ring_index,
    chip_id_t target_device_id,
    std::optional<chip_id_t> receiver_device_id,
    std::optional<chip_id_t> sender_device_id,
    ttnn::ccl::Topology topology);

};  // namespace ccl::barrier::detail

namespace operations::ccl {
// Template for the barrier tensor found in device/barrier_op.cpp
Tensor barrier_function(const Tensor& input_tensor, const ttnn::Barrier& barrier_struct);
std::vector<Tensor> barrier_function(const std::vector<Tensor>& input_tensors, const ttnn::Barrier& barrier_struct);
}  // namespace operations::ccl

}  // namespace ttnn
