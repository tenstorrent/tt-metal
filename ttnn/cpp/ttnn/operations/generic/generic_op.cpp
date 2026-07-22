// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "generic_op.hpp"
#include "device/generic_op_device_operation.hpp"

#include <algorithm>
#include <utility>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/core.hpp"

namespace ttnn {

struct PreparedGenericOp::Impl {
    std::vector<Tensor> io_tensors;
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device;
    tt::tt_metal::distributed::MeshWorkload workload;
    std::uint8_t cq_id;
    bool inactive_rank = false;
    bool has_outstanding_dispatch = false;

    Impl(
        const std::vector<Tensor>& tensors,
        const tt::tt_metal::experimental::MeshProgramDescriptor& mesh_program_descriptor,
        std::optional<std::uint8_t> requested_cq_id) :
        io_tensors(tensors), cq_id(requested_cq_id.value_or(tt::tt_metal::GetCurrentCommandQueueIdForThread())) {
        TT_FATAL(
            io_tensors.size() >= 2,
            "PreparedGenericOp requires at least one input tensor and one output tensor, got {} tensors",
            io_tensors.size());
        auto* first_device = io_tensors.front().device();
        TT_FATAL(first_device != nullptr, "PreparedGenericOp tensors must be on a MeshDevice");
        mesh_device = first_device->shared_from_this();

        for (const auto& tensor : io_tensors) {
            TT_FATAL(tensor.device() == first_device, "PreparedGenericOp tensors must be on the same MeshDevice");
            TT_FATAL(tensor.is_allocated(), "PreparedGenericOp tensors must remain allocated");
        }

        // Match device_operation::launch: ranks with no devices in this mesh
        // retain the tensor/resource owners but do not validate or materialize
        // device work. Their dispatch and synchronization methods are no-ops.
        if (mesh_device->is_remote_only()) {
            inactive_rank = true;
            return;
        }

        TT_FATAL(
            !mesh_program_descriptor.mesh_programs.empty(),
            "PreparedGenericOp requires a non-empty MeshProgramDescriptor");
        TT_FATAL(cq_id < mesh_device->num_hw_cqs(), "PreparedGenericOp cq_id {} is out of range", cq_id);

        const auto first_coords = io_tensors.front().device_storage().get_coords();
        TT_FATAL(!first_coords.empty(), "PreparedGenericOp tensors must cover at least one mesh coordinate");
        for (const auto& tensor : io_tensors) {
            const auto coords = tensor.device_storage().get_coords();
            TT_FATAL(
                coords.size() == first_coords.size() && std::equal(coords.begin(), coords.end(), first_coords.begin()),
                "PreparedGenericOp tensors must cover exactly the same mesh coordinates");
        }

        std::size_t num_programs = 0;
        for (const auto& coord : first_coords) {
            const tt::tt_metal::ProgramDescriptor* selected_descriptor = nullptr;
            for (const auto& [range, descriptor] : mesh_program_descriptor.mesh_programs) {
                if (!range.contains(coord)) {
                    continue;
                }
                TT_FATAL(
                    selected_descriptor == nullptr,
                    "PreparedGenericOp MeshProgramDescriptor has overlapping ranges at coordinate {}",
                    coord);
                selected_descriptor = &descriptor;
            }
            TT_FATAL(
                selected_descriptor != nullptr,
                "PreparedGenericOp MeshProgramDescriptor does not cover tensor coordinate {}",
                coord);

            // Match DescriptorMeshWorkloadAdapter semantics: an empty descriptor
            // means that this coordinate intentionally has no work.
            if (selected_descriptor->kernels.empty() && selected_descriptor->cbs.empty() &&
                selected_descriptor->semaphores.empty()) {
                continue;
            }
            workload.add_program(
                tt::tt_metal::distributed::MeshCoordinateRange(coord), tt::tt_metal::Program(*selected_descriptor));
            ++num_programs;
        }
        TT_FATAL(num_programs > 0, "PreparedGenericOp MeshProgramDescriptor produced an empty MeshWorkload");

        // Give the prepared workload one stable non-zero profiler identity. A
        // fresh id per enqueue is useful for ordinary TTNN operations, but would
        // add an atomic increment plus an O(programs) walk to this hot path.
        const auto runtime_id = ttnn::CoreIDs::instance().fetch_and_increment_device_operation_id();
        for (auto& [_, program] : workload.get_programs()) {
            program.set_runtime_id(runtime_id);
        }
    }

    ~Impl() {
        // Fixed-address tensors and caller-provided resource owners must not be
        // released while an asynchronous enqueue is still in flight.
        if (!has_outstanding_dispatch || mesh_device == nullptr || !mesh_device->is_initialized()) {
            return;
        }
        try {
            tt::tt_metal::distributed::Synchronize(mesh_device.get(), cq_id);
        } catch (...) {
            // Destructors must not throw, especially during interpreter/device
            // teardown. Explicit synchronize() remains the diagnostic path.
        }
    }

    void dispatch() {
        if (inactive_rank) {
            return;
        }

        // Set this conservatively before enqueue: if enqueue throws after
        // partially emitting commands, destruction must still drain the queue
        // before releasing any fixed-address resources.
        has_outstanding_dispatch = true;
        tt::tt_metal::distributed::EnqueueMeshWorkload(
            mesh_device->mesh_command_queue(cq_id), workload, /*blocking=*/false);
    }

    void synchronize() {
        if (inactive_rank) {
            return;
        }

        tt::tt_metal::distributed::Synchronize(mesh_device.get(), cq_id);
        has_outstanding_dispatch = false;
    }
};

Tensor generic_op(
    const std::vector<Tensor>& io_tensors,
    const tt::tt_metal::experimental::MeshProgramDescriptor& mesh_program_descriptor) {
    return ttnn::prim::generic_op(io_tensors, mesh_program_descriptor);
}

Tensor generic_op(const std::vector<Tensor>& io_tensors, const tt::tt_metal::ProgramDescriptor& program_descriptor) {
    TT_FATAL(!io_tensors.empty(), "io_tensors must not be empty");
    auto* mesh_device = io_tensors.front().device();
    TT_FATAL(mesh_device != nullptr, "Tensor must be on a device");

    // Create SPMD MeshProgramDescriptor; same program for the entire mesh
    tt::tt_metal::experimental::MeshProgramDescriptor mesh_program_descriptor;
    mesh_program_descriptor.mesh_programs.emplace_back(
        ttnn::MeshCoordinateRange(mesh_device->shape()), program_descriptor);

    return generic_op(io_tensors, mesh_program_descriptor);
}

PreparedGenericOp::PreparedGenericOp(
    const std::vector<Tensor>& io_tensors,
    const tt::tt_metal::experimental::MeshProgramDescriptor& mesh_program_descriptor,
    std::optional<std::uint8_t> cq_id) :
    impl_(std::make_unique<Impl>(io_tensors, mesh_program_descriptor, cq_id)) {}

PreparedGenericOp::~PreparedGenericOp() = default;
PreparedGenericOp::PreparedGenericOp(PreparedGenericOp&&) noexcept = default;
PreparedGenericOp& PreparedGenericOp::operator=(PreparedGenericOp&&) noexcept = default;

void PreparedGenericOp::dispatch() { impl_->dispatch(); }

void PreparedGenericOp::synchronize() { impl_->synchronize(); }

const Tensor& PreparedGenericOp::output_tensor() const { return impl_->io_tensors.back(); }

std::uint8_t PreparedGenericOp::cq_id() const { return impl_->cq_id; }

}  // namespace ttnn
