// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fusion_dispatch_op_nanobind.hpp"

#include <mutex>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "device/fusion_dispatch_op_device_operation.hpp"
#include "device/fusion_dispatch_op_helpers.hpp"
#include "device/fusion_semaphore_bank.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace nb = nanobind;

namespace ttnn::operations::experimental::fusion::detail {

namespace {

/// Patch the descriptor, wrap in MeshProgramDescriptor, and dispatch.
/// Used by direct launch paths with caller-owned IO tensors.
void fusion_dispatch_op_with_address_refresh(
    const std::vector<Tensor>& io_tensors,
    const tt::tt_metal::ProgramDescriptor& program_descriptor,
    const AddressSlots& address_slots,
    const std::vector<std::uint32_t>& semaphore_addresses = {}) {
    TT_FATAL(!io_tensors.empty(), "io_tensors must not be empty");
    auto* mesh_device = io_tensors.front().device();
    TT_FATAL(mesh_device != nullptr, "Tensor must be on a device");

    tt::tt_metal::experimental::MeshProgramDescriptor mesh_program_descriptor;
    mesh_program_descriptor.mesh_programs.emplace_back(
        ttnn::MeshCoordinateRange(mesh_device->shape()), program_descriptor);

    auto& desc_copy = mesh_program_descriptor.mesh_programs.back().second;
    TT_FATAL(
        address_slots.sem_rt_arg_slots.empty() == semaphore_addresses.empty(),
        "Fusion semaphore slots and fresh bank addresses must either both be present or both be absent");
    if (!semaphore_addresses.empty()) {
        patch_semaphore_addresses(desc_copy, address_slots.sem_rt_arg_slots, semaphore_addresses);
    }
    patch_stale_descriptor(desc_copy, io_tensors, address_slots);

    (void)ttnn::prim::fusion_dispatch_op(io_tensors, mesh_program_descriptor);
}

std::vector<Tensor> allocate_outputs(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const std::vector<tt::tt_metal::TensorSpec>& output_specs,
    const std::vector<std::uint32_t>& shared_output_map) {
    const auto n = output_specs.size();
    std::vector<Tensor> outputs;
    outputs.reserve(n);

    if (shared_output_map.empty()) {
        for (size_t i = 0; i < n; ++i) {
            outputs.push_back(ttnn::create_device_tensor(output_specs[i], mesh_device));
        }
    } else {
        TT_FATAL(
            shared_output_map.size() == n,
            "shared_output_map size ({}) must match output_specs size ({})",
            shared_output_map.size(),
            n);
        for (size_t i = 0; i < n; ++i) {
            if (shared_output_map[i] == static_cast<std::uint32_t>(i)) {
                outputs.push_back(ttnn::create_device_tensor(output_specs[i], mesh_device));
            } else {
                auto canonical = shared_output_map[i];
                TT_FATAL(
                    output_specs[i] == output_specs[canonical],
                    "fusion_dispatch_op: shared outputs at indices {} and {} have "
                    "mismatched TensorSpecs — this indicates a bug in the merge logic",
                    i,
                    canonical);
                outputs.push_back(outputs[canonical]);
            }
        }
    }
    return outputs;
}

/// Persistent dispatch state — caches MeshProgramDescriptor (patched in-place)
/// and output allocation metadata. Holds no persistent tensors or Python
/// objects.
///
/// ``dispatch(inputs)`` takes deduped inputs from Python, allocates outputs and
/// a command-lifetime semaphore bank, patches the cached descriptor, dispatches,
/// and returns outputs. No tensor state is retained between calls.
class FusionDispatchState {
    std::vector<tt::tt_metal::TensorSpec> output_specs_;
    std::vector<std::uint32_t> shared_output_map_;
    std::vector<std::uint32_t> result_reorder_;
    AddressSlots address_slots_;
    tt::tt_metal::distributed::MeshDevice* mesh_device_;
    tt::tt_metal::experimental::MeshProgramDescriptor mesh_desc_;
    std::optional<FusionSemaphoreBankConfig> semaphore_bank_config_;
    std::mutex dispatch_mutex_;

public:
    FusionDispatchState(
        const std::vector<tt::tt_metal::TensorSpec>& output_specs,
        const std::vector<std::uint32_t>& shared_output_map,
        const std::vector<std::uint32_t>& result_reorder,
        const tt::tt_metal::ProgramDescriptor& program_descriptor,
        const AddressSlots& address_slots,
        tt::tt_metal::distributed::MeshDevice* mesh_device,
        const std::vector<tt::tt_metal::CoreRangeSet>& semaphore_core_ranges,
        const std::vector<std::uint32_t>& semaphore_initial_values) :
        output_specs_(output_specs),
        shared_output_map_(shared_output_map),
        result_reorder_(result_reorder),
        address_slots_(address_slots),
        mesh_device_(mesh_device),
        semaphore_bank_config_(make_fusion_semaphore_bank_config(semaphore_core_ranges, semaphore_initial_values)) {
        TT_FATAL(mesh_device_ != nullptr, "FusionDispatchState requires a MeshDevice");
        TT_FATAL(
            address_slots_.sem_rt_arg_slots.empty() == !semaphore_bank_config_.has_value(),
            "Fusion semaphore slots and bank metadata must either both be present or both be absent");
        mesh_desc_.mesh_programs.emplace_back(ttnn::MeshCoordinateRange(mesh_device_->shape()), program_descriptor);
    }

    std::vector<Tensor> dispatch(const std::vector<Tensor>& inputs) {
        // One state is shared by every container with the same cache entry.
        // Keep descriptor patching and dispatch atomic across host threads.
        std::scoped_lock lock(dispatch_mutex_);
        auto outputs = allocate_outputs(mesh_device_, output_specs_, shared_output_map_);
        std::optional<FusionSemaphoreBank> semaphore_bank;
        if (semaphore_bank_config_.has_value()) {
            semaphore_bank.emplace(mesh_device_, *semaphore_bank_config_);
        }

        std::vector<Tensor> io_tensors;
        io_tensors.reserve(inputs.size() + outputs.size());
        io_tensors.insert(io_tensors.end(), inputs.begin(), inputs.end());
        io_tensors.insert(io_tensors.end(), outputs.begin(), outputs.end());

        auto& desc = mesh_desc_.mesh_programs.back().second;
        if (semaphore_bank.has_value()) {
            patch_semaphore_addresses(desc, address_slots_.sem_rt_arg_slots, semaphore_bank->addresses());
        }
        patch_stale_descriptor(desc, io_tensors, address_slots_);
        (void)ttnn::prim::fusion_dispatch_op(io_tensors, mesh_desc_);

        if (!result_reorder_.empty()) {
            std::vector<Tensor> reordered;
            reordered.reserve(result_reorder_.size());
            for (auto idx : result_reorder_) {
                reordered.push_back(outputs[idx]);
            }
            return reordered;
        }
        return outputs;
    }
};

}  // namespace

void bind_fusion_dispatch_op(nb::module_& mod) {
    nb::class_<FusionSemaphoreBank>(mod, "FusionSemaphoreBank", R"doc(
        Command-lifetime storage for a set of fusion barrier semaphores.

        Logical semaphores are 16-byte-aligned uint32 slots in one
        lockstep-sharded L1 tensor (same stride as CreateSemaphore / L1
        alignment). The object owns the tensor; dropping the object releases
        the allocation after queued users complete.
    )doc")
        .def(
            nb::init<
                tt::tt_metal::distributed::MeshDevice*,
                const std::vector<tt::tt_metal::CoreRangeSet>&,
                const std::vector<std::uint32_t>&>(),
            nb::arg("mesh_device"),
            nb::arg("semaphore_core_ranges"),
            nb::arg("initial_values"))
        .def_prop_ro("addresses", &FusionSemaphoreBank::addresses)
        .def_prop_ro("tensor", &FusionSemaphoreBank::tensor, nb::rv_policy::reference_internal);

    // NOLINTNEXTLINE(bugprone-unused-raii)
    nb::class_<AddressSlots>(mod, "AddressSlots", R"doc(
        Opaque mapping of every position in a ProgramDescriptor that references
        an IO tensor address (CB buffer pointers, per-core runtime args, common
        runtime args).  Computed once at build time via ``compute_address_slots``,
        stored by the fusion build cache, and passed to ``fusion_dispatch_op``
        on each launch to refresh stale addresses.
    )doc");

    mod.def(
        "compute_address_slots",
        &compute_address_slots,
        nb::arg("program_descriptor"),
        nb::arg("io_tensors"),
        nb::arg("sem_addrs") = std::vector<std::uint32_t>{},
        R"doc(
        Compute the full address-slot mapping for a ProgramDescriptor.

        Must be called while buffer pointers and runtime arg addresses are
        valid (at build time, before tensors are freed).  Uses the same
        address-matching logic as ``discover_address_slots`` in the program
        factory.  The returned ``AddressSlots`` should be passed to
        ``fusion_dispatch_op`` on each launch.

        If ``sem_addrs`` is provided, runtime arg positions matching those
        addresses are recorded as semaphore slots (patched separately from
        tensor addresses on each dispatch).
        )doc");

    mod.def(
        "cb_has_backing",
        [](const tt::tt_metal::CBDescriptor& descriptor) { return get_cb_backing_buffer(descriptor) != nullptr; },
        nb::arg("cb_descriptor"),
        R"doc(
        Return whether a CBDescriptor has Buffer* or tensor backing.
        )doc");

    mod.def(
        "cb_backing_address",
        [](const tt::tt_metal::CBDescriptor& descriptor) -> std::optional<std::uint32_t> {
            if (auto* buffer = get_cb_backing_buffer(descriptor); buffer != nullptr) {
                return buffer->address();
            }
            return std::nullopt;
        },
        nb::arg("cb_descriptor"),
        R"doc(
        L1 address of a CBDescriptor's Buffer* or tensor backing, or None.
        )doc");

    mod.def(
        "copy_cb_backing",
        [](tt::tt_metal::CBDescriptor& dst, const tt::tt_metal::CBDescriptor& src) {
            dst.buffer = src.buffer;
            dst.tensor = src.tensor;
        },
        nb::arg("dst"),
        nb::arg("src"),
        R"doc(
        Copy Buffer* and tensor backing from one CBDescriptor to another.
        )doc");

    mod.def(
        "fusion_dispatch_op",
        &fusion_dispatch_op_with_address_refresh,
        nb::arg("io_tensors"),
        nb::arg("program_descriptor"),
        nb::arg("address_slots"),
        nb::arg("semaphore_addresses") = std::vector<std::uint32_t>{},
        R"doc(
        Dispatch with a flat io_tensors list (inputs + outputs concatenated).
        Used by inline mode and direct dispatch paths.  When provided,
        ``semaphore_addresses`` replaces build-time barrier addresses before
        launch.
        )doc");

    nb::class_<FusionDispatchState>(mod, "FusionDispatchState", R"doc(
        Caches MeshProgramDescriptor (patched in-place) and output allocation
        metadata. Holds no persistent tensors and no Python objects.

        ``dispatch(inputs)`` takes deduped inputs, allocates ephemeral outputs
        and one command-lifetime semaphore bank, patches all addresses,
        dispatches, and returns outputs.
    )doc")
        .def(
            nb::init<
                const std::vector<tt::tt_metal::TensorSpec>&,
                const std::vector<std::uint32_t>&,
                const std::vector<std::uint32_t>&,
                const tt::tt_metal::ProgramDescriptor&,
                const AddressSlots&,
                tt::tt_metal::distributed::MeshDevice*,
                const std::vector<tt::tt_metal::CoreRangeSet>&,
                const std::vector<std::uint32_t>&>(),
            nb::arg("output_specs"),
            nb::arg("shared_output_map"),
            nb::arg("result_reorder"),
            nb::arg("program_descriptor"),
            nb::arg("address_slots"),
            nb::arg("mesh_device"),
            nb::arg("semaphore_core_ranges") = std::vector<tt::tt_metal::CoreRangeSet>{},
            nb::arg("semaphore_initial_values") = std::vector<std::uint32_t>{})
        .def("dispatch", &FusionDispatchState::dispatch, nb::arg("inputs"));
}

}  // namespace ttnn::operations::experimental::fusion::detail
