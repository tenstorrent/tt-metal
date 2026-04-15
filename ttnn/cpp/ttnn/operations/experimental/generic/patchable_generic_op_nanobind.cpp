// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "patchable_generic_op_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <unordered_set>

#include "device/patchable_generic_op_device_operation.hpp"
#include "device/patchable_generic_op_helpers.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace nb = nanobind;

namespace ttnn::operations::experimental::generic::detail {

namespace {

void dispatch_patched(
    const std::vector<Tensor>& input_tensors,
    const std::vector<Tensor>& output_tensors,
    const tt::tt_metal::ProgramDescriptor& program_descriptor,
    const AddressSlots& address_slots) {
    TT_FATAL(!input_tensors.empty(), "input_tensors must not be empty");
    auto* mesh_device = input_tensors.front().device();
    TT_FATAL(mesh_device != nullptr, "Tensor must be on a device");

    std::vector<Tensor> io_tensors;
    io_tensors.reserve(input_tensors.size() + output_tensors.size());
    io_tensors.insert(io_tensors.end(), input_tensors.begin(), input_tensors.end());
    io_tensors.insert(io_tensors.end(), output_tensors.begin(), output_tensors.end());

    tt::tt_metal::experimental::MeshProgramDescriptor mesh_program_descriptor;
    mesh_program_descriptor.mesh_programs.emplace_back(
        ttnn::MeshCoordinateRange(mesh_device->shape()), program_descriptor);

    auto& desc_copy = mesh_program_descriptor.mesh_programs.back().second;
    patch_stale_descriptor(desc_copy, io_tensors, address_slots);

    (void)ttnn::prim::patchable_generic_op(io_tensors, mesh_program_descriptor);
}

std::vector<Tensor> allocate_outputs(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const std::vector<TensorSpec>& output_specs,
    const std::vector<std::uint32_t>& shared_output_map) {
    const auto n = output_specs.size();
    std::vector<Tensor> outputs;
    outputs.reserve(n);

    if (shared_output_map.empty()) {
        for (size_t i = 0; i < n; ++i) {
            outputs.push_back(tt::tt_metal::create_device_tensor(output_specs[i], mesh_device));
        }
    } else {
        TT_FATAL(
            shared_output_map.size() == n,
            "shared_output_map size ({}) must match output_specs size ({})",
            shared_output_map.size(),
            n);
        for (size_t i = 0; i < n; ++i) {
            if (shared_output_map[i] == static_cast<std::uint32_t>(i)) {
                outputs.push_back(tt::tt_metal::create_device_tensor(output_specs[i], mesh_device));
            } else {
                auto canonical = shared_output_map[i];
                TT_FATAL(
                    output_specs[i] == output_specs[canonical],
                    "patchable_generic_op: shared outputs at indices {} and {} have "
                    "mismatched TensorSpecs — this indicates a bug in the merge logic",
                    i,
                    canonical);
                outputs.push_back(outputs[canonical]);
            }
        }
    }
    return outputs;
}

/// Persistent dispatch state — caches all marshalling work at construction.
/// ``set_input()`` writes tensors into pre-sized slots; ``dispatch()`` allocates
/// ephemeral outputs, patches the descriptor in-place, and dispatches.
class FusionDispatchState {
    // Cached at construction (never changes)
    std::vector<TensorSpec> output_specs_;
    std::vector<std::uint32_t> shared_output_map_;
    std::vector<std::uint32_t> result_reorder_;
    AddressSlots address_slots_;
    tt::tt_metal::distributed::MeshDevice* mesh_device_;

    // Pre-built MeshProgramDescriptor — patched IN PLACE each dispatch.
    // Safe because patch_stale_descriptor writes addresses unconditionally
    // and custom_program_hash is pre-set at cache-write time.
    tt::tt_metal::experimental::MeshProgramDescriptor mesh_desc_;

    // Deduped input tensor slots — written by set_input(), read by dispatch().
    std::vector<Tensor> inputs_;

    // Tracks which slots set_input() wrote since the last dispatch.
    // Dirty slots are cleared after dispatch to release L1 buffers.
    // Non-dirty slots (weights set at construction) stay alive.
    std::vector<bool> dirty_;

    // Set by set_input(), cleared by dispatch().  Tells run() whether update()
    // was called since the last dispatch — if not, fall back to _container_run
    // which reads op.input_tensors (supports direct assignment patterns).
    bool ready_ = false;

public:
    FusionDispatchState(
        nb::list ops_input_tensors,
        nb::list output_specs_py,
        const std::vector<std::uint32_t>& shared_output_map,
        const std::vector<std::uint32_t>& result_reorder,
        const tt::tt_metal::ProgramDescriptor& program_descriptor,
        const AddressSlots& address_slots) :
        shared_output_map_(shared_output_map), result_reorder_(result_reorder), address_slots_(address_slots) {
        // 1. Dedup inputs from ops' input_tensors (by Python object identity).
        std::unordered_set<PyObject*> seen;
        for (auto op_list_handle : ops_input_tensors) {
            auto op_list = nb::cast<nb::list>(op_list_handle);
            for (auto tensor_handle : op_list) {
                PyObject* py_ptr = tensor_handle.ptr();
                if (seen.insert(py_ptr).second) {
                    inputs_.push_back(nb::cast<Tensor>(tensor_handle));
                }
            }
        }
        TT_FATAL(!inputs_.empty(), "ops_input_tensors must contain at least one tensor");

        // 2. Convert output specs.
        output_specs_.reserve(output_specs_py.size());
        for (auto item : output_specs_py) {
            output_specs_.push_back(nb::cast<TensorSpec>(item));
        }

        // 3. Extract mesh device.
        auto* device = inputs_.front().device();
        mesh_device_ = dynamic_cast<tt::tt_metal::distributed::MeshDevice*>(device);
        TT_FATAL(mesh_device_ != nullptr, "Tensor must be on a MeshDevice");

        // 4. Build MeshProgramDescriptor ONCE (copies ProgramDescriptor here, never again).
        mesh_desc_.mesh_programs.emplace_back(ttnn::MeshCoordinateRange(mesh_device_->shape()), program_descriptor);

        // 5. All clean — nothing written by set_input yet.
        dirty_.resize(inputs_.size(), false);
    }

    void set_input(size_t dedup_idx, const Tensor& t) {
        inputs_[dedup_idx] = t;
        dirty_[dedup_idx] = true;
        ready_ = true;
    }

    bool ready() const { return ready_; }

    std::vector<Tensor> dispatch() {
        // 1. Allocate ephemeral outputs from cached specs.
        auto outputs = allocate_outputs(mesh_device_, output_specs_, shared_output_map_);

        // 2. Build io_tensors.
        std::vector<Tensor> io_tensors;
        io_tensors.reserve(inputs_.size() + outputs.size());
        io_tensors.insert(io_tensors.end(), inputs_.begin(), inputs_.end());
        io_tensors.insert(io_tensors.end(), outputs.begin(), outputs.end());

        // 3. Patch descriptor in place (skips null-buffer entries).
        auto& desc = mesh_desc_.mesh_programs.back().second;
        patch_stale_descriptor(desc, io_tensors, address_slots_);

        // 4. Dispatch.
        (void)ttnn::prim::patchable_generic_op(io_tensors, mesh_desc_);

        // 5. Release dirty input slots to free L1 between iterations.
        //    Non-dirty slots (weights) stay alive — the caller holds those anyway.
        for (size_t i = 0; i < dirty_.size(); ++i) {
            if (dirty_[i]) {
                inputs_[i] = Tensor{};
                dirty_[i] = false;
            }
        }
        ready_ = false;

        // 6. Reorder if needed.
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

void bind_patchable_generic_op(nb::module_& mod) {
    nb::class_<AddressSlots>(mod, "AddressSlots", R"doc(
        Opaque mapping of every position in a ProgramDescriptor that references
        an IO tensor address (CB buffer pointers, per-core runtime args, common
        runtime args).  Computed once at build time via ``compute_address_slots``,
        stored by the fusion build cache, and passed to ``patchable_generic_op``
        on each launch to refresh stale addresses.
    )doc");

    mod.def(
        "compute_address_slots",
        &compute_address_slots,
        nb::arg("program_descriptor"),
        nb::arg("io_tensors"),
        R"doc(
        Compute the full address-slot mapping for a ProgramDescriptor.

        Must be called while buffer pointers and runtime arg addresses are
        valid (at build time, before tensors are freed).  Uses the same
        address-matching logic as ``discover_address_slots`` in the program
        factory.  The returned ``AddressSlots`` should be passed to
        ``patchable_generic_op`` on each launch.
        )doc");

    mod.def(
        "patchable_generic_op",
        [](nb::list ops_input_tensors,
           nb::list output_specs_py,
           const std::vector<std::uint32_t>& shared_output_map,
           const std::vector<std::uint32_t>& result_reorder,
           const tt::tt_metal::ProgramDescriptor& program_descriptor,
           const AddressSlots& address_slots) -> std::vector<Tensor> {
            // 1. Gather unique inputs from the ops' input_tensors (dedup by Python object identity).
            std::vector<Tensor> unique_inputs;
            std::unordered_set<PyObject*> seen;
            for (auto op_list_handle : ops_input_tensors) {
                auto op_list = nb::cast<nb::list>(op_list_handle);
                for (auto tensor_handle : op_list) {
                    PyObject* py_ptr = tensor_handle.ptr();
                    if (seen.insert(py_ptr).second) {
                        unique_inputs.push_back(nb::cast<Tensor>(tensor_handle));
                    }
                }
            }
            TT_FATAL(!unique_inputs.empty(), "ops_input_tensors must contain at least one tensor");

            // 2. Allocate outputs from cached specs.
            std::vector<TensorSpec> specs;
            specs.reserve(output_specs_py.size());
            for (auto item : output_specs_py) {
                specs.push_back(nb::cast<TensorSpec>(item));
            }
            auto* device = unique_inputs.front().device();
            auto* mesh_device = dynamic_cast<tt::tt_metal::distributed::MeshDevice*>(device);
            TT_FATAL(mesh_device != nullptr, "Tensor must be on a MeshDevice");
            auto outputs = allocate_outputs(mesh_device, specs, shared_output_map);

            // 3. Patch + dispatch.
            dispatch_patched(unique_inputs, outputs, program_descriptor, address_slots);

            // 4. Apply result_reorder if non-empty.
            if (!result_reorder.empty()) {
                std::vector<Tensor> reordered;
                reordered.reserve(result_reorder.size());
                for (auto idx : result_reorder) {
                    reordered.push_back(outputs[idx]);
                }
                return reordered;
            }
            return outputs;
        },
        nb::arg("ops_input_tensors"),
        nb::arg("output_specs"),
        nb::arg("shared_output_map"),
        nb::arg("result_reorder"),
        nb::arg("program_descriptor"),
        nb::arg("address_slots"),
        R"doc(
        Full persistent-mode hot path: dedup inputs from per-op tensor lists,
        allocate outputs from TensorSpecs, patch the descriptor, dispatch, and
        apply result reordering — all in a single C++ call.

        ``ops_input_tensors`` is a list of lists: each inner list is one op's
        ``input_tensors``.  Tensors are deduped by storage identity across all
        ops.  ``result_reorder`` maps from output_sources order to the caller's
        expected return order (empty = identity).
        )doc");

    mod.def(
        "patchable_generic_op",
        &dispatch_patched,
        nb::arg("input_tensors"),
        nb::arg("output_tensors"),
        nb::arg("program_descriptor"),
        nb::arg("address_slots"),
        R"doc(
        Dispatch into pre-allocated output tensors with address patching.

        Used by the cold path (``FusedOp.launch()``) where outputs already
        exist from ``build()``.  Inputs and outputs are separate vectors.
        )doc");

    nb::class_<FusionDispatchState>(mod, "FusionDispatchState", R"doc(
        Persistent dispatch state that caches all marshalling work at construction.
        ``set_input()`` writes tensors into deduped slots; ``dispatch()`` allocates
        ephemeral outputs, patches the descriptor in-place, and dispatches — all
        with zero per-call argument marshalling.
    )doc")
        .def(
            nb::init<
                nb::list,
                nb::list,
                const std::vector<std::uint32_t>&,
                const std::vector<std::uint32_t>&,
                const tt::tt_metal::ProgramDescriptor&,
                const AddressSlots&>(),
            nb::arg("ops_input_tensors"),
            nb::arg("output_specs"),
            nb::arg("shared_output_map"),
            nb::arg("result_reorder"),
            nb::arg("program_descriptor"),
            nb::arg("address_slots"))
        .def("set_input", &FusionDispatchState::set_input, nb::arg("dedup_idx"), nb::arg("tensor"))
        .def("dispatch", &FusionDispatchState::dispatch)
        .def("ready", &FusionDispatchState::ready);
}

}  // namespace ttnn::operations::experimental::generic::detail
