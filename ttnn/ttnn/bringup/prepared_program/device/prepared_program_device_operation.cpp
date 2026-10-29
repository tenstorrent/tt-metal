// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "prepared_program_device_operation.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>

namespace ttnn::operations::bringup::prepared_program {

namespace {
uint32_t address_of(const Tensor& t) { return static_cast<uint32_t>(t.mesh_buffer().address()); }
tt::tt_metal::Buffer* buffer_of(const Tensor& t) { return t.mesh_buffer().get_reference_buffer(); }

int program_index(const Prepared& p, const std::optional<ttnn::MeshCoordinate>& coord) {
    if (!coord.has_value()) {
        TT_FATAL(
            p.desc.mesh_programs.size() == 1, "prepared_program: several mesh programs need per-coordinate dispatch");
        return 0;
    }
    return p.program_of(*coord);
}
}  // namespace

tt::tt_metal::ProgramDescriptor PreparedProgramFactory::create_descriptor(
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    const Prepared& p = *attributes.prepared;
    const int pi = program_index(p, mesh_dispatch_coordinate);
    if (pi < 0) {
        return tt::tt_metal::ProgramDescriptor{};  // (no program on this chip)
    }
    tt::tt_metal::ProgramDescriptor desc = p.desc.mesh_programs[pi].second;
    const auto& io = tensor_args.io_tensors;
    for (const auto& r : p.rt[pi]) {
        for (auto& [core, args] : desc.kernels[r.kernel].runtime_args) {
            if (core.x == r.x && core.y == r.y) {
                args[r.arg] = address_of(io[r.io]);
                break;
            }
        }
    }
    for (const auto& r : p.crt[pi]) {
        desc.kernels[r.kernel].common_runtime_args[r.arg] = address_of(io[r.io]);
    }
    for (const auto& c : p.cbs[pi]) {
        desc.cbs[c.cb].buffer = buffer_of(io[c.io]);
    }
    for (const auto& v : p.val[pi]) {
        for (auto& [core, args] : desc.kernels[v.kernel].runtime_args) {
            if (core.x == v.x && core.y == v.y) {
                args[v.arg] = attributes.values[v.slot];
                break;
            }
        }
    }
    return desc;
}

void PreparedProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    const Prepared& p = *attributes.prepared;
    const int pi = program_index(p, mesh_dispatch_coordinate);
    if (pi < 0) {
        return;
    }
    const auto& io = tensor_args.io_tensors;
    // (an io tensor's slots are always written together, so its first slot tells whether this program already holds
    // its address - the usual case, the same buffers every call: one lookup instead of every slot)
    for (size_t j = 0; j < io.size(); ++j) {
        const uint32_t a = address_of(io[j]);
        const auto& rts = p.rt_of_io[pi][j];
        if (!rts.empty()) {
            const auto& r0 = p.rt[pi][rts.front()];
            if (tt::tt_metal::GetRuntimeArgs(program, r0.kernel, CoreCoord(r0.x, r0.y))[r0.arg] != a) {
                for (const uint32_t ri : rts) {
                    const auto& r = p.rt[pi][ri];
                    tt::tt_metal::GetRuntimeArgs(program, r.kernel, CoreCoord(r.x, r.y))[r.arg] = a;
                }
            }
        }
        const auto& crts = p.crt_of_io[pi][j];
        if (!crts.empty()) {
            const auto& r0 = p.crt[pi][crts.front()];
            if (tt::tt_metal::GetCommonRuntimeArgs(program, r0.kernel)[r0.arg] != a) {
                for (const uint32_t ri : crts) {
                    const auto& r = p.crt[pi][ri];
                    tt::tt_metal::GetCommonRuntimeArgs(program, r.kernel)[r.arg] = a;
                }
            }
        }
    }
    for (const auto& v : p.val[pi]) {  // (value_slots option: every run)
        tt::tt_metal::GetRuntimeArgs(program, v.kernel, CoreCoord(v.x, v.y))[v.arg] = attributes.values[v.slot];
    }
    if (!p.cbs[pi].empty()) {
        const auto program_cbs = program.circular_buffers();
        const auto& cb_descs = p.desc.mesh_programs[pi].second.cbs;
        for (const auto& c : p.cbs[pi]) {
            tt::tt_metal::UpdateDynamicCircularBufferAddress(
                program, program_cbs[c.cb]->id(), *buffer_of(io[c.io]), cb_descs[c.cb].address_offset);
        }
    }
}

PreparedProgramDeviceOperation::program_factory_t PreparedProgramDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return PreparedProgramFactory{};
}

spec_return_value_t PreparedProgramDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    return tensor_args.output_tensor.tensor_spec();
}

tensor_return_value_t PreparedProgramDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    return tensor_args.output_tensor;  // (the caller allocates it, as generic_op)
}

void PreparedProgramDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    TT_FATAL(
        tensor_args.io_tensors.size() == attributes.prepared->n_io,
        "prepared_program: {} io tensors, prepared with {}",
        tensor_args.io_tensors.size(),
        attributes.prepared->n_io);
}

void PreparedProgramDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attributes, tensor_args);
}

ttsl::hash::hash_t PreparedProgramDeviceOperation::compute_program_hash(
    const operation_attributes_t& attributes, const tensor_args_t&) {
    return attributes.prepared->hash;
}

}  // namespace ttnn::operations::bringup::prepared_program
