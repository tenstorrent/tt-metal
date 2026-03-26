// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <optional>
#include <unordered_map>

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "tools/profiler/host_dispatch_microbench.hpp"

#include "patchable_generic_op_program_factory.hpp"

namespace ttnn::operations::experimental::generic::program {
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;
using tt::tt_metal::distributed::MeshWorkload;

namespace {

using OptionalAddr = std::optional<std::uint32_t>;

std::vector<OptionalAddr> collect_io_tensor_addresses(const patchable_tensor_args_t& tensor_args) {
    std::vector<OptionalAddr> addrs;
    addrs.reserve(tensor_args.io_tensors.size());
    for (const auto& t : tensor_args.io_tensors) {
        auto* buf = t.buffer();
        if (buf != nullptr) {
            addrs.push_back(buf->address());
        } else {
            addrs.push_back(std::nullopt);
        }
    }
    return addrs;
}

std::optional<std::uint32_t> find_io_tensor_index(std::uint32_t value, const std::vector<OptionalAddr>& addrs) {
    for (size_t i = 0; i < addrs.size(); ++i) {
        if (addrs[i].has_value() && addrs[i].value() == value) {
            return static_cast<std::uint32_t>(i);
        }
    }
    return std::nullopt;
}

void discover_address_slots(
    const ProgramDescriptor& desc,
    const std::vector<OptionalAddr>& tensor_addrs,
    PatchableGenericMeshProgramFactory::shared_variables_t& out) {
    for (size_t ki = 0; ki < desc.kernels.size(); ++ki) {
        const auto& kd = desc.kernels[ki];
        for (const auto& [coord, args] : kd.runtime_args) {
            for (size_t ai = 0; ai < args.size(); ++ai) {
                if (auto ti = find_io_tensor_index(args[ai], tensor_addrs)) {
                    out.per_core_runtime_arg_slots.push_back(PatchableGenericMeshProgramFactory::PerCoreRuntimeArgSlot{
                        .kernel_idx = static_cast<std::uint32_t>(ki),
                        .core = coord,
                        .arg_idx = static_cast<std::uint32_t>(ai),
                        .io_tensor_index = *ti,
                    });
                }
            }
        }
        for (size_t ai = 0; ai < kd.common_runtime_args.size(); ++ai) {
            if (auto ti = find_io_tensor_index(kd.common_runtime_args[ai], tensor_addrs)) {
                out.common_runtime_arg_slots.push_back(PatchableGenericMeshProgramFactory::CommonRuntimeArgSlot{
                    .kernel_idx = static_cast<std::uint32_t>(ki),
                    .arg_idx = static_cast<std::uint32_t>(ai),
                    .io_tensor_index = *ti,
                });
            }
        }
    }

    for (size_t ci = 0; ci < desc.cbs.size(); ++ci) {
        const auto* buf = desc.cbs[ci].buffer;
        if (buf != nullptr) {
            if (auto ti = find_io_tensor_index(buf->address(), tensor_addrs)) {
                out.cb_tensor_slots.push_back(PatchableGenericMeshProgramFactory::CBTensorSlot{
                    .cb_idx = static_cast<std::uint32_t>(ci), .io_tensor_index = *ti});
            }
        }
    }
}

/// Flatten io_tensor addresses into a flat uint32_t vector (0 for null buffers).
/// Unlike collect_io_tensor_addresses (which returns optional), this is cheaper
/// to compare against prev_io_addresses.
std::vector<std::uint32_t> collect_io_addresses_flat(const patchable_tensor_args_t& tensor_args) {
    std::vector<std::uint32_t> addrs;
    addrs.reserve(tensor_args.io_tensors.size());
    for (const auto& t : tensor_args.io_tensors) {
        auto* buf = t.buffer();
        addrs.push_back(buf != nullptr ? buf->address() : 0u);
    }
    return addrs;
}

void patch_program_from_io_tensors(
    Program& program,
    PatchableGenericMeshProgramFactory::shared_variables_t& shared_vars,
    const patchable_tensor_args_t& tensor_args) {
    const auto cur_addrs = [&tensor_args]() {
        tt::tt_metal::host_dispatch_microbench::ScopedTimer _collect_timer(
            tt::tt_metal::host_dispatch_microbench::Slot::PatchableCollectIoTensorAddresses);
        return collect_io_addresses_flat(tensor_args);
    }();

    const auto& prev = shared_vars.prev_io_addresses;
    const bool have_prev = prev.size() == cur_addrs.size();

    tt::tt_metal::host_dispatch_microbench::ScopedTimer _apply_patches_timer(
        tt::tt_metal::host_dispatch_microbench::Slot::PatchableApplySlotPatches);

    const auto check_io_index = [&](std::uint32_t io_idx, const char* ctx) {
        TT_FATAL(
            io_idx < cur_addrs.size(),
            "patchable_generic_op: {} io_tensor_index {} out of range ({} io tensors)",
            ctx,
            io_idx,
            cur_addrs.size());
    };

    for (const auto& slot : shared_vars.per_core_runtime_arg_slots) {
        check_io_index(slot.io_tensor_index, "per-core runtime arg");
        const auto addr = cur_addrs[slot.io_tensor_index];
        if (have_prev && addr == prev[slot.io_tensor_index]) {
            continue;
        }
        auto& cached = GetRuntimeArgs(program, slot.kernel_idx, slot.core);
        cached.at(slot.arg_idx) = addr;
    }

    for (const auto& slot : shared_vars.common_runtime_arg_slots) {
        check_io_index(slot.io_tensor_index, "common runtime arg");
        const auto addr = cur_addrs[slot.io_tensor_index];
        if (have_prev && addr == prev[slot.io_tensor_index]) {
            continue;
        }
        auto& cached = GetCommonRuntimeArgs(program, slot.kernel_idx);
        cached.at(slot.arg_idx) = addr;
    }

    for (const auto& slot : shared_vars.cb_tensor_slots) {
        check_io_index(slot.io_tensor_index, "CB tensor");
        const auto addr = cur_addrs[slot.io_tensor_index];
        if (have_prev && addr == prev[slot.io_tensor_index]) {
            continue;
        }
        auto* buf = tensor_args.io_tensors[slot.io_tensor_index].buffer();
        TT_FATAL(buf != nullptr, "patchable_generic_op: CB patch tensor has no buffer");
        auto cb_handle = shared_vars.cb_handles[slot.cb_idx];
        UpdateDynamicCircularBufferAddress(program, cb_handle, *buf);
    }

    shared_vars.prev_io_addresses = cur_addrs;
}

}  // namespace

PatchableGenericMeshProgramFactory::cached_program_t PatchableGenericMeshProgramFactory::create_at(
    const ProgramDescriptor& program_descriptor, const patchable_tensor_args_t& tensor_args) {
    Program program{program_descriptor};
    shared_variables_t shared_vars;

    {
        tt::tt_metal::host_dispatch_microbench::ScopedTimer _program_build_timer(
            tt::tt_metal::host_dispatch_microbench::Slot::PatchableCreateAtProgramBuild);
        auto cbs = program.circular_buffers();
        shared_vars.cb_handles.reserve(cbs.size());
        for (const auto& cb : cbs) {
            shared_vars.cb_handles.push_back(static_cast<CBHandle>(cb->id()));
        }
    }

    {
        tt::tt_metal::host_dispatch_microbench::ScopedTimer _discover_timer(
            tt::tt_metal::host_dispatch_microbench::Slot::PatchableDiscoverAddressSlots);
        const auto tensor_addrs = collect_io_tensor_addresses(tensor_args);
        discover_address_slots(program_descriptor, tensor_addrs, shared_vars);
    }

    return {std::move(program), std::move(shared_vars)};
}

PatchableGenericMeshProgramFactory::cached_mesh_workload_t PatchableGenericMeshProgramFactory::create_mesh_workload(
    const patchable_operation_attributes_t& operation_attributes,
    const MeshCoordinateRangeSet& /*tensor_coords*/,
    const patchable_tensor_args_t& tensor_args,
    patchable_tensor_return_value_t& /*tensor_return_value*/) {
    MeshWorkload mesh_workload;
    std::unordered_map<MeshCoordinateRange, mesh_shared_variables_t> mesh_shared_variables;

    for (const auto& [mesh_coord_range, program_descriptor] : operation_attributes.mesh_programs) {
        auto cached_program = create_at(program_descriptor, tensor_args);
        mesh_workload.add_program(mesh_coord_range, std::move(cached_program.program));
        mesh_shared_variables[mesh_coord_range] = mesh_shared_variables_t{std::move(cached_program.shared_variables)};
    }

    return cached_mesh_workload_t{std::move(mesh_workload), std::move(mesh_shared_variables)};
}

void PatchableGenericMeshProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_mesh_workload,
    const patchable_operation_attributes_t& operation_attributes,
    const patchable_tensor_args_t& tensor_args,
    patchable_tensor_return_value_t& /*tensor_return_value*/) {
    auto& workload_programs = cached_mesh_workload.workload.get_programs();
    const auto& mesh_programs = operation_attributes.mesh_programs;

    TT_FATAL(
        workload_programs.size() == mesh_programs.size(),
        "Size mismatch between cached workload programs ({}) and operation mesh_programs ({})",
        workload_programs.size(),
        mesh_programs.size());

    for (const auto& [range, _pd] : mesh_programs) {
        (void)_pd;
        auto program_it = workload_programs.find(range);
        TT_FATAL(
            program_it != workload_programs.end(),
            "MeshCoordinateRange {} not found in cached workload programs",
            range);

        auto& shared_vars = cached_mesh_workload.shared_variables.at(range);
        patch_program_from_io_tensors(program_it->second, shared_vars.program_shared_variables, tensor_args);
    }
}

}  // namespace ttnn::operations::experimental::generic::program
