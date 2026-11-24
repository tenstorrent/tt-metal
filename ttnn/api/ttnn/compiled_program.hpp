// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <string>
#include <utility>

#include <tt-metalium/experimental/kernel_config_buffer.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program_cache.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/distributed/types.hpp"

// Experimental and subject to change.

namespace ttnn::device_operation {

// A device operation's mesh workload, compiled and laid out for one set of operation attributes and tensors
// (device_operation::compile), outside the program cache. device_operation::run launches it with any operation
// attributes and tensors that have the same program cache key, applying their runtime arguments as a cache hit does,
// optionally from a kernel config buffer (create_kernel_config_buffer).
//
// Tied to the MeshDevice it was compiled for, which it records by ID only. Move-only.
class CompiledProgram {
public:
    // Finds the mesh workload in a CachedProgramFactory (device_operation::detail::get_cached_workload).
    using WorkloadOf =
        tt::tt_metal::distributed::MeshWorkload& (*)(tt::tt_metal::program_cache::detail::CachedProgramFactory&);

    CompiledProgram(
        tt::tt_metal::program_cache::detail::CachedProgramFactory cached_program_factory,
        WorkloadOf workload_of,
        tt::tt_metal::program_cache::detail::ProgramCacheKey key,
        std::string operation_name,
        int mesh_device_id) :
        cached_program_factory_(std::move(cached_program_factory)),
        workload_of_(workload_of),
        key_(std::move(key)),
        operation_name_(std::move(operation_name)),
        mesh_device_id_(mesh_device_id) {}

    CompiledProgram(CompiledProgram&&) noexcept = default;
    CompiledProgram& operator=(CompiledProgram&&) noexcept = default;
    CompiledProgram(const CompiledProgram&) = delete;
    CompiledProgram& operator=(const CompiledProgram&) = delete;

    tt::tt_metal::distributed::MeshWorkload& workload() { return workload_of_(cached_program_factory_); }
    tt::tt_metal::program_cache::detail::CachedProgramFactory& cached_program_factory() {
        return cached_program_factory_;
    }
    std::size_t program_factory_index() const { return cached_program_factory_.program_factory_index; }
    const tt::tt_metal::program_cache::detail::ProgramCacheKey& key() const { return key_; }
    // The device operation it was compiled from.
    const std::string& operation_name() const { return operation_name_; }
    // ID of the MeshDevice it was compiled for.
    int mesh_device_id() const { return mesh_device_id_; }

private:
    tt::tt_metal::program_cache::detail::CachedProgramFactory cached_program_factory_;
    WorkloadOf workload_of_;
    tt::tt_metal::program_cache::detail::ProgramCacheKey key_;
    std::string operation_name_;
    int mesh_device_id_;
};

// A kernel config buffer on `mesh_device` for `compiled` (experimental::CreateKernelConfigBuffer), for
// device_operation::run to run its Tensix kernel binaries from instead of the KERNEL_CONFIG window. `mesh_device` must
// be the MeshDevice `compiled` was compiled for. Needs slow dispatch and the device opened with
// TT_METAL_ALLOCATOR_MODE_HYBRID=1.
inline tt::tt_metal::experimental::KernelConfigBuffer create_kernel_config_buffer(
    ttnn::MeshDevice& mesh_device, CompiledProgram& compiled) {
    TT_FATAL(
        mesh_device.id() == compiled.mesh_device_id(),
        "\"{}\" was compiled for MeshDevice {}, not MeshDevice {}",
        compiled.operation_name(),
        compiled.mesh_device_id(),
        mesh_device.id());
    return tt::tt_metal::experimental::CreateKernelConfigBuffer(mesh_device, compiled.workload());
}

}  // namespace ttnn::device_operation
