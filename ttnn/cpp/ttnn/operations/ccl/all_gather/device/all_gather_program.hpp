// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/program.hpp>

#include "ttnn/device_operation.hpp"
#include "all_gather_device_operation_types.hpp"
#include "all_gather_plan.hpp"

// Program building shared by the unicast and multicast factories: pipeline constants, the slot CB, the two
// semaphores, kernel creation with the common compile-time args, route args, the per-device workload loop and
// the runtime-arg override.
namespace ttnn::operations::ccl::all_gather_program {

// Pipeline constants. Wrong values cost speed, never correctness. Measured on BH (8x p150b, 2 links, payload
// 15168): (K, D) = (3, 8) and (2, 4) are no faster than (3, 6) at 2 and 26 MB.
inline constexpr uint32_t kSlots = 6;     // CB slots: even (slot parity = packet parity) and fixed, so a cached
                                          // program never depends on later allocator state
inline constexpr uint32_t kInflight = 3;  // local-copy slots a builder holds until their reads land; <= kSlots - 2
inline constexpr uint32_t kBuilderRtInAddr = 0;
inline constexpr uint32_t kBuilderRtOutAddr = 1;
static_assert(kSlots % 2 == 0 && kInflight + 2 <= kSlots);

// Unicast data_valid cadence, in full packets. Each signal is a fabric packet of its own, so a long cadence saves
// link and pump time, while a short one lets the downstream relay start sooner, which counts when an iteration is
// only a few packets. Measured on BH (8 devices, 2 links): at 8-26 MB, 1 packet is up to 1.5x slower than 6-16
// (flat); at 2 MB (9 packets per iteration) 2 (line) or 3 (ring) is best and 8 is up to 20% slower.
uint32_t signal_packets(const all_gather_plan::Plan& plan);

// Cached per program; identical for both factories. Only the builder holds tensor addresses.
struct SharedVariables {
    std::vector<tt::tt_metal::CoreCoord> cores;
    tt::tt_metal::KernelHandle builder_kernel_id{};
    tt::tt_metal::GlobalSemaphore init_sem;
    tt::tt_metal::GlobalSemaphore sync_sem;  // unicast data_valid, multicast done
};
using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<SharedVariables>;

// The plan both the composite decision and the factories run on.
all_gather_plan::Plan make_device_plan(
    const AllGatherParams& args, const Tensor& input, const tt::tt_metal::TensorSpec& output_spec);

// What every device's program shares: worker cores (the same coordinates on every device, which makes mirror
// and partner addressing work) and the two semaphores, allocated before any program runs.
struct Workers {
    std::vector<tt::tt_metal::CoreCoord> cores;
    CoreRangeSet core_range;
    tt::tt_metal::GlobalSemaphore init_sem;
    tt::tt_metal::GlobalSemaphore sync_sem;
};

// One CB slot: [C headers | pad | payload <= P], payload 64 B aligned.
struct SlotLayout {
    uint32_t C;
    uint32_t hdr_bytes;
    uint32_t payload_off;
    uint32_t slot_bytes;
};
SlotLayout make_slot_layout(uint32_t num_connections, uint32_t payload_bytes);

using NamedArgs = std::unordered_map<std::string, uint32_t>;
// Kernel sources and the named compile-time args each adds to the shared slot args (see create_kernels).
struct KernelSpec {
    const char* builder_path;
    const char* pump_path;
    NamedArgs builder_args;
    NamedArgs pump_args;
};
struct Kernels {
    tt::tt_metal::KernelHandle builder;
    tt::tt_metal::KernelHandle pump;
};
// Creates the slot CB and both kernels on `cores`: builder on RISCV_0/NOC1, pump on RISCV_1/NOC0. The EDMs write
// everything they deliver into DRAM over NOC1; on BH, sends to the EDM over NOC1 cost ~4x more than over NOC0.
Kernels create_kernels(
    tt::tt_metal::Program& program,
    const CoreRangeSet& cores,
    const all_gather_plan::Plan& plan,
    const SlotLayout& slot,
    const KernelSpec& spec,
    const Tensor& input,
    const Tensor& output);

// Appends a RouteArgs block (see kernels/route.hpp). `dst` is the connection's first-hop neighbor.
void append_route_args(
    std::vector<uint32_t>& args, const tt::tt_fabric::FabricNodeId& dst, uint32_t range_v0, uint32_t range_v1);

// Builds one device's program (kernels plus runtime args) into `program`.
using DeviceProgramFn = std::function<Kernels(
    tt::tt_metal::Program& program,
    const all_gather_plan::Plan& plan,
    const Workers& workers,
    const ttnn::MeshCoordinate& coord)>;

// Plans, checks the plan chose `algo`, allocates workers (cores_per_link per link) and builds every device's
// program with `fn`.
cached_mesh_workload_t build_workload(
    const AllGatherParams& args,
    const ttnn::MeshCoordinateRangeSet& coords,
    const Tensor& input,
    const Tensor& output,
    all_gather_plan::AgAlgo algo,
    uint32_t cores_per_link,
    const DeviceProgramFn& fn);

// Rewrites the tensor addresses: the only runtime args that change between cached invocations.
void override_runtime_arguments(cached_mesh_workload_t& workload, const Tensor& input, const Tensor& output);

}  // namespace ttnn::operations::ccl::all_gather_program
