// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/common/queue_id.hpp"

#include <mutex>
#include <set>
#include <tuple>
#include <tt-logger/tt-logger.hpp>

#include "reduce_scatter.hpp"
#include "device/reduce_scatter_device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "ttnn/operations/experimental/ccl/composite_common.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_direct/reduce_scatter_minimal_direct.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_direct/device/reduce_scatter_minimal_direct_op_device_operation.hpp"

namespace ttnn {
using namespace ttnn::operations::ccl;

namespace {

// --- Direct (one-shot) reduce-scatter dispatch policy ---
//
// reduce_scatter_minimal_direct unicasts every destination's slice STRAIGHT to that destination: one
// fabric traversal instead of the ring's N/2 store-and-forward steps, in exchange for ~2.3x the link
// traffic (a distance-h contribution crosses h links). So it is a latency play -- it wins while the
// collective is dominated by fill latency and loses once it is bandwidth-bound, by a margin that grows
// with size.
//
// MEASURED 2026-07-31, blackhole 1x8 ring, trace, op-allocated buffers -- which is how THIS dispatch
// calls it, so the writer's start barrier is compiled in and the numbers below include it. Ratio is
// direct/ring, so < 1.00 means direct wins; per-device input bytes on the left:
//
//     bf16                              bf8
//      16 KB  0.54                       8 KB  0.66
//      64 KB  0.70 / 0.56 (repeats)     34 KB  0.52
//     128 KB  0.65                      68 KB  0.55
//     256 KB  0.59                     136 KB  0.69
//     448 KB  0.78, 0.85               238 KB  0.62, 0.63
//    1024 KB  1.29  <-- ring wins      544 KB  0.98  (tie)
//
// The crossover sits between 448 KB and 1 MB, so the gate is 512 KB. Past it the ring op keeps pulling
// away -- independently confirmed on device crit-path time: 1.55x at 512K elements, 3.52x at 5M,
// 3.95x at 25M. bf8 is still break-even at 544 KB, so a single byte-based gate is slightly
// conservative for it; that costs nothing, since it is a tie there anyway.
constexpr uint64_t k_direct_rs_max_input_bytes = 512ull << 10;

// tt::tt_fabric::Topology is a scoped enum with no fmt formatter, and Ring-vs-Torus is worth reading in
// the dispatch warning below: it separates the 1D hop-count routing regime from the 2D
// destination-node one, which the direct op handles along different code paths.
const char* direct_rs_topology_name(tt::tt_fabric::Topology topology) {
    switch (topology) {
        case tt::tt_fabric::Topology::Ring: return "Ring";
        case tt::tt_fabric::Topology::Torus: return "Torus";
        case tt::tt_fabric::Topology::Linear: return "Linear";
        case tt::tt_fabric::Topology::Mesh: return "Mesh";
        case tt::tt_fabric::Topology::NeighborExchange: return "NeighborExchange";
    }
    return "unknown";
}

bool use_direct_reduce_scatter(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    std::optional<uint32_t> cluster_axis,
    tt::tt_fabric::Topology topology,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<ttnn::MemoryConfig>& intermediate_memory_config,
    bool use_l1_small_for_semaphores) {
    // The axis has to wrap: Ring on a 1D fabric, Torus on a 2D one.
    if (!tt::tt_fabric::is_ring_or_torus(topology)) {
        return false;
    }
    // The direct op has no tuning knobs and no compute-kernel config of its own (it derives
    // fp32_dest_acc from the dtype). Honour an explicit request for any of them by staying on the ring
    // op rather than silently dropping it.
    if (chunks_per_sync.has_value() || num_workers_per_link.has_value() || num_buffers_per_channel.has_value() ||
        compute_kernel_config.has_value()) {
        return false;
    }
    // Same reasoning for the two options the direct op cannot express. Its staging buffer is not the ring
    // op's intermediate -- placement (L1-sharded / L1-interleaved / DRAM) is derived from the shape by
    // reduce_scatter_direct_staging_spec and is not caller-selectable -- so an explicit
    // intermediate_memory_config could not be honoured. Semaphore placement is likewise the factory's own
    // choice. Note use_l1_small_for_semaphores is a plain bool, so "explicitly false" is indistinguishable
    // from the default; only an explicit true can be detected and declined here.
    if (intermediate_memory_config.has_value() || use_l1_small_for_semaphores) {
        return false;
    }
    // Structural constraints (ring, TILE, whole-page split, 1D fabric) are the op's own to state.
    if (!ttnn::experimental::reduce_scatter_minimal_direct_is_applicable(input_tensor, dim, cluster_axis)) {
        return false;
    }
    // Size gate: physical per-device input bytes, so block-float dtypes are counted as they actually
    // land in DRAM rather than by logical element count.
    const auto* buffer = input_tensor.buffer();
    return buffer != nullptr && static_cast<uint64_t>(buffer->size()) <= k_direct_rs_max_input_bytes;
}

}  // namespace

ttnn::Tensor reduce_scatter(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    std::optional<uint32_t> cluster_axis,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::MemoryConfig>& intermediate_memory_config,
    const std::optional<ttnn::Tensor>& optional_output_tensor,
    std::optional<uint32_t> num_links,
    std::optional<tt::tt_fabric::Topology> topology,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    bool use_l1_small_for_semaphores) {
    // If cluster_axis is None, but mesh shape is not 1xM or Mx1, then we call reduce-scatter on cluster_axis=1, then
    // reduce-scatter on cluster_axis=0
    if (cluster_axis == std::nullopt) {
        auto mesh_shape = input_tensor.device()->get_view().shape();
        if (!mesh_shape.is_line_topology()) {
            Tensor tensor = input_tensor;
            for (size_t i = 0; i < mesh_shape.dims(); ++i) {
                tensor = ttnn::reduce_scatter(
                    tensor,
                    dim,
                    i,
                    subdevice_id,
                    memory_config,
                    intermediate_memory_config,
                    optional_output_tensor,
                    num_links,
                    topology,
                    chunks_per_sync,
                    num_workers_per_link,
                    num_buffers_per_channel,
                    compute_kernel_config,
                    use_l1_small_for_semaphores);
            }
            return tensor;
        }
    }
    auto* mesh_device = input_tensor.device();
    uint32_t normalized_dim = input_tensor.logical_shape().get_normalized_index(dim);
    tt::tt_fabric::Topology topology_ = ::ttnn::ccl::get_usable_topology(input_tensor, topology, cluster_axis);
    topology_ = ::ttnn::ccl::convert_2d_to_1d_topology(topology_);

    auto memory_config_ = memory_config.value_or(input_tensor.memory_config());
    // TODO: until #27196 is resolved, the fabric API does not subtract out the one link correctly for dispatch used
    // when not all devices are mmio capable. Manually doing it requires the use of "is_mmio_capable" counting, but as
    // the one link that's subtracted out is only along one cluster axis, we will be using less links we would like
    uint32_t num_links_ = num_links.value_or(common::get_num_links(*mesh_device, cluster_axis));

    auto resolved_compute_kernel_config =
        ttnn::ccl::resolve_fp32_acc_compute_kernel_config(compute_kernel_config, input_tensor.dtype());

    if (composite_common::use_composite_reduce_scatter(input_tensor, dim, cluster_axis)) {
        return composite_common::composite_reduce_scatter(
            input_tensor,
            dim,
            num_links_,
            topology_,
            memory_config_,
            subdevice_id,
            cluster_axis,
            chunks_per_sync,
            num_workers_per_link,
            num_buffers_per_channel,
            resolved_compute_kernel_config,
            use_l1_small_for_semaphores);
    }
    // Small-shape fast path: one fabric hop per contribution instead of the ring's N/2 relay steps.
    // Gated on topology/layout/size -- see use_direct_reduce_scatter. Called through the prim so the
    // caller's optional output tensor can be reused without also having to own the staging buffer;
    // staging stays op-allocated, which is what compiles the writer's start barrier in.
    if (use_direct_reduce_scatter(
            input_tensor,
            dim,
            cluster_axis,
            topology_,
            chunks_per_sync,
            num_workers_per_link,
            num_buffers_per_channel,
            compute_kernel_config,
            intermediate_memory_config,
            use_l1_small_for_semaphores)) {
        // Diagnostic: the direct path is otherwise completely silent (no logging, and no perf model, so
        // it reports PM_IDEAL 0.0), which makes "did this shape take the fast path?" unanswerable from a
        // CI log. Warning level so it survives default log filtering. Deduplicated by shape rather than
        // emitted once per process: reduce_scatter runs inside per-step model loops, so an unconditional
        // warning would flood, but one line per distinct call site is bounded and says which shapes hit.
        {
            // Non-const so decltype(key) is a valid std::set value_type.
            auto key = std::make_tuple(
                ::ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis),
                input_tensor.buffer()->size(),
                static_cast<int>(input_tensor.dtype()),
                normalized_dim);
            static std::mutex direct_warned_mutex;
            static std::set<decltype(key)> direct_warned;
            bool first_for_shape = false;
            {
                std::lock_guard<std::mutex> lock(direct_warned_mutex);
                first_for_shape = direct_warned.insert(key).second;
            }
            if (first_for_shape) {
                log_warning(
                    tt::LogOp,
                    "reduce_scatter: taking the DIRECT (one-shot) path -- num_devices={}, per-device input "
                    "{} B (gate {} B), dtype={}, scatter dim={}, shape={}, topology={}. The {} B gate is "
                    "calibrated on a 1x8 Blackhole ring at 2 links only; other ring sizes and link counts "
                    "are unvalidated. Emitted once per distinct shape.",
                    std::get<0>(key),
                    input_tensor.buffer()->size(),
                    k_direct_rs_max_input_bytes,
                    input_tensor.dtype(),
                    normalized_dim,
                    input_tensor.padded_shape(),
                    direct_rs_topology_name(topology_),
                    k_direct_rs_max_input_bytes);
            }
        }
        return ttnn::prim::reduce_scatter_minimal_direct(
                   input_tensor,
                   static_cast<int32_t>(normalized_dim),
                   memory_config_,
                   cluster_axis,
                   num_links_,
                   optional_output_tensor,
                   /*persistent_staging_tensor=*/std::nullopt,
                   subdevice_id,
                   /*sub_core_grid=*/std::nullopt)
            .at(0);
    }

    return ttnn::prim::reduce_scatter(
               input_tensor,
               normalized_dim,
               cluster_axis,
               subdevice_id,
               memory_config_,
               intermediate_memory_config,
               optional_output_tensor,
               num_links_,
               topology_,
               chunks_per_sync,
               num_workers_per_link,
               num_buffers_per_channel,
               resolved_compute_kernel_config,
               use_l1_small_for_semaphores)
        .at(1);  // first is the intermediate tensor
}

}  // namespace ttnn
