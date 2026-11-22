// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/gemma4/sliding_ring_sdpa/device/sliding_ring_sdpa_program_factory.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/chunked_q_mapping.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/chunked_prefill_utils.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/sliding_window_geometry.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/sliding_window_work_plan.hpp"
#include "ttnn/operations/transformer/sdpa/device/sliding_halo_layout.hpp"
#include "ttnn/operations/experimental/gemma4/sliding_ring_sdpa/device/kernels/sliding_ring_sdpa_derived_slots.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/recipe_ring_cbs.hpp"
#include "ttnn/operations/transformer/sdpa/device/recipe/sdpa_recipe_program.hpp"
#include "ttnn/operations/experimental/gemma4/sliding_ring_sdpa/device/sliding_halo_exchange.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <cmath>
#include <string>
#include <limits>
#include <mutex>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <hostdevcommon/common_values.hpp>

using namespace tt::tt_metal;
using ttnn::Tensor;

namespace {

namespace ag_rt = ttnn::ring_attention_all_gather_async_detail;
namespace halo_rt = ttnn::experimental::gemma4::halo_rt;
using ttnn::experimental::gemma4::sliding_halo_exchange_helper;
using ttnn::experimental::gemma4::SlidingHaloConfig;
namespace ring_joint = ttnn::operations::transformer::sdpa::ring_joint;

// Circular sliding KV slab count (0 = unbounded); validation already required whole slabs and >= 2.
uint32_t derived_kv_slab_count(
    const ttnn::prim::Gemma4SlidingRingSDPAParams& args, const ttnn::prim::Gemma4SlidingRingSDPAInputs& tensor_args) {
    return args.circular_kv_cache ? tensor_args.kv_slab_count() : 0;
}

// The shared op's ring-work masks (bits indexed by ring_iter) stay in the kernels' runtime-arg layout. Sliding folds
// its local and predecessor ranges into one synthetic ring iteration, so only iteration 0 is active; the
// single-valid-K-chunk mask only orders the dense writer's save/restore and stays clear.
constexpr uint32_t kSlidingActiveRingIterMask = 1;
constexpr uint32_t kSlidingSingleValidKvChunkMask = 0;

struct RingJointRuntimePlan {
    uint32_t logical_nt = 0;
    ring_joint::ChunkedQMapping kv_pad_q_mapping;
    bool kernel_chunked = false;
    bool kernel_is_causal = false;
};

struct RingJointRuntimeArgLayout {
    uint32_t reader_kv_cache_batch_idx = 0;
    uint32_t reader_logical_nt = 0;
    uint32_t reader_active_ring_iter_mask = 0;
    uint32_t writer_logical_nt = 0;
    uint32_t writer_active_ring_iter_mask = 0;
    uint32_t writer_single_valid_kv_chunk_mask = 0;
    uint32_t compute_logical_nt = 0;
    uint32_t compute_q_pre_wrap_start_tile = 0;
    uint32_t compute_q_pre_wrap_tile_count = 0;
    uint32_t compute_q_post_wrap_start_tile = 0;
    uint32_t compute_q_valid_tile_count = 0;
    uint32_t compute_active_ring_iter_mask = 0;
    CoreCoord grid_size = {0, 0};
};

// On a cluster-axis ring the transport rank is also the tensor rank.
struct RingWritePlan {
    uint32_t transport_rank = 0;
    uint32_t forward_writes_expected = 0;
    uint32_t backward_writes_expected = 0;
    std::optional<ttnn::MeshCoordinate> forward_coord;
    std::optional<ttnn::MeshCoordinate> backward_coord;
};

struct RingJointInputParams {
    bool has_joint_tensors = false;
    bool joint_is_sharded = false;
    const Tensor* joint_q = nullptr;
    const Tensor* joint_k = nullptr;
    const Tensor* joint_v = nullptr;
    const Tensor* gathered_joint_k = nullptr;
    const Tensor* gathered_joint_v = nullptr;
    uint32_t L = 0;          // full (padded) joint sequence length across the ring
    uint32_t L_local = 0;    // per-device joint length (padded L/P when sharded, else L)
    uint32_t logical_l = 0;  // true (unpadded) joint token count; <= L on the sharded path
};

// Resolved joint Q/K/V params + sequence lengths for both replicated and sharded-joint paths.
// Centralizes the optional-tensor / logical_l branching so descriptor construction and the
// runtime planners share one definition of L (full) and L_local (per-device).
RingJointInputParams resolve_ring_joint_input_params(
    const ttnn::prim::Gemma4SlidingRingSDPAParams& args, const ttnn::prim::Gemma4SlidingRingSDPAInputs& tensor_args) {
    RingJointInputParams joint_input_params;
    joint_input_params.has_joint_tensors = tensor_args.joint_q.has_value();
    joint_input_params.joint_is_sharded = tensor_args.joint_is_sharded();

    if (joint_input_params.has_joint_tensors) {
        joint_input_params.joint_q = &tensor_args.joint_q.value();
        joint_input_params.joint_k = &tensor_args.joint_k.value();
        joint_input_params.joint_v =
            tensor_args.joint_v.has_value() ? &tensor_args.joint_v.value() : joint_input_params.joint_k;
    }

    if (joint_input_params.joint_is_sharded) {
        joint_input_params.gathered_joint_k = &tensor_args.gathered_joint_k.value();
        joint_input_params.gathered_joint_v = &tensor_args.gathered_joint_v.value();
        // Per-shard physical length is the (padded, tile-aligned) joint Q shard on this device.
        // The padded total L is that shard times the ring; logical_l carries the true token count,
        // which may be smaller than L when the joint prompt does not fill the last shard. This
        // mirrors how the spatial path keeps logical_n separate from the padded gathered length.
        joint_input_params.L_local = joint_input_params.joint_q->logical_shape()[2];
        joint_input_params.L =
            joint_input_params.L_local * static_cast<uint32_t>(args.all_gather_operation_attributes.ring_size);
        joint_input_params.logical_l = static_cast<uint32_t>(args.logical_l);
    } else if (joint_input_params.has_joint_tensors) {
        joint_input_params.L = joint_input_params.joint_q->logical_shape()[2];
        joint_input_params.L_local = joint_input_params.L;
        joint_input_params.logical_l = joint_input_params.L;
    }

    return joint_input_params;
}

constexpr uint32_t kReaderKernelIndex = 0;
constexpr uint32_t kWriterKernelIndex = 1;
constexpr uint32_t kComputeKernelIndex = 2;

// Sliding halos append one reader/writer pair per exchange after the three SDPA kernels, in
// plan_halo_exchanges order.
constexpr uint32_t kNeighborHaloReaderKernelIndex = 3;
constexpr uint32_t kNeighborHaloWriterKernelIndex = 4;

// Program creation and the per-dispatch patch both go through plan_halo_exchanges, since the exchange
// list fixes the kernel count and order. Only a 1D fabric's line multicast takes a start distance.
std::vector<ring_joint::ChunkedSlidingHaloExchange> plan_halo_exchanges(
    const ttnn::prim::Gemma4SlidingRingSDPAParams& args,
    const ring_joint::ChunkedSlidingHaloLayout& layout,
    uint32_t transport_rank) {
    return ring_joint::plan_chunked_sliding_halo_exchanges(
        layout,
        transport_rank,
        args.all_gather_operation_attributes.topology == ttnn::ccl::Topology::Linear,
        tt::tt_fabric::is_1d_fabric_config(tt::tt_fabric::GetFabricConfig()));
}

// Arrival semaphore key for `source`'s hop-`hop` payload. A receiver's predecessors get distinct keys, so each
// semaphore has one sender and a predecessor running a layer ahead cannot stand in for another. Source-keyed
// layouts key by source slab, so one line multicast is the same key on every receiver; others key by hop.
uint32_t halo_arrival_key(const ring_joint::ChunkedSlidingHaloLayout& layout, uint32_t source, uint32_t hop) {
    if (layout.remote_hop_count() <= 1) {
        return 0;
    }
    if (layout.source_keyed()) {
        return source % ring_joint::chunked_sliding_halo_hop_count(layout.halo_tile_rows, layout.q_local_tile_rows);
    }
    return hop - 1;
}

// The keys `receiver`'s collector waits on, one per remote predecessor (hop 1 first).
std::vector<uint32_t> halo_collected_arrival_keys(
    const ring_joint::ChunkedSlidingHaloLayout& layout, uint32_t receiver) {
    const uint32_t ring_size = layout.ring_size;
    std::vector<uint32_t> keys;
    for (uint32_t hop = 1; hop <= layout.remote_hop_count(); ++hop) {
        keys.push_back(halo_arrival_key(layout, (receiver + ring_size - hop % ring_size) % ring_size, hop));
    }
    return keys;
}

// The collector waits once per key, so each key must name its own counter. Semaphore handles aren't in the
// program hash, so this runs at program creation and on every cache hit.
template <typename Semaphores>
void validate_halo_arrival_semaphores(const Semaphores& semaphores, const std::vector<uint32_t>& keys) {
    std::set<uint32_t> addresses;
    for (const uint32_t key : keys) {
        TT_FATAL(
            halo_rt::kHaloArrivalSemaphoreBase + key < semaphores.size(),
            "Sliding halo needs halo arrival semaphore {}, got {} semaphores",
            key,
            semaphores.size());
        addresses.insert(static_cast<uint32_t>(semaphores[halo_rt::kHaloArrivalSemaphoreBase + key].address()));
    }
    TT_FATAL(
        addresses.size() == keys.size(),
        "Sliding halo needs a distinct arrival semaphore per predecessor: {} predecessors, {} distinct semaphores",
        keys.size(),
        addresses.size());
}

// A one-hop or multicast halo spreads each exchange over every link; a unicast multi-hop halo gives each
// hop one link.
uint32_t halo_links_per_exchange(
    const std::vector<ring_joint::ChunkedSlidingHaloExchange>& exchanges, uint32_t num_links) {
    return exchanges.size() == 1 || exchanges.front().multicast ? num_links : 1;
}

// Source tile row each hop of a multicast exchange ships.
std::vector<uint32_t> multicast_origin_rows(
    const ring_joint::ChunkedSlidingHaloLayout& layout,
    uint32_t transport_rank,
    const ring_joint::ChunkedSlidingHaloExchange& exchange) {
    std::vector<uint32_t> origin_rows;
    if (!exchange.multicast) {
        return origin_rows;
    }
    for (uint32_t i = 0; i < exchange.hop_count; ++i) {
        const auto sources = layout.send_sources(transport_rank, exchange.hop + i);
        TT_FATAL(sources.count == 1, "A multicast halo hop ships one tail, got {}", sources.count);
        origin_rows.push_back(sources.first_start_tile);
    }
    return origin_rows;
}

// Runtime-arg offsets used by cache-hit patching. Descriptor construction appends the same slots through
// CheckedRuntimeArgList, so future layout edits fail on program creation instead of corrupting cache hits.
// Q, K, V, gathered K, gathered V, and attention sink (nullable).
constexpr uint32_t kReaderBaseBufferArgCount = 6;
constexpr uint32_t kReaderJointBufferArgCount = 3;
constexpr uint32_t kReaderQWorkArgCount = 3;
constexpr uint32_t kWriterBaseArgCount = 5;
constexpr uint32_t kComputeRingSequencerArgCount = 6;

struct CheckedRuntimeArgList {
    KernelDescriptor::RTArgList args;
    uint32_t size = 0;

    template <typename T>
    void push_back(const T& value) {
        args.push_back(value);
        size++;
    }

    void append(const std::vector<uint32_t>& values) {
        args.append(values);
        size += values.size();
    }

    template <typename T>
    void push_checked(uint32_t expected_index, const T& value, const char* name) {
        TT_FATAL(
            size == expected_index,
            "RingJoint runtime arg {} expected index {}, got {} before append",
            name,
            expected_index,
            size);
        push_back(value);
    }
};

ring_joint::ChunkedQMapping build_kv_pad_q_mapping(
    uint32_t kv_actual_tile_count,
    uint32_t logical_tile_count,
    uint32_t ring_size,
    uint32_t q_local_padded_tile_count,
    uint32_t device_index) {
    const uint32_t q_chunk_group_tile_count = ring_size * q_local_padded_tile_count;
    const uint32_t first_group = kv_actual_tile_count / q_chunk_group_tile_count;
    const uint32_t last_group = (logical_tile_count - 1) / q_chunk_group_tile_count;
    TT_FATAL(
        last_group <= first_group + 1,
        "KV-pad-aware rotation expects the current valid Q to fit in one fixed global chunk. "
        "Got kv_actual_tile_count={}, new_actual_tile_count={}, q_chunk_group_tile_count={}",
        kv_actual_tile_count,
        logical_tile_count - kv_actual_tile_count,
        q_chunk_group_tile_count);

    const auto mapping = ring_joint::build_chunked_q_mapping(
        kv_actual_tile_count, logical_tile_count, q_local_padded_tile_count, ring_size, device_index);
    TT_FATAL(
        mapping.q_valid_tile_count <= q_local_padded_tile_count,
        "KV-pad-aware rotation mapped more valid Q tiles to this device than its local Q slab can hold. "
        "Got q_valid_tile_count={}, q_local_padded_tile_count={}, device_index={}",
        mapping.q_valid_tile_count,
        q_local_padded_tile_count,
        device_index);
    return mapping;
}

RingWritePlan build_ring_write_plan(
    const ttnn::prim::Gemma4SlidingRingSDPAParams& args,
    const ttnn::prim::Gemma4SlidingRingSDPAInputs& tensor_args,
    const ttnn::MeshCoordinate& coord) {
    RingWritePlan plan;
    const auto& ag = args.all_gather_operation_attributes;
    TT_FATAL(
        !ag.full_mesh && ag.cluster_axis.has_value(),
        "Gemma4SlidingRingSDPA runs on a cluster-axis ring, not a full-mesh snake");
    plan.transport_rank =
        ttnn::ccl::get_linearized_index_from_physical_coord(tensor_args.input_q, coord, ag.cluster_axis);
    plan.forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.input_q, coord, 1, ag.topology, ag.cluster_axis);
    plan.backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.input_q, coord, -1, ag.topology, ag.cluster_axis);

    // Chunked sliding consumes the local slab followed by its cyclic
    // predecessor. Keep that dependency on direction 1 for every device,
    // independent of the dense ring's parity-based split.
    TT_FATAL(
        args.has_sliding_window() && tensor_args.is_chunked() && !args.is_cross,
        "Gemma4SlidingRingSDPA requires chunked causal sliding attention");
    plan.forward_writes_expected = 1;
    plan.backward_writes_expected = 0;
    return plan;
}

RingJointRuntimePlan build_runtime_plan(
    const ttnn::prim::Gemma4SlidingRingSDPAParams& args,
    const ttnn::prim::Gemma4SlidingRingSDPAInputs& tensor_args,
    const RingWritePlan& ring_write_plan) {
    const uint32_t q_local_padded_Nt = tensor_args.input_q.logical_shape()[2] / tt::constants::TILE_HEIGHT;

    RingJointRuntimePlan plan;
    plan.logical_nt = tt::div_up(static_cast<uint32_t>(args.logical_n), tt::constants::TILE_HEIGHT);
    if (args.kv_actual_isl.has_value()) {
        plan.kv_pad_q_mapping = build_kv_pad_q_mapping(
            args.kv_actual_isl.value() / tt::constants::TILE_HEIGHT,
            plan.logical_nt,
            static_cast<uint32_t>(args.ring_size),
            q_local_padded_Nt,
            ring_write_plan.transport_rank);
    }
    // Cross is non-causal on chunked-shaped tensors, so kernels use the non-chunked path.
    plan.kernel_chunked = tensor_args.is_chunked() && !args.is_cross;
    plan.kernel_is_causal = args.is_causal && !plan.kernel_chunked;
    return plan;
}

RingJointRuntimeArgLayout get_runtime_arg_layout(
    const ttnn::prim::Gemma4SlidingRingSDPAParams& args, const ttnn::prim::Gemma4SlidingRingSDPAInputs& tensor_args) {
    const RingJointInputParams joint_input_params = resolve_ring_joint_input_params(args, tensor_args);

    RingJointRuntimeArgLayout layout;
    layout.grid_size = args.program_config.has_value() ? args.program_config->compute_with_storage_grid_size
                                                       : tensor_args.input_q.device()->compute_with_storage_grid_size();

    const uint32_t joint_buffer_args = (joint_input_params.L != 0) ? kReaderJointBufferArgCount : 0;
    // 2 extra buffer slots for gathered_joint_k/v when the sharded-joint path is active
    const uint32_t gathered_joint_buffer_args = joint_input_params.joint_is_sharded ? 2 : 0;
    layout.reader_kv_cache_batch_idx = kReaderBaseBufferArgCount + joint_buffer_args + gathered_joint_buffer_args + 2;
    // Sliding builds no KV chains, so no chain-config args sit between the Q work range and the scalars.
    layout.reader_logical_nt =
        kReaderBaseBufferArgCount + joint_buffer_args + gathered_joint_buffer_args + kReaderQWorkArgCount;
    layout.reader_active_ring_iter_mask = layout.reader_logical_nt + 1;
    layout.writer_logical_nt = kWriterBaseArgCount;
    layout.writer_active_ring_iter_mask = layout.writer_logical_nt + 1;
    layout.writer_single_valid_kv_chunk_mask = layout.writer_active_ring_iter_mask + 1;
    layout.compute_logical_nt = kComputeRingSequencerArgCount;
    layout.compute_q_pre_wrap_start_tile = layout.compute_logical_nt + 1;
    layout.compute_q_pre_wrap_tile_count = layout.compute_q_pre_wrap_start_tile + 1;
    layout.compute_q_post_wrap_start_tile = layout.compute_q_pre_wrap_tile_count + 1;
    layout.compute_q_valid_tile_count = layout.compute_q_post_wrap_start_tile + 1;
    layout.compute_active_ring_iter_mask = layout.compute_q_valid_tile_count + 1;
    return layout;
}

void write_runtime_arg(RuntimeArgsData& args, uint32_t index, uint32_t value, const char* name) {
    TT_FATAL(
        index < args.size(), "Missing RingJoint runtime arg {} at index {}; args.size()={}", name, index, args.size());
    args[index] = value;
}

template <std::size_t N>
void write_runtime_arg_block(
    RuntimeArgsData& args, uint32_t index, const std::array<uint32_t, N>& values, const char* name) {
    TT_FATAL(
        index <= args.size() && N <= args.size() - index,
        "Missing RingJoint runtime arg block {} at index {}; count={}; args.size()={}",
        name,
        index,
        N,
        args.size());
    std::copy(values.begin(), values.end(), args.data() + index);
}

// Tile-rows of the KV cache the halo exchange may read for this chunk: the first
// ceil(logical_n / chunk_global) block-cyclic slabs (a contiguous per-device page prefix), so an
// oversized (growing) KV cache only moves kv_actual-sized data. Returns nullopt when KV-pad rotation
// is off (gather the full input). Passed to the halo exchange at descriptor creation.
std::optional<uint32_t> compute_gather_valid_Ht(
    const ttnn::prim::Gemma4SlidingRingSDPAParams& args, const ttnn::prim::Gemma4SlidingRingSDPAInputs& tensor_args) {
    if (!ttnn::prim::kv_pad_rotation_active(args, tensor_args)) {
        return std::nullopt;
    }
    const uint32_t ring_size = static_cast<uint32_t>(args.all_gather_operation_attributes.ring_size);
    const uint32_t n_local_q = tensor_args.q_lane_padded_rows();  // per-device Q slab (chunk_local), per lane
    const uint32_t chunk_global = n_local_q * ring_size;
    if (tensor_args.has_metadata()) {
        // Metadata path: the all-gather reader recomputes this per dispatch from kv_actual_isl
        // (ring_attention_all_gather_reader.cpp) and CLAMPS against the value baked here, so a
        // create-time bound derived from host logical_n silently caps every later dispatch at the
        // creating chunk's prefix. Under a captured trace that is permanent — the host patch that
        // would otherwise grow it per dispatch never runs on replay — and later chunks attend a
        // truncated history, which degrades with ring depth instead of failing outright.
        //
        // The device value is authoritative here, so bound to the full per-device K extent and let
        // the on-device recompute do the narrowing.
        return tensor_args.input_k.padded_shape()[2] / tt::constants::TILE_HEIGHT;
    }
    const uint32_t valid_slabs = (static_cast<uint32_t>(args.logical_n) + chunk_global - 1) / chunk_global;
    return std::min(
        valid_slabs * (n_local_q / tt::constants::TILE_HEIGHT),
        tensor_args.input_k.padded_shape()[2] / tt::constants::TILE_HEIGHT);
}

void apply_ring_joint_scalar_runtime_args(
    Program& program,
    const ttnn::prim::Gemma4SlidingRingSDPAParams& args,
    const ttnn::prim::Gemma4SlidingRingSDPAInputs& tensor_args,
    const ttnn::MeshCoordinate& mesh_dispatch_coordinate) {
    const bool patch_indexed_kv_cache = args.has_indexed_kv_cache();
    const bool patch_kv_pad_rotation = args.has_kv_pad_rotation();

    // Indexed KV-cache hits also need the current device index and logical sequence geometry for
    // the compact sliding halo. Build these plans for either kind of runtime patch; using default
    // plans here would incorrectly select device 0's halo tail when no KV-pad rotation is present.
    const RingWritePlan ring_write_plan = build_ring_write_plan(args, tensor_args, mesh_dispatch_coordinate);
    const RingJointRuntimePlan runtime_plan = build_runtime_plan(args, tensor_args, ring_write_plan);
    const RingJointRuntimeArgLayout layout = get_runtime_arg_layout(args, tensor_args);
    const uint32_t num_cores = layout.grid_size.x * layout.grid_size.y;
    const uint32_t kv_cache_batch_idx = args.cache_batch_idx().value_or(0);

    // Halo inputs (K, plus V when it isn't the latent-V alias of K). Shared by the indexed-slot and
    // halo-slice patches below.
    const Tensor& input_k = tensor_args.input_k;
    const uint32_t num_ag_inputs = tensor_args.input_v.has_value() ? 2u : 1u;
    const std::array<const Tensor*, 2> ag_inputs = {
        &input_k, tensor_args.input_v.has_value() ? &tensor_args.input_v.value() : &input_k};
    // One reader/writer kernel pair per halo hop, appended in hop order after the three SDPA kernels.
    const ring_joint::ChunkedSlidingHaloLayout runtime_chunked_sliding_layout =
        ring_joint::build_chunked_sliding_halo_layout(
            tensor_args.q_lane_padded_rows() / tt::constants::TILE_HEIGHT,
            args.get_k_chunk_size() / tt::constants::TILE_HEIGHT,
            args.sliding_window_size.value(),
            tt::constants::TILE_HEIGHT,
            static_cast<uint32_t>(args.all_gather_operation_attributes.ring_size),
            runtime_plan.logical_nt,
            tensor_args.gathered_k.logical_shape()[2] / tt::constants::TILE_HEIGHT,
            derived_kv_slab_count(args, tensor_args),
            args.kv_actual_isl.has_value() ? std::optional<uint32_t>(*args.kv_actual_isl / tt::constants::TILE_HEIGHT)
                                           : std::nullopt);
    TT_FATAL(runtime_chunked_sliding_layout.uses_neighbor_halo(), "Sliding attention requires a neighbor halo");
    const std::vector<ring_joint::ChunkedSlidingHaloExchange> halo_exchanges =
        plan_halo_exchanges(args, runtime_chunked_sliding_layout, ring_write_plan.transport_rank);

    // Semaphores aren't in the program hash: re-patch each writer's arrival semaphore and the collector's
    // trailing ones (see sliding_halo_exchange_helper).
    const auto& semaphores = args.all_gather_operation_attributes.semaphore;
    const auto arrival_semaphore_address = [&](uint32_t key) {
        return static_cast<uint32_t>(semaphores.at(halo_rt::kHaloArrivalSemaphoreBase + key).address());
    };
    const auto collected_keys =
        halo_collected_arrival_keys(runtime_chunked_sliding_layout, ring_write_plan.transport_rank);
    validate_halo_arrival_semaphores(semaphores, collected_keys);
    for (uint32_t exchange = 0; exchange < halo_exchanges.size(); ++exchange) {
        const uint32_t writer_address = arrival_semaphore_address(halo_arrival_key(
            runtime_chunked_sliding_layout, ring_write_plan.transport_rank, halo_exchanges[exchange].hop));
        for (auto& col_args : GetRuntimeArgs(program, kNeighborHaloWriterKernelIndex + 2 * exchange)) {
            for (auto& core_args : col_args) {
                if (core_args.size() > halo_rt::kNeighborWriterSemaphoreArg) {  // skip cores without this kernel
                    write_runtime_arg(
                        core_args, halo_rt::kNeighborWriterSemaphoreArg, writer_address, "halo.semaphore");
                }
            }
        }
        if (halo_exchanges[exchange].hop != 1) {
            continue;  // only the hop-1 exchange collects (SlidingHaloConfig::collects_arrivals)
        }
        for (auto& col_args : GetRuntimeArgs(program, kNeighborHaloReaderKernelIndex + 2 * exchange)) {
            for (auto& core_args : col_args) {
                if (core_args.size() == 0) {
                    continue;  // core without this kernel
                }
                TT_FATAL(
                    core_args.size() >= halo_rt::kNeighborReaderRuntimeArgHeaderCount + collected_keys.size(),
                    "Sliding halo collector has too few args for {} arrival semaphores",
                    collected_keys.size());
                const uint32_t first = core_args.size() - collected_keys.size();
                for (uint32_t i = 0; i < collected_keys.size(); ++i) {
                    write_runtime_arg(
                        core_args, first + i, arrival_semaphore_address(collected_keys[i]), "halo.arrival_semaphore");
                }
            }
        }
    }
    if (!patch_indexed_kv_cache && !patch_kv_pad_rotation) {
        return;
    }
    const uint32_t neighbor_reader_tensor_descriptor_field_count =
        tensor_args.has_metadata() ? halo_rt::kNeighborReaderMetadataTensorDescriptorFieldCount
                                   : halo_rt::kNeighborReaderTensorDescriptorFieldCount;

    // Re-patch the halo readers to read the single cache slot `kv_cache_batch_idx`. input_batch_base is
    // uniform across all halo cores/links, so patch every core that runs the reader. Mirrors the
    // helper's create-time arithmetic so miss and hit paths agree.
    if (patch_indexed_kv_cache) {
        const auto patch_reader_batch_base = [&](uint32_t kernel_id,
                                                 uint32_t header_count,
                                                 uint32_t descriptor_field_count,
                                                 uint32_t batch_base_offset) {
            auto& grid_args = GetRuntimeArgs(program, kernel_id);  // [x][y] per-core args
            for (auto& col_args : grid_args) {
                for (auto& core_args : col_args) {
                    for (uint32_t in = 0; in < num_ag_inputs; ++in) {
                        const auto& shape = ag_inputs[in]->padded_shape();
                        const uint32_t num_heads = shape[1];
                        const uint32_t Ht = shape[2] / tt::constants::TILE_HEIGHT;
                        const uint32_t Wt = shape[3] / tt::constants::TILE_WIDTH;
                        const uint32_t input_batch_base =
                            ag_rt::input_batch_base_pages(kv_cache_batch_idx, num_heads, Ht, Wt);
                        const uint32_t idx = header_count + in * descriptor_field_count + batch_base_offset;
                        if (core_args.size() > idx) {  // skip cores that don't run this kernel
                            write_runtime_arg(core_args, idx, input_batch_base, "all_gather_reader.input_batch_base");
                        }
                    }
                }
            }
        };
        for (uint32_t exchange = 0; exchange < halo_exchanges.size(); ++exchange) {
            patch_reader_batch_base(
                kNeighborHaloReaderKernelIndex + 2 * exchange,
                halo_rt::kNeighborReaderRuntimeArgHeaderCount,
                neighbor_reader_tensor_descriptor_field_count,
                halo_rt::kNeighborReaderInputBatchBaseFieldOffset);
        }
    }

    // logical_n/kv_actual_isl are runtime-patched and excluded from the program hash. For
    // compact chunked sliding they also choose which cache-group tail each hop reads.
    // Rewrite every per-link reader/writer slice for the current request.
    const uint32_t links_per_exchange =
        halo_links_per_exchange(halo_exchanges, args.all_gather_operation_attributes.num_links);
    const uint32_t reader_origins_base = halo_rt::kNeighborReaderRuntimeArgHeaderCount +
                                         num_ag_inputs * neighbor_reader_tensor_descriptor_field_count +
                                         halo_rt::kNeighborMulticastBlockOriginsOffset;
    const uint32_t writer_origins_base = halo_rt::kNeighborWriterRuntimeArgHeaderCount +
                                         num_ag_inputs * halo_rt::kNeighborWriterTensorDescriptorFieldCount +
                                         halo_rt::kNeighborMulticastBlockOriginsOffset;
    for (uint32_t exchange = 0; exchange < halo_exchanges.size(); ++exchange) {
        const auto& plan = halo_exchanges[exchange];
        const auto sources = runtime_chunked_sliding_layout.send_sources(ring_write_plan.transport_rank, plan.hop);
        const uint32_t tail_rows = runtime_chunked_sliding_layout.hop_rows(plan.hop);
        const auto origin_rows =
            multicast_origin_rows(runtime_chunked_sliding_layout, ring_write_plan.transport_rank, plan);
        auto& reader_grid_args = GetRuntimeArgs(program, kNeighborHaloReaderKernelIndex + 2 * exchange);
        auto& writer_grid_args = GetRuntimeArgs(program, kNeighborHaloWriterKernelIndex + 2 * exchange);
        TT_FATAL(reader_grid_args.size() == writer_grid_args.size(), "Directional gather runtime grids disagree");
        for (uint32_t x = 0; x < reader_grid_args.size(); ++x) {
            TT_FATAL(
                reader_grid_args[x].size() == writer_grid_args[x].size(),
                "Directional gather runtime grid columns disagree");
            for (uint32_t y = 0; y < reader_grid_args[x].size(); ++y) {
                auto& reader_args = reader_grid_args[x][y];
                auto& writer_args = writer_grid_args[x][y];
                if (reader_args.size() == 0 && writer_args.size() == 0) {
                    continue;
                }
                TT_FATAL(
                    reader_args.size() != 0 && writer_args.size() != 0, "Directional gather worker pair is incomplete");
                for (uint32_t in = 0; in < num_ag_inputs; ++in) {
                    const uint32_t reader_base = halo_rt::kNeighborReaderRuntimeArgHeaderCount +
                                                 in * neighbor_reader_tensor_descriptor_field_count;
                    const uint32_t writer_base = halo_rt::kNeighborWriterRuntimeArgHeaderCount +
                                                 in * halo_rt::kNeighborWriterTensorDescriptorFieldCount;
                    TT_FATAL(
                        writer_args.size() >= writer_base + halo_rt::kNeighborWriterTensorDescriptorFieldCount,
                        "Directional gather writer descriptor is incomplete");
                    const uint32_t input_Wt = ag_inputs[in]->padded_shape()[3] / tt::constants::TILE_WIDTH;
                    const uint32_t halo_pages = tail_rows * input_Wt;
                    const uint32_t link = reader_args[halo_rt::kNeighborReaderRuntimeArgHeaderCount - 1];
                    const uint32_t pages = sources.count * halo_pages;
                    const uint32_t start =
                        link * (pages / links_per_exchange) + std::min(link, pages % links_per_exchange);
                    const uint32_t end =
                        (link + 1) * (pages / links_per_exchange) + std::min(link + 1, pages % links_per_exchange);
                    reader_args[reader_base + halo_rt::kNeighborReaderInputTileStartFieldOffset] = start;
                    reader_args[reader_base + halo_rt::kNeighborReaderInputTileEndFieldOffset] = end;
                    reader_args[reader_base + halo_rt::kNeighborReaderFirstOriginFieldOffset] =
                        sources.first_start_tile * input_Wt;
                    reader_args[reader_base + halo_rt::kNeighborReaderSecondOriginFieldOffset] =
                        sources.second_start_tile * input_Wt;
                    reader_args[reader_base + halo_rt::kNeighborReaderHaloPagesFieldOffset] = halo_pages;
                    writer_args[writer_base + halo_rt::kNeighborWriterInputTileStartFieldOffset] = start;
                    writer_args[writer_base + halo_rt::kNeighborWriterInputTileEndFieldOffset] = end;
                }
                for (uint32_t i = 0; i < origin_rows.size(); ++i) {
                    write_runtime_arg(
                        reader_args, reader_origins_base + i, origin_rows[i], "neighbor_halo_reader.origin");
                    write_runtime_arg(
                        writer_args, writer_origins_base + i, origin_rows[i], "neighbor_halo_writer.origin");
                }
            }
        }
    }

    // Resolve each kernel's argument grid once per dispatch. Keep these references local:
    // descriptor application may change the backing storage before the next cache hit.
    auto& compute_grid_args = GetRuntimeArgs(program, kComputeKernelIndex);
    auto& reader_grid_args = GetRuntimeArgs(program, kReaderKernelIndex);
    auto* writer_grid_args = patch_kv_pad_rotation ? &GetRuntimeArgs(program, kWriterKernelIndex) : nullptr;
    const auto validate_grid = [&](const auto& grid_args) {
        TT_FATAL(grid_args.size() >= layout.grid_size.x, "RingJoint runtime argument grid is missing columns");
        for (uint32_t x = 0; x < layout.grid_size.x; ++x) {
            TT_FATAL(grid_args[x].size() >= layout.grid_size.y, "RingJoint runtime argument grid is missing rows");
        }
    };
    validate_grid(compute_grid_args);
    validate_grid(reader_grid_args);
    if (writer_grid_args != nullptr) {
        validate_grid(*writer_grid_args);
    }
    // These fields occupy adjacent slots in the descriptor. Validate the layout once,
    // then check and write each complete block on every core, including inactive receivers.
    TT_FATAL(
        layout.reader_active_ring_iter_mask == layout.reader_logical_nt + 1 &&
            layout.writer_active_ring_iter_mask == layout.writer_logical_nt + 1 &&
            layout.writer_single_valid_kv_chunk_mask == layout.writer_logical_nt + 2 &&
            layout.compute_q_pre_wrap_start_tile == layout.compute_logical_nt + 1 &&
            layout.compute_q_pre_wrap_tile_count == layout.compute_logical_nt + 2 &&
            layout.compute_q_post_wrap_start_tile == layout.compute_logical_nt + 3 &&
            layout.compute_q_valid_tile_count == layout.compute_logical_nt + 4 &&
            layout.compute_active_ring_iter_mask == layout.compute_logical_nt + 5,
        "RingJoint scalar runtime argument blocks must be contiguous");
    const std::array reader_values = {
        runtime_plan.logical_nt,
        kSlidingActiveRingIterMask,
        runtime_plan.kv_pad_q_mapping.q_pre_wrap_start_tile,
        runtime_plan.kv_pad_q_mapping.q_pre_wrap_tile_count,
        runtime_plan.kv_pad_q_mapping.q_post_wrap_start_tile,
        runtime_plan.kv_pad_q_mapping.q_valid_tile_count};
    const std::array writer_values = {
        runtime_plan.logical_nt, kSlidingActiveRingIterMask, kSlidingSingleValidKvChunkMask};
    const std::array compute_values = {
        runtime_plan.logical_nt,
        runtime_plan.kv_pad_q_mapping.q_pre_wrap_start_tile,
        runtime_plan.kv_pad_q_mapping.q_pre_wrap_tile_count,
        runtime_plan.kv_pad_q_mapping.q_post_wrap_start_tile,
        runtime_plan.kv_pad_q_mapping.q_valid_tile_count,
        kSlidingActiveRingIterMask};
    for (uint32_t i = 0; i < num_cores; ++i) {
        const CoreCoord core = {i % layout.grid_size.x, i / layout.grid_size.x};

        // Patch EVERY core, exactly as the create-time build sets these scalars on all cores
        // unconditionally, including cores with no Q chunks.
        auto& compute_args = compute_grid_args[core.x][core.y];

        auto& reader_args = reader_grid_args[core.x][core.y];
        if (patch_indexed_kv_cache) {
            write_runtime_arg(
                reader_args, layout.reader_kv_cache_batch_idx, kv_cache_batch_idx, "reader.kv_cache_batch_idx");
        }
        if (!patch_kv_pad_rotation) {
            continue;
        }

        write_runtime_arg_block(reader_args, layout.reader_logical_nt, reader_values, "reader.scalars");

        auto& writer_args = (*writer_grid_args)[core.x][core.y];
        write_runtime_arg_block(writer_args, layout.writer_logical_nt, writer_values, "writer.scalars");
        write_runtime_arg_block(compute_args, layout.compute_logical_nt, compute_values, "compute.scalars");
    }
}

}  // namespace

namespace ttnn::prim {

namespace {

// Per-coord ProgramDescriptor build. Pulled into an anonymous-namespace helper so
// create_workload_descriptor() can loop coords and reuse this body verbatim. The
// op-specific name suffix avoids Unity-build collisions with the sibling ring
// sdpa factories that share the same helper signature.
// Descriptor construction must keep host/runtime argument layouts together.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
tt::tt_metal::ProgramDescriptor build_ring_joint_sdpa_program_descriptor(
    const Gemma4SlidingRingSDPAParams& args,
    const Gemma4SlidingRingSDPAInputs& tensor_args,
    Gemma4SlidingRingSDPAResult& output_tensors,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_FATAL(
        mesh_dispatch_coordinate.has_value(),
        "build_ring_joint_sdpa_program_descriptor requires mesh_dispatch_coordinate");
    const auto& coord = mesh_dispatch_coordinate.value();
    /*
    The QKV inputs are fractured on the sequence dimension across ring_size.
    The sequence length comes in padded such that it is divisible by `TILE_HEIGHT * ring_size`.
    Therefore each device has `padded_N / ring_size` local tokens.

    Naming:
        - padded_N: the global, padded sequence length
        - kv_local_padded_N: local shard of padded sequence length for K/V (== padded_N / ring_size)
        - q_local_padded_N: local Q seq length. For chunked prefill < kv_local_padded_N; otherwise equal.
        - logical_n: the logical global sequence length. logical_n <= padded_N.
        - L: the full (global) joint sequence length
        - L_local: per-device joint length. Equals L/ring_size on the sharded path, or L on the replicated path.

    input_tensor_q: B x NH  x q_local_padded_N  x DH
    input_tensor_k: B x NHK x kv_local_padded_N x DH
    input_tensor_v: B x NH  x kv_local_padded_N x DH

    gathered_input_tensor_k: B x NHK x padded_N x DH
    gathered_input_tensor_v: B x NH  x padded_N x DH

    Replicated joint path (logical_l == 0 or joint seq == L):
        joint_tensor_q/k/v: B x NH x L x DH  (full joint on every device)
        joint_output_tensor: B x NH x L x DH

    Sharded joint path (logical_l > 0 and joint seq == L / ring_size):
        joint_tensor_q/k/v: B x NH x L_local x DH  (one shard per device)
        gathered_joint_k/v: B x NH x L x DH         (scratch buffer; filled by fused all-gather)
        joint_output_tensor: B x NH x L_local x DH

    output_tensor: B x NH x q_local_padded_N x DH

    The algorithm is roughly described below.
    - for each ring iteration:
        - read a Q chunk from input_tensor_q
        - for each KV chunk in kv_local_padded_N:
            - on the first ring iteration, read from local input_tensor_k and input_tensor_v
            - otherwise, read from gathered_input_tensor_k and gathered_input_tensor_v
            - Replicated joint: when ring_id == ring_size-1, also read from joint_tensor_k/v (full L).
            - Sharded joint: on every ring iteration, read one L_local shard from gathered_joint_k/v
              (or from the local joint_tensor_k/v when ring_id == this device's ring_index).
            - if the KV chunk is from the non-joint input and contains the global token index (logical_n - 1),
    generate a mask
            - else if the KV chunk is from non-joint input and contains the local token index (kv_local_padded_N - 1),
    generate an attention mask
            - else if the KV chunk is from the joint input and contains the local token index (L_local - 1),
    generate a mask
            - compute attention
        - write the output Q chunk
        - if this is not the first ring iteration, do the LSE update.
    */

    log_debug(tt::LogOp, "Gemma4SlidingRingSDPA create_descriptor");

    const auto& input_tensor_q = tensor_args.input_q;
    const auto& input_tensor_k = tensor_args.input_k;
    const bool v_shares_k_buffer = tensor_args.v_shares_k_buffer();
    const auto& input_tensor_v = tensor_args.input_v.has_value() ? tensor_args.input_v.value() : input_tensor_k;

    const RingJointInputParams joint_input_params = resolve_ring_joint_input_params(args, tensor_args);
    const Tensor* joint_tensor_q = joint_input_params.joint_q;
    const Tensor* joint_tensor_k = joint_input_params.joint_k;
    const Tensor* joint_tensor_v = joint_input_params.joint_v;
    const Tensor* gathered_joint_tensor_k = joint_input_params.gathered_joint_k;
    const Tensor* gathered_joint_tensor_v = joint_input_params.gathered_joint_v;
    const bool joint_is_sharded = joint_input_params.joint_is_sharded;

    const auto& gathered_input_tensor_k = tensor_args.gathered_k;
    const auto& gathered_input_tensor_v =
        tensor_args.gathered_v.has_value() ? tensor_args.gathered_v.value() : gathered_input_tensor_k;
    const auto& attention_sink = tensor_args.attention_sink;
    const bool use_attention_sink = attention_sink.has_value();

    auto& output_tensor = output_tensors[GEMMA4_SLIDING_RING_SDPA_OUTPUT_IDX];
    auto& joint_output_tensor = output_tensors[GEMMA4_SLIDING_RING_SDPA_JOINT_OUTPUT_IDX];
    auto& stats_output_tensor = output_tensors[GEMMA4_SLIDING_RING_SDPA_STATS_OUTPUT_IDX];

    std::size_t q_chunk_size = args.get_q_chunk_size();
    std::size_t k_chunk_size = args.get_k_chunk_size();

    tt::tt_metal::ProgramDescriptor desc;

    auto* mesh_device = input_tensor_q.device();
    const RingWritePlan ring_write_plan = build_ring_write_plan(args, tensor_args, coord);
    const uint32_t transport_rank = ring_write_plan.transport_rank;
    const uint32_t forward_writes_expected = ring_write_plan.forward_writes_expected;
    const uint32_t backward_writes_expected = ring_write_plan.backward_writes_expected;
    const auto& forward_coord = ring_write_plan.forward_coord;
    const auto& backward_coord = ring_write_plan.backward_coord;

    log_debug(tt::LogOp, "transport rank: {}", transport_rank);
    log_debug(tt::LogOp, "is_causal: {}", args.is_causal);
    log_debug(tt::LogOp, "is_balanced: {}", args.is_balanced);

    auto scale = args.scale;
    if (not scale.has_value()) {
        scale = 1.0f / std::sqrt(static_cast<float>(input_tensor_q.logical_shape()[-1]));
    }

    std::optional<ttnn::prim::RingSDPAFusedOpSignaler> sdpa_fused_op_signaler = ttnn::prim::RingSDPAFusedOpSignaler();

    // Minimally use matmul fused op signaler
    sdpa_fused_op_signaler->init_all_gather(
        args.all_gather_operation_attributes.ring_size,
        transport_rank,
        forward_writes_expected,
        backward_writes_expected);

    const auto& q_shape = input_tensor_q.logical_shape();
    const auto& k_shape = gathered_input_tensor_k.logical_shape();

    log_debug(tt::LogOp, "q_shape: {}", q_shape);
    log_debug(tt::LogOp, "k_shape (gathered): {}", k_shape);
    if (tensor_args.gathered_v.has_value()) {
        log_debug(tt::LogOp, "v_shape (gathered): {}", tensor_args.gathered_v->logical_shape());
    } else {
        log_debug(
            tt::LogOp,
            "v_shape (latent): [B={}, NHV=1, N=0, DH={}]",
            q_shape[0],
            tensor_args.v_head_dim(args.latent_v_head_dim));
    }

    // q_local_padded_N (Q rows per device) can be shorter than kv_local_padded_N for chunked prefill.
    // Metadata uses an on-device cache-slot value, but needs the same single-slot program structure.
    const bool slot_from_metadata = tensor_args.has_metadata();
    const bool indexed_kv_cache = ttnn::prim::indexed_kv_cache_active(args, tensor_args);
    // Latent-V mode: V tensors are omitted and the reader reads V from K's buffer.
    // Lanes: B is the lane count (validation forces Q batch 1) and each lane owns q_shape[2] / B of Q's stacked rows.
    const uint32_t num_lanes = tensor_args.num_lanes();
    const uint32_t B = q_shape[0] * num_lanes;
    const uint32_t NH = q_shape[1];
    const uint32_t NHK = k_shape[1];
    const uint32_t NHV = tensor_args.v_num_heads();
    const uint32_t DH = q_shape[3];
    const uint32_t q_local_padded_N = tensor_args.q_lane_rows();
    const uint32_t kv_local_padded_N = tensor_args.local_kv_seq_len();
    const uint32_t ring_size = static_cast<uint32_t>(args.all_gather_operation_attributes.ring_size);
    const uint32_t gathered_padded_N = k_shape[2];
    const uint32_t global_padded_N = kv_local_padded_N * ring_size;
    const uint32_t sliding_window_size = args.sliding_window_size.value_or(0);
    const bool has_sliding_window = sliding_window_size > 0;
    // The kernels carry only the sliding path: no dense ring all-gather, KV chains or multicast.
    TT_FATAL(has_sliding_window, "Gemma4SlidingRingSDPA requires a sliding window");
    // Circular sliding KV cache slab count (0 = unbounded), derived from the cache/Q geometry
    // (validated divisible with >= 2 slabs). Wraps the local-slab derivation in
    // sliding_window_work_plan.hpp for the host halo layout, the reader, and the compute kernel.
    const uint32_t circular_kv_slab_count = derived_kv_slab_count(args, tensor_args);
    const uint32_t kv_cache_batch_idx = args.cache_batch_idx().value_or(0);
    // L / L_local resolved once in resolve_ring_joint_input_params (full vs per-device joint seq).
    const uint32_t L = joint_input_params.L;
    const uint32_t L_local = joint_input_params.L_local;
    // True (unpadded) joint token count. Equals L on the replicated/aligned path; smaller than the
    // padded L when the sharded joint prompt leaves pad rows on the global tail.
    const uint32_t logical_l = joint_input_params.logical_l;
    const uint32_t vDH = tensor_args.v_head_dim(args.latent_v_head_dim);

    const uint32_t q_local_padded_Nt = q_local_padded_N / tt::constants::TILE_HEIGHT;
    const uint32_t kv_local_padded_Nt = kv_local_padded_N / tt::constants::TILE_HEIGHT;
    const uint32_t gathered_padded_Nt = gathered_padded_N / tt::constants::TILE_HEIGHT;
    // Find unpadded sequence lengths in tiles
    const uint32_t Lt = tt::div_up(L, tt::constants::TILE_HEIGHT);
    const uint32_t Lt_local = tt::div_up(L_local, tt::constants::TILE_HEIGHT);
    // True joint length in tiles (number of tiles holding any real joint token). Drives the
    // per-ring-iteration joint tail mask, mirroring logical_nt for the spatial path.
    const uint32_t logical_lt = tt::div_up(logical_l, tt::constants::TILE_HEIGHT);
    const uint32_t DHt = DH / tt::constants::TILE_WIDTH;
    const uint32_t vDHt = vDH / tt::constants::TILE_WIDTH;
    const bool kv_pad_from_metadata = tensor_args.kv_pad_from_metadata();
    const bool kv_pad_rotation_enabled = ttnn::prim::kv_pad_rotation_active(args, tensor_args);
    const RingJointRuntimePlan runtime_plan = build_runtime_plan(args, tensor_args, ring_write_plan);
    const RingJointRuntimeArgLayout runtime_arg_layout = get_runtime_arg_layout(args, tensor_args);
    const uint32_t logical_nt = runtime_plan.logical_nt;
    const ring_joint::ChunkedQMapping& kv_pad_q_mapping = runtime_plan.kv_pad_q_mapping;

    /*
    For non-causal case we must provide a padded mask if the K sequence length has been padded
    Note that we dont have this issue in non-causal case if Q is padded, since those pad tokens
    don't affect attention of unpadded tokens.
    In causal case, the causal mask takes care of masking K pad tokens.
    */

    const uint32_t Sq_chunk_t = q_chunk_size / tt::constants::TILE_HEIGHT;
    const uint32_t Sk_chunk_t = k_chunk_size / tt::constants::TILE_HEIGHT;

    // One global chunk is one Q slab per ring device.
    const uint32_t q_chunk_group_tile_count = q_local_padded_Nt * ring_size;
    // kernel_chunked drives the chunked-prefill math in the kernels.
    // Cross runs the non-causal full-prefill path on chunked-shaped tensors, so it is excluded; the
    // kernel-level is_causal flag carries the legacy local-frame causal-stamp semantics (chunked
    // prefill supersedes it via absolute-coords stamps). Both are derived once in build_runtime_plan.
    const bool kernel_chunked = runtime_plan.kernel_chunked;
    const bool kernel_is_causal = runtime_plan.kernel_is_causal;
    const ring_joint::ChunkedSlidingHaloLayout chunked_sliding_halo_layout =
        ring_joint::build_chunked_sliding_halo_layout(
            q_local_padded_Nt,
            Sk_chunk_t,
            sliding_window_size,
            tt::constants::TILE_HEIGHT,
            ring_size,
            logical_nt,
            gathered_padded_Nt,
            circular_kv_slab_count,
            args.kv_actual_isl.has_value() ? std::optional<uint32_t>(*args.kv_actual_isl / tt::constants::TILE_HEIGHT)
                                           : std::nullopt);
    TT_FATAL(
        kernel_chunked && chunked_sliding_halo_layout.uses_neighbor_halo(),
        "Sliding K/V requires neighbor-halo geometry; gathered rows={}, global rows={}",
        gathered_padded_N,
        global_padded_N);
    TT_FATAL(
        gathered_padded_N < global_padded_N,
        "Sliding K/V requires a compact halo buffer; gathered rows={}, global rows={}",
        gathered_padded_N,
        global_padded_N);
    TT_FATAL(
        gathered_padded_Nt >= chunked_sliding_halo_layout.halo_tile_rows,
        "Compact sliding K/V buffer has {} tile rows but requires at least {}",
        gathered_padded_Nt,
        chunked_sliding_halo_layout.halo_tile_rows);
    // Placeholder lengths: KV-pad rotation patches the logical length per dispatch, so it stays out of the kernels'
    // compile-time args.
    const bool has_logical_n_tensor = tensor_args.has_logical_n_tensor();
    const bool has_logical_l_tensor = tensor_args.has_logical_l_tensor();
    const uint32_t global_n_partial_col = args.logical_n % tt::constants::TILE_HEIGHT;
    const uint32_t compile_time_logical_n = kv_pad_rotation_enabled ? 0 : static_cast<uint32_t>(args.logical_n);
    const uint32_t compile_time_logical_nt = kv_pad_rotation_enabled ? 0 : logical_nt;
    const uint32_t compile_time_global_n_partial_col = kv_pad_rotation_enabled ? 0 : global_n_partial_col;
    const uint32_t joint_l_partial_col = logical_l % tt::constants::TILE_HEIGHT;

    const uint32_t num_local_q_chunks = tt::div_up(q_local_padded_N, q_chunk_size);
    // Q chunking uses L_local (per-device shard on sharded path, full L on replicated).
    const uint32_t num_joint_q_chunks = tt::div_up(L_local, q_chunk_size);
    const uint32_t num_q_chunks = num_local_q_chunks + num_joint_q_chunks;
    const uint32_t num_local_k_chunks = tt::div_up(kv_local_padded_N, k_chunk_size);
    // Sharded joint: per-iteration K chunk count for one shard (L_local). Replicated: full L.
    // Kernels process this many joint K chunks on every ring iteration (sharded) or just the last (replicated).
    const uint32_t num_joint_k_chunks = tt::div_up(L_local, k_chunk_size);

    log_debug(tt::LogOp, "B: {}", B);
    log_debug(tt::LogOp, "NH: {}", NH);
    log_debug(tt::LogOp, "NHK: {}", NHK);
    log_debug(tt::LogOp, "NHV: {}", NHV);
    log_debug(tt::LogOp, "L: {}", L);
    log_debug(tt::LogOp, "DH: {}", DH);
    log_debug(tt::LogOp, "vDH: {}", vDH);

    // Log padded dimensions
    log_debug(tt::LogOp, "q_local_padded_N: {}", q_local_padded_N);
    log_debug(tt::LogOp, "kv_local_padded_N: {}", kv_local_padded_N);
    log_debug(tt::LogOp, "L: {}", L);

    // Log tile dimensions
    log_debug(tt::LogOp, "DHt: {}", DHt);
    log_debug(tt::LogOp, "vDHt: {}", vDHt);
    log_debug(tt::LogOp, "q_local_padded_Nt: {}", q_local_padded_Nt);
    log_debug(tt::LogOp, "kv_local_padded_Nt: {}", kv_local_padded_Nt);
    log_debug(tt::LogOp, "Lt: {}", Lt);

    // Log chunking parameters
    log_debug(tt::LogOp, "Sq_chunk_t: {}", Sq_chunk_t);
    log_debug(tt::LogOp, "Sk_chunk_t: {}", Sk_chunk_t);
    log_debug(tt::LogOp, "num_local_q_chunks: {}", num_local_q_chunks);
    log_debug(tt::LogOp, "num_joint_q_chunks: {}", num_joint_q_chunks);
    log_debug(tt::LogOp, "q_chunk_size: {}", q_chunk_size);
    log_debug(tt::LogOp, "k_chunk_size: {}", k_chunk_size);
    log_debug(tt::LogOp, "num_q_chunks: {}", num_q_chunks);
    log_debug(tt::LogOp, "num_local_k_chunks: {}", num_local_k_chunks);
    log_debug(tt::LogOp, "num_joint_k_chunks: {}", num_joint_k_chunks);

    MeshDevice* device = input_tensor_q.device();

    // The precision recipe owns the numerics (transformer/sdpa/device/recipe/sdpa_precision_policy.hpp).
    namespace recipes = ttnn::operations::transformer::sdpa::detail;
    const recipes::PrecisionPolicy policy =
        recipes::resolve_recipe_policy(input_tensor_q, gathered_input_tensor_k, args.precision, scale, args.program_config);

    CoreCoord grid_size = args.program_config.has_value() ? args.program_config->compute_with_storage_grid_size
                                                          : mesh_device->compute_with_storage_grid_size();

    auto core_grid = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    CoreRangeSet core_grid_set(core_grid);
    uint32_t num_cores = grid_size.x * grid_size.y;

    // Init fused op signaler — descriptor-pattern equivalent of
    // RingSDPAFusedOpSignaler::init_fused_op. The signaler stores the receiver-cores
    // noc list and two signal semaphore IDs (one for forward, one for backward).
    // Semaphore IDs match insertion order into desc.semaphores.
    {
        sdpa_fused_op_signaler->fused_op_signaler_mode = ttnn::experimental::ccl::FusedOpSignalerMode::MULTI;
        sdpa_fused_op_signaler->fused_op_receiver_cores_noc.clear();
        const auto cores = tt::tt_metal::corerange_to_cores(core_grid_set, std::nullopt, /*row_wise=*/true);
        for (const auto& core : cores) {
            sdpa_fused_op_signaler->fused_op_receiver_cores_noc.push_back(
                mesh_device->worker_core_from_logical_core(core));
        }
        const uint32_t fused_sem0_id = static_cast<uint32_t>(desc.semaphores.size());
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = fused_sem0_id,
            .core_type = tt::CoreType::WORKER,
            .core_ranges = core_grid_set,
            .initial_value = 0,
        });
        const uint32_t fused_sem1_id = static_cast<uint32_t>(desc.semaphores.size());
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = fused_sem1_id,
            .core_type = tt::CoreType::WORKER,
            .core_ranges = core_grid_set,
            .initial_value = 0,
        });
        sdpa_fused_op_signaler->fused_op_receiver_signal_semaphores.clear();
        sdpa_fused_op_signaler->fused_op_receiver_signal_semaphores.push_back(fused_sem0_id);
        sdpa_fused_op_signaler->fused_op_receiver_signal_semaphores.push_back(fused_sem1_id);
        sdpa_fused_op_signaler->num_fused_op_cores_to_signal =
            sdpa_fused_op_signaler->fused_op_receiver_cores_noc.size();
        sdpa_fused_op_signaler->initialized_fused_op = true;
    }

    log_debug(tt::LogOp, "num_cores: {}", num_cores);
    log_debug(
        tt::LogOp, "mesh_device->compute_with_storage_grid_size(): {}", mesh_device->compute_with_storage_grid_size());
    log_debug(tt::LogOp, "grid_size: {}", grid_size);

    TT_FATAL(
        num_cores <= mesh_device->compute_with_storage_grid_size().x * mesh_device->compute_with_storage_grid_size().y,
        "Provided grid must not contain more cores than the device. Got {} cores, expected at most {} cores.",
        num_cores,
        mesh_device->compute_with_storage_grid_size().x * mesh_device->compute_with_storage_grid_size().y);

    /**
     * This parallelization scheme is efficient because it divides the global work,
     * the total number of Q chunks across all batches and heads, evenly across the cores.
     *
     */
    const uint32_t all_heads_num_q_chunks = B * NH * num_q_chunks;
    const uint32_t max_q_per_core = tt::div_up(all_heads_num_q_chunks, num_cores);
    log_debug(tt::LogOp, "max_q_per_core: {}", max_q_per_core);

    // The recipe's matmul subblock height: one Q tile row per group for the FP32 recipes, two for the paired BF16
    // recipes (an odd Q chunk ends with a single row; the reader then pushes Q one row at a time).
    const uint32_t qk_out_subblock_h = policy.fp32_destination ? 1 : 2;
    // The writer drains the output in the rows compute packs (one per row); pairs of rows when the Q chunk is a whole
    // number of them, so a pair never straddles the output CB's wrap.
    const uint32_t writer_out_row_group_h = Sq_chunk_t % 2 == 0 ? 2 : 1;
    // K split is not applied (as RingJointSDPA's sliding path): the recipes run every Q chunk's K chunks on its own core.
    // Lanes still reject it (validate_on_program_cache_miss). Segmented accumulation is dense-only.
    if (args.program_config.has_value() && args.program_config->k_splits > 1) {
        static std::once_flag ignored_once;
        std::call_once(ignored_once, [] {
            log_warning(
                tt::LogOp,
                "Gemma4SlidingRingSDPA ignores SDPAProgramConfig k_splits: the precision recipes run every Q chunk's "
                "K chunks on its own core.");
        });
    }
    TT_FATAL(
        !(args.program_config.has_value() && args.program_config->segmented_accumulation),
        "segmented_accumulation is not supported with a sliding window. Turn segmented_accumulation off.");

    const float scale_value = scale.value_or(1.0f);
    const uint32_t scale_packed = std::bit_cast<uint32_t>(scale_value);

    // log scale
    log_debug(tt::LogOp, "scale: {}", scale_value);

    // Enable per-head zigzag for load balancing in balanced causal mode
    // Requires even num_q_chunks for symmetric light/heavy work distribution
    // Chunked prefill rides its own absolute-coords path, not the legacy local-frame causal stamp,
    // so the zigzag asymmetry doesn't apply — gate on kernel_is_causal, not args.is_causal.
    const bool enable_zigzag_balancing = args.is_balanced && kernel_is_causal && (num_q_chunks % 2 == 0);

    // The shared op's ring-work masks, kept as runtime args: sliding folds its local and predecessor ranges into one
    // synthetic ring iteration (see kSlidingActiveRingIterMask).
    const uint32_t active_ring_iter_mask = kSlidingActiveRingIterMask;
    const uint32_t single_valid_kv_chunk_mask = kSlidingSingleValidKvChunkMask;
    // Transport rank is tensor rank on a cluster-axis ring; the kernels keep the rank-mapping slots of the
    // shared op's full-mesh snake, pinned to identity.
    constexpr RingAttentionRankMapping rank_mapping{};

    // Cores actually issuing Q reads. When the flat q-chunk distribution is smaller
    // than the grid the trailing cores get zero work; zigzag distributes pairs, so
    // the unit count is total_pairs = all_heads_num_q_chunks / 2.
    const uint32_t num_active_cores = enable_zigzag_balancing ? std::min(num_cores, all_heads_num_q_chunks / 2)
                                                              : std::min(num_cores, all_heads_num_q_chunks);

    std::vector<uint32_t> reader_compile_time_args = {
        B,
        NH,
        NHK,
        DHt,
        vDHt,
        Sq_chunk_t,
        Sk_chunk_t,
        q_local_padded_Nt,
        kv_local_padded_Nt,
        Lt,
        L,
        num_local_q_chunks,
        num_joint_q_chunks,
        num_local_k_chunks,
        num_joint_k_chunks,
        num_q_chunks,
        args.all_gather_operation_attributes.ring_size,
        qk_out_subblock_h,
        kernel_is_causal,
        args.is_balanced,
        static_cast<uint32_t>(enable_zigzag_balancing),
        // Reader chunked-prefill control.
        static_cast<uint32_t>(kernel_chunked),
        num_active_cores,
        q_chunk_group_tile_count,
        static_cast<uint32_t>(indexed_kv_cache),
        static_cast<uint32_t>(kv_pad_rotation_enabled),
        NHV,
        static_cast<uint32_t>(v_shares_k_buffer),
        static_cast<uint32_t>(use_attention_sink),
        sliding_window_size,
        gathered_padded_Nt,
        static_cast<uint32_t>(slot_from_metadata),
        static_cast<uint32_t>(kv_pad_from_metadata),
        // Slots 33-34, sharded-joint path: Lt_local (Q-axis tile count = L_local/TILE_HEIGHT) and flag.
        Lt_local,
        static_cast<uint32_t>(joint_is_sharded),
        // Slot 35: true (unpadded) joint length in tiles (twins spatial logical_nt). The reader uses it
        // to skip joint K chunks that lie entirely beyond the real joint tail (padding).
        logical_lt,
        // Slots 36-39: transport-to-tensor rank mapping.
        static_cast<uint32_t>(rank_mapping.full_mesh),
        static_cast<uint32_t>(rank_mapping.orientation),
        rank_mapping.mesh_rows,
        rank_mapping.mesh_cols,
        // Slot 40: circular sliding KV slab count (0 = unbounded). Feeds the reader's
        // build_sliding_q_work_plan so local slab addressing wraps identically to the host halo
        // layout and the compute kernel.
        circular_kv_slab_count,
        // Slots 41-42: logical-length transport. The reader NoC-reads the live values, re-derives
        // logical_nt / logical_lt / the ring masks, and fans them to compute via cb_kv_pad_derived.
        // Tensor accessors start at slot 43.
        static_cast<uint32_t>(has_logical_n_tensor),
        static_cast<uint32_t>(has_logical_l_tensor),
    };

    TensorAccessorArgs(input_tensor_q.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(input_tensor_k.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(input_tensor_v.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(gathered_input_tensor_k.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(gathered_input_tensor_v.buffer()).append_to(reader_compile_time_args);
    if (L != 0) {
        TensorAccessorArgs(joint_tensor_q->buffer()).append_to(reader_compile_time_args);
        TensorAccessorArgs(joint_tensor_k->buffer()).append_to(reader_compile_time_args);
        TensorAccessorArgs(joint_tensor_v->buffer()).append_to(reader_compile_time_args);
    }
    // Sharded-joint path: the gathered joint K/V accessors follow the local joint accessors, matching
    // get_post_tensor_args_offset<has_joint_inputs, has_gathered_joint_k, ...>() in the reader kernel.
    // They must precede the attention-sink and metadata accessors below, which the kernel places at
    // post_joint_tensor_args_offset / post_tensor_args_offset respectively.
    if (joint_is_sharded) {
        TensorAccessorArgs(gathered_joint_tensor_k->buffer()).append_to(reader_compile_time_args);
        TensorAccessorArgs(gathered_joint_tensor_v->buffer()).append_to(reader_compile_time_args);
    }
    TensorAccessorArgs(attention_sink.has_value() ? attention_sink->buffer() : nullptr)
        .append_to(reader_compile_time_args);
    // Metadata accessors follow the tensor accessors (metadata path only) and precede the CB compile args;
    // the reader kernel gates their offsets on slot_from_metadata / kv_pad_from_metadata.
    // slot_id and kv_actual_isl are SEPARATELY allocated single-page DRAM tensors that can land in different
    // DRAM banks, so each needs its OWN accessor -- a shared accessor's dspec (bank for page 0) is baked
    // from one buffer and reads the wrong bank for the other (kv read silently returned 0, breaking the
    // rotation derivation). The writer already appends kv_actual_isl's own accessor for the same reason.
    if (slot_from_metadata) {
        TensorAccessorArgs(tensor_args.slot_id->buffer()).append_to(reader_compile_time_args);
        if (kv_pad_from_metadata) {
            TensorAccessorArgs(tensor_args.kv_actual_isl->buffer()).append_to(reader_compile_time_args);
            if (tensor_args.has_kv_valid_end()) {
                TensorAccessorArgs(tensor_args.kv_valid_end->buffer()).append_to(reader_compile_time_args);
            }
        }
    }
    // Appended as a pair when either is present (null accessor for the absent one) so kernel offsets do not
    // depend on which was supplied. Separate accessors: the two single-page tensors can land in different
    // DRAM banks, the same trap documented for slot_id/kv_actual_isl.
    if (has_logical_n_tensor || has_logical_l_tensor) {
        TensorAccessorArgs(has_logical_n_tensor ? tensor_args.logical_n_tensor->buffer() : nullptr)
            .append_to(reader_compile_time_args);
        TensorAccessorArgs(has_logical_l_tensor ? tensor_args.logical_l_tensor->buffer() : nullptr)
            .append_to(reader_compile_time_args);
    }

    std::vector<uint32_t> writer_compile_time_args = {
        B,
        NH,
        DHt,
        vDHt,
        Sq_chunk_t,
        Sk_chunk_t,
        q_local_padded_Nt,
        kv_local_padded_Nt,
        compile_time_logical_n,
        Lt_local,  // slot 9: per-device joint tile count (Lt_local == Lt on replicated path)
        L,
        num_local_q_chunks,
        num_joint_q_chunks,
        num_local_k_chunks,
        num_joint_k_chunks,
        num_q_chunks,
        0u,  // slot 16: reserved (the writer generates the recipe's identities)
        scale_packed,
        args.all_gather_operation_attributes.ring_size,
        compile_time_global_n_partial_col,
        joint_l_partial_col,
        0u,  // slot 21: reserved
        kernel_is_causal,
        args.is_balanced,
        static_cast<uint32_t>(enable_zigzag_balancing),
        static_cast<std::uint32_t>(writer_out_row_group_h),
        static_cast<uint32_t>(kernel_chunked),
        q_chunk_group_tile_count,
        sliding_window_size,
        // Slot 29: trace-safe KV-pad derivation -- the writer recomputes logical_nt + masks from
        // metadata[1] on-device (it's dataflow).
        static_cast<uint32_t>(kv_pad_from_metadata),
        // Slot 30: sharded-joint flag. When true, one shard arrives per ring iteration.
        static_cast<uint32_t>(joint_is_sharded),
        // Slot 31: true (unpadded) joint length in tiles (twins spatial logical_nt). Combined with
        // joint_l_partial_col it drives the joint mask-generation gate.
        logical_lt,
        // Slots 32-35: transport-to-tensor rank mapping.
        static_cast<uint32_t>(rank_mapping.full_mesh),
        static_cast<uint32_t>(rank_mapping.orientation),
        rank_mapping.mesh_rows,
        rank_mapping.mesh_cols,
        // Slots 36-37: logical-length transport. The writer is a dataflow kernel, so it NoC-reads the live
        // values itself rather than sharing the reader's L1 mailbox. Output accessors start at slot 38.
        static_cast<uint32_t>(has_logical_n_tensor),
        static_cast<uint32_t>(has_logical_l_tensor),
    };

    TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_time_args);
    TensorAccessorArgs(joint_output_tensor.buffer()).append_to(writer_compile_time_args);
    TensorAccessorArgs(stats_output_tensor.buffer()).append_to(writer_compile_time_args);
    if (kv_pad_from_metadata) {
        TensorAccessorArgs(tensor_args.kv_actual_isl->buffer()).append_to(writer_compile_time_args);
        if (tensor_args.has_kv_valid_end()) {
            TensorAccessorArgs(tensor_args.kv_valid_end->buffer()).append_to(writer_compile_time_args);
        }
    }
    if (has_logical_n_tensor || has_logical_l_tensor) {
        TensorAccessorArgs(has_logical_n_tensor ? tensor_args.logical_n_tensor->buffer() : nullptr)
            .append_to(writer_compile_time_args);
        TensorAccessorArgs(has_logical_l_tensor ? tensor_args.logical_l_tensor->buffer() : nullptr)
            .append_to(writer_compile_time_args);
    }

    std::vector<uint32_t> compute_compile_time_args = {
        NH,
        DHt,
        vDHt,
        Sq_chunk_t,
        Sk_chunk_t,
        q_local_padded_Nt,
        kv_local_padded_Nt,
        compile_time_logical_n,
        compile_time_logical_nt,
        Lt,
        L,
        num_local_q_chunks,
        num_local_k_chunks,
        num_joint_k_chunks,
        num_q_chunks,
        args.all_gather_operation_attributes.ring_size,
        // Slots 16-25: reserved (the recipe sets its own matmul subblocks).
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        scale_packed,
        0u,  // slot 27: reserved
        compile_time_global_n_partial_col,
        joint_l_partial_col,
        kernel_is_causal,
        args.is_balanced,
        static_cast<uint32_t>(enable_zigzag_balancing),
        static_cast<uint32_t>(kernel_chunked),
        q_chunk_group_tile_count,
        static_cast<uint32_t>(kv_pad_rotation_enabled),
        static_cast<uint32_t>(v_shares_k_buffer),
        static_cast<uint32_t>(use_attention_sink),
        sliding_window_size,
        // Slot 39: trace-safe KV-pad derivation. When set, compute reads logical_nt / q-mapping /
        // active_ring_iter_mask from cb_kv_pad_derived (produced by the reader) instead of its runtime
        // args, so a captured trace replays across chunks.
        static_cast<uint32_t>(kv_pad_from_metadata),
        // Slot 40: sharded-joint flag. When true, one shard per ring iteration; do_joint_kv fires every iter.
        static_cast<uint32_t>(joint_is_sharded),
        // Slot 41: true (unpadded) joint length in tiles (twins spatial logical_nt). Drives the
        // per-ring-iteration joint tail mask and the joint out-of-bounds K-chunk skip.
        logical_lt,
        // Slots 42-45: transport-to-tensor rank mapping.
        static_cast<uint32_t>(rank_mapping.full_mesh),
        static_cast<uint32_t>(rank_mapping.orientation),
        rank_mapping.mesh_rows,
        rank_mapping.mesh_cols,
        // Slot 46: circular sliding KV slab count (0 = unbounded). Feeds the compute-side
        // build_sliding_q_work_plan so it stays in lockstep with the reader.
        circular_kv_slab_count,
        // Slots 47-48: logical-length transport. Compute cannot NoC-read DRAM, so it takes the live values
        // from cb_kv_pad_derived, which the reader fills.
        static_cast<uint32_t>(has_logical_n_tensor),
        static_cast<uint32_t>(has_logical_l_tensor)};

    // Sliding work plans: the compact halo's predecessor slots, the most source ranges a Q chunk's plan holds, and
    // whether the local slab goes first (only for a multi-hop halo, whose exchange is long enough to hide; a one-hop
    // halo keeps the oldest-first order).
    const uint32_t sliding_halo_slot_count = chunked_sliding_halo_layout.halo_slot_count;
    const uint32_t sliding_halo_hops =
        ring_joint::chunked_sliding_halo_hop_count(chunked_sliding_halo_layout.halo_tile_rows, q_local_padded_Nt);
    const uint32_t sliding_max_source_ranges =
        ring_joint::sliding_q_work_plan_source_ranges(sliding_halo_hops, chunked_sliding_halo_layout.halo_slot_count);
    const uint32_t sliding_local_first = sliding_halo_hops > 1 ? 1 : 0;

    // Circular buffers: the recipe compute layout (recipe_compute_program, CBs 0-16 and 29-31; recipe_ring_cbs.hpp),
    // then the reader's derived lengths (RingJointSDPA's sliding layout less its checkpoint and sink CBs: a sliding Q
    // chunk normalizes in one pass and this op takes no sink). Every CB spans the whole worker grid.
    const tt::DataFormat q_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_q.dtype());
    const tt::DataFormat k_df = tt::tt_metal::datatype_to_dataformat_converter(gathered_input_tensor_k.dtype());
    const tt::DataFormat v_df = tt::tt_metal::datatype_to_dataformat_converter(gathered_input_tensor_v.dtype());
    const uint32_t q_tile_size = tt::tile_size(q_df);
    const uint32_t k_tile_size = tt::tile_size(k_df);
    const uint32_t v_tile_size = tt::tile_size(v_df);
    const auto v_storage = recipes::recipe_kv_storage(gathered_input_tensor_v.dtype());
    tt::tt_metal::ProgramDescriptor recipe_program = recipes::recipe_compute_program(
        policy,
        core_grid_set,
        1,
        Sq_chunk_t,
        Sk_chunk_t,
        DHt,
        scale,
        vDHt,
        v_storage != policy.selection.kv_storage ? std::optional<recipes::KVStorage>(v_storage) : std::nullopt,
        q_df);
    auto& compute_recipe_kernel = recipe_program.kernels.front();
    desc.cbs = std::move(recipe_program.cbs);
    for (auto& cb : desc.cbs) {
        auto& format = cb.format_descriptors.front();
        if (format.buffer_index == recipe_ring_cb::q) {
            // Q double-buffered only when a core runs several Q chunks.
            cb.total_size = Sq_chunk_t * DHt * (max_q_per_core > 1 ? 2 : 1) * q_tile_size;
        } else if (format.buffer_index == recipe_ring_cb::k) {
            cb.total_size = Sk_chunk_t * DHt * 2 * k_tile_size;
        } else if (format.buffer_index == recipe_ring_cb::v) {
            cb.total_size = Sk_chunk_t * vDHt * 2 * v_tile_size;
        } else if (format.buffer_index == recipe_ring_cb::out) {
            // The output tensor is BF16 whatever Q's type.
            const uint32_t tiles = cb.total_size / format.page_size;
            format.data_format = tt::DataFormat::Float16_b;
            format.page_size = tt::tile_size(tt::DataFormat::Float16_b);
            cb.total_size = tiles * format.page_size;
        }
    }
    // Reader-to-compute mailbox for the metadata-derived logical geometry (lanes: one record per lane).
    const uint32_t derived_page_bytes = num_lanes > 1 ? ring_joint::kDerivedLanePageBytes : 64;
    desc.cbs.push_back(CBDescriptor{
        .total_size = derived_page_bytes,
        .core_ranges = core_grid_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(recipe_ring_cb::derived),
            .data_format = tt::DataFormat::UInt32,
            .page_size = derived_page_bytes}}},
    });
    // Fit L1 as RingJointSDPA does: first give up Q prefetch, then output staging beyond one row group, then the fused
    // chunks' CBs (every K chunk then runs on the reduce path, slower than either).
    const uint64_t l1_available = recipes::recipe_free_l1(*device);
    uint64_t cb_bytes = recipes::recipe_cb_bytes(desc);
    if (cb_bytes > l1_available && max_q_per_core > 1) {
        for (auto& cb : desc.cbs) {
            if (cb.format_descriptors.front().buffer_index == recipe_ring_cb::q) {
                cb.total_size /= 2;
            }
        }
        cb_bytes = recipes::recipe_cb_bytes(desc);
        log_debug(tt::LogOp, "Gemma4SlidingRingSDPA recipe: single-slot Q to fit L1");
    }
    for (const uint32_t rows : {2u, 1u}) {
        if (cb_bytes <= l1_available || rows < std::min(Sq_chunk_t, 2u)) {
            break;
        }
        for (auto& cb : desc.cbs) {
            if (cb.format_descriptors.front().buffer_index == recipe_ring_cb::out) {
                cb.total_size = std::min<uint32_t>(cb.total_size, rows * vDHt * cb.format_descriptors.front().page_size);
            }
        }
        cb_bytes = recipes::recipe_cb_bytes(desc);
        log_debug(tt::LogOp, "Gemma4SlidingRingSDPA recipe: output staged in {} row(s) to fit L1", rows);
    }
    if (cb_bytes > l1_available) {
        cb_bytes -= recipes::recipe_drop_fused(desc.cbs, compute_recipe_kernel.named_compile_time_args, Sq_chunk_t);
        log_debug(tt::LogOp, "Gemma4SlidingRingSDPA recipe: unfused chunks to fit L1");
    }
    TT_FATAL(
        cb_bytes <= l1_available,
        "Gemma4SlidingRingSDPA recipe needs {} bytes of L1 per core at Q{}/K{}, but only {} are available; use a smaller "
        "Q or K chunk",
        cb_bytes,
        Sq_chunk_t * tt::constants::TILE_HEIGHT,
        Sk_chunk_t * tt::constants::TILE_HEIGHT,
        l1_available);
    if (device->arch() == tt::ARCH::WORMHOLE_B0 &&
        recipes::named_arg(compute_recipe_kernel.named_compile_time_args, "recipe_fused") != 0) {
        // Wormhole runs every ring K chunk on the reduce path, as RingJointSDPA (its fused chunks deadlock in the ring's
        // one-chunk segments; open item there).
        desc.cbs.erase(std::remove_if(desc.cbs.begin(), desc.cbs.end(), recipes::recipe_fused_cb), desc.cbs.end());
        recipes::set_named_arg(compute_recipe_kernel.named_compile_time_args, "recipe_fused", 0);
    }

    const std::vector<uint32_t> reader_cb_compile_time_args = {
        recipe_ring_cb::q, recipe_ring_cb::k, recipe_ring_cb::v, recipe_ring_cb::q, recipe_ring_cb::derived};
    reader_compile_time_args.insert(
        reader_compile_time_args.end(), reader_cb_compile_time_args.begin(), reader_cb_compile_time_args.end());

    auto* const q_buf = input_tensor_q.buffer();
    auto* const k_buf = input_tensor_k.buffer();
    auto* const v_buf = input_tensor_v.buffer();
    auto* const gathered_k_buf = gathered_input_tensor_k.buffer();
    auto* const gathered_v_buf = gathered_input_tensor_v.buffer();
    auto* const attention_sink_buf = attention_sink.has_value() ? attention_sink->buffer() : nullptr;
    auto* const out_buf = output_tensor.buffer();
    auto* const joint_out_buf = joint_output_tensor.buffer();
    auto* const stats_buf = stats_output_tensor.buffer();

    // Per-core flat Q range. Sliding attention reads K/V independently on every core, so no store-and-forward
    // chains or K/V multicast are built and no core runs padded multicast slots.
    struct CoreWork {
        CoreCoord physical_core;
        uint32_t global_q_start = 0;
        uint32_t global_q_count = 0;
    };

    std::vector<CoreWork> core_work(num_cores);

    // Evenly distribute flat global q chunks across cores
    const uint32_t total_q_chunks = B * NH * num_q_chunks;

    uint32_t base_chunks_per_core = 0;
    uint32_t extra_chunks_per_core = 0;
    uint32_t cores_doing_extra_work = 0;
    if (enable_zigzag_balancing) {
        log_debug(tt::LogOp, "Enabling zigzag balancing with even num_q_chunks: {}", num_q_chunks);
        const uint32_t total_pairs = total_q_chunks / 2;
        cores_doing_extra_work = total_pairs % num_cores;
        base_chunks_per_core = (num_cores == 0) ? 0 : (total_pairs / num_cores) * 2;
        extra_chunks_per_core = (num_cores == 0) ? 0 : 2;
    } else {
        cores_doing_extra_work = total_q_chunks % num_cores;
        base_chunks_per_core = (num_cores == 0) ? 0 : (total_q_chunks / num_cores);
        extra_chunks_per_core = (num_cores == 0) ? 0 : 1;
    }

    uint32_t next_global_chunk = 0;

    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};
        uint32_t chunk_count = base_chunks_per_core + ((i < cores_doing_extra_work) ? extra_chunks_per_core : 0);
        if (next_global_chunk >= total_q_chunks) {
            chunk_count = 0;
        } else if (chunk_count > total_q_chunks - next_global_chunk) {
            chunk_count = total_q_chunks - next_global_chunk;
        }

        auto& work = core_work.at(i);
        work.physical_core = device->worker_core_from_logical_core(core);
        work.global_q_start = next_global_chunk;
        work.global_q_count = chunk_count;

        next_global_chunk += chunk_count;
    }

    // The kernels keep the shared op's compile-time ABI, whose LAST compile-time arg is the rotated Q split's
    // maximum owned chunk count; they read it back from that position. Sliding never rotates Q, so it is always 0.
    // The TT_FATAL below pins that "last" so a future append fails loudly instead of silently handing the kernels
    // some other value.
    constexpr uint32_t rotated_max_slots_ct = 0;
    for (auto* args : {&reader_compile_time_args, &writer_compile_time_args, &compute_compile_time_args}) {
        args->push_back(rotated_max_slots_ct);
    }
    const std::array<size_t, 3> ct_arg_sizes_with_rotated_last = {
        reader_compile_time_args.size(), writer_compile_time_args.size(), compute_compile_time_args.size()};

    // Build kernel descriptors locally so we can append per-core runtime args
    // before pushing them into desc.kernels at the end.
    KernelDescriptor reader_kernel{};
    reader_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/gemma4/sliding_ring_sdpa/device/kernels/dataflow/"
        "sliding_ring_sdpa_reader.cpp";
    reader_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel.core_ranges = core_grid_set;
    // The kernels read rotated_max_slots as their final compile-time arg, so nothing may be
    // appended after it (in-place patching of earlier indices is fine and does not change size).
    TT_FATAL(
        reader_compile_time_args.size() == ct_arg_sizes_with_rotated_last[0] &&
            writer_compile_time_args.size() == ct_arg_sizes_with_rotated_last[1] &&
            compute_compile_time_args.size() == ct_arg_sizes_with_rotated_last[2],
        "rotated_max_slots must stay the last compile-time arg; args were appended after it "
        "(reader {}->{}, writer {}->{}, compute {}->{})",
        ct_arg_sizes_with_rotated_last[0],
        reader_compile_time_args.size(),
        ct_arg_sizes_with_rotated_last[1],
        writer_compile_time_args.size(),
        ct_arg_sizes_with_rotated_last[2],
        compute_compile_time_args.size());

    reader_kernel.compile_time_args = reader_compile_time_args;
    reader_kernel.config = ReaderConfigDescriptor{};
    // Layout: metadata block first (when present), then the logical-length pair, so the kernel's base index
    // is ring_joint::kReaderMetadataCommonArgCount (or kWriterMetadataCommonArgCount) when its metadata block
    // is present, else 0. Both slots are emplaced whenever either tensor is present. Passed as Buffer* (not
    // ->address()) so a program-cache hit re-patches them if the tensor was reallocated.
    const bool has_logical_length_tensor = has_logical_n_tensor || has_logical_l_tensor;
    const auto append_logical_length_common_args = [&](KernelDescriptor::RTArgList& into) {
        if (!has_logical_length_tensor) {
            return;
        }
        for (const auto* logical_tensor : {&tensor_args.logical_n_tensor, &tensor_args.logical_l_tensor}) {
            if (logical_tensor->has_value()) {
                into.push_back((*logical_tensor)->buffer());
            } else {
                into.push_back(uint32_t{0});
            }
        }
    };
    if (slot_from_metadata || has_logical_length_tensor) {
        KernelDescriptor::RTArgList reader_common_args;
        if (slot_from_metadata) {
            // Bound as buffers so a cache hit with different metadata tensors follows them; the reader takes
            // these as get_common_arg_val(0..4).
            reader_common_args.push_back(tensor_args.slot_id->buffer());
            reader_common_args.push_back(args.kv_cache_num_layers);
            reader_common_args.push_back(args.kv_cache_layer_idx);
            reader_common_args.push_back(std::min(
                tensor_args.input_k.logical_shape()[0],
                tensor_args.input_v.has_value() ? tensor_args.input_v->logical_shape()[0]
                                                : tensor_args.input_k.logical_shape()[0]));
            reader_common_args.push_back(tensor_args.kv_actual_isl->buffer());
        }
        append_logical_length_common_args(reader_common_args);
        if (tensor_args.has_kv_valid_end()) {
            reader_common_args.push_back(tensor_args.kv_valid_end->buffer());
        }
        reader_kernel.emplace_common_runtime_args(reader_common_args);
    }

    KernelDescriptor writer_kernel{};
    writer_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/gemma4/sliding_ring_sdpa/device/kernels/dataflow/"
        "sliding_ring_sdpa_writer.cpp";
    writer_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel.core_ranges = core_grid_set;
    writer_kernel.compile_time_args = writer_compile_time_args;
    writer_kernel.config = WriterConfigDescriptor{};
    if (kv_pad_from_metadata || has_logical_length_tensor) {
        KernelDescriptor::RTArgList writer_common_args;
        if (kv_pad_from_metadata) {
            // Bound as a buffer so a cache hit with a different metadata tensor follows it; the writer takes
            // it as get_common_arg_val(0).
            writer_common_args.push_back(tensor_args.kv_actual_isl->buffer());
        }
        append_logical_length_common_args(writer_common_args);
        if (tensor_args.has_kv_valid_end()) {
            writer_common_args.push_back(tensor_args.kv_valid_end->buffer());
        }
        writer_kernel.emplace_common_runtime_args(writer_common_args);
    }

    KernelDescriptor compute_kernel{};
    compute_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/gemma4/sliding_ring_sdpa/device/kernels/compute/sliding_ring_sdpa.cpp";
    compute_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel.core_ranges = core_grid_set;
    compute_kernel.compile_time_args = compute_compile_time_args;
    // One KV stripe (kv_stripe_split is a divisor, so off is 1). The recipe's named arguments (after the L1 fallbacks
    // above) on every kernel; the ring continuation without checkpoints (every sliding Q chunk normalizes in its one
    // pass) and the sequence-frame positional masks (recipe/recipe_config.hpp).
    auto recipe_args = recipes::recipe_named_args(compute_recipe_kernel.named_compile_time_args);
    recipes::set_named_arg(recipe_args, "recipe_ring", 1);
    recipes::set_named_arg(recipe_args, "recipe_ring_stream_state", 0);
    recipes::set_named_arg(recipe_args, "recipe_ring_mask", 2);
    recipes::set_named_arg(recipe_args, "recipe_ring_window", sliding_window_size);
    recipes::set_named_arg(recipe_args, "recipe_ring_rotation", kv_pad_rotation_enabled ? 1 : 0);
    recipes::set_named_arg(recipe_args, "recipe_ring_circular", circular_kv_slab_count > 1 ? 1 : 0);
    for (auto* kernel : {&reader_kernel, &writer_kernel, &compute_kernel}) {
        kernel->named_compile_time_args = {
            {"kv_stripe_split", 1u},
            {"num_lanes", num_lanes},
            {"kv_valid_end", tensor_args.has_kv_valid_end() ? 1u : 0u},
            {"sliding_halo_slot_count", sliding_halo_slot_count},
            {"sliding_max_source_ranges", sliding_max_source_ranges},
            {"sliding_local_first", sliding_local_first},
        };
        kernel->named_compile_time_args.insert(
            kernel->named_compile_time_args.end(), recipe_args.begin(), recipe_args.end());
    }
    reader_kernel.named_compile_time_args.emplace_back(
        "k_row_Wt", input_tensor_k.padded_shape()[3] / tt::constants::TILE_WIDTH);
    reader_kernel.named_compile_time_args.emplace_back(
        "v_row_Wt", input_tensor_v.padded_shape()[3] / tt::constants::TILE_WIDTH);
    // The recipe's compute config (fidelity, FP32 DEST and unpack-to-dest modes) and defines.
    compute_kernel.defines = compute_recipe_kernel.defines;
    compute_kernel.config = compute_recipe_kernel.config;
    // Code size, as RingJointSDPA's masked rings: Blackhole unpack and pack at -O2 (math is -O2 in the kernel).
    // Wormhole keeps the default: its link-time inlining needs it (the math LLK's address-mode setup takes constants).
    if (device->arch() != tt::ARCH::WORMHOLE_B0) {
        compute_kernel.opt_level = recipes::recipe_opt_level(tt::tt_metal::KernelBuildOptLevel::O2);
    }

    // Set reader rt args
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};

        const auto& work = core_work.at(i);
        const uint32_t global_q_start = work.global_q_start;
        const uint32_t global_q_end = global_q_start + work.global_q_count;

        // log the above
        log_debug(tt::LogOp, "core: {}", i);
        log_debug(tt::LogOp, "x={},y={}", core.x, core.y);
        log_debug(tt::LogOp, "global_q_start: {}", global_q_start);
        log_debug(tt::LogOp, "global_q_end: {}", global_q_end);

        CheckedRuntimeArgList reader_args;
        reader_args.push_back(q_buf);
        reader_args.push_back(k_buf);
        reader_args.push_back(v_buf);
        reader_args.push_back(gathered_k_buf);
        reader_args.push_back(gathered_v_buf);
        if (L != 0) {
            reader_args.push_back(joint_tensor_q->buffer());
            reader_args.push_back(joint_tensor_k->buffer());
            reader_args.push_back(joint_tensor_v->buffer());
        }
        reader_args.push_back(attention_sink_buf);
        // Read by the kernel right after attention_sink_addr and before global_q_start.
        if (joint_is_sharded) {
            reader_args.push_back(gathered_joint_tensor_k->buffer());
            reader_args.push_back(gathered_joint_tensor_v->buffer());
        }
        reader_args.push_back(global_q_start);
        reader_args.push_back(global_q_end);
        reader_args.push_checked(
            runtime_arg_layout.reader_kv_cache_batch_idx, kv_cache_batch_idx, "reader.kv_cache_batch_idx");
        reader_args.push_checked(runtime_arg_layout.reader_logical_nt, logical_nt, "reader.logical_nt");
        reader_args.push_checked(
            runtime_arg_layout.reader_active_ring_iter_mask, active_ring_iter_mask, "reader.active_ring_iter_mask");
        reader_args.push_back(kv_pad_q_mapping.q_pre_wrap_start_tile);
        reader_args.push_back(kv_pad_q_mapping.q_pre_wrap_tile_count);
        reader_args.push_back(kv_pad_q_mapping.q_post_wrap_start_tile);
        reader_args.push_back(kv_pad_q_mapping.q_valid_tile_count);

        // Inject fused-op synchronization RT args (AllGather) here; it will append to reader_args
        std::vector<uint32_t> reader_signaler_args;
        sdpa_fused_op_signaler->push_ring_sdpa_fused_op_rt_args(reader_signaler_args);
        reader_args.append(reader_signaler_args);
        reader_kernel.emplace_runtime_args(core, reader_args.args);

        // Writer args
        CheckedRuntimeArgList writer_args;
        writer_args.push_back(out_buf);
        writer_args.push_back(joint_out_buf);
        writer_args.push_back(stats_buf);
        writer_args.push_back(global_q_start);
        writer_args.push_back(global_q_end);
        writer_args.push_checked(runtime_arg_layout.writer_logical_nt, logical_nt, "writer.logical_nt");
        writer_args.push_checked(
            runtime_arg_layout.writer_active_ring_iter_mask, active_ring_iter_mask, "writer.active_ring_iter_mask");
        writer_args.push_checked(
            runtime_arg_layout.writer_single_valid_kv_chunk_mask,
            single_valid_kv_chunk_mask,
            "writer.single_valid_kv_chunk_mask");
        std::vector<uint32_t> writer_signaler_args;
        sdpa_fused_op_signaler->push_ring_sdpa_fused_op_rt_args(writer_signaler_args);
        writer_args.append(writer_signaler_args);
        writer_kernel.emplace_runtime_args(core, writer_args.args);

        // Compute args
        CheckedRuntimeArgList compute_args;
        compute_args.push_back(global_q_start);
        compute_args.push_back(global_q_end);
        compute_args.push_back(ring_size);
        compute_args.push_back(transport_rank);
        compute_args.push_back(forward_writes_expected);
        compute_args.push_back(backward_writes_expected);
        compute_args.push_checked(runtime_arg_layout.compute_logical_nt, logical_nt, "compute.logical_nt");
        compute_args.push_checked(
            runtime_arg_layout.compute_q_pre_wrap_start_tile,
            kv_pad_q_mapping.q_pre_wrap_start_tile,
            "compute.q_pre_wrap_start_tile");
        compute_args.push_checked(
            runtime_arg_layout.compute_q_pre_wrap_tile_count,
            kv_pad_q_mapping.q_pre_wrap_tile_count,
            "compute.q_pre_wrap_tile_count");
        compute_args.push_checked(
            runtime_arg_layout.compute_q_post_wrap_start_tile,
            kv_pad_q_mapping.q_post_wrap_start_tile,
            "compute.q_post_wrap_start_tile");
        compute_args.push_checked(
            runtime_arg_layout.compute_q_valid_tile_count,
            kv_pad_q_mapping.q_valid_tile_count,
            "compute.q_valid_tile_count");
        compute_args.push_checked(
            runtime_arg_layout.compute_active_ring_iter_mask, active_ring_iter_mask, "compute.active_ring_iter_mask");
        // The shared op's static multicast loop count. Without multicast padding it is this core's own Q count.
        compute_args.push_back(global_q_end - global_q_start);
        compute_kernel.emplace_runtime_args(core, compute_args.args);
    }

    // Push the SDPA kernels into desc before invoking the halo exchange helper so
    // the helper appends its own kernels after these. Their indices in
    // desc.kernels will be 0/1/2 respectively (they are the first kernels appended).
    desc.kernels.push_back(std::move(reader_kernel));
    desc.kernels.push_back(std::move(writer_kernel));
    desc.kernels.push_back(std::move(compute_kernel));

    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler> all_gather_fused_op_signaler =
        ttnn::experimental::ccl::AllGatherFusedOpSignaler();

    all_gather_fused_op_signaler->init_fused_op(
        sdpa_fused_op_signaler->fused_op_receiver_cores_noc,
        sdpa_fused_op_signaler->fused_op_receiver_signal_semaphores,
        sdpa_fused_op_signaler->fused_op_signaler_mode);

    std::vector<Tensor> all_gather_input_tensors = {input_tensor_k};
    std::vector<Tensor> all_gather_output_tensors = {gathered_input_tensor_k};
    if (tensor_args.input_v.has_value()) {
        all_gather_input_tensors.push_back(input_tensor_v);
        all_gather_output_tensors.push_back(gathered_input_tensor_v);
    }
    // Sharded-joint path: include joint K/V in the same fused gather so gathered_joint_k/v are
    // produced by the gather that actually runs on device. Same as spatial K/V: the AG omits each
    // device's own local slice from the gathered buffer; the SDPA reader fetches that slice from
    // the local joint tensor when ring_id == ring_index, and remote slices from the gathered buffer.
    // Sliding-window attention does not support joint tokens, so this never feeds the halo path below.
    if (joint_is_sharded) {
        TT_FATAL(
            tensor_args.gathered_joint_k.has_value() && tensor_args.gathered_joint_v.has_value(),
            "joint_is_sharded but gathered_joint_k/v not set in tensor_args");
        all_gather_input_tensors.push_back(*joint_tensor_k);
        all_gather_output_tensors.push_back(tensor_args.gathered_joint_k.value());
        all_gather_input_tensors.push_back(*joint_tensor_v);
        all_gather_output_tensors.push_back(tensor_args.gathered_joint_v.value());
    }
    // A halo that fits in one Q slab is a single exchange with the +1 neighbour. A wider one reaches
    // several predecessors, and hop d ships the tail of the slab d positions back into its own block
    // of the receiver's compact buffer; the SDPA reader waits for all of them before its K loop. A
    // hop that lands back on this device (halo spanning the whole ring) is a local cache read and
    // gets no exchange. validate_on_program_cache_miss bounds the hop count.
    // plan_chunked_sliding_halo_exchanges groups the hops into exchanges.
    const uint32_t halo_remote_hops = chunked_sliding_halo_layout.remote_hop_count();
    const auto& ag_attrs = args.all_gather_operation_attributes;
    const auto halo_exchanges = plan_halo_exchanges(args, chunked_sliding_halo_layout, transport_rank);
    // The first exchange's workers collect every arrival (SlidingHaloConfig::collects_arrivals).
    TT_FATAL(halo_exchanges.front().hop == 1, "The first sliding halo exchange must cover hop 1");
    const bool halo_multicast = halo_exchanges.front().multicast;
    const uint32_t halo_exchange_count = halo_exchanges.size();
    // Unicast hops beyond the link count time-share a link (see SlidingHaloConfig).
    const uint32_t links_per_exchange = halo_links_per_exchange(halo_exchanges, ag_attrs.num_links);
    // Each exchange needs its own workers. The CCL offset points at a reserved column (or row), so
    // walk along it: allocation order is the same for every call, so the blocks are disjoint.
    const bool halo_column_major = ag_attrs.core_allocation_strategy == ttnn::ccl::CoreAllocationStrategy::COL_MAJOR;
    const auto halo_exchange_core_grid_offset = [&](uint32_t exchange) {
        const uint32_t worker_stride = exchange * links_per_exchange;
        return CoreCoord{
            args.ccl_core_grid_offset.x + (halo_column_major ? 0 : worker_stride),
            args.ccl_core_grid_offset.y + (halo_column_major ? worker_stride : 0)};
    };
    // Same allocator the helper uses, so the two agree.
    std::vector<CoreCoord> first_exchange_link_cores;
    // Each exchange's link-0 worker, which carries its link hand-off.
    std::vector<CoreCoord> exchange_lead_cores;
    exchange_lead_cores.reserve(halo_exchange_count);
    for (uint32_t exchange = 0; exchange < halo_exchange_count; ++exchange) {
        auto [exchange_core_range, cores] = ttnn::ccl::choose_worker_cores(
            links_per_exchange,
            1,
            mesh_device,
            ag_attrs.sub_device_id,
            halo_exchange_core_grid_offset(exchange),
            std::nullopt,
            ag_attrs.core_allocation_strategy);
        exchange_lead_cores.push_back(cores.front());
        if (exchange == 0) {
            first_exchange_link_cores = std::move(cores);
        }
    }
    // Every exchange of a multi-hop halo increments the receiver's first-exchange workers, one per link;
    // every device allocates its first exchange at the same offset, so the cores match.
    std::vector<CoreCoord> halo_rendezvous;
    if (halo_remote_hops > 1) {
        for (const auto& core : first_exchange_link_cores) {
            halo_rendezvous.push_back(mesh_device->worker_core_from_logical_core(core));
        }
    }
    // Forward and backward are separate fabric connections, so each direction time-shares only its
    // own links: within a direction, exchange k (0-based) runs on link k % span and, beyond the
    // first span exchanges, waits for exchange k - span to close its connection first. Queued
    // exchanges still overlap their DRAM reads with the predecessor's send, so only the fabric
    // transfer serialises.
    std::array<std::vector<uint32_t>, 2> direction_exchanges;  // [0] forward, [1] backward
    for (uint32_t exchange = 0; exchange < halo_exchange_count; ++exchange) {
        direction_exchanges[halo_exchanges[exchange].send_backward].push_back(exchange);
    }
    std::array<uint32_t, 2> direction_link_span{0, 0};
    for (uint32_t direction = 0; direction < 2; ++direction) {
        if (direction_exchanges[direction].empty()) {
            continue;
        }
        const auto& neighbour = direction == 0 ? forward_coord : backward_coord;
        TT_FATAL(
            neighbour.has_value(),
            "Sliding halo hop {} has no fabric neighbour",
            halo_exchanges[direction_exchanges[direction][0]].hop);
        const uint32_t links_available = tt::tt_fabric::get_forwarding_link_indices(
                                             *mesh_device,
                                             mesh_device->get_fabric_node_id(coord),
                                             mesh_device->get_fabric_node_id(neighbour.value()))
                                             .size();
        TT_FATAL(links_available >= 1, "Chunked sliding halo found no forwarding fabric link");
        // Every predecessor must use the same link count, or arrival counts would not match.
        TT_FATAL(
            !halo_multicast || links_available >= links_per_exchange,
            "Sliding halo multicast needs {} fabric links but only {} are available",
            links_per_exchange,
            links_available);
        // Never use more links than the caller asked for.
        direction_link_span[direction] = std::min(
            {static_cast<uint32_t>(direction_exchanges[direction].size()), ag_attrs.num_links, links_available});
    }
    const bool halo_links_shared = direction_exchanges[0].size() > direction_link_span[0] ||
                                   direction_exchanges[1].size() > direction_link_span[1];
    // One local semaphore, same id on every exchange's worker, carries the hand-off. It is allocated
    // before the helper's own per-core semaphores, so pick an id free on all of them.
    uint32_t halo_chain_semaphore_id = 0;
    if (halo_links_shared) {
        // Per-core ids are handed out densely from 0, so the largest first-free id is free on
        // every exchange core; asserted below, since a taken id would silently alias a semaphore.
        for (const auto& core : exchange_lead_cores) {
            const auto sem_id = desc.find_available_semaphore_id(core, tt::CoreType::WORKER);
            TT_FATAL(
                sem_id.has_value(), "Ran out of semaphore IDs on sliding-halo worker core ({}, {})", core.x, core.y);
            halo_chain_semaphore_id = std::max(halo_chain_semaphore_id, sem_id.value());
        }
        for (const auto& sem : desc.semaphores) {
            TT_FATAL(
                sem.core_type != tt::CoreType::WORKER || sem.id != halo_chain_semaphore_id ||
                    std::none_of(
                        exchange_lead_cores.begin(),
                        exchange_lead_cores.end(),
                        [&](const CoreCoord& core) { return sem.core_ranges.contains(core); }),
                "Sliding-halo link hand-off semaphore id {} is already used on a hop worker core",
                halo_chain_semaphore_id);
        }
        std::vector<CoreRange> chain_core_ranges;
        chain_core_ranges.reserve(exchange_lead_cores.size());
        for (const auto& core : exchange_lead_cores) {
            chain_core_ranges.emplace_back(core, core);
        }
        desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = halo_chain_semaphore_id,
            .core_type = tt::CoreType::WORKER,
            .core_ranges = CoreRangeSet(chain_core_ranges),
            .initial_value = 0,
        });
    }
    const auto collector_keys = halo_collected_arrival_keys(chunked_sliding_halo_layout, transport_rank);
    validate_halo_arrival_semaphores(ag_attrs.semaphore, collector_keys);
    for (uint32_t exchange = 0; exchange < halo_exchange_count; ++exchange) {
        const auto& plan = halo_exchanges[exchange];
        const uint32_t destination_rank = (transport_rank + plan.hop) % ring_size;
        const int32_t signed_hops =
            plan.send_backward ? -static_cast<int32_t>(plan.distance) : static_cast<int32_t>(plan.distance);
        // The nearest receiver; a multicast continues past it.
        const auto halo_destination_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
            tensor_args.input_q, coord, signed_hops, ag_attrs.topology, ag_attrs.cluster_axis);
        const auto halo_transport_coord = plan.send_backward ? backward_coord : forward_coord;
        TT_FATAL(
            halo_transport_coord.has_value() && halo_destination_coord.has_value(),
            "Sliding attention hop {} requires a route to ring device {}",
            plan.hop,
            destination_rank);
        // Exchanges beyond their direction's link count queue up behind the one `span` earlier in
        // the same direction, which shares their link, and hand it on to the one `span` later.
        const auto& lane = direction_exchanges[plan.send_backward];
        const uint32_t lane_span = direction_link_span[plan.send_backward];
        const uint32_t lane_index = std::find(lane.begin(), lane.end(), exchange) - lane.begin();
        const bool exchange_waits = lane_index >= lane_span;
        const bool exchange_signals = lane_index + lane_span < lane.size();
        const CoreCoord exchange_successor =
            exchange_signals
                ? mesh_device->worker_core_from_logical_core(exchange_lead_cores[lane[lane_index + lane_span]])
                : CoreCoord{0, 0};
        // send_to_next_start_Ht is linear in the chunk index, so on the scalar path the host relocates
        // the halo page ranges every dispatch (apply_ring_joint_scalar_runtime_args). A captured trace
        // never replays that, so on the metadata path hand the halo kernels the same kv_actual_isl the
        // rest of the op reads and let them derive the start themselves; the value here then serves as
        // the baked origin they shift away from.
        // The tail(s) this exchange ships follow from each receiver's Q mapping: two when block-cyclic
        // Q wraps. A multicast lists every hop's origin so its kernels can
        // group equal ones into runs. Metadata kernels re-derive them on-device each replay.
        const auto hop_sources = chunked_sliding_halo_layout.send_sources(transport_rank, plan.hop);
        const uint32_t hop_tail_rows = chunked_sliding_halo_layout.hop_rows(plan.hop);
        const SlidingHaloConfig neighbor_halo{
            .send_to_next_start_Ht = hop_sources.first_start_tile,
            .send_to_next_count_Ht = hop_sources.count * hop_tail_rows,
            .send_second_start_Ht = hop_sources.second_start_tile,
            .send_backward = plan.send_backward,
            .distance = plan.distance,
            .hop = plan.hop,
            .tail_tile_rows = hop_tail_rows,
            .dest_row_base = chunked_sliding_halo_layout.dest_row(transport_rank, plan.hop),
            .second_dest_row_base = chunked_sliding_halo_layout.dest_row(transport_rank, plan.hop, 1),
            .hop_origin_rows = multicast_origin_rows(chunked_sliding_halo_layout, transport_rank, plan),
            .link_base = lane_index % lane_span,
            .rendezvous_noc = halo_rendezvous,
            .arrival_key = halo_arrival_key(chunked_sliding_halo_layout, transport_rank, plan.hop),
            .collected_keys = plan.hop == 1 ? collector_keys : std::vector<uint32_t>{},
            .waits_for_predecessor = exchange_waits,
            .signals_successor = exchange_signals,
            .chain_semaphore_id = halo_chain_semaphore_id,
            .successor_noc_x = static_cast<uint32_t>(exchange_successor.x),
            .successor_noc_y = static_cast<uint32_t>(exchange_successor.y),
            .slot_id = tensor_args.has_metadata() ? &tensor_args.slot_id.value() : nullptr,
            .kv_actual_isl = tensor_args.has_metadata() ? &tensor_args.kv_actual_isl.value() : nullptr,
            .kv_cache_num_layers = args.kv_cache_num_layers,
            .kv_cache_layer_idx = args.kv_cache_layer_idx,
            .q_local_tile_rows = chunked_sliding_halo_layout.q_local_tile_rows,
            .halo_tile_rows = chunked_sliding_halo_layout.halo_tile_rows,
            .source_device = transport_rank,
            .num_lanes = num_lanes,
        };
        log_debug(
            tt::LogOp,
            "Chunked sliding K/V halo: device={}, exchange={}/{} hop={} -> device={} ({} {} x{}), link={}, "
            "chain(w={}, s={}), tail=[{}, {}), payload_rows={}, compact rows [{}, {})",
            transport_rank,
            exchange + 1,
            halo_exchange_count,
            plan.hop,
            destination_rank,
            plan.send_backward ? "backward" : "forward",
            plan.distance,
            plan.hop_count,
            neighbor_halo.link_base,
            exchange_waits,
            exchange_signals,
            neighbor_halo.send_to_next_start_Ht,
            neighbor_halo.send_to_next_start_Ht + neighbor_halo.send_to_next_count_Ht,
            neighbor_halo.send_to_next_count_Ht,
            neighbor_halo.dest_row_base,
            neighbor_halo.dest_row_base + neighbor_halo.send_to_next_count_Ht);
        sliding_halo_exchange_helper(
            desc,
            all_gather_input_tensors,
            coord,
            halo_transport_coord.value(),
            halo_destination_coord.value(),
            all_gather_output_tensors,
            links_per_exchange,
            ag_attrs.ring_size,
            transport_rank,
            ag_attrs.topology,
            ag_attrs.semaphore,
            ag_attrs.sub_device_id,
            all_gather_fused_op_signaler.value(),
            halo_exchange_core_grid_offset(exchange),
            ag_attrs.core_allocation_strategy,
            args.cache_batch_idx(),
            compute_gather_valid_Ht(args, tensor_args),
            neighbor_halo);
    }

    return desc;
}

}  // namespace

// Ring-joint SDPA returns a WorkloadDescriptor with one ProgramDescriptor per coord:
// transport rank / forward_coord / backward_coord (used by the halo exchange) all
// depend on the mesh coordinate, so descriptors cannot be shared across coords. Returning
// a WorkloadDescriptor (rather than a per-coord ProgramDescriptor) keeps the framework on
// its no-rebuild cache-hit fast path; the dynamic scalar runtime args (indexed kv-cache /
// kv-pad rotation) are still re-applied every dispatch by override_runtime_arguments below.
tt::tt_metal::WorkloadDescriptor Gemma4SlidingRingSDPAProgramFactory::create_workload_descriptor(
    const Gemma4SlidingRingSDPAParams& args,
    const Gemma4SlidingRingSDPAInputs& tensor_args,
    Gemma4SlidingRingSDPAResult& output_tensors,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor wd;
    const auto coords = tensor_coords.coords();
    wd.programs.reserve(coords.size());
    for (const auto& coord : coords) {
        auto desc = build_ring_joint_sdpa_program_descriptor(args, tensor_args, output_tensors, coord);
        wd.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
    }
    return wd;
}

Gemma4SlidingRingSDPAMeshWorkloadFactory::cached_mesh_workload_t
Gemma4SlidingRingSDPAMeshWorkloadFactory::create_mesh_workload(
    const Gemma4SlidingRingSDPAParams& args,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const Gemma4SlidingRingSDPAInputs& tensor_args,
    Gemma4SlidingRingSDPAResult& output_tensors) {
    return descriptor_adapter_t::create_mesh_workload(args, tensor_coords, tensor_args, output_tensors);
}

void Gemma4SlidingRingSDPAMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const Gemma4SlidingRingSDPAParams& args,
    const Gemma4SlidingRingSDPAInputs& tensor_args,
    Gemma4SlidingRingSDPAResult& output_tensors) {
    descriptor_adapter_t::apply_descriptor(cached_workload, args, tensor_args, output_tensors);

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        const ttnn::MeshCoordinate coord = coordinate_range.start_coord();
        TT_FATAL(
            coord == coordinate_range.end_coord(),
            "Expected Gemma4SlidingRingSDPA cached programs to cover a single coordinate, got range {} to {}",
            coord,
            coordinate_range.end_coord());
        apply_ring_joint_scalar_runtime_args(program, args, tensor_args, coord);
    }
}

}  // namespace ttnn::prim
