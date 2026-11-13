// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/sparse_ring_sdpa_program_factory.hpp"

#include <string>
#include "ttnn/operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/ring_attention_all_gather_async_multi_core_with_workers_program_factory.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_joint_chain_layout.hpp"
#include "ttnn/operations/transformer/sdpa/device/ring_joint_kv_chains.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_id_sequencer.hpp"
#include "ttnn/operations/transformer/sdpa/device/sdpa_subblock_utils.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/sparse_dense_units.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/sparse_ring_balance_plan.hpp"
#include <ttnn/global_semaphore.hpp>
#include "ttnn/tensor/tensor_ops.hpp"

#include <algorithm>
#include <array>
#include <vector>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <cmath>
#include <limits>
#include <memory>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <internal/cluster_noc_helpers.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <hostdevcommon/common_values.hpp>

using namespace tt::tt_metal;
using ttnn::Tensor;

namespace {

namespace ag_rt = ttnn::ring_attention_all_gather_async_detail;
namespace ag_dyn = ttnn::experimental::prim::ring_attention_all_gather_async_dynamic;
namespace ring_joint = ttnn::operations::transformer::sdpa::ring_joint;

// Host-side summary of which ring-loop iterations do useful SDPA work. Bits are indexed by ring_iter,
// not ring_id; kernels still advance their sync/ring-id sequence on every iter before checking the mask.
struct RingWorkMasks {
    uint32_t active_ring_iter_mask = 0;
    uint32_t single_valid_kv_chunk_mask = 0;
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
};

struct RingWritePlan {
    uint32_t transport_rank = 0;
    uint32_t forward_writes_expected = 0;
    uint32_t backward_writes_expected = 0;
    std::optional<ttnn::MeshCoordinate> forward_coord;
    std::optional<ttnn::MeshCoordinate> backward_coord;
};

constexpr uint32_t kReaderKernelIndex = 0;
constexpr uint32_t kWriterKernelIndex = 1;
constexpr uint32_t kComputeKernelIndex = 2;
// Q blocks a core may hold in L1 at once (the kernels' per-block state is a 32-bit bitmask).
constexpr uint32_t kResidentMaxSlots = 32;

// The all-gather appends reader-forward, writer-forward, reader-backward, writer-backward after the three SDPA kernels.
constexpr uint32_t kFirstAllGatherKernelIndex = 3;
constexpr uint32_t kAllGatherReaderForwardKernelIndex = kFirstAllGatherKernelIndex + ag_rt::kReaderForwardKernelOffset;
constexpr uint32_t kAllGatherWriterForwardKernelIndex = kFirstAllGatherKernelIndex + ag_rt::kWriterForwardKernelOffset;
constexpr uint32_t kAllGatherReaderBackwardKernelIndex =
    kFirstAllGatherKernelIndex + ag_rt::kReaderBackwardKernelOffset;
constexpr uint32_t kAllGatherWriterBackwardKernelIndex =
    kFirstAllGatherKernelIndex + ag_rt::kWriterBackwardKernelOffset;

// Runtime-arg offsets. Descriptor construction appends the same slots through CheckedRuntimeArgList, so layout
// edits fail on program creation.
// Q, K, V, gathered K, gathered V, and attention sink (always null).
constexpr uint32_t kReaderBaseBufferArgCount = 6;
constexpr uint32_t kReaderQWorkArgCount = 3;
constexpr uint32_t kRingJointChainConfigArgCount = ring_joint::kChainConfigRuntimeArgCount;
constexpr uint32_t kRingJointChainCompileArgCount = ring_joint::kChainCompileArgCount;
constexpr uint32_t kRingJointChainSemaphoreCompileArgCount = ring_joint::kChainSemaphoreCompileArgCount;
constexpr uint32_t kRingJointChainMcastEnabledCompileArgOffset = ring_joint::kChainMcastEnabledCompileArgOffset;
constexpr uint32_t kReaderBatchChainExtraArgCount = 1;
constexpr uint32_t kReaderGQAChainExtraArgCount = 1;
constexpr uint32_t kWriterBaseArgCount = 3;  // output address, Q range
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
            "SparseRingSDPA runtime arg {} expected index {}, got {} before append",
            name,
            expected_index,
            size);
        push_back(value);
    }
};

// Build the per-device ring-loop masks passed to reader/compute/writer. This mirrors the kernel ring-id order and
// marks the ring_iter entries whose shard has K/V inside logical_n.
RingWorkMasks build_ring_work_masks(
    const ttnn::prim::SparseRingSDPAParams& args,
    const ttnn::prim::SparseRingSDPAInputs& tensor_args,
    const RingWritePlan& ring_write_plan) {
    const uint32_t ring_size = static_cast<uint32_t>(args.all_gather_operation_attributes.ring_size);
    TT_FATAL(
        ring_size <= std::numeric_limits<uint32_t>::digits,
        "SparseRingSDPA host ring-work masks support up to {} ring iterations. Got ring_size={}",
        std::numeric_limits<uint32_t>::digits,
        ring_size);
    const uint32_t logical_nt = tt::div_up(static_cast<uint32_t>(args.logical_n), tt::constants::TILE_HEIGHT);
    const uint32_t kv_local_padded_Nt = tensor_args.local_kv_seq_len() / tt::constants::TILE_HEIGHT;
    const uint32_t k_chunk_tile_count = args.get_k_chunk_size() / tt::constants::TILE_HEIGHT;
    const uint32_t num_local_k_chunks = tt::div_up(tensor_args.local_kv_seq_len(), args.get_k_chunk_size());

    RingWorkMasks masks;
    RingIdSequencer seq(
        ring_write_plan.transport_rank,
        ring_size,
        ring_write_plan.backward_writes_expected,
        ring_write_plan.forward_writes_expected);
    // RingIdSequencer accepts a sync callback for kernel semaphore waits. Host planning only needs the
    // same ring-id sequence, so use a no-op callback.
    auto noop_sync = [](uint32_t, uint32_t) {};

    for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
        const uint32_t ring_id = seq.get_next_ring_id(noop_sync);
        uint32_t valid_kv_chunks = 0;
        for (uint32_t k_chunk = 0; k_chunk < num_local_k_chunks; ++k_chunk) {
            const uint32_t local_tile_start = k_chunk * k_chunk_tile_count;
            if (local_tile_start < kv_local_padded_Nt && ring_id * kv_local_padded_Nt + local_tile_start < logical_nt) {
                valid_kv_chunks++;
            }
        }
        if (valid_kv_chunks > 0) {
            masks.active_ring_iter_mask |= (1u << ring_iter);
        }
        if (valid_kv_chunks <= 1) {
            masks.single_valid_kv_chunk_mask |= (1u << ring_iter);
        }
    }

    return masks;
}

RingWritePlan build_ring_write_plan(
    const ttnn::prim::SparseRingSDPAParams& args,
    const ttnn::prim::SparseRingSDPAInputs& tensor_args,
    const ttnn::MeshCoordinate& coord) {
    RingWritePlan plan;
    const auto& ag = args.all_gather_operation_attributes;
    plan.transport_rank =
        ttnn::ccl::get_linearized_index_from_physical_coord(tensor_args.input_q, coord, ag.cluster_axis);
    plan.forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.input_q, coord, 1, ag.topology, ag.cluster_axis);
    plan.backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.input_q, coord, -1, ag.topology, ag.cluster_axis);

    auto [num_targets_forward, num_targets_backward, dynamic_alternate] =
        ttnn::ccl::get_forward_backward_configuration(ag.ring_size, plan.transport_rank, ag.topology);
    (void)dynamic_alternate;
    if (ag.topology == ttnn::ccl::Topology::Ring && plan.transport_rank % 2 == 0) {
        std::swap(num_targets_forward, num_targets_backward);
    }

    if (ag.topology == ttnn::ccl::Topology::Linear) {
        plan.forward_writes_expected = num_targets_backward;
        plan.backward_writes_expected = num_targets_forward;
    } else {
        TT_FATAL(ag.topology == ttnn::ccl::Topology::Ring, "Topology must be Linear or Ring");
        plan.forward_writes_expected = num_targets_forward;
        plan.backward_writes_expected = num_targets_backward;
    }

    return plan;
}

RingJointRuntimeArgLayout get_runtime_arg_layout(bool use_head_chain, bool k_uses_batch_chain, bool gqa_grouped_kv) {
    const uint32_t head_chain_args = use_head_chain ? kRingJointChainConfigArgCount : 0;
    const uint32_t batch_chain_args =
        k_uses_batch_chain ? (kRingJointChainConfigArgCount + kReaderBatchChainExtraArgCount) : 0;
    const uint32_t gqa_chain_args = gqa_grouped_kv ? (kRingJointChainConfigArgCount + kReaderGQAChainExtraArgCount) : 0;

    RingJointRuntimeArgLayout layout;
    layout.reader_kv_cache_batch_idx = kReaderBaseBufferArgCount + 2;
    layout.reader_logical_nt =
        kReaderBaseBufferArgCount + kReaderQWorkArgCount + head_chain_args + batch_chain_args + gqa_chain_args;
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

}  // namespace

namespace ttnn::prim {

namespace {

// dense_blocks split across devices (sparse_dense_units.hpp): the hop staging and receive buffers and the ready /
// arrival semaphores (per SDPA core, running counts of units staged for the + / - neighbour and arrived from the - / +
// neighbour), made once per cached workload (not with dense_partials_debug).
struct DenseResources {
    std::shared_ptr<Tensor> staging;
    std::shared_ptr<Tensor> recv;
    GlobalSemaphore ready_plus;
    GlobalSemaphore ready_minus;
    GlobalSemaphore arrival_minus;
    GlobalSemaphore arrival_plus;
};

// Load balancing (kernels/dataflow/sparse_ring_balance.cpp, kernels/sparse_ring_balance_plan.hpp), decided per build
// from the shapes (see make_balance_config); cross-device balancing (remote) on a ring of 2+ devices with a 1D fabric.
struct BalanceConfig {
    bool on = false;
    bool remote = false;
    CoreCoord core;           // the balance core (logical), next to the gather workers
    uint32_t units = 0;       // NH x q_blocks, head-major
    uint32_t cores = 0;       // SDPA cores
    uint32_t q_blocks = 0;    // Q blocks per head
    uint32_t cost_words = 0;  // cost words per core in a table page
    uint32_t swappable = 0;   // full Q blocks (a partial last block never moves)
    uint32_t devices = 1;     // cost pages (devices) the balance kernel reads
    uint32_t scratch_bytes = 0;
};

// The balance core: the first core the gather's worker selection takes after its workers (the CCL column's next core
// on the Galaxy), when it is on the worker grid.
std::optional<CoreCoord> choose_balance_core(
    const SparseRingSDPAParams& args, const SparseRingSDPAInputs& tensor_args) {
    const auto& ag = args.all_gather_operation_attributes;
    auto* mesh = tensor_args.input_q.device();
    const auto gather = ttnn::ccl::try_choose_worker_cores(
        ag.num_links, 2, mesh, ag.sub_device_id, args.ccl_core_grid_offset, std::nullopt, ag.core_allocation_strategy);
    const auto more = ttnn::ccl::try_choose_worker_cores(
        1,
        2 * ag.num_links + 1,
        mesh,
        ag.sub_device_id,
        args.ccl_core_grid_offset,
        std::nullopt,
        ag.core_allocation_strategy);
    if (!gather.all_placeable() || !more.all_placeable() || more.cores.size() != gather.cores.size() + 1) {
        return std::nullopt;
    }
    for (const auto& c : more.cores) {
        if (std::find(gather.cores.begin(), gather.cores.end(), c) == gather.cores.end()) {
            return c;
        }
    }
    return std::nullopt;
}

// Every device must map DRAM banks to the same NoC coordinates: the cross-device swaps write into other devices'
// DRAM with this device's bank map (as the gather does).
bool same_dram_banks_on_every_device(tt::tt_metal::distributed::MeshDevice* mesh) {
    const auto banks_of = [](const tt::tt_metal::IDevice* device) {
        std::vector<std::array<uint64_t, 3>> banks;
        for (const auto& bank : tt::tt_metal::internal::get_dram_bank_table(device->id())) {
            banks.push_back({bank.noc_x, bank.noc_y, bank.base_addr});
        }
        return banks;
    };
    const auto devices = mesh->get_devices();
    const auto first = banks_of(devices.front());
    return std::all_of(devices.begin(), devices.end(), [&](const auto* device) { return banks_of(device) == first; });
}

BalanceConfig make_balance_config(const SparseRingSDPAParams& args, const SparseRingSDPAInputs& tensor_args) {
    BalanceConfig c;
    // Kept-only route without dense_blocks only: dense_blocks already spread the dense rows over the devices, and the
    // pooled route measured slower with balancing on balanced masks.
    if (!args.dense_blocks.empty() || tensor_args.pooled_route()) {
        return c;
    }
    const uint32_t q_tiles = tensor_args.input_q.padded_shape()[2] / tt::constants::TILE_HEIGHT;
    const uint32_t mask_row_bytes = tensor_args.block_mask.logical_shape()[3] * sizeof(uint32_t);
    const CoreCoord grid = args.program_config.compute_with_storage_grid_size;
    c.q_blocks = tt::div_up(q_tiles, args.block_tiles());
    c.cores = grid.x * grid.y;
    c.units = tensor_args.input_q.logical_shape()[1] * c.q_blocks;
    const uint32_t max_q = tt::div_up(c.units, c.cores);
    c.cost_words = sparse_balance::cost_words_per_core(max_q);
    c.swappable = q_tiles / args.block_tiles();
    const auto core = choose_balance_core(args, tensor_args);
    // Unit tables hold 15-bit unit indices and kMaxCoreUnits units; the reader stages a core's costs in its second plan
    // page; the balance kernel's local scratch fits its budget, a head's bins a byte.
    const sparse_balance::StaticSplit split{c.units, c.cores, c.q_blocks};
    if (!core.has_value() || c.units >= (1u << 15) || c.q_blocks > sparse_balance::kMaxUnitsPerHead ||
        max_q > sparse_balance::kMaxCoreUnits || c.cost_words * sizeof(uint32_t) + 63 > 64 + mask_row_bytes ||
        split.max_bins() > sparse_balance::kMaxBins ||
        sparse_balance::local_scratch_bytes(c.q_blocks, split.max_bins()) > sparse_balance::kMaxLocalScratchBytes) {
        return c;
    }
    constexpr uint32_t kMaxScratchBytes = 512 * 1024;  // the balance core's L1 holds the cost pages and the plan
    const auto scratch = [&](uint32_t devices) {
        return sparse_balance::Scratch(devices, c.cores, c.cost_words, c.q_blocks, c.swappable, c.units).words *
                   sizeof(uint32_t) +
               64;
    };
    c.core = *core;
    c.on = scratch(1) <= kMaxScratchBytes;
    // Remote devices are addressed through this device's NoC coordinates for the balance core (fabric increments):
    // every device must map it the same way. Ranks are tensor ranks, which are the transport ranks off the full mesh.
    auto* mesh = tensor_args.input_q.device();
    const CoreCoord noc = mesh->worker_core_from_logical_core(c.core);
    bool same_noc = true;
    for (auto* device : mesh->get_devices()) {
        same_noc = same_noc && device->worker_core_from_logical_core(c.core) == noc;
    }
    c.remote = c.on && same_noc && args.ring_size > 1 && !args.all_gather_operation_attributes.full_mesh &&
               tt::tt_fabric::is_1d_fabric_config(tt::tt_fabric::GetFabricConfig()) &&
               scratch(args.ring_size) <= kMaxScratchBytes && same_dram_banks_on_every_device(mesh);
    c.devices = c.remote ? static_cast<uint32_t>(args.ring_size) : 1u;
    c.scratch_bytes = scratch(c.devices);
    return c;
}

// The balance kernel's buffers and fabric counters, made once per cached workload: the cost table (R pages), and
// with cross-device balancing the partner page, the import buffers (Q and mask rows of the swapped-in row blocks, laid
// out as Q and the mask by local slot), the export buffer (their outputs, laid out as the output) and the counters of
// arrived tables, imports and returns on the balance core.
struct BalanceResources {
    std::shared_ptr<Tensor> table;
    std::shared_ptr<Tensor> partner;
    std::shared_ptr<Tensor> import_q;
    std::shared_ptr<Tensor> import_mask;
    std::shared_ptr<Tensor> export_out;
    std::optional<GlobalSemaphore> tables_arrived;
    std::optional<GlobalSemaphore> imports_arrived;
    std::optional<GlobalSemaphore> returns_arrived;
};

// Dense units of the op: one plan per owner shard (a shard with a Q block in a range), in shard order. Units are
// head-major (sparse_dense_units.hpp): n_units = NH x the owners' dense blocks.
struct DenseOwnerPlan {
    uint32_t shard;
    uint32_t n_blocks;
    uint32_t block_offset;
    uint32_t slab_out_row;
    uint32_t slab_row_start;
    uint32_t slab_rows;
};

std::vector<DenseOwnerPlan> dense_owner_plans(
    const SparseRingSDPAParams& args, const SparseRingSDPAInputs& tensor_args, uint32_t& n_units) {
    std::vector<DenseOwnerPlan> plans;
    n_units = 0;
    if (!tensor_args.gathered_dense_q.has_value()) {
        return plans;
    }
    const uint32_t ring_size = static_cast<uint32_t>(args.ring_size);
    const uint32_t nb_pad = tensor_args.block_mask.logical_shape()[3] / ring_size;
    const uint32_t local_tiles = tensor_args.input_q.padded_shape()[2] / tt::constants::TILE_HEIGHT;
    const uint32_t nb = tt::div_up(local_tiles, args.block_tiles());
    const uint32_t NH = tensor_args.input_q.logical_shape()[1];
    const auto slabs = dense_q_slabs(args.dense_blocks, ring_size, nb_pad, local_tiles, args.block_tiles());
    uint32_t out_row = 0;
    uint32_t n_dense_blocks = 0;
    for (uint32_t s = 0; s < ring_size; ++s) {
        uint32_t n = 0;
        for (uint32_t b = 0; b < nb; ++b) {
            const uint32_t g = s * nb_pad + b;
            n += std::any_of(
                     args.dense_blocks.begin(),
                     args.dense_blocks.end(),
                     [g](const auto& r) { return g >= r.first && g < r.second; })
                     ? 1u
                     : 0u;
        }
        if (n > 0) {
            plans.push_back({s, n, n_dense_blocks, out_row, slabs[s].row_start, slabs[s].rows});
            n_dense_blocks += n;
        }
        out_row += slabs[s].rows;
    }
    n_units = NH * n_dense_blocks;
    return plans;
}

// Per-coord ProgramDescriptor build, looped over coords by create_workload_descriptor().
//
// The kernels keep the dense ring op's argument layout: positional compile-time and runtime args for features this
// op never uses (joint tokens, KV cache, causal, sliding window, latent V) are still emitted, as constants, so every
// index the kernels read stays where it was.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
tt::tt_metal::ProgramDescriptor build_sparse_ring_sdpa_program_descriptor(
    const SparseRingSDPAParams& args,
    const SparseRingSDPAInputs& tensor_args,
    SparseRingSDPAResult& output_tensors,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate,
    const DenseResources* dense_res,
    const BalanceResources* balance_res) {
    TT_FATAL(
        mesh_dispatch_coordinate.has_value(),
        "build_sparse_ring_sdpa_program_descriptor requires mesh_dispatch_coordinate");
    const auto& coord = mesh_dispatch_coordinate.value();
    /*
        input_tensor_q / k / v: 1 x NH x local_N x DH (sequence-sharded across the ring)
        gathered_k / gathered_v: 1 x NH x padded_N x DH (fused all-gather)
        block_mask: 1 x NH x nqb_pad x n_blocks_pad (uint32); pooled_k / pooled_v (optional): 1 x NH x n_blocks_pad x DH
        output_tensor: 1 x NH x local_N x DH

    Every core keeps the Q blocks of its current pass and their softmax state in L1; each pass replays the ring over
    the K/V gathered once. Each K/V piece a head group keeps is delivered once to the group.
    */

    log_debug(tt::LogOp, "SparseRingSDPA create_descriptor");

    const auto& input_tensor_q = tensor_args.input_q;
    const auto& input_tensor_k = tensor_args.input_k;
    const auto& input_tensor_v = tensor_args.input_v;
    const auto& gathered_input_tensor_k = tensor_args.gathered_k;
    const auto& gathered_input_tensor_v = tensor_args.gathered_v;

    auto& output_tensor = output_tensors[SPARSE_RING_SDPA_OUTPUT_IDX];

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

    const float scale_value =
        args.scale.value_or(1.0f / std::sqrt(static_cast<float>(input_tensor_q.logical_shape()[-1])));

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
    log_debug(tt::LogOp, "v_shape (gathered): {}", gathered_input_tensor_v.logical_shape());

    const uint32_t B = q_shape[0];
    const uint32_t NH = q_shape[1];
    const uint32_t NHK = k_shape[1];
    const uint32_t NHV = input_tensor_v.logical_shape()[1];
    const uint32_t DH = q_shape[3];
    const uint32_t vDH = input_tensor_v.logical_shape()[3];
    const uint32_t q_local_padded_N = q_shape[2];
    const uint32_t kv_local_padded_N = tensor_args.local_kv_seq_len();
    const uint32_t ring_size = static_cast<uint32_t>(args.all_gather_operation_attributes.ring_size);
    const uint32_t gathered_padded_N = k_shape[2];

    const bool gqa_grouped_kv = ring_joint::is_gqa_grouped_kv_head_mode(/*v_shares_k_buffer=*/false, NH, NHK, NHV);
    const bool k_uses_batch_chain = ring_joint::uses_shared_k_batch_chain(gqa_grouped_kv, NHK);
    const bool use_head_chain =
        ring_joint::uses_v_head_chain(/*enable_kv_chains=*/true, gqa_grouped_kv, /*v_shares_k_buffer=*/false);

    const uint32_t q_local_padded_Nt = q_local_padded_N / tt::constants::TILE_HEIGHT;
    const uint32_t kv_local_padded_Nt = kv_local_padded_N / tt::constants::TILE_HEIGHT;
    const uint32_t gathered_padded_Nt = gathered_padded_N / tt::constants::TILE_HEIGHT;
    const uint32_t DHt = DH / tt::constants::TILE_WIDTH;
    const uint32_t vDHt = vDH / tt::constants::TILE_WIDTH;
    const RingWorkMasks ring_work_masks = build_ring_work_masks(args, tensor_args, ring_write_plan);
    const RingJointRuntimeArgLayout runtime_arg_layout =
        get_runtime_arg_layout(use_head_chain, k_uses_batch_chain, gqa_grouped_kv);
    const uint32_t logical_nt = tt::div_up(static_cast<uint32_t>(args.logical_n), tt::constants::TILE_HEIGHT);
    // Chunked-prefill Q mapping words of the shared runtime-arg layout; unused here.
    constexpr uint32_t kUnusedQMappingWord = 0;

    const uint32_t Sq_chunk_t = q_chunk_size / tt::constants::TILE_HEIGHT;
    const uint32_t Sk_chunk_t = k_chunk_size / tt::constants::TILE_HEIGHT;

    // Block geometry. Blocks are block_size tokens (block_tiles tiles) inside each shard; the mask row (and the
    // pooled tensors, when given) hold ring_size x mask_nb_pad blocks; a pooled "chunk" is Sk_chunk_t pooled rows.
    // Without pooled K / V (kept blocks only) the pooled route is compiled out and there are no pooled chunks.
    const bool pooled_route = tensor_args.pooled_route();
    const uint32_t block_tiles = args.block_tiles();
    const uint32_t block_tokens = block_tiles * tt::constants::TILE_HEIGHT;
    const uint32_t mask_nkc = tt::div_up(kv_local_padded_Nt, Sk_chunk_t);
    const uint32_t mask_n_blocks_pad = tensor_args.block_mask.logical_shape()[3];
    const uint32_t mask_nb_pad = mask_n_blocks_pad / ring_size;
    const uint32_t mask_nb = tt::div_up(kv_local_padded_Nt, block_tiles);
    const uint32_t mask_nqb_pad = tensor_args.block_mask.logical_shape()[2];
    const uint32_t pooled_tiles = pooled_route ? mask_n_blocks_pad / tt::constants::TILE_HEIGHT : 0u;
    const uint32_t pooled_chunks = tt::div_up(pooled_tiles, Sk_chunk_t);
    const uint32_t pooled_last_tiles = pooled_chunks > 0 ? pooled_tiles - (pooled_chunks - 1) * Sk_chunk_t : 0u;
    // The reader reads whole mask rows.
    const uint32_t mask_row_bytes = tensor_args.block_mask.logical_shape()[3] * sizeof(uint32_t);
    // Validated on the op.
    TT_ASSERT(mask_nkc <= 64);
    TT_ASSERT(mask_nb <= mask_nb_pad);
    TT_ASSERT(Sk_chunk_t % block_tiles == 0);
    TT_ASSERT(Sq_chunk_t == block_tiles);

    // Global chunk width the kernels divide by the ring size (one Q slab per device).
    const uint32_t q_chunk_group_tile_count = q_local_padded_Nt * ring_size;

    const uint32_t compile_time_logical_n = static_cast<uint32_t>(args.logical_n);
    const uint32_t compile_time_global_n_partial_col = compile_time_logical_n % tt::constants::TILE_HEIGHT;

    const uint32_t num_local_q_chunks = tt::div_up(q_local_padded_N, q_chunk_size);
    const uint32_t num_local_k_chunks = tt::div_up(kv_local_padded_N, k_chunk_size);

    // dense_blocks split across devices: every device computes every dense unit (owner shard, head, dense Q block)
    // against its own K/V shard, and this shard's dense Q blocks leave its resident split (res_remap).
    const bool dense_h = tensor_args.gathered_dense_q.has_value();
    const bool dense_debug = tensor_args.dense_partials_debug.has_value();
    // One partial (sparse_dense::partial_tiles): out, per-column sums, row max.
    const uint32_t partial_tiles = Sq_chunk_t * vDHt + 2 * Sq_chunk_t;
    uint32_t dense_units = 0;
    const auto owner_plans = dense_owner_plans(args, tensor_args, dense_units);
    // The dense-Q buffer's tile rows per head (its page layout; the slabs fill a prefix).
    const uint32_t dense_q_rows_t =
        dense_h ? tensor_args.gathered_dense_q->padded_shape()[2] / tt::constants::TILE_HEIGHT : 0;
    uint32_t own_dense_blocks = 0;
    for (const auto& o : owner_plans) {
        if (o.shard == transport_rank) {
            own_dense_blocks = o.n_blocks;
        }
    }
    if (dense_h) {
        TT_FATAL(
            own_dense_blocks < num_local_q_chunks,
            "SparseRingSDPA: every shard must keep a Q block outside dense_blocks (shard {} has {} of {} dense)",
            transport_rank,
            own_dense_blocks,
            num_local_q_chunks);
        TT_FATAL(
            logical_nt > (ring_size - 1) * kv_local_padded_Nt,
            "SparseRingSDPA: dense_blocks need a real token in every shard (logical_n {})",
            args.logical_n);
    }
    const uint32_t num_res_q_chunks = num_local_q_chunks - own_dense_blocks;
    const bool res_remap = own_dense_blocks > 0;
    // Load balancing (see BalanceConfig); the workload made its resources iff it is on.
    const BalanceConfig balancing = make_balance_config(args, tensor_args);
    TT_FATAL(
        balancing.on == (balance_res != nullptr), "SparseRingSDPA: load-balancing resources do not match the build");
    const bool balance = balancing.on;
    log_debug(
        tt::LogOp,
        "SparseRingSDPA load balancing: within the device {}, across devices {}, balance core {}",
        balancing.on,
        balancing.remote,
        balancing.core);
    // The balance kernel's scratch (sparse_balance::Scratch) first: the unit-table CB, laid out with the SDPA CBs, also
    // spans the balance core and so lands above both.
    constexpr uint32_t kBalanceScratchCb = 0;
    if (balance) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = balancing.scratch_bytes,
            .core_ranges = CoreRangeSet(CoreRange(balancing.core)),
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(kBalanceScratchCb),
                .data_format = tt::DataFormat::RawUInt32,
                .page_size = balancing.scratch_bytes,
            }}},
        });
    }
    // Each device's role for each owner's units: normalize (the owner) or emit toward it, one hop along the shorter
    // way round (ties: +); and which neighbours' partials arrive here.
    const bool ring_topology = args.all_gather_operation_attributes.topology == ttnn::ccl::Topology::Ring;
    const auto route = [&](uint32_t n, uint32_t o) -> int {
        if (ring_topology) {
            return (o + ring_size - n) % ring_size <= (n + ring_size - o) % ring_size ? 1 : -1;
        }
        return o > n ? 1 : -1;
    };
    std::vector<uint32_t> dense_records;
    for (const auto& o : owner_plans) {
        const uint32_t d = transport_rank;
        uint32_t up_sides = 0;
        const bool has_minus = ring_topology || d > 0;
        const bool has_plus = ring_topology || d + 1 < ring_size;
        const uint32_t m = (d + ring_size - 1) % ring_size;
        const uint32_t p = (d + 1) % ring_size;
        if (has_minus && m != o.shard && route(m, o.shard) == 1) {
            up_sides |= 1u;
        }
        if (has_plus && p != o.shard && p != m && route(p, o.shard) == -1) {
            up_sides |= 2u;
        }
        const uint32_t role = d == o.shard             ? sparse_dense::kRoleOwner
                              : route(d, o.shard) == 1 ? sparse_dense::kRoleEmitPlus
                                                       : sparse_dense::kRoleEmitMinus;
        dense_records.insert(
            dense_records.end(),
            {o.shard, o.n_blocks, o.block_offset, o.slab_out_row, o.slab_row_start, o.slab_rows, up_sides, role});
    }

    log_debug(tt::LogOp, "B: {}, NH: {}, NHK: {}, NHV: {}, DH: {}, vDH: {}", B, NH, NHK, NHV, DH, vDH);
    log_debug(tt::LogOp, "q_local_padded_N: {}, kv_local_padded_N: {}", q_local_padded_N, kv_local_padded_N);
    log_debug(
        tt::LogOp,
        "Sq_chunk_t: {}, Sk_chunk_t: {}, num_local_q_chunks: {}, num_local_k_chunks: {}",
        Sq_chunk_t,
        Sk_chunk_t,
        num_local_q_chunks,
        num_local_k_chunks);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(mesh_device->arch(), args.compute_kernel_config);

    const CoreCoord grid_size = args.program_config.compute_with_storage_grid_size;
    const bool exp_approx_mode = args.program_config.exp_approx_mode.value_or(true);

    auto core_grid = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    CoreRangeSet core_grid_set(core_grid);
    uint32_t num_cores = grid_size.x * grid_size.y;

    // Init fused op signaler — descriptor-pattern equivalent of RingSDPAFusedOpSignaler::init_fused_op. The signaler
    // stores the receiver-cores noc list and two signal semaphore IDs (one for forward, one for backward).
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

    // Single host-derived split-forwarding decision, shared with the all-gather (passed to the helper below), so
    // producer and consumer cannot disagree.
    sdpa_fused_op_signaler->split_forwarding_enabled =
        (args.all_gather_operation_attributes.topology == ttnn::ccl::Topology::Ring) &&
        (args.all_gather_operation_attributes.ring_size % 2 == 0) &&
        (args.all_gather_operation_attributes.ring_size > 2);

    log_debug(tt::LogOp, "num_cores: {}, grid_size: {}", num_cores, grid_size);

    TT_FATAL(
        num_cores <= mesh_device->compute_with_storage_grid_size().x * mesh_device->compute_with_storage_grid_size().y,
        "Provided grid must not contain more cores than the device. Got {} cores, expected at most {} cores.",
        num_cores,
        mesh_device->compute_with_storage_grid_size().x * mesh_device->compute_with_storage_grid_size().y);

    // The (batch, head, Q block) units are divided evenly across the cores, in flat order.
    const uint32_t all_heads_num_q_chunks = B * NH * num_res_q_chunks;
    const uint32_t max_q_per_core = tt::div_up(all_heads_num_q_chunks, num_cores);

    log_debug(tt::LogOp, "max_q_per_core: {}", max_q_per_core);

    // The plan page's 16-word header carries the shard's kept-block bits (one word per 32 blocks) plus the union
    // piece bits (2 words) and the pooled flag.
    TT_FATAL(
        tt::div_up(mask_nb, 32u) + 3 <= 16,
        "SparseRingSDPA: {} blocks per shard exceed the plan header (at most 416)",
        mask_nb);
    // Group delivery: one producer per head group multicasts each piece. The K/V CBs hold 3 pieces with it (the
    // producer sends the next piece while members compute), 2 without.
    const bool group_mcast = use_head_chain;
    const uint32_t k_piece_tiles = Sk_chunk_t * DHt;
    const uint32_t v_piece_tiles = Sk_chunk_t * vDHt;
    // Pair steps (one step over two consecutive pieces of a head, halving the per-step fixed cost) apply to small
    // pieces (<= 256 keys). They need a 4-deep K/V CB, so a pair and the next piece's delivery fit together, and twice
    // the score scratch; the pass split below turns them on only when that costs no extra pass. Off with dense units:
    // their code would not fit the kernel config buffer next to the dense units'.
    const bool pair_steps_eligible = Sk_chunk_t <= 8 && !dense_h;
    bool pair_steps = false;                       // decided with the pass split
    uint32_t kv_cb_pages = group_mcast ? 3u : 2u;  // decided with the pass split
    uint32_t qk_tiles = Sq_chunk_t * Sk_chunk_t;
    uint32_t out_im_tiles = Sq_chunk_t * vDHt;
    uint32_t out0_t = Sq_chunk_t * vDHt;
    uint32_t scale_tiles = 1;
    uint32_t statistics_tiles = Sq_chunk_t;  // Single column of values in each iteration

    log_debug(
        tt::LogOp,
        "k_piece_tiles: {}, v_piece_tiles: {}, qk_tiles: {}, out0_t: {}, statistics_tiles: {}",
        k_piece_tiles,
        v_piece_tiles,
        qk_tiles,
        out0_t,
        statistics_tiles);

    // Host code is responsible for determining matmul configuration
    const uint32_t dst_size = ttnn::get_dest_reg_count(args.compute_kernel_config);
    const uint32_t qk_in0_block_w = DHt;
    auto [qk_out_subblock_h, qk_out_subblock_w] =
        detail::determine_largest_subblock_size(Sq_chunk_t, Sk_chunk_t, dst_size);

    TT_FATAL(
        Sq_chunk_t % qk_out_subblock_h == 0,
        "Sq_chunk_t ({}) must be divisible by qk_out_subblock_h ({})",
        Sq_chunk_t,
        qk_out_subblock_h);
    const uint32_t qk_in0_num_subblocks = Sq_chunk_t / qk_out_subblock_h;
    const uint32_t qk_in1_num_subblocks = Sk_chunk_t / qk_out_subblock_w;

    // now for out0
    const uint32_t out_in0_block_w = Sk_chunk_t;

    // Validated: fp32_dest_acc_en is off, so compute takes the streaming path.
    const bool use_streaming_compute = !fp32_dest_acc_en;
    TT_ASSERT(use_streaming_compute);

    auto [out_out_subblock_h, out_out_subblock_w] = detail::determine_largest_subblock_size(
        Sq_chunk_t,
        vDHt,
        dst_size,
        /*max_subblock_h=*/2,
        /*max_subblock_w=*/UINT32_MAX);
    // Streaming compute may widen the QKT@V row group beyond the host matmul subblock height for odd Q chunks. The
    // writer must drain cb_out with the same row-group cadence that compute pushes.
    const uint32_t writer_out_row_group_h =
        ttnn::transformer::sdpa::streaming_qktv_h(out_out_subblock_h, out_out_subblock_w, dst_size, Sq_chunk_t);

    const uint32_t out_in0_num_subblocks = Sq_chunk_t / out_out_subblock_h;
    const uint32_t out_in1_num_subblocks = vDHt / out_out_subblock_w;

    // Shrink cb_out to a 2-slot ping-pong (see sdpa_subblock_utils.hpp) when the save-to-staging branch cannot fire
    // (ring_size == 1 or one Q block per core) and the full output fits in 2 row groups.
    const bool streaming_shrink_fits = Sq_chunk_t <= 2 * writer_out_row_group_h;
    // With load balancing a core may hold more units than the static split gives it.
    const bool streaming_shrink_safe = (ring_size == 1 || max_q_per_core == 1) && streaming_shrink_fits && !balance;
    if (streaming_shrink_safe) {
        out0_t = detail::streaming_cb_out_tiles(out_out_subblock_h, out_out_subblock_w, dst_size, Sq_chunk_t, vDHt);
        TT_FATAL(
            Sq_chunk_t % out_out_subblock_h == 0,
            "Streaming cb_out drain requires Sq_chunk_t ({}) divisible by out_out_subblock_h ({})",
            Sq_chunk_t,
            out_out_subblock_h);
    }
    log_debug(tt::LogOp, "streaming_shrink_safe={} out0_t={}", streaming_shrink_safe, out0_t);
    log_debug(
        tt::LogOp,
        "qk subblock {}x{} ({}x{} subblocks), out subblock {}x{} ({}x{} subblocks)",
        qk_out_subblock_h,
        qk_out_subblock_w,
        qk_in0_num_subblocks,
        qk_in1_num_subblocks,
        out_out_subblock_h,
        out_out_subblock_w,
        out_in0_num_subblocks,
        out_in1_num_subblocks);

    // Determine granularity for statistics computation
    // Each granularity must evenly divide its tile count to avoid dropping tiles
    const uint32_t stats_granularity = detail::find_valid_granularity(Sq_chunk_t, dst_size);
    const uint32_t sub_exp_granularity = detail::find_valid_granularity(Sk_chunk_t, dst_size);
    // DHT_GRANULARITY is used in the kernel with both DHt and vDHt as the cols parameter,
    // so the granularity must evenly divide both to avoid dropping tiles.
    uint32_t dht_granularity = std::min({DHt, vDHt, dst_size});
    while (dht_granularity > 1 && (DHt % dht_granularity != 0 || vDHt % dht_granularity != 0)) {
        dht_granularity--;
    }
    const uint32_t reduce_granularity = detail::find_valid_granularity(Sq_chunk_t, dst_size / 2);

    log_debug(
        tt::LogOp,
        "granularity: stats {}, sub_exp {}, dht {}, reduce {}",
        stats_granularity,
        sub_exp_granularity,
        dht_granularity,
        reduce_granularity);

    // Reduce ops need to multiply by a scalar. We always want to multiply by 1.0f
    class bfloat16 bfloat_identity_scalar(1.0f);
    uint32_t packed_identity_scalar = pack_two_bfloat16_into_uint32({bfloat_identity_scalar, bfloat_identity_scalar});

    const uint32_t scale_packed = std::bit_cast<uint32_t>(scale_value);

    log_debug(tt::LogOp, "scale: {}", scale_value);

    const uint32_t active_ring_iter_mask = ring_work_masks.active_ring_iter_mask;
    const uint32_t single_valid_kv_chunk_mask = ring_work_masks.single_valid_kv_chunk_mask;
    const auto& ag_attributes = args.all_gather_operation_attributes;
    const RingAttentionRankMapping rank_mapping{
        .full_mesh = ag_attributes.full_mesh,
        .orientation = ag_attributes.snake_orientation,
        .mesh_rows = ag_attributes.mesh_rows,
        .mesh_cols = ag_attributes.mesh_cols};

    const uint32_t q_heads_per_kv = NH / NHK;

    // Cores actually issuing Q reads. When the flat Q block distribution is smaller than the grid the trailing cores
    // get zero work.
    const uint32_t num_active_cores = std::min(num_cores, all_heads_num_q_chunks);

    // Positional reader args. Slots for features this op does not use are constants (see the function comment).
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
        0,  // Lt (no joint tokens)
        0,  // L
        num_local_q_chunks,
        0,  // num_joint_q_chunks
        num_local_k_chunks,
        0,  // num_joint_k_chunks
        num_res_q_chunks,
        ring_size,
        qk_out_subblock_h,
        0,  // is_causal
        0,  // is_balanced
        0,  // zigzag balancing
        0,  // chunked prefill
        num_active_cores,
        q_chunk_group_tile_count,
        0,  // indexed KV cache
        0,  // KV-pad rotation
        NHV,
        0,  // v_shares_k_buffer
        0,  // attention sink
        0,  // sliding_window_size
        gathered_padded_Nt,
        0,  // slot from metadata
        0,  // KV pad from metadata
        0,  // Lt_local
        0,  // joint is sharded
        0,  // logical_lt
        // Transport-to-tensor rank mapping.
        static_cast<uint32_t>(rank_mapping.full_mesh),
        static_cast<uint32_t>(rank_mapping.orientation),
        rank_mapping.mesh_rows,
        rank_mapping.mesh_cols,
        0,  // circular KV slab count
        0,  // logical_n tensor
        0,  // logical_l tensor
    };

    // Accessors: Q, K, V, gathered K, gathered V, attention sink (null), then block mask, pooled K, pooled V (null
    // without the pooled route, so the accessor offsets do not move).
    tt::tt_metal::Buffer* const attention_sink_buf = nullptr;
    tt::tt_metal::Buffer* const pooled_k_buf = pooled_route ? tensor_args.pooled_k->buffer() : nullptr;
    tt::tt_metal::Buffer* const pooled_v_buf = pooled_route ? tensor_args.pooled_v->buffer() : nullptr;
    tt::tt_metal::Buffer* const block_counts_buf =
        tensor_args.block_counts.has_value() ? tensor_args.block_counts->buffer() : nullptr;
    TensorAccessorArgs(input_tensor_q.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(input_tensor_k.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(input_tensor_v.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(gathered_input_tensor_k.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(gathered_input_tensor_v.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(attention_sink_buf).append_to(reader_compile_time_args);
    TensorAccessorArgs(tensor_args.block_mask.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(pooled_k_buf).append_to(reader_compile_time_args);
    TensorAccessorArgs(pooled_v_buf).append_to(reader_compile_time_args);
    TensorAccessorArgs(block_counts_buf).append_to(reader_compile_time_args);
    tt::tt_metal::Buffer* const dense_q_buf = dense_h ? tensor_args.gathered_dense_q->buffer() : nullptr;
    tt::tt_metal::Buffer* const dense_recv_buf = dense_res != nullptr ? dense_res->recv->buffer() : nullptr;
    tt::tt_metal::Buffer* const dense_partials_buf =
        dense_debug ? tensor_args.dense_partials_debug->buffer()
                    : (dense_res != nullptr ? dense_res->staging->buffer() : nullptr);
    TensorAccessorArgs(dense_q_buf).append_to(reader_compile_time_args);
    TensorAccessorArgs(dense_recv_buf).append_to(reader_compile_time_args);
    tt::tt_metal::Buffer* const balance_table_buf = balance ? balance_res->table->buffer() : nullptr;
    TensorAccessorArgs(balance_table_buf).append_to(reader_compile_time_args);

    /**
     * Create semaphores used for L1-L1 store-and-forward of KV between cores.
     * ChainSemaphores groups the three semaphore IDs for a single chain (sender,
     * receiver, valid) and pushes them as SemaphoreDescriptor entries on the
     * descriptor. The IDs are sequential indices into desc.semaphores.
     */
    struct ChainSemaphores {
        uint32_t sender_id;
        uint32_t receiver_id;
        uint32_t valid_id;

        static ChainSemaphores create(ProgramDescriptor& desc, const CoreRangeSet& cores) {
            ChainSemaphores out;
            out.sender_id = static_cast<uint32_t>(desc.semaphores.size());
            desc.semaphores.push_back(SemaphoreDescriptor{
                .id = out.sender_id,
                .core_type = tt::CoreType::WORKER,
                .core_ranges = cores,
                .initial_value = INVALID,
            });
            out.receiver_id = static_cast<uint32_t>(desc.semaphores.size());
            desc.semaphores.push_back(SemaphoreDescriptor{
                .id = out.receiver_id,
                .core_type = tt::CoreType::WORKER,
                .core_ranges = cores,
                .initial_value = INVALID,
            });
            out.valid_id = static_cast<uint32_t>(desc.semaphores.size());
            desc.semaphores.push_back(SemaphoreDescriptor{
                .id = out.valid_id,
                .core_type = tt::CoreType::WORKER,
                .core_ranges = cores,
                .initial_value = VALID,
            });
            return out;
        }

        void append_to_compile_args(std::vector<uint32_t>& args) const {
            const size_t start_size = args.size();
            args.push_back(sender_id);
            args.push_back(receiver_id);
            args.push_back(valid_id);
            TT_FATAL(
                args.size() == start_size + kRingJointChainSemaphoreCompileArgCount,
                "ChainSemaphores expected to append {} compile-time args, appended {}",
                kRingJointChainSemaphoreCompileArgCount,
                args.size() - start_size);
        }
    };

    std::optional<ChainSemaphores> head_sems;
    std::optional<ChainSemaphores> batch_sems;
    std::optional<ChainSemaphores> gqa_sems;
    if (use_head_chain) {
        head_sems = ChainSemaphores::create(desc, core_grid_set);  // head chain (MHA or separate-V V)
    }
    if (k_uses_batch_chain) {
        batch_sems = ChainSemaphores::create(desc, core_grid_set);  // shared-K chain
    }
    if (gqa_grouped_kv) {
        gqa_sems = ChainSemaphores::create(desc, core_grid_set);  // grouped K/V chain
    }
    // The dense groups' delivery (credits, delivered count, source word), apart from the resident groups'.
    std::optional<ChainSemaphores> dense_sems;
    if (dense_h && group_mcast) {
        dense_sems = ChainSemaphores::create(desc, core_grid_set);
    }
    // Load balancing: per SDPA core the release of its unit table (written after the table) and the imports-arrived
    // signal; on the balance core the count of cores that wrote their costs and of swapped-in units whose outputs are
    // written.
    struct BalanceSemaphores {
        uint32_t table = 0, import = 0, costs = 0, outputs = 0;
    } balance_sems;
    if (balance) {
        const auto add_semaphore = [&](const CoreRangeSet& cores) {
            const uint32_t id = static_cast<uint32_t>(desc.semaphores.size());
            desc.semaphores.push_back(SemaphoreDescriptor{
                .id = id, .core_type = tt::CoreType::WORKER, .core_ranges = cores, .initial_value = 0});
            return id;
        };
        const CoreRangeSet balance_core_set(CoreRange(balancing.core));
        balance_sems.table = add_semaphore(core_grid_set);
        balance_sems.import = add_semaphore(core_grid_set);
        balance_sems.costs = add_semaphore(balance_core_set);
        balance_sems.outputs = add_semaphore(balance_core_set);
        // The gather adds up to 3 sync semaphores and the writers' flag after these: ids stay below 16.
        TT_FATAL(
            desc.semaphores.size() + 4 <= 16,
            "SparseRingSDPA load balancing: {} semaphores leave no room for the gather's",
            desc.semaphores.size());
    }
    const CoreCoord balance_core_noc =
        balance ? mesh_device->worker_core_from_logical_core(balancing.core) : CoreCoord{0, 0};

    // Append semaphore ids to reader compile-time args (must match reader kernel expectations)
    const auto sem_args_offset = reader_compile_time_args.size();
    if (use_head_chain) {
        head_sems->append_to_compile_args(reader_compile_time_args);
        reader_compile_time_args.push_back(0);  // head_mcast_enabled: never (group delivery replaces it)
    }
    if (k_uses_batch_chain) {
        batch_sems->append_to_compile_args(reader_compile_time_args);
        reader_compile_time_args.push_back(0);  // shared-K mcast placeholder (patched after chain construction)
    }
    if (gqa_grouped_kv) {
        gqa_sems->append_to_compile_args(reader_compile_time_args);
        reader_compile_time_args.push_back(0);  // GQA mcast placeholder (patched after chain construction)
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
        0,  // Lt_local
        0,  // L
        num_local_q_chunks,
        0,  // num_joint_q_chunks
        num_local_k_chunks,
        0,  // num_joint_k_chunks
        num_res_q_chunks,
        packed_identity_scalar,
        scale_packed,
        ring_size,
        compile_time_global_n_partial_col,
        0,  // joint_l_partial_col
        static_cast<std::uint32_t>(use_streaming_compute),
        0,  // is_causal
        0,  // is_balanced
        0,  // zigzag balancing
        static_cast<std::uint32_t>(writer_out_row_group_h),
        0,  // chunked prefill
        q_chunk_group_tile_count,
        0,  // sliding_window_size
        0,  // KV pad from metadata
        0,  // joint is sharded
        0,  // logical_lt
        // Transport-to-tensor rank mapping.
        static_cast<uint32_t>(rank_mapping.full_mesh),
        static_cast<uint32_t>(rank_mapping.orientation),
        rank_mapping.mesh_rows,
        rank_mapping.mesh_cols,
        0,  // logical_n tensor
        0,  // logical_l tensor
    };

    TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_time_args);
    TensorAccessorArgs(dense_partials_buf).append_to(writer_compile_time_args);

    std::vector<uint32_t> compute_compile_time_args = {
        NH,
        DHt,
        vDHt,
        Sq_chunk_t,
        Sk_chunk_t,
        q_local_padded_Nt,
        kv_local_padded_Nt,
        compile_time_logical_n,
        logical_nt,
        0,  // Lt
        0,  // L
        num_local_q_chunks,
        num_local_k_chunks,
        0,  // num_joint_k_chunks
        num_res_q_chunks,
        ring_size,
        qk_in0_block_w,
        qk_out_subblock_w,
        qk_out_subblock_h,
        qk_in0_num_subblocks,
        qk_in1_num_subblocks,
        out_in0_block_w,
        out_out_subblock_w,
        out_out_subblock_h,
        out_in0_num_subblocks,
        out_in1_num_subblocks,
        scale_packed,
        static_cast<std::uint32_t>(use_streaming_compute),
        compile_time_global_n_partial_col,
        0,  // joint_l_partial_col
        0,  // is_causal
        0,  // is_balanced
        0,  // zigzag balancing
        0,  // chunked prefill
        q_chunk_group_tile_count,
        0,  // KV-pad rotation
        0,  // v_shares_k_buffer
        0,  // attention sink
        0,  // sliding_window_size
        0,  // KV pad from metadata
        0,  // joint is sharded
        0,  // logical_lt
        // Transport-to-tensor rank mapping.
        static_cast<uint32_t>(rank_mapping.full_mesh),
        static_cast<uint32_t>(rank_mapping.orientation),
        rank_mapping.mesh_rows,
        rank_mapping.mesh_cols,
        0,  // circular KV slab count
        0,  // logical_n tensor
        0,  // logical_l tensor
    };

    std::map<std::string, std::string> defines;
    defines["GROUPED_KV_SOURCE_COUNT"] = "1";
    defines["STATS_GRANULARITY"] = std::to_string(stats_granularity);
    defines["SUB_EXP_GRANULARITY"] = std::to_string(sub_exp_granularity);
    defines["DHT_GRANULARITY"] = std::to_string(dht_granularity);
    defines["REDUCE_GRANULARITY"] = std::to_string(reduce_granularity);
    defines["EXP_APPROX_MODE"] = std::to_string(exp_approx_mode);
    if (args.program_config.matmul_math_fidelity.has_value()) {
        defines["SDPA_MATMUL_FIDELITY"] =
            std::to_string(static_cast<uint32_t>(*args.program_config.matmul_math_fidelity));
    }
    // MATH_FIDELITY is not defined on the unpack TRISC, and all three must agree on how P.V is set up.
    if (math_fidelity == MathFidelity::LoFi) {
        defines["SDPA_COMPUTE_LOFI"] = "1";
    }
    // Dense units: one copy of the bodies reached from the ring loop and the dense batches (sparse_dense_units.hpp);
    // the kernel config buffer holds all kernel text.
    if (dense_h) {
        defines["SPARSE_DENSE_UNITS"] = "1";
    }

    // NOTE: KernelDescriptors are built after chain construction.

    // Create circular buffers

    tt::DataFormat q_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_q.dtype());
    tt::DataFormat k_df = tt::tt_metal::datatype_to_dataformat_converter(gathered_input_tensor_k.dtype());
    tt::DataFormat v_df = tt::tt_metal::datatype_to_dataformat_converter(gathered_input_tensor_v.dtype());

    // The pooled-column bias tiles are Float16_b: the step adds them into the scores by L1 accumulation.
    tt::DataFormat mask_df = tt::DataFormat::Float16_b;
    tt::DataFormat out_df = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    tt::DataFormat scalar_df = tt::DataFormat::Float16_b;
    // Intermediates stay in bf16 to save L1 (fp32 dest accumulation is off).
    tt::DataFormat im_df = tt::DataFormat::Float16_b;
    tt::DataFormat stats_df = im_df;
    tt::DataFormat qk_im_df = tt::DataFormat::Float16_b;
    // The output and sum slots are Float32: the packer accumulates every step onto them, so small late increments
    // are not rounded away.
    tt::DataFormat state_df = tt::DataFormat::Float32;

    uint32_t q_tile_size = tt::tile_size(q_df);
    uint32_t k_tile_size = tt::tile_size(k_df);
    uint32_t v_tile_size = tt::tile_size(v_df);
    uint32_t mask_tile_size = tt::tile_size(mask_df);
    uint32_t out_tile_size = tt::tile_size(out_df);
    uint32_t scalar_tile_size = tt::tile_size(scalar_df);
    uint32_t im_tile_size = tt::tile_size(im_df);
    uint32_t stats_tile_size = tt::tile_size(stats_df);
    uint32_t qk_im_tile_size = tt::tile_size(qk_im_df);
    uint32_t state_tile_size = tt::tile_size(state_df);

    log_debug(tt::LogOp, "data formats: q {}, k {}, v {}, out {}, scalar {}", q_df, k_df, v_df, out_df, scalar_df);

    uint32_t next_cb_index = 0;
    const auto allocate_cb = [&](uint32_t page_size_bytes, uint32_t num_pages, tt::DataFormat data_format) -> uint32_t {
        const uint32_t cb_index = next_cb_index++;
        desc.cbs.push_back(CBDescriptor{
            .total_size = page_size_bytes * num_pages,
            .core_ranges = core_grid_set,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_index),
                .data_format = data_format,
                .page_size = page_size_bytes,
            }}},
        });
        return cb_index;
    };

    // Slot-dependent CBs ("slots" = Q blocks per core per pass) are laid out by lay_out_cbs, which the pass split runs
    // with a byte counter per candidate and then with the real allocator, so the fit check and allocation agree.
    // A step's scores (cb_qk_im) hold up to kResidentGroup Q blocks of Sq_chunk_t x Sk_chunk_t tiles, twice that with
    // pair steps; the compute kernel derives the same capacity.
    constexpr uint32_t kResidentGroup = 4;
    const bool has_block_counts = tensor_args.block_counts.has_value();
    const uint32_t block_counts_bytes = tt::round_up(mask_n_blocks_pad, 32u);
    const uint32_t key_bias_tiles =
        (has_block_counts || args.logical_n % tt::constants::TILE_WIDTH != 0) ? 2 * (Sk_chunk_t / block_tiles) : 0u;
    struct CbLayout {
        uint32_t q_in, k_in, v_in, identity_scale_in, col_identity, qk_im, max_A, exp_max_diff, out, stats_in, prev_out,
            recip_scratch, sum_in, sparse_plan, pooled_bias, block_counts, key_bias, sum_identity, dense_in, balance;
    };
    // group: Q blocks a step holds at full width (kResidentGroup; more for dense units, see below).
    const auto lay_out_cbs =
        [&](uint32_t slots, uint32_t kv_pages, bool pairs, uint32_t group, const auto& alloc) -> CbLayout {
        const auto tile_cb = [&](uint32_t num_tiles, uint32_t tile_size, tt::DataFormat data_format) {
            return alloc(tile_size, num_tiles, data_format);
        };
        CbLayout l{};
        l.q_in = tile_cb(Sq_chunk_t * DHt * slots, q_tile_size, q_df);
        l.k_in = tile_cb(k_piece_tiles * kv_pages, k_tile_size, k_df);
        l.v_in = tile_cb(v_piece_tiles * kv_pages, v_tile_size, v_df);
        l.identity_scale_in = tile_cb(scale_tiles, scalar_tile_size, scalar_df);
        l.col_identity = tile_cb(scale_tiles, scalar_tile_size, scalar_df);
        // The compute step's row-sum operand (an identity tile, from the writer): the sum slots keep one partial sum
        // per column, which normalization collapses with col_identity.
        l.sum_identity = tile_cb(1, scalar_tile_size, scalar_df);
        const uint32_t qk_im_cb_tiles =
            std::max(qk_tiles, std::min(slots, group) * Sq_chunk_t * Sk_chunk_t * (pairs ? 2u : 1u));
        const uint32_t res_stats_tiles = statistics_tiles * slots;
        l.qk_im = tile_cb(qk_im_cb_tiles, qk_im_tile_size, qk_im_df);
        // Staging CBs (max, out, sum): one accumulator set per slot, updated in place and never pushed or popped
        // until the pass's final normalization, so read and write indices name the same tile.
        l.max_A = tile_cb(res_stats_tiles, stats_tile_size, stats_df);
        l.exp_max_diff = tile_cb(res_stats_tiles, stats_tile_size, stats_df);
        l.out = tile_cb(out0_t, out_tile_size, out_df);
        l.stats_in = tile_cb(statistics_tiles * slots, im_tile_size, im_df);
        l.prev_out = tile_cb(out_im_tiles * slots, state_tile_size, state_df);
        l.recip_scratch = tile_cb(1, im_tile_size, im_df);  // 1-tile recip scratch for res_normalize_slot
        l.sum_in = tile_cb(statistics_tiles * slots, state_tile_size, state_df);
        // Plan pages: a 16-word summary [kept-block words][union piece bits lo, hi][do_pooled], then the Q block's
        // mask row. Two pages: compute reads one iteration's records while the reader fills the next.
        l.sparse_plan = alloc(64 + mask_row_bytes, 2, tt::DataFormat::UInt32);
        // One pooled chunk of bias tiles in flight; only with the pooled route.
        l.pooled_bias = pooled_route ? tile_cb(Sk_chunk_t, mask_tile_size, mask_df)
                                     : std::numeric_limits<uint32_t>::max();  // inactive
        // One byte per global block, filled once by the reader.
        l.block_counts = has_block_counts ? alloc(block_counts_bytes, 1, tt::DataFormat::UInt32)
                                          : std::numeric_limits<uint32_t>::max();  // inactive
        // Key bias: one column-bias tile per block whose real tokens end inside a tile, two pieces' worth. Needed
        // only with block_counts or a logical_n off a tile boundary.
        l.key_bias = key_bias_tiles > 0 ? tile_cb(key_bias_tiles, mask_tile_size, mask_df)
                                        : std::numeric_limits<uint32_t>::max();  // inactive
        // One arriving dense partial (bf16, the output's format), merged one at a time.
        l.dense_in = dense_h && !dense_debug ? tile_cb(partial_tiles, out_tile_size, out_df)
                                             : std::numeric_limits<uint32_t>::max();  // inactive
        // Load balancing: the core's unit table, written by the balance kernel (sparse_balance::kTableWords words).
        l.balance = balance ? alloc(sparse_balance::kTableWords * sizeof(uint32_t), 1, tt::DataFormat::UInt32)
                            : std::numeric_limits<uint32_t>::max();  // inactive
        return l;
    };
    const auto cb_bytes_for_slots =
        [&](uint32_t slots, uint32_t kv_pages, bool pairs, uint32_t group = kResidentGroup) -> uint64_t {
        uint64_t bytes = 0;
        uint32_t index = 0;
        lay_out_cbs(
            slots,
            kv_pages,
            pairs,
            group,
            [&](uint32_t page_size_bytes, uint32_t num_pages, tt::DataFormat) -> uint32_t {
                bytes += static_cast<uint64_t>(page_size_bytes) * num_pages;
                return index++;
            });
        return bytes;
    };

    // Pass split. S = slots per pass: the largest count (at most kResidentMaxSlots, the core's block count and
    // q_blocks_per_pass) whose CBs end below the lowest live L1 buffer. P passes split every core's blocks evenly, so
    // each working core needs at least P blocks (no empty pass; group members stay in lockstep).
    // The live L1 top is not in the program hash: a cached program keeps the S of the call that built it.
    const auto lowest_l1_buffer = mesh_device->lowest_occupied_compute_l1_address();
    const uint32_t cb_space_top = lowest_l1_buffer.has_value() ? static_cast<uint32_t>(lowest_l1_buffer.value())
                                                               : mesh_device->l1_size_per_core();
    const uint32_t usable_l1 =
        cb_space_top - mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    uint32_t max_slots = std::min(kResidentMaxSlots, max_q_per_core);
    if (args.q_blocks_per_pass.has_value()) {
        max_slots = std::min(max_slots, *args.q_blocks_per_pass);
    }
    const auto fit_slots_for = [&](uint32_t kv_pages, bool pairs) -> uint32_t {
        uint32_t slots = max_slots;
        while (slots > 0 && cb_bytes_for_slots(slots, kv_pages, pairs) > usable_l1) {
            --slots;
        }
        return slots;
    };
    const auto passes_for = [&](uint32_t slots) {
        return slots > 0 ? tt::div_up(max_q_per_core, slots) : std::numeric_limits<uint32_t>::max();
    };
    // The K/V depth with the fewest passes (a pass re-streams every piece and repeats every step's fixed cost); at
    // equal passes the deeper one: pair steps (4 pages), then the default depth (3 with group delivery), then 2.
    struct KvLayout {
        uint32_t pages;
        bool pairs;
    };
    std::vector<KvLayout> kv_layouts;
    if (pair_steps_eligible) {
        kv_layouts.push_back({4, true});
    }
    kv_layouts.push_back({kv_cb_pages, false});
    if (kv_cb_pages > 2) {
        kv_layouts.push_back({2, false});
    }
    uint32_t fit_slots = 0;
    for (const auto& layout : kv_layouts) {
        const uint32_t slots = fit_slots_for(layout.pages, layout.pairs);
        if (slots > 0 && passes_for(slots) < passes_for(fit_slots)) {
            fit_slots = slots;
            kv_cb_pages = layout.pages;
            pair_steps = layout.pairs;
        }
    }
    TT_FATAL(
        fit_slots > 0,
        "SparseRingSDPA CBs need {} B with one Q block per core but only {} B of L1 are usable (k_chunk={}); reduce "
        "k_chunk_size.",
        cb_bytes_for_slots(1, 2, false),
        usable_l1,
        Sk_chunk_t * tt::constants::TILE_HEIGHT);
    const uint32_t num_q_passes = passes_for(fit_slots);
    // Every working core holds base or base + 1 blocks (all hold one when there are fewer blocks than cores).
    const uint32_t min_q_per_working_core = std::max(1u, all_heads_num_q_chunks / num_cores);
    TT_FATAL(
        num_q_passes <= min_q_per_working_core,
        "SparseRingSDPA: {} passes of at most {} Q blocks would leave a core with {} blocks an empty pass; allow more "
        "Q blocks per pass (q_blocks_per_pass) or reduce k_chunk_size",
        num_q_passes,
        fit_slots,
        min_q_per_working_core);
    TT_FATAL(
        num_q_passes == 1 || (!gqa_grouped_kv && !k_uses_batch_chain),
        "SparseRingSDPA: several Q passes need separate K and V heads per query head (MHA), got NH {} NHK {} NHV {}",
        NH,
        NHK,
        NHV);
    const uint32_t pass_slots = tt::div_up(max_q_per_core, num_q_passes);
    // Dense builds: a dense batch takes up to all of a core's dense units and a step up to all of a batch's units (one
    // step per piece pays the step's fixed cost once), as far as that fits L1 next to the pass split's choice (the
    // same passes, K/V depth and pair steps): more slots first (fewer batches), then wider steps.
    // Dense units: every head's n_dense_blocks units on dense_cores_per_head cores of their own (see dense_range).
    const uint32_t n_dense_blocks = dense_units / NH;
    const uint32_t dense_cores_per_head = dense_h ? num_cores / NH : 0;
    TT_FATAL(!dense_h || dense_cores_per_head > 0, "SparseRingSDPA: dense_blocks need a core per head");
    uint32_t dense_slots = pass_slots;
    uint32_t step_members = kResidentGroup;
    if (dense_h) {
        const uint32_t du_max = tt::div_up(n_dense_blocks, dense_cores_per_head);  // the most units on one core
        for (uint32_t n = std::min(kResidentMaxSlots, std::max(pass_slots, du_max)); n > pass_slots; --n) {
            if (cb_bytes_for_slots(n, kv_cb_pages, pair_steps) <= usable_l1) {
                dense_slots = n;
                break;
            }
        }
        for (uint32_t g = std::min(dense_slots, du_max); g > kResidentGroup; --g) {
            if (cb_bytes_for_slots(dense_slots, kv_cb_pages, pair_steps, g) <= usable_l1) {
                step_members = g;
                break;
            }
        }
    }
    uint32_t resident_slots = dense_slots;
    if (balance) {
        // Load balancing moves units between a head's cores: a quarter more slots (at least 2) where L1 allows them
        // (never past q_blocks_per_pass, never an extra pass). Light cores take the extra units of heavy ones. The
        // extra slots are optional, so they leave kBalanceL1Reserve free: the cached program's CBs must not reach L1
        // buffers allocated after it was built (a call's fresh inputs or semaphores).
        constexpr uint32_t kBalanceL1Reserve = 32 * 1024;
        const uint32_t headroom_l1 = usable_l1 > kBalanceL1Reserve ? usable_l1 - kBalanceL1Reserve : 0;
        const uint32_t most = std::min(
            {kResidentMaxSlots,
             pass_slots + std::max(2u, tt::div_up(pass_slots, 4u)),
             args.q_blocks_per_pass.value_or(kResidentMaxSlots)});
        for (uint32_t n = most; n > resident_slots; --n) {
            if (cb_bytes_for_slots(n, kv_cb_pages, pair_steps) <= headroom_l1) {
                resident_slots = n;
                break;
            }
        }
    }
    // A core's most units under load balancing (its unit table's room).
    const uint32_t balance_cap = std::min(sparse_balance::kMaxCoreUnits, num_q_passes * resident_slots);
    log_debug(
        tt::LogOp,
        "SparseRingSDPA: {} Q blocks per core at most, {} pass(es) of at most {}, {} K/V pages, pair steps {} "
        "({} of {} B of L1 for CBs)",
        max_q_per_core,
        num_q_passes,
        resident_slots,
        kv_cb_pages,
        pair_steps ? "on" : "off",
        cb_bytes_for_slots(resident_slots, kv_cb_pages, pair_steps, step_members),
        usable_l1);

    const CbLayout cbs = lay_out_cbs(resident_slots, kv_cb_pages, pair_steps, step_members, allocate_cb);
    if (balance) {
        // The unit-table CB (the last one laid out) also on the balance core: a CB has one address on all its cores,
        // so the balance kernel writes each SDPA core's table at its own copy's address.
        auto& table_cb = desc.cbs.back();
        TT_FATAL(table_cb.format_descriptors[0].buffer_index == cbs.balance, "SparseRingSDPA: unit-table CB not last");
        table_cb.core_ranges = table_cb.core_ranges.merge(CoreRangeSet(CoreRange(balancing.core)));
    }
    const uint32_t cb_q_in = cbs.q_in;
    const uint32_t cb_k_in = cbs.k_in;
    const uint32_t cb_v_in = cbs.v_in;
    const uint32_t cb_identity_scale_in = cbs.identity_scale_in;
    const uint32_t cb_col_identity = cbs.col_identity;
    const uint32_t cb_qk_im = cbs.qk_im;
    const uint32_t cb_max_A = cbs.max_A;
    const uint32_t cb_exp_max_diff = cbs.exp_max_diff;
    const uint32_t cb_out = cbs.out;
    const uint32_t cb_stats_in = cbs.stats_in;
    const uint32_t cb_prev_out = cbs.prev_out;
    const uint32_t cb_recip_scratch = cbs.recip_scratch;
    const uint32_t cb_sum_in = cbs.sum_in;
    const uint32_t cb_sparse_plan = cbs.sparse_plan;
    // Without the pooled route the bias CB is not allocated; the kernels still take an id (and its tile size), so
    // they get Q's, which they never touch through it.
    const uint32_t cb_pooled_bias = pooled_route ? cbs.pooled_bias : cbs.q_in;
    const uint32_t cb_block_counts = has_block_counts ? cbs.block_counts : cbs.q_in;
    const uint32_t cb_key_bias = key_bias_tiles > 0 ? cbs.key_bias : cbs.q_in;

    // The dense op's padding-mask and scale CBs: the resident step masks padding by narrowing its K width and applies
    // the scale inside its exp, so neither is allocated. Their slots in the shared CB argument block stay.
    constexpr uint32_t inactive_cb = std::numeric_limits<uint32_t>::max();
    const uint32_t cb_mask_in = inactive_cb;
    const uint32_t cb_scale_in = inactive_cb;
    // No attention sink: alias a valid CB so compile-time tile-size queries remain valid.
    const uint32_t cb_attention_sink = cb_q_in;
    // Slots of the dense op's CB argument block no kernel of this op reads: the second accumulator set, the
    // save / restore and signal CBs, and the metadata mailbox. Not allocated; the slots stay so the indices hold.
    const uint32_t cb_out_im_A = inactive_cb;
    const uint32_t cb_out_im_B = inactive_cb;
    const uint32_t cb_max_B = inactive_cb;
    const uint32_t cb_sum_A = inactive_cb;
    const uint32_t cb_sum_B = inactive_cb;
    const uint32_t cb_stats_out = inactive_cb;
    const uint32_t cb_sum_out = inactive_cb;
    const uint32_t cb_signal = inactive_cb;
    const uint32_t cb_kv_pad_derived = inactive_cb;
    // All Q blocks' plan records go in one page per ring iteration, after the 16-word header and the shard slice
    // (mask_nb_pad words), when they fit in the page; otherwise each block gets its own page (plan_rec_base = 0).
    const uint32_t plan_rec_words = tt::div_up(mask_nb, 32u) + 3;
    const bool plan_fits_one_page =
        (16 + mask_nb_pad + resident_slots * plan_rec_words) * sizeof(uint32_t) <= 64 + mask_row_bytes;
    const uint32_t plan_rec_base = plan_fits_one_page ? 16 + mask_nb_pad : 0;
    {
        uint64_t cb_bytes = 0;
        for (const auto& cb : desc.cbs) {
            cb_bytes += cb.core_ranges.intersects(core_grid_set) ? cb.total_size : 0;
        }
        log_debug(
            tt::LogOp,
            "SparseRingSDPA CBs: {} buffers, {} bytes ({} slots; k {} t, v {} t, qk {} t, out_im {} t, stats {} t; "
            "im tile {} B)",
            desc.cbs.size(),
            cb_bytes,
            resident_slots,
            k_piece_tiles * kv_cb_pages,
            v_piece_tiles * kv_cb_pages,
            qk_tiles,
            out_im_tiles,
            statistics_tiles,
            im_tile_size);
    }

    const std::vector<uint32_t> cb_compile_time_args = {
        cb_q_in,     cb_k_in,     cb_v_in,         cb_mask_in,       cb_scale_in,     cb_identity_scale_in,
        cb_stats_in, cb_prev_out, cb_col_identity, cb_recip_scratch, cb_sum_out,      cb_sum_in,
        cb_signal,   cb_out,      cb_stats_out,    cb_qk_im,         cb_out_im_A,     cb_out_im_B,
        cb_max_A,    cb_max_B,    cb_sum_A,        cb_sum_B,         cb_exp_max_diff, cb_kv_pad_derived};
    const std::vector<uint32_t> reader_cb_compile_time_args = {
        cb_q_in, cb_k_in, cb_v_in, cb_attention_sink, cb_kv_pad_derived};
    reader_compile_time_args.insert(
        reader_compile_time_args.end(), reader_cb_compile_time_args.begin(), reader_cb_compile_time_args.end());
    writer_compile_time_args.insert(
        writer_compile_time_args.end(), cb_compile_time_args.begin(), cb_compile_time_args.end());
    auto compute_cb_compile_time_args = cb_compile_time_args;
    compute_cb_compile_time_args.push_back(cb_attention_sink);
    compute_compile_time_args.insert(
        compute_compile_time_args.end(), compute_cb_compile_time_args.begin(), compute_cb_compile_time_args.end());

    auto* const q_buf = input_tensor_q.buffer();
    auto* const k_buf = input_tensor_k.buffer();
    auto* const v_buf = input_tensor_v.buffer();
    auto* const gathered_k_buf = gathered_input_tensor_k.buffer();
    auto* const gathered_v_buf = gathered_input_tensor_v.buffer();
    auto* const out_buf = output_tensor.buffer();

    // K/V chains and multicast groups (ring_joint_kv_chains.hpp, shared with the ring joint op).
    using ring_joint::ChainConfig;
    using ring_joint::ChainSegment;
    using ring_joint::CoreWork;
    using ring_joint::HeadSegmentRef;
    std::vector<CoreWork> core_work(num_cores);
    std::vector<ChainConfig> head_chain_configs(use_head_chain ? num_cores : 0);       // MHA K/V or separate-V V
    std::vector<ChainConfig> batch_chain_configs(k_uses_batch_chain ? num_cores : 0);  // Shared K for separate-V
    std::vector<ChainConfig> gqa_chain_configs(gqa_grouped_kv ? num_cores : 0);        // Grouped K/V for GQA
    std::vector<std::vector<HeadSegmentRef>> head_segments(use_head_chain ? NH : 0);

    // Evenly distribute flat global Q blocks across cores
    const uint32_t cores_doing_extra_work = all_heads_num_q_chunks % num_cores;
    const uint32_t base_chunks_per_core = all_heads_num_q_chunks / num_cores;
    uint32_t next_global_chunk = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};
        uint32_t chunk_count = base_chunks_per_core + ((i < cores_doing_extra_work) ? 1 : 0);
        if (next_global_chunk >= all_heads_num_q_chunks) {
            chunk_count = 0;
        } else if (chunk_count > all_heads_num_q_chunks - next_global_chunk) {
            chunk_count = all_heads_num_q_chunks - next_global_chunk;
        }

        auto& work = core_work.at(i);
        work.physical_core = mesh_device->worker_core_from_logical_core(core);
        work.global_q_start = next_global_chunk;
        work.global_q_count = chunk_count;
        ring_joint::append_head_work(core_work, head_segments, i, next_global_chunk, chunk_count, num_res_q_chunks, NH);

        next_global_chunk += chunk_count;
    }

    std::vector<uint32_t> gqa_chain_max_q(gqa_chain_configs.size(), 0);  // per-core loop-padding count
    ring_joint::GqaKvChains gqa_chains;
    if (gqa_grouped_kv) {
        gqa_chains = ring_joint::build_gqa_kv_chains(
            core_work, gqa_chain_configs, gqa_chain_max_q, NHK, q_heads_per_kv, grid_size, mesh_device);
    }
    if (k_uses_batch_chain) {
        ring_joint::build_shared_k_chain(core_work, batch_chain_configs);
    }
    std::vector<uint32_t> k_chain_max_q(batch_chain_configs.size(), 0);  // per-core loop-padding count
    std::string k_mcast_fallback_reason;
    const bool k_mcast_enabled =
        k_uses_batch_chain &&
        ring_joint::build_shared_k_mcast(
            core_work, batch_chain_configs, k_chain_max_q, grid_size, mesh_device, k_mcast_fallback_reason);

    // The kernels read rotated_max_slots (the dense op's rotated Q split) as their LAST compile-time arg. Resident Q
    // blocks never change owner, so it is 0.
    constexpr uint32_t rotated_max_slots_ct = 0;
    for (auto* ct_args : {&reader_compile_time_args, &writer_compile_time_args, &compute_compile_time_args}) {
        ct_args->push_back(rotated_max_slots_ct);
    }
    const std::array<size_t, 3> ct_arg_sizes_with_rotated_last = {
        reader_compile_time_args.size(), writer_compile_time_args.size(), compute_compile_time_args.size()};

    // Build static MHA/shared-K V head chains. GQA uses KV-head-grouped chains instead.
    if (use_head_chain) {
        for (uint32_t head_id = 0; head_id < static_cast<uint32_t>(head_segments.size()); ++head_id) {
            const auto& segs = head_segments[head_id];
            if (segs.size() < 2) {
                continue;
            }
            std::vector<ChainSegment> chain_segs;
            chain_segs.reserve(segs.size());
            for (const auto& seg : segs) {
                chain_segs.emplace_back(
                    seg.core_idx, core_work[seg.core_idx].head_work[seg.head_work_index].q_chunk_count);
            }
            // A core spanning two heads streams both into the same K/V CBs, so its CB pointer drifts from the chain's
            // and a forward could overwrite an entry in use. Such cores read their own K/V: drop them from the tail.
            while (!chain_segs.empty() && core_work[chain_segs.back().first].head_work.size() > 1) {
                chain_segs.pop_back();
            }
            ring_joint::build_linear_chain(chain_segs, head_id, head_chain_configs, core_work);
        }
    }

    // The head chains are never multicast (group delivery below replaces that). Patch the shared-K and GQA
    // multicast flags.
    {
        const uint32_t head_chain_compile_args = use_head_chain ? kRingJointChainCompileArgCount : 0;
        if (k_uses_batch_chain) {
            const uint32_t batch_mcast_arg_index =
                sem_args_offset + head_chain_compile_args + kRingJointChainMcastEnabledCompileArgOffset;
            reader_compile_time_args[batch_mcast_arg_index] = k_mcast_enabled ? 1 : 0;
        }
        if (gqa_grouped_kv) {
            const uint32_t batch_chain_compile_args = k_uses_batch_chain ? kRingJointChainCompileArgCount : 0;
            const uint32_t gqa_mcast_arg_index = sem_args_offset + head_chain_compile_args + batch_chain_compile_args +
                                                 kRingJointChainMcastEnabledCompileArgOffset;
            reader_compile_time_args[gqa_mcast_arg_index] = gqa_chains.mcast_enabled ? 1 : 0;
        }
    }

    if (gqa_grouped_kv) {
        log_debug(
            tt::LogOp,
            "K/V chain mode: grouped GQA {} (chains={}, participant_cores={}, local_fallback_cores={})",
            gqa_chains.mcast_enabled ? "mcast" : fmt::format("unicast, {}", gqa_chains.mcast_fallback_reason),
            gqa_chains.grouped_chains,
            gqa_chains.grouped_participant_cores,
            gqa_chains.local_fallback_cores);
    } else if (k_uses_batch_chain) {
        log_debug(
            tt::LogOp,
            "K chain mode: batch ({})",
            k_mcast_enabled ? "mcast" : fmt::format("unicast, {}", k_mcast_fallback_reason));
    } else {
        log_debug(tt::LogOp, "K chain mode: head");
    }

    // Convert std::map<string,string> defines to KernelDescriptor::Defines vector form.
    KernelDescriptor::Defines kernel_defines(defines.begin(), defines.end());

    KernelDescriptor reader_kernel{};
    reader_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/sparse_ring_reader.cpp";
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
    reader_kernel.defines = kernel_defines;
    reader_kernel.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_kernel{};
    writer_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/sparse_ring_writer.cpp";
    writer_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel.core_ranges = core_grid_set;
    writer_kernel.compile_time_args = writer_compile_time_args;
    writer_kernel.defines = kernel_defines;
    writer_kernel.config = WriterConfigDescriptor{};
    // The reader and writer are built for size: the kernel config buffer holds all five kernels' text, and the long
    // builds (several passes, pair steps, LoFi with block-float K / V) come within a few hundred bytes of it at O2.
    // Their time is mostly NoC waits.
    reader_kernel.opt_level = tt::tt_metal::KernelBuildOptLevel::Os;
    writer_kernel.opt_level = tt::tt_metal::KernelBuildOptLevel::Os;

    KernelDescriptor compute_kernel{};
    compute_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/sparse_ring_sdpa.cpp";
    compute_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel.core_ranges = core_grid_set;
    compute_kernel.compile_time_args = compute_compile_time_args;
    // Named compile-time args shared by the three kernels.
    const KernelDescriptor::NamedCompileTimeArgs named_args = {
        // The kernels share the dense op's chain and ring helpers, which take a KV split factor; no split here.
        {"kv_stripe_split", 1u},
        // Mask block size in tiles (square blocks: Q block rows == K block columns). Only 2 (64 tokens) for now.
        {"block_tiles", block_tiles},
        // Q blocks per core per pass, and the pass count (the same on every core).
        {"resident_slots", resident_slots},
        // Q blocks a step holds at full width (the score scratch's capacity; kResidentGroup without dense units).
        {"step_members", step_members},
        {"num_q_passes", num_q_passes},
        // Pair steps (see pair_steps_eligible above) and the K/V CB depth in pieces.
        {"pair_steps", pair_steps ? 1u : 0u},
        {"kv_cb_pages", kv_cb_pages},
        // Words of per-shard kept-block bits in each plan page (block b: word b/32, bit b%32).
        {"plan_block_words", tt::div_up(mask_nb, 32u)},
        // Group delivery by multicast (see group_mcast above).
        {"group_mcast", group_mcast ? 1u : 0u},
        // The one-page plan's record base word (0 = a page per block); see plan_rec_base above.
        {"plan_rec_base", plan_rec_base},
        {"mask_nb", mask_nb},
        {"mask_nb_pad", mask_nb_pad},
        {"mask_n_blocks_pad", mask_n_blocks_pad},
        {"mask_nqb_pad", mask_nqb_pad},
        // The pooled route (Sol-Attn): pooled entries and their column bias. 0: kept blocks only.
        {"pooled_route", pooled_route ? 1u : 0u},
        {"pooled_chunks", pooled_chunks},
        {"pooled_last_tiles", pooled_last_tiles},
        {"mask_row_bytes", mask_row_bytes},
        // Logical length in tokens, for the on-device pooled-column bias.
        {"sparse_logical_n", static_cast<uint32_t>(args.logical_n)},
        {"cb_sparse_plan", cb_sparse_plan},
        {"cb_pooled_bias", cb_pooled_bias},
        // Real tokens per block: given (block_counts) or the geometric lengths; see the CB layout above.
        {"has_block_counts", has_block_counts ? 1u : 0u},
        {"cb_block_counts", cb_block_counts},
        {"key_bias_tiles", key_bias_tiles},
        {"cb_key_bias", cb_key_bias},
        {"cb_sum_identity", cbs.sum_identity},
        // dense_blocks split across devices (sparse_dense_units.hpp): owner records, partials to dense_partials_debug
        // instead of the hop path, this shard's resident blocks skip its dense ones.
        {"dense_owners", static_cast<uint32_t>(owner_plans.size())},
        {"dense_debug", dense_debug ? 1u : 0u},
        {"res_remap", res_remap ? 1u : 0u},
        {"cb_dense_in", dense_h && !dense_debug ? cbs.dense_in : cbs.q_in},
        {"dense_units", dense_units},
        // DRAM banks: a unit's partial is one bank's consecutive pages in the hop buffers.
        {"dense_banks", mesh_device->allocator()->get_num_banks(tt::tt_metal::BufferType::DRAM)},
        {"dense_q_rows_t", dense_q_rows_t},
        {"dense_blocks_total", std::max(1u, dense_units / NH)},
        // Load balancing (sparse_ring_balance.cpp): on, cross-device, a core's cost words, the semaphores (see
        // BalanceSemaphores) and the unit table's CB.
        {"balance", balance ? 1u : 0u},
        {"balance_remote", balancing.remote ? 1u : 0u},
        {"balance_cost_words", balancing.cost_words},
        {"balance_sem_table", balance_sems.table},
        {"balance_sem_import", balance_sems.import},
        {"balance_sem_costs", balance_sems.costs},
        {"balance_sem_outputs", balance_sems.outputs},
        {"cb_balance", balance ? cbs.balance : cbs.q_in},
    };
    for (auto* kernel : {&reader_kernel, &writer_kernel, &compute_kernel}) {
        kernel->named_compile_time_args = named_args;
    }
    reader_kernel.named_compile_time_args.emplace_back(
        "k_row_Wt", input_tensor_k.padded_shape()[3] / tt::constants::TILE_WIDTH);
    reader_kernel.named_compile_time_args.emplace_back(
        "v_row_Wt", input_tensor_v.padded_shape()[3] / tt::constants::TILE_WIDTH);
    // dense_blocks: the range count (the ranges are runtime args) and whether every Q block also keeps them.
    reader_kernel.named_compile_time_args.emplace_back(
        "dense_n_ranges", static_cast<uint32_t>(args.dense_blocks.size()));
    reader_kernel.named_compile_time_args.emplace_back("dense_keys", args.dense_keys ? 1u : 0u);
    reader_kernel.named_compile_time_args.emplace_back("dense_sem_sender", dense_sems ? dense_sems->sender_id : 0u);
    reader_kernel.named_compile_time_args.emplace_back("dense_sem_receiver", dense_sems ? dense_sems->receiver_id : 0u);
    reader_kernel.named_compile_time_args.emplace_back("dense_sem_valid", dense_sems ? dense_sems->valid_id : 0u);
    writer_kernel.named_compile_time_args.emplace_back(
        "dense_n_ranges", static_cast<uint32_t>(args.dense_blocks.size()));
    compute_kernel.defines = kernel_defines;
    // Built at O2 to fit the kernel config buffer.
    compute_kernel.opt_level = tt::tt_metal::KernelBuildOptLevel::O2;
    compute_kernel.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
    };

    // Pooled-column bias per block token count: bf16 (round-to-nearest-even) ln(len) / scale, -30000 for len 0,
    // two per word for len = 0 .. block_tokens + 1 (the last unused). The generator weights each pooled column by its
    // count through exp((QK + ln(len)/scale - max) * scale) = len * exp(...). The reader derives the same word count.
    const uint32_t kPooledBiasTableWords = block_tiles * 16 + 1;
    std::vector<uint32_t> pooled_bias_table(kPooledBiasTableWords, 0u);
    {
        const auto f2bf = [](float f) -> uint32_t {
            uint32_t u = std::bit_cast<uint32_t>(f);
            const uint32_t lsb = (u >> 16) & 1u;
            u += 0x7FFFu + lsb;
            return u >> 16;
        };
        constexpr float kPooledBiasNeg = -30000.0f;  // exp underflows to 0 in bf16
        for (uint32_t len = 0; len <= block_tokens + 1; ++len) {
            const float v =
                (len == 0 || len > block_tokens) ? kPooledBiasNeg : std::log(static_cast<float>(len)) / scale_value;
            pooled_bias_table[len / 2] |= f2bf(v) << (16 * (len % 2));
        }
    }

    // Group delivery: per core [role, producer x, producer y, receivers, rectangles, (x0, y0, x1, y1, cores)...].
    // A group is a head chain's members; the producer (first member in core order) reads every piece from DRAM and
    // multicasts it to rectangles of consecutive NoC x on one row each, so no multicast lands outside the group.
    constexpr uint32_t kGroupMaxRects = 8;
    // The receivers of a group as rectangles of consecutive NoC x on one row each (x0, y0, x1, y1, cores).
    const auto group_rects = [&](const std::vector<uint32_t>& members, uint32_t producer) {
        std::map<uint32_t, std::vector<uint32_t>> receiver_xs_by_row;
        for (const uint32_t ci : members) {
            if (ci != producer) {
                receiver_xs_by_row[core_work[ci].physical_core.y].push_back(core_work[ci].physical_core.x);
            }
        }
        std::vector<std::array<uint32_t, 5>> rects;
        for (auto& [y, xs] : receiver_xs_by_row) {
            std::sort(xs.begin(), xs.end());
            size_t run_start = 0;
            for (size_t k = 1; k <= xs.size(); ++k) {
                if (k == xs.size() || xs[k] != xs[k - 1] + 1) {
                    rects.push_back({xs[run_start], y, xs[k - 1], y, static_cast<uint32_t>(k - run_start)});
                    run_start = k;
                }
            }
        }
        return rects;
    };
    const auto set_group_args = [&](std::vector<std::vector<uint32_t>>& out,
                                    const std::vector<uint32_t>& members,
                                    uint32_t producer,
                                    const std::vector<std::array<uint32_t, 5>>& rects) {
        const CoreCoord prod_phys = core_work[producer].physical_core;
        for (const uint32_t ci : members) {
            auto& a = out[ci];
            a = {
                ci == producer ? 1u : 2u,
                static_cast<uint32_t>(prod_phys.x),
                static_cast<uint32_t>(prod_phys.y),
                static_cast<uint32_t>(members.size() - 1),
                0u};
            if (ci == producer) {
                a[4] = static_cast<uint32_t>(rects.size());
                for (const auto& r : rects) {
                    a.insert(a.end(), r.begin(), r.end());
                }
            }
        }
    };
    std::vector<std::vector<uint32_t>> group_args(num_cores, std::vector<uint32_t>{0u, 0u, 0u, 0u, 0u});
    if (group_mcast) {
        for (uint32_t h = 0; h < static_cast<uint32_t>(head_segments.size()); ++h) {
            std::vector<uint32_t> members;
            for (const auto& seg : head_segments[h]) {
                const auto& cfg = head_chain_configs.at(seg.core_idx);
                if (!cfg.participates || cfg.head != h ||
                    std::find(members.begin(), members.end(), seg.core_idx) != members.end()) {
                    continue;
                }
                TT_FATAL(
                    core_work[seg.core_idx].head_work.size() == 1,
                    "SparseRingSDPA group delivery: core {} of head {} spans two heads",
                    seg.core_idx,
                    h);
                members.push_back(seg.core_idx);
            }
            if (members.size() < 2) {
                continue;
            }
            const uint32_t producer = *std::min_element(members.begin(), members.end());
            const auto rects = group_rects(members, producer);
            TT_FATAL(
                rects.size() <= kGroupMaxRects,
                "SparseRingSDPA group delivery: head {} needs {} multicast rectangles (at most {})",
                h,
                rects.size(),
                kGroupMaxRects);
            set_group_args(group_args, members, producer, rects);
            log_debug(
                tt::LogOp,
                "SparseRingSDPA group: head {} producer core {} -> {} receivers in {} rectangles",
                h,
                producer,
                members.size() - 1,
                rects.size());
        }
        // The reader leaves the head chain's transfers out with group delivery (v_chain_moves): no member may be
        // left outside a group.
        for (uint32_t i = 0; i < num_cores; ++i) {
            TT_FATAL(
                !head_chain_configs[i].participates || group_args[i][0] != 0,
                "SparseRingSDPA: head chain member core {} has no delivery group",
                i);
        }
    }

    // Dense units: every head's n_dense_blocks units on dense_cores_per_head cores of their own (cores h x that ..,
    // base or base + 1 units each, the first ones one more), so a head's cores share its local pieces in one dense
    // group; the cores past NH x dense_cores_per_head hold none. (A flat split over every core puts cores across head
    // boundaries and unequal batches in a group, and their own K/V reads cost more than the cores save.)
    const auto dense_range = [&](uint32_t i) -> std::pair<uint32_t, uint32_t> {
        if (!dense_h || i >= NH * dense_cores_per_head) {
            return {0, 0};
        }
        const uint32_t h = i / dense_cores_per_head;
        const uint32_t c = i % dense_cores_per_head;
        const uint32_t base = n_dense_blocks / dense_cores_per_head;
        const uint32_t extra = n_dense_blocks % dense_cores_per_head;
        const uint32_t start = h * n_dense_blocks + c * base + std::min(c, extra);
        return {start, start + base + (c < extra ? 1u : 0u)};
    };
    // Dense groups: a head's cores with units, when each runs the same batches per owner (sparse_dense::for_each_batch
    // over resident_slots): they stream identical piece sequences, and a batch waits for one owner's arrivals on every
    // member, so the owners' hop chains stay independent. Else they read their own pieces.
    std::vector<std::vector<uint32_t>> dense_group_args(num_cores, std::vector<uint32_t>{0u, 0u, 0u, 0u, 0u});
    if (dense_sems.has_value()) {
        const auto n_batches = [&](uint32_t i) {
            const auto [lo, hi] = dense_range(i);
            std::vector<uint32_t> n;
            for (const auto& o : owner_plans) {
                const uint32_t start = (lo / std::max(1u, n_dense_blocks)) * n_dense_blocks + o.block_offset;
                const uint32_t a = std::max(lo, start);
                const uint32_t b = std::min(hi, start + o.n_blocks);
                n.push_back(b > a ? tt::div_up(b - a, resident_slots) : 0u);
            }
            return n;
        };
        for (uint32_t h = 0; h < NH; ++h) {
            std::vector<uint32_t> members;
            for (uint32_t i = h * dense_cores_per_head; i < (h + 1) * dense_cores_per_head; ++i) {
                if (dense_range(i).second > dense_range(i).first) {
                    members.push_back(i);
                }
            }
            if (members.size() < 2 || std::any_of(members.begin(), members.end(), [&](uint32_t i) {
                    return n_batches(i) != n_batches(members.front());
                })) {
                continue;
            }
            const auto rects = group_rects(members, members.front());
            if (rects.size() > kGroupMaxRects) {
                continue;
            }
            set_group_args(dense_group_args, members, members.front(), rects);
        }
    }
    const uint32_t ready_plus_addr = dense_res != nullptr ? static_cast<uint32_t>(dense_res->ready_plus.address()) : 0;
    const uint32_t ready_minus_addr =
        dense_res != nullptr ? static_cast<uint32_t>(dense_res->ready_minus.address()) : 0;
    const uint32_t arrival_minus_addr =
        dense_res != nullptr ? static_cast<uint32_t>(dense_res->arrival_minus.address()) : 0;
    const uint32_t arrival_plus_addr =
        dense_res != nullptr ? static_cast<uint32_t>(dense_res->arrival_plus.address()) : 0;

    // Set reader rt args
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};
        const auto [du_start, du_end] = dense_range(i);

        const auto& work = core_work.at(i);
        const uint32_t global_q_start = work.global_q_start;
        const uint32_t global_q_end = global_q_start + work.global_q_count;

        log_debug(tt::LogOp, "core {} ({},{}): q [{}, {})", i, core.x, core.y, global_q_start, global_q_end);

        CheckedRuntimeArgList reader_args;
        reader_args.push_back(q_buf);
        reader_args.push_back(k_buf);
        reader_args.push_back(v_buf);
        reader_args.push_back(gathered_k_buf);
        reader_args.push_back(gathered_v_buf);
        reader_args.push_back(attention_sink_buf);
        reader_args.push_back(global_q_start);
        reader_args.push_back(global_q_end);
        reader_args.push_checked(runtime_arg_layout.reader_kv_cache_batch_idx, 0u, "reader.kv_cache_batch_idx");
        if (use_head_chain) {
            // Head chain: MHA uses it for K/V; separate-V shared-K uses it for V only.
            std::vector<uint32_t> head_chain_args;
            head_chain_configs.at(i).append_to_args(head_chain_args);
            reader_args.append(head_chain_args);
        }
        if (k_uses_batch_chain) {
            std::vector<uint32_t> batch_chain_args;
            batch_chain_configs.at(i).append_to_args(batch_chain_args);
            reader_args.append(batch_chain_args);
            reader_args.push_back(k_chain_max_q[i]);
        }
        if (gqa_grouped_kv) {
            std::vector<uint32_t> gqa_chain_args;
            gqa_chain_configs.at(i).append_to_args(gqa_chain_args);
            reader_args.append(gqa_chain_args);
            reader_args.push_back(gqa_chain_max_q[i]);
        }

        reader_args.push_checked(runtime_arg_layout.reader_logical_nt, logical_nt, "reader.logical_nt");
        reader_args.push_checked(
            runtime_arg_layout.reader_active_ring_iter_mask, active_ring_iter_mask, "reader.active_ring_iter_mask");
        reader_args.push_back(kUnusedQMappingWord);  // q_pre_wrap_start_tile
        reader_args.push_back(kUnusedQMappingWord);  // q_pre_wrap_tile_count
        reader_args.push_back(kUnusedQMappingWord);  // q_post_wrap_start_tile
        reader_args.push_back(kUnusedQMappingWord);  // q_valid_tile_count

        // Inject fused-op synchronization RT args (AllGather) here; it will append to reader_args
        std::vector<uint32_t> reader_signaler_args;
        sdpa_fused_op_signaler->push_ring_sdpa_fused_op_rt_args(reader_signaler_args);
        reader_args.append(reader_signaler_args);

        // Block-sparse args, read by the kernel at rotated_args_base. Word offsets from there:
        //   +0 mask, +1 pooled K, +2 pooled V, +3 block counts (null: geometric), +4 n_members, +5 my_start, then the
        //   kPooledBiasTableWords packed bf16 table, then the dense_blocks ranges (start, end)..., then members
        //   (start, count)..., then the group delivery words.
        // Buffer entries are registered as bindings and re-patched on cache hits by the descriptor adapter.
        reader_args.push_back(tensor_args.block_mask.buffer());
        reader_args.push_back(pooled_k_buf);  // null without the pooled route; positions do not move
        reader_args.push_back(pooled_v_buf);
        reader_args.push_back(block_counts_buf);
        // Linear head chains: the chain's members as (Q block start within the head, count) plus this core's own
        // start. At stream position j every member is on Q block start_m + j, so each core derives the same
        // chain-wide kept set and the chain skips a piece nobody needs in lockstep. Non-participants (head-boundary
        // cores before the injector) get no members and skip on their own mask.
        std::vector<std::pair<uint32_t, uint32_t>> group_members;
        uint32_t sparse_my_start = 0;
        if (use_head_chain && head_chain_configs[i].participates) {
            const uint32_t h = head_chain_configs[i].head;
            for (const auto& seg : head_segments[h]) {
                const auto& cfg = head_chain_configs[seg.core_idx];
                if (!cfg.participates || cfg.head != h) {
                    continue;
                }
                const auto& w = core_work[seg.core_idx];
                uint32_t flat = w.global_q_start;
                for (uint32_t s = 0; s < seg.head_work_index; ++s) {
                    flat += w.head_work[s].q_chunk_count;
                }
                const uint32_t start_in_head = flat % num_res_q_chunks;
                const uint32_t count = w.head_work[seg.head_work_index].q_chunk_count;
                if (seg.core_idx == i) {
                    sparse_my_start = start_in_head;
                }
                group_members.emplace_back(start_in_head, count);
            }
            TT_FATAL(
                group_members.size() <= 64,
                "SparseRingSDPA: head chain of {} members is too long",
                group_members.size());
        }
        reader_args.push_back(static_cast<uint32_t>(group_members.size()));
        reader_args.push_back(sparse_my_start);
        if (balance) {
            // [core index, cost table, balance core NoC x, y, import Q, import mask].
            reader_args.push_back(i);
            reader_args.push_back(balance_table_buf);
            reader_args.push_back(static_cast<uint32_t>(balance_core_noc.x));
            reader_args.push_back(static_cast<uint32_t>(balance_core_noc.y));
            reader_args.push_back(balancing.remote ? balance_res->import_q->buffer() : nullptr);
            reader_args.push_back(balancing.remote ? balance_res->import_mask->buffer() : nullptr);
        }
        for (uint32_t w = 0; w < kPooledBiasTableWords; ++w) {
            reader_args.push_back(pooled_bias_table[w]);
        }
        for (const auto& [start, end] : args.dense_blocks) {
            reader_args.push_back(start);
            reader_args.push_back(end);
        }
        if (dense_h) {
            // [first unit, end unit, arrivals from -, dense-Q buffer, receive buffer, arrivals from +], the owner
            // records.
            reader_args.push_back(du_start);
            reader_args.push_back(du_end);
            reader_args.push_back(arrival_minus_addr);
            reader_args.push_back(dense_q_buf);
            reader_args.push_back(dense_recv_buf);
            reader_args.push_back(arrival_plus_addr);
            reader_args.append(dense_records);
        }
        for (const auto& [start, count] : group_members) {
            reader_args.push_back(start);
            reader_args.push_back(count);
        }
        if (group_mcast) {
            for (const uint32_t w : group_args[i]) {
                reader_args.push_back(w);
            }
            if (dense_h) {
                for (const uint32_t w : dense_group_args[i]) {
                    reader_args.push_back(w);
                }
            }
        }
        reader_kernel.emplace_runtime_args(core, reader_args.args);

        // Writer args
        CheckedRuntimeArgList writer_args;
        writer_args.push_back(out_buf);
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
        if (balancing.remote) {
            // [export buffer, balance core NoC x, y].
            writer_args.push_back(balance_res->export_out->buffer());
            writer_args.push_back(static_cast<uint32_t>(balance_core_noc.x));
            writer_args.push_back(static_cast<uint32_t>(balance_core_noc.y));
        }
        // dense_blocks ranges, then [first unit, end unit, partials buffer, ready semaphores + / -], the records.
        for (const auto& [start, end] : args.dense_blocks) {
            writer_args.push_back(start);
            writer_args.push_back(end);
        }
        if (dense_h) {
            writer_args.push_back(du_start);
            writer_args.push_back(du_end);
            writer_args.push_back(dense_partials_buf);
            writer_args.push_back(ready_plus_addr);
            writer_args.push_back(ready_minus_addr);
            writer_args.append(dense_records);
        }
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
            runtime_arg_layout.compute_q_pre_wrap_start_tile, kUnusedQMappingWord, "compute.q_pre_wrap_start_tile");
        compute_args.push_checked(
            runtime_arg_layout.compute_q_pre_wrap_tile_count, kUnusedQMappingWord, "compute.q_pre_wrap_tile_count");
        compute_args.push_checked(
            runtime_arg_layout.compute_q_post_wrap_start_tile, kUnusedQMappingWord, "compute.q_post_wrap_start_tile");
        compute_args.push_checked(
            runtime_arg_layout.compute_q_valid_tile_count, kUnusedQMappingWord, "compute.q_valid_tile_count");
        compute_args.push_checked(
            runtime_arg_layout.compute_active_ring_iter_mask, active_ring_iter_mask, "compute.active_ring_iter_mask");
        if (dense_h) {
            compute_args.push_back(du_start);
            compute_args.push_back(du_end);
            compute_args.append(dense_records);
        }
        TT_FATAL(
            reader_args.size <= tt::tt_metal::max_runtime_args,
            "SparseRingSDPA: core {} needs {} reader runtime args (at most {})",
            i,
            reader_args.size,
            tt::tt_metal::max_runtime_args);
        compute_kernel.emplace_runtime_args(core, compute_args.args);
    }

    // Push the SDPA kernels into desc before invoking the all-gather helper so the helper appends its own kernels
    // after these, at kFirstAllGatherKernelIndex.
    static_assert(kReaderKernelIndex == 0 && kWriterKernelIndex == 1 && kComputeKernelIndex == 2);
    desc.kernels.push_back(std::move(reader_kernel));
    desc.kernels.push_back(std::move(writer_kernel));
    desc.kernels.push_back(std::move(compute_kernel));

    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler> all_gather_fused_op_signaler =
        ttnn::experimental::ccl::AllGatherFusedOpSignaler();

    all_gather_fused_op_signaler->init_fused_op(
        sdpa_fused_op_signaler->fused_op_receiver_cores_noc,
        sdpa_fused_op_signaler->fused_op_receiver_signal_semaphores,
        sdpa_fused_op_signaler->fused_op_signaler_mode);

    std::vector<Tensor> all_gather_input_tensors = {input_tensor_k, input_tensor_v};
    std::vector<Tensor> all_gather_output_tensors = {gathered_input_tensor_k, gathered_input_tensor_v};
    // dense_blocks: the dense rows' Q rides the gather after K and V, one slab per shard (dense_q_slabs).
    std::vector<RingAttentionRankSlab> dense_q_rank_slabs;
    if (tensor_args.gathered_dense_q.has_value()) {
        uint32_t output_row = 0;
        for (const auto& slab :
             dense_q_slabs(args.dense_blocks, ring_size, mask_nb_pad, q_local_padded_Nt, block_tiles)) {
            dense_q_rank_slabs.push_back({.input_row = slab.row_start, .output_row = output_row, .rows = slab.rows});
            output_row += slab.rows;
        }
        all_gather_input_tensors.push_back(input_tensor_q);
        all_gather_output_tensors.push_back(*tensor_args.gathered_dense_q);
    }
    // Append the all-gather portion to `desc`. Buffer addresses are auto-patched on cache hits.
    // The gather writers forward the dense partials one hop after the gather (not with dense_partials_debug).
    std::optional<RingAttentionDenseReturn> dense_return;
    if (dense_res != nullptr) {
        RingAttentionDenseReturn dr;
        dr.staging = dense_res->staging->buffer();
        dr.recv = dense_res->recv->buffer();
        dr.ready_plus_addr = ready_plus_addr;
        dr.ready_minus_addr = ready_minus_addr;
        dr.arrival_minus_addr = arrival_minus_addr;
        dr.arrival_plus_addr = arrival_plus_addr;
        dr.n_cores = NH * dense_cores_per_head;  // the cores holding dense units, in flat order
        dr.cores_per_head = dense_cores_per_head;
        dr.n_dense_blocks = n_dense_blocks;
        for (uint32_t x = 0; x < grid_size.x; ++x) {
            dr.core_x.push_back(mesh_device->worker_core_from_logical_core({x, 0}).x);
        }
        for (uint32_t y = 0; y < grid_size.y; ++y) {
            dr.core_y.push_back(mesh_device->worker_core_from_logical_core({0, y}).y);
        }
        for (uint32_t i = 0; i < num_cores; ++i) {
            const CoreCoord v = mesh_device->worker_core_from_logical_core({i % grid_size.x, i / grid_size.x});
            TT_FATAL(
                v.x == dr.core_x[i % grid_size.x] && v.y == dr.core_y[i / grid_size.x],
                "SparseRingSDPA dense return: the SDPA grid's NoC coordinates are not a product of rows and columns");
        }
        dr.n_banks = mesh_device->allocator()->get_num_banks(tt::tt_metal::BufferType::DRAM);
        dr.n_units = tt::round_up(dense_units, dr.n_banks);
        dr.partial_tiles = partial_tiles;
        dr.partial_tile_bytes = tt::tile_size(out_df);
        for (uint32_t k = 0; k < owner_plans.size(); ++k) {
            const uint32_t role = dense_records[k * sparse_dense::kRecordWords + 7];
            const std::array<uint32_t, 3> send = {k, owner_plans[k].block_offset, owner_plans[k].n_blocks};
            if (role == sparse_dense::kRoleEmitPlus) {
                dr.sends_plus.push_back(send);
            } else if (role == sparse_dense::kRoleEmitMinus) {
                dr.sends_minus.push_back(send);
            }
        }
        dense_return = std::move(dr);
    }
    // Cross-device load balancing rides the gather writers; the helper adds their flag semaphore.
    RingAttentionBalance ring_balance;
    if (balancing.remote) {
        ring_balance.table = balance_res->table->buffer();
        ring_balance.partner = balance_res->partner->buffer();
        ring_balance.q = q_buf;
        ring_balance.import_q = balance_res->import_q->buffer();
        ring_balance.mask = tensor_args.block_mask.buffer();
        ring_balance.import_mask = balance_res->import_mask->buffer();
        ring_balance.export_out = balance_res->export_out->buffer();
        ring_balance.out = out_buf;
        ring_balance.tables_arrived_addr = static_cast<uint32_t>(balance_res->tables_arrived->address());
        ring_balance.imports_arrived_addr = static_cast<uint32_t>(balance_res->imports_arrived->address());
        ring_balance.returns_arrived_addr = static_cast<uint32_t>(balance_res->returns_arrived->address());
        ring_balance.balance_core_noc = balance_core_noc;
        ring_balance.heads = NH;
        ring_balance.q_blocks = num_local_q_chunks;
        ring_balance.q_row_tiles = q_local_padded_Nt;
        ring_balance.dht = DHt;
        ring_balance.vdht = vDHt;
        ring_balance.block_tiles = block_tiles;
        ring_balance.mask_rows = mask_nqb_pad;
        ring_balance.mask_row_bytes = mask_row_bytes;
        ring_balance.table_page_bytes =
            sparse_balance::table_page_words(balancing.cores, balancing.cost_words, balancing.q_blocks) *
            sizeof(uint32_t);
        ring_balance.q_page_bytes = q_buf->page_size();
        ring_balance.out_page_bytes = out_buf->page_size();
    }
    ring_attention_all_gather_async_multi_core_with_workers_helper(
        desc,
        all_gather_input_tensors,
        coord,
        forward_coord,
        backward_coord,
        all_gather_output_tensors,
        args.all_gather_operation_attributes.dim,
        args.all_gather_operation_attributes.num_links,
        args.all_gather_operation_attributes.ring_size,
        transport_rank,
        args.all_gather_operation_attributes.topology,
        args.all_gather_operation_attributes.semaphore,
        args.all_gather_operation_attributes.sub_device_id,
        all_gather_fused_op_signaler,
        args.ccl_core_grid_offset,
        args.all_gather_operation_attributes.core_allocation_strategy,
        /*input_batch_slice_idx=*/std::nullopt,
        /*gather_valid_Ht=*/std::nullopt,
        /*slot_id=*/std::nullopt,
        /*kv_actual_isl=*/std::nullopt,
        /*chunk_local_tiles=*/tensor_args.input_q.padded_shape()[2] / tt::constants::TILE_HEIGHT,
        /*kv_cache_num_layers=*/1,
        /*kv_cache_layer_idx=*/0,
        // Share the split-forwarding decision derived above so the all-gather only splits when this
        // consumer implements the second-half wait.
        sdpa_fused_op_signaler->split_forwarding_enabled,
        /*partial_readiness_enabled=*/false,
        rank_mapping,
        dense_q_rank_slabs,
        dense_return,
        balancing.remote ? &ring_balance : nullptr);

    // The balance kernel (after the gather's kernels, which supply the writers' flag), on its own core with its
    // scratch: cost pages, row-block costs, the plan (sparse_balance::Scratch).
    if (balance) {
        const CoreRangeSet balance_core_set(CoreRange(balancing.core));
        KernelDescriptor balance_kernel{};
        balance_kernel.kernel_source =
            "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/sparse_ring_balance.cpp";
        balance_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
        balance_kernel.core_ranges = balance_core_set;
        balance_kernel.config = ReaderConfigDescriptor{};
        std::vector<uint32_t> balance_compile_args;
        TensorAccessorArgs(balance_res->table->buffer()).append_to(balance_compile_args);
        TensorAccessorArgs(balancing.remote ? balance_res->partner->buffer() : nullptr).append_to(balance_compile_args);
        balance_kernel.compile_time_args = balance_compile_args;
        balance_kernel.named_compile_time_args = {
            {"balance_units", balancing.units},
            {"balance_cores", balancing.cores},
            {"balance_q_blocks", balancing.q_blocks},
            {"num_q_passes", num_q_passes},
            {"balance_cap", balance_cap},
            {"balance_cost_words", balancing.cost_words},
            {"balance_remote", balancing.remote ? 1u : 0u},
            {"balance_devices", balancing.devices},
            {"balance_swappable", balancing.swappable},
            {"balance_grid_x", static_cast<uint32_t>(grid_size.x)},
            {"balance_sem_costs", balance_sems.costs},
            {"balance_sem_outputs", balance_sems.outputs},
            {"balance_sem_table", balance_sems.table},
            {"balance_sem_import", balance_sems.import},
            {"cb_balance", cbs.balance},
            {"balance_writer_flag", ring_balance.writer_flag_semaphore},
            {"balance_cb", kBalanceScratchCb},
        };
        // [cost table, partner page, this device's rank, tables / imports / returns arrived, writers], the SDPA grid's
        // column NoC x and row NoC y, the writers' NoC (x, y).
        KernelDescriptor::RTArgList balance_args;
        balance_args.push_back(balance_res->table->buffer());
        balance_args.push_back(balancing.remote ? balance_res->partner->buffer() : nullptr);
        balance_args.push_back(transport_rank);
        balance_args.push_back(ring_balance.tables_arrived_addr);
        balance_args.push_back(ring_balance.imports_arrived_addr);
        balance_args.push_back(ring_balance.returns_arrived_addr);
        balance_args.push_back(static_cast<uint32_t>(ring_balance.writer_cores_noc.size()));
        for (uint32_t x = 0; x < grid_size.x; ++x) {
            balance_args.push_back(static_cast<uint32_t>(mesh_device->worker_core_from_logical_core({x, 0}).x));
        }
        for (uint32_t y = 0; y < grid_size.y; ++y) {
            balance_args.push_back(static_cast<uint32_t>(mesh_device->worker_core_from_logical_core({0, y}).y));
        }
        for (uint32_t i = 0; i < num_cores; ++i) {
            const CoreCoord v = mesh_device->worker_core_from_logical_core({i % grid_size.x, i / grid_size.x});
            TT_FATAL(
                v.x == mesh_device->worker_core_from_logical_core({i % grid_size.x, 0}).x &&
                    v.y == mesh_device->worker_core_from_logical_core({0, i / grid_size.x}).y,
                "SparseRingSDPA load balancing: the SDPA grid's NoC coordinates are not a product of rows and columns");
        }
        for (const CoreCoord& w : ring_balance.writer_cores_noc) {
            balance_args.push_back(static_cast<uint32_t>(w.x));
            balance_args.push_back(static_cast<uint32_t>(w.y));
        }
        balance_kernel.emplace_runtime_args(balancing.core, balance_args);
        desc.kernels.push_back(std::move(balance_kernel));
    }

    return desc;
}

}  // namespace

// One ProgramDescriptor per coord: transport rank / forward_coord / backward_coord (used by the all-gather portion)
// depend on the mesh coordinate, so descriptors cannot be shared across coords. Returning a WorkloadDescriptor keeps
// the framework on its no-rebuild cache-hit fast path.
tt::tt_metal::WorkloadDescriptor SparseRingSDPAProgramFactory::create_workload_descriptor(
    const SparseRingSDPAParams& args,
    const SparseRingSDPAInputs& tensor_args,
    SparseRingSDPAResult& output_tensors,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor wd;
    const auto coords = tensor_coords.coords();
    wd.programs.reserve(coords.size());
    // The dense hop path's buffers and semaphores, owned by the cached workload. Made before the per-coord builds so
    // their L1 counts in the CB fit.
    std::optional<DenseResources> dense_res;
    if (tensor_args.gathered_dense_q.has_value() && !tensor_args.dense_partials_debug.has_value()) {
        auto* mesh = tensor_args.input_q.device();
        uint32_t n_units = 0;
        dense_owner_plans(args, tensor_args, n_units);
        const uint32_t sq_t = args.block_tiles();
        const uint32_t vdh_t = tensor_args.input_v.logical_shape()[3] / tt::constants::TILE_WIDTH;
        // Units padded to whole banks (sparse_dense::banked_page).
        const uint32_t n_banks = mesh->allocator()->get_num_banks(BufferType::DRAM);
        const uint32_t pages = tt::round_up(n_units, n_banks) * (sq_t * vdh_t + 2 * sq_t);
        const auto make_buffer = [&](uint32_t n_pages) {
            const tt::tt_metal::TensorSpec spec(
                ttnn::Shape({1, 1, n_pages * tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH}),
                tt::tt_metal::TensorLayout(
                    DataType::BFLOAT16,
                    tt::tt_metal::PageConfig(Layout::TILE),
                    tt::tt_metal::MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}));
            return std::make_shared<Tensor>(create_device_tensor(spec, mesh));
        };
        const CoreCoord grid = args.program_config.compute_with_storage_grid_size;
        const CoreRangeSet sdpa_cores(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
        const auto make_sem = [&] {
            return ttnn::global_semaphore::create_global_semaphore(mesh, sdpa_cores, 0, BufferType::L1);
        };
        dense_res =
            DenseResources{make_buffer(pages), make_buffer(2 * pages), make_sem(), make_sem(), make_sem(), make_sem()};
        tt::tt_metal::distributed::Synchronize(*mesh, std::nullopt, {});
        for (const auto& t : {dense_res->staging, dense_res->recv}) {
            wd.buffers.push_back({t, t->buffer()});
        }
        for (const auto* sem :
             {&dense_res->ready_plus, &dense_res->ready_minus, &dense_res->arrival_minus, &dense_res->arrival_plus}) {
            wd.semaphores.push_back(*sem);
        }
    }
    // Load balancing's buffers and counters (BalanceResources), likewise owned by the cached workload.
    std::optional<BalanceResources> balance_res;
    const BalanceConfig balancing = make_balance_config(args, tensor_args);
    if (balancing.on) {
        auto* mesh = tensor_args.input_q.device();
        const auto dram = tt::tt_metal::MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
        const auto words = [&](uint32_t rows, uint32_t width) {
            return std::make_shared<Tensor>(create_device_tensor(
                tt::tt_metal::TensorSpec(
                    ttnn::Shape({1, 1, rows, width}),
                    tt::tt_metal::TensorLayout(DataType::UINT32, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), dram)),
                mesh));
        };
        balance_res.emplace();
        // One page per device (indexed by rank) even when only this device's is read.
        balance_res->table = words(
            static_cast<uint32_t>(args.ring_size),
            sparse_balance::table_page_words(balancing.cores, balancing.cost_words, balancing.q_blocks));
        if (balancing.remote) {
            balance_res->partner = words(1, tt::round_up(balancing.q_blocks, 16u));
            balance_res->import_q =
                std::make_shared<Tensor>(create_device_tensor(tensor_args.input_q.tensor_spec(), mesh));
            balance_res->import_mask =
                std::make_shared<Tensor>(create_device_tensor(tensor_args.block_mask.tensor_spec(), mesh));
            balance_res->export_out =
                std::make_shared<Tensor>(create_device_tensor(output_tensors.at(0).tensor_spec(), mesh));
            const CoreRangeSet balance_core_set(CoreRange(balancing.core));
            const auto make_sem = [&] {
                return ttnn::global_semaphore::create_global_semaphore(mesh, balance_core_set, 0, BufferType::L1);
            };
            balance_res->tables_arrived = make_sem();
            balance_res->imports_arrived = make_sem();
            balance_res->returns_arrived = make_sem();
            tt::tt_metal::distributed::Synchronize(*mesh, std::nullopt, {});
            for (const auto* sem :
                 {&*balance_res->tables_arrived, &*balance_res->imports_arrived, &*balance_res->returns_arrived}) {
                wd.semaphores.push_back(*sem);
            }
        }
        for (const auto& t :
             {balance_res->table,
              balance_res->partner,
              balance_res->import_q,
              balance_res->import_mask,
              balance_res->export_out}) {
            if (t != nullptr) {
                wd.buffers.push_back({t, t->buffer()});
            }
        }
    }
    for (const auto& coord : coords) {
        auto desc = build_sparse_ring_sdpa_program_descriptor(
            args,
            tensor_args,
            output_tensors,
            coord,
            dense_res.has_value() ? &*dense_res : nullptr,
            balance_res.has_value() ? &*balance_res : nullptr);
        wd.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
    }
    return wd;
}

SparseRingSDPAMeshWorkloadFactory::cached_mesh_workload_t SparseRingSDPAMeshWorkloadFactory::create_mesh_workload(
    const SparseRingSDPAParams& args,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const SparseRingSDPAInputs& tensor_args,
    SparseRingSDPAResult& output_tensors) {
    return descriptor_adapter_t::create_mesh_workload(args, tensor_coords, tensor_args, output_tensors);
}

void SparseRingSDPAMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const SparseRingSDPAParams& args,
    const SparseRingSDPAInputs& tensor_args,
    SparseRingSDPAResult& output_tensors) {
    descriptor_adapter_t::apply_descriptor(cached_workload, args, tensor_args, output_tensors);

    // The fused all-gather's GlobalSemaphore addresses are excluded from the program hash and baked into the
    // all-gather reader/writer runtime args at creation, so re-apply them on every cache hit (the standalone
    // all-gather op does the same); otherwise a cached program keeps signalling the ping-pong semaphore it was
    // created with.
    const auto& ag_semaphores = args.all_gather_operation_attributes.semaphore;
    TT_FATAL(
        ag_semaphores.size() > std::max(ag_dyn::kForwardSemaphoreIdx, ag_dyn::kBackwardSemaphoreIdx),
        "SparseRingSDPA needs {} all-gather semaphores (forward and backward), got {}",
        std::max(ag_dyn::kForwardSemaphoreIdx, ag_dyn::kBackwardSemaphoreIdx) + 1,
        ag_semaphores.size());
    const auto forward_sem_addr = static_cast<uint32_t>(ag_semaphores.at(ag_dyn::kForwardSemaphoreIdx).address());
    const auto backward_sem_addr = static_cast<uint32_t>(ag_semaphores.at(ag_dyn::kBackwardSemaphoreIdx).address());

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        TT_FATAL(
            coordinate_range.start_coord() == coordinate_range.end_coord(),
            "Expected SparseRingSDPA cached programs to cover a single coordinate, got range {} to {}",
            coordinate_range.start_coord(),
            coordinate_range.end_coord());
        // The all-gather always appends its four kernels at kFirstAllGatherKernelIndex.
        const auto patch_semaphore_arg = [&program](uint32_t kernel_idx, uint32_t arg_idx, uint32_t address) {
            auto& args_by_core = tt::tt_metal::GetRuntimeArgs(program, kernel_idx);
            for (auto& args_by_y : args_by_core) {
                for (auto& core_args : args_by_y) {
                    if (arg_idx < core_args.size()) {
                        core_args[arg_idx] = address;
                    }
                }
            }
        };
        patch_semaphore_arg(kAllGatherReaderForwardKernelIndex, ag_dyn::kReaderSemaphoreArg, forward_sem_addr);
        patch_semaphore_arg(kAllGatherWriterForwardKernelIndex, ag_dyn::kWriterSemaphoreArg, forward_sem_addr);
        patch_semaphore_arg(kAllGatherReaderBackwardKernelIndex, ag_dyn::kReaderSemaphoreArg, backward_sem_addr);
        patch_semaphore_arg(kAllGatherWriterBackwardKernelIndex, ag_dyn::kWriterSemaphoreArg, backward_sem_addr);
    }
}

}  // namespace ttnn::prim
