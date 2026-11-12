// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/transformer/sdpa/device/sparse_ring_sdpa_device_operation_types.hpp"
#include "ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/ring_attention_all_gather_async_multi_core_with_workers_program_factory.hpp"

namespace ttnn::prim {

namespace detail {

struct SparseRingSDPADescriptorAdapterOperation {
    using operation_attributes_t = SparseRingSDPAParams;
    using tensor_args_t = SparseRingSDPAInputs;
    using spec_return_value_t = SparseRingSDPAResultSpec;
    using tensor_return_value_t = SparseRingSDPAResult;
};

}  // namespace detail

inline constexpr const char* kSparseRingQPermPreset = "stride:7";

inline uint32_t sparse_ring_fast_pbias() {
    const char* s = std::getenv("TT_SPARSE_RING_FAST_PBIAS");
    const int v = s == nullptr ? 0 : std::atoi(s);
    return v <= 0 ? 0u : (v >= 2 ? 2u : 1u);
}

inline std::string sparse_ring_kv_pages_request() {
    const char* s = std::getenv("TT_SPARSE_RING_KV_PAGES");
    return s == nullptr ? std::string() : std::string(s);
}

// TT_SPARSE_RING_MEMBER_RESCALE=1: the lazy-rescale test runs per resident Q block. A step rescales only the blocks whose
// own row max grew past the threshold; the others keep their stored max (as every block does on a step that does not
// rescale). Changes the order of the online-softmax updates, not the mask or the routed set. Default off.
inline uint32_t sparse_ring_member_rescale() {
    const char* s = std::getenv("TT_SPARSE_RING_MEMBER_RESCALE");
    return (s != nullptr && std::atoi(s) > 0) ? 1u : 0u;
}

// TT_SPARSE_RING_DYN_EXTRA_PASS=1 (with TT_SPARSE_RING_DYN_SPLIT=on): the dynamic split's per-core cap
// (ceil(N / floor(C / U)) Q blocks) may need one pass more than the even split; accept that plan instead of falling
// back to the even split. The later pass replays the gathered K/V from DRAM (the ring gather runs once), and halving
// the slots per pass leaves L1 for deeper K/V buffers. No effect when the cap fits the even split's passes. Default off.
inline uint32_t sparse_ring_dyn_extra_pass() {
    const char* s = std::getenv("TT_SPARSE_RING_DYN_EXTRA_PASS");
    return (s != nullptr && std::atoi(s) > 0) ? 1u : 0u;
}

// TT_SPARSE_RING_MULTIPASS_PAIRS=1: keep pair steps (two K/V pieces per compute step) when the plan needs several Q
// passes. The compute kernel otherwise takes single steps there, because the pair path's text (~1.5 KB per TRISC) does
// not fit the kernel config buffer next to the multi-pass reader; with this switch the reader is built at -Os (about
// 3 KB smaller) to make room. One-pass plans are unchanged. Default off.
inline uint32_t sparse_ring_multipass_pairs() {
    const char* s = std::getenv("TT_SPARSE_RING_MULTIPASS_PAIRS");
    return (s != nullptr && std::atoi(s) > 0) ? 1u : 0u;
}

inline uint32_t sparse_ring_dyn_split() {
    const char* s = std::getenv("TT_SPARSE_RING_DYN_SPLIT");
    if (s == nullptr) {
        return 0u;
    }
    const std::string v(s);
    return v == "on" ? 2u : 0u;
}

inline std::string sparse_ring_q_perm_request() {
    const char* s = std::getenv("TT_SPARSE_RING_Q_PERM");
    return s == nullptr ? std::string() : std::string(s);
}

inline std::string sparse_ring_q_perm_resolve(const std::string& request) {
    return (request == "1" || request == "on" || request == "preset") ? std::string(kSparseRingQPermPreset) : request;
}

std::vector<uint32_t> sparse_ring_q_perm_table(
    const std::string& request, uint32_t num_q_chunks, uint32_t slots_per_pass);

std::vector<uint32_t> sparse_ring_q_split_aligned(
    uint32_t units, uint32_t q_per_unit, uint32_t num_cores, std::string& why);

struct SparseRingSDPAProgramFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const SparseRingSDPAParams& args,
        const SparseRingSDPAInputs& tensor_args,
        SparseRingSDPAResult& output_tensors,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

struct SparseRingSDPAMeshWorkloadFactory {
    using descriptor_adapter_t = ttnn::device_operation::MeshDeviceOperationAdapter<
        detail::SparseRingSDPADescriptorAdapterOperation>::DescriptorMeshWorkloadAdapter<SparseRingSDPAProgramFactory>;
    using cached_mesh_workload_t = typename descriptor_adapter_t::cached_mesh_workload_t;

    static cached_mesh_workload_t create_mesh_workload(
        const SparseRingSDPAParams& args,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const SparseRingSDPAInputs& tensor_args,
        SparseRingSDPAResult& output_tensors);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const SparseRingSDPAParams& args,
        const SparseRingSDPAInputs& tensor_args,
        SparseRingSDPAResult& output_tensors);
};

static_assert(ttnn::device_operation::MeshWorkloadFactoryConcept<SparseRingSDPAMeshWorkloadFactory>);

}  // namespace ttnn::prim
