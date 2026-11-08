// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sparse_dispatch_2d_device_operation.hpp"

#include <set>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt_stl/assert.hpp>

#include "sparse_dispatch_2d_plan.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::operations::experimental::moe_block::sparse_dispatch_2d {

namespace sparse_dispatch_2d_detail {

void validate_dram_row_major(const Tensor& t, const char* name) {
    TT_FATAL(
        t.storage_type() == StorageType::DEVICE && t.is_allocated(), "sparse_dispatch_2d: {} must be on device", name);
    TT_FATAL(t.layout() == Layout::ROW_MAJOR, "sparse_dispatch_2d: {} must be ROW_MAJOR, got {}", name, t.layout());
    TT_FATAL(
        t.memory_config().buffer_type() == BufferType::DRAM &&
            t.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "sparse_dispatch_2d: {} must be DRAM interleaved (a sender addresses it by page on another chip)",
        name);
}

}  // namespace sparse_dispatch_2d_detail

ttsl::hash::hash_t SparseDispatch2dDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& t) {
    return tt::tt_metal::operation::hash_operation<SparseDispatch2dDeviceOperation>(
        args.plan_key,
        args.group,
        args.manifest_batch,
        args.pack,
        t.xbuf.tensor_spec(),
        t.route.tensor_spec(),
        t.sems.tensor_spec(),
        t.arena.has_value() ? std::optional<tt::tt_metal::TensorSpec>(t.arena->tensor_spec()) : std::nullopt);
}

void SparseDispatch2dDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& t) {
    using namespace sparse_dispatch_2d_detail;
    TT_FATAL(args.ports != nullptr, "sparse_dispatch_2d: no port plan (call through ttnn.experimental.moe_block)");
    const auto& topo = args.ports->plan.topo;
    const uint32_t N = topo.num_chips();
    TT_FATAL(N > 1 && N <= 32, "sparse_dispatch_2d: {} chips (2..32 supported)", N);
    TT_FATAL(
        gmax(topo) <= 15,
        "sparse_dispatch_2d: END generations are 4 bits; a {}x{} mesh needs {}",
        topo.R,
        topo.C,
        gmax(topo));
    auto* mesh = t.xbuf.device();
    TT_FATAL(
        t.route.device() == mesh && t.sems.device() == mesh, "sparse_dispatch_2d: every tensor must be on one mesh");
    TT_FATAL(
        mesh->shape().dims() == 2 && mesh->shape()[0] == topo.R && mesh->shape()[1] == topo.C,
        "sparse_dispatch_2d: the mesh is {}, the resources' plan {}x{}",
        mesh->shape(),
        topo.R,
        topo.C);

    validate_dram_row_major(t.xbuf, "xbuf");
    validate_dram_row_major(t.route, "route");
    TT_FATAL(t.route.dtype() == DataType::UINT32, "sparse_dispatch_2d: route must be UINT32, got {}", t.route.dtype());
    const uint32_t RW = route_words(N);
    const uint32_t route_len = t.route.logical_shape()[-1];
    TT_FATAL(
        t.route.logical_volume() == route_len && route_len % RW == 0 && route_len > 0,
        "sparse_dispatch_2d: route must be [1, 1, 1, {} * S], got {}",
        RW,
        t.route.logical_shape());
    const uint32_t S = route_len / RW;
    const auto& xs = t.xbuf.logical_shape();
    TT_FATAL(
        xs.rank() >= 2 && xs[-2] == N * S && t.xbuf.logical_volume() == static_cast<uint64_t>(N) * S * xs[-1],
        "sparse_dispatch_2d: xbuf must be [1, 1, N * S = {}, W], got {}",
        N * S,
        xs);
    TT_FATAL(N * S < (1u << 16), "sparse_dispatch_2d: N * S = {} slots; manifest entries hold 16-bit slots", N * S);

    const uint32_t row_bytes = t.xbuf.buffer()->aligned_page_size();
    const size_t payload = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
    TT_FATAL(args.pack == 1 || args.pack == 2, "sparse_dispatch_2d: pack must be 1 or 2, got {}", args.pack);
    TT_FATAL(
        args.pack * row_bytes <= payload,
        "sparse_dispatch_2d: {} row(s) of {} B a packet exceed the fabric payload {} B",
        args.pack,
        row_bytes,
        payload);
    TT_FATAL(args.group >= 1 && args.manifest_batch >= 1, "sparse_dispatch_2d: group and manifest_batch must be >= 1");
    TT_FATAL(
        args.manifest_batch * 16 <= payload,
        "sparse_dispatch_2d: a manifest batch of {} entries exceeds the fabric payload {} B",
        args.manifest_batch,
        payload);

    const auto& sm = t.sems.memory_config();
    TT_FATAL(
        t.sems.dtype() == DataType::UINT32 && sm.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED &&
            (sm.buffer_type() == BufferType::L1 || sm.buffer_type() == BufferType::L1_SMALL) &&
            sm.shard_spec().has_value(),
        "sparse_dispatch_2d: sems must be UINT32 words HEIGHT_SHARDED in L1 / L1_SMALL");
    TT_FATAL(sm.shard_spec()->shape[1] >= SEM_WORDS, "sparse_dispatch_2d: sems needs {} words a port core", SEM_WORDS);
    for (const auto& [key, core] : args.ports->cores) {
        TT_FATAL(
            sm.shard_spec()->grid.contains(core),
            "sparse_dispatch_2d: port core {} has no semaphore words",
            core.str());
    }
    if (t.arena.has_value()) {
        const auto& am = t.arena->memory_config();
        TT_FATAL(
            t.arena->device() == mesh && am.buffer_type() == BufferType::L1 &&
                am.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED && am.shard_spec().has_value(),
            "sparse_dispatch_2d: the arena must be an L1 HEIGHT_SHARDED tensor on the same mesh");
        for (const auto& [key, core] : args.ports->cores) {
            TT_FATAL(
                am.shard_spec()->grid.contains(core),
                "sparse_dispatch_2d: port core {} is outside the arena",
                core.str());
        }
    }
}

void SparseDispatch2dDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*args*/, const tensor_args_t& t) {
    TT_FATAL(t.xbuf.is_allocated() && t.route.is_allocated(), "sparse_dispatch_2d: xbuf and route must be allocated");
    TT_FATAL(!t.arena.has_value() || t.arena->is_allocated(), "sparse_dispatch_2d: the arena must be allocated");
}

SparseDispatch2dDeviceOperation::spec_return_value_t SparseDispatch2dDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& t) {
    return t.xbuf.tensor_spec();
}

SparseDispatch2dDeviceOperation::topology_return_value_t SparseDispatch2dDeviceOperation::compute_output_topologies(
    const operation_attributes_t&, const tensor_args_t& t) {
    return {t.xbuf.tensor_topology()};
}

SparseDispatch2dDeviceOperation::tensor_return_value_t SparseDispatch2dDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& t) {
    return t.xbuf;  // (in place)
}

}  // namespace ttnn::operations::experimental::moe_block::sparse_dispatch_2d

namespace ttnn::prim {

Tensor moe_block_sparse_dispatch_2d(
    const Tensor& xbuf,
    const Tensor& route,
    const Tensor& sems,
    const ttnn::operations::experimental::moe_block::sparse_dispatch_2d::SparseDispatch2dParams& params,
    const std::optional<Tensor>& arena) {
    using namespace ttnn::operations::experimental::moe_block::sparse_dispatch_2d;
    return ttnn::device_operation::launch<SparseDispatch2dDeviceOperation>(
        params, SparseDispatch2dInputs{.xbuf = xbuf, .route = route, .sems = sems, .arena = arena});
}

}  // namespace ttnn::prim
