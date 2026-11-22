// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "kv_fabric_push_device_operation.hpp"

#include <algorithm>
#include <vector>

#include <tt-metalium/mesh_device.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

namespace {

// The cached program binds each run-time buffer address to the source slot that held it, so a buffer given twice
// would leave a cache entry patching two addresses from one slot.
void check_sources(const KvFabricPushParams& args, const KvFabricPushInputs& tensor_args) {
    const auto& sources = tensor_args.sources;
    TT_FATAL(!sources.empty(), "kv_fabric_push: no sources");
    TT_FATAL(sources.size() <= 255, "kv_fabric_push: {} sources, a run names at most 255", sources.size());
    TT_FATAL(args.tilize.size() == sources.size(), "kv_fabric_push: a tilize flag per source");
    std::vector<const Buffer*> seen;
    for (size_t s = 0; s < sources.size(); ++s) {
        const auto& t = sources[s];
        TT_FATAL(
            t.storage_type() == StorageType::DEVICE && t.is_allocated(), "kv_fabric_push: source {} not on device", s);
        TT_FATAL(t.device() == sources[0].device(), "kv_fabric_push: source {} on another mesh", s);
        const Buffer* b = t.buffer();
        TT_FATAL(b->is_dram(), "kv_fabric_push: source {} is not in DRAM", s);
        TT_FATAL(std::find(seen.begin(), seen.end(), b) == seen.end(), "kv_fabric_push: source {} given twice", s);
        seen.push_back(b);
        if (args.tilize[s]) {
            TT_FATAL(
                t.layout() == Layout::ROW_MAJOR && t.dtype() == DataType::BFLOAT16 && t.logical_shape()[-1] % 32 == 0,
                "kv_fabric_push: tilized source {} must be bf16 ROW_MAJOR rows of a multiple of 32 values",
                s);
        }
    }
}

}  // namespace

void KvFabricPushOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    check_sources(args, tensor_args);
    auto* mesh = tensor_args.sources[0].device();
    const auto shape = mesh->shape();
    TT_FATAL(
        shape.dims() == 2 && shape[1] == args.mesh_num_cols,
        "kv_fabric_push: a 2D mesh of {} columns",
        args.mesh_num_cols);
    TT_FATAL(
        args.socket_config_addrs.size() == shape.mesh_size(),
        "kv_fabric_push: a socket per chip ({} for {} chips)",
        args.socket_config_addrs.size(),
        shape.mesh_size());
    TT_FATAL(!args.dst_chips.empty() && args.dst_banks > 0, "kv_fabric_push: no destination chips");
    TT_FATAL(
        args.socket_page_size % 16 == 0 && args.max_dst_bytes % args.socket_page_size == 0,
        "kv_fabric_push: socket pages of whole runs, the destination table of whole pages");
}

void KvFabricPushOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    check_sources(args, tensor_args);
}

ttsl::hash::hash_t KvFabricPushOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    std::vector<TensorSpec> specs;
    specs.reserve(tensor_args.sources.size());
    for (const auto& t : tensor_args.sources) {
        specs.push_back(t.tensor_spec());
    }
    return tt::tt_metal::operation::hash_operation<KvFabricPushOperation>(
        args.dst_mesh_id,
        args.dst_chips,
        args.dst_banks,
        args.max_dst_bytes,
        args.socket_page_size,
        args.mesh_num_cols,
        args.socket_config_addrs,
        args.tilize,
        specs);
}

KvFabricPushOperation::spec_return_value_t KvFabricPushOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    return tensor_args.sources[0].tensor_spec();
}

KvFabricPushOperation::topology_return_value_t KvFabricPushOperation::compute_output_topologies(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    return {tensor_args.sources[0].tensor_topology()};
}

KvFabricPushOperation::tensor_return_value_t KvFabricPushOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    return tensor_args.sources[0];
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

ttnn::Tensor kv_fabric_push(
    const ttnn::experimental::prim::KvFabricPushParams& params, const std::vector<ttnn::Tensor>& sources) {
    using OperationType = ttnn::experimental::prim::KvFabricPushOperation;
    return ttnn::device_operation::launch<OperationType>(params, OperationType::tensor_args_t{sources});
}

}  // namespace ttnn::prim
