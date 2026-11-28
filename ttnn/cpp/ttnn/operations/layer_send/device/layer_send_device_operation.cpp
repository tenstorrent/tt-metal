// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "layer_send_device_operation.hpp"

#include "ttnn/operation.hpp"

namespace ttnn::operations::layer_send {

using namespace tt::tt_metal;

namespace {

// io tensors and values a kind takes (layer_send.hpp)
void check_counts(const SendPlan& p, size_t n_io, size_t n_values) {
    const auto& k = p.consts;
    switch (p.kind) {
        case SendKind::WINDOW: TT_FATAL(n_io == 2 && n_values == 0, "layer_send window: io [region, tail]"); break;
        case SendKind::WINDOW_FUSED:
            TT_FATAL(n_io == 4 && n_values == 3 * k.at(9), "layer_send window fused: io [region, tail, rows, ring]");
            break;
        case SendKind::SNAP_STATIC:
            TT_FATAL(n_io == 2 && n_values == 5, "layer_send snapshot: io [ring, slab], values [page0, 4 ring rows]");
            break;
        case SendKind::SNAP_DYN:
            TT_FATAL(
                n_io == 2 && n_values == 3, "layer_send snapshot dyn: io [ring, slab], values [page0, rows, codes]");
            break;
        case SendKind::SNAP_MULTI:
            TT_FATAL(
                n_io >= 2 && n_io % 2 == 0 && n_io / 2 <= p.sems.size() / 4 && n_values == 3 * (n_io / 2),
                "layer_send snapshot multi: io [rings, slabs], values (page0, rows, codes) a segment");
            break;
        case SendKind::GLUED:
            TT_FATAL(
                n_io == 2 + (2 * k.at(7)) && n_values == k.at(7) * (4 + (3 * k.at(9))),
                "layer_send glued: io [region, rows, tails, rings], values (carries, ring pieces)");
            break;
    }
}

}  // namespace

LayerSendDeviceOperation::program_factory_t LayerSendDeviceOperation::select_program_factory(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {
    return LayerSendMeshWorkloadFactory{};
}

void LayerSendDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    TT_FATAL(attrs.plan != nullptr, "layer_send: no plan");
    const SendPlan& p = *attrs.plan;
    const auto& io = tensor_args.io;
    check_counts(p, io.size(), attrs.values.size());
    auto* mesh = io.at(0).device();
    TT_FATAL(mesh != nullptr, "layer_send: io must be on a device");
    TT_FATAL(
        mesh->shape().dims() == 2 && mesh->shape()[0] == p.rows && mesh->shape()[1] == p.cols,
        "layer_send: the plan was made for a {}x{} mesh",
        p.rows,
        p.cols);
    TT_FATAL(p.hops.size() == p.rows * p.cols, "layer_send: a hop list a chip");
    for (const auto& t : io) {
        TT_FATAL(t.device() == mesh, "layer_send: every io tensor on one mesh");
        TT_FATAL(t.layout() == Layout::ROW_MAJOR, "layer_send: ROW_MAJOR io (one row a page)");
    }
}

void LayerSendDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    check_counts(*attrs.plan, tensor_args.io.size(), attrs.values.size());
}

ttsl::hash::hash_t LayerSendDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    // The plan (routes, constants, semaphores, pattern) and what each io tensor's accessor compile-time args hold:
    // its buffer type and memory layout, and for a sharded one its memory config and padded shape (FIX.32). The
    // fused window send's q_rows (the rows' height) is the plan's (validated). Never an address or a value.
    auto h = static_cast<ttsl::hash::hash_t>(attrs.plan->hash);
    h = ttsl::hash::hash_objects(h, tensor_args.io.size());
    for (const auto& t : tensor_args.io) {
        const auto& mc = t.memory_config();
        h = ttsl::hash::hash_objects(h, mc.buffer_type(), mc.memory_layout());
        if (mc.is_sharded()) {
            h = ttsl::hash::hash_objects(h, mc, t.padded_shape());
        }
    }
    return h;
}

LayerSendDeviceOperation::spec_return_value_t LayerSendDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    return tensor_args.io.at(0).tensor_spec();
}

LayerSendDeviceOperation::tensor_return_value_t LayerSendDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& tensor_args) {
    return tensor_args.io.at(0);
}

}  // namespace ttnn::operations::layer_send

namespace ttnn::prim {

Tensor layer_send(
    const std::vector<Tensor>& io,
    const std::shared_ptr<const ttnn::operations::layer_send::SendPlan>& plan,
    const std::vector<uint32_t>& values) {
    using OperationType = ttnn::operations::layer_send::LayerSendDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{.plan = plan, .values = values}, OperationType::tensor_args_t{.io = io});
}

}  // namespace ttnn::prim
