// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sparse_sdpa_lanes_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include <tt-metalium/mesh_device.hpp>

namespace ttnn::prim::bringup {

namespace {

SparseSDPAInputs base_inputs(const SparseSDPALanesInputs& t, const Tensor& kv) {
    return SparseSDPAInputs{
        .q = t.q, .kv = kv, .indices = t.indices, .attention_sink = t.attention_sink, .kv_window = t.kv_window};
}

const Tensor& lane_kv(
    const SparseSDPALanesParams& a, const SparseSDPALanesInputs& t, const std::optional<ttnn::MeshCoordinate>& coord) {
    TT_FATAL(coord.has_value(), "sparse_sdpa kv_lanes: each device's program needs its mesh coordinate");
    const auto shape = t.q.device()->shape();
    TT_FATAL(
        coord->dims() == shape.dims(), "sparse_sdpa kv_lanes: coordinate rank {} vs mesh {}", coord->dims(), shape);
    uint32_t chip = 0;
    for (size_t d = 0; d < coord->dims(); ++d) {
        chip = chip * shape[d] + (*coord)[d];
    }
    TT_FATAL(
        chip < a.lane_of_chip.size(),
        "sparse_sdpa kv_lanes: chip {} has no lane ({} given)",
        chip,
        a.lane_of_chip.size());
    const uint32_t lane = a.lane_of_chip[chip];
    TT_FATAL(
        lane < t.kv_lanes.size(), "sparse_sdpa kv_lanes: chip {} lane {} of {} tensors", chip, lane, t.kv_lanes.size());
    return t.kv_lanes[lane];
}

void validate_lanes(const SparseSDPALanesParams& a, const SparseSDPALanesInputs& t) {
    TT_FATAL(
        !t.kv_lanes.empty() && t.kv_lanes.size() <= 8, "sparse_sdpa kv_lanes: 1 to 8 tensors ({})", t.kv_lanes.size());
    TT_FATAL(
        a.lane_of_chip.size() == t.q.device()->shape().mesh_size(),
        "sparse_sdpa kv_lanes: lane_of_chip has {} entries for {} devices",
        a.lane_of_chip.size(),
        t.q.device()->shape().mesh_size());
    const auto& k0 = t.kv_lanes.front();
    // (an interleaved, non-block-cyclic kv may differ in rows T: T is not in the source program's key and each device's
    // program is built and patched with its own lane's tensor; anything else must match exactly)
    const bool rows_free = !k0.memory_config().is_sharded() && !a.base.has_block_cyclic();
    for (const auto& k : t.kv_lanes) {
        const auto s = k.logical_shape(), s0 = k0.logical_shape();
        const bool same = rows_free ? (s.rank() == s0.rank() && s[0] == s0[0] && s[1] == s0[1] && s[-1] == s0[-1])
                                    : (s == s0 && k.padded_shape() == k0.padded_shape());
        TT_FATAL(
            same && k.dtype() == k0.dtype() && k.layout() == k0.layout() && k.memory_config() == k0.memory_config(),
            "sparse_sdpa kv_lanes: every lane tensor must have kv_lanes[0]'s shape (rows may differ when interleaved), "
            "dtype, layout and memory config");
        TT_FATAL(k.device() == t.q.device(), "sparse_sdpa kv_lanes: every lane tensor on q's mesh");
    }
}

}  // namespace

tt::tt_metal::ProgramDescriptor SparseSDPALanesOperation::SparseSDPALanesProgramFactory::create_descriptor(
    const operation_attributes_t& attrs,
    const tensor_args_t& t,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    return SparseSDPAOperation::SparseSDPAProgramFactory::create_descriptor(
        attrs.base, base_inputs(t, lane_kv(attrs, t, mesh_dispatch_coordinate)), output);
}

void SparseSDPALanesOperation::SparseSDPALanesProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& attrs,
    const tensor_args_t& t,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    SparseSDPAOperation::SparseSDPAProgramFactory::override_runtime_arguments(
        program,
        attrs.base,
        base_inputs(t, lane_kv(attrs, t, mesh_dispatch_coordinate)),
        output,
        mesh_dispatch_coordinate);
}

void SparseSDPALanesOperation::validate_on_program_cache_miss(const operation_attributes_t& a, const tensor_args_t& t) {
    validate_lanes(a, t);
    SparseSDPAOperation::validate_on_program_cache_miss(a.base, base_inputs(t, t.kv_lanes.front()));
}

void SparseSDPALanesOperation::validate_on_program_cache_hit(const operation_attributes_t& a, const tensor_args_t& t) {
    validate_lanes(a, t);
    SparseSDPAOperation::validate_on_program_cache_hit(a.base, base_inputs(t, t.kv_lanes.front()));
}

SparseSDPALanesOperation::spec_return_value_t SparseSDPALanesOperation::compute_output_specs(
    const operation_attributes_t& a, const tensor_args_t& t) {
    return SparseSDPAOperation::compute_output_specs(a.base, base_inputs(t, t.kv_lanes.front()));
}

SparseSDPALanesOperation::tensor_return_value_t SparseSDPALanesOperation::create_output_tensors(
    const operation_attributes_t& a, const tensor_args_t& t) {
    return SparseSDPAOperation::create_output_tensors(a.base, base_inputs(t, t.kv_lanes.front()));
}

ttsl::hash::hash_t SparseSDPALanesOperation::compute_program_hash(
    const operation_attributes_t& a, const tensor_args_t& t) {
    // the source program's key (kv_lanes[0] stands for every lane: same spec, validated) + the lane map
    const auto base = SparseSDPAOperation::compute_program_hash(a.base, base_inputs(t, t.kv_lanes.front()));
    return tt::tt_metal::operation::hash_operation<SparseSDPALanesOperation>(
        base, a.lane_of_chip, static_cast<uint32_t>(t.kv_lanes.size()));
}

Tensor sparse_sdpa_lanes(
    const Tensor& q,
    const std::vector<Tensor>& kv_lanes,
    const std::vector<uint32_t>& lane_of_chip,
    const Tensor& indices,
    const SparseSDPAParams& base,
    const std::optional<Tensor>& attention_sink,
    const std::optional<Tensor>& kv_window) {
    using OperationType = ttnn::prim::bringup::SparseSDPALanesOperation;
    TT_FATAL(!kv_lanes.empty(), "sparse_sdpa kv_lanes: at least one tensor");
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{.base = base, .lane_of_chip = lane_of_chip},
        OperationType::tensor_args_t{
            .q = q,
            .kv = kv_lanes.front(),
            .indices = indices,
            .attention_sink = attention_sink,
            .kv_window = kv_window,
            .kv_lanes = kv_lanes,
        });
}

}  // namespace ttnn::prim::bringup
