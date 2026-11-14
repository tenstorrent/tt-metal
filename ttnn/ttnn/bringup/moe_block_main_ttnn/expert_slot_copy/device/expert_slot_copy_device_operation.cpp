// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "expert_slot_copy_device_operation.hpp"

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/operation.hpp"

namespace ttnn::operations::bringup::moe_block_main::expert_slot_copy {

namespace expert_slot_copy_detail {

void validate_banked(const Tensor& t, const char* name) {
    TT_FATAL(
        t.storage_type() == StorageType::DEVICE && t.is_allocated(), "expert_slot_copy: {} must be on device", name);
    const auto& mc = t.memory_config();
    TT_FATAL(
        mc.buffer_type() == BufferType::DRAM && mc.is_sharded(),
        "expert_slot_copy: {} must be DRAM sharded over the banks (the flat expert's weight layout)",
        name);
}

}  // namespace expert_slot_copy_detail

ttsl::hash::hash_t SlotCopyDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& t) {
    return tt::tt_metal::operation::hash_operation<SlotCopyDeviceOperation>(
        args.plan_key,
        args.chunk_bytes,
        args.depth,
        args.batch,
        args.links,
        t.src_gu.tensor_spec(),
        t.src_d.tensor_spec(),
        t.dst_gu.tensor_spec(),
        t.dst_d.tensor_spec(),
        t.runs.tensor_spec(),
        t.sems.tensor_spec());
}

void SlotCopyDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& t) {
    using namespace expert_slot_copy_detail;
    TT_FATAL(args.ports != nullptr, "expert_slot_copy: no port plan (call through ttnn.bringup.moe_block_main)");
    const auto& topo = args.ports->plan.topo;
    TT_FATAL(topo.num_chips() <= 255, "expert_slot_copy: {} chips (chip ids are 8 bits in a job)", topo.num_chips());
    auto* mesh = t.dst_gu.device();
    for (const Tensor* x : {&t.src_gu, &t.src_d, &t.dst_d, &t.runs, &t.sems}) {
        TT_FATAL(x->device() == mesh, "expert_slot_copy: every tensor must be on one mesh");
    }
    TT_FATAL(
        mesh->shape().dims() == 2 && mesh->shape()[0] == topo.R && mesh->shape()[1] == topo.C,
        "expert_slot_copy: the mesh is {}, the resources' plan {}x{}",
        mesh->shape(),
        topo.R,
        topo.C);
    validate_banked(t.src_gu, "src_gu");
    validate_banked(t.src_d, "src_d");
    validate_banked(t.dst_gu, "dst_gu");
    validate_banked(t.dst_d, "dst_d");
    TT_FATAL(
        t.runs.dtype() == DataType::UINT32 && t.runs.layout() == Layout::ROW_MAJOR &&
            t.runs.memory_config().buffer_type() == BufferType::DRAM &&
            t.runs.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "expert_slot_copy: runs must be uint32 ROW_MAJOR DRAM interleaved (a run table a page)");
    const size_t payload = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
    TT_FATAL(
        args.chunk_bytes % 64 == 0 && args.chunk_bytes > 0 && args.chunk_bytes <= payload,
        "expert_slot_copy: chunk {} B (a 64 B multiple within the fabric payload {} B)",
        args.chunk_bytes,
        payload);
    TT_FATAL(
        args.batch >= 1 && args.depth >= args.batch,
        "expert_slot_copy: depth {} >= batch {} >= 1",
        args.depth,
        args.batch);
    TT_FATAL(args.links >= 1 && args.links <= kMaxLinks, "expert_slot_copy: links {} (1..{})", args.links, kMaxLinks);
    const auto& sm = t.sems.memory_config();
    TT_FATAL(
        t.sems.dtype() == DataType::UINT32 && sm.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED &&
            sm.shard_spec().has_value() && sm.shard_spec()->shape[1] * 4 >= (1 + topo.num_chips()) * kSemStride,
        "expert_slot_copy: sems must be HEIGHT_SHARDED uint32 words, {} B a core",
        (1 + topo.num_chips()) * kSemStride);
    for (const auto& [key, core] : args.ports->cores) {
        TT_FATAL(sm.shard_spec()->grid.contains(core), "expert_slot_copy: port core {} has no words", core.str());
    }
}

void SlotCopyDeviceOperation::validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t& t) {
    for (const Tensor* x : {&t.src_gu, &t.src_d, &t.dst_gu, &t.dst_d, &t.runs}) {
        TT_FATAL(x->is_allocated(), "expert_slot_copy: every tensor must be allocated");
    }
}

SlotCopyDeviceOperation::spec_return_value_t SlotCopyDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& t) {
    return t.dst_gu.tensor_spec();
}

SlotCopyDeviceOperation::topology_return_value_t SlotCopyDeviceOperation::compute_output_topologies(
    const operation_attributes_t&, const tensor_args_t& t) {
    return {t.dst_gu.tensor_topology()};
}

SlotCopyDeviceOperation::tensor_return_value_t SlotCopyDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& t) {
    return t.dst_gu;  // (in place)
}

}  // namespace ttnn::operations::bringup::moe_block_main::expert_slot_copy

namespace ttnn::prim {

Tensor moe_block_main_expert_slot_copy(
    const ttnn::operations::bringup::moe_block_main::expert_slot_copy::SlotCopyInputs& inputs,
    const ttnn::operations::bringup::moe_block_main::expert_slot_copy::SlotCopyParams& params) {
    using namespace ttnn::operations::bringup::moe_block_main::expert_slot_copy;
    return ttnn::device_operation::launch<SlotCopyDeviceOperation>(params, inputs);
}

}  // namespace ttnn::prim
