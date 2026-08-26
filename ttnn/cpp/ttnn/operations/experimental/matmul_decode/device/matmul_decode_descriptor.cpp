// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_decode_descriptor.hpp"

#include <memory>
#include <mutex>
#include <vector>
#include <optional>

#include <tt-metalium/mesh_coord.hpp>

namespace ttnn::prim {

namespace {

using RealOp = ttnn::operations::experimental::matmul_decode::MatmulDecodeDeviceOperation;

// The real operation_attributes_t/tensor_args_t are trivial field-for-field mirrors of
// MatmulDecodeParams/MatmulDecodeInputs (see matmul_decode_device_operation.hpp); this just
// re-packs the by-value Python-facing types into the reference-holding ones the real device
// operation and its factories expect.
RealOp::operation_attributes_t to_real_attributes(const MatmulDecodeParams& p) {
    return RealOp::operation_attributes_t{
        p.M,
        p.N,
        p.K,
        p.output_mem_config,
        p.output_dtype,
        p.partial_width_sharded,
        p.batch,
        p.b_blocks,
        p.n_blocks,
        p.global_cb,
        p.global_cb_k_blocks,
        p.packed_weight,
        p.all_gather,
        p.ring_size,
    };
}

RealOp::tensor_args_t to_real_tensor_args(const MatmulDecodeInputs& t) {
    return RealOp::tensor_args_t{t.input_tensor_a, t.input_tensor_b};
}

// Each program factory's create_descriptor() embeds a raw
// `const GlobalCircularBuffer*` (see CBDescriptor::global_circular_buffer in
// program_descriptors.hpp) straight into the returned ProgramDescriptor, on the
// assumption that whatever owns `operation_attributes_t` -- and therefore its
// by-value `global_cb` -- outlives the descriptor's use. That holds for the normal
// synchronous device-operation invoke path (operation_attributes_t lives on the
// calling stack through Program build), but not here: `to_real_attributes(...)`
// below builds a fresh by-value copy that is a temporary of the single
// create_descriptor() call, so it would be destroyed before
// models/experimental/ops/descriptors/matmul_decode.py's cached/deferred
// ProgramDescriptor is actually dispatched (build()/launch() can run well after
// this call returns, and fusion.py's build cache can dispatch it many times) --
// leaving CBDescriptor::global_circular_buffer dangling and segfaulting on launch.
//
// Fix: keep one owning copy of operation_attributes_t alive for the process
// lifetime per create_descriptor() call and hand create_descriptor a reference into
// that copy instead of a bare temporary. GlobalCircularBuffers are long-lived,
// model-scoped objects in practice (built once per weight layout, not per call), so
// the leaked set stays small; this trades a bounded, permanent allocation for
// correctness without changing the shared ProgramDescriptor/CBDescriptor ownership
// model.
const RealOp::operation_attributes_t& keep_attributes_alive(RealOp::operation_attributes_t&& attrs) {
    static std::mutex mutex;
    static std::vector<std::unique_ptr<RealOp::operation_attributes_t>> registry;
    std::lock_guard<std::mutex> lock(mutex);
    registry.push_back(std::make_unique<RealOp::operation_attributes_t>(std::move(attrs)));
    return *registry.back();
}

}  // namespace

MatmulDecodeDeviceOperation::tensor_return_value_t MatmulDecodeDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return RealOp::create_output_tensors(to_real_attributes(operation_attributes), to_real_tensor_args(tensor_args));
}

MatmulDecodeDeviceOperation::spec_return_value_t MatmulDecodeDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return RealOp::compute_output_specs(to_real_attributes(operation_attributes), to_real_tensor_args(tensor_args));
}

ttsl::hash::hash_t MatmulDecodeDeviceOperation::compute_descriptor_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return RealOp::compute_program_hash(to_real_attributes(operation_attributes), to_real_tensor_args(tensor_args));
}

RealOp::program_factory_t matmul_decode_select_program_factory(
    const MatmulDecodeParams& operation_attributes, const MatmulDecodeInputs& tensor_args) {
    return RealOp::select_program_factory(to_real_attributes(operation_attributes), to_real_tensor_args(tensor_args));
}

tt::tt_metal::ProgramDescriptor matmul_decode_full_width_sharded_create_descriptor(
    const MatmulDecodeParams& operation_attributes,
    const MatmulDecodeInputs& tensor_args,
    Tensor& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    const auto& real_attrs = keep_attributes_alive(to_real_attributes(operation_attributes));
    return RealOp::FullWidthSharded::create_descriptor(
        real_attrs, to_real_tensor_args(tensor_args), tensor_return_value, mesh_dispatch_coordinate);
}

tt::tt_metal::ProgramDescriptor matmul_decode_partial_width_sharded_create_descriptor(
    const MatmulDecodeParams& operation_attributes,
    const MatmulDecodeInputs& tensor_args,
    Tensor& tensor_return_value) {
    const auto& real_attrs = keep_attributes_alive(to_real_attributes(operation_attributes));
    return RealOp::PartialWidthSharded::create_descriptor(
        real_attrs, to_real_tensor_args(tensor_args), tensor_return_value);
}

tt::tt_metal::ProgramDescriptor matmul_decode_batched_width_sharded_create_descriptor(
    const MatmulDecodeParams& operation_attributes,
    const MatmulDecodeInputs& tensor_args,
    Tensor& tensor_return_value) {
    const auto& real_attrs = keep_attributes_alive(to_real_attributes(operation_attributes));
    return RealOp::BatchedWidthSharded::create_descriptor(
        real_attrs, to_real_tensor_args(tensor_args), tensor_return_value);
}

}  // namespace ttnn::prim
