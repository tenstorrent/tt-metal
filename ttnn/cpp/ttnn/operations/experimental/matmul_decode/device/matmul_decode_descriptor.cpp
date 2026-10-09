// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_decode_descriptor.hpp"

#include <memory>
#include <optional>

#include <tt-metalium/global_circular_buffer.hpp>
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
        p.ring_gather,
        /*mesh_coords=*/std::nullopt,
        p.in0_row_major_height_sharded,
        p.output_core_grid,
        p.output_mcast_two_hub,
        p.rms_norm,
        p.rms_norm_gamma,
        p.rms_norm_epsilon,
        p.rms_norm_group_size,
    };
}

RealOp::tensor_args_t to_real_tensor_args(const MatmulDecodeInputs& t) {
    return RealOp::tensor_args_t{t.input_tensor_a, t.input_tensor_b, t.rms_norm_gamma};
}

// create_descriptor() stores `const GlobalCircularBuffer*` into the ProgramDescriptor.
// That pointer must stay valid until program build, which fusion can defer and repeat
// after this call returns. Copy the GCB onto each CB that referenced the temporary
// attributes. The copy shares the device buffer; the CB's shared_ptr drops it when
// the last descriptor copy is destroyed.
void retain_global_circular_buffer(
    tt::tt_metal::ProgramDescriptor& descriptor, const RealOp::operation_attributes_t& attrs) {
    if (!attrs.global_cb.has_value()) {
        return;
    }
    const auto* original = std::addressof(*attrs.global_cb);
    auto owned = std::make_shared<const tt::tt_metal::experimental::GlobalCircularBuffer>(*attrs.global_cb);
    for (auto& cb : descriptor.cbs) {
        if (cb.global_circular_buffer == original) {
            cb.owned_global_circular_buffer = owned;
            cb.global_circular_buffer = owned.get();
        }
    }
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
    auto real_attrs = to_real_attributes(operation_attributes);
    auto descriptor = RealOp::FullWidthSharded::create_descriptor(
        real_attrs, to_real_tensor_args(tensor_args), tensor_return_value, mesh_dispatch_coordinate);
    retain_global_circular_buffer(descriptor, real_attrs);
    return descriptor;
}

tt::tt_metal::ProgramDescriptor matmul_decode_partial_width_sharded_create_descriptor(
    const MatmulDecodeParams& operation_attributes,
    const MatmulDecodeInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto real_attrs = to_real_attributes(operation_attributes);
    auto descriptor = RealOp::PartialWidthSharded::create_descriptor(
        real_attrs, to_real_tensor_args(tensor_args), tensor_return_value);
    retain_global_circular_buffer(descriptor, real_attrs);
    return descriptor;
}

tt::tt_metal::ProgramDescriptor matmul_decode_batched_width_sharded_create_descriptor(
    const MatmulDecodeParams& operation_attributes,
    const MatmulDecodeInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto real_attrs = to_real_attributes(operation_attributes);
    auto descriptor = RealOp::BatchedWidthSharded::create_descriptor(
        real_attrs, to_real_tensor_args(tensor_args), tensor_return_value);
    retain_global_circular_buffer(descriptor, real_attrs);
    return descriptor;
}

}  // namespace ttnn::prim
