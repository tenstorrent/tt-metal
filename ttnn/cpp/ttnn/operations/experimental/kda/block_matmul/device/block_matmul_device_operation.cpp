// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "block_matmul_device_operation.hpp"

#include "ttnn/operations/experimental/kda/factory/kda_block_matmul.hpp"

namespace ttnn::experimental::prim {

void BlockMatmulOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    TT_FATAL(attrs.cores.size() == 1, "block_matmul: cores must be one rectangle");
    kda_factory_detail::check_block_matmul(in.a, in.w, attrs.cores.bounding_box(), attrs.k_block, "block_matmul");
}

BlockMatmulOperation::spec_return_value_t BlockMatmulOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    auto shape = in.a.logical_shape();
    shape[-1] = in.w.logical_shape()[-1];
    return {tt::tt_metal::TensorSpec(
        shape,
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            attrs.output_mem_config))};
}

BlockMatmulOperation::tensor_return_value_t BlockMatmulOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    return {create_device_tensor(compute_output_specs(attrs, in)[0], in.a.device())};
}

tt::tt_metal::ProgramDescriptor BlockMatmulProgramFactory::create_descriptor(
    const BlockMatmulParams& attrs, const BlockMatmulInputs& in, std::vector<Tensor>& outputs) {
    return kda_factory_detail::block_matmul_program(
        *in.a.device(),
        in.a,
        in.w,
        outputs[0],
        attrs.cores.bounding_box(),
        attrs.k_block,
        0,
        attrs.compute_kernel_config);
}

}  // namespace ttnn::experimental::prim
