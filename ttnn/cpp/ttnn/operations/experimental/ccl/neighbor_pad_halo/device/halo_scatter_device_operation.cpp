// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "halo_scatter_device_operation.hpp"
#include "halo_scatter_device_operation_types.hpp"
#include "halo_scatter_program_factory.hpp"

#include <tt-metalium/tt_metal.hpp>

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

void NpHaloScatterDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& compact = tensor_args.compact_buffer;
    const auto& x = tensor_args.interior_src;

    TT_FATAL(compact.layout() == Layout::ROW_MAJOR, "HaloScatter: compact_buffer must be row-major.");
    TT_FATAL(x.layout() == Layout::ROW_MAJOR, "HaloScatter: interior_src must be row-major.");
    TT_FATAL(
        compact.dtype() == x.dtype(),
        "HaloScatter: compact ({}) and interior_src ({}) dtype must match.",
        compact.dtype(),
        x.dtype());
    TT_FATAL(
        compact.dtype() == DataType::BFLOAT16 || compact.dtype() == DataType::FLOAT32,
        "HaloScatter: dtype must be bfloat16 or float32. got {}",
        compact.dtype());
    TT_FATAL(
        x.logical_shape().size() == 5,
        "HaloScatter: interior_src must be rank 5 [B,T,H,W,C]. got {}",
        x.logical_shape().size());
    TT_FATAL(args.np_padding_h > 0, "HaloScatter: np_padding_h must be > 0.");
}

TensorSpec NpHaloScatterDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // Padded output = interior_src [B,T,H,W,C] with H,W grown by 2*pad; allocated (every page written).
    const auto& s = tensor_args.interior_src.logical_shape();
    ttnn::Shape output_shape(
        {s[0], s[1], s[2] + 2 * args.np_padding_h, s[3] + 2 * args.np_padding_w, s[4]});
    return TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            tensor_args.interior_src.dtype(),
            tt::tt_metal::PageConfig(Layout::ROW_MAJOR),
            args.output_mem_config));
}

Tensor NpHaloScatterDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.interior_src.device());
}

ttsl::hash::hash_t NpHaloScatterDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& x = tensor_args.interior_src;
    return operation::hash_operation<NpHaloScatterDeviceOperation>(
        args, x.dtype(), x.memory_config(), x.logical_shape());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor halo_scatter(
    const Tensor& compact_buffer,
    const Tensor& interior_src,
    const ttnn::experimental::prim::NpHaloScatterParams& params) {
    using OperationType = ttnn::experimental::prim::NpHaloScatterDeviceOperation;
    auto tensor_args =
        OperationType::tensor_args_t{.compact_buffer = compact_buffer, .interior_src = interior_src};
    return ttnn::device_operation::launch<OperationType>(params, tensor_args);
}

}  // namespace ttnn::prim
