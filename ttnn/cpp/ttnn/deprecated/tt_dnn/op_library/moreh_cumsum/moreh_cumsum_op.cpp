// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/moreh_cumsum/moreh_cumsum_op.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {
using namespace constants;
namespace operations {
namespace primary {

////////////////////////////////////////////////////////////////////////////
//                         MorehCumSum
////////////////////////////////////////////////////////////////////////////
void MorehCumSum::validate(const std::vector<Tensor>& inputs) const {
    TT_ASSERT((dim >= 0 && dim <= 3), "dim should be 0 - 3");
    const auto& input = inputs.at(0);
    const auto& output = inputs.at(1);

    auto input_shape = input.get_legacy_shape();
    const auto& output_shape = output.get_legacy_shape();
    auto input_shape_wo_padding = input.get_legacy_shape().without_padding();
    const auto& output_shape_wo_padding = output.get_legacy_shape().without_padding();

    for (int i = 0; i < input_shape.rank(); ++i) {
        TT_ASSERT(input_shape[i] == output_shape[i]);
        TT_ASSERT(input_shape_wo_padding[i] == output_shape_wo_padding[i]);
    }
}

std::vector<Tensor> MorehCumSum::create_output_tensors(const std::vector<Tensor>& inputs) const {
    // Inplace
    return {};
}

std::vector<Shape> MorehCumSum::compute_output_shapes(const std::vector<Tensor>& inputs) const {
    // Inplace
    return {};
}

operation::ProgramWithCallbacks MorehCumSum::create_program(
    const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) const {
    TT_ASSERT((dim >= 0 && dim <= 3), "dim should be 0 - 3");
    auto& input = inputs.at(0);
    auto& output = inputs.at(1);

    if (dim == 2 || dim == 3) {
        TT_ASSERT(false, "currenty only support moreh_cumsum op for dim 0, 1");
    }

    return moreh_cumsum_nc(input, output, dim, flip);
}

Tensor moreh_cumsum_(const Tensor& input, const Tensor& output, const int64_t& dim, const bool flip = false) {
    std::vector<Tensor> dummy_output_tensors = {Tensor(operation::get_workers_for_op_output({input, output}))};

    operation::launch_op(
        [dim, flip](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            return operation::run(
                MorehCumSum{.dim = dim, .flip = flip}, input_tensors, optional_input_tensors, optional_output_tensors);
        },
        {input, output},
        dummy_output_tensors);
    return output;
}

Tensor moreh_cumsum_backward(const Tensor& output_grad, const Tensor& input_grad, const int64_t& dim) {
    return moreh_cumsum_(output_grad, input_grad, dim, true);
}

Tensor moreh_cumsum(const Tensor& input, const Tensor& output, const int64_t& dim) {
    return moreh_cumsum_(input, output, dim);
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
