// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/moreh_adamw/moreh_adamw_op.hpp"

#include <optional>
#include <utility>

#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt::tt_metal;

namespace tt {
namespace operations {
namespace primary {

void MorehAdamW::validate_with_output_tensors(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        const std::vector<std::optional<Tensor>> &output_tensors) const {
    TT_ASSERT(
        input_tensors.size() == 4 and optional_input_tensors.size() <= 1,
        "moreh_adamw must have between 4 to 5 input tensors");

    const auto& param_in = input_tensors.at(0);
    const auto& grad = input_tensors.at(1);
    const auto& exp_avg_in = input_tensors.at(2);
    const auto& exp_avg_sq_in = input_tensors.at(3);

    const auto& max_exp_avg_sq_in = optional_input_tensors.at(0);

    check_tensor(param_in, "moreh_adamw", "param_in");
    check_tensor(grad, "moreh_adamw", "grad");
    check_tensor(exp_avg_in, "moreh_adamw", "exp_avg_in");
    check_tensor(exp_avg_sq_in, "moreh_adamw", "exp_avg_sq_in");

    if (max_exp_avg_sq_in.has_value()) {
        check_tensor(max_exp_avg_sq_in.value(), "moreh_adamw", "max_exp_avg_sq_in");
    }

    const auto& param_out = output_tensors.at(0);
    const auto& exp_avg_out = output_tensors.at(1);
    const auto& exp_avg_sq_out = output_tensors.at(2);
    const auto& max_exp_avg_sq_out = output_tensors.at(3);

    if (param_out.has_value()) {
        check_tensor(param_out.value(), "moreh_adamw", "param_out");
    }
    if (exp_avg_out.has_value()) {
        check_tensor(exp_avg_out.value(), "moreh_adamw", "exp_avg_out");
    }
    if (exp_avg_sq_out.has_value()) {
        check_tensor(exp_avg_sq_out.value(), "moreh_adamw", "exp_avg_sq_out");
    }
    if (max_exp_avg_sq_out.has_value()) {
        check_tensor(max_exp_avg_sq_out.value(), "moreh_adamw", "max_exp_avg_sq_out");
    }
}

std::vector<tt::tt_metal::LegacyShape> MorehAdamW::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    auto output_shape = input_tensors.at(0).get_legacy_shape();
    return {output_shape, output_shape, output_shape, output_shape};
}

std::vector<Tensor> MorehAdamW::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {

    const auto& output_shapes = this->compute_output_shapes(input_tensors);
    auto dtype = input_tensors.at(0).get_dtype();
    Layout layout{Layout::TILE};
    auto device = input_tensors.at(0).device();

    std::vector<Tensor> result;

    for (uint32_t idx = 0 ; idx < 4; idx++) {
        if (output_tensors.at(idx).has_value()) {
            result.push_back(output_tensors.at(idx).value());
        } else {
            result.push_back(create_device_tensor(output_shapes.at(idx), dtype, layout, device, this->output_mem_config));
        }
    }

    return result;
}

operation::ProgramWithCallbacks MorehAdamW::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    const auto& param_in = input_tensors.at(0);
    const auto& grad = input_tensors.at(1);
    const auto& exp_avg_in = input_tensors.at(2);
    const auto& exp_avg_sq_in = input_tensors.at(3);

    const auto& max_exp_avg_sq_in = optional_input_tensors.at(0);

    const auto& param_out = output_tensors.at(0);
    const auto& exp_avg_out = output_tensors.at(1);
    const auto& exp_avg_sq_out = output_tensors.at(2);

    const auto& max_exp_avg_sq_out =
        this->amsgrad ? std::make_optional<Tensor>(output_tensors.at(3)) : std::nullopt;

    return moreh_adamw_(
        param_in,
        grad,
        exp_avg_in,
        exp_avg_sq_in,
        this->lr,
        this->beta1,
        this->beta2,
        this->eps,
        this->weight_decay,
        this->step,
        this->amsgrad,
        max_exp_avg_sq_in,
        param_out,
        exp_avg_out,
        exp_avg_sq_out,
        max_exp_avg_sq_out,
        this->core_range,
        this->compute_kernel_config);
}

std::vector<std::optional<Tensor>> moreh_adamw(
    const Tensor& param_in,
    const Tensor& grad,
    const Tensor& exp_avg_in,
    const Tensor& exp_avg_sq_in,

    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    uint32_t step,
    bool amsgrad,

    const std::optional<const Tensor> max_exp_avg_sq_in,
    const std::optional<const Tensor> param_out,
    const std::optional<const Tensor> exp_avg_out,
    const std::optional<const Tensor> exp_avg_sq_out,
    const std::optional<const Tensor> max_exp_avg_sq_out,
    const MemoryConfig& mem_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {

    auto device = param_in.device();
    auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange all_cores({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});

    auto compute_kernel_config_val =
        init_device_compute_kernel_config(device->arch(), compute_kernel_config, MathFidelity::HiFi4);

    std::vector<Tensor> output_tensors = {
        Tensor(operation::get_workers_for_op_output({param_in, grad, exp_avg_in, exp_avg_sq_in}, {max_exp_avg_sq_in})),
        Tensor(operation::get_workers_for_op_output({param_in, grad, exp_avg_in, exp_avg_sq_in}, {max_exp_avg_sq_in})),
        Tensor(operation::get_workers_for_op_output({param_in, grad, exp_avg_in, exp_avg_sq_in}, {max_exp_avg_sq_in})),
        Tensor(operation::get_workers_for_op_output({param_in, grad, exp_avg_in, exp_avg_sq_in}, {max_exp_avg_sq_in}))
        };

    operation::launch_op(
        [lr, beta1, beta2, eps, weight_decay, step, amsgrad, all_cores, mem_config, compute_kernel_config_val](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            return operation::run(
                MorehAdamW{
                    .lr = lr,
                    .beta1 = beta1,
                    .beta2 = beta2,
                    .eps = eps,
                    .weight_decay = weight_decay,
                    .step = step,
                    .amsgrad = amsgrad,
                    .core_range = all_cores,
                    .output_mem_config = mem_config,
                    .compute_kernel_config = compute_kernel_config_val},
                input_tensors,
                optional_input_tensors,
                optional_output_tensors);
        },
        {param_in, grad, exp_avg_in, exp_avg_sq_in},
        output_tensors,
        {max_exp_avg_sq_in},
        {param_out, exp_avg_out, exp_avg_sq_out, max_exp_avg_sq_out});

    std::vector<std::optional<Tensor>> optional_outputs(4);
    optional_outputs[0] = output_tensors[0]; // param_out
    optional_outputs[1] = output_tensors[1]; // exp_avg_out
    optional_outputs[2] = output_tensors[2]; // exp_avg_sq_out
    if (max_exp_avg_sq_out.has_value()) {
        optional_outputs[3] = output_tensors[3]; // max_exp_avg_sq_out
    }
    return optional_outputs;
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
