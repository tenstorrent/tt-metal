// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/moreh_adam/moreh_adam_op.hpp"

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

void MorehAdam::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    const std::vector<std::optional<Tensor>>& output_tensors) const {
    TT_ASSERT(
        input_tensors.size() == 4 and optional_input_tensors.size() <= 1,
        "moreh_adam must have between 4 to 6 input tensors");

    const auto& param_in = input_tensors.at(0);
    const auto& grad = input_tensors.at(1);
    const auto& exp_avg_in = input_tensors.at(2);
    const auto& exp_avg_sq_in = input_tensors.at(3);

    const auto& max_exp_avg_sq_in = optional_input_tensors.at(0);

    check_tensor(param_in, "moreh_adam", "param_in");
    check_tensor(grad, "moreh_adam", "grad");
    check_tensor(exp_avg_in, "moreh_adam", "exp_avg_in");
    check_tensor(exp_avg_sq_in, "moreh_adam", "exp_avg_sq_in");

    if (max_exp_avg_sq_in.has_value()) {
        check_tensor(max_exp_avg_sq_in.value(), "moreh_adam", "max_exp_avg_sq_in");
    }

    const auto& param_out = output_tensors.at(0);
    const auto& exp_avg_out = output_tensors.at(1);
    const auto& exp_avg_sq_out = output_tensors.at(2);
    const auto& max_exp_avg_sq_out = output_tensors.at(3);

    if (param_out.has_value()) {
        check_tensor(param_out.value(), "moreh_adam", "param_out");
    }
    if (exp_avg_out.has_value()) {
        check_tensor(exp_avg_out.value(), "moreh_adam", "exp_avg_out");
    }
    if (exp_avg_sq_out.has_value()) {
        check_tensor(exp_avg_sq_out.value(), "moreh_adam", "exp_avg_sq_out");
    }
    if (max_exp_avg_sq_out.has_value()) {
        check_tensor(max_exp_avg_sq_out.value(), "moreh_adam", "max_exp_avg_sq_out");
    }
}

std::vector<tt::tt_metal::LegacyShape> MorehAdam::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    auto output_shape = input_tensors.at(0).get_legacy_shape();
    return {output_shape, output_shape, output_shape, output_shape};
}

std::vector<Tensor> MorehAdam::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    const auto& output_shapes = this->compute_output_shapes(input_tensors);
    auto dtype = input_tensors.at(0).get_dtype();
    Layout layout{Layout::TILE};
    auto device = input_tensors.at(0).device();

    std::vector<Tensor> result;

    auto idx = uint32_t{0};
    if (output_tensors.at(idx).has_value()) {
        result.push_back(output_tensors.at(idx).value());
    } else {
        result.push_back(create_device_tensor(output_shapes.at(idx), dtype, layout, device, this->output_mem_config));
    }
    ++idx;

    if (output_tensors.at(idx).has_value()) {
        result.push_back(output_tensors.at(idx).value());
    } else {
        result.push_back(create_device_tensor(output_shapes.at(idx), dtype, layout, device, this->output_mem_config));
    }
    ++idx;

    if (output_tensors.at(idx).has_value()) {
        result.push_back(output_tensors.at(idx).value());
    } else {
        result.push_back(create_device_tensor(output_shapes.at(idx), dtype, layout, device, this->output_mem_config));
    }
    ++idx;

    if (output_tensors.at(idx).has_value()) {
        result.push_back(output_tensors.at(idx).value());
    } else if (amsgrad) {
        result.push_back(create_device_tensor(output_shapes.at(idx), dtype, layout, device, this->output_mem_config));
    }

    return std::move(result);
}

operation::ProgramWithCallbacks MorehAdam::create_program(
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
        amsgrad ? std::make_optional<Tensor>(output_tensors.at(3)) : std::nullopt;

    return moreh_adam_(
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
        this->compute_kernel_config);
}

std::vector<std::optional<Tensor>> moreh_adam(
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

    if(amsgrad){
        TT_ASSERT(max_exp_avg_sq_in.has_value());
    }

    auto device = param_in.device();

    auto compute_kernel_config_val =
        init_device_compute_kernel_config(device->arch(), compute_kernel_config, MathFidelity::HiFi4);

    std::vector<Tensor> output_tensors = {
        Tensor(operation::get_workers_for_op_output({param_in, grad, exp_avg_in, exp_avg_sq_in}, {max_exp_avg_sq_in})),
        Tensor(operation::get_workers_for_op_output({param_in, grad, exp_avg_in, exp_avg_sq_in}, {max_exp_avg_sq_in})),
        Tensor(operation::get_workers_for_op_output({param_in, grad, exp_avg_in, exp_avg_sq_in}, {max_exp_avg_sq_in}))};

    if (max_exp_avg_sq_in.has_value()) {
        output_tensors.push_back(Tensor(
            operation::get_workers_for_op_output({param_in, grad, exp_avg_in, exp_avg_sq_in}, {max_exp_avg_sq_in})));
    }


    operation::launch_op(
        [lr,
         beta1,
         beta2,
         eps,
         weight_decay,
         step,
         amsgrad,
         mem_config,
         compute_kernel_config_val](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            return operation::run(
                MorehAdam{
                    .lr = lr,
                    .beta1 = beta1,
                    .beta2 = beta2,
                    .eps = eps,
                    .weight_decay = weight_decay,
                    .step = step,
                    .amsgrad = amsgrad,
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
    optional_outputs[0] = output_tensors[0];
    optional_outputs[1] = output_tensors[1];
    optional_outputs[2] = output_tensors[2];
    if (max_exp_avg_sq_in.has_value()) {
        optional_outputs[3] = output_tensors[3];
    }
    return optional_outputs;
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
