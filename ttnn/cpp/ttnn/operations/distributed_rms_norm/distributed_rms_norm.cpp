// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "distributed_rms_norm.hpp"

#include <stdexcept>
#include <type_traits>

#include <fmt/format.h>

#include "device/distributed_rms_norm_device_operation.hpp"
#include "device/distributed_rms_norm_planner.hpp"
#include "device/kernels/distributed_rms_norm_args.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn {

namespace {

namespace drms = operations::distributed_rms_norm;

drms::ExchangeMode parse_exchange_mode(const std::optional<std::string>& exchange_mode) {
    if (!exchange_mode.has_value() || *exchange_mode == "direct") {
        return drms::ExchangeMode::DIRECT;
    }
    if (*exchange_mode == "relay") {
        return drms::ExchangeMode::RELAY;
    }
    if (*exchange_mode == "mcast") {
        return drms::ExchangeMode::MCAST;
    }
    throw std::invalid_argument(fmt::format(
        "distributed_rms_norm: exchange_mode must be one of ('direct', 'relay', 'mcast'), got '{}'", *exchange_mode));
}

// The Python's default_compute_kernel_config and _compute_config: fidelity, fp32 DEST, approx mode and DEST sync of
// the caller's (or the default) config, and the lossless fp32 unpack of the gathered records when fp32 DEST is on.
tt::tt_metal::ComputeConfigDescriptor resolve_compute_config(
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config) {
    tt::tt_metal::ComputeConfigDescriptor out;
    using Fidelity = decltype(out.math_fidelity);
    if (!compute_kernel_config.has_value()) {
        out.math_fidelity = Fidelity::HiFi4;
        out.fp32_dest_acc_en = true;
        out.math_approx_mode = false;
    } else {
        out.math_fidelity = compute_kernel_config->math_fidelity;
        out.fp32_dest_acc_en = compute_kernel_config->fp32_dest_acc_en;
        out.math_approx_mode = compute_kernel_config->math_approx_mode;
        out.dst_full_sync_en = compute_kernel_config->dst_full_sync_en;
    }
    if (out.fp32_dest_acc_en) {
        using Mode = std::decay_t<decltype(out.unpack_to_dest_mode)>::value_type;
        constexpr size_t kDefaultModes = 64;
        auto modes = compute_kernel_config.has_value() ? compute_kernel_config->unpack_to_dest_mode
                                                       : decltype(out.unpack_to_dest_mode){};
        if (modes.empty()) {
            modes.assign(kDefaultModes, Mode::Default);
        }
        modes.at(::distributed_rms_norm::cb::STATS_GATHER) = Mode::UnpackToDestFp32;
        out.unpack_to_dest_mode = std::move(modes);
    }
    return out;
}

MemoryConfig resolve_output_memory_config(const Tensor& input, const std::optional<MemoryConfig>& memory_config) {
    if (memory_config.has_value()) {
        return *memory_config;
    }
    if (input.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED) {
        return ttnn::DRAM_MEMORY_CONFIG;
    }
    return input.memory_config();
}

// validate() of the Python entry: the caller errors, then the support contract, then the regime refusals.
drms::DistributedRmsNormParams resolve_params(
    const Tensor& input_tensor,
    int32_t cluster_axis,
    const std::optional<Tensor>& gamma,
    double epsilon,
    tt::tt_fabric::Topology topology,
    std::optional<int32_t> num_links,
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<std::string>& exchange_mode,
    const std::optional<Tensor>& output_tensor,
    const std::optional<Tensor>& scratch_tensor) {
    TT_FATAL(is_device_tensor(input_tensor), "distributed_rms_norm: input_tensor must be on a mesh device");
    const drms::ExchangeMode mode = parse_exchange_mode(exchange_mode);
    tt::tt_metal::ComputeConfigDescriptor config = resolve_compute_config(compute_kernel_config);
    const drms::GroupResolution group =
        drms::check_caller_inputs(input_tensor, cluster_axis, gamma, topology, num_links);
    const drms::DistributedRmsNormParams params{
        .cluster_axis = static_cast<uint32_t>(cluster_axis),
        .epsilon = epsilon,
        .topology = topology,
        .group_size = group.group_size,
        .num_links = num_links.has_value() ? static_cast<uint32_t>(*num_links) : group.usable_links,
        .ring = group.ring,
        .exchange_mode = mode,
        .compute_config = std::move(config),
        .output_memory_config = resolve_output_memory_config(input_tensor, memory_config),
    };
    drms::check_support(
        params,
        {.input = input_tensor, .gamma = gamma, .output = output_tensor, .scratch = scratch_tensor},
        memory_config);
    return params;
}

Tensor allocate_output(
    const drms::DistributedRmsNormParams& params, const Tensor& input_tensor, const std::optional<Tensor>& gamma) {
    const std::optional<Tensor> none;
    return create_device_tensor(
        drms::DistributedRmsNormDeviceOperation::compute_output_specs(
            params, {.input = input_tensor, .gamma = gamma, .output = none, .scratch = none}),
        input_tensor.device());
}

Tensor allocate_landing(
    const drms::DistributedRmsNormParams& params,
    const Tensor& input_tensor,
    const std::optional<Tensor>& gamma,
    const Tensor& output) {
    const std::optional<Tensor> out{output};
    const std::optional<Tensor> none;
    const drms::DistributedRmsNormInputs inputs{.input = input_tensor, .gamma = gamma, .output = out, .scratch = none};
    const auto& plan = drms::detail::solve_plan(drms::detail::plan_inputs(params, inputs, output));
    auto* mesh = input_tensor.device();
    return create_device_tensor(drms::detail::landing_spec(*mesh, plan.landing_alloc_words), mesh);
}

}  // namespace

Tensor distributed_rms_norm(
    const Tensor& input_tensor,
    int32_t cluster_axis,
    const std::optional<Tensor>& gamma,
    double epsilon,
    tt::tt_fabric::Topology topology,
    std::optional<int32_t> num_links,
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<std::string>& exchange_mode,
    const std::optional<Tensor>& output_tensor,
    const std::optional<Tensor>& scratch_tensor) {
    const drms::DistributedRmsNormParams params = resolve_params(
        input_tensor,
        cluster_axis,
        gamma,
        epsilon,
        topology,
        num_links,
        compute_kernel_config,
        memory_config,
        exchange_mode,
        output_tensor,
        scratch_tensor);
    // The output first, then the landing planned next to it (the Python's allocation order). A landing this call
    // allocates is dropped once the program is enqueued.
    const std::optional<Tensor> output =
        output_tensor.has_value() ? output_tensor : std::optional<Tensor>(allocate_output(params, input_tensor, gamma));
    const std::optional<Tensor> scratch =
        scratch_tensor.has_value() ? scratch_tensor
                                   : std::optional<Tensor>(allocate_landing(params, input_tensor, gamma, *output));
    return ttnn::prim::distributed_rms_norm(input_tensor, gamma, output, scratch, params);
}

Tensor distributed_rms_norm_scratch_tensor(
    const Tensor& input_tensor,
    int32_t cluster_axis,
    const std::optional<Tensor>& gamma,
    double epsilon,
    tt::tt_fabric::Topology topology,
    std::optional<int32_t> num_links,
    const std::optional<tt::tt_metal::ComputeConfigDescriptor>& compute_kernel_config,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<std::string>& exchange_mode) {
    const std::optional<Tensor> none;
    const drms::DistributedRmsNormParams params = resolve_params(
        input_tensor,
        cluster_axis,
        gamma,
        epsilon,
        topology,
        num_links,
        compute_kernel_config,
        memory_config,
        exchange_mode,
        none,
        none);
    const Tensor output = allocate_output(params, input_tensor, gamma);
    return allocate_landing(params, input_tensor, gamma, output);
}

}  // namespace ttnn
