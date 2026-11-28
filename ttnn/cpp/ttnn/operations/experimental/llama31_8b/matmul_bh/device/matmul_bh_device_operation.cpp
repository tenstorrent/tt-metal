// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/llama31_8b/matmul_bh/device/matmul_bh_device_operation.hpp"

#include "ttnn/operations/experimental/llama31_8b/matmul_bh/device/matmul_bh_l1.hpp"
#include "ttnn/operations/experimental/llama31_8b/matmul_bh/device/matmul_bh_shapes.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation.hpp"

namespace ttnn::prim::llama31_8b_matmul_bh {

ttnn::device_operation::ProgramArtifacts Mcast1DProgramFactory::create_program_artifacts(
    const MatmulParams& operation_attributes,
    const MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    auto route = bh_matmul_route(operation_attributes, tensor_args);
    TT_FATAL(route.has_value(), "llama31_8b.matmul_bh: no 1D program for this call");
    MatmulParams routed = operation_attributes;
    routed.program_config = route->program_config;
    return mcast_1d::create_program_artifacts(routed, tensor_args, tensor_return_value, *route->plan);
}

ttnn::device_operation::ProgramArtifacts Mcast2DProgramFactory::create_program_artifacts(
    const MatmulParams& operation_attributes,
    const MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    auto route = bh_matmul_route_2d(operation_attributes, tensor_args);
    TT_FATAL(route.has_value(), "llama31_8b.matmul_bh: no 2D program for this call");
    MatmulParams routed = operation_attributes;
    routed.program_config = route->program_config;
    return mcast_2d::create_program_artifacts(routed, tensor_args, tensor_return_value, *route->plan);
}

ttnn::device_operation::ProgramArtifacts DramShardedProgramFactory::create_program_artifacts(
    const MatmulParams& operation_attributes,
    const MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    auto route = bh_matmul_route_dram_sharded(operation_attributes, tensor_args);
    TT_FATAL(route.has_value(), "llama31_8b.matmul_bh: no DRAM-sharded program for this call");
    MatmulParams routed = operation_attributes;
    routed.program_config = route->program_config;
    return dram_sharded::create_program_artifacts(
        routed, tensor_args, tensor_return_value, route->read_options, *route->plan);
}

bool covers(const MatmulParams& operation_attributes, const MatmulInputs& tensor_args) {
    return bh_matmul_route(operation_attributes, tensor_args).has_value() ||
           bh_matmul_route_2d(operation_attributes, tensor_args).has_value() ||
           bh_matmul_route_dram_sharded(operation_attributes, tensor_args).has_value();
}

bool fits(const MatmulParams& operation_attributes, const MatmulInputs& tensor_args) {
    const auto& a = tensor_args.input_tensors.at(0);
    const auto& b = tensor_args.input_tensors.at(1);
    const auto out_dtype = operation_attributes.output_dtype.value_or(a.dtype());
    if (const auto route = bh_matmul_route(operation_attributes, tensor_args)) {
        return l1_fits(a, l1_bytes_1d(*route->plan, a, b, out_dtype));
    }
    if (const auto route = bh_matmul_route_2d(operation_attributes, tensor_args)) {
        return l1_fits(a, l1_bytes_2d(*route->plan, a, b, out_dtype));
    }
    return true;
}

MatmulBhDeviceOperation::program_factory_t MatmulBhDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (bh_matmul_route(operation_attributes, tensor_args).has_value()) {
        return Mcast1DProgramFactory{};
    }
    if (bh_matmul_route_2d(operation_attributes, tensor_args).has_value()) {
        return Mcast2DProgramFactory{};
    }
    TT_FATAL(
        bh_matmul_route_dram_sharded(operation_attributes, tensor_args).has_value(),
        "llama31_8b.matmul_bh: this call is not one of the Llama-3.1-8B Blackhole p100a shapes it covers");
    return DramShardedProgramFactory{};
}

static bool swiglu_epilogue_route(
    const MatmulBhDeviceOperation::operation_attributes_t& operation_attributes,
    const MatmulBhDeviceOperation::tensor_args_t& tensor_args) {
    const auto route = bh_matmul_route_2d(operation_attributes, tensor_args);
    return route.has_value() && route->plan->swiglu;
}

void MatmulBhDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    TT_FATAL(
        covers(operation_attributes, tensor_args),
        "llama31_8b.matmul_bh: this call is not one of the Llama-3.1-8B Blackhole p100a shapes it covers; "
        "use ttnn.linear");
    TT_FATAL(
        fits(operation_attributes, tensor_args),
        "llama31_8b.matmul_bh: not enough free L1 for this program; use ttnn.linear");
    if (!swiglu_epilogue_route(operation_attributes, tensor_args)) {
        MatmulDeviceOperation::validate_on_program_cache_miss(operation_attributes, tensor_args);
        return;
    }
    auto paired = operation_attributes;
    auto* config =
        std::get_if<operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>(&paired.program_config.value());
    TT_FATAL(config != nullptr, "llama31_8b.matmul_bh: the gate / up call takes the model's 2D program config");
    config->per_core_N *= 2;
    MatmulDeviceOperation::validate_on_program_cache_miss(paired, tensor_args);
}

MatmulBhDeviceOperation::spec_return_value_t MatmulBhDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto specs = MatmulDeviceOperation::compute_output_specs(operation_attributes, tensor_args);
    if (swiglu_epilogue_route(operation_attributes, tensor_args) && !specs.empty()) {
        const auto& s0 = specs[0];
        ttsl::SmallVector<uint32_t> dims(s0.logical_shape().cbegin(), s0.logical_shape().cend());
        dims.back() /= 2;
        specs[0] = tt::tt_metal::TensorSpec(ttnn::Shape(dims), s0.tensor_layout());
    }
    return specs;
}

MatmulBhDeviceOperation::tensor_return_value_t MatmulBhDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (!swiglu_epilogue_route(operation_attributes, tensor_args)) {
        return MatmulDeviceOperation::create_output_tensors(operation_attributes, tensor_args);
    }
    tensor_return_value_t out;
    for (const auto& spec : compute_output_specs(operation_attributes, tensor_args)) {
        out.emplace_back(create_device_tensor(spec, tensor_args.input_tensors.at(0).device()));
    }
    return out;
}

}  // namespace ttnn::prim::llama31_8b_matmul_bh

namespace ttnn::experimental::prim::llama31_8b_matmul_bh {

ttsl::hash::hash_t MinimalBhDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return MinimalMatmulDeviceOperation::compute_program_hash(operation_attributes, tensor_args);
}

void MinimalBhDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    MinimalMatmulDeviceOperation::validate_on_program_cache_hit(operation_attributes, tensor_args);
}

bool minimal_fits(const MinimalMatmulParams& operation_attributes, const MinimalMatmulInputs& tensor_args) {
    const auto* plan = ttnn::prim::llama31_8b_matmul_bh::minimal_plan(operation_attributes, tensor_args);
    if (plan == nullptr) {
        return true;
    }
    const auto& input = tensor_args.input_tensor;
    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(input.device()->arch(), operation_attributes.compute_kernel_config);
    return ttnn::prim::llama31_8b_matmul_bh::l1_fits(
        input,
        ttnn::prim::llama31_8b_matmul_bh::l1_bytes_minimal(
            *plan,
            input.dtype(),
            tensor_args.weight_tensor.dtype(),
            operation_attributes.output_dtype.value_or(input.dtype()),
            fp32_dest_acc_en));
}

void MinimalBhDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    MinimalMatmulDeviceOperation::validate_on_program_cache_miss(operation_attributes, tensor_args);
    TT_FATAL(
        minimal_fits(operation_attributes, tensor_args),
        "llama31_8b.minimal_matmul_bh: not enough free L1 for this program; use ttnn.experimental.minimal_matmul");
}

MinimalBhDeviceOperation::spec_return_value_t MinimalBhDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return MinimalMatmulDeviceOperation::compute_output_specs(operation_attributes, tensor_args);
}

MinimalBhDeviceOperation::tensor_return_value_t MinimalBhDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return MinimalMatmulDeviceOperation::create_output_tensors(operation_attributes, tensor_args);
}

}  // namespace ttnn::experimental::prim::llama31_8b_matmul_bh

namespace ttnn::experimental::prim::llama31_8b_matmul_bh {

namespace {
constexpr uint32_t kHeadDim = 128;
}  // namespace

ttnn::device_operation::ProgramArtifacts QkvHeadsProgramFactory::create_program_artifacts(
    const QkvHeadsParams& operation_attributes,
    const MinimalMatmulInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    return create_minimal_program_artifacts(
        operation_attributes.matmul,
        tensor_args,
        tensor_return_value,
        HeadsLayout{operation_attributes.num_heads, operation_attributes.num_kv_heads, kHeadDim});
}

tt::tt_metal::experimental::ProgramRunArgs QkvHeadsProgramFactory::override_runtime_arguments(
    const QkvHeadsParams& operation_attributes,
    const MinimalMatmulInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& coord) {
    return MinimalProgramFactory::override_runtime_arguments(
        operation_attributes.matmul, tensor_args, tensor_return_value, coord);
}

ttsl::hash::hash_t QkvHeadsDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ttsl::hash::hash_objects_with_default_seed(
        MinimalMatmulDeviceOperation::compute_program_hash(operation_attributes.matmul, tensor_args),
        operation_attributes.num_heads,
        operation_attributes.num_kv_heads);
}

void QkvHeadsDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    MinimalMatmulDeviceOperation::validate_on_program_cache_hit(operation_attributes.matmul, tensor_args);
}

void QkvHeadsDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    MinimalMatmulDeviceOperation::validate_on_program_cache_miss(operation_attributes.matmul, tensor_args);
    TT_FATAL(
        minimal_fits(operation_attributes.matmul, tensor_args),
        "llama31_8b.qkv_heads_bh: not enough free L1 for this program; use ttnn.experimental.minimal_matmul");
    const auto& shape = tensor_args.input_tensor.logical_shape();
    TT_FATAL(
        operation_attributes.matmul.chunks == 3 && shape.rank() == 4 && shape[-3] == 1 &&
            tensor_args.weight_tensor.logical_shape()[-1] ==
                (operation_attributes.num_heads + 2 * operation_attributes.num_kv_heads) * kHeadDim,
        "llama31_8b.qkv_heads_bh: needs a [B, 1, S, K] input and a fused QKV weight of (q + 2 kv) heads of 128");
}

QkvHeadsDeviceOperation::spec_return_value_t QkvHeadsDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto base = MinimalMatmulDeviceOperation::compute_output_specs(operation_attributes.matmul, tensor_args);
    const auto& in = tensor_args.input_tensor.logical_shape();
    spec_return_value_t specs;
    for (uint32_t heads :
         {operation_attributes.num_heads, operation_attributes.num_kv_heads, operation_attributes.num_kv_heads}) {
        ttnn::Shape shape(in);
        shape[-3] = heads;
        shape[-1] = kHeadDim;
        specs.emplace_back(shape, base.at(0).tensor_layout());
    }
    return specs;
}

QkvHeadsDeviceOperation::tensor_return_value_t QkvHeadsDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    tensor_return_value_t outputs;
    for (const auto& spec : compute_output_specs(operation_attributes, tensor_args)) {
        outputs.push_back(create_device_tensor(spec, tensor_args.input_tensor.device()));
    }
    return outputs;
}

}  // namespace ttnn::experimental::prim::llama31_8b_matmul_bh
