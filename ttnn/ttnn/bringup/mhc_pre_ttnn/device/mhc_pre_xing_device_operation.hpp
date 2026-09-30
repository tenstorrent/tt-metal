// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <optional>
#include <variant>
#include <vector>

#include <tt_stl/reflection.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::bringup::mhc_pre_ttnn {

// ttnn.bringup.mhc_pre_xing (mhc_pre_xing.hpp). The scalars are runtime args patched on every call (one program
// serves every layer); the hash holds only the mode, n and the tensor specs.
struct MhcPreXingParams {
    uint32_t n = 4;
    bool compute_coef = true;  // false: coefficients_given (the input is a finished hc)
    bool pack_stats = false;   // true: input = partial mix row, output = it with column n (n + 2) = sum x^2 of streams
    std::array<double, 3> scale{};
    std::vector<double> base;  // n (n + 2)
    double inv_nc = 0.0;
    double norm_eps = 1e-6;
    double hc_eps = 1e-6;
    uint32_t sinkhorn_iters = 20;
    double clamp_min = -30.0;
    double clamp_max = 30.0;
    tt::tt_metal::ComputeConfigDescriptor compute_config;
};

struct MhcPreXingInputs {
    Tensor input;                   // (..., T, 32) mix row, or (..., T, n(n+2)) hc
    std::optional<Tensor> streams;  // (..., T, n*C) fp32
};

tt::tt_metal::ProgramDescriptor create_xing_program_descriptor(
    const MhcPreXingParams& params, const MhcPreXingInputs& inputs, const std::vector<Tensor>& outputs);

struct MhcPreXingProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const MhcPreXingParams& operation_attributes,
        const MhcPreXingInputs& tensor_args,
        std::vector<Tensor>& outputs);

    // Cache hit: patch the buffer addresses of the reader / writer and the scalars of the compute kernel.
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const MhcPreXingParams& operation_attributes,
        const MhcPreXingInputs& tensor_args,
        std::vector<Tensor>& outputs,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

struct MhcPreXingDeviceOperation {
    using operation_attributes_t = MhcPreXingParams;
    using tensor_args_t = MhcPreXingInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t =
        std::vector<Tensor>;  // [hc] (compute_coef), then [y] (streams); [packed] (pack_stats)
    using program_factory_t = std::variant<MhcPreXingProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::bringup::mhc_pre_ttnn

namespace ttnn::prim::bringup {
std::vector<ttnn::Tensor> mhc_pre_xing(
    const ttnn::Tensor& input,
    const std::optional<ttnn::Tensor>& streams,
    const ttnn::operations::bringup::mhc_pre_ttnn::MhcPreXingParams& params);
}  // namespace ttnn::prim::bringup
