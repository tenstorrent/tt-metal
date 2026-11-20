// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
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
    bool deepseek = false;   // DeepSeek-V4 coefficient math (compute define MHC_XING_DEEPSEEK; off: Xing's program)
    bool given_pre = false;  // y collapses with tensor_args.collapse_pre (DM define MHC_XING_GIVEN_PRE; off: unchanged)
    uint32_t partials = 1;   // > 1: the input row is (1, partials, T, 32), summed in dim-1 order (define
                             // MHC_XING_PARTIALS; 1 = off: the program and hash it always had)
    tt::tt_metal::ComputeConfigDescriptor compute_config;
    // Fork option max_grid (deepseek_v41_flash_d_p FIX.15, agent rule 18): the op runs on the rectangle (0, 0) ..
    // (max_grid.x - 1, max_grid.y - 1) of the device's compute grid. Default (none) = the whole grid. Hashed when set.
    std::optional<tt::tt_metal::CoreCoord> max_grid = std::nullopt;
};

// the device's compute grid cut to max_grid (FIX.15)
inline tt::tt_metal::CoreCoord cut_grid(
    tt::tt_metal::CoreCoord g, const std::optional<tt::tt_metal::CoreCoord>& max_grid) {
    if (max_grid.has_value()) {
        g.x = std::max<std::size_t>(1, std::min<std::size_t>(g.x, max_grid->x));
        g.y = std::max<std::size_t>(1, std::min<std::size_t>(g.y, max_grid->y));
    }
    return g;
}

struct MhcPreXingInputs {
    Tensor input;                        // (..., T, 32) mix row, or (..., T, n(n+2)) hc
    std::optional<Tensor> streams;       // (..., T, n*C) fp32
    std::optional<Tensor> collapse_pre;  // (..., T, n..32) fp32: the pre y collapses with (given_pre)
    // P.16 (default absent = the op allocates): the caller's buffers for hc (or the packed row under pack_stats)
    // and y; each must have exactly the spec compute_output_specs gives it (checked on every call).
    std::optional<Tensor> hc_output;
    std::optional<Tensor> y_output;
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
    const ttnn::operations::bringup::mhc_pre_ttnn::MhcPreXingParams& params,
    const std::optional<ttnn::Tensor>& collapse_pre = std::nullopt,
    const std::optional<ttnn::Tensor>& hc_output = std::nullopt,
    const std::optional<ttnn::Tensor>& y_output = std::nullopt);
}  // namespace ttnn::prim::bringup
