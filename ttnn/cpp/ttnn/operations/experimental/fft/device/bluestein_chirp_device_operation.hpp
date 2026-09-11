// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>
#include <variant>

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct BluesteinChirpParams {
    uint32_t output_size = 0;
    bool input_imag_provided = false;
};

struct BluesteinChirpTensorArgs {
    Tensor input_real;
    Tensor input_imag;
    Tensor chirp_real;
    Tensor chirp_imag;
};

struct BluesteinChirpFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const BluesteinChirpParams&, const BluesteinChirpTensorArgs&, std::tuple<Tensor, Tensor>&);

    static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
        const BluesteinChirpParams&,
        const BluesteinChirpTensorArgs&,
        std::tuple<Tensor, Tensor>&,
        const std::optional<ttnn::MeshCoordinate>& = std::nullopt);
};

struct BluesteinChirpDeviceOperation {
    using operation_attributes_t = BluesteinChirpParams;
    using tensor_args_t = BluesteinChirpTensorArgs;
    using spec_return_value_t = std::tuple<tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::tuple<Tensor, Tensor>;
    using program_factory_t = std::variant<BluesteinChirpFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

// Internal Bluestein PRE/POST primitive. Multiply the first chirp_size elements
// by the cached chirp, then zero-extend or truncate to output_size. No scaling
// is introduced here: the inverse chirp already contains the normalization.
std::tuple<Tensor, Tensor> bluestein_chirp(
    const Tensor& input_real,
    const std::optional<Tensor>& input_imag,
    const Tensor& chirp_real,
    const Tensor& chirp_imag,
    uint32_t output_size);

}  // namespace ttnn::prim
