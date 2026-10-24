// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <vector>

#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_tile_columns.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

// The tile column activation (kda_tile_columns.hpp) as a program of its own on the worker grid: what a KDA program runs
// beside its own kernels when it leaves worker cores idle, for when it leaves none.
struct TileColumnActivationInputs {
    Tensor input;
    // Written in place: its window of the activation's tile columns.
    Tensor output;
};

struct TileColumnActivationProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const kda_factory_detail::TileColumnActivationParams&, const TileColumnActivationInputs&, std::vector<Tensor>&);
};

struct TileColumnActivationOperation {
    using operation_attributes_t = kda_factory_detail::TileColumnActivationParams;
    using tensor_args_t = TileColumnActivationInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<TileColumnActivationProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

// Run the activation on the worker grid, into `output` in place; returns `output`.
Tensor activate_tile_columns(
    const kda_factory_detail::TileColumnActivationParams& params, const Tensor& input, const Tensor& output);

}  // namespace ttnn::experimental::prim
