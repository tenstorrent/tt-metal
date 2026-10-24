// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "tile_column_activation_device_operation.hpp"

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

namespace {
constexpr std::string_view operation_name = "activate_tile_columns";
}  // namespace

TileColumnActivationOperation::program_factory_t TileColumnActivationOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return TileColumnActivationProgramFactory{};
}

void TileColumnActivationOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    kda_factory_detail::check_tile_column_activation(attrs, in.input, in.output, operation_name);
}

TileColumnActivationOperation::spec_return_value_t TileColumnActivationOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& in) {
    return {in.output.tensor_spec()};
}

TileColumnActivationOperation::tensor_return_value_t TileColumnActivationOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& in) {
    return {in.output};
}

ttnn::device_operation::ProgramArtifacts TileColumnActivationProgramFactory::create_program_artifacts(
    const kda_factory_detail::TileColumnActivationParams& attrs,
    const TileColumnActivationInputs& in,
    std::vector<Tensor>& outputs) {
    const auto grid = in.input.mesh_tensor().device().compute_with_storage_grid_size();
    const auto cores = kda_factory_detail::idle_cores(grid, tt::tt_metal::CoreRangeSet{});
    tt::tt_metal::experimental::ProgramSpec spec{.name = std::string(operation_name)};
    tt::tt_metal::experimental::ProgramRunArgs run_args;
    kda_factory_detail::add_tile_column_activation(spec, run_args, attrs, in.input, outputs[0], cores);
    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

Tensor activate_tile_columns(
    const kda_factory_detail::TileColumnActivationParams& params, const Tensor& input, const Tensor& output) {
    return ttnn::device_operation::launch<TileColumnActivationOperation>(
        params, TileColumnActivationInputs{.input = input, .output = output})[0];
}

}  // namespace ttnn::experimental::prim
