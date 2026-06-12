// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_tile.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_descriptor_builders.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

// The descriptor logic now lives in build_slice_tile_descriptor (a standalone
// function shared with the pybind / fusion / mesh_partition consumers). This
// thin delegate is the seam: when the factory migrates to Metal 2.0 it gains
// create_program_spec and this create_descriptor is dropped, while the consumers
// keep calling build_slice_tile_descriptor directly.
tt::tt_metal::ProgramDescriptor SliceTileProgramFactory::create_descriptor(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    return build_slice_tile_descriptor(args, tensor_args, output);
}

}  // namespace ttnn::prim
