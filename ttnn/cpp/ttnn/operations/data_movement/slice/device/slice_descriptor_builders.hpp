// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/data_movement/slice/device/slice_device_operation_types.hpp"

// Standalone ProgramDescriptor builders for slice.
//
// These hold the descriptor-construction logic that used to live as static
// methods on the slice program factories. They are kept as free functions so
// that descriptor consumers — the Python fusion compiler (via the pybind), and
// mesh_partition's std::visit reuse — keep getting a ProgramDescriptor, even
// after the slice program factories migrate to Metal 2.0 (where a factory may
// not also expose create_descriptor; see operation_concepts.hpp
// all_factories_valid "exactly one concept").
//
// The factory's Metal 2.0 create_program_spec and these builders are separate
// entry points: descriptor IR for fusion, ProgramSpec for standalone dispatch.

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor build_slice_tile_descriptor(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);
tt::tt_metal::ProgramDescriptor build_slice_rm_descriptor(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);
tt::tt_metal::ProgramDescriptor build_slice_rm_stride_descriptor(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);
tt::tt_metal::ProgramDescriptor build_slice_rm_sharded_descriptor(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);
tt::tt_metal::ProgramDescriptor build_slice_tile_tensor_args_descriptor(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output);

}  // namespace ttnn::prim
