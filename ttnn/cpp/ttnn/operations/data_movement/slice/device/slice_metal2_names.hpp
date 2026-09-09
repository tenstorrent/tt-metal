// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>

// Spec names shared by slice's five program factories.
//
// Each factory builds its own ProgramSpec, so reusing a name across factories is not a collision at
// the spec level. A name only has to be unique within the spec that declares it. The reason these
// live in a header rather than in each factory's own anonymous namespace is the C++ one: this target
// is a unity build, so same-named anonymous-namespace constants in sibling factory .cpp files would
// merge into one scope and collide at link time. `inline const` at namespace scope gives every
// translation unit the same single definition instead.
//
// The dataflow buffer names are deliberately absent. They differ between factories, so each factory
// declares its own with a factory-specific C++ identifier, for the same unity-build reason.

namespace ttnn::prim::slice_metal2 {

inline const tt::tt_metal::experimental::KernelSpecName READER{"reader"};
inline const tt::tt_metal::experimental::KernelSpecName WRITER{"writer"};

inline const tt::tt_metal::experimental::TensorParamName INPUT{"input"};
inline const tt::tt_metal::experimental::TensorParamName OUTPUT{"output"};
// Only SliceTileTensorArgsProgramFactory declares these two: the device-resident tensors carrying
// the slice bounds.
inline const tt::tt_metal::experimental::TensorParamName START{"start"};
inline const tt::tt_metal::experimental::TensorParamName END{"end"};

}  // namespace ttnn::prim::slice_metal2
