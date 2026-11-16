// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <tt-metalium/workload/specification/dataflow_buffer_spec.hpp>
#include <tt-metalium/workload/specification/kernel_spec.hpp>
#include <tt-metalium/workload/specification/scratchpad_spec.hpp>
#include <tt-metalium/workload/specification/tensor_parameter.hpp>

// Spec names for slice's five program factories.
//
// Each factory builds its own ProgramSpec, so the name strings only need to be unique within a
// factory; the C++ constants live in one header because the five factory .cpp files share a CMake
// target, where per-file anonymous-namespace constants of the same name would collide under a unity
// build. The cache-hit helper that serves all five factories also needs the kernel and tensor names,
// which is the other reason they have a single home.

namespace ttnn::prim::slice_metal2 {

// Tensor parameters. Every factory binds `input` and `output`; only the tensor-args tile factory
// binds `start` and `end`.
inline const tt::tt_metal::TensorParamName INPUT{"input"};
inline const tt::tt_metal::TensorParamName OUTPUT{"output"};
inline const tt::tt_metal::TensorParamName START_TENSOR{"start_tensor"};
inline const tt::tt_metal::TensorParamName END_TENSOR{"end_tensor"};

// SliceRmProgramFactory
inline const tt::tt_metal::KernelSpecName RM_READER{"rm_reader"};
inline const tt::tt_metal::KernelSpecName RM_WRITER{"rm_writer"};
inline const tt::tt_metal::DFBSpecName RM_IN{"rm_in"};
inline const tt::tt_metal::ScratchpadSpecName RM_ID_PER_DIM{"rm_id_per_dim"};

// SliceRmShardedProgramFactory
inline const tt::tt_metal::KernelSpecName SHARDED_READER{"sharded_reader"};

// SliceRmStrideProgramFactory
inline const tt::tt_metal::KernelSpecName STRIDE_READER{"stride_reader"};
inline const tt::tt_metal::KernelSpecName STRIDE_WRITER{"stride_writer"};
inline const tt::tt_metal::DFBSpecName STRIDE_IN{"stride_in"};

// SliceTileProgramFactory
inline const tt::tt_metal::KernelSpecName TILE_READER{"tile_reader"};
inline const tt::tt_metal::KernelSpecName TILE_WRITER{"tile_writer"};
inline const tt::tt_metal::DFBSpecName TILE_IN{"tile_in"};
inline const tt::tt_metal::ScratchpadSpecName TILE_ID_PER_DIM{"tile_id_per_dim"};

// SliceTileTensorArgsProgramFactory
inline const tt::tt_metal::KernelSpecName TA_READER{"ta_reader"};
inline const tt::tt_metal::KernelSpecName TA_WRITER{"ta_writer"};
inline const tt::tt_metal::DFBSpecName TA_IN{"ta_in"};
inline const tt::tt_metal::ScratchpadSpecName TA_TENSOR{"ta_tensor"};
inline const tt::tt_metal::ScratchpadSpecName TA_ID_PER_DIM{"ta_id_per_dim"};

}  // namespace ttnn::prim::slice_metal2
