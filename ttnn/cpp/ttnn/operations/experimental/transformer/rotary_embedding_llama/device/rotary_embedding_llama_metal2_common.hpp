// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/scratchpad_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>

// Shared Metal 2.0 named-resource vocabulary for the three rotary_embedding_llama program
// factories. Hoisted into a header (with inline linkage) because ttnn enables unity builds:
// the factory .cpp files concatenate into one TU, and per-.cpp anonymous-namespace constants
// of the same name would collide. The three factories also share kernel sources (writer and
// compute/rotary_embedding_llama.cpp across factories 1 & 2), so their DFB / tensor accessor
// names must be defined once and reused.

namespace ttnn::experimental::prim::rope_metal2 {

using tt::tt_metal::experimental::DFBSpecName;
using tt::tt_metal::experimental::KernelSpecName;
using tt::tt_metal::experimental::ScratchpadSpecName;
using tt::tt_metal::experimental::TensorParamName;

// Kernels
inline const KernelSpecName READER{"reader"};
inline const KernelSpecName WRITER{"writer"};
inline const KernelSpecName COMPUTE{"compute"};

// Dataflow buffers (one per legacy CB index)
inline const DFBSpecName INPUT_DFB{"input"};                    // c_0
inline const DFBSpecName COS_DFB{"cos"};                        // c_1
inline const DFBSpecName SIN_DFB{"sin"};                        // c_2
inline const DFBSpecName TRANS_MAT_DFB{"trans_mat"};            // c_3
inline const DFBSpecName ROTATED_INTERM_DFB{"rotated_interm"};  // c_24
inline const DFBSpecName COS_INTERM_DFB{"cos_interm"};          // c_25
inline const DFBSpecName SIN_INTERM_DFB{"sin_interm"};          // c_26
inline const DFBSpecName OUT_DFB{"out"};                        // c_16

// Scratchpads. The writer's zero-fill staging region (legacy c_27) was ported as a self-looped
// DFB (the writer bound both PRODUCER and CONSUMER); Gen2 rejects DM self-loop DFBs, so it is a
// Scratchpad (per ai/post_port/semantic/dm_self_loop_dfbs.md) — a plain node-local L1 region with
// no FIFO semantics, which is all the kernel ever used it as.
inline const ScratchpadSpecName ZERO_SCRATCH{"zero"};  // c_27

// Tensor parameters
inline const TensorParamName INPUT_PARAM{"input"};
inline const TensorParamName COS_PARAM{"cos"};
inline const TensorParamName SIN_PARAM{"sin"};
inline const TensorParamName TRANS_MAT_PARAM{"trans_mat"};
inline const TensorParamName OUTPUT_PARAM{"output"};

// Kernel source paths. Defined once here (uniquely named, inline) rather than in each factory's
// anonymous namespace: under unity builds the factory .cpp files can share a translation unit, where
// duplicate anon-namespace names collide. The writer and prefill compute source are shared by
// factories 1 & 2.
inline const std::filesystem::path kReaderInterleavedSource{
    "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/kernels/dataflow/"
    "reader_rotary_embedding_llama_interleaved_start_id.cpp"};
inline const std::filesystem::path kReaderPrefillShardedSource{
    "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/kernels/dataflow/"
    "reader_rotary_embedding_llama_prefill_sharded.cpp"};
inline const std::filesystem::path kWriterSource{
    "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/kernels/dataflow/"
    "writer_rotary_embedding_llama_interleaved_start_id.cpp"};
inline const std::filesystem::path kComputeSource{
    "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/kernels/compute/"
    "rotary_embedding_llama.cpp"};
inline const std::filesystem::path kComputeShardedSource{
    "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/kernels/compute/"
    "rotary_embedding_llama_sharded.cpp"};

}  // namespace ttnn::experimental::prim::rope_metal2
