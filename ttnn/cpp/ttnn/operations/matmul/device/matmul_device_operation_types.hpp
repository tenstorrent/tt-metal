// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "tt-metalium/experimental/prefetcher_pipe.hpp"
#include "ttnn/prefetcher_pipe.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "tt-metalium/global_circular_buffer.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operation.hpp"  // for DEFAULT_OUTPUT_MEMORY_CONFIG
#include "ttnn/assigned_sub_device.hpp"

namespace ttnn::prim {

struct MatmulParams {
    std::optional<operations::matmul::MatmulProgramConfig> program_config = std::nullopt;
    std::optional<bool> bcast_batch = std::nullopt;
    tt::tt_metal::MemoryConfig output_mem_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG;
    std::optional<tt::tt_metal::DataType> output_dtype = std::nullopt;
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt;
    bool untilize_out = false;
    std::optional<tt::tt_metal::CoreCoord> user_core_coord = std::nullopt;
    std::optional<ttnn::operations::unary::UnaryWithParam> user_fused_activation = std::nullopt;
    bool user_run_batched = false;
    bool transpose_a = false;
    bool transpose_b = false;
    std::optional<tt::tt_metal::Tile> output_tile = std::nullopt;
    std::optional<tt::tt_metal::experimental::GlobalCircularBuffer> global_cb = std::nullopt;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt;
    // Alternative in1 transport to `global_cb`: PrefetcherPipes whose receivers are this matmul's workers
    // and that deliver receiver i the in1 K-blocks of output column block i, in K order. Either every
    // pipe of one create_prefetcher_pipes_for_tensor_prefetcher call (DRAM senders), or worker-sender
    // pipes fed by the caller's own producer. Empty means none; at most one of the two transports may
    // be set. Keep the pipes alive for as long as the program cache may hold a program built against
    // them: the Program binds each pipe, and cb_in1 is laid over its ring.
    ttnn::PrefetcherPipeList prefetcher_pipes;
    // Read A's K columns from this tile-aligned column of a wider interleaved A, instead of all of A's columns.
    // Set through ttnn::linear; only the 2D multicast program config (its Metal 2.0 spec factory) supports it.
    std::optional<uint32_t> in0_column_offset = std::nullopt;
    // The sub-device the matmul runs on (see ttnn::AssignedSubDevice), assigned by its prim, or by a composite before
    // it sizes the program config; sub_device_id is only the caller's request. Program configs are placed in its cores
    // (allowed_worker_cores), where the multicast and MultiCore factories anchor their grids. Unassigned when a fused
    // operation builds MatmulParams itself: its programs then go where sub_device_id (by default the first sub-device)
    // puts them.
    ttnn::AssignedSubDevice assigned_sub_device = {};
};

// Where a matmul's programs go: its assigned sub-device's cores or, unassigned, the cores matmul used before
// sub-devices were assigned.
struct MatmulPlacementFrame {
    // The rectangle an automatic program config fills; a program config's grid is placed at its start.
    tt::tt_metal::CoreRange rectangle;
    // The cores a MultiCore config splits its work over.
    tt::tt_metal::CoreRangeSet multicore_cores;
    // The cores the gather path's ring may take: the sub-device's.
    tt::tt_metal::CoreRangeSet gather_cores;
};

MatmulPlacementFrame matmul_placement_frame(
    const MatmulParams& attributes, const tt::tt_metal::distributed::MeshDevice& device);

struct MatmulInputs {
    std::vector<Tensor> input_tensors;                                // a,b, weights
    std::vector<std::optional<const Tensor>> optional_input_tensors;  // bias
    std::vector<std::optional<Tensor>> optional_output_tensors;       // output
};

}  // namespace ttnn::prim
