// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/matmul/device/tile_sparse/tile_sparse_matmul_types.hpp"
#include "ttnn/operations/matmul/device/matmul_1d_type.hpp"

namespace ttnn::prim {

/**
 * @brief Program factory for tile-sparse matrix multiplication.
 *
 * Creates Tensix programs that exploit tile-level sparsity by:
 * 1. Reading only non-zero tiles based on sparsity mask
 * 2. Computing only non-zero tile products
 * 3. Accumulating results into dense output
 *
 * Uses multicore reuse with multicast for efficient distribution
 * of sparse tiles across compute cores.
 */
struct TileSparseMatmulProgramFactory {
    struct shared_variables_t {
        std::vector<tt::tt_metal::KernelHandle> kernels;
        std::vector<tt::tt_metal::CBHandle> cbs;
        CoreCoord start_core;
        std::vector<CoreCoord> cores;
        uint32_t num_cores_with_work{};
        uint32_t nnz_tiles{};                                       // Number of non-zero tiles to process
        std::shared_ptr<tt::tt_metal::Buffer> tile_indices_buffer;  // Device buffer for tile indices
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    /**
     * @brief Creates the tile-sparse matmul program.
     *
     * Sets up reader/compute/writer kernels with sparsity-aware logic:
     * - Reader: Uses tile indices to read only non-zero tiles
     * - Compute: Processes non-zero tile pairs
     * - Writer: Writes accumulated results to output
     */
    static cached_program_t create(
        const TileSparseMatmulParams& operation_attributes,
        const TileSparseMatmulInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);

    /**
     * @brief Updates runtime arguments for cached program.
     */
    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const TileSparseMatmulParams& operation_attributes,
        const TileSparseMatmulInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);
};

/**
 * @brief Mesh workload factory for multi-device tile-sparse matmul.
 */
struct TileSparseMatmulMeshWorkloadFactory {
    using shared_variables_t = TileSparseMatmulProgramFactory::shared_variables_t;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const TileSparseMatmulParams& attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const TileSparseMatmulInputs& tensor_args,
        std::vector<Tensor>& output);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const TileSparseMatmulParams& operation_attributes,
        const TileSparseMatmulInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);
};

}  // namespace ttnn::prim
