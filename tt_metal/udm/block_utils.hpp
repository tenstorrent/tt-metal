// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/udm/types.hpp"
#include "tt_metal/udm/block_builder.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::udm {

/**
 * @brief Map a tensor to global cores based on work partitioning strategy
 *
 * This function intelligently distributes tensor work to gcores:
 * - Understands data sharding (how data is distributed across grids)
 * - Partitions work on specified dimension (which tensor dimension to split work on)
 * - Assigns gcores from appropriate grids based on data locality
 *
 * Example:
 *   Tensor (4, 16) width-sharded on 1×4 mesh → each device has (4, 4)
 *   partition_dim = 0: Split on rows, each worker gets 1 row
 *     → Assigns 4 workers per device (16 total)
 *     → Each worker reads 4 tiles (its local row portion)
 *
 * @param builder The TensorBuilder containing device and tensor information
 * @param tensor The distributed tensor to map
 * @param partition_dim The tensor dimension to partition work on (0=rows, 1=cols, etc.)
 *                      Use -1 to auto-detect based on tensor sharding
 * @return GcoresInfo Information about the global cores and their work assignments
 */
GcoresInfo map_tensor_to_gcores(const class TensorBuilder& builder, const ttnn::Tensor& tensor, int partition_dim = -1);

}  // namespace tt::tt_metal::udm
