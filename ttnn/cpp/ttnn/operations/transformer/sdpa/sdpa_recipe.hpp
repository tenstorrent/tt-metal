// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "sdpa_precision_policy.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include "sdpa.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::transformer::sdpa::detail {

RecipeSelection select_recipe(ttnn::transformer::SDPAPrecision precision, DataType kv_type);
// Dense/joint recipes: any tile-aligned Q chunk up to 1024 rows and any tile-aligned K chunk (ring and exp ring
// validate through recipe_geometry_rejection in sdpa_recipe_blocking.hpp; L1 fit is checked when the program is
// built).
uint32_t recipe_dense_q_tiles(const std::optional<SDPAProgramConfig>& program_config);
uint32_t recipe_dense_k_tiles(const std::optional<SDPAProgramConfig>& program_config);

// Recipe QK/PV matmul subblock width for a K chunk or head dim of `tiles` tiles: the largest of 4, 2 and 1
// dividing it (SDPA_RECIPE_QK_W / SDPA_RECIPE_PV_W). Shared by the dense, ring and exp ring recipe hosts.
uint32_t recipe_subblock_width(uint32_t tiles);

tt::tt_metal::ProgramDescriptor recipe_compute_program(
    const PrecisionPolicy& policy,
    const CoreRangeSet& grid,
    uint32_t k_chunks,
    uint32_t q_tiles = 8,
    uint32_t k_tiles = 16,
    uint32_t d_tiles = 4);

// Bytes of the Tensix kernel config ring buffer that holds every program's kernel binaries: the L1 between the
// kernel config base and the allocator's unreserved base. It depends on the device's worker_l1_size (a smaller
// worker L1 leaves a larger buffer): 70656 B at the default Blackhole worker L1, ~187 KB at 1344544.
uint64_t recipe_kernel_config_bytes(const tt::tt_metal::distributed::MeshDevice& device);

// Kernel config buffer below which the exp ring B-E recipes with a BF16 destination (COMPENSATED, LOW_PRECISION)
// build with the generic-geometry flags (SDPA_RECIPE_SIZE_OPTIMIZED + SDPA_RECIPE_GENERIC_GEOMETRY: pack and
// unpack at -Os) at every geometry. At the previously qualified geometries those builds otherwise keep pack
// (odd Q) or no TRISC (even Q) at -Os, and with the MUX writer the exp ring programs measure 71.3-76.0 KB on
// Blackhole, over the 70656 B buffer of a default-worker-L1 device (the qualified runs use H3's
// worker_l1_size=1344544, a ~187 KB buffer, and keep their flags). FP32-destination recipes (C, D) and the
// generic-geometry builds already fit 70656 B.
inline constexpr uint64_t kExpRingRecipeQualifiedKernelConfigBytes = 96 * 1024;

// True when the exp ring recipe build at (policy, geometry) must add the generic-geometry size flags for a
// device whose kernel config buffer is `kernel_config_bytes` (0: unknown, treated as large). Shared by the exp
// ring recipe program factory and the blocking chooser.
bool exp_ring_recipe_size_optimized_for_config_buffer(
    const PrecisionPolicy& policy, uint32_t q_tiles, uint32_t k_tiles, uint32_t d_tiles, uint64_t kernel_config_bytes);

PrecisionPolicy resolve_recipe_policy(
    const Tensor& q,
    const Tensor& k,
    ttnn::transformer::SDPAPrecision precision,
    bool inputs_prepared,
    std::optional<float> scale,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<SDPAProgramConfig>& program_config);

std::tuple<Tensor, Tensor> run_joint_recipe(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& joint_q,
    const Tensor& joint_k,
    const Tensor& joint_v,
    const PrecisionPolicy& policy,
    const std::optional<SDPAProgramConfig>& program_config);

// Checks an additive attn_mask for the dense recipe path (shape, dtype, layout) before any dispatch.
void validate_recipe_mask(const Tensor& q, const Tensor& k, const Tensor& mask, const PrecisionPolicy& policy);

// attn_mask: optional additive mask [1|B, 1|H, Sq, Sk] (BF16/BFP8/BFP4, tiled DRAM), already
// multiplied by 1/scale like legacy SDPA; it is L1-accumulated onto the QK scores before the row max.
Tensor run_recipe(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const PrecisionPolicy& policy,
    const std::optional<SDPAProgramConfig>& program_config,
    const std::optional<Tensor>& attn_mask = std::nullopt);

}  // namespace ttnn::operations::transformer::sdpa::detail
