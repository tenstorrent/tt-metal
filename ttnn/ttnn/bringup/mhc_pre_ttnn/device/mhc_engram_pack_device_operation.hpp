// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>
#include <vector>

#include <tt_stl/reflection.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "mhc_pre_xing_device_operation.hpp"  // cut_grid

namespace ttnn::operations::bringup::mhc_pre_ttnn {

// ttnn.bringup.mhc_pre_xing_pack(None, streams, n=n, engram_keys=K, engram_weight=W) (deepseek_v41_flash_d_p P.14):
// the engram gate's per-stream partial sums of one chip in ONE pass over the streams and the keys (exact fp32 SFPU):
//     row[t, i]       = sum_c h_i[t, c] w_i[c] k_i[t, c]      (i = 0..n-1; h_i = streams[..., i C:(i+1) C])
//     row[t, n + i]   = sum_c h_i[t, c]^2
//     row[t, 2n + i]  = sum_c k_i[t, c]^2                      (k_i = engram_keys[..., i C:(i+1) C])
// (..., T, 32) fp32, the other columns zero. W (..., 32, n C) fp32: the weight row replicated over the tile's 32 rows.
struct MhcEngramPackParams {
    uint32_t n = 4;
    // Fork option max_grid (deepseek_v41_flash_d_p FIX.15, agent rule 18): the op runs on the rectangle (0, 0) ..
    // (max_grid.x - 1, max_grid.y - 1) of the device's compute grid. Default (none) = the whole grid. Hashed when set.
    std::optional<tt::tt_metal::CoreCoord> max_grid = std::nullopt;
    // Fork option engram_gate_eps (deepseek_v41_flash_d_p P.24, TINY_KERNELS): when set, the streams' C columns are
    // the whole hidden (no TP split) and the root also writes the engram gates into columns 3n .. 4n-1 of the row:
    //     g_i = sigmoid(copysign(sqrt(max(|d_i|, 1e-6)), d_i)),
    //     d_i = row[i] * (rsqrt(row[n + i] / C + eps) * rsqrt(row[2n + i] / C + eps)) * C^-0.5
    // (fp32 SFPU, the op chain's steps and roundings). Default (none) = the program it always was. Hashed when set.
    std::optional<float> gate_eps = std::nullopt;
};

struct MhcEngramPackInputs {
    Tensor streams;  // (..., T, n*C) fp32
    Tensor keys;     // (..., T, >= n*C) fp32, the keys in columns 0 .. n*C - 1
    Tensor weight;   // (..., 32, n*C) fp32, leading dims 1
};

// Units: (stream i, column block g of Bc tiles, token tile-row r), stream/block-major (u = (i G + g) RT + r), split
// contiguously over the grid; the root of row r (owner of unit ((r mod n G) RT + r)) sums the row's n G partials.
struct MhcEngramPackPlan {
    uint32_t n = 0, col_tiles = 0, key_tiles = 0, row_tiles = 0, cols_per_block = 0, groups = 0, units = 0;
};
MhcEngramPackPlan mhc_engram_pack_plan(const MhcEngramPackParams& p, const MhcEngramPackInputs& in);

struct MhcEngramPackProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const MhcEngramPackParams& operation_attributes,
        const MhcEngramPackInputs& tensor_args,
        std::vector<Tensor>& outputs);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const MhcEngramPackParams& operation_attributes,
        const MhcEngramPackInputs& tensor_args,
        std::vector<Tensor>& outputs,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

struct MhcEngramPackDeviceOperation {
    using operation_attributes_t = MhcEngramPackParams;
    using tensor_args_t = MhcEngramPackInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;  // [row, scratch]
    using program_factory_t = std::variant<MhcEngramPackProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::bringup::mhc_pre_ttnn

namespace ttnn::prim::bringup {
std::vector<ttnn::Tensor> mhc_engram_pack(
    const ttnn::Tensor& streams,
    const ttnn::Tensor& keys,
    const ttnn::Tensor& weight,
    uint32_t n,
    std::optional<tt::tt_metal::CoreCoord> max_grid = std::nullopt,
    std::optional<float> gate_eps = std::nullopt);
}  // namespace ttnn::prim::bringup
