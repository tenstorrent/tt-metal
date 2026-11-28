// SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <utility>

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation_types.hpp"
#include "ttnn/operations/common/hash_without_rows.hpp"

namespace ttnn::experimental::prim {

struct AllGatherMinimalMatmulAsyncParams {
    std::optional<const MinimalMatmulConfig> config;
    std::optional<operations::unary::UnaryWithParam> fused_activation;
    std::optional<tt::tt_metal::MemoryConfig> output_mem_config;
    std::optional<tt::tt_metal::DataType> output_dtype;
    DeviceComputeKernelConfig compute_kernel_config;
    uint32_t num_links = 0;
    uint32_t ring_size = 0;
    ttnn::ccl::Topology topology;
    std::vector<GlobalSemaphore> semaphore;
    std::optional<uint32_t> cluster_axis;
    const std::optional<GlobalSemaphore>& barrier_semaphore;
    bool using_persistent_buffers = false;
    bool force_transpose = false;
    uint32_t num_workers_per_link = 0;
    uint32_t num_buffers_per_channel = 0;

    // Fused addcmul: ternary_a + scalar * matmul_output * ternary_b
    std::optional<float> fused_ternary_scalar;

    int32_t chunks = 1;  // Number of output tensors to split into (default 1 for backward compat)
    int32_t dim = -1;    // Dimension to split along (default -1)

    // Per-chunk output widths along `dim`, in ELEMENTS
    std::vector<uint32_t> chunk_sizes;

    // FSDP fusion: when set, the weight tensor is sharded along its K dim across
    // `fsdp_cluster_axis` with size `fsdp_ring_size`, and the op all-gathers it
    // into `persistent_weight_buffer` before/concurrently-with the matmul.
    std::optional<uint32_t> fsdp_cluster_axis;
    uint32_t fsdp_ring_size = 1;
    std::vector<GlobalSemaphore> fsdp_semaphore;  // ping-pong pair, same shape as `semaphore`
    bool using_persistent_weight_buffer = false;
    ttnn::ccl::Topology fsdp_topology;

    // Fused SwiGLU: weight is a tile-pair-interleaved [gate|up] matrix of width 2N; the op
    // emits silu(gate)*up of width N in a single matmul. Mutually exclusive with
    // fused_activation / ternary.
    bool fuse_swiglu = false;

    // Core-grid orientation: nullopt keeps the legacy `force_transpose || M > N` rule, a value pins it
    // so the program (and its cache entry) does not depend on M. Neither this nor force_transpose is in
    // attribute_values(): the program hash keys on the resolved orientation (resolve_transpose_core_grid),
    // so a pinned and an unpinned call that resolve alike share one program.
    std::optional<bool> transpose_core_grid;

    // Input blocks reach their whole row or column of the matmul grid in one NoC multicast instead of a core-to-core
    // relay. Independent of the fabric config.
    bool use_mcast = false;

    // Gather in1 instead of in0. The caller passes the whole in0, [M, K], and a stripe of in1, [K / ring_size, N], and
    // gets in0 @ gathered(in1), [M, N]. Inside the op the two tensors trade places: input_tensor, the one the op
    // gathers, holds the in1 stripe and weight_tensor holds in0.
    bool gather_in1 = false;
    // With gather_in1: the caller's stripe is in1's transpose, [N, K / ring_size].
    bool transpose_b = false;

    AllGatherMinimalMatmulAsyncParams(
        std::optional<const MinimalMatmulConfig> config,
        std::optional<ttnn::operations::unary::UnaryWithParam> fused_activation,
        std::optional<tt::tt_metal::MemoryConfig> output_mem_config,
        std::optional<tt::tt_metal::DataType> output_dtype,
        DeviceComputeKernelConfig compute_kernel_config,
        uint32_t num_links,
        uint32_t ring_size,
        ttnn::ccl::Topology topology,
        std::vector<GlobalSemaphore> semaphore,
        std::optional<uint32_t> cluster_axis,
        const std::optional<GlobalSemaphore>& barrier_semaphore,
        bool using_persistent_buffers,
        bool force_transpose,
        uint32_t num_workers_per_link,
        uint32_t num_buffers_per_channel,
        std::optional<float> fused_ternary_scalar,
        int32_t chunks,
        int32_t dim,
        std::optional<uint32_t> fsdp_cluster_axis,
        uint32_t fsdp_ring_size,
        std::vector<GlobalSemaphore> fsdp_semaphore,
        bool using_persistent_weight_buffer,
        ttnn::ccl::Topology fsdp_topology,
        bool fuse_swiglu = false,
        std::vector<uint32_t> chunk_sizes = {},
        std::optional<bool> transpose_core_grid = std::nullopt,
        bool use_mcast = false,
        bool gather_in1 = false,
        bool transpose_b = false) :
        config(config),
        fused_activation(std::move(fused_activation)),
        output_mem_config(std::move(output_mem_config)),
        output_dtype(output_dtype),
        compute_kernel_config(compute_kernel_config),
        num_links(num_links),
        ring_size(ring_size),
        topology(topology),
        semaphore(std::move(semaphore)),
        cluster_axis(cluster_axis),
        barrier_semaphore(barrier_semaphore),
        using_persistent_buffers(using_persistent_buffers),
        force_transpose(force_transpose),
        num_workers_per_link(num_workers_per_link),
        num_buffers_per_channel(num_buffers_per_channel),
        fused_ternary_scalar(fused_ternary_scalar),
        chunks(chunks),
        dim(dim),
        chunk_sizes(std::move(chunk_sizes)),
        fsdp_cluster_axis(fsdp_cluster_axis),
        fsdp_ring_size(fsdp_ring_size),
        fsdp_semaphore(std::move(fsdp_semaphore)),
        using_persistent_weight_buffer(using_persistent_weight_buffer),
        fsdp_topology(fsdp_topology),
        fuse_swiglu(fuse_swiglu),
        transpose_core_grid(transpose_core_grid),
        use_mcast(use_mcast),
        gather_in1(gather_in1),
        transpose_b(transpose_b) {}

    // Structural fields that affect program-cache key.
    static constexpr auto attribute_names = std::make_tuple(
        "num_links",
        "ring_size",
        "output_mem_config",
        "topology",
        "cluster_axis",
        "num_workers_per_link",
        "num_buffers_per_channel",
        "config",
        "chunks",
        "dim",
        "chunk_sizes",
        "fsdp_cluster_axis",
        "fsdp_ring_size",
        "using_persistent_weight_buffer",
        "fuse_swiglu",
        "use_mcast",
        "gather_in1",
        "transpose_b",
        "fused_activation",
        "output_dtype",
        // Selects DST_SYNC_MODE and the other compile-time compute settings.
        "compute_kernel_config");

    auto attribute_values() const {
        return std::forward_as_tuple(
            this->num_links,
            this->ring_size,
            this->output_mem_config,
            this->topology,
            this->cluster_axis,
            this->num_workers_per_link,
            this->num_buffers_per_channel,
            this->config,
            this->chunks,
            this->dim,
            this->chunk_sizes,
            this->fsdp_cluster_axis,
            this->fsdp_ring_size,
            this->using_persistent_weight_buffer,
            this->fuse_swiglu,
            this->use_mcast,
            this->gather_in1,
            this->transpose_b,
            this->fused_activation,
            this->output_dtype,
            this->compute_kernel_config);
    }
};

// The orientation the program is built with. M and N are the matmul's padded row count and weight width. Fabric2D
// only runs the untransposed grid, so it is not picked there automatically.
inline bool resolve_transpose_core_grid(
    const AllGatherMinimalMatmulAsyncParams& params, uint32_t M, uint32_t N, bool fabric_2d) {
    return params.transpose_core_grid.value_or(params.force_transpose || (M > N && !fabric_2d));
}
inline bool resolve_transpose_core_grid(
    const AllGatherMinimalMatmulAsyncParams& params,
    uint32_t M,
    uint32_t N,
    const tt::tt_metal::distributed::MeshDevice& mesh_device) {
    return resolve_transpose_core_grid(
        params, M, N, tt::tt_fabric::is_2d_fabric_config(tt::tt_fabric::GetFabricConfig(mesh_device)));
}

// The M, N and K per device the program runs with, read from the tensors in its two slots:
//
//                              input_tensor (gathered)      weight_tensor (local)    M      N      K per device
//   default                    in0 stripe [rows, K / n]     in1 [K, N]               rows   N      K / n
//   gather_in1                 in1 stripe [K / n, N]        in0 [rows, K]            N      rows   K / n
//   gather_in1 + transpose_b   in1 stripe [N, K / n]        in0 [rows, K]            N      rows   K / n
inline bool agmm_stripes_are_rows(const AllGatherMinimalMatmulAsyncParams& params) {
    return params.gather_in1 && !params.transpose_b;
}
inline uint32_t agmm_m(const AllGatherMinimalMatmulAsyncParams& params, const Tensor& gathered) {
    return agmm_stripes_are_rows(params) ? gathered.padded_shape()[-1] : operations::common::folded_rows(gathered);
}
inline uint32_t agmm_n(const AllGatherMinimalMatmulAsyncParams& params, const Tensor& local) {
    return params.gather_in1 ? operations::common::folded_rows(local) : local.padded_shape()[-1];
}
inline uint32_t agmm_k_per_device(const AllGatherMinimalMatmulAsyncParams& params, const Tensor& gathered) {
    return gathered.padded_shape()[agmm_stripes_are_rows(params) ? -2 : -1];
}

// Per-chunk widths (elements): explicit `chunk_sizes`, else the uniform N/chunks split.
inline std::vector<uint32_t> resolve_chunk_sizes(const AllGatherMinimalMatmulAsyncParams& params, uint32_t N) {
    if (!params.chunk_sizes.empty()) {
        return params.chunk_sizes;
    }
    const auto chunks = static_cast<uint32_t>(params.chunks);
    return std::vector<uint32_t>(chunks, N / chunks);
}

struct AllGatherMinimalMatmulAsyncInputs {
    Tensor input_tensor;
    Tensor weight_tensor;
    std::optional<Tensor> bias_tensor;
    std::optional<Tensor> persistent_output_buffer;

    // Fused addcmul: ternary_a + scalar * matmul_output * ternary_b
    std::optional<Tensor> fused_ternary_input_a;  // residual/base (broadcast like bias)
    std::optional<Tensor> fused_ternary_input_b;  // gate/multiplier (full MxN shape)

    // FSDP fusion: optional pre-allocated buffer holding the gathered weight `[K, N/tp]`.
    // When fsdp_cluster_axis is set on params, this should also be set; the op will
    // populate it with the gathered weight before the matmul reads from it.
    std::optional<Tensor> persistent_weight_buffer;
};

}  // namespace ttnn::experimental::prim
