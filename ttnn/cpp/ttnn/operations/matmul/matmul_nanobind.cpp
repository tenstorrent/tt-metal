// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_nanobind.hpp"

#include <cstddef>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include <tt-metalium/core_coord.hpp>
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config.hpp"
#include "ttnn-nanobind/decorators.hpp"
#include "ttnn-nanobind/json_class.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::matmul {

struct MatmulProgramConfigPlaceholder {};

using ttnn::operations::unary::UnaryWithParam;

void py_module(nb::module_& mod) {
    auto matmul_program_config = nb::class_<MatmulProgramConfigPlaceholder>(mod, "MatmulProgramConfig", R"doc(
        Variant defining matmul program config
    )doc");

    auto matmul_multi_core_reuse_program_config =
        tt_serializable_class<MatmulMultiCoreReuseProgramConfig>(mod, "MatmulMultiCoreReuseProgramConfig", R"doc(
        Configuration class for multi-core reusable matmul operations.

        This program config is used for basic multi-core matmul operations that can reuse
        intermediate results across cores for better performance.
    )doc");

    matmul_multi_core_reuse_program_config.def(
        nb::init<CoreCoord, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t>(),
        nb::kw_only(),
        nb::arg("compute_with_storage_grid_size"),
        nb::arg("in0_block_w").noconvert(),
        nb::arg("out_subblock_h").noconvert(),
        nb::arg("out_subblock_w").noconvert(),
        nb::arg("per_core_M").noconvert(),
        nb::arg("per_core_N").noconvert());
    matmul_multi_core_reuse_program_config.def_rw(
        "compute_with_storage_grid_size", &MatmulMultiCoreReuseProgramConfig::compute_with_storage_grid_size, R"doc(
        Grid size for compute cores with storage capability.

        Specifies the 2D grid of cores (x, y) that will be used for computation and have
        access to storage. This determines how the computation is distributed across cores.
    )doc");
    matmul_multi_core_reuse_program_config.def_rw("in0_block_w", &MatmulMultiCoreReuseProgramConfig::in0_block_w, R"doc(
        Block width for both input tensors along the K dimension (shared inner dimension).

        This parameter determines the granularity of data blocks by specifying how many tiles wide
        each block is along the K dimension. It affects the size of data chunks processed together
        and impacts memory usage and compute efficiency for both tensors. Must be a divisor of the
        K dimension. Suggested to be a multiple of 32 for tile alignment.
    )doc");
    matmul_multi_core_reuse_program_config.def_rw(
        "out_subblock_h", &MatmulMultiCoreReuseProgramConfig::out_subblock_h, R"doc(
        Height of output subblocks in tiles.

        Controls the granularity of computation within each output block along the M dimension.
        Smaller values can reduce memory usage but may decrease efficiency. Must divide evenly
        into the output block height.
    )doc");
    matmul_multi_core_reuse_program_config.def_rw(
        "out_subblock_w", &MatmulMultiCoreReuseProgramConfig::out_subblock_w, R"doc(
        Width of output subblocks in tiles.

        Controls the granularity of computation within each output block along the N dimension.
        Smaller values can reduce memory usage but may decrease efficiency. Must divide evenly
        into the output block width.
    )doc");
    matmul_multi_core_reuse_program_config.def_rw("per_core_M", &MatmulMultiCoreReuseProgramConfig::per_core_M, R"doc(
        Number of output tiles each core processes along the M dimension.

        Determines how the M dimension of the output is distributed across cores.
        Larger values mean fewer cores are used but each core does more work.
        Must be chosen such that (total_M / per_core_M) cores are available.
    )doc");
    matmul_multi_core_reuse_program_config.def_rw("per_core_N", &MatmulMultiCoreReuseProgramConfig::per_core_N, R"doc(
        Number of output tiles each core processes along the N dimension.

        Determines how the N dimension of the output is distributed across cores.
        Larger values mean fewer cores are used but each core does more work.
        Must be chosen such that (total_N / per_core_N) cores are available.
    )doc");

    auto matmul_multi_core_reuse_multicast_program_config =
        tt_serializable_class<MatmulMultiCoreReuseMultiCastProgramConfig>(
            mod, "MatmulMultiCoreReuseMultiCastProgramConfig", R"doc(
        The "2D" matmul program config is used for block sharded tensors, and general interleaved tensors.
    )doc");

    matmul_multi_core_reuse_multicast_program_config.def(
        "__init__",
        [](MatmulMultiCoreReuseMultiCastProgramConfig* t,
           CoreCoord compute_with_storage_grid_size,
           std::size_t in0_block_w,
           std::size_t out_subblock_h,
           std::size_t out_subblock_w,
           std::optional<std::size_t> out_block_h,
           std::optional<std::size_t> out_block_w,
           std::size_t per_core_M,
           std::size_t per_core_N,
           bool transpose_mcast,
           std::optional<UnaryWithParam> fused_activation,
           bool fuse_batch) {
            // Set out_block_h and out_block_w to defaults if they are not provided
            std::size_t actual_out_block_h = out_block_h.value_or(per_core_M);
            std::size_t actual_out_block_w = out_block_w.value_or(per_core_N);

            new (t) MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size,
                in0_block_w,
                out_subblock_h,
                out_subblock_w,
                actual_out_block_h,
                actual_out_block_w,
                per_core_M,
                per_core_N,
                transpose_mcast,
                std::move(fused_activation),
                fuse_batch);
        },
        nb::kw_only(),
        nb::arg("compute_with_storage_grid_size"),
        nb::arg("in0_block_w").noconvert(),
        nb::arg("out_subblock_h").noconvert(),
        nb::arg("out_subblock_w").noconvert(),
        nb::arg("out_block_h") = nb::none(),
        nb::arg("out_block_w") = nb::none(),
        nb::arg("per_core_M").noconvert(),
        nb::arg("per_core_N").noconvert(),
        nb::arg("transpose_mcast").noconvert(),
        nb::arg("fused_activation") = nb::none(),
        nb::arg("fuse_batch").noconvert() = true);

    matmul_multi_core_reuse_multicast_program_config.def_rw(
        "compute_with_storage_grid_size",
        &MatmulMultiCoreReuseMultiCastProgramConfig::compute_with_storage_grid_size,
        R"doc(
        Grid size for compute cores with storage capability.

        Specifies the 2D grid of cores (x, y) that will be used for computation and have
        access to storage. This determines how the computation is distributed across cores
        and affects multicast communication patterns.
    )doc");

    matmul_multi_core_reuse_multicast_program_config.def_rw(
        "in0_block_w", &MatmulMultiCoreReuseMultiCastProgramConfig::in0_block_w, R"doc(
        Block width for both input tensors along the K dimension (shared inner dimension).

        Determines the data granularity by specifying how many tiles wide each block is along
        the K dimension for both input_tensor_a and input_tensor_b in multicast operations.
        Must be a divisor of the K dimension. Smaller blocks can improve load balancing but
        may increase communication overhead in multicast scenarios.
    )doc");

    matmul_multi_core_reuse_multicast_program_config.def_rw(
        "out_subblock_h", &MatmulMultiCoreReuseMultiCastProgramConfig::out_subblock_h, R"doc(
        Height of output subblocks in tiles.

        Controls the granularity of computation within each output block along the M dimension.
        Must divide evenly into out_block_h. Affects memory usage and compute scheduling
        in the multicast implementation.
    )doc");

    matmul_multi_core_reuse_multicast_program_config.def_rw(
        "out_subblock_w", &MatmulMultiCoreReuseMultiCastProgramConfig::out_subblock_w, R"doc(
        Width of output subblocks in tiles.

        Controls the granularity of computation within each output block along the N dimension.
        Must divide evenly into out_block_w. Affects memory usage and compute scheduling
        in the multicast implementation.
    )doc");

    matmul_multi_core_reuse_multicast_program_config.def_rw(
        "out_block_h", &MatmulMultiCoreReuseMultiCastProgramConfig::out_block_h, R"doc(
        Height of output blocks in tiles.

        Specifies the block size for output tensor along the M dimension. If not provided,
        defaults to per_core_M. Must be divisible by out_subblock_h and should be chosen
        to optimize multicast efficiency and memory usage.
    )doc");

    matmul_multi_core_reuse_multicast_program_config.def_rw(
        "out_block_w", &MatmulMultiCoreReuseMultiCastProgramConfig::out_block_w, R"doc(
        Width of output blocks in tiles.

        Specifies the block size for output tensor along the N dimension. If not provided,
        defaults to per_core_N. Must be divisible by out_subblock_w and should be chosen
        to optimize multicast efficiency and memory usage.
    )doc");

    matmul_multi_core_reuse_multicast_program_config.def_rw(
        "per_core_M", &MatmulMultiCoreReuseMultiCastProgramConfig::per_core_M, R"doc(
        Number of output tiles each core processes along the M dimension.

        Determines how the M dimension is distributed across cores in the multicast setup.
        Used as the default value for out_block_h if not explicitly specified.
    )doc");

    matmul_multi_core_reuse_multicast_program_config.def_rw(
        "per_core_N", &MatmulMultiCoreReuseMultiCastProgramConfig::per_core_N, R"doc(
        Number of output tiles each core processes along the N dimension.

        Determines how the N dimension is distributed across cores in the multicast setup.
        Used as the default value for out_block_w if not explicitly specified.
    )doc");

    matmul_multi_core_reuse_multicast_program_config.def_rw(
        "transpose_mcast", &MatmulMultiCoreReuseMultiCastProgramConfig::transpose_mcast, R"doc(
        Whether to transpose the multicast communication pattern.

        When true, the multicast direction is transposed, which can be beneficial for
        certain tensor shapes and grid configurations. This affects how data is broadcast
        across cores and can impact performance depending on the access patterns.
    )doc");

    matmul_multi_core_reuse_multicast_program_config.def_rw(
        "fused_activation", &MatmulMultiCoreReuseMultiCastProgramConfig::fused_activation, R"doc(
        Optional fused activation function to apply to the output.

        If provided, the specified activation function (e.g., ReLU, GELU) is applied
        directly during the matmul computation, avoiding the need for a separate activation
        operation and improving performance.
    )doc");

    matmul_multi_core_reuse_multicast_program_config.def_rw(
        "fuse_batch", &MatmulMultiCoreReuseMultiCastProgramConfig::fuse_batch, R"doc(
        Whether to fuse batch dimensions into the matrix dimensions.

        When true, batch dimensions are fused with the M dimension, allowing for more
        efficient processing of batched matrix multiplications. This can improve performance
        for operations with large batch sizes. Defaults to true.

        Note: the batch dimensions need to all be 1 for the second input tensor when fuse_batch is true.
    )doc");

    auto matmul_multi_core_reuse_multicast_1d_program_config =
        tt_serializable_class<MatmulMultiCoreReuseMultiCast1DProgramConfig>(
            mod, "MatmulMultiCoreReuseMultiCast1DProgramConfig", R"doc(
        Configuration class for 1D multicast matmul operations with advanced features.

        This program config is for use with width and height sharded tensors, or very narrow interleaved tensors.
    )doc");

    matmul_multi_core_reuse_multicast_1d_program_config
        .def(
            "__init__",
            [](MatmulMultiCoreReuseMultiCast1DProgramConfig* t,
               CoreCoord compute_with_storage_grid_size,
               std::size_t in0_block_w,
               std::size_t out_subblock_h,
               std::size_t out_subblock_w,
               std::optional<std::size_t> out_block_h,
               std::optional<std::size_t> out_block_w,
               std::size_t per_core_M,
               std::size_t per_core_N,
               bool fuse_batch,
               std::optional<UnaryWithParam> fused_activation,
               bool mcast_in0,
               bool gather_in0,
               CoreRangeSet hop_cores,
               std::size_t num_global_cb_receivers,
               bool untilize_out) {
                // Set out_block_h and out_block_w to defaults if they are not provided
                std::size_t actual_out_block_h = out_block_h.value_or(per_core_M);
                std::size_t actual_out_block_w = out_block_w.value_or(per_core_N);

                new (t) MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size,
                    in0_block_w,
                    out_subblock_h,
                    out_subblock_w,
                    actual_out_block_h,
                    actual_out_block_w,
                    per_core_M,
                    per_core_N,
                    fuse_batch,
                    std::move(fused_activation),
                    mcast_in0,
                    gather_in0,
                    std::move(hop_cores),
                    num_global_cb_receivers,
                    untilize_out);
            },
            nb::kw_only(),
            nb::arg("compute_with_storage_grid_size"),
            nb::arg("in0_block_w").noconvert(),
            nb::arg("out_subblock_h").noconvert(),
            nb::arg("out_subblock_w").noconvert(),
            nb::arg("out_block_h") = nb::none(),
            nb::arg("out_block_w") = nb::none(),
            nb::arg("per_core_M").noconvert(),
            nb::arg("per_core_N").noconvert(),
            nb::arg("fuse_batch").noconvert(),
            nb::arg("fused_activation") = nb::none(),
            nb::arg("mcast_in0").noconvert(),
            nb::arg("gather_in0").noconvert() = false,
            nb::arg("hop_cores").noconvert() = nb::cast(CoreRangeSet()),
            nb::arg("num_global_cb_receivers").noconvert() = 1,
            nb::arg("untilize_out").noconvert() = false)
        .def_rw(
            "compute_with_storage_grid_size",
            &MatmulMultiCoreReuseMultiCast1DProgramConfig::compute_with_storage_grid_size,
            R"doc(
            Grid size for compute cores with storage capability.

            Defines the 2D grid of cores that will be used for computation. In 1D multicast,
            this grid is used to determine the communication pattern for broadcasting data
            along one dimension while distributing computation.
        )doc")
        .def_rw("in0_block_w", &MatmulMultiCoreReuseMultiCast1DProgramConfig::in0_block_w, R"doc(
            Block width for both input tensors along the K dimension (shared inner dimension).

            Determines the data granularity by specifying how many tiles wide each block is
            along the inner dimension for both input_tensor_a and input_tensor_b. This parameter
            impacts 1D multicast performance as it affects the size of data chunks that
            are broadcast across cores and memory access patterns for both tensors.
        )doc")
        .def_rw("out_subblock_h", &MatmulMultiCoreReuseMultiCast1DProgramConfig::out_subblock_h, R"doc(
            Height of output subblocks in tiles.

            Controls computation granularity within output blocks along the M dimension.
            In 1D multicast, this affects how computation is scheduled and memory usage
            patterns across the participating cores.
        )doc")
        .def_rw("out_subblock_w", &MatmulMultiCoreReuseMultiCast1DProgramConfig::out_subblock_w, R"doc(
            Width of output subblocks in tiles.

            Controls computation granularity within output blocks along the N dimension.
            This parameter affects the efficiency of the 1D multicast communication pattern
            and compute scheduling.
        )doc")
        .def_rw("out_block_h", &MatmulMultiCoreReuseMultiCast1DProgramConfig::out_block_h, R"doc(
            Height of output blocks in tiles.

            Defines the output block size along the M dimension. If not specified, defaults
            to per_core_M. This parameter is important for optimizing the 1D multicast
            pattern and memory access efficiency.
        )doc")
        .def_rw("out_block_w", &MatmulMultiCoreReuseMultiCast1DProgramConfig::out_block_w, R"doc(
            Width of output blocks in tiles.

            Defines the output block size along the N dimension. If not specified, defaults
            to per_core_N. This affects the efficiency of data distribution in the 1D
            multicast implementation.
        )doc")
        .def_rw("per_core_M", &MatmulMultiCoreReuseMultiCast1DProgramConfig::per_core_M, R"doc(
            Number of output tiles each core processes along the M dimension.

            Determines the workload distribution along the M dimension in the 1D multicast
            pattern. This affects both load balancing and communication efficiency.
        )doc")
        .def_rw("per_core_N", &MatmulMultiCoreReuseMultiCast1DProgramConfig::per_core_N, R"doc(
            Number of output tiles each core processes along the N dimension.

            Determines the workload distribution along the N dimension in the 1D multicast
            pattern. This parameter is crucial for achieving optimal performance in
            1D multicast scenarios.
        )doc")
        .def_rw("fuse_batch", &MatmulMultiCoreReuseMultiCast1DProgramConfig::fuse_batch, R"doc(
            Whether to fuse batch dimensions into matrix dimensions.

            When true, batch dimensions are incorporated into the matrix computation,
            allowing for more efficient processing of batched operations in the 1D
            multicast implementation.

            Note: the batch dimensions need to all be 1 for the second input tensor when fuse_batch is true.
        )doc")
        .def_rw("fused_activation", &MatmulMultiCoreReuseMultiCast1DProgramConfig::fused_activation, R"doc(
            Optional fused activation function to apply during computation.

            If specified, the activation function is applied directly during the matmul
            operation, eliminating the need for a separate activation pass and improving
            overall performance in 1D multicast scenarios.
        )doc")
        .def_rw("mcast_in0", &MatmulMultiCoreReuseMultiCast1DProgramConfig::mcast_in0, R"doc(
            Whether to multicast the first input tensor (input_tensor_a).

            When true, input_tensor_a is broadcast across cores using the 1D multicast
            pattern, which can significantly reduce memory bandwidth requirements for
            certain matrix shapes and improve performance.
        )doc")
        .def_rw("gather_in0", &MatmulMultiCoreReuseMultiCast1DProgramConfig::gather_in0, R"doc(
            Defaults to false.
            Used by ops that call matmul internally. Should not be specified or left as the default value for all other uses.
        )doc")
        .def_rw("hop_cores", &MatmulMultiCoreReuseMultiCast1DProgramConfig::hop_cores, R"doc(
            Defaults to empty set.
            Used by ops that call matmul internally. Should not be specified or left as the default value for all other uses.
        )doc")
        .def_rw(
            "num_global_cb_receivers", &MatmulMultiCoreReuseMultiCast1DProgramConfig::num_global_cb_receivers, R"doc(
            Defaults to 1.
            Used by ops that call matmul internally. Should not be specified or left as the default value for all other uses.
        )doc")
        .def_rw("untilize_out", &MatmulMultiCoreReuseMultiCast1DProgramConfig::untilize_out, R"doc(
            Whether to untilize the output tensor.

            When true, the output is converted from tiled layout to row-major layout during
            the operation. This can be useful when the subsequent operation expects row-major
            data and can eliminate a separate untilization pass. Defaults to false.
        )doc");

    auto matmul_multi_core_reuse_multicast_dram_sharded_program_config =
        tt_serializable_class<MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>(
            mod, "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig", R"doc(
        This program config is a specialized config for very narrow tensors stored in DRAM.
    )doc");

    matmul_multi_core_reuse_multicast_dram_sharded_program_config
        .def(
            nb::init<std::size_t, std::size_t, std::size_t, std::optional<UnaryWithParam>>(),
            nb::kw_only(),
            nb::arg("in0_block_w").noconvert(),
            nb::arg("per_core_M").noconvert(),
            nb::arg("per_core_N").noconvert(),
            nb::arg("fused_activation") = nb::none())
        .def_rw("in0_block_w", &MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig::in0_block_w, R"doc(
            Block width for both input tensors along the K dimension (shared inner dimension).

            Determines the data granularity by specifying how many tiles wide each block is
            along the inner dimension for both input_tensor_a and input_tensor_b in DRAM-sharded
            operations. This parameter must be chosen to align with the DRAM sharding
            strategy and optimize memory bandwidth utilization for both tensors.
        )doc")
        .def_rw("per_core_M", &MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig::per_core_M, R"doc(
            Number of output tiles each core processes along the M dimension.

            Determines how the M dimension is distributed across cores in DRAM-sharded
            scenarios. This must align with the DRAM sharding pattern to ensure optimal
            performance and avoid memory access conflicts.
        )doc")
        .def_rw("per_core_N", &MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig::per_core_N, R"doc(
            Number of output tiles each core processes along the N dimension.

            Determines how the N dimension is distributed across cores in DRAM-sharded
            scenarios. This parameter affects the multicast efficiency and must be
            compatible with the DRAM sharding configuration.
        )doc")
        .def_rw("fused_activation", &MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig::fused_activation, R"doc(
            Optional fused activation function to apply during computation.

            If specified, the activation function is applied directly during the DRAM-sharded
            matmul operation. This can provide significant performance benefits by avoiding
            additional memory round-trips in DRAM-based operations.
        )doc");

    bind_registered_operation(
        mod,
        ::ttnn::matmul,
        R"doc(
        Returns the matrix product of two tensors.

        The input tensors need to be tiled and at least 1-dimensional.

        - If both input tensors are 1-dimensional, then the operation is a dot product.
        - If first input tensor is 1-dimensional and the other input tensor is at least 2-dimensional,
          the batched vector-matrix multiplication is performed.
        - If the first input tensor is at least 2-dimensional and the second input tensor is 1-dimensional,
          the batched matrix-vector multiplication is performed.
        - If both input tensors are at least 2-dimensional, then a batched matrix multiply is performed.

        The following are the allowed possibilities for batch dimensions.
        Examples below show concrete operations and tensor sizes.

        - If all batch dimensions are of size 1, then there is no batched operation.

        - If both inputs have batch dimensions that are not all of size 1, then the
          batch dimensions of both inputs should be the same. If the dimensions are
          not the same then, although there may be combinations that may work, in most
          cases various errors will be reported.

        - If the first input has batch dimensions that are not all of size 1, and the
          second input has no batch dimensions or has batch dimensions all of size 1,
          then the second input is broadcasted to align appropriately with the first
          input.

        - Matrix multiplication will not work if the first input has batch
          dimensions that are all of size 1 and the second input has batch dimensions
          that are not all of size 1.

        - Note: In general, the number of dimensions between the two inputs should
          match. There may be cases where they don't. In that case, if the inputs
          are not valid based on the above criteria, the error messages may
          be unexpected and refer to non-obvious issues.

        - Note: There are various combinations of dimensions possible. The behaviour
          is the same as PyTorch, except for two exceptions.
          These exceptions are for the following scenarios related to batch
          dimensions:

              - The two batch dimensions are swapped. E.g. the first input has (`j` x `1`)
                and the second input has (`1` x `j`)
                or the first input has (`1` x `j`) and the second input has
                (`j` x `1`)
              - When a batch dimension is implicitly extended, the two patch dimensions are swapped.
                E.g.  (`j` x `1`) and (`j`) which is treated as
                (`j` x `1`) and (`1` x `j`)

        - In order to leverage sharded matmul implementations we can shard both `input_tensor_a` and `input_tensor_b`. The sharding strategy used will be according
          to the sharding strategy on the respective tensor. A sharded 1D matmul can be either HEIGHT or WIDTH sharded, 2D matmuls can be BLOCK sharded.

          - For 1D sharded matmul variants (width- or height-sharded inputs), if a sharded :attr:`memory_config` is
            provided for the output, its memory layout and buffer type must match those of :attr:`input_tensor_a`.

          Note: the broadcasting logic only looks at the batch dimensions when determining if the inputs
          are broadcastable, and not the matrix dimensions. For example, if :attr:`input_tensor_a` is a
          (`j` x `1` x `n_size` x `m_size`) tensor and :attr:`input_tensor_b` is a (`k_size` x `m_size` x `p`)
          tensor, these inputs are valid for broadcasting even though the final two dimensions (i.e. the
          matrix dimensions) are different. The operation will return a (`j` x `k_size` x `n_size` x `p`) tensor.

        - Note: there are various additional constraints related to specific program
          configs chosen. Please look at the error messages carefully and fix
          problems appropriately.
        - Note: If optional output tensor is specified, then dtype and memory config need to be checked as follows:
          - if they are default then they should be set based on optional output tensor
          - if the are not default then they should be compared and if there is a difference an error is reported

        Args:
            input_tensor_a (ttnn.Tensor): the first tensor to be multiplied. Needs to be on the device.
            input_tensor_b (ttnn.Tensor): the second tensor to be multiplied. Needs to be on the device.

        Keyword Args:
            transpose_a (bool, optional): Whether to transpose input_tensor_a. Defaults to `False`.
            transpose_b (bool, optional): Whether to transpose input_tensor_b. Defaults to `False`.
            memory_config(ttnn.MemoryConfig, optional): the memory configuration of the output tensor. Defaults to `None`, which will result in using ttnn.DRAM_MEMORY_CONFIG.
            dtype (ttnn.DataType): the data type of the output tensor. Defaults to `None`.
            program_config (ttnn.MatmulProgramConfig): the program configuration for the matmul operation. Defaults to `None`.
            activation (str or ttnn.UnaryWithParam, optional): the activation function to be applied. When using sharded tensors, the :attr:`fused_activation` parameter of the :attr:`program_config` should be used instead. Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig): the compute kernel configuration for the matmul operation. Defaults to `None`.
            core_grid (ttnn.CoreGrid): the grid on which to distribute the sharded tensor on (writes to the cores L1s). Defaults to `None`.
            output_tile (List of [int], optional): Specifies the output tile configuration. Defaults to `None`.
            optional_output_tensor (ttnn.Tensor, optional): User provided on-device output tensor where the result of matmul is to be written. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            The input tensors support the following data types and layouts:

            .. list-table:: input_tensor_a
                :header-rows: 1

                * - dtype
                  - layout
                * - BFLOAT8_B, BFLOAT4_B, BFLOAT16, FLOAT32
                  - TILE

            .. list-table:: input_tensor_b
                :header-rows: 1

                * - dtype
                  - layout
                * - BFLOAT8_B, BFLOAT4_B, BFLOAT16, FLOAT32
                  - TILE

        Memory Support:
            The supported memory configurations for the two input tensors are program config dependent, as described below:

            .. list-table:: Supported Memory Configurations
                :header-rows: 1

                * - Config
                  - Input A
                  - Input B
                * - MatmulMultiCoreReuseProgramConfig
                  - Interleaved (L1/DRAM), Height Sharded (L1), or Block Sharded (L1)
                  - Interleaved (L1/DRAM), Height Sharded (L1), or Block Sharded (L1)
                * - MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig
                  - Width Sharded (L1)
                  - Width Sharded (DRAM)
                * - MatmulMultiCoreReuseMultiCastProgramConfig
                  - Interleaved (L1/DRAM), Block Sharded (L1)
                  - Interleaved (L1/DRAM)
                * - MatmulMultiCoreReuseMultiCastProgramConfig (only for row major orientation without transpose multicast)
                  - Interleaved (L1/DRAM), Height Sharded (L1)
                  - Interleaved (L1/DRAM), Width Sharded (L1)
                * - MatmulMultiCoreReuseMultiCast1DProgramConfig (mcast_in0=False)
                  - Interleaved (L1/DRAM), Width Sharded (L1)
                  - Interleaved (L1/DRAM), Width Sharded (L1)
                * - MatmulMultiCoreReuseMultiCast1DProgramConfig (mcast_in0=True)
                  - Interleaved (L1/DRAM), Height Sharded (L1)
                  - Interleaved (L1/DRAM)

            When sharded output tensors are provided, they should match :attr:`input_tensor_a`'s buffer type and memory layout.
        )doc",
        ttnn::nanobind_overload_t{
            [](decltype(::ttnn::matmul)& self,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const bool transpose_a,
               const bool transpose_b,
               const std::optional<const ttnn::MemoryConfig>& memory_config,
               const std::optional<const DataType> dtype,
               const std::optional<const MatmulProgramConfig>& program_config,
               const std::optional<const ::ttnn::Activation>& activation,
               const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
               const std::optional<const ttnn::CoreGrid> core_grid,
               const std::optional<const tt::tt_metal::Tile>& output_tile,
               std::optional<Tensor>& optional_output_tensor,
               const std::optional<const GlobalCircularBuffer>& global_cb,
               const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) -> ttnn::Tensor {
                return self(
                    input_tensor_a,
                    input_tensor_b,
                    transpose_a,
                    transpose_b,
                    memory_config,
                    dtype,
                    program_config,
                    activation,
                    compute_kernel_config,
                    core_grid,
                    output_tile,
                    optional_output_tensor,
                    global_cb,
                    sub_device_id);
            },
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg("transpose_a") = false,
            nb::arg("transpose_b") = false,
            nb::arg("memory_config") = nb::none(),
            nb::arg("dtype") = nb::none(),
            nb::arg("program_config") = nb::none(),
            nb::arg("activation") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none(),
            nb::arg("core_grid") = nb::none(),
            nb::arg("output_tile") = nb::none(),
            nb::arg("optional_output_tensor") = nb::none(),
            nb::arg("global_cb") = nb::none(),
            nb::arg("sub_device_id") = nb::none(),
        });

    bind_registered_operation(
        mod,
        ::ttnn::linear,
        R"doc(
        Returns the linear transformation of the inputs.

        The limitations and behaviours are the same as for matmul.

        Note:
            The tensors support the following data types and layouts:

            .. list-table:: input_tensor_a
                :header-rows: 1

                * - dtype
                  - layout
                * - BFLOAT8_B, BFLOAT4_B, BFLOAT16, FLOAT32
                  - TILE

            .. list-table:: input_tensor_b
                :header-rows: 1

                * - dtype
                  - layout
                * - BFLOAT8_B, BFLOAT4_B, BFLOAT16, FLOAT32
                  - TILE

            .. list-table:: bias
                :header-rows: 1

                * - dtype
                  - layout
                * - BFLOAT8_B, BFLOAT4_B, BFLOAT16, FLOAT32
                  - TILE

        Args:
            input_tensor_a (ttnn.Tensor): the first tensor to be multiplied. Needs to be on the device.
            input_tensor_b (ttnn.Tensor): the second tensor to be multiplied. Needs to be on the device.

        Keyword Args:
            bias (ttnn.Tensor, optional): the bias tensor to be added. If specified, needs to be on the device. Defaults to `None`.
            transpose_a (bool, optional): Whether to transpose input_tensor_a. Defaults to `False`.
            transpose_b (bool, optional): Whether to transpose input_tensor_b. Defaults to `False`.
            memory_config (ttnn.MemoryConfig, optional): the memory configuration of the output tensor. Defaults to `None`, which will result in using `ttnn.DRAM_MEMORY_CONFIG`.
            dtype (ttnn.DataType, optional): the data type of the output tensor. Defaults to `None`.
            program_config (MatmulProgramConfig, optional): the program configuration for the matmul operation. Defaults to `None`.
            activation (str or ttnn.UnaryWithParam, optional): the activation function to be applied. Defaults to `None`. When using sharded tensors, the :attr:`fused_activation` parameter of the :attr:`program_config` should be used instead.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): the compute kernel configuration for the matmul operation. Defaults to `None`.
            core_grid (ttnn.CoreGrid, optional): the grid on which to distribute the sharded tensor on (writes to the cores L1s). Defaults to `None`.
            output_tile (List of [int], optional): Specifies the output tile configuration. Defaults to `None`.
            optional_output_tensor (ttnn.Tensor, optional): User provided on-device output tensor where the result of linear is to be written. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.
        )doc",
        ttnn::nanobind_overload_t{
            [](decltype(::ttnn::linear)& self,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const std::optional<const ttnn::Tensor>& bias,
               const bool transpose_a,
               const bool transpose_b,
               const std::optional<const ttnn::MemoryConfig>& memory_config,
               const std::optional<const DataType> dtype,
               const std::optional<const MatmulProgramConfig>& program_config,
               const std::optional<const ::ttnn::Activation>& activation,
               const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
               const std::optional<const ttnn::CoreGrid> core_grid,
               const std::optional<const tt::tt_metal::Tile>& output_tile,
               std::optional<Tensor>& optional_output_tensor,
               const std::optional<const GlobalCircularBuffer>& global_cb,
               const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) -> ttnn::Tensor {
                return self(
                    input_tensor_a,
                    input_tensor_b,
                    bias,
                    transpose_a,
                    transpose_b,
                    memory_config,
                    dtype,
                    program_config,
                    activation,
                    compute_kernel_config,
                    core_grid,
                    output_tile,
                    optional_output_tensor,
                    global_cb,
                    sub_device_id);
            },
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg("bias") = nb::none(),
            nb::arg("transpose_a") = false,
            nb::arg("transpose_b") = false,
            nb::arg("memory_config") = nb::none(),
            nb::arg("dtype") = nb::none(),
            nb::arg("program_config") = nb::none(),
            nb::arg("activation") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none(),
            nb::arg("core_grid") = nb::none(),
            nb::arg("output_tile") = nb::none(),
            nb::arg("optional_output_tensor") = nb::none(),
            nb::arg("global_cb") = nb::none(),
            nb::arg("sub_device_id") = nb::none(),
        });

    bind_registered_operation(
        mod,
        ::ttnn::matmul_batched_weights,
        R"doc(
        DEPRECATED: This is for experimental internal use and is not supported.

        Performs matrix multiplication for a single input tensor a with multiple tensors b, and returns a vector of output tensors.

        Args:
            input_tensor_a (ttnn.Tensor): the first tensor to be multiplied. Needs to be on the device.
            input_tensors_b (List of ttnn.Tensor): the tensors to be multiplied. Needs to be on the device.

        Note:
            The tensors support the following data types and layouts:

            .. list-table:: input_tensor_a
                :header-rows: 1

                * - dtype
                  - layout
                * - BFLOAT8_B, BFLOAT4_B, BFLOAT16, FLOAT32
                  - TILE

            .. list-table:: input_tensors_b
                :header-rows: 1

                * - dtype
                  - layout
                * - BFLOAT8_B, BFLOAT4_B, BFLOAT16, FLOAT32
                  - TILE

        Keyword Args:
            bias (ttnn.Tensor, optional): the bias tensor to be added. If specified, needs to be on the device. Defaults to `None`.
            transpose_a (bool, optional): Whether to transpose input_tensor_a. Defaults to `False`.
            transpose_b (bool, optional): Whether to transpose input_tensor_b. Defaults to `False`.
            memory_config (ttnn.MemoryConfig, optional): the memory configuration of the output tensor. Defaults to `None`, which will result in using `ttnn.DRAM_MEMORY_CONFIG`.
            dtype (ttnn.DataType, optional): the data type of the output tensor. Defaults to `None`.
            program_config (MatmulProgramConfig, optional): the program configuration for the matmul operation. Defaults to `None`.
            activation (str or ttnn.UnaryWithParam, optional): the activation function to be applied. Defaults to `None`. When using sharded tensors, the :attr:`fused_activation` parameter of the :attr:`program_config` should be used instead.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): the compute kernel configuration for the matmul operation. Defaults to `None`.
            core_grid (ttnn.CoreGrid, optional): the grid on which to distribute the sharded tensor on (writes to the cores L1s). Defaults to `None`.
            output_tile (List of [int], optional): Specifies the output tile configuration. Defaults to `None`.
            optional_output_tensor (ttnn.Tensor, optional): User provided on-device output tensor where the result of linear is to be written. Defaults to `None`.
            global_cb (ttnn.GlobalCircularBuffer, optional): the global circular buffer to be used for the matmul operation. Defaults to `None`.
            sub_device_id (ttnn.SubDeviceId, optional): the sub device id to be used for the matmul operation. Defaults to `None`.

        Returns:
            List of ttnn.Tensor: the output tensors.
        )doc",
        ttnn::nanobind_overload_t{
            [](decltype(::ttnn::matmul_batched_weights)& self,
               const ttnn::Tensor& input_tensor_a,
               const std::vector<ttnn::Tensor>& input_tensors_b,
               const bool transpose_a,
               const bool transpose_b,
               const std::optional<const ttnn::MemoryConfig>& memory_config,
               const std::optional<const DataType> dtype,
               const std::optional<const MatmulProgramConfig>& program_config,
               const std::optional<const ::ttnn::Activation>& activation,
               const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
               const std::optional<const ttnn::CoreGrid> core_grid,
               const std::optional<const tt::tt_metal::Tile>& output_tile,
               std::optional<Tensor>& optional_output_tensor,
               const std::optional<const GlobalCircularBuffer>& global_cb,
               const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) -> std::vector<ttnn::Tensor> {
                return self(
                    input_tensor_a,
                    input_tensors_b,
                    transpose_a,
                    transpose_b,
                    memory_config,
                    dtype,
                    program_config,
                    activation,
                    compute_kernel_config,
                    core_grid,
                    output_tile,
                    optional_output_tensor,
                    global_cb,
                    sub_device_id);
            },
            nb::arg("input_tensor_a"),
            nb::arg("input_tensors_b"),
            nb::kw_only(),
            nb::arg("transpose_a") = false,
            nb::arg("transpose_b") = false,
            nb::arg("memory_config") = nb::none(),
            nb::arg("dtype") = nb::none(),
            nb::arg("program_config") = nb::none(),
            nb::arg("activation") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none(),
            nb::arg("core_grid") = nb::none(),
            nb::arg("output_tile") = nb::none(),
            nb::arg("optional_output_tensor") = nb::none(),
            nb::arg("global_cb") = nb::none(),
            nb::arg("sub_device_id") = nb::none(),
        });

    bind_registered_operation(
        mod,
        ::ttnn::addmm,
        R"doc(
        Returns a matrix products of tensors mat1_tensor and mat2_tensor. Tensor input_tensor is added to the final result.

        - If mat1_tensor has shape (n, m) and mat2_tensor has shape (m, p), input_tensor needs to be of shape (n, p) and
          result will also be (n, p).

        - If optional_output_tensor is provided, it needs to be of shape (n, p) and result will be stored there; all
          previous content will be overwritten, reference to this object will also be returned.

        - Arguments alpha and beta are scaling factors, result calculation look like this:

            out = beta * input_tensor + alpha * (mat1_tensor @ mat2_tensor)

        - If beta is 0, then content of input_tensor is ignored.

        - Arguments beta and alpha should be real numbers;

        Note:
            The tensors support the following data types and layouts:

            .. list-table:: input_tensor
                :header-rows: 1

                * - dtype
                  - layout
                * - BFLOAT8_B, BFLOAT4_B, BFLOAT16, FLOAT32
                  - TILE

            .. list-table:: mat1_tensor
                :header-rows: 1

                * - dtype
                  - layout
                * - BFLOAT8_B, BFLOAT4_B, BFLOAT16, FLOAT32
                  - TILE

            .. list-table:: mat2_tensor
                :header-rows: 1

                * - dtype
                  - layout
                * - BFLOAT8_B, BFLOAT4_B, BFLOAT16, FLOAT32
                  - TILE

        Args:
            input_tensor (ttnn.Tensor): tensor to be added to result of matrix multiplication of mat1_tensor and mat2_tensor
            mat1_tensor (ttnn.Tensor): the first tensor to be matrix multiplied
            mat2_tensor (ttnn.Tensor): the second tensor to be matrix multiplied

        Keyword Args:
            alpha (float): multiplier for mat1_tensor @ mat2_tensor
            beta (float): multiplier for input_tensor
            memory_config(ttnn.MemoryConfig, optional): the memory configuration of the output tensor. Defaults to `None`, which will result in using ttnn.DRAM_MEMORY_CONFIG.
            dtype (ttnn.DataType): the data type of the output tensor. Supported types: `ttnn.bfloat16`, `ttnn.float32`, `ttnn.bfloat8_b`  Defaults to `None` which means it will default to the highest precision of `input_tensor`, `mat1_tensor` or `mat2_tensor`.
            program_config (ttnn.MatmulProgramConfig): the program configuration for the matmul operation. Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig): the compute kernel configuration for the matmul operation. Defaults to `None`.
            core_grid (ttnn.CoreGrid): the grid on which to distribute the sharded tensor on (writes to the cores L1s). Defaults to `None`.
            output_tile (List of [int], optional): Specifies the output tile configuration. Defaults to `None`.
            optional_output_tensor (ttnn.Tensor, optional): User-provided on-device output tensor where the result of matmul is to be written. Defaults to `None`.
            global_cb (ttnn.GlobalCircularBuffer): TBD
            sub_device_id (ttnn.SubDeviceId): TBD

        Returns:
            ttnn.Tensor: output tensor of shape (n, p)

        )doc",
        ttnn::nanobind_overload_t{
            [](decltype(::ttnn::addmm)& self,
               const Tensor& input_tensor,
               const Tensor& mat1_tensor,
               const Tensor& mat2_tensor,
               const float alpha,
               const float beta,
               const std::optional<const MemoryConfig>& memory_config,
               const std::optional<const DataType> dtype,
               const std::optional<const MatmulProgramConfig>& program_config,
               const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
               const std::optional<const CoreGrid> core_grid,
               const std::optional<const tt::tt_metal::Tile>& output_tile,
               const std::optional<Tensor>& optional_output_tensor) -> ttnn::Tensor {
                return self(
                    input_tensor,
                    mat1_tensor,
                    mat2_tensor,
                    alpha,
                    beta,
                    memory_config,
                    dtype,
                    program_config,
                    compute_kernel_config,
                    core_grid,
                    output_tile,
                    optional_output_tensor);
            },
            nb::arg("input_tensor"),
            nb::arg("mat1_tensor"),
            nb::arg("mat2_tensor"),
            nb::kw_only(),
            nb::arg("alpha") = 1.0,
            nb::arg("beta") = 1.0,
            nb::arg("memory_config") = nb::none(),
            nb::arg("dtype") = nb::none(),
            nb::arg("program_config") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none(),
            nb::arg("core_grid") = nb::none(),
            nb::arg("output_tile") = nb::none(),
            nb::arg("optional_output_tensor") = nb::none()});

    bind_registered_operation(
        mod,
        ::ttnn::sparse_matmul,
        R"doc(
        Returns the matrix product of two tensors. Based on `is_input_a_sparse`, `is_input_b_sparse` and the sparsity tensor, some parts of the output computation is skipped.

        The two input tensors must be be tiled and each have a rank of 4.
        The sparsity tensor must be a rank 4 tensor in row major layout.

        Based on the input tensor shapes and `is_input_a_sparse` and `is_input_b_sparse` values, the output tensor shape is computed. See the supported modes table below.

        Args:
            input_tensor_a (ttnn.Tensor): the first tensor to be multiplied. Needs to be on the device.
            input_tensor_b (ttnn.Tensor): the second tensor to be multiplied. Needs to be on the device.
        Keyword Args:
            sparsity (ttnn.Tensor): the sparsity tensor containing the mask values. Needs to be on the device. The data type must be bfloat16.
            program_config (ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig): the program configuration for the matmul operation. Only this config type is supported. ``mcast_in0`` must be set to True.
            nnz (int, optional): the number of non-zero values in the sparsity tensor. If not provided, it will be inferred from the sparsity tensor at runtime.
            is_input_a_sparse (bool, optional): boolean indicating whether `input_tensor_a` is sparse. Defaults to `False`. Together with `is_input_b_sparse`, it determines how the sparsity tensor is interpreted. See the supported modes table below.
            is_input_b_sparse (bool, optional): boolean indicating whether `input_tensor_b` is sparse. Defaults to `True`. Together with `is_input_a_sparse`, it determines how the sparsity tensor is interpreted. See the supported modes table below.
            memory_config (ttnn.MemoryConfig, optional): the memory configuration of the output tensor. Defaults to `None`, which will result in using ttnn.DRAM_MEMORY_CONFIG.
            dtype (ttnn.DataType, optional): the data type of the output tensor. Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): the compute kernel configuration for the matmul operation. Defaults to `None`.
            core_grid (ttnn.CoreGrid, optional): the grid on which to distribute the sharded tensor on (writes to the cores L1s). Defaults to `None`.
            output_tile (List of [int], optional): Specifies the output tile configuration. Defaults to `None`.
            optional_output_tensor (ttnn.Tensor, optional): User provided on-device output tensor where the result of matmul is to be written. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor with sparse results.

        Supported Modes
            .. list-table::
                :header-rows: 1

                * - is_input_a_sparse
                  - is_input_b_sparse
                  - input_tensor_a shape
                  - input_tensor_b shape
                  - sparsity shape
                  - nnz
                  - output shape
                * - True
                  - True
                  - [1, E, M, K]
                  - [1, E, K, N]
                  - [1, 1, 1, E]
                  - None or 0 ≤ nnz ≤ E
                  - [1, E, M, N]
                * - False
                  - True
                  - [A, B, M, K]
                  - [1, E, K, N]
                  - [A, B, 1, E]
                  - None or 0 ≤ nnz ≤ A * B * E
                  - [A, B, 1, E, M, N]
                * - True
                  - False
                  - [A, E, M, K]
                  - [1, E, K, N]
                  - [1, 1, A, E]
                  - None or 0 ≤ nnz ≤ A * E
                  - [A, E, M, N]
                * - False
                  - False
                  - Invalid
                  - --
                  - --
                  - --
                  - --

        Note:
            The input tensors support the following data types and layouts:

            .. list-table:: input_tensor_a
                :header-rows: 1

                * - dtype
                  - layout
                * - BFLOAT4_B, BFLOAT8_B, BFLOAT16, FLOAT32
                  - TILE

            .. list-table:: input_tensor_b
                :header-rows: 1

                * - dtype
                  - layout
                * - BFLOAT4_B, BFLOAT8_B, BFLOAT16, FLOAT32
                  - TILE

            .. list-table:: sparsity
                :header-rows: 1

                * - dtype
                  - layout
                * - BFLOAT16
                  - ROW_MAJOR

        Memory Support:
            The supported memory configurations for the two input tensors are program config dependent, as described below:

            .. list-table:: Supported Memory Configurations
                :header-rows: 1

                * - Config
                  - Input A
                  - Input B
                * - ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig with (mcast_in0=True)
                  - Interleaved (L1/DRAM)
                  - Interleaved (L1/DRAM)
        )doc",
        ttnn::nanobind_overload_t{
            [](decltype(::ttnn::sparse_matmul)& self,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const ttnn::Tensor& sparsity,
               const MatmulProgramConfig& program_config,
               const std::optional<uint32_t> nnz,
               const bool is_input_a_sparse,
               const bool is_input_b_sparse,
               const std::optional<const ttnn::MemoryConfig>& memory_config,
               const std::optional<const DataType> dtype,
               const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
               const std::optional<const ttnn::CoreGrid> core_grid,
               const std::optional<const tt::tt_metal::Tile>& output_tile,
               std::optional<Tensor>& optional_output_tensor,
               const std::optional<const GlobalCircularBuffer>& global_cb,
               const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) -> ttnn::Tensor {
                return self(
                    input_tensor_a,
                    input_tensor_b,
                    sparsity,
                    nnz,
                    is_input_a_sparse,
                    is_input_b_sparse,
                    memory_config,
                    dtype,
                    program_config,
                    compute_kernel_config,
                    core_grid,
                    output_tile,
                    optional_output_tensor,
                    global_cb,
                    sub_device_id);
            },
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"),
            nb::kw_only(),
            nb::arg("sparsity"),
            nb::arg("program_config"),
            nb::arg("nnz") = nb::none(),
            nb::arg("is_input_a_sparse") = false,
            nb::arg("is_input_b_sparse") = true,
            nb::arg("memory_config") = nb::none(),
            nb::arg("dtype") = nb::none(),
            nb::arg("compute_kernel_config") = nb::none(),
            nb::arg("core_grid") = nb::none(),
            nb::arg("output_tile") = nb::none(),
            nb::arg("optional_output_tensor") = nb::none(),
            nb::arg("global_cb") = nb::none(),
            nb::arg("sub_device_id") = nb::none()});
}

}  // namespace ttnn::operations::matmul
