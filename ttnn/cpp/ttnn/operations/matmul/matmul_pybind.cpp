// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/matmul_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <utility>

#include "ttnn-pybind/decorators.hpp"
#include <tt-metalium/core_coord.hpp>
#include "ttnn-pybind/json_class.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::matmul {

using ttnn::operations::unary::UnaryWithParam;

void py_module(py::module& module) {
    auto matmul_program_config = tt_serializable_class<MatmulProgramConfig>(module, "MatmulProgramConfig", R"doc(
        Class defining matmul program config
    )doc");

    auto matmul_multi_core_reuse_program_config =
        tt_serializable_class<MatmulMultiCoreReuseProgramConfig>(module, "MatmulMultiCoreReuseProgramConfig", R"doc(
        Class defining matmul multi core reuse program config
    )doc");

    matmul_multi_core_reuse_program_config
        .def(
            py::init<CoreCoord, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t>(),
            py::kw_only(),
            py::arg("compute_with_storage_grid_size"),
            py::arg("in0_block_w").noconvert(),
            py::arg("out_subblock_h").noconvert(),
            py::arg("out_subblock_w").noconvert(),
            py::arg("per_core_M").noconvert(),
            py::arg("per_core_N").noconvert())
        .def_readwrite(
            "compute_with_storage_grid_size", &MatmulMultiCoreReuseProgramConfig::compute_with_storage_grid_size)
        .def_readwrite("in0_block_w", &MatmulMultiCoreReuseProgramConfig::in0_block_w)
        .def_readwrite("out_subblock_h", &MatmulMultiCoreReuseProgramConfig::out_subblock_h)
        .def_readwrite("out_subblock_w", &MatmulMultiCoreReuseProgramConfig::out_subblock_w)
        .def_readwrite("per_core_M", &MatmulMultiCoreReuseProgramConfig::per_core_M)
        .def_readwrite("per_core_N", &MatmulMultiCoreReuseProgramConfig::per_core_N);

    auto matmul_multi_core_reuse_multicast_program_config =
        tt_serializable_class<MatmulMultiCoreReuseMultiCastProgramConfig>(
            module, "MatmulMultiCoreReuseMultiCastProgramConfig", R"doc(
        Class defining matmul multi core reuse multi cast program config
    )doc");

    matmul_multi_core_reuse_multicast_program_config
        .def(
            py::init([](CoreCoord compute_with_storage_grid_size,
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

                return MatmulMultiCoreReuseMultiCastProgramConfig(
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
            }),
            py::kw_only(),
            py::arg("compute_with_storage_grid_size"),
            py::arg("in0_block_w").noconvert(),
            py::arg("out_subblock_h").noconvert(),
            py::arg("out_subblock_w").noconvert(),
            py::arg("out_block_h") = py::none(),
            py::arg("out_block_w") = py::none(),
            py::arg("per_core_M").noconvert(),
            py::arg("per_core_N").noconvert(),
            py::arg("transpose_mcast").noconvert(),
            py::arg("fused_activation"),
            py::arg("fuse_batch").noconvert() = true)
        .def_readwrite(
            "compute_with_storage_grid_size",
            &MatmulMultiCoreReuseMultiCastProgramConfig::compute_with_storage_grid_size)
        .def_readwrite("in0_block_w", &MatmulMultiCoreReuseMultiCastProgramConfig::in0_block_w)
        .def_readwrite("out_subblock_h", &MatmulMultiCoreReuseMultiCastProgramConfig::out_subblock_h)
        .def_readwrite("out_subblock_w", &MatmulMultiCoreReuseMultiCastProgramConfig::out_subblock_w)
        .def_readwrite("out_block_h", &MatmulMultiCoreReuseMultiCastProgramConfig::out_block_h)
        .def_readwrite("out_block_w", &MatmulMultiCoreReuseMultiCastProgramConfig::out_block_w)
        .def_readwrite("per_core_M", &MatmulMultiCoreReuseMultiCastProgramConfig::per_core_M)
        .def_readwrite("per_core_N", &MatmulMultiCoreReuseMultiCastProgramConfig::per_core_N)
        .def_readwrite("transpose_mcast", &MatmulMultiCoreReuseMultiCastProgramConfig::transpose_mcast)
        .def_readwrite("fused_activation", &MatmulMultiCoreReuseMultiCastProgramConfig::fused_activation)
        .def_readwrite("fuse_batch", &MatmulMultiCoreReuseMultiCastProgramConfig::fuse_batch);

    auto matmul_multi_core_reuse_multicast_1d_program_config =
        tt_serializable_class<MatmulMultiCoreReuseMultiCast1DProgramConfig>(
            module, "MatmulMultiCoreReuseMultiCast1DProgramConfig", R"doc(
        Class defining matmul multi core reuse multi cast 1D program config
    )doc");

    matmul_multi_core_reuse_multicast_1d_program_config
        .def(
            py::init([](CoreCoord compute_with_storage_grid_size,
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

                return MatmulMultiCoreReuseMultiCast1DProgramConfig(
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
            }),
            py::kw_only(),
            py::arg("compute_with_storage_grid_size"),
            py::arg("in0_block_w").noconvert(),
            py::arg("out_subblock_h").noconvert(),
            py::arg("out_subblock_w").noconvert(),
            py::arg("out_block_h") = py::none(),
            py::arg("out_block_w") = py::none(),
            py::arg("per_core_M").noconvert(),
            py::arg("per_core_N").noconvert(),
            py::arg("fuse_batch").noconvert(),
            py::arg("fused_activation"),
            py::arg("mcast_in0").noconvert(),
            py::arg("gather_in0").noconvert() = false,
            py::arg("hop_cores").noconvert() = CoreRangeSet(),
            py::arg("num_global_cb_receivers").noconvert() = 1,
            py::arg("untilize_out").noconvert() = false)
        .def_readwrite(
            "compute_with_storage_grid_size",
            &MatmulMultiCoreReuseMultiCast1DProgramConfig::compute_with_storage_grid_size)
        .def_readwrite("in0_block_w", &MatmulMultiCoreReuseMultiCast1DProgramConfig::in0_block_w)
        .def_readwrite("out_subblock_h", &MatmulMultiCoreReuseMultiCast1DProgramConfig::out_subblock_h)
        .def_readwrite("out_subblock_w", &MatmulMultiCoreReuseMultiCast1DProgramConfig::out_subblock_w)
        .def_readwrite("out_block_h", &MatmulMultiCoreReuseMultiCast1DProgramConfig::out_block_h)
        .def_readwrite("out_block_w", &MatmulMultiCoreReuseMultiCast1DProgramConfig::out_block_w)
        .def_readwrite("per_core_M", &MatmulMultiCoreReuseMultiCast1DProgramConfig::per_core_M)
        .def_readwrite("per_core_N", &MatmulMultiCoreReuseMultiCast1DProgramConfig::per_core_N)
        .def_readwrite("fuse_batch", &MatmulMultiCoreReuseMultiCast1DProgramConfig::fuse_batch)
        .def_readwrite("fused_activation", &MatmulMultiCoreReuseMultiCast1DProgramConfig::fused_activation)
        .def_readwrite("mcast_in0", &MatmulMultiCoreReuseMultiCast1DProgramConfig::mcast_in0)
        .def_readwrite("gather_in0", &MatmulMultiCoreReuseMultiCast1DProgramConfig::gather_in0)
        .def_readwrite("hop_cores", &MatmulMultiCoreReuseMultiCast1DProgramConfig::hop_cores)
        .def_readwrite(
            "num_global_cb_receivers", &MatmulMultiCoreReuseMultiCast1DProgramConfig::num_global_cb_receivers)
        .def_readwrite("untilize_out", &MatmulMultiCoreReuseMultiCast1DProgramConfig::untilize_out);

    auto matmul_multi_core_reuse_multicast_dram_sharded_program_config =
        tt_serializable_class<MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>(
            module, "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig", R"doc(
        Class defining matmul multi core reuse multi cast DRAM sharded program config
    )doc");

    matmul_multi_core_reuse_multicast_dram_sharded_program_config
        .def(
            py::init<std::size_t, std::size_t, std::size_t, std::optional<UnaryWithParam>>(),
            py::kw_only(),
            py::arg("in0_block_w").noconvert(),
            py::arg("per_core_M").noconvert(),
            py::arg("per_core_N").noconvert(),
            py::arg("fused_activation"))
        .def_readwrite("in0_block_w", &MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig::in0_block_w)
        .def_readwrite("per_core_M", &MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig::per_core_M)
        .def_readwrite("per_core_N", &MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig::per_core_N)
        .def_readwrite("fused_activation", &MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig::fused_activation);

    bind_registered_operation(
        module,
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
            activation (str, optional): the activation function to be applied. Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig): the compute kernel configuration for the matmul operation. Defaults to `None`.
            core_grid (ttnn.CoreGrid): the grid on which to distribute the sharded tensor on (writes to the cores L1s). Defaults to `None`.
            output_tile (List of [int], optional): Specifies the output tile configuration. Defaults to `None`.
            optional_output_tensor (ttnn.Tensor, optional): User provided on-device output tensor where the result of matmul is to be written. Defaults to `None`.


        Returns:
            ttnn.Tensor: the output tensor.

        Example:
            >>> # matrix x matrix - no batch dimensions
            >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.randn((64, 32), dtype=torch.bfloat16)), device)
            >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.randn((32, 64), dtype=torch.bfloat16)), device)
            >>> output = ttnn.matmul(tensor1, tensor2)
            >>> print(output.shape)
            [64, 64]
            >>> # extended matrix x extended matrix - all batch dimensions of size 1
            >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.randn((1, 1, 64, 32), dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT), device=device)
            >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.randn((1, 1, 32, 64), dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT), device=device)
            >>> output = ttnn.matmul(tensor1, tensor2)
            >>> print(output.shape)
            [1, 1, 64, 64]
            >>> # extended matrix x extended matrix - all batch dimensions of size 1
            >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.randn((1, 1, 64, 32), dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT), device=device)
            >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.randn((1, 32, 64), dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT), device=device)
            >>> output = ttnn.matmul(tensor1, tensor2)
            >>> print(output.shape)
            [1, 1, 64, 64]
            >>> # batched matrix x broadcasted matrix - first input has batch dimensions not of size 1
            >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.randn((10, 64, 32), dtype=torch.bfloat16)), device)
            >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.randn((32, 64), dtype=torch.bfloat16)), device)
            >>> output = ttnn.matmul(tensor1, tensor2)
            >>> print(output.shape)
            [10, 64, 64]
            >>> # batched matrix x batched matrix - both inputs have batch dimensions
            >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.randn((10, 64, 32), dtype=torch.bfloat16)), device)
            >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.randn((10, 32, 128), dtype=torch.bfloat16)), device)
            >>> output = tensor1 @ tensor2 # alternative to ttnn.matmul(tensor1, tensor2)
            >>> print(output.shape)
            [10, 64, 128]
            >>> # batched matrix x broadcasted extended matrix - first input has batch dimensions not of size 1
            >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.randn((10, 64, 32), dtype=torch.bfloat16)), device)
            >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.randn((1, 1, 32, 128), dtype=torch.bfloat16)), device)
            >>> output = tensor1 @ tensor2
            >>> print(output.shape)
            [1, 10, 64, 128]
        )doc",
        ttnn::pybind_overload_t{
            [](decltype(::ttnn::matmul)& self,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const bool transpose_a,
               const bool transpose_b,
               const std::optional<const ttnn::MemoryConfig>& memory_config,
               const std::optional<const DataType> dtype,
               const std::optional<const MatmulProgramConfig>& program_config,
               const std::optional<const std::string>& activation,
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
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("transpose_a") = false,
            py::arg("transpose_b") = false,
            py::arg("memory_config") = std::nullopt,
            py::arg("dtype") = std::nullopt,
            py::arg("program_config") = std::nullopt,
            py::arg("activation") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt,
            py::arg("core_grid") = std::nullopt,
            py::arg("output_tile") = std::nullopt,
            py::arg("optional_output_tensor") = std::nullopt,
            py::arg("global_cb") = std::nullopt,
            py::arg("sub_device_id") = std::nullopt,
        });

    bind_registered_operation(
        module,
        ::ttnn::linear,
        R"doc(
        Returns the linear transformation of the inputs.

        The limitations and behaviours are the same as for matmul.

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
            activation (str, optional): the activation function to be applied. Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): the compute kernel configuration for the matmul operation. Defaults to `None`.
            core_grid (ttnn.CoreGrid, optional): the grid on which to distribute the sharded tensor on (writes to the cores L1s). Defaults to `None`.
            output_tile (List of [int], optional): Specifies the output tile configuration. Defaults to `None`.
            optional_output_tensor (ttnn.Tensor, optional): User provided on-device output tensor where the result of linear is to be written. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Example:
            >>> # batched matrix x broadcasted matrix
            >>> activations = ttnn.to_device(ttnn.from_torch(torch.randn((10, 64, 32), dtype=torch.bfloat16)), device)
            >>> weight = ttnn.to_device(ttnn.from_torch(torch.randn((32, 128), dtype=torch.bfloat16)), device)
            >>> bias = ttnn.to_device(ttnn.from_torch(torch.randn((128,), dtype=torch.bfloat16)), device)
            >>> output = ttnn.linear(activations, weight, bias=bias)
            >>> print(output.shape)
            [10, 64, 128]
        )doc",
        ttnn::pybind_overload_t{
            [](decltype(::ttnn::linear)& self,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const std::optional<const ttnn::Tensor>& bias,
               const bool transpose_a,
               const bool transpose_b,
               const std::optional<const ttnn::MemoryConfig>& memory_config,
               const std::optional<const DataType> dtype,
               const std::optional<const MatmulProgramConfig>& program_config,
               const std::optional<const std::string>& activation,
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
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("bias") = std::nullopt,
            py::arg("transpose_a") = false,
            py::arg("transpose_b") = false,
            py::arg("memory_config") = std::nullopt,
            py::arg("dtype") = std::nullopt,
            py::arg("program_config") = std::nullopt,
            py::arg("activation") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt,
            py::arg("core_grid") = std::nullopt,
            py::arg("output_tile") = std::nullopt,
            py::arg("optional_output_tensor") = std::nullopt,
            py::arg("global_cb") = std::nullopt,
            py::arg("sub_device_id") = std::nullopt,
        });

    bind_registered_operation(
        module,
        ::ttnn::matmul_batched_weights,
        R"doc(
        performs matrix multiplication for a single input tensor a with multiple tensors b, return a vector of output tensors.

        Args:
            input_tensor_a (ttnn.Tensor): the first tensor to be multiplied. Needs to be on the device.
            input_tensors_b (ttnn.Tensor): the second tensor vector to be multiplied. Needs to be on the device.

        Keyword Args:
            bias (ttnn.Tensor, optional): the bias tensor to be added. If specified, needs to be on the device. Defaults to `None`.
            transpose_a (bool, optional): Whether to transpose input_tensor_a. Defaults to `False`.
            transpose_b (bool, optional): Whether to transpose input_tensor_b. Defaults to `False`.
            memory_config (ttnn.MemoryConfig, optional): the memory configuration of the output tensor. Defaults to `None`, which will result in using `ttnn.DRAM_MEMORY_CONFIG`.
            dtype (ttnn.DataType, optional): the data type of the output tensor. Defaults to `None`.
            program_config (MatmulProgramConfig, optional): the program configuration for the matmul operation. Defaults to `None`.
            activation (str, optional): the activation function to be applied. Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): the compute kernel configuration for the matmul operation. Defaults to `None`.
            core_grid (ttnn.CoreGrid, optional): the grid on which to distribute the sharded tensor on (writes to the cores L1s). Defaults to `None`.
            output_tile (List of [int], optional): Specifies the output tile configuration. Defaults to `None`.
            optional_output_tensor (ttnn.Tensor, optional): User provided on-device output tensor where the result of linear is to be written. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.
        )doc",
        ttnn::pybind_overload_t{
            [](decltype(::ttnn::matmul_batched_weights)& self,
               const ttnn::Tensor& input_tensor_a,
               const std::vector<ttnn::Tensor>& input_tensors_b,
               const bool transpose_a,
               const bool transpose_b,
               const std::optional<const ttnn::MemoryConfig>& memory_config,
               const std::optional<const DataType> dtype,
               const std::optional<const MatmulProgramConfig>& program_config,
               const std::optional<const std::string>& activation,
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
            py::arg("input_tensor_a"),
            py::arg("input_tensors_b"),
            py::kw_only(),
            py::arg("transpose_a") = false,
            py::arg("transpose_b") = false,
            py::arg("memory_config") = std::nullopt,
            py::arg("dtype") = std::nullopt,
            py::arg("program_config") = std::nullopt,
            py::arg("activation") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt,
            py::arg("core_grid") = std::nullopt,
            py::arg("output_tile") = std::nullopt,
            py::arg("optional_output_tensor") = std::nullopt,
            py::arg("global_cb") = std::nullopt,
            py::arg("sub_device_id") = std::nullopt,
        });

    bind_registered_operation(
        module,
        ::ttnn::sparse_matmul,
        R"doc(
        Performs sparse matrix multiplication on input tensors based on sparsity tensor that has scale factor for each token.

        Args:
            input_tensor_a (ttnn.Tensor): the first tensor to be multiplied containing the weights of the experts. Needs to be on the device.
            input_tensor_b (ttnn.Tensor): the second tensor to be multiplied, containing the tokens to be processed. Needs to be on the device.
        Keyword Args:
            sparsity (ttnn.Tensor): the sparsity tensor containing the scale factor for each token for each expert. Needs to be on the device.
            nnz (int): the number of non-zero values in the sparsity tensor.
            memory_config (ttnn.MemoryConfig, optional): the memory configuration of the output tensor. Defaults to `None`, which will result in using `ttnn.DRAM_MEMORY_CONFIG`.
            dtype (ttnn.DataType, optional): the data type of the output tensor. Defaults to `None`.
            program_config (MatmulProgramConfig, optional): the program configuration for the matmul operation. Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): the compute kernel configuration for the matmul operation. Defaults to `None`.
            core_grid (ttnn.CoreGrid, optional): the grid on which to distribute the sharded tensor on (writes to the cores L1s). Defaults to `None`.
            output_tile (List of [int], optional): Specifies the output tile configuration. Defaults to `None`.
            optional_output_tensor (ttnn.Tensor, optional): User provided on-device output tensor where the result of sparse_matmul is to be written. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor with sparse results.

        Example:
            >>> # Sparse matmul for 64 batch, 128 sequence, 512 hidden dimensions, 8 experts
            >>> expert_weights = ttnn.ones([1, 8, 512, 512])
            >>> tokens = ttnn.ones([1, 64, 128, 512])
            >>> # Create sparsity bitmask
            >>> sparsity_bitmask = torch.zeros([1, 64, 128, 8])
            >>> # Set some tokens to be processed by different experts (simplified pattern)
            >>> sparsity_bitmask[0, 0, 0, 0] = 1.0  # First token goes to expert 0
            >>> sparsity_bitmask[0, 0, 0, 10] = 1.0  # First token goes to expert 10 as well
            >>> sparsity_bitmask[0, 0, 1, 2] = 0.7  # Second token goes to expert 2
            >>> sparsity_bitmask[0, 1, 0, 1] = 0.3  # Another token goes to expert 1
            >>> # Move sparsity bitmask to device
            >>> sparsity_bitmask = ttnn.to_device(sparsity_bitmask, device)
            >>> # Perform sparse matmul
            >>> output = ttnn.sparse_matmul(expert_weights, tokens, sparsity=sparsity_bitmask, nnz=4)
            >>> print(output.shape)
            [64, 128, 8, 512]
        )doc",
        ttnn::pybind_overload_t{
            [](decltype(::ttnn::sparse_matmul)& self,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor& input_tensor_b,
               const ttnn::Tensor& sparsity,
               uint32_t nnz,
               const std::optional<const ttnn::MemoryConfig>& memory_config,
               const std::optional<const DataType> dtype,
               const std::optional<const MatmulProgramConfig>& program_config,
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
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::kw_only(),
            py::arg("sparsity"),
            py::arg("nnz"),
            py::arg("memory_config") = std::nullopt,
            py::arg("dtype") = std::nullopt,
            py::arg("program_config") = std::nullopt,
            py::arg("compute_kernel_config") = std::nullopt,
            py::arg("core_grid") = std::nullopt,
            py::arg("output_tile") = std::nullopt,
            py::arg("optional_output_tensor") = std::nullopt,
            py::arg("global_cb") = std::nullopt,
            py::arg("sub_device_id") = std::nullopt,
        });
}

}  // namespace ttnn::operations::matmul
