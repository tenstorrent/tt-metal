// SPDX-FileCopyrightText: © 2043 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/matmul_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pybind11/decorators.hpp"
#include "tt_metal/common/core_coord.h"
#include "ttnn/cpp/pybind11/json_class.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/types.hpp"

namespace ttnn {
namespace operations {
namespace matmul {

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
            py::init<
                CoreCoord,
                std::size_t,
                std::size_t,
                std::size_t,
                std::size_t,
                std::size_t,
                bool,
                std::optional<UnaryWithParam>,
                bool>(),
            py::kw_only(),
            py::arg("compute_with_storage_grid_size"),
            py::arg("in0_block_w").noconvert(),
            py::arg("out_subblock_h").noconvert(),
            py::arg("out_subblock_w").noconvert(),
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
            py::init<
                CoreCoord,
                std::size_t,
                std::size_t,
                std::size_t,
                std::size_t,
                std::size_t,
                bool,
                std::optional<UnaryWithParam>,
                bool>(),
            py::kw_only(),
            py::arg("compute_with_storage_grid_size"),
            py::arg("in0_block_w").noconvert(),
            py::arg("out_subblock_h").noconvert(),
            py::arg("out_subblock_w").noconvert(),
            py::arg("per_core_M").noconvert(),
            py::arg("per_core_N").noconvert(),
            py::arg("fuse_batch").noconvert(),
            py::arg("fused_activation"),
            py::arg("mcast_in0").noconvert())
        .def_readwrite(
            "compute_with_storage_grid_size",
            &MatmulMultiCoreReuseMultiCast1DProgramConfig::compute_with_storage_grid_size)
        .def_readwrite("in0_block_w", &MatmulMultiCoreReuseMultiCast1DProgramConfig::in0_block_w)
        .def_readwrite("out_subblock_h", &MatmulMultiCoreReuseMultiCast1DProgramConfig::out_subblock_h)
        .def_readwrite("out_subblock_w", &MatmulMultiCoreReuseMultiCast1DProgramConfig::out_subblock_w)
        .def_readwrite("per_core_M", &MatmulMultiCoreReuseMultiCast1DProgramConfig::per_core_M)
        .def_readwrite("per_core_N", &MatmulMultiCoreReuseMultiCast1DProgramConfig::per_core_N)
        .def_readwrite("fuse_batch", &MatmulMultiCoreReuseMultiCast1DProgramConfig::fuse_batch)
        .def_readwrite("fused_activation", &MatmulMultiCoreReuseMultiCast1DProgramConfig::fused_activation)
        .def_readwrite("mcast_in0", &MatmulMultiCoreReuseMultiCast1DProgramConfig::mcast_in0);

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

        The input tensors need to be tiled. Therefore, the input tensors have to be
        at least 2-dimensional.

        If the input tensors have more than two dimensions, the additional, front,
        dimensions may be used for batched matrix multiply.
        These front dimensions may also be referred to as batch dimensions.
        E.g. a tensor with dimensions :math:`(a \\times b \\times c \\times d)`
        has batch dimensions a and b.
        The following are the allowed possibilities for batch dimensions.
        Examples below show concrete operations and tensor sizes.

        - If all batch dimensions are all of size 1, then there is no batched operation.

        - If both inputs have batch dimensions that are not all of size 1, then the
          batch dimensions of both inputs should be the same. If the dimensions are
          not the same then, although there may be combinations that may work, in most
          cases various errors will be reported.

        - If the first input has batch dimensions that are not all of size 1, and the
          second input has no batch dimensions or has batch dimensions all of size 1,
          then the second input is broadcast to align appropriately with the first
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

              - The two batch dimensions are swapped. E.g. the first input has :math:`(j \\times 1)`
                and the second input has :math:`(1 \\times j)`
                or the first input has :math:`(1 \\times j)` and the second input has
                :math:`(j \\times 1)`
              - When a batch dimension is implicitly extended then the two patch dimensions are swapped.
                E.g.  :math:`(j \\times 1)` and :math:`(j)` which is treated as
                :math:`(j \\times 1)` and :math:`(1 \\times j)`

        - In order to leverage sharded matmul implementations we can shard both input_tensor_a and input_tensor_b. The sharding strategy used will be according
          to the sharding strategy on the respective tensor. A sharded 1D matmul can be either HEIGHT or WIDTH sharded, 2D matmuls can be block sharded.

          Note: the broadcasting logic only looks at the batch dimensions when determining if the inputs
          are broadcastable, and not the matrix dimensions. For example, if :attr:`input_tensor_a` is a
          :math:`(j \\times 1 \\times n\_size \\times m\_size)` tensor and :attr:`input_tensor_b` is a :math:`(k\_size \\times m\_size \\times p)`
          tensor, these inputs are valid for broadcasting even though the final two dimensions (i.e. the
          matrix dimensions) are different. The operation will return a :math:`(j \\times k\_size \\times n\_size \\times p)` tensor.

        - Note: there are various additional constraints related to specific program
          configs chosen. Please look at the error messages carefully and fix
          problems appropriately.

        Args:
            input_tensor_a (ttnn.Tensor): the first tensor to be multiplied. Needs to be on the device.
            input_tensor_b (ttnn.Tensor): the second tensor to be multiplied. Needs to be on the device.

        Keyword Args:
            memory_config(ttnn.MemoryConfig, optional): the memory configuration of the output tensor. Defaults to `None`, which will result in using ttnn.DRAM_MEMORY_CONFIG.
            dtype (ttnn.DataType): the data type of the output tensor. Defaults to `None`.
            core_grid (ttnn.CoreGrid): the grid on which to distribute the sharded tensor on (writes to the cores L1s). Defaults to `None`.
            program_config (ttnn.MatmulProgramConfig): the program configuration for the matmul operation. Defaults to `None`.
            activation (str, optional): the activation function to be applied. Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig): the compute kernel configuration for the matmul operation. Defaults to `None`.

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
            >>> # batched matrix x broadcasted tensor - first input has batch dimensions not of size 1
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
               const std::optional<const ttnn::MemoryConfig> memory_config,
               const std::optional<const DataType> dtype,
               const std::optional<const MatmulProgramConfig> program_config,
               const std::optional<const std::string>& activation,
               const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
               const std::optional<const ttnn::CoreGrid> core_grid,
               const std::optional<const Tile>& output_tile) -> ttnn::Tensor {
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
                    output_tile);
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
            memory_config (ttnn.MemoryConfig, optional): the memory configuration of the output tensor. Defaults to `None`, which will result in using `ttnn.DRAM_MEMORY_CONFIG`.
            dtype (ttnn.DataType, optional): the data type of the output tensor. Defaults to `None`.
            core_grid (ttnn.CoreGrid, optional): the grid on which to distribute the sharded tensor on (writes to the cores L1s). Defaults to `None`.
            program_config (MatmulProgramConfig, optional): the program configuration for the matmul operation. Defaults to `None`.
            activation (str, optional): the activation function to be applied. Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): the compute kernel configuration for the matmul operation. Defaults to `None`.

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
               const std::optional<const ttnn::MemoryConfig> memory_config,
               const std::optional<const DataType> dtype,
               const std::optional<const MatmulProgramConfig> program_config,
               const std::optional<const std::string>& activation,
               const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
               const std::optional<const ttnn::CoreGrid> core_grid,
               const std::optional<const Tile>& output_tile) -> ttnn::Tensor {
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
                    output_tile);
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
        });
}

}  // namespace matmul
}  // namespace operations
}  // namespace ttnn
