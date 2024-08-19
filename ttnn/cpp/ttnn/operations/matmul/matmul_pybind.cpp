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

        The behavior depends on the dimensionality of the tensors as follows:

        - If both arguments are 2-dimensional, the matrix-matrix product is returned.
        - If the first argument is 1-dimensional and the second argument is 2-dimensional,
        a 1 is prepended to its dimension for the purpose of the matrix multiply.
        After the matrix multiply, the prepended dimension is removed.
        - If the first argument is 2-dimensional and the second argument is 1-dimensional,
        the matrix-vector product is returned in 2 dimensions.
        - If both arguments are at least 1-dimensional and at least one argument is
        N-dimensional (where N > 2), then a batched matrix multiply is returned.  If the first
        argument is 1-dimensional, a 1 is prepended to its dimension for the purpose of the
        batched matrix multiply.  If the second argument is 1-dimensional, a
        1 is appended to its dimension for the purpose of the batched matrix multiple.
        The non-matrix (i.e. batch) dimensions must be broadcastable.
        The behaviour is the same as PyTorch, with the exception of two cases of batch dimensions:

            - The two batch dimensions are swapped. E.g. :math:`(j \times 1)` and :math:`(1 \times j)`
                or :math:`(1 \times j)` and :math:`(j \times 1)`
            - When a batch dimension is implicitly extended then the two patch dimensions are swapped.
                E.g.  :math:`(j \times 1)` and :math:`(j)` which is treated as
                :math:`(j \times 1)` and :math:`(1 \times j)`

        - In order to leverage sharded matmul implementations we can shard both :attr:`input_tensor_a` and :attr:`input_tensor_b`. The sharding strategy used will be according
        to the sharding strategy on the respective tensor. A sharded 1D matmul can be either HEIGHT or WIDTH sharded, 2D matmuls can be block sharded.

        Note that the broadcasting logic only looks at the batch dimensions when determining if the inputs
        are broadcastable, and not the matrix dimensions. For example, if :attr:`input_tensor_a` is a
        :math:`(j \times 1 \times n \times m)` tensor and :attr:`input_tensor_b` is a :math:`(k \times m \times p)`
        tensor, these inputs are valid for broadcasting even though the final two dimensions (i.e. the
        matrix dimensions) are different. The operation will return a :math:`(j \times k \times n \times p)` tensor.

        Note:
            The 1-dimensional dot product version of this function is currently returning the Tensor with a non-empty shape. This is expected to be fixed in an upcoming release.

        Args:
            input_tensor_a (ttnn.Tensor): the first tensor to be multiplied. Needs to be on the device.
            input_tensor_b (ttnn.Tensor): the second tensor to be multiplied. Needs to be on the device.

        Keyword Args:
            memory_config(ttnn.MemoryConfig, optional): the memory configuration of the output tensor. Defaults to `None`, which will result in using ttnn.DRAM_MEMORY_CONFIG
            dtype (ttnn.DataType): the data type of the output tensor. Defaults to `None`.
            core_grid (ttnn.CoreGrid): the grid on which to distribute the sharded tensor on (writes to the cores L1s). Defaults to `None`.
            program_config (ttnn.MatmulProgramConfig): the program configuration for the matmul operation. Defaults to `None`.
            activation (str, optional): the activation function to be applied. Defaults to `None`.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig): the compute kernel configuration for the matmul operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Example:
            >>> # vector x vector
            >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.randn((32), dtype=torch.bfloat16)), device)
            >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.randn((32), dtype=torch.bfloat16)), device)
            >>> output = tensor1 @ tensor2
            >>> print(output.shape)
            [32]
            >>> # matrix x vector
            >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.randn((64, 32), dtype=torch.bfloat16)), device)
            >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.randn((32), dtype=torch.bfloat16)), device)
            >>> output = tensor1 @ tensor2
            >>> print(output.shape)
            [64, 1]
            >>> # batched matrix x broadcasted vector
            >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.randn((10, 64, 32), dtype=torch.bfloat16)), device)
            >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.randn((32), dtype=torch.bfloat16)), device)
            >>> output = tensor1 @ tensor2
            >>> print(output.shape)
            [10, 64, 1]
            >>> # batched matrix x batched matrix
            >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.randn((10, 64, 32), dtype=torch.bfloat16)), device)
            >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.randn((10, 32, 128), dtype=torch.bfloat16)), device)
            >>> output = tensor1 @ tensor2
            >>> print(output.shape)
            [10, 64, 128]
            >>> # batched matrix x broadcasted matrix
            >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.randn((10, 64, 32), dtype=torch.bfloat16)), device)
            >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.randn((32, 128), dtype=torch.bfloat16)), device)
            >>> output = tensor1 @ tensor2
            >>> print(output.shape)
            [10, 64, 128]
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
               const std::optional<const ttnn::CoreGrid> core_grid) -> ttnn::Tensor {
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
                    core_grid);
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
        });

    bind_registered_operation(
        module,
        ::ttnn::linear,
        R"doc(
        Returns the linear transformation of the inputs.

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
               const std::optional<const ttnn::CoreGrid> core_grid) -> ttnn::Tensor {
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
                    core_grid);
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
        });
}

}  // namespace matmul
}  // namespace operations
}  // namespace ttnn
