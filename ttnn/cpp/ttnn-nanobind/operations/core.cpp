// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "core.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>  // needed for DeviceComputerKernelConfig

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn-nanobind/export_enum.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/operations/experimental/core_subset_write/copy_to_device_filtered.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/base_types.hpp>
#include <tt-metalium/core_coord.hpp>

// NOLINTBEGIN(bugprone-unused-raii)

namespace ttnn::operations::core {

struct DeviceComputeKernelConfigPlaceholder {};

void py_module_types(nb::module_& mod) {
    export_enum<compute_throttle_utils::ThrottleLevel>(mod, "ThrottleLevel", R"doc(
        Enum for controlling compute throttling.

        Higher levels insert NOP instructions to reduce compute throughput:
        - LEVEL_1: Throttle to 73% of max performance
        - LEVEL_2: Throttle to 67% of max performance
        - LEVEL_3: Throttle to 50% of max performance
        - LEVEL_4: Throttle to 40% of max performance
        - LEVEL_5: Throttle to 33% of max performance

        Used to prevent di/dt (power supply current) issues on large core counts.
    )doc");

    // Unified ComputeKernelConfig - all architecture-specific names are now aliases
    nb::class_<DeviceComputeKernelConfigPlaceholder>(mod, "DeviceComputeKernelConfig", R"doc(
        Abstract base type used in op signatures to accept any architecture-specific compute kernel config.

        Example concrete, instantiable class is :class:`ttnn.WormholeComputeKernelConfig`, which is valid for both
        Wormhole and Blackhole devices. Construct one of the concrete classes and pass it wherever a
        ``ttnn.DeviceComputeKernelConfig`` is expected.
    )doc");

    // Primary config class (ComputeKernelConfig is the canonical name, but we expose as WormholeComputeKernelConfig
    // for backward compatibility)
    nb::class_<ComputeKernelConfig>(mod, "WormholeComputeKernelConfig", R"doc(
        Compute kernel configuration for Wormhole devices. Note that ``BlackholeComputeKernelConfig``
        is a type alias for the same class (``WormholeComputeKernelConfig``).

        Controls the precision/throughput trade-offs of the on-chip matrix engine and SFPU.
        Pass an instance of this class as the ``compute_kernel_config`` argument to ops such as
        :func:`ttnn.matmul`, :func:`ttnn.linear`, and others that accept a
        :class:`ttnn.DeviceComputeKernelConfig`.

        See the per-attribute docs below for what each field does, and
        ``tech_reports/matrix_engine/matrix_engine.md`` for the full architectural background.
    )doc")
        .def(
            nb::init<
                tt::tt_metal::MathFidelity,
                bool,
                bool,
                bool,
                bool,
                ttnn::operations::compute_throttle_utils::ThrottleLevel>(),
            nb::kw_only(),
            nb::arg("math_fidelity") = nb::cast(tt::tt_metal::MathFidelity::Invalid),
            nb::arg("math_approx_mode") = true,
            nb::arg("fp32_dest_acc_en") = false,
            nb::arg("packer_l1_acc") = false,
            nb::arg("dst_full_sync_en") = false,
            nb::arg("throttle_level") = compute_throttle_utils::ThrottleLevel::NO_THROTTLE)
        .def_rw("math_fidelity", &ComputeKernelConfig::math_fidelity, R"doc(
            Controls the number of mantissa-bit passes through the 5b×7b multiplier array, trading
            throughput for precision. See :class:`ttnn.MathFidelity` for the available values.
        )doc")
        .def_rw("math_approx_mode", &ComputeKernelConfig::math_approx_mode, R"doc(
            When ``True``, SFPU ops that have an approximate variant (e.g. ``exp``, ``gelu``, ``sqrt``)
            run in high-performance / lower-precision mode instead of high-precision / lower-performance mode.
            Has no effect on ops that do not have an approximate variant.
        )doc")
        .def_rw("fp32_dest_acc_en", &ComputeKernelConfig::fp32_dest_acc_en, R"doc(
            When ``True``, the FPU accumulates results into the math destination (DST) register in FP32
            instead of BFLOAT16. This is required when the output dtype is ``FLOAT32`` and improves
            numerical accuracy for intermediate accumulations.

            Warning: enabling this halves the number of tiles that fit in DST. With ``DstSync::Half``,
            BFLOAT16 holds 8 tiles while FP32 holds only 4.
        )doc")
        .def_rw("packer_l1_acc", &ComputeKernelConfig::packer_l1_acc, R"doc(
            When ``True``, the packer reads the current value at the output SRAM address, adds the value
            from DST, and writes the result back (in-place accumulation in SRAM). This is useful for
            accumulating partial sums in higher precision circular buffer (e.g. FP32) across multiple kernel launches,
            and then doing a final pack to a lower-precision circular buffer (e.g. BFLOAT16).
        )doc")
        .def_rw("dst_full_sync_en", &ComputeKernelConfig::dst_full_sync_en, R"doc(
            When ``True``, the math and pack units synchronize over the full DST register rather than
            the default half-DST split. Leave at ``False`` unless you explicitly need full-DST sync
            semantics.
        )doc")
        .def_rw("throttle_level", &ComputeKernelConfig::throttle_level, R"doc(
            Inserts NOP instructions into the compute kernel to reduce throughput and limit di/dt
            (power-supply current transients) when running on large core grids. See
            :class:`ttnn.ThrottleLevel` for the available levels and their throughput percentages.
        )doc");
}

void py_module(nb::module_& mod) {
    mod.def(
        "init_device_compute_kernel_config",
        &ttnn::init_device_compute_kernel_config,
        nb::arg("arch"),
        nb::arg("device_kernel_config") = nb::none(),
        nb::kw_only(),
        nb::arg("math_fidelity") = nb::cast(tt::tt_metal::MathFidelity::LoFi),
        nb::arg("math_approx_mode") = true,
        nb::arg("fp32_dest_acc_en") = false,
        nb::arg("packer_l1_acc") = false,
        nb::arg("dst_full_sync_en") = false,
        nb::arg("throttle_level") = nb::cast(ttnn::operations::compute_throttle_utils::ThrottleLevel::NO_THROTTLE));

    mod.def("unsqueeze_to_4D", &ttnn::unsqueeze_to_4D, nb::arg("tensor"));

    mod.def(
        "to_device",
        nb::overload_cast<
            const ttnn::Tensor&,
            MeshDevice*,
            const std::optional<MemoryConfig>&,
            std::optional<ttnn::QueueId>>(&ttnn::operations::core::to_device),
        nb::arg("tensor"),
        nb::arg("device"),
        nb::arg("memory_config") = nb::none(),
        nb::kw_only(),
        nb::arg("queue_id") = nb::none(),
        nb::keep_alive<0, 2>(),  // test
        R"doc(
            Copy tensor from host to device.

            Args:
                tensor (ttnn.Tensor): The tensor to be copied from host to device.
                device (ttnn.Device | ttnn.MeshDevice): The target device where the tensor will be copied.
                memory_config (ttnn.MemoryConfig, optional): The memory configuration to use. Defaults to `None`.

            Kwargs:
                queue_id (ttnn.QueueId, optional): The queue id to use. Defaults to `null`.

            Returns:
                ttnn.Tensor: The device tensor copy.
        )doc");

    mod.def(
        "from_device",
        &ttnn::operations::core::from_device,
        nb::arg("tensor"),
        nb::arg("blocking") = true,
        nb::kw_only(),
        nb::arg("queue_id") = nb::none(),
        R"doc(
            Copy tensor from device to host.

            Args:
                tensor (ttnn.Tensor): the tensor to be copied from device to host.
                blocking (bool, optional): whether the operation should be blocked until the copy is complete. Defaults to `True`.

            Kwargs:
                queue_id (ttnn.QueueId, optional): The queue id to use. Defaults to `null`.

            Returns:
                ttnn.Tensor: the host tensor copy.
        )doc");

    mod.def(
        "deallocate",
        &ttnn::operations::core::deallocate,
        nb::arg("tensor"),
        nb::arg("force") = true,
        R"doc(
        Deallocates device tensor. Releases the resources for `ttnn.Tensor` :attr:`tensor` explicitly.

        Args:
            tensor (ttnn.Tensor): the input tensor.
            force (bool, optional): force deallocation even if the buffer may have multiple references. Defaults to `True`.

        Returns:
            `None`: deallocates the tensor.
    )doc");

    mod.def(
        "reallocate",
        [](ttnn::Tensor& input_tensor, const std::optional<ttnn::MemoryConfig>& memory_config) -> ttnn::Tensor {
            return reallocate(input_tensor, memory_config);
        },
        nb::arg("tensor"),
        nb::arg("memory_config") = nb::none(),
        R"doc(
            Deallocates device tensor and returns a reallocated tensor.

            Args:
                tensor (ttnn.Tensor): the input tensor.
                memory_config (ttnn.MemoryConfig, optional): memory configuration for the reallocated tensor. Defaults to `None`.

            Returns:
                ttnn.Tensor: the reallocated tensor.
        )doc");

    ttnn::bind_function<"to_memory_config">(
        mod,
        R"doc(
        Converts a tensor to the desired memory configuration. Used for converting tensors to sharded tensors, interleaved tensors, or converting between DRAM and L1 memory.

        Args:
            tensor (ttnn.Tensor): the input tensor to be converted.
            memory_config (ttnn.MemoryConfig): the desired memory configuration for the tensor.
            dtype (ttnn.DataType, optional): the optional `ttnn` data type. Defaults to `None`.
            output_tensor (ttnn.Tensor, optional): the optional output tensor. Defaults to `None`.

        Returns:
            ttnn.Tensor: the converted tensor.
        )doc",
        &ttnn::to_memory_config,
        nb::arg("tensor"),
        nb::arg("memory_config"),
        nb::arg("dtype") = nb::none(),
        nb::arg("output_tensor") = nb::none());

    ttnn::bind_function<"to_dtype">(
        mod,
        R"doc(
        Converts a host tensor to the desired dtype.

        Args:
            tensor (ttnn.Tensor): the tensor to be converted to the desired dtype.
            dtype (ttnn.DataType): the desired data type.

        Note:
            This operations supports tensors according to the following data types and layout:

            .. list-table:: tensor
                :header-rows: 1

                * - dtype
                    - layout
                * - BFLOAT16, BFLOAT8_B, BFLOAT4_B, FLOAT32, UINT32, INT32, UINT16, UINT8
                    - TILE
                * - BFLOAT16, FLOAT32, UINT32, INT32, UINT16, UINT8
                    - ROW_MAJOR

            Memory Support:
                - Interleaved: DRAM and L1
                - Height, Width, Block, and ND Sharded: DRAM and L1

            Limitations:
                -  tensor must be on the host.
        )doc",
        &ttnn::to_dtype,
        nb::arg("tensor"),
        nb::arg("dtype"));

    mod.def(
           "allocate_tensor_on_device",
           [](const ttnn::TensorSpec& spec, MeshDevice* device) {
               return tt::tt_metal::create_device_tensor(spec, device);
           },
           nb::arg("tensor_spec"),
           nb::arg("mesh_device"))
        .def(
            "allocate_tensor_on_host",
            [](const ttnn::TensorSpec& spec, MeshDevice* device) {
                return tt::tt_metal::allocate_tensor_on_host(spec, device);
            },
            nb::arg("tensor_spec"),
            nb::arg("mesh_device"));

    mod.def(
           "allocate_tensor_on_device",
           [](const ttnn::Shape& shape,
              ttnn::DataType dtype,
              ttnn::Layout layout,
              MeshDevice* device,
              const std::optional<ttnn::MemoryConfig>& mem_config) {
               return tt::tt_metal::create_device_tensor(
                   TensorSpec(
                       shape,
                       tt::tt_metal::TensorLayout(
                           dtype, tt::tt_metal::PageConfig(layout), mem_config.value_or(MemoryConfig{}))),
                   device);
           },
           nb::arg("shape"),
           nb::arg("dtype"),
           nb::arg("layout"),
           nb::arg("mesh_device"),
           nb::arg("memory_config") = nb::none())
        .def(
            "allocate_tensor_on_host",
            [](const ttnn::Shape& shape,
               ttnn::DataType dtype,
               ttnn::Layout layout,
               MeshDevice* device,
               const std::optional<ttnn::MemoryConfig>& mem_config) {
                return tt::tt_metal::allocate_tensor_on_host(
                    TensorSpec(
                        shape,
                        tt::tt_metal::TensorLayout(
                            dtype, tt::tt_metal::PageConfig(layout), mem_config.value_or(MemoryConfig{}))),
                    device);
            },
            nb::arg("shape"),
            nb::arg("dtype"),
            nb::arg("layout"),
            nb::arg("mesh_device"),
            nb::arg("memory_config") = nb::none());

    mod.def(
        "copy_host_to_device_tensor",
        [](const ttnn::Tensor& host_tensor, ttnn::Tensor& device_tensor, const std::optional<QueueId>& cq_id) {
            copy_to_device(host_tensor, device_tensor, cq_id);
        },
        nb::arg("host_tensor"),
        nb::arg("device_tensor"),
        nb::arg("cq_id") = nb::none(),
        R"doc(
        Copies a tensor from host to device.

        Args:
            host_tensor (ttnn.Tensor): the tensor to be copied from host to device.
            device_tensor (ttnn.Tensor): the tensor to be copied to.
            cq_id (ttnn.QueueId, optional): The queue id to use. Defaults to `None`.

        Note:
            This operations supports tensors according to the following data types and layout:

            .. list-table:: host/device tensor
                :header-rows: 1

                * - dtype
                    - layout
                * - BFLOAT16, BFLOAT8_B, BFLOAT4_B, FLOAT32, UINT32, INT32, UINT16, UINT8
                    - TILE
                * - BFLOAT16, FLOAT32, UINT32, INT32, UINT16, UINT8
                    - ROW_MAJOR

            Memory Support:
                - Interleaved: DRAM and L1
                - Height, Width, Block, and ND Sharded: DRAM and L1

            Limitations:
                -  Host and Device tensors must be the same shape, have the same datatype, and have the same data layout (ROW_MAJOR or TILE).
        )doc");

    mod.def(
        "copy_host_to_device_tensor_partial",
        [](const ttnn::Tensor& host_tensor,
           ttnn::Tensor& device_tensor,
           const tt::tt_metal::CoreRangeSet& logical_core_filter,
           const std::optional<QueueId>& cq_id) {
            ttnn::experimental::core_subset_write::copy_to_device_filtered(
                host_tensor, device_tensor, logical_core_filter, cq_id);
        },
        nb::arg("host_tensor"),
        nb::arg("device_tensor"),
        nb::arg("logical_core_filter"),
        nb::arg("cq_id") = nb::none(),
        R"doc(
        Copies host tensor data into the pre-allocated device tensor, writing only shards mapped to
        cores in ``logical_core_filter``. The device tensor must use a sharded buffer layout.

        - Empty ``logical_core_filter`` is a no-op (no shards are written).
        - Non-empty ``logical_core_filter`` with an interleaved device buffer raises an error.
        )doc");

    mod.def(
        "copy_device_to_host_tensor",
        [](const ttnn::Tensor& device_tensor,
           ttnn::Tensor& host_tensor,
           bool blocking = true,
           std::optional<ttnn::QueueId> cq_id = std::nullopt) {
            copy_to_host(device_tensor, host_tensor, blocking, cq_id);
        },
        nb::arg("device_tensor"),
        nb::arg("host_tensor"),
        nb::arg("blocking") = true,
        nb::arg("cq_id") = nb::none(),
        R"doc(
        Copies a tensor from device to host.

        Args:
            device_tensor (ttnn.Tensor): the tensor to be copied from device to host.
            host_tensor (ttnn.Tensor): the tensor to be copied to.
            blocking (bool, optional): whether the operation should be blocked until the copy is complete. Defaults to `True`.
            cq_id (ttnn.QueueId, optional): The queue id to use. Defaults to `None`.

        Note:
            This operations supports tensors according to the following data types and layout:

            .. list-table:: device/host tensor
                :header-rows: 1

                * - dtype
                    - layout
                * - BFLOAT16, BFLOAT8_B, BFLOAT4_B, FLOAT32, UINT32, INT32, UINT16, UINT8
                    - TILE
                * - BFLOAT16, FLOAT32, UINT32, INT32, UINT16, UINT8
                    - ROW_MAJOR

            Memory Support:
                - Interleaved: DRAM and L1
                - Height, Width, Block, and ND Sharded: DRAM and L1

            Limitations:
                -  Host and Device tensors must be the same shape, have the same datatype, and have the same data layout (ROW_MAJOR or TILE).
        )doc");

    ttnn::bind_function<"to_layout">(
        mod,
        R"doc(
        Organizes the `ttnn.Tensor` tensor into either `ttnn.ROW_MAJOR_LAYOUT` or `ttnn.TILE_LAYOUT`.

        When requesting `ttnn.ROW_MAJOR_LAYOUT`, the tensor will be returned unpadded in the last two dimensions.
        When requesting `ttnn.TILE_LAYOUT`, the tensor will be automatically padded where the width and height
        become multiples of 32. In the case where the layout is the same, the operation simply pads or unpads
        the last two dimensions depending on the requested layout.

        Args:
            tensor (ttnn.Tensor): the input tensor to be organized.
            layout (ttnn.Layout): the desired layout, either `ttnn.ROW_MAJOR_LAYOUT` or `ttnn.TILE_LAYOUT`.
            dtype (ttnn.DataType, optional): the optional output data type.
            memory_config (ttnn.MemoryConfig, optional): the optional output memory configuration.
            sub_core_grids (ttnn.CoreRangeSet, optional): the optional sub core grids. Defaults to `None`.
            pad_value (float, optional): the optional pad value. Defaults to `0.0f`.

        Returns:
            ttnn.Tensor: the tensor with the requested layout.
        )doc",
        &ttnn::to_layout,
        nb::arg("tensor"),
        nb::arg("layout"),
        nb::arg("dtype") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("sub_core_grids") = nb::none(),
        nb::arg("pad_value") = 0.0f);

    mod.def(
        "num_cores_to_corerangeset",
        nb::overload_cast<const uint32_t, const CoreCoord, const bool>(&tt::tt_metal::num_cores_to_corerangeset),
        R"doc(Create a CoreRangeSet containing the specified number of cores)doc");

    mod.def(
        "num_cores_to_corerangeset_in_subcoregrids",
        nb::overload_cast<const CoreCoord, const uint32_t, const CoreRangeSet&, const bool>(
            &tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids),
        R"doc(Create a CoreRangeSet containing the specified number of cores starting from start_core in given subcoregrids)doc");

    mod.def(
        "split_work_to_cores",
        nb::overload_cast<const CoreCoord, const uint32_t, const bool>(&tt::tt_metal::split_work_to_cores),
        nb::arg("grid_size"),
        nb::arg("units_to_divide"),
        nb::arg("row_wise") = false,
        R"doc(
        Split work units across cores in a grid.

        This function divides a specified number of work units across cores in a grid.
        It returns information about how the work is distributed, including core ranges
        for different groups if work cannot be evenly divided.

        Args:
            grid_size (ttnn.CoreCoord): The size of the core grid (x, y dimensions).
            units_to_divide (int): The total number of work units to distribute.
            row_wise (bool, optional): Whether to distribute work by iterating row-wise. Defaults to False.

        Returns:
            tuple: A tuple containing:
                - num_cores (int): Number of cores being used
                - all_cores (CoreRangeSet): All cores involved
                - core_group_1 (CoreRangeSet): Cores doing more work
                - core_group_2 (CoreRangeSet): Cores doing less work (empty if evenly divisible)
                - units_per_core_group_1 (int): Work units per core in group 1
                - units_per_core_group_2 (int): Work units per core in group 2

        Example:
        >>> # Split 100 tiles across an 8x8 core grid
        >>> num_cores, all_cores, core_group_1, core_group_2, units_1, units_2 = \\
        ...     ttnn.split_work_to_cores(ttnn.CoreCoord(8, 8), 100)
        >>> print(f"Using {num_cores} cores, {units_1} units per core in group 1, {units_2} in group 2")
        )doc");

    mod.def(
        "split_work_to_cores",
        nb::overload_cast<const CoreRangeSet&, const uint32_t, const bool>(&tt::tt_metal::split_work_to_cores),
        nb::arg("core_grid"),
        nb::arg("units_to_divide"),
        nb::arg("row_wise") = false,
        R"doc(
        Split work units across cores in a CoreRangeSet.

        This function divides a specified number of work units across cores in a CoreRangeSet.
        It returns information about how the work is distributed, including core ranges
        for different groups if work cannot be evenly divided.

        Args:
            core_grid (ttnn.CoreRangeSet): The set of core ranges to distribute work across.
            units_to_divide (int): The total number of work units to distribute.
            row_wise (bool, optional): Whether to distribute work by iterating row-wise. Defaults to False.

        Returns:
            tuple: A tuple containing:
                - num_cores (int): Number of cores being used
                - all_cores (CoreRangeSet): All cores involved
                - core_group_1 (CoreRangeSet): Cores doing more work
                - core_group_2 (CoreRangeSet): Cores doing less work (empty if evenly divisible)
                - units_per_core_group_1 (int): Work units per core in group 1
                - units_per_core_group_2 (int): Work units per core in group 2

        Example:
        >>> # Split 100 tiles across an 8x8 core grid
        >>> core_rangeset = ttnn.CoreRangeSet(ttnn.CoreRange(ttnn.CoreCoord(0,0), ttnn.CoreCoord(7,7)))
        >>> num_cores, all_cores, core_group_1, core_group_2, units_1, units_2 = \\
        ...     ttnn.split_work_to_cores(core_rangeset, 100)
        >>> print(f"Using {num_cores} cores, {units_1} units per core in group 1, {units_2} in group 2")
        )doc");
}

}  // namespace ttnn::operations::core

// NOLINTEND(bugprone-unused-raii)
