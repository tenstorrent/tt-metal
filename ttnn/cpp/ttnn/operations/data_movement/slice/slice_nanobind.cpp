// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "slice_nanobind.hpp"

#include <array>
#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/small_vector_caster.hpp"
#include "ttnn-nanobind/bind_function.hpp"

#include "slice.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_tile.hpp"

namespace ttnn::operations::data_movement::detail {

namespace {

ttnn::Tensor slice_small_vector_wrapper(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<int>& slice_start,
    const ttsl::SmallVector<int>& slice_end,
    const std::optional<ttsl::SmallVector<int>>& step,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<float>& pad_value,
    const std::optional<CoreRangeSet>&& sub_core_grids) {
    const auto step_value = step.value_or(ttsl::SmallVector<int>(slice_end.size(), 1));
    return ttnn::slice(
        input_tensor,
        slice_start,
        slice_end,
        step_value,
        memory_config,
        optional_output_tensor,
        pad_value,
        sub_core_grids);
}

}  // namespace

void bind_slice(nb::module_& mod) {
    const auto* doc = R"doc(
        Returns a slice of the input tensor, given the start and end indices of each dimension and an optional step.

        ``slice_start`` and ``slice_end`` can also be passed positionally as ``ttnn.Tensor`` objects; that overload also
        takes the ``slice_dim`` and ``num_devices`` keywords.

        Args:
            input_tensor (ttnn.Tensor): Input tensor, on device.
            slice_start (List[int]): Start indices of input tensor. Negative values count from the end of the dim; other values must be in ``[0, input_tensor_shape[i])``.
            slice_end (List[int]): End indices of input tensor (exclusive). Negative values count from the end of the dim; other values must be in ``(0, input_tensor_shape[i]]``.
            slice_step (List[int], optional): Step size for each dim. Defaults to `None`, which uses a step of 1 for every dim.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the output tensor. Defaults to `None`, which uses the input tensor's memory configuration.
            output_tensor (ttnn.Tensor, optional): Pre-allocated output tensor. Its shape must match the slice output. Defaults to `None`.
            pad_value (float, optional): Fill value for the implicit tile padding of a TILE output. It is applied when a unit-step slice, with start indices in the last two dims that are multiples of 32, changes the size of one of the last two dims to a size that is not a multiple of 32. Defaults to `None`, which leaves the padding undefined.
            sub_core_grids (ttnn.CoreRangeSet, optional): sub core grids for the operation. Defaults to `None`.

        Returns:
            ttnn.Tensor: the output tensor.

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32, INT32, UINT32
                 - TILE, ROW_MAJOR
               * - UINT16
                 - TILE, ROW_MAJOR
               * - UINT8
                 - ROW_MAJOR
               * - BFLOAT8_B
                 - TILE

            ``BFLOAT8_B`` tensors need a ``slice_step`` of 1.

            TILE layout tensors must use the standard 32x32 tile, and a pre-allocated ``output_tensor`` must
            carry the same tile as the input.

        Memory Support:
            - Interleaved: DRAM and L1
            - Sharded: HEIGHT_SHARDED, WIDTH_SHARDED and BLOCK_SHARDED input and output; DRAM-sharded input is also accepted with an interleaved output
            - ND sharded: input and output, for tensors of rank 2 or more
    )doc";

    // TODO: implementing the array version and overloading the nanobind with all the possible array sizes is better
    // than a vector with a fixed size default value
    ttnn::bind_function<"slice">(
        mod,
        doc,
        // Overload 1: Tensor args version (uint32_t template parameter)
        ttnn::overload_t(
            nb::overload_cast<
                const ttnn::Tensor&,
                const ttnn::Tensor&,
                const ttnn::Tensor&,
                const std::optional<ttsl::SmallVector<uint32_t>>&,
                const std::optional<MemoryConfig>&,
                const std::optional<Tensor>&,
                const std::optional<float>&,
                const std::optional<uint32_t>&,
                const std::optional<uint32_t>&,
                const std::optional<CoreRangeSet>&>(&ttnn::slice<uint32_t>),
            nb::arg("input_tensor"),
            nb::arg("starts"),
            nb::arg("ends"),
            nb::arg("slice_step") = nb::none(),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("pad_value") = nb::none(),
            nb::arg("slice_dim") = nb::none(),
            nb::arg("num_devices") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()),
        // Overload 2: std::array version (uint32_t template parameter, size 4)
        ttnn::overload_t(
            nb::overload_cast<
                const ttnn::Tensor&,
                const std::array<uint32_t, 4>&,
                const std::array<uint32_t, 4>&,
                const std::array<uint32_t, 4>&,
                const std::optional<MemoryConfig>&,
                const std::optional<Tensor>&,
                const std::optional<float>&,
                const std::optional<CoreRangeSet>&>(&ttnn::slice<uint32_t, 4>),
            nb::arg("input_tensor"),
            nb::arg("starts"),
            nb::arg("ends"),
            nb::arg("steps"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("pad_value") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()),
        // Overload 3: ttsl::SmallVector<int> version (int32_t template parameter)
        ttnn::overload_t(
            &slice_small_vector_wrapper,
            nb::arg("input_tensor"),
            nb::arg("slice_start"),
            nb::arg("slice_end"),
            nb::arg("slice_step") = nb::none(),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("pad_value") = nb::none(),
            nb::arg("sub_core_grids") = nb::none()));
}
void bind_slice_descriptor(nb::module_& mod) {
    nb::class_<ttnn::prim::SliceParams>(mod, "SliceParams")
        .def(nb::init<>())
        .def_rw("slice_start", &ttnn::prim::SliceParams::slice_start)
        .def_rw("slice_end", &ttnn::prim::SliceParams::slice_end)
        .def_rw("step", &ttnn::prim::SliceParams::step)
        .def_rw("output_mem_config", &ttnn::prim::SliceParams::output_mem_config)
        .def_rw("sub_core_grids", &ttnn::prim::SliceParams::sub_core_grids)
        .def_rw(
            "output_mem_config_inherited",
            &ttnn::prim::SliceParams::output_mem_config_inherited,
            "True if output_mem_config was implicitly inherited from the input tensor (no memory_config or "
            "output_tensor given by the caller) rather than explicitly requested. When set, the device operation "
            "may adjust an inherited sharded spec to fit the sliced output shape.");

    nb::class_<ttnn::prim::SliceInputs>(mod, "SliceInputs")
        .def(
            "__init__",
            [](ttnn::prim::SliceInputs* t, const ttnn::Tensor& input) {
                new (t) ttnn::prim::SliceInputs{input, std::nullopt, std::nullopt, std::nullopt};
            },
            nb::arg("input"))
        .def_rw("input", &ttnn::prim::SliceInputs::input)
        .def_rw("preallocated_output", &ttnn::prim::SliceInputs::preallocated_output);

    nb::class_<ttnn::prim::SliceDeviceOperation>(mod, "SliceDeviceOperation")
        .def_static(
            "create_output_tensors",
            &ttnn::prim::SliceDeviceOperation::create_output_tensors,
            nb::arg("operation_attributes"),
            nb::arg("tensor_args"))
        .def_static(
            "compute_output_specs",
            &ttnn::prim::SliceDeviceOperation::compute_output_specs,
            nb::arg("operation_attributes"),
            nb::arg("tensor_args"));

    // Bound without create_descriptor: the Metal 2.0 factory produces a ProgramSpec, and the
    // fusion branches that used to drive create_descriptor consume a ProgramDescriptor. The class
    // stays exposed so those call sites resolve and report the missing method themselves.
    nb::class_<ttnn::prim::SliceTileProgramFactory>(mod, "SliceTileProgramFactory");
}

}  // namespace ttnn::operations::data_movement::detail
