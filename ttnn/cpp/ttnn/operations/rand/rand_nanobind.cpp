// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rand_nanobind.hpp"

#include <cstdint>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "rand.hpp"

namespace ttnn::operations::rand {
void bind_rand_operation(nb::module_& mod) {
    nb::enum_<ttnn::RandGenerator>(mod, "RandGenerator")
        .value("LFSR", ttnn::RandGenerator::LFSR)
        .value("THREEFRY", ttnn::RandGenerator::THREEFRY);

    std::string doc =
        R"doc(
        Generates a tensor with the given shape, filled with random values from a uniform distribution.
        Output semantics depend on the specified data type:

        - DataType.float32 / bfloat16:
            Values are generated natively in the half-open range [`low`, `high`).

        - DataType.bfloat4_b / bfloat8_b / int8 / int32 / uint16 / uint32:
            These legacy outputs are generated in float32 and then typecast. Typecast rounding applies, so the
            converted values are not guaranteed to remain in [`low`, `high`). Integer output is not a discrete
            uniform distribution. DataType.uint8 and fp8_e4m3 are not supported.

        Args:
            shape (list[int]): A list of integers defining the shape of the output tensor.
            device (ttnn.Device | ttnn.MeshDevice): The device on which the tensor will be allocated.

        Keyword Args:
            dtype (ttnn.DataType, optional): The data type of the tensor. Defaults to `ttnn.bfloat16`.
            layout (ttnn.Layout, optional): The layout of the tensor. Defaults to `ttnn.TILE_LAYOUT`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `ttnn.DRAM_MEMORY_CONFIG`.
            low (float, optional): The lower bound of the range (inclusive). Defaults to 0.0.
            high (float, optional): The upper bound of the range (exclusive). Defaults to 1.0.
            seed (int, optional): An optional seed to initialize the random number generator
                                for reproducible results. Defaults to 0.
            mesh_mapper (ttnn.MeshMapperConfig, optional): Distribution strategy for multi-device tensors.
                Use ``ttnn.MeshMapperConfig([ttnn.PlacementShard(dim)])`` to shard across mesh devices along the
                given tensor dimension, or ``ttnn.MeshMapperConfig([ttnn.PlacementReplicate()])`` to replicate.
                When sharding, each device generates unique random values; when replicating with a fixed seed,
                all devices produce the same values. Defaults to `None`.
            generator (ttnn.RandGenerator, optional): ``LFSR`` uses the hardware PRNG with hashed per-core seeds and
                position-salted output. ``THREEFRY`` uses the Threefry-2x32 counter-based generator: output depends
                only on the key (seed, shard index, epoch) and the element position, so it is identical across core
                grids. Defaults to ``ttnn.RandGenerator.LFSR``.
            state (ttnn.Tensor, optional): Per-device epoch counters from :func:`ttnn.rand_state`. The op reads and
                advances them on the device, so a captured trace yields fresh values on every replay. Defaults to
                `None`, in which case the same seed always reproduces the same tensor.

        Returns:
            ttnn.Tensor: A tensor with specified shape, dtype, and layout containing random values.

        Note:
            Supported dtypes and layouts:

            .. list-table::
               :header-rows: 1

               * - Dtypes
                 - Layouts
               * - BFLOAT16, FLOAT32, BFLOAT8_B, BFLOAT4_B, INT8, INT32, UINT16, UINT32
                 - TILE, ROW_MAJOR

            BFLOAT8_B and BFLOAT4_B are supported only on TILE layout.
        )doc";

    ttnn::bind_function<"rand">(
        mod,
        doc.c_str(),
        &ttnn::rand,
        nb::arg("shape"),
        nb::arg("device"),
        nb::kw_only(),
        nb::arg("dtype") = nb::cast(DataType::BFLOAT16),
        nb::arg("layout") = nb::cast(Layout::TILE),
        nb::arg("memory_config") = nb::cast(ttnn::DRAM_MEMORY_CONFIG),
        nb::arg("low") = 0.0f,
        nb::arg("high") = 1.0f,
        nb::arg("seed") = 0,
        nb::arg("mesh_mapper") = nb::none(),
        nb::arg("generator") = nb::cast(ttnn::RandGenerator::LFSR),
        nb::arg("state") = nb::none());

    ttnn::bind_function<"rand_state">(
        mod,
        R"doc(
        Allocates the per-device RNG state used by ``ttnn.rand(..., state=...)``: one uint32 row of epoch counters
        per core, zero-initialised and replicated across the mesh. Rewriting it resets the stream.

        Args:
            device (ttnn.Device | ttnn.MeshDevice): The device the state lives on.

        Returns:
            ttnn.Tensor: A uint32 row-major tensor of shape [num_cores, 32].
        )doc",
        &ttnn::rand_state,
        nb::arg("device"));
}
}  // namespace ttnn::operations::rand
