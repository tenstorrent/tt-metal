// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_decode_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/matmul_decode/matmul_decode.hpp"
#include "ttnn/operations/experimental/matmul_decode/device/matmul_decode_descriptor.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/mesh_coord.hpp>

namespace ttnn::operations::experimental::matmul_decode::detail {

void bind_matmul_decode_operation(nb::module_& mod) {
    // The details of one weight packed inside a larger fused weight tensor, for the
    // packed_weight argument below and for MatmulDecodeParams.packed_weight.
    nb::class_<ttnn::experimental::PackedWeightSpec>(mod, "MatmulDecodePackedWeightSpec")
        .def(nb::init<>())
        .def(
            "__init__",
            [](ttnn::experimental::PackedWeightSpec* t,
               uint32_t tile_offset,
               uint32_t K,
               uint32_t N,
               const tt::tt_metal::CoreRangeSet& cores,
               uint32_t k_blocks,
               uint32_t batch,
               uint32_t b_blocks) {
                new (t) ttnn::experimental::PackedWeightSpec{tile_offset, K, N, cores, k_blocks, batch, b_blocks};
            },
            nb::arg("tile_offset"),
            nb::arg("K"),
            nb::arg("N"),
            nb::arg("cores"),
            nb::arg("k_blocks") = 1,
            nb::arg("batch") = 1,
            nb::arg("b_blocks") = 1)
        .def_rw("tile_offset", &ttnn::experimental::PackedWeightSpec::tile_offset)
        .def_rw("K", &ttnn::experimental::PackedWeightSpec::K)
        .def_rw("N", &ttnn::experimental::PackedWeightSpec::N)
        .def_rw("cores", &ttnn::experimental::PackedWeightSpec::cores)
        .def_rw("k_blocks", &ttnn::experimental::PackedWeightSpec::k_blocks)
        .def_rw("batch", &ttnn::experimental::PackedWeightSpec::batch)
        .def_rw("b_blocks", &ttnn::experimental::PackedWeightSpec::b_blocks)
        .def_prop_ro("num_cores", &ttnn::experimental::PackedWeightSpec::num_cores)
        .def_prop_ro("n_blocks", &ttnn::experimental::PackedWeightSpec::n_blocks);

    ttnn::bind_function<"matmul_decode", "ttnn.experimental.">(
        mod,
        R"doc(matmul_decode(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, *, partial_width_sharded: bool = False, dtype: Optional[ttnn.DataType] = None, output_mem_config: Optional[ttnn.MemoryConfig] = None, global_cb: Optional[ttnn.GlobalCircularBuffer] = None, global_cb_k_blocks: int = 1, packed_weight: Optional[ttnn.experimental.MatmulDecodePackedWeightSpec] = None, all_gather: bool = False, mesh_coords: Optional[list[ttnn.MeshCoordinate]] = None, ring_gather: bool = False) -> ttnn.Tensor

        Returns the matrix product of two tensors.

        Args:
            input_tensor_a (ttnn.Tensor): the first tensor to be multiplied. A rank-4 tensor
                ([d0, d1, M, K]) whose leading dims multiply to a batch > 1 selects the batched
                program factory; the fold geometry is inferred from the operand shapes.
            input_tensor_b (ttnn.Tensor): the second tensor to be multiplied.

        Keyword Args:
            partial_width_sharded (bool, optional): force the partial width-sharded program
                factory, where B is sharded along both K and N and the K-partials are reduced
                across cores. Ignored when the batched factory is selected. Defaults to False.
            dtype (ttnn.DataType, optional): data type for the output tensor. Defaults to None.
            output_mem_config (ttnn.MemoryConfig, optional): memory config for the output tensor.
                Defaults to None.
            global_cb (ttnn.GlobalCircularBuffer, optional): DRAM-sender global circular buffer
                supplying the weights from the tensor prefetcher. Supported by all three program
                factories; requires a DRAM ND-sharded (receiver-contiguous) weight, whose ND
                shard is exactly one receiver's slab. Defaults to None.

                Each mode delivers one whole slab per receiver per invocation, and assigns
                weight blocks to receiver cores in the GCB's receiver-core order (row-major
                within the receiver grid):

                    mode     receivers             slab shape        slab idx -> block
                    full     N_blocks              [K, N/N_blocks]   n_idx = idx
                    partial  K_blocks*N_blocks     [Kc, Nc]          k_idx = idx / N_blocks, n_idx = idx % N_blocks
                    batched  b_blocks*n_blocks     [Bc*K, Nc]        b_idx = idx / n_blocks, n_idx = idx % n_blocks

                N is the fast-varying dimension in the two two-dimensional modes.

                The caller owns this ordering contract and it is not checked at runtime: the
                weight's ND shard order and the prefetcher's ring order must both agree with
                the receiver order. When they disagree the result is silently wrong rather
                than an error -- each core multiplies against another core's weights. Prefer
                building the pair with
                `ttnn._experimental.tensor_prefetcher_matmul_decode.make_matmul_decode_gcb`,
                which derives the ring layout from the receiver grid for you.
            global_cb_k_blocks (int, optional): how many GCB pages carry one receiver's slab.
                Defaults to 1, one page per slab, which requires the GCB to hold a whole slab
                per receiver. A higher value cuts the slab into that many equal K-blocks, which
                the reader streams and the compute kernel accumulates over, so the GCB only has
                to hold two pages -- letting one small GCB feed weights whose slabs differ in
                size, as long as they agree on the page size.

                It must equal the `block_count` of the prefetch request filling this GCB. A
                mismatch is a device hang, not an error, so derive both from
                `ttnn._experimental.tensor_prefetcher_matmul_decode.matmul_decode_k_blocks`
                rather than by hand.
            packed_weight (ttnn.experimental.MatmulDecodePackedWeightSpec, optional): where this
                op's weight lives inside `input_tensor_b` when B is a larger fused weight tensor:
                one HEIGHT_SHARDED L1 tensor packing many weights, one equal one-tile-wide shard
                per core, this weight's slab occupying tiles
                [tile_offset, tile_offset + slab_tiles) of the shard on each of `cores` (slab
                tiles in row-major order, exactly like a dedicated width-sharded weight).

                The fused tensor's shape says nothing about the weight, so the spec carries the
                logical geometry: [K, N], the holding cores (row-major shard order), and the cut
                -- `k_blocks > 1` selects the partial width-sharded factory, `batch > 1` the
                batched factory (with `b_blocks`), otherwise full width-sharded with N split
                across the cores. `partial_width_sharded` is ignored when this is set. Mutually
                exclusive with `global_cb`. Defaults to None.
            all_gather (bool, optional): fuse a fabric all-gather of the local N-shard into
                the same program. Every device then holds `[..., M, N_local * ring_size]`,
                where the ring is the full mesh of `input_tensor_a`. Requires a multi-device
                mesh. Not supported with `global_cb` or the batched factory. Defaults to False.
            mesh_coords (list[ttnn.MeshCoordinate], optional): dispatch the matmul only on
                these mesh coordinates while retaining output storage on the complete mesh.
                Intended for an explicit point-to-point broadcast of the selected rank's
                result. Not supported with `global_cb`, `all_gather`, or the batched factory.
                Defaults to None (all coordinates).
            ring_gather (bool, optional): gather in0 over a pipelined closed ring on the
                union of the source and compute grids instead of the two-hub gather.
                Full- and partial-width L1-resident paths only (plain or packed_weight).
                Not supported with `global_cb` or the batched factory. Defaults to False.

        Returns:
            ttnn.Tensor: the output tensor.
        )doc",
        &ttnn::experimental::matmul_decode,
        nb::arg("input_tensor_a"),
        nb::arg("input_tensor_b"),
        nb::kw_only(),
        nb::arg("partial_width_sharded") = false,
        nb::arg("dtype") = nb::none(),
        nb::arg("output_mem_config") = nb::none(),
        nb::arg("global_cb") = nb::none(),
        nb::arg("global_cb_k_blocks") = 1,
        nb::arg("packed_weight") = nb::none(),
        nb::arg("all_gather") = false,
        nb::arg("mesh_coords") = nb::none(),
        nb::arg("ring_gather") = false);
}

// Descriptor-level bindings for models/experimental/ops/descriptors/matmul_decode.py, mirroring
// the ttnn::prim::MatmulDeviceOperation block in matmul_nanobind.cpp. Bound onto the same
// "ttnn.experimental" submodule as bind_matmul_decode_operation above, so these appear as
// ttnn._ttnn.operations.experimental.MatmulDecode{Params,Inputs,DeviceOperation,...}.
void bind_matmul_decode_descriptor(nb::module_& mod) {
    nb::class_<ttnn::prim::MatmulDecodeParams>(mod, "MatmulDecodeParams")
        .def(nb::init<>())
        .def_rw("M", &ttnn::prim::MatmulDecodeParams::M)
        .def_rw("N", &ttnn::prim::MatmulDecodeParams::N)
        .def_rw("K", &ttnn::prim::MatmulDecodeParams::K)
        .def_rw("output_mem_config", &ttnn::prim::MatmulDecodeParams::output_mem_config)
        .def_rw("output_dtype", &ttnn::prim::MatmulDecodeParams::output_dtype)
        .def_rw("partial_width_sharded", &ttnn::prim::MatmulDecodeParams::partial_width_sharded)
        .def_rw("batch", &ttnn::prim::MatmulDecodeParams::batch)
        .def_rw("b_blocks", &ttnn::prim::MatmulDecodeParams::b_blocks)
        .def_rw("n_blocks", &ttnn::prim::MatmulDecodeParams::n_blocks)
        .def_rw("global_cb", &ttnn::prim::MatmulDecodeParams::global_cb)
        .def_rw("global_cb_k_blocks", &ttnn::prim::MatmulDecodeParams::global_cb_k_blocks)
        .def_rw("packed_weight", &ttnn::prim::MatmulDecodeParams::packed_weight)
        .def_rw("all_gather", &ttnn::prim::MatmulDecodeParams::all_gather)
        .def_rw("ring_size", &ttnn::prim::MatmulDecodeParams::ring_size)
        .def_rw("ring_gather", &ttnn::prim::MatmulDecodeParams::ring_gather);

    nb::class_<ttnn::prim::MatmulDecodeInputs>(mod, "MatmulDecodeInputs")
        .def(
            "__init__",
            [](ttnn::prim::MatmulDecodeInputs* t, const Tensor& a, const Tensor& b) {
                new (t) ttnn::prim::MatmulDecodeInputs{a, b};
            },
            nb::arg("input_tensor_a"),
            nb::arg("input_tensor_b"))
        .def_rw("input_tensor_a", &ttnn::prim::MatmulDecodeInputs::input_tensor_a)
        .def_rw("input_tensor_b", &ttnn::prim::MatmulDecodeInputs::input_tensor_b);

    nb::class_<ttnn::prim::MatmulDecodeDeviceOperation>(mod, "MatmulDecodeDeviceOperation")
        .def_static(
            "create_output_tensors",
            &ttnn::prim::MatmulDecodeDeviceOperation::create_output_tensors,
            nb::arg("operation_attributes"),
            nb::arg("tensor_args"))
        .def_static(
            "compute_output_specs",
            &ttnn::prim::MatmulDecodeDeviceOperation::compute_output_specs,
            nb::arg("operation_attributes"),
            nb::arg("tensor_args"))
        .def_static(
            "compute_program_hash",
            &ttnn::prim::MatmulDecodeDeviceOperation::compute_descriptor_program_hash,
            nb::arg("operation_attributes"),
            nb::arg("tensor_args"));

    // Each program factory's create_descriptor, mirroring the MatmulMultiCoreReuse*ProgramFactory
    // bindings in matmul_nanobind.cpp. matmul_decode has no "core_range_set" override argument --
    // unlike plain matmul, core placement is entirely derived from where the caller already
    // sharded input_tensor_a / input_tensor_b / the output tensor (or, on the prefetcher path,
    // from the GlobalCircularBuffer's receiver set), so there is nothing extra to pass here.
    nb::class_<ttnn::operations::experimental::matmul_decode::MatmulDecodeDeviceOperation::FullWidthSharded>(
        mod, "MatmulDecodeFullWidthShardedProgramFactory")
        .def_static(
            "create_descriptor",
            &ttnn::prim::matmul_decode_full_width_sharded_create_descriptor,
            nb::arg("operation_attributes"),
            nb::arg("tensor_args"),
            nb::arg("tensor_return_value"),
            nb::arg("mesh_dispatch_coordinate") = nb::none());

    nb::class_<ttnn::operations::experimental::matmul_decode::MatmulDecodeDeviceOperation::PartialWidthSharded>(
        mod, "MatmulDecodePartialWidthShardedProgramFactory")
        .def_static(
            "create_descriptor",
            &ttnn::prim::matmul_decode_partial_width_sharded_create_descriptor,
            nb::arg("operation_attributes"),
            nb::arg("tensor_args"),
            nb::arg("tensor_return_value"));

    nb::class_<ttnn::operations::experimental::matmul_decode::MatmulDecodeDeviceOperation::BatchedWidthSharded>(
        mod, "MatmulDecodeBatchedWidthShardedProgramFactory")
        .def_static(
            "create_descriptor",
            &ttnn::prim::matmul_decode_batched_width_sharded_create_descriptor,
            nb::arg("operation_attributes"),
            nb::arg("tensor_args"),
            nb::arg("tensor_return_value"));

    mod.def(
        "matmul_decode_select_program_factory",
        &ttnn::prim::matmul_decode_select_program_factory,
        nb::arg("operation_attributes"),
        nb::arg("tensor_args"));
}

}  // namespace ttnn::operations::experimental::matmul_decode::detail
