// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_decode_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/matmul_decode/matmul_decode.hpp"
#include "ttnn/operations/experimental/matmul_decode/device/matmul_decode_descriptor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::matmul_decode::detail {

void bind_matmul_decode_operation(nb::module_& mod) {
    ttnn::bind_function<"matmul_decode", "ttnn.experimental.">(
        mod,
        R"doc(matmul_decode(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, *, partial_width_sharded: bool = False, dtype: Optional[ttnn.DataType] = None, output_mem_config: Optional[ttnn.MemoryConfig] = None, global_cb: Optional[ttnn.GlobalCircularBuffer] = None, global_cb_k_blocks: int = 1) -> ttnn.Tensor

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
        nb::arg("global_cb_k_blocks") = 1);
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
        .def_rw("global_cb_k_blocks", &ttnn::prim::MatmulDecodeParams::global_cb_k_blocks);

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
            nb::arg("tensor_return_value"));

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
