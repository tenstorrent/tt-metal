// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_decode_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/matmul_decode/matmul_decode.hpp"
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

}  // namespace ttnn::operations::experimental::matmul_decode::detail
