// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dram_prefetcher_consumer_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "dram_prefetcher_consumer.hpp"
#include "dram_prefetcher_validator.hpp"

namespace ttnn::operations::experimental::test {

void bind_test_dram_prefetcher_consumer(nb::module_& mod) {
    ttnn::bind_function<"test_dram_prefetcher_consumer", "ttnn.experimental.">(
        mod,
        R"doc(
            Bench-only consumer companion to ttnn.dram_prefetcher. Builds and enqueues a
            program that loads a discard receiver kernel on each receiver core of the supplied
            GCB, looping num_iters times of wait_front(1)+pop_front(1). Used to measure
            prefetcher push bandwidth without matmul receiver-side effects.
            Used for debugging purposes, please avoid to use in any production code.

            Args:
                mesh_device: the MeshDevice to enqueue on.
                num_iters (int): total pages each receiver should consume (= num_layers * num_blocks).
                page_size_bytes (int): receiver-side page size; must match what the sender pushes
                    (in0_block_w_tiles * n_tiles_per_receiver * tile_bytes).
                global_cb (GlobalCircularBuffer): either a worker-sender or DRAM-sender GCB.
        )doc",
        &test_dram_prefetcher_consumer,
        nb::arg("mesh_device"),
        nb::arg("num_iters"),
        nb::arg("page_size_bytes"),
        nb::kw_only(),
        nb::arg("global_cb"));

    ttnn::bind_function<"test_dram_prefetcher_contention_consumer", "ttnn.experimental.">(
        mod,
        R"doc(
            Benchmark-only Tensor Prefetcher consumer. After each prefetched page
            arrives, every receiver issues an ordinary NoC DRAM read against the same
            bank before releasing the page. This keeps the Device-side Tensor
            Prefetcher MPFE port and the ordinary-operation MPFE port contended.

            Args:
                mesh_device: the MeshDevice to enqueue on.
                ordinary_source_tensor (ttnn.Tensor): DRAM tensor read by the competing
                    ordinary-operation traffic.
                timing_tensor (ttnn.Tensor): row-major L1-sharded uint32 tensor with one
                    eight-word shard per receiver; receives cycle counters.
                num_iters (int): prefetched pages consumed by each receiver.
                page_size_bytes (int): receiver-side prefetched page size.
                ordinary_read_bytes (int): bytes read through the ordinary DRAM path
                    after each prefetched page arrives.
                global_cb (GlobalCircularBuffer): DRAM-sender GCB to consume.
        )doc",
        &test_dram_prefetcher_contention_consumer,
        nb::arg("mesh_device"),
        nb::arg("ordinary_source_tensor"),
        nb::arg("timing_tensor"),
        nb::arg("num_iters"),
        nb::arg("page_size_bytes"),
        nb::arg("ordinary_read_bytes"),
        nb::kw_only(),
        nb::arg("global_cb"));

    ttnn::bind_function<"test_dram_prefetcher_validator", "ttnn.experimental.">(
        mod,
        R"doc(
            Byte-for-byte validator receiver: for each pushed page, reads the expected
            tile range from source_tensor via TensorAccessor and memcmps against the
            received bytes. On any mismatch DPRINTs (layer, block, word) + the diverging
            bytes and hangs the core; otherwise DPRINTs progress and a final OK. After
            the expected num_layers * ring_size iterations, polls briefly for extra
            pages (sender overshoot) and hangs on overflow.

            The kernel derives its expected bytes from the (bank, receiver, block) ->
            tile-range mapping documented in
            tt_metal/impl/buffers/prefetcher_matmul_design.md §3 ("Per-block source tiles"). Used for
            debugging purposes; please avoid in any production code.

            Args:
                mesh_device: the MeshDevice to enqueue on.
                source_tensor (ttnn.Tensor): the same width-sharded DRAM tensor the
                    prefetcher is being driven with.
                num_layers (int): number of layers the prefetcher will push.
                print_stride (int): DPRINT every Nth iter; first/last always logged. 0 = first/last only.
                global_cb (GlobalCircularBuffer): worker-sender or DRAM-sender GCB.
                streaming (bool): when True, expect the streaming prefetcher's ring-rotated
                    delivery (block at FIFO position p is physical block (lead_block + p) mod
                    num_blocks). Must match the streaming flag passed to the prefetcher.
                    Defaults to False.
                rotation (List[int]): per-receiver streaming rotation indexed by global ring
                    position (must match the rotation queued to the prefetcher). Empty == identity
                    (lead_block = ring_pos), the natural topology order. Defaults to empty.
        )doc",
        &test_dram_prefetcher_validator,
        nb::arg("mesh_device"),
        nb::arg("source_tensor"),
        nb::arg("num_layers"),
        nb::arg("print_stride"),
        nb::kw_only(),
        nb::arg("global_cb"),
        nb::arg("streaming") = false,
        nb::arg("rotation") = std::vector<uint32_t>{});
}

}  // namespace ttnn::operations::experimental::test
