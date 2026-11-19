// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "relay_chunk_scan_nanobind.hpp"

#include "relay_chunk_scan.hpp"
#include "ttnn-nanobind/bind_function.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace ttnn::operations::experimental::kda::relay_chunk_scan::detail {

void bind_relay_chunk_scan(nb::module_& mod) {
    ttnn::bind_function<"relay_chunk_scan", "ttnn.experimental.kda.">(
        mod,
        R"doc(
        The sequence-parallel prefix as a relay along the SP axis, with the chunk's final carry: rank ``p``
        (chronological position, from ``actual_start``) takes its entry state ``S_p`` from its predecessor, computes
        ``S_{p+1} = A_p S_p + B_p = S_p + (E_p S_p + B_p)`` from its packed complement-form summary ``[E | B]``
        (``E = A - I``) and forwards it, one head per head's first core and one tile column at a time; the last rank
        sends the final state ``P`` on to the first. ``S_p`` and ``B_p`` are added exactly in FP32 DST (``B_p`` as
        ``I B_p`` in the matmul); only ``E_p S_p`` passes through the matmul's source registers at its fidelity.

        Every core of a head then forms its chunks' outputs ``x = F0 + (FI - F0) @ seed`` from the summary's
        ``terms`` (the entry correction), the seed being ``S_p``, or ``P`` for the chunks of a separated tail,
        and writes each row's gated RMSNorm ``x / sqrt(mean(x^2) + norm_epsilon) * norm_weight *
        sigmoid(gate)``, token-major. The gate ``gate_input @ gate_weight`` is formed in the same program at
        ``gate_math_fidelity``, while the relay's hops are on their way: each head's other cores form their chunks'
        rows and a helper core per head those of its first core. The first rank forms the final carry as ``P`` lands: ``P + (E P + B)``
        with its final transition ``final_e`` / ``final_b`` when the interval ends in a separated tail, else
        ``P``. It multicasts it from L1 along the SP axis into every rank's carry output, and every other rank
        waits for its arrival, which separates consecutive calls across the SP line.

        ``cores`` lists the same number of cores per head, head by head (the first of each also runs the relay's
        hop), then two sender cores per link, then a helper core per head (any other core, a sender included). The ``landing`` buffer and the global ``semaphores`` outlive the call:
        a peer writes them before this rank's call starts.

        Args:
            packed (ttnn.Tensor): BFLOAT16 TILE-layout interleaved ``[1, B*H, K, 2K]`` rows of this rank's
                packed transition in complement form ``[E | B]``, ``E = A - I``.
            terms (ttnn.Tensor): FLOAT32 TILE-layout chunk terms, each chunk's output for a zero seed ``F(0)`` and
                an identity seed ``F(I)``: ``[B*H * cores per head, 2, rows per core, K]`` height-sharded in L1 over
                the heads' cores in order (each core's ``F(0)`` rows, then its ``F(I)`` rows), or interleaved
                ``[B*H, 2, local_rows, K]``, streamed.
            initial_state (ttnn.Tensor): FLOAT32 TILE-layout interleaved ``[B*H, K, K]`` state the
                chronologically first rank enters at, or with ``slab_index`` a ``[entries, H, K, K]`` DRAM slab.
            final_e (ttnn.Tensor): FLOAT32 TILE-layout interleaved ``[B*H, K, K]`` E of this rank's final
                transition.
            final_b (ttnn.Tensor): FLOAT32 TILE-layout interleaved ``[B*H, K, K]`` B of the same transition.
            gate_input (ttnn.Tensor): BF16 TILE-layout interleaved ``[B, local_rows, X]`` input the output gate
                projects (the layer's hidden states).
            gate_weight (ttnn.Tensor): TILE-layout interleaved ``[X, H*K]`` output gate projection.
            norm_weight (ttnn.Tensor): BF16 TILE-layout interleaved ``[K]`` RMSNorm weight.
            actual_start (ttnn.Tensor): Replicated UINT32 row-major scalar with the absolute position of the
                chunk's first token. Keep its address stable and update its contents before replay of a
                captured trace.
            landing (ttnn.Tensor): FLOAT32 TILE-layout ``[2*B*H*K, K]`` height-sharded in L1 over each head's
                first core, two ``K x K`` states per core: the entry state the predecessor's sender writes and the
                final state the last rank writes on the first rank.
            semaphores (list[ttnn.GlobalSemaphore]): Eight zeroed counters on every core of the program: the
                entry-state, final-state and final-carry arrivals, then the output gate's x and W ready and valid
                counts and its landed count.

        Keyword Args:
            local_rows (int): Positive, 32-aligned token rows per SP device.
            num_links (int): Fabric links to the successor, two sender cores each.
            cores (list[ttnn.CoreCoord]): The heads' cores, head by head (as many per head, dividing its chunks),
                then two senders per link, then a helper per head (distinct from the heads' cores and each other).
            num_heads (int): Heads per batch ``H``; ``B*H`` heads are ``B`` batches.
            actual_end (ttnn.Tensor, optional): Replicated UINT32 exclusive valid end, which decides whether the
                interval ends in a separated tail. Omitted, the chunk runs to its full capacity.
            sequence_parallel_axis (int, optional): Mesh axis partitioning the sequence, of at least two ranks
                (the last rank's hand-off to the first crosses the axis when it does not wrap).
            gate_local (ttnn.Tensor, optional): BF16 TILE-layout interleaved ``[B, local_rows, X / TP]``, this rank's
                TP slice of ``gate_input`` (TP the mesh axis other than ``sequence_parallel_axis``), read in place of
                those columns of ``gate_input``, which then need not hold them.
            gate_math_fidelity (ttnn.MathFidelity, optional): The output gate's fidelity. Defaults to HiFi2.
            norm_epsilon (float, optional): Added to the mean square. Defaults to 1e-6.
            output_dtype (ttnn.DataType, optional): FLOAT32 or BFLOAT16 output. Defaults to BFLOAT16.
            memory_config (ttnn.MemoryConfig, optional): Interleaved memory of the output. Defaults to DRAM.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): The matmuls' fidelity; their
                products accumulate in DST. Requires ``fp32_dest_acc_en`` and HiFi3 or above. Defaults to HiFi3
                with FP32 destination accumulation.
            slab_index (ttnn.Tensor, optional): Replicated UINT32 scalar, read on device: the first rank enters at
                slab entry ``slab_index * slab_index_scale + slab_index_offset``, or zeros when ``actual_start`` is 0,
                and every rank's final carry is written back into that entry, in place (nothing past the slab).
            slab_index_scale (int, optional): Defaults to 1.
            slab_index_offset (int, optional): Defaults to 0.

        Returns:
            tuple[ttnn.Tensor, ttnn.Tensor]: TILE-layout ``[B, local_rows, H*K]`` gated-norm outputs, token-major,
                and the FLOAT32 ``[B*H, K, K]`` final carry in DRAM on every rank (with ``slab_index``, the slab
                itself).

        Note:
            ``K`` must be positive, tile aligned and at most the DST's tiles wide; the fabric payload must hold
            an FP32 tile. Inputs are not modified, except a slab's entry.
        )doc",
        &ttnn::experimental::kda::relay_chunk_scan,
        nb::arg("packed").noconvert(),
        nb::arg("terms").noconvert(),
        nb::arg("initial_state").noconvert(),
        nb::arg("final_e").noconvert(),
        nb::arg("final_b").noconvert(),
        nb::arg("gate_input").noconvert(),
        nb::arg("gate_weight").noconvert(),
        nb::arg("norm_weight").noconvert(),
        nb::arg("actual_start").noconvert(),
        nb::arg("landing").noconvert(),
        nb::arg("semaphores"),
        nb::kw_only(),
        nb::arg("local_rows"),
        nb::arg("num_links"),
        nb::arg("cores"),
        nb::arg("num_heads"),
        nb::arg("actual_end") = nb::none(),
        nb::arg("sequence_parallel_axis") = 0,
        nb::arg("gate_local") = nb::none(),
        nb::arg("gate_math_fidelity") = tt::tt_metal::MathFidelity::HiFi2,
        nb::arg("norm_epsilon") = 1e-6f,
        nb::arg("output_dtype") = tt::tt_metal::DataType::BFLOAT16,
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("slab_index") = nb::none(),
        nb::arg("slab_index_scale") = 1,
        nb::arg("slab_index_offset") = 0);
}

}  // namespace ttnn::operations::experimental::kda::relay_chunk_scan::detail
