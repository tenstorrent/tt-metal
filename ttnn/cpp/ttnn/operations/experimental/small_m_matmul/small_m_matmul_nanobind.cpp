// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "small_m_matmul_nanobind.hpp"

#include <optional>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "small_m_matmul.hpp"
#include "device/small_m_matmul_config.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/types.hpp"
#include "ttnn/device.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::operations::experimental::small_m_matmul::detail {

void bind_small_m_matmul(nb::module_& mod) {
    ttnn::bind_function<"small_m_matmul", "ttnn.experimental.">(
        mod,
        R"doc(
        small_m_matmul(input_tensor, weight_tensor, config=None, *, bias_tensor=None, fused_activation=None, fused_ternary_scalar=None, fused_ternary_input_a=None, fused_ternary_input_b=None)

        Experimental DRAM-bandwidth-optimal matrix multiply (A @ B) for low-arithmetic-intensity
        low-arithmetic-intensity shapes with small M and wide N (M << N), with optional fused epilogue.
        The transpose case (N << M) is NOT supported: narrow N cannot fill the 8-bank in1 width shard
        (see SUPPORTED DOMAIN below), so those shapes are refused at program build. Numerics are FIXED:
        BFLOAT16 in/out, HiFi2 math, FP32 dest accumulation, DRAM-interleaved output (there are no
        dtype / memory_config / compute_kernel_config arguments).

        Fusions (applied at the output/compute stage; for split-K they run exactly once after reduction):
          - bias:       Y = A@B + bias
          - activation: Y = activation(A@B + bias)                (bias applied before activation)
          - addcmul:    Y = residual + scalar*(A@B + bias)*gate   (activation and addcmul are exclusive)

        The activation A ([.., M, K]) is DRAM interleaved. The weight B ([.., K, N]) must be DRAM
        WIDTH_SHARDED across 8 banks — build its MemoryConfig with
        ``ttnn.create_small_m_weight_memory_config``. Output is [.., M, N] in TILE layout.

        SUPPORTED DOMAIN
        ----------------
        Blackhole only. Rank >= 2 with all leading dims 1 (no batching), TILE layout, BFLOAT16 in/out.
        Two SHAPE-DOMAIN limits are structural rather than tuning gaps, so a shape hitting either can only
        be served by a standard matmul:

        1. N must be wide enough to fill the 8-bank in1 width shard: ``7*ceil(Nt/8) < Nt`` where
           ``Nt = ceil(N/32)``, else the trailing banks would be entirely padding. The smallest workable
           widths are Nt = 8, 15, 16, 22, 23, 24, 29.. (N = 256, 480, 512, 704, 736, 768, 928..). So narrow
           N — e.g. N=8 or N=128 — is out of domain.
        2. The in0 k-slice is kept L1-RESIDENT, so cb0 alone is ~``(Mt/Sm)*(Kt/Pk)`` tiles while core
           feasibility caps ``8*Pk*Ns*Sm <= 104`` cores (``Pk*Ns*Sm <= 13``). Large Mt with deep K cannot
           satisfy both: e.g. Mt=152 (M=4864), Kt=128 (K=4096) would need ``Sm*Pk >~ 32``.

        Subblocks are additionally bounded by the 4-tile fp32 DST limit, and ``small_m_matmul_split``
        requires ``dim == -1`` with N divisible by ``chunks`` and each chunk a multiple of TILE_WIDTH.

        FALLBACK BEHAVIOUR
        ------------------
        There is no silent fallback to another matmul: an out-of-domain shape raises at program build with a
        message naming which of the two limits above it hit and why, so it fails loudly rather than running
        slowly or wrongly. Within the domain, ``config=None`` degrades gracefully instead of failing:

        - the measured lookup table is consulted first; if a table entry no longer fits L1 once fusion CBs
          are added, it is skipped and the cost model is used, yielding a slower-but-valid config rather
          than an error;
        - if no ``m_slices == 1`` config fits L1, an M-split (``Sm > 1``) config is searched as a rescue
          before giving up — this is what serves large-Mt deep-K shapes such as 512x15360x768.

        An explicit ``config`` is NOT rescued: it is validated and rejected if infeasible, because silently
        substituting a different config would invalidate any measurement taken against it.

        Parameters
        ----------
        input_tensor : ttnn.Tensor
            Activation A. TILE layout, BFLOAT16, on device. Shape [.., M, K] (leading dims must be 1).
        weight_tensor : ttnn.Tensor
            Weight B. TILE layout, BFLOAT16, on device, DRAM WIDTH_SHARDED. Shape [.., K, N].
        config : Optional[SmallMMatmulConfig], default: None
            Manual execution config. None => auto-select via the FLUX/LTX picker.
        bias_tensor : Optional[ttnn.Tensor], default: None
            Row-broadcast bias [.., 1, N] / [.., N], TILE, on device.
        fused_activation : Optional[UnaryWithParam], default: None
            Fused unary activation applied after bias.
        fused_ternary_scalar : Optional[float], default: None
            addcmul scalar. If set, fused_ternary_input_a (residual) and fused_ternary_input_b (gate)
            are required and fused_activation must be None.
        fused_ternary_input_a : Optional[ttnn.Tensor], default: None
            addcmul residual [M, N], BFLOAT16, TILE.
        fused_ternary_input_b : Optional[ttnn.Tensor], default: None
            addcmul gate [1, N] (broadcast) or [M, N] (full), TILE.

        Returns
        -------
        ttnn.Tensor
            Output tensor [.., M, N], TILE layout, BFLOAT16, DRAM interleaved.
        )doc",
        &ttnn::experimental::small_m_matmul,
        nb::arg("input_tensor"),
        nb::arg("weight_tensor"),
        nb::arg("config") = nb::none(),
        nb::kw_only(),
        nb::arg("bias_tensor") = nb::none(),
        nb::arg("fused_activation") = nb::none(),
        nb::arg("fused_ternary_scalar") = nb::none(),
        nb::arg("fused_ternary_input_a") = nb::none(),
        nb::arg("fused_ternary_input_b") = nb::none());

    ttnn::bind_function<"small_m_matmul_split", "ttnn.experimental.">(
        mod,
        R"doc(
        small_m_matmul_split(input_tensor, weight_tensor, chunks, dim=-1, config=None, *, bias_tensor=None, fused_activation=None, fused_ternary_scalar=None, fused_ternary_input_a=None, fused_ternary_input_b=None)

        Output column-split sibling of small_m_matmul. Returns `chunks` equal-width [.., M, N/chunks]
        output tensors, written directly (no full-output materialize + slice). Requires dim==-1,
        N % chunks == 0 and N/chunks tile-aligned. All fusions compose with chunking. Fixed numerics
        (BFLOAT16, HiFi2, FP32 acc, DRAM interleaved).

        Returns
        -------
        List[ttnn.Tensor]
            `chunks` output tensors [.., M, N/chunks], TILE layout.
        )doc",
        &ttnn::experimental::small_m_matmul_split,
        nb::arg("input_tensor"),
        nb::arg("weight_tensor"),
        nb::arg("chunks"),
        nb::arg("dim") = -1,
        nb::arg("config") = nb::none(),
        nb::kw_only(),
        nb::arg("bias_tensor") = nb::none(),
        nb::arg("fused_activation") = nb::none(),
        nb::arg("fused_ternary_scalar") = nb::none(),
        nb::arg("fused_ternary_input_a") = nb::none(),
        nb::arg("fused_ternary_input_b") = nb::none());

    auto py_config = nb::class_<SmallMMatmulConfig>(
                         mod,
                         "SmallMMatmulConfig",
                         R"doc(
                         Configuration for the small-M matmul operation (all values in tiles / slice counts).
                         A manual config must set the scheduling fields explicitly; there is deliberately NO
                         zero-argument constructor (the old all-ones default builds only 8 workers, which is
                         invalid for most shapes). Use config=None for the auto-picker instead.
                         )doc")
                         .def(
                             nb::init<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>(),
                             nb::kw_only(),
                             nb::arg("k_slices"),
                             nb::arg("n_slices"),
                             nb::arg("m_slices"),
                             nb::arg("k_block_tiles"),
                             nb::arg("n_subblock_tiles"));

    // Read-only: a config is immutable after construction (set all fields via the constructor).
    py_config.def_ro("k_slices", &SmallMMatmulConfig::k_slices, "");
    py_config.def_ro("n_slices", &SmallMMatmulConfig::n_slices, "");
    py_config.def_ro("m_slices", &SmallMMatmulConfig::m_slices, "");
    py_config.def_ro("k_block_tiles", &SmallMMatmulConfig::k_block_tiles, "");
    py_config.def_ro("n_subblock_tiles", &SmallMMatmulConfig::n_subblock_tiles, "");
    // Build the repr manually (this file is compiled standalone / SKIP_UNITY, so the generic
    // reflection-based fmt formatter for aggregates is not in scope here).
    py_config.def("__repr__", [](const SmallMMatmulConfig& c) {
        return "SmallMMatmulConfig(k_slices=" + std::to_string(c.k_slices) +
               ", n_slices=" + std::to_string(c.n_slices) + ", m_slices=" + std::to_string(c.m_slices) +
               ", k_block_tiles=" + std::to_string(c.k_block_tiles) +
               ", n_subblock_tiles=" + std::to_string(c.n_subblock_tiles) + ")";
    });

    // Build the canonical DRAM width-sharded MemoryConfig for the in1 (weight) tensor.
    mod.def(
        "create_small_m_weight_memory_config",
        [](const ttnn::Shape& weight_shape, tt::tt_metal::DataType dtype, ttnn::MeshDevice* device) {
            return ttnn::experimental::prim::create_small_m_weight_memory_config(weight_shape, dtype, device);
        },
        nb::arg("weight_shape"),
        nb::arg("dtype"),
        nb::arg("device"),
        R"doc(
        create_small_m_weight_memory_config(weight_shape, dtype, device)

        Return the DRAM WIDTH_SHARDED (8-bank, ROW_MAJOR) MemoryConfig required for the small-M matmul
        weight tensor. K is padded up to a multiple of 8 tiles, N up to a multiple of 8 tiles; the shard
        spec depends only on (K, N), never on the execution config.
        )doc");
}

}  // namespace ttnn::operations::experimental::small_m_matmul::detail
