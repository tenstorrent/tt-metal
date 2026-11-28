// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "pair_pool_nanobind.hpp"

#include <exception>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "device/pair_pool_device_operation_types.hpp"
#include "pair_pool.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::pair_pool {

namespace {

// The Python op refused with `assert`; its C++ refusals reach Python as the builtin AssertionError with the same
// message. Any other exception falls through to the next translator.
void translate_support_refusal(const std::exception_ptr& exception, void* /*payload*/) {
    try {
        std::rethrow_exception(exception);
    } catch (const PairPoolAssertion& refusal) {
        PyErr_SetString(PyExc_AssertionError, refusal.what());
    } catch (...) {
        throw;  // not a refusal: the next translator takes it
    }
}

// A caller's output_tensor is returned as is (the Python op returned the very object it was given): it arrives as the
// pointer to the C++ tensor its Python object holds, and returning that pointer by reference makes nanobind hand back
// the existing Python object instead of wrapping a copy.
Tensor* pair_pool_into(
    const Tensor& x,
    Tensor* output_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<CoreCoord>& max_grid) {
    ttnn::pair_pool(x, *output_tensor, memory_config, max_grid);
    return output_tensor;
}

}  // namespace

void bind_pair_pool_operation(nb::module_& mod) {
    nb::register_exception_translator(&translate_support_refusal);

    ttnn::bind_function<"pair_pool">(
        mod,
        R"doc(
        Softmax-gated pooling of row pairs: x [.., R, 2 hd] = [kv | gate] per row -> out [.., R / 2, hd],
        out[t] = kv[2t] w0 + kv[2t + 1] w1, (w0, w1) = softmax(gate[2t], gate[2t + 1]) per column, in fp32.
        On a mesh, the same program runs on every chip.

        Args:
            x (ttnn.Tensor): float32, TILE layout, DRAM interleaved; R % 64 == 0 (unpadded), hd % 32 == 0.

        Keyword Args:
            output_tensor (ttnn.Tensor, optional): the caller's buffer of exactly the output spec (float32, TILE,
                interleaved), returned as is. Default None allocates.
            memory_config (ttnn.MemoryConfig, optional): placement of an allocated output. Default DRAM.
            max_grid (ttnn.CoreCoord, optional): the cores are taken from the rectangle (0, 0) .. (max_grid - 1).
                Default None uses the whole compute grid.

        Returns:
            ttnn.Tensor: [.., R / 2, hd] float32 TILE.

        Raises:
            AssertionError: an input outside the contract above.
        )doc",
        // tried first: a caller's output_tensor (None does not match a pointer argument without .none())
        ttnn::overload_t(
            &pair_pool_into,
            nb::arg("x"),
            nb::kw_only(),
            nb::arg("output_tensor"),
            nb::arg("memory_config") = nb::none(),
            nb::arg("max_grid") = nb::none(),
            nb::rv_policy::reference),
        ttnn::overload_t(
            &ttnn::pair_pool,
            nb::arg("x"),
            nb::kw_only(),
            nb::arg("output_tensor") = nb::none(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("max_grid") = nb::none()));
}

}  // namespace ttnn::operations::pair_pool
