// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "grouped_matmul_nanobind.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <tt_stl/indestructible.hpp>

#include "device/grouped_matmul_device_operation_types.hpp"
#include "grouped_matmul.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::grouped_matmul {

namespace {

// The bank layouts already converted, by Python object. bank.BankLayout is a frozen dataclass of tuples, so a layout
// object converts to the same value on every call; a model passes the same object on every call, and reading its slot
// table attribute by attribute each time cost host time (the Python op cached its layout key for the same reason).
// Each entry holds a reference to its object, so the object's id is not reused while it is cached. Bounded like the
// Python op's prepared-program cache. Indestructible: the references are not released after the interpreter is gone.
constexpr std::size_t kBankLayoutCacheCap = 64;
using BankLayoutCache = std::unordered_map<PyObject*, std::pair<nanobind::object, BankLayout>>;

BankLayoutCache& bank_layout_cache() {
    static ttsl::Indestructible<BankLayoutCache> cache;
    return cache.get();
}

}  // namespace

}  // namespace ttnn::operations::grouped_matmul

// w_layout arrives as the Python bank.BankLayout (planned by bank.plan, a frozen dataclass): read its fields into
// the C++ BankLayout, once per layout object (bank_layout_cache). Only Python -> C++ is needed. The
// conversion runs with the GIL held (before the call releases it), which serializes the cache.
namespace nanobind::detail {

template <>
struct type_caster<ttnn::operations::grouped_matmul::BankLayout> {
    NB_TYPE_CASTER(ttnn::operations::grouped_matmul::BankLayout, const_name("BankLayout"))

    bool from_python(handle src, uint8_t /*flags*/, cleanup_list* /*cleanup*/) noexcept {
        try {
            auto& cache = ttnn::operations::grouped_matmul::bank_layout_cache();
            if (const auto it = cache.find(src.ptr()); it != cache.end()) {
                value = it->second.second;
                return true;
            }
            auto u32 = [](const handle& h, const char* name) { return nanobind::cast<uint32_t>(h.attr(name)); };
            Value v;
            v.G = u32(src, "G");
            v.KT = u32(src, "KT");
            v.NT = u32(src, "NT");
            v.banks = u32(src, "banks");
            v.cores_per_bank = u32(src, "cores_per_bank");
            for (const handle s : src.attr("slots")) {
                const nanobind::object core = s.attr("core");
                v.slots.push_back(ttnn::operations::grouped_matmul::BankSlot{
                    .bank = u32(s, "bank"),
                    .core_x = nanobind::cast<uint32_t>(core[0]),
                    .core_y = nanobind::cast<uint32_t>(core[1]),
                    .g = u32(s, "g"),
                    .n0 = u32(s, "n0"),
                    .pn = u32(s, "pn"),
                    .tile0 = u32(s, "tile0"),
                    .cluster = u32(s, "cluster"),
                    .sender = nanobind::cast<bool>(s.attr("sender")),
                });
            }
            for (const handle r : src.attr("rect")) {
                v.rect.push_back(ttnn::operations::grouped_matmul::BankRect{
                    .x0 = nanobind::cast<uint32_t>(r[0]),
                    .y0 = nanobind::cast<uint32_t>(r[1]),
                    .x1 = nanobind::cast<uint32_t>(r[2]),
                    .y1 = nanobind::cast<uint32_t>(r[3]),
                });
            }
            for (const handle c : src.attr("sender")) {
                v.sender.push_back(ttnn::operations::grouped_matmul::BankCore{
                    .x = nanobind::cast<uint32_t>(c[0]),
                    .y = nanobind::cast<uint32_t>(c[1]),
                });
            }
            if (cache.size() >= ttnn::operations::grouped_matmul::kBankLayoutCacheCap) {
                cache.clear();
            }
            cache.emplace(src.ptr(), std::make_pair(nanobind::borrow(src), v));
            value = std::move(v);
            return true;
        } catch (...) {
            PyErr_Clear();
            return false;
        }
    }

    static handle from_cpp(const Value& /*value*/, rv_policy /*policy*/, cleanup_list* /*cleanup*/) noexcept {
        return {};
    }
};

}  // namespace nanobind::detail

namespace ttnn::operations::grouped_matmul {

namespace {

// The Python op refused only through `assert`, so a refusal reaches Python as AssertionError. Any other exception
// falls through to the next translator.
void translate_support_refusal(const std::exception_ptr& exception, void* /*payload*/) {
    try {
        std::rethrow_exception(exception);
    } catch (const GroupedMatmulAssertion& refusal) {
        PyErr_SetString(PyExc_AssertionError, refusal.what());
    }
}

// The bound call: ttnn::grouped_matmul, returning the caller's output_tensor object itself when one is given (the
// Python op returned it as is: grouped_matmul(..., output_tensor=out) is out); a Tensor returned by value would be
// a new Python object on the same buffer. bind_function releases the GIL across this body, so the Python objects
// are touched only under a re-acquired GIL, and the op itself runs without it.
nb::object grouped_matmul_returning_output(
    const Tensor& x,
    const Tensor& w,
    const std::optional<DataType>& dtype,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    const nb::object& output_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<CoreCoord>& max_grid,
    std::optional<uint32_t> block_w,
    uint32_t depth,
    uint32_t rt,
    const std::optional<BankLayout>& w_layout,
    uint32_t w_inflight,
    uint32_t x_ahead,
    bool one_group,
    const std::vector<std::string>& diag) {
    std::optional<Tensor> output;
    {
        nb::gil_scoped_acquire gil;
        if (!output_tensor.is_none()) {
            output = nb::cast<Tensor>(output_tensor);
        }
    }
    Tensor y = ttnn::grouped_matmul(
        x,
        w,
        dtype,
        compute_kernel_config,
        output,
        memory_config,
        max_grid,
        block_w,
        depth,
        rt,
        w_layout,
        w_inflight,
        x_ahead,
        one_group,
        diag);
    nb::gil_scoped_acquire gil;
    return output_tensor.is_none() ? nb::cast(std::move(y)) : nb::borrow(output_tensor);
}

}  // namespace

void bind_grouped_matmul_operation(nb::module_& mod) {
    nb::register_exception_translator(&translate_support_refusal);

    ttnn::bind_function<"grouped_matmul">(
        mod,
        R"doc(
        A batched matmul with one weight per group: y[g] = x[g] @ w[g], one grid row per group.

        Args:
            x (ttnn.Tensor): [1, G, M, K], TILE, DRAM interleaved.
            w (ttnn.Tensor): [1, G, K, N], TILE, DRAM interleaved (or, with ``w_layout``, the layout's
                WIDTH_SHARDED DRAM weight from bank.upload).

        Keyword Args:
            dtype (ttnn.DataType, optional): the output dtype (default x's).
            compute_kernel_config (optional): math fidelity, fp32 DEST and approx mode (default HiFi2, fp32 DEST).
            output_tensor (ttnn.Tensor, optional): the caller's buffer of exactly the output spec, returned as is.
            memory_config (ttnn.MemoryConfig, optional): the output placement (default DRAM interleaved).
            max_grid (ttnn.CoreCoord, optional): cores from (0, 0) .. (max_grid - 1) (default the whole grid).
            block_w (int, optional): K tiles a block (default the widest of 32 / 16 / 8 / 4 / 2 / 1 that fits L1).
            depth (int): K blocks of x and of w buffered a core (default 2, >= 2).
            rt (int): row tiles a DST block (default 1).
            w_layout (bank.BankLayout, optional): bank-local weights; every core reads only its own bank.
            w_inflight (int): bank path: K blocks of weights in flight (default 1, <= min(depth, 8)).
            x_ahead (int): bank path: x blocks the sender reads ahead (default 1, <= min(3, depth - 1)).
            one_group (bool): bank path: one kernel group for every core width (default False).
            diag (tuple of str): diagnostic stubs "no_w", "no_x", "no_compute" (timing probes only: wrong output).

        Returns:
            ttnn.Tensor: [1, G, M, N], TILE.

        Raises:
            AssertionError: an input outside the op's layout (e.g. "w_layout needs the bank tensor").
        )doc",
        &grouped_matmul_returning_output,
        nb::arg("x"),
        nb::arg("w"),
        nb::kw_only(),
        nb::arg("dtype") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("output_tensor") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("max_grid") = nb::none(),
        nb::arg("block_w") = nb::none(),
        nb::arg("depth") = 2u,
        nb::arg("rt") = 1u,
        nb::arg("w_layout") = nb::none(),
        nb::arg("w_inflight") = 1u,
        nb::arg("x_ahead") = 1u,
        nb::arg("one_group") = false,
        nb::arg("diag") = std::vector<std::string>{});
}

}  // namespace ttnn::operations::grouped_matmul
