// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "groupnorm_sc_N_1_HW_C_b_nanobind.hpp"

#include <cstdint>
#include <exception>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "device/groupnorm_sc_N_1_HW_C_b_device_operation_types.hpp"
#include "groupnorm_sc_N_1_HW_C_b.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::groupnorm_sc_N_1_HW_C_b {

namespace {

void raise_contract(const char* type_name, const char* message) {
    const nb::object type = nb::module_::import_("ttnn.operations._op_contract").attr(type_name);
    PyErr_SetString(type.ptr(), message);
}

// A support refusal reaches Python as the ttnn.operations._op_contract exception of the same name, the
// NotImplementedError subclasses the eval harness recognizes as a deliberate refusal; the planner's refusal is
// Python's builtin NotImplementedError. Any other exception falls through to the next translator.
void translate_support_refusal(const std::exception_ptr& exception, void* /*payload*/) {
    try {
        std::rethrow_exception(exception);
    } catch (const UnsupportedAxisValue& refusal) {
        raise_contract("UnsupportedAxisValue", refusal.what());
    } catch (const ExcludedCell& refusal) {
        raise_contract("ExcludedCell", refusal.what());
    } catch (const PlanNotImplemented& refusal) {
        PyErr_SetString(PyExc_NotImplementedError, refusal.what());
    } catch (...) {
        throw;  // not a refusal: the next translator takes it
    }
}

std::optional<int32_t> tuning_int(const nb::dict& tuning, const char* key) {
    if (!tuning.contains(key)) {
        return std::nullopt;
    }
    const nb::object value = tuning[key];
    if (value.is_none()) {
        return std::nullopt;
    }
    return nb::cast<int32_t>(nb::int_(value));
}

struct TuningOverrides {
    std::optional<int32_t> max_resident_chunks;
    std::optional<int32_t> chunk_tiles_target;
    std::optional<int32_t> max_cores;
    bool chunk_zones = false;
};

// The package's TUNING_OVERRIDES (tests monkeypatch it) is read on every call. ttnn::bind_function calls the bound
// function with the GIL released, so the dict is read under a re-acquired GIL, and no Python object outlives the call
// (a function-static one would be released after the interpreter finalizes).
TuningOverrides read_tuning_overrides() {
    const nb::gil_scoped_acquire gil;
    const nb::dict tuning =
        nb::cast<nb::dict>(nb::module_::import_("ttnn.operations.groupnorm_sc_N_1_HW_C_b").attr("TUNING_OVERRIDES"));
    bool chunk_zones = false;
    if (tuning.contains("chunk_zones")) {
        const nb::object zones = tuning["chunk_zones"];
        chunk_zones = !zones.is_none() && nb::cast<bool>(nb::bool_(zones));
    }
    return TuningOverrides{
        .max_resident_chunks = tuning_int(tuning, "max_resident_chunks"),
        .chunk_tiles_target = tuning_int(tuning, "chunk_tiles_target"),
        .max_cores = tuning_int(tuning, "max_cores"),
        .chunk_zones = chunk_zones,
    };
}

Tensor groupnorm_with_tuning(
    const Tensor& input_tensor,
    int num_groups,
    const std::optional<Tensor>& gamma,
    const std::optional<Tensor>& beta,
    float eps,
    bool in_place,
    const std::optional<Tensor>& output_tensor) {
    const TuningOverrides tuning = read_tuning_overrides();
    return ttnn::groupnorm_sc_N_1_HW_C_b(
        input_tensor,
        num_groups,
        gamma,
        beta,
        eps,
        in_place,
        output_tensor,
        tuning.max_resident_chunks,
        tuning.chunk_tiles_target,
        tuning.max_cores,
        tuning.chunk_zones);
}

}  // namespace

void bind_groupnorm_sc_N_1_HW_C_b_operation(nb::module_& mod) {
    nb::register_exception_translator(&translate_support_refusal);

    ttnn::bind_function<"groupnorm_sc_N_1_HW_C_b">(
        mod,
        R"doc(
        GroupNorm over an (N, 1, HW, C) channels-last activation: per image and group of C / num_groups channels,
        y = (x - mean) * rstd * gamma + beta, rstd = 1 / sqrt(var + eps).

        DRAM interleaved input -> DRAM interleaved TILE output. L1 BLOCK_SHARDED input -> an output with the input's
        layout and memory config; in_place=True (BLOCK_SHARDED only) writes the result over the input's shard and
        returns the input tensor.

        Args:
            input_tensor (ttnn.Tensor): (N, 1, HW, C), bfloat16 / float32 / bfloat8_b, TILE or ROW_MAJOR.
            num_groups (int): number of groups; must divide C.

        Keyword Args:
            gamma (ttnn.Tensor, optional): (1, 1, 1, C) scale. Defaults to None.
            beta (ttnn.Tensor, optional): (1, 1, 1, C) shift. Defaults to None.
            eps (float): variance epsilon, > 0. Defaults to 1e-5.
            in_place (bool): overwrite the block-sharded input. Defaults to False.
            output_tensor (ttnn.Tensor, optional): preallocated output matching the output spec. Defaults to None.

        Returns:
            ttnn.Tensor: the normalized tensor.

        Raises:
            ValueError: structurally invalid inputs (rank, shape[1], num_groups, gamma / beta shape, eps, in_place).
            UnsupportedAxisValue, ExcludedCell: inputs outside the support contract.
            NotImplementedError: no feasible plan for the shape.
        )doc",
        &groupnorm_with_tuning,
        nb::arg("input_tensor"),
        nb::arg("num_groups"),
        nb::kw_only(),
        nb::arg("gamma") = nb::none(),
        nb::arg("beta") = nb::none(),
        nb::arg("eps") = 1e-5f,
        nb::arg("in_place") = false,
        nb::arg("output_tensor") = nb::none());
}

}  // namespace ttnn::operations::groupnorm_sc_N_1_HW_C_b
