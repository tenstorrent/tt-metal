// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Non-template JSON assembly for op_profiler.
// This file is compiled once and keeps nlohmann/json.hpp + tt-metalium/tt_metal.hpp
// out of the 1000+ TU include chain that goes through device_operation.hpp.

#include "tools/profiler/op_profiler_serialize.hpp"

#include <nlohmann/json.hpp>
#include <set>
#include <string_view>
#include <unordered_set>
#include <tt-metalium/tt_metal.hpp>
#include "ttnn/operation.hpp"
#include <enchantum/enchantum.hpp>
#include <fmt/format.h>

using json = nlohmann::json;

namespace tt::tt_metal::op_profiler {

// ---------------------------------------------------------------------------
// Internal helpers — only needed when Tracy is enabled
// ---------------------------------------------------------------------------

#if defined(TRACY_ENABLE)

static json tensor_meta_to_json(const TensorMeta& m) {
    json ret;
    if (m.is_device) {
        ret["storage_type"]["device_id"] = m.device_id;
        ret["storage_type"]["memory_config"]["buffer_type"] = m.buffer_type;
        ret["storage_type"]["memory_config"]["memory_layout"] = m.memory_layout;
    } else {
        ret["storage_type"] = m.storage_type_str;
    }
    ret["shape"]["W"] = m.shape_W;
    ret["shape"]["Z"] = m.shape_Z;
    ret["shape"]["Y"] = m.shape_Y;
    ret["shape"]["X"] = m.shape_X;
    ret["layout"] = m.layout;
    ret["dtype"] = m.dtype;
    return ret;
}

static json get_kernels_json(ChipId device_id, const Program& program) {
    std::vector<json> computeKernels;
    std::vector<json> datamovementKernels;

    IDevice* device = tt::tt_metal::detail::GetActiveDevice(device_id);

    json kernelSizes;
    // TODO(HalProcessorClassType): all the combinations can be queried from HAL instead of hardcoded here, but
    // currently HAL does not correctly report the number of processors under DM.
    // It should report (DM, 0) and (DM, 1), but instead it currently reports (DM, 0) and (DM+1, 0).
    // So hardcode for now, this is on par with previously hardcoded brisc, ncrisc, etc.
    kernelSizes["TENSIX_DM_0_max_kernel_size"] = 0;
    kernelSizes["TENSIX_DM_1_max_kernel_size"] = 0;
    kernelSizes["TENSIX_COMPUTE_0_max_kernel_size"] = 0;
    kernelSizes["TENSIX_COMPUTE_1_max_kernel_size"] = 0;
    kernelSizes["TENSIX_COMPUTE_2_max_kernel_size"] = 0;
    kernelSizes["ACTIVE_ETH_DM_0_max_kernel_size"] = 0;
    kernelSizes["ACTIVE_ETH_DM_1_max_kernel_size"] = 0;
    kernelSizes["IDLE_ETH_DM_0_max_kernel_size"] = 0;
    kernelSizes["IDLE_ETH_DM_1_max_kernel_size"] = 0;

    // Fused ops (e.g. DeepSeek decoder block) can have hundreds of kernels compiled
    // from the same source with different compile-time args.  Deduplicate by
    // (source, math_fidelity) for compute and by source for datamovement to keep the
    // profiler JSON well within Tracy's 64 KiB message limit.
    std::set<std::pair<std::string_view, std::string>> seenCompute;
    std::set<std::string_view> seenDatamovement;
    for (const auto& kernel : detail::collect_kernel_meta(program, device)) {
        auto processor_class = kernel.processor_class;
        if (processor_class == HalProcessorClassType::COMPUTE) {
            auto fidelityStr = enchantum::to_string(kernel.math_fidelity.value());
            if (seenCompute.emplace(kernel.source, fidelityStr).second) {
                json kernelObj;
                kernelObj["source"] = kernel.source;
                kernelObj["name"] = kernel.name;
                kernelObj["math_fidelity"] = fidelityStr;
                computeKernels.push_back(std::move(kernelObj));
            }
        } else {
            if (seenDatamovement.emplace(kernel.source).second) {
                json kernelObj;
                kernelObj["source"] = kernel.source;
                kernelObj["name"] = kernel.name;
                datamovementKernels.push_back(std::move(kernelObj));
            }
        }

        auto core_type = kernel.programmable_core_type;
        auto core_type_name = enchantum::to_string(core_type);
        auto processor_class_name = enchantum::to_string(kernel.processor_class);

        for (const auto& binary_meta : kernel.binary_meta) {
            auto key = fmt::format(
                "{}_{}_{}_max_kernel_size", core_type_name, processor_class_name, binary_meta.processor_type);
            if (kernelSizes.value(key, 0) < binary_meta.packed_size) {
                kernelSizes[key] = binary_meta.packed_size;
            }
        }
    }

    json ret;
    ret["compute_kernels"] = std::move(computeKernels);
    ret["datamovement_kernels"] = std::move(datamovementKernels);
    ret["kernel_sizes"] = std::move(kernelSizes);
    return ret;
}

#endif  // TRACY_ENABLE

// ---------------------------------------------------------------------------
// assemble_device_op_json — non-template, compiled once
// Builds the full JSON string, updates global caches, returns Tracy message.
// ---------------------------------------------------------------------------

std::string assemble_device_op_json(
    [[maybe_unused]] const OpProfileData& data,
    [[maybe_unused]] ttsl::hash::hash_t program_hash,
    [[maybe_unused]] ChipId device_id,
    [[maybe_unused]] bool program_cache_hit,
    [[maybe_unused]] const tt::tt_metal::Program& program) {
#if defined(TRACY_ENABLE)
    json j;
    j["global_call_count"] = data.operation_id;
    j["op_code"] = data.op_name;

    json attributesObj;
    for (const auto& [name, value] : data.attributes) {
        attributesObj[name] = value;
    }
    j["attributes"] = attributesObj;

    std::vector<json> input_tensors_json;
    input_tensors_json.reserve(data.input_tensors.size());
    for (const auto& tm : data.input_tensors) {
        input_tensors_json.push_back(tensor_meta_to_json(tm));
    }
    j["input_tensors"] = std::move(input_tensors_json);

    std::vector<json> output_tensors_json;
    output_tensors_json.reserve(data.output_tensors.size());
    for (const auto& tm : data.output_tensors) {
        output_tensors_json.push_back(tensor_meta_to_json(tm));
    }
    j["output_tensors"] = std::move(output_tensors_json);

    j["op_type"] = enchantum::to_string(OpType::tt_dnn_device);
    j["device_id"] = device_id;
    j["op_hash"] = program_hash;
    j["program_cache_hit"] = program_cache_hit;
    j["kernel_info"] = get_kernels_json(device_id, program);

    auto opname = j["op_code"].template get<std::string>();
    runtime_id_to_opname_.insert({device_id, program.get_runtime_id()}, opname);
    program_hash_to_opname_.insert({device_id, program_hash}, opname);

    j["optional_input_tensors"] = std::vector<json>{};

    j["performance_model"]["compute_ns"] = data.perf_compute_ns;
    j["performance_model"]["ideal_ns"] = data.perf_ideal_ns;
    j["performance_model"]["bandwidth_ns"] = data.perf_bandwidth_ns;
    j["performance_model"]["input_bws"] = data.perf_input_bws;
    j["performance_model"]["output_bws"] = data.perf_output_bws;

    std::string short_str = fmt::format(
        "`TT_DNN_DEVICE_OP: {}, {}, {}, {}, ", j["op_code"].dump(), program_hash, device_id, program_cache_hit);

    if (cached_ops.find(device_id) == cached_ops.end()) {
        cached_ops.emplace(device_id, (std::unordered_map<ttsl::hash::hash_t, std::string>){{program_hash, short_str}});
    } else {
        cached_ops.at(device_id).emplace(program_hash, short_str);
    }

    // Tracy hard limit is uint16_t::max bytes including null terminator.
    // The message is wrapped in backticks which serve as the CSV quotechar.
    // If the message is truncated by tracy_message(), the closing backtick is
    // lost, causing the CSV reader to absorb subsequent rows into this field.
    // Build with pretty JSON first, then compact, then drop kernel_info to fit.
    constexpr size_t tracy_limit = std::numeric_limits<uint16_t>::max() - 1;
    auto msg = fmt::format("{}{} ->\n{}`", short_str, data.operation_id, j.dump(4));
    if (msg.size() > tracy_limit) {
        msg = fmt::format("{}{} ->\n{}`", short_str, data.operation_id, j.dump(-1));
    }
    if (msg.size() > tracy_limit) {
        j.erase("kernel_info");
        msg = fmt::format("{}{} ->\n{}`", short_str, data.operation_id, j.dump(-1));
    }
    if (msg.size() > tracy_limit) {
        log_warning(
            tt::LogMetal,
            "Tracy op profiler message for op '{}' (call {}, device {}) exceeded the {} byte limit even after "
            "dropping kernel_info ({} bytes). Message discarded.",
            j.value("op_code", "?"),
            data.operation_id,
            device_id,
            tracy_limit,
            msg.size());
        return {};
    }
    return msg;
#else
    return {};
#endif
}

}  // namespace tt::tt_metal::op_profiler
