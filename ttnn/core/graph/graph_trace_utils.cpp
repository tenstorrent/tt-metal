// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/graph/graph_trace_utils.hpp"

#include <cstdint>
#include <cstdlib>  // std::strtoul
#include <string>

#include <nlohmann/json.hpp>

#include "ttnn/graph/graph_consts.hpp"
#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/levelized_graph.hpp"
#include <tt_stl/assert.hpp>

namespace ttnn::graph {

namespace {
ttnn::Shape parse_shape(std::string_view shape_string) {
    // Extract shape values from string like "ttnn.Shape([1, 3, 32, 32])"
    auto start = shape_string.find('[') + 1;
    auto end = shape_string.find(']');
    std::string_view shape_values = shape_string.substr(start, end - start);

    // Vector to hold the parsed shape values
    SmallVector<uint32_t> shape;
    const char* str = shape_values.data();
    const char* end_str = str + shape_values.size();

    while (str < end_str) {
        char* next;
        uint32_t value = std::strtoul(str, &next, 10);
        if (str == next) {
            break;  // no conversion happened
        }
        shape.push_back(value);
        str = next;
        if (*str == ',') {
            ++str;  // skip the comma
        }
        if (*str == ' ') {
            ++str;  // skip spaces, assume a single space
        }
    }

    return ttnn::Shape(std::move(shape));
}
}  // namespace

uint32_t extract_peak_L1_memory_usage(const nlohmann::json& trace) {
    uint32_t total_cb = 0;
    uint32_t total_buffer = 0;
    uint32_t peak_memory_usage = 0;
    std::vector<std::string> current_op;

    for (size_t i = 0; i < trace.size(); ++i) {
        const auto& v = trace[i];

        if (v[kNodeType] == kNodeFunctionStart) {
            if (current_op.empty()) {
                while (++i < trace.size()) {
                    const auto& inner_v = trace[i];
                    if (inner_v[kNodeType] == "buffer" && inner_v[kParams][kType] == "L1") {
                        total_buffer += std::stoi(inner_v[kParams][kSize].get<std::string>());
                    } else if (inner_v[kNodeType] == kNodeTensor) {
                        continue;
                    } else {
                        break;
                    }
                }
                --i;  // adjust for loop increment
            }
            current_op.push_back(v[kParams][kName]);
        } else if (v[kNodeType] == kNodeCBAllocate) {
            total_cb += stoi(v[kParams][kSize].get<std::string>());
        } else if (v[kNodeType] == kNodeCBDeallocateAll) {
            total_cb = 0;
        } else if (v[kNodeType] == kNodeBufferAllocate && v[kParams][kType] == "L1") {
            total_buffer += stoi(v[kParams][kSize].get<std::string>());
        } else if (v[kNodeType] == kNodeBufferDeallocate) {
            auto connection = v[kConnections][0].get<int>();
            auto buffer = trace[connection];
            if (buffer[kParams][kType] == "L1") {
                total_buffer -= stoi(buffer[kParams][kSize].get<std::string>());
            }
        } else if (v[kNodeType] == kNodeFunctionEnd) {
            current_op.pop_back();
        }

        peak_memory_usage = std::max(peak_memory_usage, total_cb + total_buffer);
    }

    return peak_memory_usage;
}

std::pair<uint32_t, uint32_t> count_intermediate_and_output_tensors(const nlohmann::json& trace) {
    bool first_begin_found = false;
    bool last_end_found = false;

    std::unordered_set<int> intermediate_tensors;
    std::unordered_set<int> output_tensors;

    int last_end_index = -1;

    for (int i = 0; i < trace.size(); ++i) {
        const auto& v = trace[i];
        if (v[kNodeType] == kNodeFunctionStart && !first_begin_found) {
            first_begin_found = true;
        } else if (v[kNodeType] == kNodeFunctionEnd) {
            last_end_found = true;
            last_end_index = i;

            if (v[kParams][kName] == "create_device_tensor") {
                auto id = v[kConnections][0].get<int>();
                intermediate_tensors.insert(id);
            }
        }
    }

    TT_ASSERT(first_begin_found);
    TT_ASSERT(last_end_found);

    auto connections = trace[last_end_index][kConnections].get<std::unordered_set<uint32_t>>();
    for (auto index : connections) {
        // It can be tensor or some other node like
        if (trace[index][kNodeType] == kNodeTensor) {
            output_tensors.insert(index);
        }
    }

    for (int index : output_tensors) {
        intermediate_tensors.erase(index);
    }

    // Return the counts of intermediate and output tensors
    return {intermediate_tensors.size(), output_tensors.size()};
}

std::vector<std::string> extract_calltrace(const nlohmann::json& trace) {
    std::vector<std::string> op_calls;
    size_t i = 0;

    while (i < trace.size()) {
        const auto& v = trace[i];
        i++;

        if (v[kNodeType] == kNodeFunctionStart) {
            op_calls.push_back(v[kParams][kName]);
        }
    }

    return op_calls;
}

nlohmann::json extract_levelized_graph(const nlohmann::json& trace, size_t max_level) {
    LevelizedGraph levelized_graph(trace, max_level);
    return levelized_graph.to_json();
}

std::vector<OperationInfo> extract_arguments(const nlohmann::json& trace) {
    std::vector<OperationInfo> operations;
    size_t i = 0;
    while (i < trace.size()) {
        const auto& v = trace[i];
        i++;
        OperationInfo info;
        if (!v[kArguments].empty()) {
            info.operation_name = v[kParams][kName];
            info.arguments = v[kArguments];
            operations.push_back(info);
        }
    }

    return operations;
}

std::unordered_set<uint32_t> extract_output_tensors(const nlohmann::json& trace) {
    // Lambda to find the last 'function_end' node
    auto find_function_end_node = [](const auto& trace) -> const nlohmann::json& {
        for (int i = trace.size() - 1; i >= 0; --i) {
            const auto& v = trace[i];
            if (v[kNodeType] == kNodeFunctionEnd) {
                return v;
            }
        }
        TT_THROW("No function_end node found in the trace");
    };

    const auto& function_end_node = find_function_end_node(trace);

    // Lambda to extract output tensors from the 'function_end' node
    auto extract_output_tensors = [&trace](const nlohmann::json& function_end_node) {
        std::unordered_set<uint32_t> output;
        auto connections = function_end_node[kConnections].get<std::unordered_set<uint32_t>>();
        for (const auto& output_id : connections) {
            const auto& output_node = trace[output_id];
            if (output_node[kNodeType] == kNodeTensor) {
                output.insert(output_id);
            }
        }
        return output;
    };

    const auto output_tensors = extract_output_tensors(function_end_node);
    return output_tensors;
}

std::vector<TensorInfo> extract_output_info(const nlohmann::json& trace) {
    std::vector<TensorInfo> output;
    auto output_tensors = extract_output_tensors(trace);

    for (const auto& node : trace) {
        if (node[kNodeType] != kNodeBuffer) {
            continue;
        }

        auto connections = node[kConnections].get<std::unordered_set<uint32_t>>();
        for (const auto& tensor_id : connections) {
            if (!output_tensors.contains(tensor_id)) {
                continue;
            }

            const auto type =
                node[kParams][kType] == "L1" ? tt::tt_metal::BufferType::L1 : tt::tt_metal::BufferType::DRAM;
            const auto size = stoi(node[kParams][kSize].get<std::string>());

            const auto& tensor = trace[tensor_id];
            const std::string shape_string = tensor[kParams][kShape];
            const auto shape = parse_shape(shape_string);

            output.emplace_back(TensorInfo{.shape = shape, .size = size, .type = type});
        }
    }

    return output;
}

namespace detail {
// This function computes the worst-case memory allocation per core for a given total size, page size, and number of
// cores.
size_t worst_case_per_core_allocation(size_t total_size, size_t page_size, size_t num_of_cores) {
    size_t pages = std::ceil(float(total_size) / page_size);
    size_t pages_per_core = std::ceil(float(pages) / num_of_cores);
    return pages_per_core * page_size;
}
}  // namespace detail

uint32_t extract_l1_output_buffer_allocation_size_per_core(
    const Tensor& output_tensor, size_t interleaved_storage_cores) {
    tt::tt_metal::Buffer* buffer = output_tensor.buffer();
    if (buffer->is_dram()) {
        TT_THROW("No L1 allocation. Tensor is in DRAM");
    }

    uint32_t output_buffer_allocate_total_size = buffer->size();
    uint32_t page_size = buffer->page_size();
    uint32_t num_cores = buffer->num_cores().value_or(interleaved_storage_cores);

    return detail::worst_case_per_core_allocation(output_buffer_allocate_total_size, page_size, num_cores);
}

uint32_t extract_l1_buffer_allocation_peak_size_per_core(
    const nlohmann::json& trace, size_t interleaved_storage_cores) {
    const auto& [cb_peak_size_per_core, l1_buffers_peak_per_core, peak_memory_usage_per_core] =
        extract_resource_usage_per_core(trace, interleaved_storage_cores);
    return l1_buffers_peak_per_core;
}

uint32_t extract_circular_buffers_peak_size_per_core(const nlohmann::json& trace) {
    const auto& [cb_peak_size_per_core, l1_buffers_peak_per_core, peak_memory_usage_per_core] =
        extract_resource_usage_per_core(trace, 1);
    return cb_peak_size_per_core;
}

// calculate the size of buffer allocated/deallocated on each core
static uint32_t calculate_buffer_allocation_size(const nlohmann::json& node, size_t interleaved_storage_cores) {
    uint32_t page_size = std::stoi(node.at(kParams).at(kPageSize).get<std::string>());
    uint32_t num_of_cores = std::stoi(node.at(kParams).at(kNumCores).get<std::string>());
    if (num_of_cores == 0) {
        num_of_cores = interleaved_storage_cores;
    }

    uint32_t total_size = std::stoi(node.at(kParams).at(kSize).get<std::string>());
    return detail::worst_case_per_core_allocation(total_size, page_size, num_of_cores);
}

uint32_t extract_peak_memory_usage(const nlohmann::json& trace, size_t interleaved_storage_cores) {
    const auto& [cb_peak_size_per_core, l1_buffers_peak_per_core, peak_memory_usage_per_core] =
        extract_resource_usage_per_core(trace, interleaved_storage_cores);
    return peak_memory_usage_per_core;
}

PeakMemoryUsagePerCore extract_resource_usage_per_core(const nlohmann::json& trace, size_t interleaved_storage_cores) {
    size_t current_cb = 0, peak_cb = 0;
    size_t current_l1 = 0, peak_l1 = 0;
    size_t current_total = 0, peak_total = 0;

    size_t counter_expected = 0;
    for (const auto& node : trace) {
        // expect a trace to be sorted by counter (execution order)
        if (node.at(kCounter).get<size_t>() == counter_expected) {
            counter_expected++;
        } else {
            TT_THROW("Graph trace counter/execution out of order");
        }

        if (node.at(kNodeType) == kNodeCBAllocate) {
            bool is_globally_allocated = std::stoi(node.at(kParams).at(kGloballyAllocated).get<std::string>()) == 1;
            if (!is_globally_allocated) {
                uint32_t alloc_size = std::stoi(node.at(kParams).at(kSize).get<std::string>());
                current_cb += alloc_size;
                peak_cb = std::max(peak_cb, current_cb);
                current_total += alloc_size;
                peak_total = std::max(peak_total, current_total);
            }
        } else if (node.at(kNodeType) == kNodeCBDeallocateAll) {
            current_total -= current_cb;
            current_cb = 0;
        } else if (node.at(kNodeType) == kNodeBufferAllocate || node.at(kNodeType) == kNodeBufferDeallocate) {
            if (node.at(kParams).at(kType) == "DRAM") {
                continue;
            }
            size_t alloc_size = calculate_buffer_allocation_size(node, interleaved_storage_cores);
            if (node.at(kNodeType) == kNodeBufferAllocate) {
                current_l1 += alloc_size;
                peak_l1 = std::max(peak_l1, current_l1);
                current_total += alloc_size;
                peak_total = std::max(peak_total, current_total);
            } else {  // kNodeBufferDeallocate
                current_l1 -= alloc_size;
                current_total -= alloc_size;
            }
        }
    }
    return PeakMemoryUsagePerCore{.peak_cb = peak_cb, .peak_l1 = peak_l1, .peak_total = peak_total};
}

DRAMUsage extract_dram_usage(const nlohmann::json& trace) {
    DRAMUsage result;
    long long current_buffer = 0;

    for (size_t i = 0; i < trace.size(); ++i) {
        const auto& v = trace[i];

        if (v[kNodeType] == kNodeBufferAllocate && v[kParams][kType] == "DRAM") {
            size_t buffer_size = std::stoll(v[kParams][kSize].get<std::string>());
            current_buffer += buffer_size;
            result.total_allocations += buffer_size;
        } else if (v[kNodeType] == kNodeBufferDeallocate) {
            auto connection = v[kConnections][0].get<int>();
            auto buffer = trace[connection];
            if (buffer[kParams][kType] == "DRAM") {
                size_t buffer_size = std::stoll(buffer[kParams][kSize].get<std::string>());
                current_buffer -= buffer_size;
                result.total_deallocations += buffer_size;
            }
        }

        // Track peak
        result.peak = std::max(result.peak, current_buffer);
    }

    return result;
}

}  // namespace ttnn::graph
