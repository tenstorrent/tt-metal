// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "graph_trace_utils.hpp"

#include <cstdlib>  // std::strtoul
#include <string>
#include <cstdlib>  // std::strtoul

#include "graph_consts.hpp"
#include "graph_processor.hpp"
#include <tt-metalium/assert.hpp>

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

    return ttnn::Shape(shape);
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

// Returns count of intermediate and output tensors
std::pair<uint32_t, uint32_t> count_intermediate_and_output_tensors(const nlohmann::json& trace) {
    bool first_begin_found = false;
    bool last_end_found = false;

    std::unordered_set<int> intermediate_tensors;
    std::unordered_set<int> output_tensors;

    int first_begin_index = -1;
    int last_end_index = -1;

    for (int i = 0; i < trace.size(); ++i) {
        const auto& v = trace[i];
        if (v[kNodeType] == kNodeFunctionStart && !first_begin_found) {
            first_begin_found = true;
            first_begin_index = i;
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
            if (output_tensors.find(tensor_id) == output_tensors.end()) {
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

// This function returns the worst-case memory allocation per core for the output L1 buffer. 0 for DRAM buffers.
uint32_t extract_l1_output_buffer_allocation_size_per_core(
    const nlohmann::json& trace, size_t interleaved_storage_cores) {
    // we are lookin for buffer_allocate that is connected to a buffer,
    // that is connected to a same tensor as the function_end connected to capture_end
    // buffer_allocate -> buffer -> tensor <- function_end -> capture_end

    // Find the 'capture_end' node
    const auto& capture_end_node =
        std::find_if(trace.rbegin(), trace.rend(), [](const auto& v) { return v.at(kNodeType) == kNodeCaptureEnd; });

    if (capture_end_node == trace.rend()) {
        throw std::runtime_error("No capture_end node found in the trace");
    }

    // helper function to find a node of a specific type that points to a given node
    auto find_a_first_node_of_a_type_pointing_to_me([&](const auto& trace, const char* node_type, const auto& my_node) {
        return std::find_if(trace.begin(), trace.end(), [&](const auto& v) {
            // check if v.at(kNodeType) starts with node_type because buffers and tensors have suffixes \"(counter)\"
            std::string v_node_type = v.at(kNodeType).dump();
            v_node_type.erase(std::remove(v_node_type.begin(), v_node_type.end(), '"'), v_node_type.end());

            return (v_node_type.starts_with(node_type)) &&
                   (std::find(v.at(kConnections).begin(), v.at(kConnections).end(), my_node.at(kCounter)) !=
                    v.at(kConnections).end());
        });
    });

    // helper function to find a node by counter
    auto find_node_by_counter([&](const auto& trace, int counter) {
        return std::find_if(trace.begin(), trace.end(), [&](const auto& v) { return v.at(kCounter) == counter; });
    });

    const auto& last_function_end_node =
        find_a_first_node_of_a_type_pointing_to_me(trace, kNodeFunctionEnd, *capture_end_node);

    if (last_function_end_node == trace.end()) {
        throw std::runtime_error("No function_end node connected to capture_end found in the trace");
    }

    const auto& output_tensor_node =
        find_node_by_counter(trace, last_function_end_node->at(kConnections).at(0).get<int>());
    if (output_tensor_node == trace.end()) {
        throw std::runtime_error("No tensor node connected to function_end found in the trace");
    }

    const auto& output_buffer_node =
        find_a_first_node_of_a_type_pointing_to_me(trace, kNodeBuffer, *output_tensor_node);
    if (output_buffer_node == trace.end()) {
        throw std::runtime_error("No buffer node connected to tensor found in the trace");
    }

    const auto& output_buffer_allocate_node =
        find_a_first_node_of_a_type_pointing_to_me(trace, kNodeBufferAllocate, *output_buffer_node);
    if (output_buffer_allocate_node == trace.end()) {
        throw std::runtime_error("No buffer_allocate node connected to buffer found in the trace");
    }

    uint32_t output_buffer_allocate_total_size =
        std::stoi(output_buffer_allocate_node->at(kParams).at(kSize).get<std::string>());

    // skip dram buffer allocation checks
    if (output_buffer_allocate_node->at(kParams).at(kType) == "DRAM") {
        return 0;
    }

    uint32_t page_size = std::stoi(output_buffer_allocate_node->at(kParams).at(kPageSize).get<std::string>());
    uint32_t num_of_cores = std::stoi(output_buffer_allocate_node->at(kParams).at(kNumCores).get<std::string>());
    if (num_of_cores == 0) {
        num_of_cores = interleaved_storage_cores;
    }

    return detail::worst_case_per_core_allocation(output_buffer_allocate_total_size, page_size, num_of_cores);
}

// This function returns the worst-case memory allocation per core for the peak L1 usage. Ignores DRAM buffers.
uint32_t extract_l1_buffer_allocation_peak_size_per_core(
    const nlohmann::json& trace, size_t interleaved_storage_cores) {
    uint32_t current_size_per_core = 0;
    uint32_t peak_size_per_core = 0;

    for (const auto& node : trace) {
        // process only buffer allocation and deallocation nodes
        if (node.at(kNodeType) != kNodeBufferAllocate && node.at(kNodeType) != kNodeBufferDeallocate) {
            continue;
        }

        // skip dram buffer allocation/deallocation
        if (node.at(kParams).at(kType) == "DRAM") {
            continue;
        }

        uint32_t page_size = std::stoi(node.at(kParams).at(kPageSize).get<std::string>());
        uint32_t num_of_cores = std::stoi(node.at(kParams).at(kNumCores).get<std::string>());
        if (num_of_cores == 0) {
            num_of_cores = interleaved_storage_cores;
        }

        if (node.at(kNodeType) == kNodeBufferAllocate) {
            current_size_per_core += detail::worst_case_per_core_allocation(
                std::stoi(node.at(kParams).at(kSize).get<std::string>()), page_size, num_of_cores);
            peak_size_per_core = std::max(peak_size_per_core, current_size_per_core);
        } else  // kNodeBufferDeallocate
        {
            current_size_per_core -= detail::worst_case_per_core_allocation(
                std::stoi(node.at(kParams).at(kSize).get<std::string>()), page_size, num_of_cores);
        }
    }

    return peak_size_per_core;
}

// returns peak size of circular buffer allocations for a given trace
uint32_t extract_circular_buffers_peak_size_per_core(const nlohmann::json& trace) {
    uint32_t current_size_per_core = 0;
    uint32_t peak_size_per_core = 0;

    size_t counter_expected = 0;
    for (const auto& node : trace) {
        // expect a trace to be sorted by counter (execution order)
        if (node.at(kCounter).get<size_t>() == counter_expected) {
            counter_expected++;
        } else {
            TT_THROW("Graph trace counter/execution out of order");
        }

        // process only circular buffer allocation and deallocation nodes
        if (node.at(kNodeType) != kNodeCBAllocate && node.at(kNodeType) != kNodeCBDeallocateAll) {
            continue;
        }

        if (node.at(kNodeType) == kNodeCBAllocate) {
            bool is_globally_allocated = std::stoi(node.at(kParams).at(kGloballyAllocated).get<std::string>()) == 1;
            if (!is_globally_allocated) {
                current_size_per_core += std::stoi(node.at(kParams).at(kSize).get<std::string>());
                peak_size_per_core = std::max(peak_size_per_core, current_size_per_core);
            }
        } else {  // kNodeCBDeallocateAll
            current_size_per_core = 0;
        }
    }

    return peak_size_per_core;
}

}  // namespace ttnn::graph
