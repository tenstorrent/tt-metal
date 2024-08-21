// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "graph_trace_utils.hpp"

#include "graph_processor.hpp"

#include "tt_metal/common/assert.hpp"

#include <unordered_set>
#include <string>
#include <cstdlib> // std::strtoul


namespace ttnn::graph {

namespace {
ttnn::Shape parse_shape(std::string_view shape_string) {
    // Extract shape values from string like "ttnn.Shape([1, 3, 32, 32])"
    auto start = shape_string.find('[') + 1;
    auto end = shape_string.find(']');
    std::string_view shape_values = shape_string.substr(start, end - start);

    // Vector to hold the parsed shape values
    std::vector<uint32_t> shape;
    const char* str = shape_values.data();
    const char* end_str = str + shape_values.size();

    while (str < end_str) {
        char* next;
        uint32_t value = std::strtoul(str, &next, 10);
        if (str == next) break; // no conversion happened
        shape.push_back(value);
        str = next;
        if (*str == ',') {
            ++str; // skip the comma
        }
        if (*str == ' ') {
            ++str; // skip spaces, assume a single space
        }
    }

    return ttnn::Shape(shape);
}
} // namespace

uint32_t extract_peak_memory_usage(const nlohmann::json& trace) {
    uint32_t total_cb = 0;
    uint32_t total_buffer = 0;
    uint32_t peak_memory_usage = 0;
    std::vector<std::string> current_op;

    for (size_t i = 0; i < trace.size(); ++i) {
        const auto& v = trace[i];

        if (v["name"] == "function_start") {
            if (current_op.empty()) {
                while (++i < trace.size()) {
                    const auto& inner_v = trace[i];
                    if (inner_v["name"] == "buffer" && inner_v["params"]["type"] == "L1") {
                        total_buffer += std::stoi(inner_v["params"]["size"].get<std::string>());
                    } else if (inner_v["name"].get<std::string>().find("tensor") != std::string::npos) {
                        continue;
                    } else {
                        break;
                    }
                }
                --i;  // adjust for loop increment
            }
            current_op.push_back(v["params"]["name"]);
        } else if (v["name"] == "circular_buffer_allocate") {
            total_cb += stoi(v["params"]["size"].get<std::string>());
        } else if (v["name"] == "circular_buffer_deallocate_all") {
            total_cb = 0;
        } else if (v["name"] == "buffer_allocate" && v["params"]["type"] == "L1") {
            total_buffer += stoi(v["params"]["size"].get<std::string>());
        } else if (v["name"] == "buffer_deallocate") {
            auto connection = v["connections"][0].get<int>();
            auto buffer = trace[connection];
            if(buffer["params"]["type"] == "L1") {
                total_buffer -= stoi(buffer["params"]["size"].get<std::string>());
            }
        } else if (v["name"] == "function_end") {
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
        if (v["name"] == "function_start" && !first_begin_found) {
            first_begin_found = true;
            first_begin_index = i;
        } else if (v["name"] == "function_end") {
            last_end_found = true;
            last_end_index = i;

            if("create_device_tensor") {
                auto id = v["connections"][0].get<int>();
                intermediate_tensors.insert(id);
            }
        }
    }

    TT_ASSERT(first_begin_found);
    TT_ASSERT(last_end_found);

    for(int index : trace[last_end_index]["connections"]) {
        // It can be tensor or some other node like
        if(trace[index]["name"].get<std::string>().find("tensor") != std::string::npos) {
            output_tensors.insert(index);
        }
    }

    for(int index : output_tensors) {
        intermediate_tensors.erase(index);
    }

    // Return the counts of intermediate and output tensors
    return {intermediate_tensors.size(), output_tensors.size()};
}

std::vector<std::string> extract_calltrace(const nlohmann::json& trace){
    std::vector<std::string> op_calls;
    size_t i = 0;

    while (i < trace.size()) {
        const auto& v = trace[i];
        i++;

        if (v["name"] == "function_start") {
            op_calls.push_back(v["params"]["name"]);
        }
    }

    return op_calls;
}

std::unordered_set<uint32_t> extract_output_tensors(const nlohmann::json& trace)
{
    // Lambda to find the last 'function_end' node
    auto find_function_end_node = [](const auto& trace) -> const nlohmann::json& {
        for(int i = trace.size() - 1; i >= 0; --i) {
            const auto& v = trace[i];
            if (v["name"] == "function_end") {
                return v;
            }
        }
        TT_FATAL(false, "No function_end node found in the trace");
    };

    const auto& function_end_node = find_function_end_node(trace);

    // Lambda to extract output tensors from the 'function_end' node
    auto extract_output_tensors = [&trace](const nlohmann::json& function_end_node) {
        std::unordered_set<uint32_t> output;
        auto connections = function_end_node["connections"].get<std::vector<uint32_t>>();
        for (const auto& output_id : connections) {
            const auto& output_node = trace[output_id];
            if (output_node["name"].template get<std::string>().find("tensor") != std::string::npos) {
                output.insert(output_id);
            }
        }
        return output;
    };

    const auto output_tensors = extract_output_tensors(function_end_node);
    return output_tensors;
}

std::vector<TensorInfo> extract_output_info(const nlohmann::json& trace)
{
    std::vector<TensorInfo> output;
    auto output_tensors = extract_output_tensors(trace);

    for (const auto& node : trace) {
        if (node["name"] != "buffer" )
            continue;

        auto connections = node["connections"].get<std::vector<uint32_t>>();
        for (const auto& tensor_id : connections) {
            if (output_tensors.find(tensor_id) == output_tensors.end())
                continue;

            const auto type = node["params"]["type"] == "L1" ? tt::tt_metal::BufferType::L1 : tt::tt_metal::BufferType::DRAM;
            const auto size = stoi(node["params"]["size"].get<std::string>());

            const auto& tensor = trace[tensor_id];
            const std::string shape_string = tensor["params"]["shape"].get<std::string>();
            const auto shape = parse_shape(shape_string);

            output.emplace_back(TensorInfo {.shape = shape, .size = size, .type = type});
        }
    }

    return output;
}


} // namespace ttnn::graph
