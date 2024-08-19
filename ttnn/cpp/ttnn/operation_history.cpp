// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operation_history.hpp"

#include "tt_metal/common/core_coord.h"

namespace tt {

namespace tt_metal {

#ifdef DEBUG

namespace operation_history {

namespace detail {

OperationHistory::~OperationHistory() {
    this->dump_to_csv(csv_file_name());
    this->dump_to_json(json_file_name());
}

void OperationHistory::append(OperationRecord&& record) {
    std::scoped_lock<std::mutex> lock(op_history_mutex);
    TT_ASSERT(record.input_tensor_records.size() <= 5);
    this->records.push_back(std::move(record));
}

template <typename RowType>
void write_row(std::ofstream& output_file_stream, const std::size_t num_columns, const RowType& row) {
    TT_ASSERT(row.size() == num_columns);
    for (const auto& element : row) {
        output_file_stream << '"' << element << '"' << ",";
    }
    output_file_stream << std::endl;
}

std::size_t write_header(
    std::ofstream& output_file_stream, const std::size_t num_attributes, const std::size_t num_input_tensors) {
    auto column_names = std::vector<std::string>{
        "ttnn_operation_id", "operation_type", "operation_name", "program_cache_hit", "program_hash"};

    for (auto attribute_index = 0; attribute_index < num_attributes; attribute_index++) {
        column_names.push_back(fmt::format("attribute_{}_name", attribute_index));
        column_names.push_back(fmt::format("attribute_{}_value", attribute_index));
    }

    for (auto input_tensor_index = 0; input_tensor_index < num_input_tensors; input_tensor_index++) {
        column_names.push_back(fmt::format("input_tensor_{}_storage_type", input_tensor_index));
        column_names.push_back(fmt::format("input_tensor_{}_shape", input_tensor_index));
        column_names.push_back(fmt::format("input_tensor_{}_dtype", input_tensor_index));
        column_names.push_back(fmt::format("input_tensor_{}_layout", input_tensor_index));
        column_names.push_back(fmt::format("input_tensor_{}_memory_config", input_tensor_index));
    }

    write_row(output_file_stream, column_names.size(), column_names);
    return column_names.size();
}

void write_record(
    std::ofstream& output_file_stream,
    const std::size_t num_columns,
    const OperationRecord& record,
    const std::size_t num_attributes,
    const std::size_t num_input_tensors) {
    std::vector<std::string> row;
    row.reserve(num_columns);

    row.push_back(fmt::format("{}", record.ttnn_operation_id));
    row.push_back(record.operation_type);
    row.push_back(record.operation_name);
    row.push_back(fmt::format("{}", record.program_cache_hit));
    row.push_back(fmt::format("{}", record.program_hash));
    for (auto attribute_index = 0; attribute_index < num_attributes; attribute_index++) {
        if (attribute_index < record.attributes.size()) {
            const auto& [name, value] = record.attributes.at(attribute_index);
            row.push_back(fmt::format("{}", name));
            row.push_back(fmt::format("{}", value));
        } else {
            row.push_back("");
            row.push_back("");
        }
    }
    for (auto input_tensor_index = 0; input_tensor_index < num_input_tensors; input_tensor_index++) {
        if (input_tensor_index < record.input_tensor_records.size()) {
            const auto& tensor_record = record.input_tensor_records.at(input_tensor_index);
            row.push_back(fmt::format("{}", tensor_record.storage_type));
            row.push_back(fmt::format("{}", tensor_record.shape));
            row.push_back(fmt::format("{}", tensor_record.data_type));
            row.push_back(fmt::format("{}", tensor_record.layout));
            row.push_back(fmt::format("{}", tensor_record.memory_config));
        } else {
            row.push_back("");
            row.push_back("");
            row.push_back("");
            row.push_back("");
            row.push_back("");
        }
    }
    write_row(output_file_stream, num_columns, row);
}

void OperationHistory::dump_to_csv(const char* file_name) {
    if (not enabled())
        return;

    std::ofstream output_file_stream(file_name);

    std::size_t num_attributes = 0;
    for (const auto& record : this->records) {
        num_attributes = std::max(num_attributes, record.attributes.size());
    }

    std::size_t num_input_tensors = 0;
    for (const auto& record : this->records) {
        num_input_tensors = std::max(num_input_tensors, record.input_tensor_records.size());
    }

    auto num_columns = write_header(output_file_stream, num_attributes, num_input_tensors);
    for (const auto& record : this->records) {
        write_record(output_file_stream, num_columns, record, num_attributes, num_input_tensors);
    }
}

void OperationHistory::dump_to_json(const char* file_name) {
    if (not enabled())
        return;

    nlohmann::json json;
    for (const auto& record : this->records) {
        nlohmann::json record_json;
        record_json["ttnn_operation_id"] = tt::stl::json::to_json(record.ttnn_operation_id);
        record_json["operation_type"] = tt::stl::json::to_json(record.operation_type);
        record_json["operation_name"] = tt::stl::json::to_json(record.operation_name);
        record_json["program_cache_hit"] = tt::stl::json::to_json(record.program_cache_hit);
        record_json["program_hash"] = tt::stl::json::to_json(record.program_hash);

        nlohmann::json attributes_json;
        for (const auto& [name, value] : record.attributes) {
            attributes_json[fmt::format("{}", name)] = tt::stl::json::to_json(value);
        }
        // record_json["attributes"] = attributes_json;
        nlohmann::json input_tensor_records_json;
        for (const auto& tensor_record : record.input_tensor_records) {
            nlohmann::json tensor_record_json;
            tensor_record_json["storage_type"] = tt::stl::json::to_json(tensor_record.storage_type);
            tensor_record_json["shape"] = tt::stl::json::to_json(tensor_record.shape);
            tensor_record_json["dtype"] = tt::stl::json::to_json(tensor_record.data_type);
            tensor_record_json["layout"] = tt::stl::json::to_json(tensor_record.layout);
            tensor_record_json["memory_config"] = tt::stl::json::to_json(tensor_record.memory_config);
            input_tensor_records_json.push_back(tensor_record_json);
        }
        record_json["input_tensor_records"] = input_tensor_records_json;
        json.push_back(record_json);
    }
    std::ofstream output_file_stream(file_name);
    output_file_stream << json << std::endl;
}

void OperationHistory::clear() {
    std::scoped_lock<std::mutex> lock(op_history_mutex);
    this->records.clear();
}

OperationHistory OPERATION_HISTORY{};

}  // namespace detail

const char* csv_file_name() { return std::getenv("OPERATION_HISTORY_CSV"); }
const char* json_file_name() { return std::getenv("OPERATION_HISTORY_JSON"); }

bool enabled() { return csv_file_name() != nullptr or json_file_name() != nullptr; }

void dump_to_csv() { detail::OPERATION_HISTORY.dump_to_csv(csv_file_name()); }
void dump_to_json() { detail::OPERATION_HISTORY.dump_to_json(json_file_name()); }

void clear() { detail::OPERATION_HISTORY.clear(); }

}  // namespace operation_history

#endif

}  // namespace tt_metal

}  // namespace tt
