// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "ttnn/reports.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace ttnn::database {

// Initialize or get the SQLite database connection for the given report path.
// Creates the database and tables if they don't exist.
void init_database(const std::filesystem::path& report_path);

// Close the database connection
void close_database();

// Insert an operation into the database.
// If duration is nullopt, inserts with NULL duration (pre-operation insert).
// If duration has a value, updates the existing operation with the duration (post-operation update).
void insert_operation(
    const std::filesystem::path& report_path,
    uint64_t operation_id,
    const std::string& operation_name,
    std::optional<double> duration_ms);

// Insert device information into the database.
void insert_device(
    const std::filesystem::path& report_path, uint32_t device_id, const ttnn::reports::DeviceInfo& device_info);

// Insert devices from mesh devices
void insert_devices(
    const std::filesystem::path& report_path, const std::vector<tt::tt_metal::distributed::MeshDevice*>& devices);

// Insert buffer information for an operation.
void insert_buffers(
    const std::filesystem::path& report_path,
    uint64_t operation_id,
    const std::vector<tt::tt_metal::distributed::MeshDevice*>& devices);

// Insert detailed buffer page information for an operation.
void insert_buffer_pages(
    const std::filesystem::path& report_path,
    uint64_t operation_id,
    const std::vector<tt::tt_metal::distributed::MeshDevice*>& devices);

// Insert captured graph JSON for an operation.
void insert_captured_graph(
    const std::filesystem::path& report_path, uint64_t operation_id, const nlohmann::json& captured_graph);

// Insert input tensors for an operation.
void insert_input_tensors(
    const std::filesystem::path& report_path, uint64_t operation_id, const std::vector<Tensor>& tensors);

// Insert output tensors for an operation.
void insert_output_tensors(
    const std::filesystem::path& report_path, uint64_t operation_id, const std::vector<Tensor>& tensors);

// Insert stack trace for an operation.
void insert_stack_trace(const std::filesystem::path& report_path, uint64_t operation_id);

// Get and increment the global operation ID counter
uint64_t get_next_operation_id();

// Save config.json to the report path
void save_config_json(const std::filesystem::path& report_path);

// Save cluster_descriptor.yaml to the report path
void save_cluster_descriptor(const std::filesystem::path& report_path);

}  // namespace ttnn::database
