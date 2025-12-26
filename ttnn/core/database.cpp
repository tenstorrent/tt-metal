// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "database.hpp"

#include <atomic>
#include <fstream>
#include <mutex>
#include <set>

#include <sqlite3.h>

#include <tt-metalium/mesh_device.hpp>
#include <tt-logger/tt-logger.hpp>

#include "ttnn/cluster.hpp"
#include "ttnn/config.hpp"
#include "ttnn/reports.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::database {

namespace {

// Global state
sqlite3* g_db = nullptr;
std::mutex g_db_mutex;
std::set<uint32_t> g_inserted_device_ids;
std::atomic<uint64_t> g_operation_id_counter{0};
std::filesystem::path g_current_report_path;

constexpr const char* SQLITE_DB_PATH = "db.sqlite";
constexpr const char* CONFIG_PATH = "config.json";

void execute_sql(const std::string& sql) {
    char* err_msg = nullptr;
    int rc = sqlite3_exec(g_db, sql.c_str(), nullptr, nullptr, &err_msg);
    if (rc != SQLITE_OK) {
        std::string error = err_msg ? err_msg : "Unknown error";
        sqlite3_free(err_msg);
        log_warning(tt::LogAlways, "SQLite error: {} (SQL: {})", error, sql);
    }
}

void create_tables() {
    // devices table
    execute_sql(R"(
        CREATE TABLE IF NOT EXISTS devices (
            device_id int,
            num_y_cores int,
            num_x_cores int,
            num_y_compute_cores int,
            num_x_compute_cores int,
            worker_l1_size int,
            l1_num_banks int,
            l1_bank_size int,
            address_at_first_l1_bank int,
            address_at_first_l1_cb_buffer int,
            num_banks_per_storage_core int,
            num_compute_cores int,
            num_storage_cores int,
            total_l1_memory int,
            total_l1_for_tensors int,
            total_l1_for_interleaved_buffers int,
            total_l1_for_sharded_buffers int,
            cb_limit int
        )
    )");

    // tensors table
    execute_sql(R"(
        CREATE TABLE IF NOT EXISTS tensors (
            tensor_id int UNIQUE, shape text, dtype text, layout text,
            memory_config text, device_id int, address int, buffer_type int
        )
    )");

    // device_tensors table
    execute_sql(R"(
        CREATE TABLE IF NOT EXISTS device_tensors (
            tensor_id int, device_id int, address int
        )
    )");

    // comparison records tables
    execute_sql(R"(
        CREATE TABLE IF NOT EXISTS local_tensor_comparison_records (
            tensor_id int UNIQUE, golden_tensor_id int, matches bool,
            desired_pcc bool, actual_pcc float
        )
    )");

    execute_sql(R"(
        CREATE TABLE IF NOT EXISTS global_tensor_comparison_records (
            tensor_id int UNIQUE, golden_tensor_id int, matches bool,
            desired_pcc bool, actual_pcc float
        )
    )");

    // operations table
    execute_sql(R"(
        CREATE TABLE IF NOT EXISTS operations (
            operation_id int UNIQUE, name text, duration float
        )
    )");

    // operation_arguments table
    execute_sql(R"(
        CREATE TABLE IF NOT EXISTS operation_arguments (
            operation_id int, name text, value text
        )
    )");

    // stack_traces table
    execute_sql(R"(
        CREATE TABLE IF NOT EXISTS stack_traces (
            operation_id int, stack_trace text
        )
    )");

    // input_tensors table
    execute_sql(R"(
        CREATE TABLE IF NOT EXISTS input_tensors (
            operation_id int, input_index int, tensor_id int
        )
    )");

    // output_tensors table
    execute_sql(R"(
        CREATE TABLE IF NOT EXISTS output_tensors (
            operation_id int, output_index int, tensor_id int
        )
    )");

    // buffers table
    execute_sql(R"(
        CREATE TABLE IF NOT EXISTS buffers (
            operation_id int, device_id int, address int,
            max_size_per_bank int, buffer_type int, buffer_layout int
        )
    )");

    // buffer_pages table
    execute_sql(R"(
        CREATE TABLE IF NOT EXISTS buffer_pages (
            operation_id int, device_id int, address int, core_y int,
            core_x int, bank_id int, page_index int, page_address int,
            page_size int, buffer_type int
        )
    )");

    // nodes table
    execute_sql(R"(
        CREATE TABLE IF NOT EXISTS nodes (
            operation_id int, unique_id int, node_operation_id int, name text
        )
    )");

    // edges table
    execute_sql(R"(
        CREATE TABLE IF NOT EXISTS edges (
            operation_id int, source_unique_id int, sink_unique_id int,
            source_output_index int, sink_input_index int, key int
        )
    )");

    // captured_graph table
    execute_sql(R"(
        CREATE TABLE IF NOT EXISTS captured_graph (
            operation_id int, captured_graph text
        )
    )");

    // errors table
    execute_sql(R"(
        CREATE TABLE IF NOT EXISTS errors (
            operation_id int, operation_name text, error_type text,
            error_message text, stack_trace text, timestamp text
        )
    )");
}

}  // namespace

void init_database(const std::filesystem::path& report_path) {
    std::lock_guard<std::mutex> lock(g_db_mutex);

    // If we already have a database open for this path, return
    if (g_db != nullptr && g_current_report_path == report_path) {
        return;
    }

    // Close existing database if any
    if (g_db != nullptr) {
        sqlite3_close(g_db);
        g_db = nullptr;
    }

    // Create report directory if it doesn't exist
    std::filesystem::create_directories(report_path);

    // Save config.json
    auto config_path = report_path / CONFIG_PATH;
    if (!std::filesystem::exists(config_path)) {
        save_config_json(report_path);
    }

    // Save cluster_descriptor.yaml
    save_cluster_descriptor(report_path);

    // Open database
    auto db_path = report_path / SQLITE_DB_PATH;
    int rc = sqlite3_open(db_path.string().c_str(), &g_db);
    if (rc != SQLITE_OK) {
        log_warning(tt::LogAlways, "Failed to open SQLite database at {}: {}", db_path.string(), sqlite3_errmsg(g_db));
        sqlite3_close(g_db);
        g_db = nullptr;
        return;
    }

    g_current_report_path = report_path;
    g_inserted_device_ids.clear();

    log_debug(
        tt::LogAlways, "Creating reports path at {} and sqlite database at {}", report_path.string(), db_path.string());

    // Create tables
    create_tables();
}

void close_database() {
    std::lock_guard<std::mutex> lock(g_db_mutex);
    if (g_db != nullptr) {
        sqlite3_close(g_db);
        g_db = nullptr;
        g_current_report_path.clear();
        g_inserted_device_ids.clear();
    }
}

void insert_operation(
    const std::filesystem::path& report_path,
    uint64_t operation_id,
    const std::string& operation_name,
    std::optional<double> duration_ms) {
    init_database(report_path);

    std::lock_guard<std::mutex> lock(g_db_mutex);
    if (g_db == nullptr) {
        return;
    }

    std::string sql;
    if (duration_ms.has_value()) {
        // Update with duration (post-operation)
        sql = fmt::format(
            "INSERT INTO operations VALUES ({}, '{}', {}) "
            "ON CONFLICT (operation_id) DO UPDATE SET duration=EXCLUDED.duration",
            operation_id,
            operation_name,
            duration_ms.value());
    } else {
        // Insert without duration (pre-operation)
        sql = fmt::format(
            "INSERT INTO operations VALUES ({}, '{}', NULL) "
            "ON CONFLICT (operation_id) DO NOTHING",
            operation_id,
            operation_name);
    }

    execute_sql(sql);
}

void insert_device(
    const std::filesystem::path& report_path, uint32_t device_id, const ttnn::reports::DeviceInfo& info) {
    init_database(report_path);

    std::lock_guard<std::mutex> lock(g_db_mutex);
    if (g_db == nullptr) {
        return;
    }

    // Check if device already inserted
    if (g_inserted_device_ids.count(device_id) > 0) {
        return;
    }

    std::string sql = fmt::format(
        "INSERT INTO devices VALUES ({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})",
        device_id,
        info.num_y_cores,
        info.num_x_cores,
        info.num_y_compute_cores,
        info.num_x_compute_cores,
        info.worker_l1_size,
        info.l1_num_banks,
        info.l1_bank_size,
        info.address_at_first_l1_bank,
        info.address_at_first_l1_cb_buffer,
        info.num_banks_per_storage_core,
        info.num_compute_cores,
        info.num_storage_cores,
        info.total_l1_memory,
        info.total_l1_for_tensors,
        info.total_l1_for_interleaved_buffers,
        info.total_l1_for_sharded_buffers,
        info.cb_limit);

    execute_sql(sql);
    g_inserted_device_ids.insert(device_id);
}

void insert_devices(
    const std::filesystem::path& report_path, const std::vector<tt::tt_metal::distributed::MeshDevice*>& devices) {
    for (auto* device : devices) {
        auto device_info = ttnn::reports::get_device_info(device);
        insert_device(report_path, device->id(), device_info);
    }
}

void insert_buffers(
    const std::filesystem::path& report_path,
    uint64_t operation_id,
    const std::vector<tt::tt_metal::distributed::MeshDevice*>& devices) {
    init_database(report_path);

    std::lock_guard<std::mutex> lock(g_db_mutex);
    if (g_db == nullptr) {
        return;
    }

    auto buffers = ttnn::reports::get_buffers(devices);
    for (const auto& buffer : buffers) {
        std::string sql = fmt::format(
            "INSERT INTO buffers VALUES ({}, {}, {}, {}, {}, {})",
            operation_id,
            buffer.device_id,
            buffer.address,
            buffer.max_size_per_bank,
            static_cast<int>(buffer.buffer_type),
            static_cast<int>(buffer.buffer_layout));

        execute_sql(sql);
    }
}

void insert_buffer_pages(
    const std::filesystem::path& report_path,
    uint64_t operation_id,
    const std::vector<tt::tt_metal::distributed::MeshDevice*>& devices) {
    init_database(report_path);

    std::lock_guard<std::mutex> lock(g_db_mutex);
    if (g_db == nullptr) {
        return;
    }

    auto buffer_pages = ttnn::reports::get_buffer_pages(devices);
    for (const auto& page : buffer_pages) {
        std::string sql = fmt::format(
            "INSERT INTO buffer_pages VALUES ({}, {}, {}, {}, {}, {}, {}, {}, {}, {})",
            operation_id,
            page.device_id,
            page.address,
            page.core_y,
            page.core_x,
            page.bank_id,
            page.page_index,
            page.page_address,
            page.page_size,
            static_cast<int>(page.buffer_type));

        execute_sql(sql);
    }
}

void insert_captured_graph(
    const std::filesystem::path& report_path, uint64_t operation_id, const nlohmann::json& captured_graph) {
    init_database(report_path);

    std::lock_guard<std::mutex> lock(g_db_mutex);
    if (g_db == nullptr) {
        return;
    }

    // Escape single quotes in JSON string
    std::string json_str = captured_graph.dump();
    std::string escaped_json;
    escaped_json.reserve(json_str.size() * 2);
    for (char c : json_str) {
        if (c == '\'') {
            escaped_json += "''";
        } else {
            escaped_json += c;
        }
    }

    std::string sql = fmt::format("INSERT INTO captured_graph VALUES ({}, '{}')", operation_id, escaped_json);

    execute_sql(sql);
}

uint64_t get_next_operation_id() { return g_operation_id_counter.fetch_add(1); }

void save_config_json(const std::filesystem::path& report_path) {
    auto config_path = report_path / CONFIG_PATH;

    nlohmann::json config_json;

    // Required fields for visualizer
    config_json["cache_path"] = ttnn::CONFIG.get<"cache_path">().string();
    config_json["model_cache_path"] = ttnn::CONFIG.get<"model_cache_path">().string();
    config_json["tmp_dir"] = ttnn::CONFIG.get<"tmp_dir">().string();
    config_json["enable_model_cache"] = ttnn::CONFIG.get<"enable_model_cache">();
    config_json["enable_fast_runtime_mode"] = ttnn::CONFIG.get<"enable_fast_runtime_mode">();
    config_json["throw_exception_on_fallback"] = ttnn::CONFIG.get<"throw_exception_on_fallback">();
    config_json["enable_logging"] = ttnn::CONFIG.get<"enable_logging">();
    config_json["enable_graph_report"] = ttnn::CONFIG.get<"enable_graph_report">();
    config_json["enable_detailed_buffer_report"] = ttnn::CONFIG.get<"enable_detailed_buffer_report">();
    config_json["enable_detailed_tensor_report"] = ttnn::CONFIG.get<"enable_detailed_tensor_report">();
    config_json["enable_comparison_mode"] = ttnn::CONFIG.get<"enable_comparison_mode">();
    config_json["comparison_mode_should_raise_exception"] =
        ttnn::CONFIG.get<"comparison_mode_should_raise_exception">();
    config_json["comparison_mode_pcc"] = ttnn::CONFIG.get<"comparison_mode_pcc">();
    config_json["root_report_path"] = ttnn::CONFIG.get<"root_report_path">().string();
    config_json["report_path"] = report_path.string();

    auto report_name = ttnn::CONFIG.get<"report_name">();
    if (report_name.has_value()) {
        config_json["report_name"] = report_name.value().string();
    }

    std::ofstream file(config_path);
    if (file.is_open()) {
        file << config_json.dump(4);
        file.close();
    } else {
        log_warning(tt::LogAlways, "Failed to write config.json to {}", config_path.string());
    }
}

void save_cluster_descriptor(const std::filesystem::path& report_path) {
    auto cluster_desc_path = report_path / "cluster_descriptor.yaml";

    if (std::filesystem::exists(cluster_desc_path)) {
        return;  // Already exists
    }

    // Get the temporary cluster descriptor path
    std::string temp_path = ttnn::cluster::serialize_cluster_descriptor();
    if (temp_path.empty()) {
        log_warning(tt::LogAlways, "Failed to serialize cluster descriptor");
        return;
    }

    // Copy to destination
    try {
        std::filesystem::copy_file(temp_path, cluster_desc_path);
    } catch (const std::exception& e) {
        log_warning(tt::LogAlways, "Failed to copy cluster descriptor: {}", e.what());
    }
}

namespace {
std::set<uint32_t> g_inserted_tensor_ids;

// Helper to insert a single tensor into the tensors table
void insert_tensor(const Tensor& tensor) {
    if (g_db == nullptr) {
        return;
    }

    uint64_t tid = tensor.tensor_id;

    // Check if tensor already inserted
    if (g_inserted_tensor_ids.count(static_cast<uint32_t>(tid)) > 0) {
        return;
    }

    std::string shape_str = fmt::format("{}", tensor.logical_shape());
    std::string dtype_str = fmt::format("{}", tensor.dtype());
    std::string layout_str = fmt::format("{}", tensor.layout());

    std::string memory_config_str = "NULL";
    int device_id = -1;
    uint64_t address = 0;
    int buffer_type = -1;

    if (tensor.storage_type() == StorageType::DEVICE && tensor.is_allocated()) {
        auto mem_config = tensor.memory_config();
        memory_config_str = fmt::format("'{}'", mem_config);
        device_id = static_cast<int>(tensor.device()->id());
        address = tensor.buffer()->address();
        buffer_type = static_cast<int>(mem_config.buffer_type());
    }

    std::string sql = fmt::format(
        "INSERT INTO tensors VALUES ({}, '{}', '{}', '{}', {}, {}, {}, {})",
        tid,
        shape_str,
        dtype_str,
        layout_str,
        memory_config_str,
        device_id == -1 ? "NULL" : std::to_string(device_id),
        address == 0 ? "NULL" : std::to_string(address),
        buffer_type == -1 ? "NULL" : std::to_string(buffer_type));

    execute_sql(sql);
    g_inserted_tensor_ids.insert(static_cast<uint32_t>(tid));
}
}  // namespace

void insert_input_tensors(
    const std::filesystem::path& report_path, uint64_t operation_id, const std::vector<Tensor>& tensors) {
    init_database(report_path);

    std::lock_guard<std::mutex> lock(g_db_mutex);
    if (g_db == nullptr) {
        return;
    }

    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto& tensor = tensors[i];
        insert_tensor(tensor);

        std::string sql =
            fmt::format("INSERT INTO input_tensors VALUES ({}, {}, {})", operation_id, i, tensor.tensor_id);
        execute_sql(sql);
    }
}

void insert_output_tensors(
    const std::filesystem::path& report_path, uint64_t operation_id, const std::vector<Tensor>& tensors) {
    init_database(report_path);

    std::lock_guard<std::mutex> lock(g_db_mutex);
    if (g_db == nullptr) {
        return;
    }

    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto& tensor = tensors[i];
        insert_tensor(tensor);

        std::string sql =
            fmt::format("INSERT INTO output_tensors VALUES ({}, {}, {})", operation_id, i, tensor.tensor_id);
        execute_sql(sql);
    }
}

void insert_stack_trace(const std::filesystem::path& report_path, uint64_t operation_id) {
    init_database(report_path);

    std::lock_guard<std::mutex> lock(g_db_mutex);
    if (g_db == nullptr) {
        return;
    }

    // For now, insert a placeholder stack trace
    // Getting C++ stack traces at runtime is complex and platform-specific
    std::string stack_trace = "C++ stack trace not available";

    std::string sql = fmt::format("INSERT INTO stack_traces VALUES ({}, '{}')", operation_id, stack_trace);
    execute_sql(sql);
}

}  // namespace ttnn::database
