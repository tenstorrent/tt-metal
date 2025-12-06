// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/tools/inspector/inspector.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <array>
#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>

namespace ttnn::inspector {

namespace {  // anonymous namespace

constexpr char kDefaultDir[] = "generated/ttnn/inspector";
constexpr char kTmpInspectorDir[] = "/tmp/ttnn/inspector";  // Default directory for inspector
constexpr char kRolloverSuffix[] = ".prev";                 // Suffix for previous file
constexpr std::chrono::seconds kFlushInterval{1};           // Flush interval for background thread
constexpr size_t kRuntimeIdLineBufferSize = 256;            // Buffer size for runtime id line
constexpr size_t kRuntimeIdRolloverEntryLimit = 8192;       // Write every 8k runtime ids to a new file

uint64_t get_timestamp_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

std::string get_rollover_filename(std::string_view base_path) { return std::string(base_path) + kRolloverSuffix; }

std::filesystem::path get_default_log_directory() {
    // Check environment variable first
    const char* env_path = std::getenv("TT_METAL_INSPECTOR_LOG_PATH");
    if (env_path) {
        return std::filesystem::path(env_path);
    }

    // Default to {TT_METAL_HOME}/generated/ttnn/inspector
    const char* root_dir = std::getenv("TT_METAL_HOME");
    if (root_dir) {
        return std::filesystem::path(root_dir) / kDefaultDir;
    }

    // Fallback to /tmp if TT_METAL_RUNTIME_ROOT not set
    return std::filesystem::path(kTmpInspectorDir);
}

// Internal file writer class with optional rollover
class FileWriter {
public:
    ~FileWriter() { close(); }

    // Open file with optional rollover
    // max_entries = 0 means no rollover (but starts fresh each run)
    // max_entries > 0 means rollover after that many entries
    void open(const std::filesystem::path& file_path, std::string header, size_t max_entries = 0) {
        std::lock_guard<std::mutex> lock(mutex_);

        file_path_ = file_path;
        header_ = std::move(header);
        max_entries_ = max_entries;
        total_entry_count_ = 0;

        // Always open in truncate mode (start fresh)
        file_.open(file_path, std::ios::out | std::ios::trunc);

        // Write header
        if (file_.is_open()) {
            file_ << header_;
            file_.flush();
        }
    }

    void write(std::string_view line) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!file_.is_open()) {
            return;
        }

        // Check if rollover is needed (only if max_entries > 0)
        if (max_entries_ > 0 && total_entry_count_ >= max_entries_) {
            rollover();
        }

        // Write directly to file (it has its own buffering)
        file_.write(line.data(), line.size());
        total_entry_count_++;
    }

    void flush() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (file_.is_open()) {
            file_.flush();
        }
    }

    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (file_.is_open()) {
            file_.flush();
            file_.close();
        }
    }

private:
    void rollover() {
        // Close and rename current file
        file_.close();
        std::string rollover_filename = get_rollover_filename(file_path_.string());
        std::error_code ec;
        std::filesystem::remove(rollover_filename, ec);
        std::filesystem::rename(file_path_, rollover_filename);

        // Open new file and write header
        file_.open(file_path_, std::ios::out | std::ios::trunc);
        if (file_.is_open()) {
            file_ << header_;
            file_.flush();
        }

        // Reset counters
        total_entry_count_ = 0;
    }

    std::ofstream file_;
    std::mutex mutex_;
    std::filesystem::path file_path_;
    std::string header_;
    size_t max_entries_ = 0;
    size_t total_entry_count_ = 0;
};

// Singleton logger with lazy initialization and auto-cleanup
class InspectorLogger {
public:
    static InspectorLogger& instance() {
        static InspectorLogger logger;  // Constructed on first use, destroyed at program exit
        return logger;
    }

    void log_operation_info(
        uint32_t workflow_id, std::string_view operation_name, std::string_view operation_attributes) {
        if (!ensure_initialized()) {
            return;
        }

        uint64_t timestamp = get_timestamp_ns();

        auto line = fmt::format(
            "{{\"timestamp_ns\":{},\"workflow_id\":{},\"operation_name\":\"{}\",\"operation_attributes\":\"{}\"}}\n",
            timestamp,
            workflow_id,
            operation_name,
            operation_attributes);

        operation_info_writer_.write(line);
    }

    void log_runtime_id(uint32_t workflow_id, uint32_t runtime_id) {
        if (!ensure_initialized()) {
            return;
        }

        std::array<char, kRuntimeIdLineBufferSize> line_buffer{};
        uint64_t timestamp = get_timestamp_ns();

        auto result = fmt::format_to_n(
            line_buffer.data(),
            line_buffer.size() - 1,
            "{{\"timestamp_ns\":{},\"workflow_id\":{},\"runtime_id\":{}}}\n",
            timestamp,
            workflow_id,
            runtime_id);

        size_t len = static_cast<size_t>(result.out - line_buffer.data());
        if (len > 0) {
            runtime_id_writer_.write(std::string_view(line_buffer.data(), len));
        }
    }

    void flush() {
        if (!ensure_initialized()) {
            return;
        }

        operation_info_writer_.flush();
        runtime_id_writer_.flush();
    }

private:
    InspectorLogger() = default;

    // Auto-cleanup on destruction (program exit)
    ~InspectorLogger() {
        stop_background_flush();
        if (initialized_.load()) {
            operation_info_writer_.close();
            runtime_id_writer_.close();
        }
    }

    // Lazy initialization with thread-safe once-flag
    // Returns true if initialized successfully, false otherwise
    bool ensure_initialized() {
        if (initialized_.load(std::memory_order_acquire)) {
            return true;  // Fast path: already initialized
        }

        std::call_once(init_flag_, [this]() {
            // Determine log directory
            auto log_dir = get_default_log_directory();
            // Recreate the logging directory if it doesn't exist or clear it if it does (like Inspector)
            try {
                std::filesystem::remove_all(log_dir);  // Clear old logs
                std::filesystem::create_directories(log_dir);
            } catch (const std::exception& e) {
                log_warning(
                    tt::LogOp, "Failed to create inspector log directory: {}. Error: {}", log_dir.string(), e.what());
                return;
            }

            // Open files in the directory
            auto operation_info_path = log_dir / "operation_info.jsonl";
            auto runtime_id_path = log_dir / "runtime_id.jsonl";

            try {
                // Open operation_info as JSONL (no header needed, no rollover)
                operation_info_writer_.open(
                    operation_info_path,
                    "",  // No header for JSONL
                    0    // No rollover
                );

                // Open runtime_id with rollover (keep CSV for simple numeric data)
                runtime_id_writer_.open(
                    runtime_id_path,
                    "",
                    kRuntimeIdRolloverEntryLimit  // Rollover after 8k entries
                );
            } catch (const std::exception& e) {
                log_warning(
                    tt::LogOp, "Failed to open inspector log files in: {}. Error: {}", log_dir.string(), e.what());
                return;
            }
            start_background_flush_thread();
            initialized_.store(true, std::memory_order_release);
        });

        return initialized_.load(std::memory_order_acquire);
    }

    // Start a background thread that flushes the files every second
    void start_background_flush_thread() {
        stop_flag_.store(false);
        flush_thread_ = std::thread([this]() {
            std::unique_lock<std::mutex> lk(flush_mutex_);
            while (!stop_flag_.load()) {
                flush_cv_.wait_for(lk, kFlushInterval);
                if (stop_flag_.load()) {
                    break;
                }
                operation_info_writer_.flush();
                runtime_id_writer_.flush();
            }
        });
    }

    // Stop the background thread that flushes the files
    void stop_background_flush() {
        stop_flag_.store(true);
        flush_cv_.notify_all();
        if (flush_thread_.joinable()) {
            flush_thread_.join();
        }
    }

    InspectorLogger(const InspectorLogger&) = delete;
    InspectorLogger& operator=(const InspectorLogger&) = delete;

    FileWriter operation_info_writer_;
    FileWriter runtime_id_writer_;
    std::atomic<bool> initialized_{false};
    std::once_flag init_flag_;

    std::thread flush_thread_;
    std::mutex flush_mutex_;
    std::condition_variable flush_cv_;
    std::atomic<bool> stop_flag_{false};
};

}  // anonymous namespace

// Public API implementation
void log_operation_info(uint32_t workflow_id, std::string_view operation_name, std::string_view operation_attributes) {
    InspectorLogger::instance().log_operation_info(workflow_id, operation_name, operation_attributes);
}

void log_runtime_id(uint32_t workflow_id, uint32_t runtime_id) {
    InspectorLogger::instance().log_runtime_id(workflow_id, runtime_id);
}

}  // namespace ttnn::inspector
