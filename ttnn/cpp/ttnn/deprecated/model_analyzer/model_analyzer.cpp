// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>

struct CSVTable {
    std::vector<std::vector<std::string>> data;
    std::unordered_map<std::string, size_t> col_index;
};

struct OperationSpec {
    int gl_cnt;
    std::string op_code;
    std::string op_name;
    int batch_size;
    int in0_height;
    int in0_width;
    int in0_channels;

    int filter_height;
    int filter_width;
    int stride;
    int pad;

    int output_height;
    int output_width;
    int output_channels;
};

struct OperationAnalysis {
    long long num_mul_adds;
    int in0_read_bytes;
    int weights_bytes;
    int ideal_dev_clock_cycles;
    int ideal_dev_nano_sec;
    int measured_nano_sec;
    float dev_util_pct;
};

using OpList = std::vector<std::pair<OperationSpec, OperationAnalysis>>;

struct ExecutionConfig {
    int tensix_mul_adds_per_cycle_lofi;
    int num_fidelity_phases;
    int device_num_rows;
    int device_num_cols;
    float frequency_GHZ;
    int dram_bw_GBPS;
    int noc_bw_BPC;
    int num_nocs;
    int fw_launch_latency_nano_sec;

    int weights_bytes_per_datum;
    int tile_act_bytes_per_datum;
    int row_major_act_bytes_per_datum;
    int batch_size;
};

struct ModelSummary {
    long long total_measured_nano_sec;
    long long total_ideal_dev_nano_sec;
    float measured_throughput;
    float ideal_throughput;
    float utilization;
    long long total_in0_read_bytes;
    long long total_params_bytes;
};

CSVTable read_csv(const std::string& file_name) {
    std::ifstream file(file_name);
    CSVTable table;
    std::string line;

    // Read header and create column index map
    if (std::getline(file, line)) {
        std::stringstream line_stream(line);
        std::string cell;
        size_t index = 0;
        while (std::getline(line_stream, cell, ',')) {
            table.col_index[cell] = index++;
        }
    }

    // Read data rows, the header row is skipped automatically
    while (std::getline(file, line)) {
        std::stringstream line_stream(line);
        std::string cell;
        std::vector<std::string> row;
        while (std::getline(line_stream, cell, ',')) {
            row.push_back(cell);
        }
        table.data.push_back(row);
    }

    return table;
}

std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(' ');
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last - first + 1));
}

void trim_spaces(CSVTable& table) {
    for (auto& row : table.data) {
        for (auto& cell : row) {
            cell = trim(cell);
        }
    }

    // Collect updated keys and values in temporary map
    std::unordered_map<std::string, size_t> update_col_index;
    for (const auto& pair : table.col_index) {
        std::string trimmed_key = trim(pair.first);
        update_col_index[trimmed_key] = pair.second;
    }

    // Replace the old col_index map with the updated one
    table.col_index.swap(update_col_index);
}

void print_table_data(const CSVTable& table) {
    for (const auto& row : table.data) {
        for (const auto& cell : row) {
            std::cout << cell << ", ";
        }
        std::cout << '\n';
    }
}

std::string get_table_cell(const CSVTable& table, size_t row_index, const std::string& col_key) {
    if (table.col_index.find(col_key) != table.col_index.end() && row_index < table.data.size()) {
        size_t col_index = table.col_index.at(col_key);
        std::cout << "col_index: " << col_index << '\n';
        std::cout << "row_index: " << row_index << '\n';
        return table.data[row_index][col_index];
    }
    return "Invalid row index or column key";
}

void print_op_spec(const OperationSpec& op_spec) {
    std::cout << "Op spec: " << std::endl;
    std::cout << "GL_CNT: " << op_spec.gl_cnt << std::endl;
    std::cout << "OP_NAME: " << op_spec.op_name << std::endl;
    std::cout << "Input Height: " << op_spec.in0_height << std::endl;
    std::cout << "Input Width: " << op_spec.in0_width << std::endl;
    std::cout << "Batch Size: " << op_spec.batch_size << std::endl;
    std::cout << "Input Channels: " << op_spec.in0_channels << std::endl;
    std::cout << "Filter Height: " << op_spec.filter_height << std::endl;
    std::cout << "Filter Width: " << op_spec.filter_width << std::endl;
    std::cout << "Output Channels: " << op_spec.output_channels << std::endl;
    std::cout << "Stride: " << op_spec.stride << std::endl;
    std::cout << "Pad: " << op_spec.pad << std::endl;
}

void analyze_op(OperationSpec& op_spec, OperationAnalysis& op_analysis, const ExecutionConfig& exec_config) {
    // Calculate output dimensions: relevant for window/stride based OPs (conv, maxpool, downsample)
    op_spec.output_height = std::floor((op_spec.in0_height - op_spec.filter_height + 2 * op_spec.pad) / op_spec.stride + 1);
    op_spec.output_width = std::floor((op_spec.in0_width - op_spec.filter_width + 2 * op_spec.pad) / op_spec.stride + 1);

    // deduct the FW launch latency to get more accurate kernel execution time
    op_analysis.measured_nano_sec -= exec_config.fw_launch_latency_nano_sec;

    // compute the amout of data that needs to be consumed on in0 (Bytes)
    if (op_spec.op_code == "OptimizedConv" || op_spec.op_code == "MaxPool" || op_spec.op_code == "Downsample") {
        // filter window is only relevant for maxpool/convs, it's 1x1, s1 for all ohter OPs
        // for each output we gather a filter window of data from input0
        // for strided OPs, the output is smaller than input, so we need to read less data (eg, downsample)
        op_analysis.in0_read_bytes = op_spec.output_height * op_spec.output_width * op_spec.in0_channels * op_spec.batch_size * exec_config.row_major_act_bytes_per_datum;
        op_analysis.in0_read_bytes *= op_spec.filter_height * op_spec.filter_width;
    } else {
        // other OPs modeled as reading full input
        // for eltwise binary OPs, we read both inputs but the each input and output is limited to 32 B/c because unpacker's 64 B/c is split across in0/in1, so each input and output get 32 B/c
        op_analysis.in0_read_bytes = op_spec.in0_height * op_spec.in0_width * op_spec.in0_channels * op_spec.batch_size * exec_config.tile_act_bytes_per_datum;
    }

    if (op_spec.op_code == "Matmul" || op_spec.op_code == "OptimizedConv") {
        // Calculate number of mul/add operations
        // TODO: add bias modeling
        long long num_mul_adds_per_elem = op_spec.in0_channels * op_spec.filter_height * op_spec.filter_width * 2; // 1 multiply and 1 add per element
        op_analysis.num_mul_adds = num_mul_adds_per_elem * op_spec.output_height * op_spec.output_width * op_spec.output_channels * op_spec.batch_size;

        op_analysis.ideal_dev_clock_cycles = std::ceil(((float)op_analysis.num_mul_adds / (float)(exec_config.device_num_rows * exec_config.device_num_cols * exec_config.tensix_mul_adds_per_cycle_lofi)) * (float)exec_config.num_fidelity_phases);
        op_analysis.weights_bytes = op_spec.filter_height * op_spec.filter_width * op_spec.in0_channels * op_spec.output_channels * exec_config.weights_bytes_per_datum;
    } else {
        // eltwise and data movement OPs
        op_analysis.weights_bytes = 0;
        op_analysis.num_mul_adds = 0; // not modeled for eltwise and data movement OPs
        // divide in0_read_bytes by 32B/c , the ideal BW unpackerA / single NOC can achieve
        op_analysis.ideal_dev_clock_cycles = (float)op_analysis.in0_read_bytes / (32 * exec_config.device_num_rows * exec_config.device_num_cols);
    }

    // common for all OPs
    op_analysis.ideal_dev_nano_sec = std::ceil((float)op_analysis.ideal_dev_clock_cycles / (float)exec_config.frequency_GHZ);
    op_analysis.dev_util_pct = ((float)op_analysis.ideal_dev_nano_sec / (float)op_analysis.measured_nano_sec) * 100;
}

OpList extract_op_list_from_table(const CSVTable& table, const ExecutionConfig& exec_config) {
    OpList op_list;

    std::unordered_set<std::string> supported_OP_CODE {
        "EltwiseBinary",

        "InterleavedToSharded",
        "Untilize",
        "Unpad",
        "Pad",
        "Tilize",
        "UntilizeWithUnpadding",
        "UntilizeWithHalo",
        "Downsample",

        "Reduce",
        "MaxPool",

        "OptimizedConv",
        "tt::operations::primary::Matmul"
    };

    for (const auto& row : table.data) {
        std::string op_code = row[table.col_index.at("OP_CODE")];

        if (supported_OP_CODE.find(op_code) != supported_OP_CODE.end()) {
            OperationSpec op_spec;

            std::unordered_map<std::string, std::string> shorten_OP_CODE = {
                {"tt::operations::primary::Matmul", "Matmul"},
                {"UntilizeWithUnpadding",           "UntileWithUnpad"},
                {"UntilizeWithHalo",                "UntileWithHalo"},
                {"InterleavedToSharded",            "IntrlevToShard"}
            };
            if (shorten_OP_CODE.find(op_code) != shorten_OP_CODE.end()) {
                op_spec.op_code = shorten_OP_CODE.at(op_code);
            } else {
                op_spec.op_code = op_code;
            }

            op_spec.op_name = row[table.col_index.at("OP_NAME")];

            // std::cout << op_code << std::endl;

            op_spec.gl_cnt = std::stoi(row[table.col_index.at("GL_CNT")]);
            op_spec.in0_height = std::stoi(row[table.col_index.at("INPUT_0_Z")]);
            op_spec.in0_width = std::stoi(row[table.col_index.at("INPUT_0_Y")]);
            op_spec.batch_size = exec_config.batch_size;
            op_spec.in0_channels = std::stoi(row[table.col_index.at("INPUT_0_X")]);

            if (op_spec.op_code == "Matmul" || op_spec.op_code == "OptimizedConv") {
                op_spec.output_channels = std::stoi(row[table.col_index.at("INPUT_1_X")]);
            } else {
                op_spec.output_channels = op_spec.in0_channels;
            }

            // window/stride based OPs
            if (op_spec.op_code == "OptimizedConv" || op_spec.op_code == "MaxPool" || op_spec.op_code == "Downsample") {
                op_spec.filter_height = std::stoi(row[table.col_index.at("FILT_H")]);
                op_spec.filter_width = std::stoi(row[table.col_index.at("FILT_W")]);
                op_spec.stride = std::stoi(row[table.col_index.at("STRIDE")]);
                op_spec.pad = std::stoi(row[table.col_index.at("PAD")]);
            } else {
                // non-window OPs
                op_spec.filter_height = 1;
                op_spec.filter_width = 1;
                op_spec.stride = 1;
                op_spec.pad = 0;
            }

            OperationAnalysis op_analysis;
            op_analysis.measured_nano_sec = std::stoi(row[table.col_index.at("MEASURED_NANO_SEC")]);
            analyze_op(op_spec, op_analysis, exec_config);

            op_list.push_back(std::make_pair(op_spec, op_analysis));
        } else {
            std::cout << "Skipping row with OP_CODE: " << row[table.col_index.at("OP_CODE")] << std::endl;
        }
    }
    return op_list;
}

void print_op_table(const OpList& op_list, std::unordered_set<std::string> filter_ops = {}, bool inclusive_filtering = true) {
    // print the filtered ops
    std::cout << std::endl;
    std::cout << "Inclusive filtering: " << std::boolalpha << inclusive_filtering << std::endl;
    std::cout << "Filtered ops: ";
    for (const auto& op_code : filter_ops) {
        std::cout << op_code << ", ";
    }
    std::cout << std::endl;

    std::vector<std::pair<std::string, int>> headers = {
        {"GL_CNT", 6}, {"OP_CODE", 15}, {"OP_NAME", 15},
        {"BS", 2}, {"IN_H", 4}, {"IN_W", 4}, {"IN_CH", 5},
        {"F_H", 4}, {"F_W", 4}, {"ST", 3}, {"PAD", 4},
        {"OUT_H", 5}, {"OUT_W", 5}, {"OUT_CH", 6},
        {"MUL_ADDs", 10}, {"IN0_RD_KB", 10}, {"WGHT_KB", 8},
        {"IDL D CC", 8}, {"IDL D NS", 8}, {"MEAS NS", 7}, {"DEV UTIL", 8}
    };

    // Define the color codes
    const std::string RESET_COLOR = "\033[0m";
    const std::string RED = "\033[31m";
    const std::string ORANGE = "\033[38;5;208m";
    const std::string YELLOW = "\033[33m";
    const std::string GREEN = "\033[32m";

    auto print_border = [&]() {
        for (const auto& header : headers) {
            std::cout << "+";
            for (int i = 0; i < header.second + 2; ++i) std::cout << "-";
        }
        std::cout << "+" << std::endl;
    };

    auto print_header = [&]() {
        for (const auto& header : headers) {
            std::cout << "| " << std::setw(header.second) << header.first << " ";
        }
        std::cout << "|" << std::endl;
    };

    print_border();
    print_header();
    print_border();

    for (const auto& op_meta_data : op_list) {
        const auto& op_spec = op_meta_data.first;
        const auto& op_analysis = op_meta_data.second;

        if (!filter_ops.empty()) {
            if (inclusive_filtering) {
                // if inclusive filtering is enabled and the op is not in the filter list, skip it
                if (filter_ops.find(op_spec.op_code) == filter_ops.end()) continue;
            } else {
                // if inclusive filtering is disabled and the op is in the filter list, skip it
                if (filter_ops.find(op_spec.op_code) != filter_ops.end()) continue;
            }
        }

        int col = 0;

        std::string color;
        if (op_analysis.dev_util_pct > 70.0) {
            color = GREEN;
        } else if (op_analysis.dev_util_pct >= 50.0) {
            color = YELLOW;
        } else if (op_analysis.dev_util_pct >= 25.0) {
            color = ORANGE;
        } else {
            color = RED;
        }

        std::cout << "| " << std::setw(headers[col++].second) << op_spec.gl_cnt << " "
                  << "| " << std::setw(headers[col++].second) << op_spec.op_code << " "
                  << "| " << std::setw(headers[col++].second) << op_spec.op_name << " "

                  << "| " << std::setw(headers[col++].second) << op_spec.batch_size << " "
                  << "| " << std::setw(headers[col++].second) << op_spec.in0_height << " "
                  << "| " << std::setw(headers[col++].second) << op_spec.in0_width << " "
                  << "| " << std::setw(headers[col++].second) << op_spec.in0_channels << " "

                  << "| " << std::setw(headers[col++].second) << op_spec.filter_height << " "
                  << "| " << std::setw(headers[col++].second) << op_spec.filter_width << " "
                  << "| " << std::setw(headers[col++].second) << op_spec.stride << " "
                  << "| " << std::setw(headers[col++].second) << op_spec.pad << " "

                  << "| " << std::setw(headers[col++].second) << op_spec.output_height << " "
                  << "| " << std::setw(headers[col++].second) << op_spec.output_width << " "
                  << "| " << std::setw(headers[col++].second) << op_spec.output_channels << " "

                  << "| " << std::setw(headers[col++].second) << op_analysis.num_mul_adds << " "
                  << "| " << std::setw(headers[col++].second) << (float)op_analysis.in0_read_bytes/1024 << " "
                  << "| " << std::setw(headers[col++].second) << (float)op_analysis.weights_bytes/1024 << " "

                  << "| " << std::setw(headers[col++].second) << op_analysis.ideal_dev_clock_cycles << " "
                  << "| " << std::setw(headers[col++].second) << op_analysis.ideal_dev_nano_sec << " "
                  << "| " << std::setw(headers[col++].second) << op_analysis.measured_nano_sec << " "
                  << "| " << color << std::setw(headers[col++].second) << std::fixed << std::setprecision(1) << op_analysis.dev_util_pct << RESET_COLOR << " "
                  << "|"
                  << std::endl;
        print_border();
    }
}

ModelSummary summarize_model(const OpList& op_list, const ExecutionConfig& exec_config) {
    ModelSummary summary = {0, 0, 0.0f, 0.0f, 0.0f, 0, 0};

    for (const auto& op_meta_data : op_list) {
        const auto& op_spec = op_meta_data.first;
        const auto& op_analysis = op_meta_data.second;

        // Accumulate total execution time
        summary.total_measured_nano_sec += op_analysis.measured_nano_sec;
        summary.total_ideal_dev_nano_sec += op_analysis.ideal_dev_nano_sec;

        // Accumulate total weights size
        if (op_spec.op_code == "Matmul" || op_spec.op_code == "OptimizedConv") {
            summary.total_params_bytes += op_analysis.weights_bytes;
        }
        summary.total_in0_read_bytes += op_analysis.in0_read_bytes;
    }

    summary.measured_throughput = ((float)1e9 / (float)summary.total_measured_nano_sec) * exec_config.batch_size;
    summary.ideal_throughput = ((float)1e9 / (float)summary.total_ideal_dev_nano_sec) * exec_config.batch_size;

    summary.utilization = ((float)summary.total_ideal_dev_nano_sec / (float)summary.total_measured_nano_sec) * 100;

    return summary;
}

void print_model_summary(const ModelSummary& summary) {
    std::cout << std::endl;
    std::cout << "Model Summary:" << std::endl;
    std::cout << "Total Execution Time (ns): " << summary.total_measured_nano_sec << std::endl;
    std::cout << "Total Execution Time (ms): " << (float)summary.total_measured_nano_sec/(1000*1000) << std::endl;
    std::cout << "Measured throughput: " << summary.measured_throughput << std::endl;
    std::cout << "Ideal throughput: " << summary.ideal_throughput << std::endl;
    std::cout << "Utilization: " << summary.utilization << "%" << std::endl;
    std::cout << "Total Parameters Size (Bytes, MB): " << summary.total_params_bytes << ", " << (float)summary.total_params_bytes/(1024*1024) << std::endl;
    std::cout << "Total IN0 Act Read (Bytes, MB): " << summary.total_in0_read_bytes << ", " << (float)summary.total_in0_read_bytes/(1024*1024) << std::endl;
}

void print_execution_breakdown(const OpList& op_list, const ModelSummary& model_summary, const std::vector<std::unordered_set<std::string>>& op_sets) {

    std::vector<long long> op_set_time_nano_sec;
    op_set_time_nano_sec.reserve(op_sets.size());
    long long op_sets_total_nano_sec = 0;

    for (const auto& op_set : op_sets) {
        long long total_time_nano_sec = 0;

        for (const auto& op_meta_data : op_list) {
            const auto& op_spec = op_meta_data.first;
            const auto& op_analysis = op_meta_data.second;
            if (op_set.find(op_spec.op_code) != op_set.end()) {
                total_time_nano_sec += op_analysis.measured_nano_sec;
            }
        }

        op_set_time_nano_sec.push_back(total_time_nano_sec);
        op_sets_total_nano_sec += total_time_nano_sec;
    }

    auto print_border = [&]() {
        std::cout << "--------------------------------------------------------" << std::endl;
    };

    std::cout << "\nModel exec time breakdown: " << std::endl;
    size_t i = 0;  // Index to look up times in op_set_time_nano_sec
    for (const auto& op_set : op_sets) {
        print_border();
        std::cout << "Op Set: ";
        for (const auto& op_name : op_set) {
            std::cout << op_name << ", ";
        }
        std::cout << '\n';

        // print the time for each op set
        std::cout << "\tTime (ns): " << op_set_time_nano_sec[i] << ", ";

        // calculate the percentage of time for each op set based on model summary total time
        float percentage = (float)(op_set_time_nano_sec[i]) / model_summary.total_measured_nano_sec * 100;
        std::cout << "Pct: " << "\t" << percentage << "%\n";

        ++i;
    }

    print_border();
    // print the remaining time
    long long remaining_time_nano_sec = model_summary.total_measured_nano_sec - op_sets_total_nano_sec;
    std::cout << "Other Ops\n\tTime (ns): " << remaining_time_nano_sec << ", ";
    float percentage = (float)(remaining_time_nano_sec) / model_summary.total_measured_nano_sec * 100;
    std::cout << "Pct: " << "\t" << percentage << "%\n";
    print_border();
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <csv-file-name>" << std::endl;
        return 1;
    }

    std::string csv_file_name = argv[1];
    std::cout << "Processing CSV file: " << csv_file_name << std::endl;

    //const std::string csv_file_name = "ResNet50-ops-1089fps.csv"; // ensure the file doesn't have ^M at the end of each line, use dos2unix to remove them
    //const std::string csv_file_name = "ResNet50-ops-1071fps.csv"; // ensure the file doesn't have ^M at the end of each line, use dos2unix to remove them

    CSVTable table = read_csv(csv_file_name);

    trim_spaces(table);

    // print_table_data(table);

    ExecutionConfig exec_config = {
        .tensix_mul_adds_per_cycle_lofi = 2048,
        .num_fidelity_phases = 4,
        .device_num_rows = 9,
        .device_num_cols = 12,
        .frequency_GHZ = 1.2,
        .dram_bw_GBPS = 100,
        .noc_bw_BPC = 32,
        .num_nocs = 2,
        .fw_launch_latency_nano_sec = 1500,

        .weights_bytes_per_datum = 2,
        .tile_act_bytes_per_datum = 2,
        .row_major_act_bytes_per_datum = 2,
        .batch_size = 8
    };

    OpList op_list = extract_op_list_from_table(table, exec_config);
    ModelSummary model_summary = summarize_model(op_list, exec_config);

    print_op_table(op_list);

    print_op_table(op_list, {"Matmul"});
    print_op_table(op_list, {"OptimizedConv"});
    print_op_table(op_list, {"EltwiseBinary"});
    print_op_table(op_list, {"MaxPool", "Reduce"});
    print_op_table(op_list, {"Matmul", "OptimizedConv", "EltwiseBinary", "MaxPool", "Reduce"}, false);

    print_execution_breakdown(op_list, model_summary, {{"Matmul"}, {"OptimizedConv"},
                                       {"EltwiseBinary"}, {"MaxPool", "Reduce"},
                                       {"Untilize", "Tilize", "UntileWithUnpad", "UntileWithHalo", "Pad", "Unpad", "IntrlevToShard", "Downsample"}
                                       });

    print_execution_breakdown(op_list, model_summary, {{"Matmul"}, {"OptimizedConv"},
                                       });

    print_model_summary(model_summary);

    return 0;
}
