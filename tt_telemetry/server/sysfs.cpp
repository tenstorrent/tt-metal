/*
 * sysfs.cpp
 *
 * Sysfs metrics discovery and printing functionality.
 * Equivalent to the Python sysfs scraping code for discovering Tenstorrent card metrics.
 *
 * TODO:
 * -----
 * - Joel Smith mentioned an issue that could occur:

Joel Smith
  7 minutes ago
If you're working with WH 6U and telemetry there are two things you should know regarding reset:
if you reset the chips, you'll lose telemetry: https://tenstorrent.atlassian.net/browse/SYS-1920
you cannot assume /dev/tenstorrent/x will still be the same ASIC pre- and post-reset
I have solutions/workarounds for both of these, but they are not deployed anywhere.  Just prototypes.





Joel Smith
  6 minutes ago
Quick workaround for the first one is to wait about a minute after reset, then unload/reload the driver.  Happy to chat if you run into any issues here.


Bart Trzynadlowski
  3 minutes ago
Ah I won’t be aware of a reset


Bart Trzynadlowski
  3 minutes ago
Should I just always read the corresponding serial as well and match at sample time?
New


Joel Smith
  Just now
There's an attribute named asic_id that will show up if your FW is new enough.  I think it's guaranteed to be unique.  I am not sure if you can/should rely on serial being unique, although maybe it's a reasonable assumption for 6U.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <regex>
#include <filesystem>
#include <optional>
#include <unistd.h>
#include <cstdlib>
#include <array>
#include <memory>

namespace fs = std::filesystem;

// Constants
const fs::path ROOT_TT_DIR = "/sys/class/tenstorrent";
const fs::path ROOT_HWMON_DIR = "/sys/class/hwmon";
const std::regex PCI_PATTERN(R"(pci(.{4}:.{2}(?:/.{4}:.{2}:.{2}\..)+))");

// Data types enum equivalent to Python DataType
enum class DataType {
    // Host metadata
    HOST_NAME,
    
    // Card metadata
    CARD_NAME,
    CARD_TYPE,
    SERIAL,
    TT_DRIVER_VERSION,
    TT_FIRMWARE_BUNDLE_VERSION,
    TT_FLASH_VERSION,
    
    // ASIC metadata
    TT_ASIC_ID_HASHED,
    
    // Card metrics
    AI_CLK,
    ARC_CLK,
    CURRENT,
    POWER,
    TEMPERATURE,
    VOLTAGE
};

// Data type information structure
struct DataTypeInfo {
    std::string name;
    std::string docs;
    std::string unit;
    bool is_nullable;
    bool is_string_type;
};

// Data type information map
const std::map<DataType, DataTypeInfo> DATA_TYPE_INFO = {
    {DataType::HOST_NAME, {"host_name", "Hostname of the system", "", false, true}},
    {DataType::CARD_NAME, {"card_name", "Name of the card", "", false, true}},
    {DataType::CARD_TYPE, {"card_type", "Type of the card", "", false, true}},
    {DataType::SERIAL, {"card_serial", "Serial number of the card", "", false, true}},
    {DataType::TT_DRIVER_VERSION, {"tt_driver_version", "Tenstorrent driver version", "", false, true}},
    {DataType::TT_FIRMWARE_BUNDLE_VERSION, {"tt_firmware_bundle_version", "Firmware bundle version", "", false, true}},
    {DataType::TT_FLASH_VERSION, {"tt_flash_version", "Flash version", "", false, true}},
    {DataType::TT_ASIC_ID_HASHED, {"tt_asic_id_hashed", "Unique ASIC identifier (hashed)", "", true, true}},
    {DataType::AI_CLK, {"ai_clk", "AI clock frequency", "megahertz", false, false}},
    {DataType::ARC_CLK, {"arc_clk", "ARC clock frequency; note that this is unrelated to the AI part of the chip", "megahertz", false, false}},
    {DataType::CURRENT, {"current", "Amount of current that the sole voltage regulator provides to the chip", "milliamps", false, false}},
    {DataType::POWER, {"power", "Output power of the voltage regulator", "microwatts", false, false}},
    {DataType::TEMPERATURE, {"temperature", "Average temperature of the chip over a number of temperature sensors, ignoring some bugs in the individual readings", "millicelsius", false, false}},
    {DataType::VOLTAGE, {"voltage", "Input voltage to the chip", "millivolts", false, false}}
};

// Card information structure
struct CardInfo {
    fs::path tt_dir;
    std::optional<fs::path> hwmon_dir;  // Optional - not all cards have hwmon
    std::string pci_path;
};

// Utility functions
std::string exec_command(const std::string& cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    // Remove trailing newline
    if (!result.empty() && result.back() == '\n') {
        result.pop_back();
    }
    return result;
}

std::optional<std::string> read_file(const fs::path& path) {
    if (!fs::exists(path)) {
        return std::nullopt;
    }
    
    std::ifstream file(path);
    if (!file.is_open()) {
        return std::nullopt;
    }
    
    std::string content;
    std::string line;
    while (std::getline(file, line)) {
        if (!content.empty()) {
            content += "\n";
        }
        content += line;
    }
    
    // Trim whitespace
    content.erase(0, content.find_first_not_of(" \t\n\r"));
    content.erase(content.find_last_not_of(" \t\n\r") + 1);
    
    return content;
}

std::string get_hostname() {
    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        return std::string(hostname);
    }
    return "unknown";
}

std::optional<std::string> extract_pci_path(const fs::path& sys_dir) {
    try {
        fs::path resolved = fs::canonical(sys_dir);
        std::string resolved_str = resolved.string();
        
        std::smatch match;
        if (std::regex_search(resolved_str, match, PCI_PATTERN)) {
            return match[1].str();
        }
    } catch (const std::exception&) {
        // Ignore errors in path resolution
    }
    return std::nullopt;
}

std::optional<std::string> read_data_type(DataType data_type, const fs::path& tt_dir, const std::optional<fs::path>& hwmon_dir) {
    try {
        switch (data_type) {
            case DataType::HOST_NAME:
                return get_hostname();
                
            case DataType::CARD_NAME:
                return hwmon_dir ? read_file(*hwmon_dir / "name") : std::nullopt;
                
            case DataType::CARD_TYPE:
                return read_file(tt_dir / "tt_card_type");
                
            case DataType::SERIAL:
                return read_file(tt_dir / "tt_serial");
                
            case DataType::TT_DRIVER_VERSION:
                try {
                    return exec_command("modinfo tenstorrent | grep ^version | awk '{print $2}'");
                } catch (const std::exception&) {
                    return std::nullopt;
                }
                
            case DataType::TT_FIRMWARE_BUNDLE_VERSION:
                return read_file(tt_dir / "tt_fw_bundle_ver");
                
            case DataType::TT_FLASH_VERSION:
                return read_file(tt_dir / "tt_ttflash_ver");
                
            case DataType::TT_ASIC_ID_HASHED:
                return read_file(tt_dir / "tt_asic_id");
                
            case DataType::AI_CLK:
                return read_file(tt_dir / "tt_aiclk");
                
            case DataType::ARC_CLK:
                return read_file(tt_dir / "tt_arcclk");
                
            case DataType::CURRENT:
                return hwmon_dir ? read_file(*hwmon_dir / "curr1_input") : std::nullopt;
                
            case DataType::POWER:
                return hwmon_dir ? read_file(*hwmon_dir / "power1_input") : std::nullopt;
                
            case DataType::VOLTAGE:
                return hwmon_dir ? read_file(*hwmon_dir / "in0_input") : std::nullopt;
                
            case DataType::TEMPERATURE:
                return hwmon_dir ? read_file(*hwmon_dir / "temp1_input") : std::nullopt;
                
            default:
                return std::nullopt;
        }
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

std::vector<CardInfo> discover_cards() {
    std::vector<CardInfo> cards;
    
    if (!fs::exists(ROOT_TT_DIR) || !fs::exists(ROOT_HWMON_DIR)) {
        std::cerr << "Warning: sysfs directories not found (" << ROOT_TT_DIR << " or " << ROOT_HWMON_DIR << ")" << std::endl;
        return cards;
    }
    
    // Discover tenstorrent directories
    std::map<std::string, fs::path> tt_dirs_by_pci;
    std::regex tt_pattern(R"(tenstorrent!\d+)");
    
    try {
        for (const auto& entry : fs::directory_iterator(ROOT_TT_DIR)) {
            if (entry.is_directory()) {
                std::string stem = entry.path().stem().string();
                if (std::regex_match(stem, tt_pattern)) {
                    auto pci_path = extract_pci_path(entry.path());
                    if (pci_path) {
                        tt_dirs_by_pci[*pci_path] = entry.path();
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error scanning tenstorrent directory: " << e.what() << std::endl;
    }
    
    // Discover hwmon directories
    std::map<std::string, fs::path> hwmon_dirs_by_pci;
    std::regex hwmon_pattern(R"(hwmon\d+)");
    
    try {
        for (const auto& entry : fs::directory_iterator(ROOT_HWMON_DIR)) {
            if (entry.is_directory()) {
                std::string stem = entry.path().stem().string();
                if (std::regex_match(stem, hwmon_pattern)) {
                    auto pci_path = extract_pci_path(entry.path());
                    if (pci_path) {
                        hwmon_dirs_by_pci[*pci_path] = entry.path();
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error scanning hwmon directory: " << e.what() << std::endl;
    }
    
    // Create CardInfo for all tenstorrent directories, with optional hwmon
    for (const auto& [pci_path, tt_dir] : tt_dirs_by_pci) {
        auto hwmon_it = hwmon_dirs_by_pci.find(pci_path);
        std::optional<fs::path> hwmon_dir = (hwmon_it != hwmon_dirs_by_pci.end()) ? 
            std::optional<fs::path>(hwmon_it->second) : std::nullopt;
        cards.push_back({tt_dir, hwmon_dir, pci_path});
    }
    
    return cards;
}

void print_sysfs_metrics() {
    std::cout << "\n=== Sysfs Metrics Discovery ===" << std::endl;
    
    auto cards = discover_cards();
    
    if (cards.empty()) {
        std::cout << "No Tenstorrent cards found in sysfs." << std::endl;
        return;
    }
    
    std::cout << "Found " << cards.size() << " Tenstorrent card(s):" << std::endl;
    
    for (size_t card_idx = 0; card_idx < cards.size(); ++card_idx) {
        const auto& card = cards[card_idx];
        
        std::cout << "\n--- Card " << (card_idx + 1) << " ---" << std::endl;
        std::cout << "TT Directory: " << card.tt_dir << std::endl;
        std::cout << "HWMON Directory: " << (card.hwmon_dir ? card.hwmon_dir->string() : "<not available>") << std::endl;
        std::cout << "PCI Path: " << card.pci_path << std::endl;
        
        // Print all data types
        std::cout << "\nMetrics and Metadata:" << std::endl;
        
        for (const auto& [data_type, info] : DATA_TYPE_INFO) {
            auto value = read_data_type(data_type, card.tt_dir, card.hwmon_dir);
            
            std::cout << "  " << info.name << ":";
            
            if (value) {
                std::cout << " " << *value;
                if (!info.unit.empty()) {
                    std::cout << " " << info.unit;
                }
            } else {
                if (info.is_nullable) {
                    std::cout << " <not available>";
                } else {
                    std::cout << " <ERROR: missing required file>";
                }
            }
            
            std::cout << std::endl;
            std::cout << "    Description: " << info.docs << std::endl;
            
            // Show the file path being read for debugging
            switch (data_type) {
                case DataType::HOST_NAME:
                    std::cout << "    Source: hostname()" << std::endl;
                    break;
                case DataType::CARD_NAME:
                    std::cout << "    Source: " << (card.hwmon_dir ? (*card.hwmon_dir / "name").string() : "<hwmon not available>") << std::endl;
                    break;
                case DataType::CARD_TYPE:
                    std::cout << "    Source: " << (card.tt_dir / "tt_card_type") << std::endl;
                    break;
                case DataType::SERIAL:
                    std::cout << "    Source: " << (card.tt_dir / "tt_serial") << std::endl;
                    break;
                case DataType::TT_DRIVER_VERSION:
                    std::cout << "    Source: modinfo tenstorrent command" << std::endl;
                    break;
                case DataType::TT_FIRMWARE_BUNDLE_VERSION:
                    std::cout << "    Source: " << (card.tt_dir / "tt_fw_bundle_ver") << std::endl;
                    break;
                case DataType::TT_FLASH_VERSION:
                    std::cout << "    Source: " << (card.tt_dir / "tt_ttflash_ver") << std::endl;
                    break;
                case DataType::TT_ASIC_ID_HASHED:
                    std::cout << "    Source: " << (card.tt_dir / "tt_asic_id") << std::endl;
                    break;
                case DataType::AI_CLK:
                    std::cout << "    Source: " << (card.tt_dir / "tt_aiclk") << std::endl;
                    break;
                case DataType::ARC_CLK:
                    std::cout << "    Source: " << (card.tt_dir / "tt_arcclk") << std::endl;
                    break;
                case DataType::CURRENT:
                    std::cout << "    Source: " << (card.hwmon_dir ? (*card.hwmon_dir / "curr1_input").string() : "<hwmon not available>") << std::endl;
                    break;
                case DataType::POWER:
                    std::cout << "    Source: " << (card.hwmon_dir ? (*card.hwmon_dir / "power1_input").string() : "<hwmon not available>") << std::endl;
                    break;
                case DataType::VOLTAGE:
                    std::cout << "    Source: " << (card.hwmon_dir ? (*card.hwmon_dir / "in0_input").string() : "<hwmon not available>") << std::endl;
                    break;
                case DataType::TEMPERATURE:
                    std::cout << "    Source: " << (card.hwmon_dir ? (*card.hwmon_dir / "temp1_input").string() : "<hwmon not available>") << std::endl;
                    break;
            }
            std::cout << std::endl;
        }
    }
    
    // Print summary
    size_t cards_with_hwmon = 0;
    size_t cards_without_hwmon = 0;
    for (const auto& card : cards) {
        if (card.hwmon_dir) {
            cards_with_hwmon++;
        } else {
            cards_without_hwmon++;
        }
    }
    
    std::cout << "\n--- Summary ---" << std::endl;
    std::cout << "Total Tenstorrent devices found: " << cards.size() << std::endl;
    std::cout << "Devices with hwmon data: " << cards_with_hwmon << std::endl;
    std::cout << "Devices without hwmon data: " << cards_without_hwmon << std::endl;
    
    if (cards_without_hwmon > 0) {
        std::cout << "\nNote: Devices without hwmon data will not have power/thermal metrics available." << std::endl;
        std::cout << "This is normal for some Tenstorrent device configurations." << std::endl;
    }
    
    std::cout << "=== End Sysfs Metrics Discovery ===\n" << std::endl;
}
