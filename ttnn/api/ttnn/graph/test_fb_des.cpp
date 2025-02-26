#include <iostream>
#include <fstream>
#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/reshard_generated.h"

using namespace ttnn;

int main() {
    std::ifstream in("build/v.bin", std::ios::binary | std::ios::in);
    if (!in.is_open()) {
        std::cerr << "FATAL: file cannot be opened!" << std::endl;
        return 1;
    }
    in.seekg(0, std::ios::end);
    int size = in.tellg();
    std::cout << "  size = " << size << std::endl;
    in.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (in.read(buffer.data(), size)) {
        in.close();
    }

    auto table = GetReshardTable(buffer.data());
    size_t num_rows = table->rows()->size();
    std::cout << "Found " << num_rows << " rows of data." << std::endl;
    for (size_t i = 0; i < num_rows; i++) {
        auto row = table->rows()->Get(i);
        std::cout << "  Row " << i << ", in_grid=" << row->in_grid()->c_str()
                  << ", out_grid=" << row->out_grid()->c_str() << std::endl;
    }

    return 0;
}
