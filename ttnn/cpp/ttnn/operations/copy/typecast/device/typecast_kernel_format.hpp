// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/data_formats/data_format.hpp>

namespace ttnn::prim {

// The typecast compute kernel takes the input and output formats as template arguments and casts them to the
// DataFormat of the architecture it compiles for. On Quasar the codes of MxInt8, MxInt4 and MxInt2 differ from the
// host enumerators (host 12, 16, 17; Quasar 2, 3, 11, tensix_types.h). Passed as host values they name other
// formats (host MxInt4 = 16 is Quasar Fp8P, host MxInt2 = 17 is Quasar UInt8) or none (12), so the kernel would
// not treat the tensor as MX. The other MX formats (MxFp8R/P, MxFp6R/P, MxFp4) have the same code on both sides.
inline std::uint32_t typecast_kernel_data_format(tt::DataFormat format, bool is_quasar) {
    if (is_quasar) {
        switch (format) {
            case tt::DataFormat::MxInt8: return 2;
            case tt::DataFormat::MxInt4: return 3;
            case tt::DataFormat::MxInt2: return 11;
            default: break;
        }
    }
    return static_cast<std::uint32_t>(format);
}

}  // namespace ttnn::prim
