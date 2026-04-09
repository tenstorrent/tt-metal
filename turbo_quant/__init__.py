from turbo_quant.quantizer import TurboQuantMSE, TurboQuantProd, OutlierAwareTurboQuant
from turbo_quant.kv_cache import TurboQuantCache, TurboQuantLayer
from turbo_quant.llama_integration import create_turbo_quant_cache, generate_with_turbo_quant
from turbo_quant.bitpack import pack, unpack

__all__ = [
    "TurboQuantMSE",
    "TurboQuantProd",
    "OutlierAwareTurboQuant",
    "TurboQuantCache",
    "TurboQuantLayer",
    "create_turbo_quant_cache",
    "generate_with_turbo_quant",
    "pack",
    "unpack",
]

# Conditional TTNN exports (only available with Tenstorrent hardware)
try:
    from turbo_quant.ttnn_integration import (
        TTNNTurboQuantSetup,
        TTNNTurboQuantCache,
        turbo_quant_quantize,
        turbo_quant_dequantize,
    )

    __all__ += [
        "TTNNTurboQuantSetup",
        "TTNNTurboQuantCache",
        "turbo_quant_quantize",
        "turbo_quant_dequantize",
    ]
except ImportError:
    pass
