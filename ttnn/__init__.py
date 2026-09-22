"""Small, device-independent quantization helpers.

The saturation is deliberately performed before conversion to an unsigned
integer.  Converting a negative integer to uint8 first would wrap it modulo
256 and is the source of the lower-bound bug this module guards against.
"""

from .quantization import quantize, requantize

__all__ = ["quantize", "requantize"]
