"""This file holds the definitions for floating point types"""

from . import base


class Float(base.BaseDataType):
    """Class representing a float"""

    # bit_width is the total number of bits used to represent a floating point number, including
    # sign bit, exponent and mantissa
    bit_width: int

    def __init__(self, bit_width: int) -> None:
        self.bit_width = bit_width

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}<{self.bit_width} bits>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and self.bit_width == other.bit_width


Float32 = lambda: Float(32)
Float64 = lambda: Float(64)