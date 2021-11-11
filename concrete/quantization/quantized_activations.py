"""Quantized activation functions."""
import copy
from abc import ABC, abstractmethod
from typing import Optional

import numpy

from .quantized_array import QuantizedArray


class QuantizedActivation(ABC):
    """Base class for quantized activation function."""

    q_out: Optional[QuantizedArray]

    def __init__(self, n_bits) -> None:
        self.n_bits = n_bits
        self.q_out = None

    @abstractmethod
    def __call__(self, q_input: QuantizedArray) -> QuantizedArray:
        """Execute the forward pass."""

    @abstractmethod
    def calibrate(self, x: numpy.ndarray) -> None:
        """Create corresponding QuantizedArray for the output of the activation function.

        Args:
            x (numpy.ndarray): Inputs.
        """

    @staticmethod
    def dequant_input(q_input: QuantizedArray) -> numpy.ndarray:
        """Dequantize the input of the activation function.

        Args:
            q_input (QuantizedArray): Quantized array for the inputs

        Returns:
            numpy.ndarray: Return dequantized input in a numpy array
        """
        return (q_input.qvalues - q_input.zero_point) * q_input.scale

    def quant_output(self, qoutput_activation: numpy.ndarray) -> QuantizedArray:
        """Quantize the output of the activation function.

        Args:
            q_out (numpy.ndarray): Output of the activation function.

        Returns:
            QuantizedArray: Quantized output.
        """
        assert self.q_out is not None

        qoutput_activation = qoutput_activation / self.q_out.scale + self.q_out.zero_point
        qoutput_activation = (
            (qoutput_activation).round().clip(0, 2 ** self.q_out.n_bits - 1).astype(int)
        )

        # TODO find a better way to do the following (see issue #832)
        q_out = copy.copy(self.q_out)
        q_out.update_qvalues(qoutput_activation)
        return q_out


class QuantizedSigmoid(QuantizedActivation):
    """Quantized sigmoid activation function."""

    def calibrate(self, x: numpy.ndarray):
        self.q_out = QuantizedArray(self.n_bits, 1 / (1 + numpy.exp(-x)))

    def __call__(self, q_input: QuantizedArray) -> QuantizedArray:
        """Process the forward pass of the quantized sigmoid.

        Args:
            q_input (QuantizedArray): Quantized input.

        Returns:
            q_out (QuantizedArray): Quantized output.
        """

        quant_sigmoid = self.dequant_input(q_input)
        quant_sigmoid = 1 + numpy.exp(-quant_sigmoid)
        quant_sigmoid = 1 / quant_sigmoid

        q_out = self.quant_output(quant_sigmoid)
        return q_out