"""
Declaration of `Node` class.
"""

from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ..internal.utils import assert_that
from ..values import Value
from .evaluator import ConstantEvaluator, GenericEvaluator, GenericTupleEvaluator, InputEvaluator
from .operation import Operation
from .utils import KWARGS_IGNORED_IN_FORMATTING, format_constant, format_indexing_element


class Node:
    """
    Node class, to represent computation in a computation graph.
    """

    inputs: List[Value]
    output: Value

    operation: Operation
    evaluator: Callable

    properties: Dict[str, Any]

    @staticmethod
    def constant(constant: Any) -> "Node":
        """
        Create an Operation.Constant node.

        Args:
            constant (Any):
                constant to represent

        Returns:
            Node:
                node representing constant

        Raises:
            ValueError:
                if the constant is not representable
        """

        try:
            value = Value.of(constant)
        except Exception as error:
            raise ValueError(f"Constant {repr(constant)} is not supported") from error

        properties = {"constant": np.array(constant)}
        return Node([], value, Operation.Constant, ConstantEvaluator(properties), properties)

    @staticmethod
    def generic(
        name: str,
        inputs: List[Value],
        output: Value,
        operation: Callable,
        args: Optional[Tuple[Any, ...]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """
        Create an Operation.Generic node.

        Args:
            name (str):
                name of the operation

            inputs (List[Value]):
                inputs to the operation

            output (Value):
                output of the operation

            operation (Callable):
                operation itself

            args (Optional[Tuple[Any, ...]]):
                args to pass to operation during evaluation

            kwargs (Optional[Dict[str, Any]]):
                kwargs to pass to operation during evaluation

            attributes (Optional[Dict[str, Any]]):
                attributes of the operation

        Returns:
            Node:
                node representing operation
        """

        properties = {
            "name": name,
            "args": args if args is not None else (),
            "kwargs": kwargs if kwargs is not None else {},
            "attributes": attributes if attributes is not None else {},
        }

        return Node(
            inputs,
            output,
            Operation.Generic,
            (
                GenericTupleEvaluator(operation, properties)  # type: ignore
                if name in ["concatenate"]
                else GenericEvaluator(operation, properties)  # type: ignore
            ),
            properties,
        )

    @staticmethod
    def input(name: str, value: Value) -> "Node":
        """
        Create an Operation.Input node.

        Args:
            name (Any):
                name of the input

            value (Any):
                value of the input

        Returns:
            Node:
                node representing input
        """

        return Node([value], value, Operation.Input, InputEvaluator(), {"name": name})

    def __init__(
        self,
        inputs: List[Value],
        output: Value,
        operation: Operation,
        evaluator: Callable,
        properties: Optional[Dict[str, Any]] = None,
    ):
        self.inputs = inputs
        self.output = output

        self.operation = operation
        self.evaluator = evaluator  # type: ignore

        self.properties = properties if properties is not None else {}

    def __call__(self, *args: List[Any]) -> Union[np.bool_, np.integer, np.floating, np.ndarray]:
        def generic_error_message() -> str:
            result = f"Evaluation of {self.operation.value} '{self.label()}' node"
            if len(args) != 0:
                result += f" using {', '.join(repr(arg) for arg in args)}"
            return result

        if len(args) != len(self.inputs):
            raise ValueError(
                f"{generic_error_message()} failed because of invalid number of arguments"
            )

        for arg, input_ in zip(args, self.inputs):
            try:
                arg_value = Value.of(arg)
            except Exception as error:
                arg_str = "the argument" if len(args) == 1 else f"argument {repr(arg)}"
                raise ValueError(
                    f"{generic_error_message()} failed because {arg_str} is not valid"
                ) from error

            if input_.shape != arg_value.shape:
                arg_str = "the argument" if len(args) == 1 else f"argument {repr(arg)}"
                raise ValueError(
                    f"{generic_error_message()} failed because "
                    f"{arg_str} does not have the expected "
                    f"shape of {input_.shape}"
                )

        result = self.evaluator(*args)

        if isinstance(result, int) and -(2**63) < result < (2**63) - 1:
            result = np.int64(result)
        if isinstance(result, float):
            result = np.float64(result)

        if isinstance(result, list):
            try:
                np_result = np.array(result)
                result = np_result
            except Exception:  # pylint: disable=broad-except
                # here we try our best to convert the list to np.ndarray
                # if it fails we raise the exception below
                pass

        if not isinstance(result, (np.bool_, np.integer, np.floating, np.ndarray)):
            raise ValueError(
                f"{generic_error_message()} resulted in {repr(result)} "
                f"of type {result.__class__.__name__} "
                f"which is not acceptable either because of the type or because of overflow"
            )

        if isinstance(result, np.ndarray):
            dtype = result.dtype
            if (
                not np.issubdtype(dtype, np.integer)
                and not np.issubdtype(dtype, np.floating)
                and not np.issubdtype(dtype, np.bool_)
            ):
                raise ValueError(
                    f"{generic_error_message()} resulted in {repr(result)} "
                    f"of type np.ndarray and of underlying type '{type(dtype).__name__}' "
                    f"which is not acceptable because of the underlying type"
                )

        if result.shape != self.output.shape:
            raise ValueError(
                f"{generic_error_message()} resulted in {repr(result)} "
                f"which does not have the expected "
                f"shape of {self.output.shape}"
            )

        return result

    def format(self, predecessors: List[str], maximum_constant_length: int = 45) -> str:
        """
        Get the textual representation of the `Node` (for printing).

        Args:
            predecessors (List[str]):
                predecessor names to this node

            maximum_constant_length (int, default = 45):
                maximum length of formatted constants

        Returns:
            str:
                textual representation of the `Node` (for printing)
        """

        if self.operation == Operation.Constant:
            return format_constant(self(), maximum_constant_length)

        if self.operation == Operation.Input:
            return self.properties["name"]

        assert_that(self.operation == Operation.Generic)

        name = self.properties["name"]

        if name == "index.static":
            index = self.properties["attributes"]["index"]
            elements = [format_indexing_element(element) for element in index]
            return f"{predecessors[0]}[{', '.join(elements)}]"

        if name == "concatenate":
            args = [f"({', '.join(predecessors)})"]
        else:
            args = deepcopy(predecessors)

        if name == "array":
            values = str(np.array(predecessors).reshape(self.output.shape).tolist()).replace(
                "'", ""
            )
            return f"array({format_constant(values, maximum_constant_length)})"

        args.extend(
            format_constant(value, maximum_constant_length) for value in self.properties["args"]
        )
        args.extend(
            f"{name}={format_constant(value, maximum_constant_length)}"
            for name, value in self.properties["kwargs"].items()
            if name not in KWARGS_IGNORED_IN_FORMATTING
        )

        return f"{name}({', '.join(args)})"

    def label(self) -> str:
        """
        Get the textual representation of the `Node` (for drawing).

        Returns:
            str:
                textual representation of the `Node` (for drawing).
        """

        if self.operation == Operation.Constant:
            return format_constant(self(), maximum_length=45, keep_newlines=True)

        if self.operation == Operation.Input:
            return self.properties["name"]

        assert_that(self.operation == Operation.Generic)

        name = self.properties["name"]
        return name if name != "index.static" else self.format(["index"])

    @property
    def converted_to_table_lookup(self) -> bool:
        """
        Get whether the node is converted to a table lookup during MLIR conversion.

        Returns:
            bool:
                True if the node is converted to a table lookup, False otherwise
        """

        return self.operation == Operation.Generic and self.properties["name"] not in [
            "add",
            "array",
            "concatenate",
            "conv1d",
            "conv2d",
            "conv3d",
            "dot",
            "index.static",
            "matmul",
            "multiply",
            "negative",
            "ones",
            "reshape",
            "subtract",
            "sum",
            "transpose",
            "zeros",
        ]

    @property
    def is_fusable(self) -> bool:
        """
        Get whether the node is can be fused into a table lookup.

        Returns:
            bool:
                True if the node can be fused into a table lookup, False otherwise
        """

        if self.converted_to_table_lookup:
            return True

        return self.operation != Operation.Generic or self.properties["name"] in [
            "add",
            "multiply",
            "negative",
            "ones",
            "subtract",
            "zeros",
        ]