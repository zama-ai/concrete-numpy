"""
Declaration of `NodeConverter` class.
"""

# pylint: disable=no-member,no-name-in-module

from typing import Dict, List, Tuple

import numpy as np
from concrete.lang.dialects import fhe, fhelinalg
from concrete.lang.dialects.fhe import EncryptedIntegerType
from mlir.dialects import arith, linalg, tensor
from mlir.ir import (
    ArrayAttr,
    Attribute,
    BoolAttr,
    Context,
    DenseElementsAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    OpResult,
    RankedTensorType,
    Type,
)

from ..dtypes import Integer
from ..internal.utils import assert_that
from ..representation import Graph, Node, Operation
from ..values import Value
from .utils import construct_deduplicated_tables

# pylint: enable=no-member,no-name-in-module


class NodeConverter:
    """
    NodeConverter class, to convert computation graph nodes to their MLIR equivalent.
    """

    ctx: Context
    graph: Graph
    node: Node
    preds: List[OpResult]

    all_of_the_inputs_are_encrypted: bool
    all_of_the_inputs_are_tensors: bool
    one_of_the_inputs_is_a_tensor: bool

    nodes_to_mlir_names: Dict[Node, str]
    mlir_names_to_mlir_types: Dict[str, str]
    scalar_to_1d_tensor_conversion_hacks: Dict[str, List[str]]

    @staticmethod
    def value_to_mlir_type(ctx: Context, value: Value) -> Type:
        """
        Convert a `Value` to its corresponding MLIR `Type`.

        Args:
            ctx (Context):
                MLIR context to perform the conversion

            value (Value):
                value to convert

        Returns:
            Type:
                MLIR `Type` corresponding to `value`
        """

        dtype = value.dtype

        if isinstance(dtype, Integer):
            if value.is_encrypted:
                result = EncryptedIntegerType.get(ctx, dtype.bit_width)
            else:
                result = IntegerType.get_signless(dtype.bit_width)

            return result if value.is_scalar else RankedTensorType.get(value.shape, result)

        # the branch above is always taken due to compatibility checks
        # still, it's a good idea to raise an appropriate error, just in case

        raise ValueError(f"{value} cannot be converted to MLIR")  # pragma: no cover

    def __init__(
        self,
        ctx: Context,
        graph: Graph,
        node: Node,
        preds: List[OpResult],
        nodes_to_mlir_names: Dict[OpResult, str],
        mlir_names_to_mlir_types: Dict[str, str],
        scalar_to_1d_tensor_conversion_hacks: Dict[str, List[str]],
    ):
        self.ctx = ctx
        self.graph = graph
        self.node = node
        self.preds = preds

        self.all_of_the_inputs_are_encrypted = True
        self.all_of_the_inputs_are_tensors = True
        self.one_of_the_inputs_is_a_tensor = False

        for inp in node.inputs:
            if not inp.is_encrypted:
                self.all_of_the_inputs_are_encrypted = False

            if inp.is_scalar:
                self.all_of_the_inputs_are_tensors = False
            else:
                self.one_of_the_inputs_is_a_tensor = True

        self.nodes_to_mlir_names = nodes_to_mlir_names
        self.mlir_names_to_mlir_types = mlir_names_to_mlir_types
        self.scalar_to_1d_tensor_conversion_hacks = scalar_to_1d_tensor_conversion_hacks

    def convert(self) -> OpResult:
        """
        Convert a node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """

        # pylint: disable=too-many-branches

        if self.node.operation == Operation.Constant:
            result = self.convert_constant()
        else:
            assert_that(self.node.operation == Operation.Generic)

            name = self.node.properties["name"]

            if name == "add":
                result = self.convert_add()

            elif name == "concatenate":
                result = self.convert_concat()

            elif name == "conv1d":
                result = self.convert_conv1d()

            elif name == "conv2d":
                result = self.convert_conv2d()

            elif name == "conv3d":
                result = self.convert_conv3d()

            elif name == "dot":
                result = self.convert_dot()

            elif name == "index.static":
                result = self.convert_static_indexing()

            elif name == "matmul":
                result = self.convert_matmul()

            elif name == "multiply":
                result = self.convert_mul()

            elif name == "negative":
                result = self.convert_neg()

            elif name == "reshape":
                result = self.convert_reshape()

            elif name == "subtract":
                result = self.convert_sub()

            elif name == "sum":
                result = self.convert_sum()

            elif name == "transpose":
                result = self.convert_transpose()

            else:
                result = self.convert_tlu()

        mlir_name = str(result).replace("Value(", "").split("=", maxsplit=1)[0].strip()

        self.nodes_to_mlir_names[self.node] = mlir_name
        self.mlir_names_to_mlir_types[mlir_name] = str(result.type)

        if self.node.operation == Operation.Generic:
            name = self.node.properties["name"]
            if name in ["add", "dot", "multiply", "subtract"]:
                if self.one_of_the_inputs_is_a_tensor and not self.all_of_the_inputs_are_tensors:
                    to_be_converted = []
                    for pred in self.graph.ordered_preds_of(self.node):
                        if pred.output.is_scalar:
                            to_be_converted.append(self.nodes_to_mlir_names[pred])
                    self.scalar_to_1d_tensor_conversion_hacks[mlir_name] = to_be_converted

        return result

        # pylint: enable=too-many-branches

    def convert_add(self) -> OpResult:
        """
        Convert "add" node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """

        resulting_type = NodeConverter.value_to_mlir_type(self.ctx, self.node.output)
        preds = self.preds

        if self.all_of_the_inputs_are_encrypted:
            if self.one_of_the_inputs_is_a_tensor:
                result = fhelinalg.AddEintOp(resulting_type, *preds).result
            else:
                result = fhe.AddEintOp(resulting_type, *preds).result
        else:
            if self.node.inputs[0].is_clear:
                preds = preds[::-1]

            if self.one_of_the_inputs_is_a_tensor:
                result = fhelinalg.AddEintIntOp(resulting_type, *preds).result
            else:
                result = fhe.AddEintIntOp(resulting_type, *preds).result

        return result

    def convert_concat(self) -> OpResult:
        """
        Convert "concatenate" node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """

        resulting_type = NodeConverter.value_to_mlir_type(self.ctx, self.node.output)
        axis = self.node.properties["kwargs"].get("axis", 0)

        if axis is not None:
            if axis < 0:
                axis += len(self.node.inputs[0].shape)
            return fhelinalg.ConcatOp(
                resulting_type,
                self.preds,
                IntegerAttr.get(IntegerType.get_signless(64), axis),
            ).result

        flattened_preds = []
        for pred, input_value in zip(self.preds, self.node.inputs):
            input_shape = input_value.shape
            input_size = np.prod(input_shape)

            flattened_pred_type = RankedTensorType.get(
                [input_size],
                NodeConverter.value_to_mlir_type(
                    self.ctx,
                    Value(input_value.dtype, shape=(), is_encrypted=input_value.is_encrypted),
                ),
            )
            flattened_pred = linalg.TensorCollapseShapeOp(
                flattened_pred_type,
                pred,
                ArrayAttr.get(
                    [
                        ArrayAttr.get(
                            [
                                IntegerAttr.get(IndexType.parse("index"), i)
                                for i in range(len(input_shape))
                            ]
                        )
                    ]
                ),
            ).result
            flattened_preds.append(flattened_pred)

        return fhelinalg.ConcatOp(
            resulting_type,
            flattened_preds,
            IntegerAttr.get(IntegerType.get_signless(64), 0),
        ).result

    def convert_constant(self) -> OpResult:
        """
        Convert Operation.Constant node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """

        resulting_type = NodeConverter.value_to_mlir_type(self.ctx, self.node.output)
        data = self.node()

        if self.node.output.is_scalar:
            attr = IntegerAttr.get(resulting_type, data)
        else:
            # usage of `Attribute.parse` is the result of some limitations in the MLIR module
            # provided by LLVM

            # what should have been used is `DenseElementsAttr` but it's impossible to assign
            # custom bit-widths using it (e.g., uint5)

            # since we couldn't create a `DenseElementsAttr` with a custom bit width using
            # the python api we use `Attribute.parse` to let the underlying library do it by itself

            attr = Attribute.parse(f"dense<{str(data.tolist())}> : {resulting_type}")

        return arith.ConstantOp(resulting_type, attr).result

    def convert_conv1d(self) -> OpResult:
        """
        Convert "conv1d" node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """

        raise NotImplementedError("conv1d conversion to MLIR is not yet implemented")

    def convert_conv2d(self) -> OpResult:
        """
        Convert "conv2d" node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """

        resulting_type = NodeConverter.value_to_mlir_type(self.ctx, self.node.output)
        preds = self.preds

        integer_type = IntegerType.get_signless(64, context=self.ctx)

        strides = DenseElementsAttr.get(
            np.array(list(self.node.properties["kwargs"]["strides"]), dtype=np.uint64),
            type=integer_type,
            context=self.ctx,
        )
        dilations = DenseElementsAttr.get(
            np.array(list(self.node.properties["kwargs"]["dilations"]), dtype=np.uint64),
            type=integer_type,
            context=self.ctx,
        )
        pads = DenseElementsAttr.get(
            np.array(list(self.node.properties["kwargs"]["pads"]), dtype=np.uint64),
            type=integer_type,
            context=self.ctx,
        )

        has_bias = len(self.node.inputs) == 3
        if not has_bias:
            preds.append(None)

        return fhelinalg.Conv2dOp(resulting_type, *preds, pads, strides, dilations).result

    def convert_conv3d(self) -> OpResult:
        """
        Convert "conv3d" node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """

        raise NotImplementedError("conv3d conversion to MLIR is not yet implemented")

    def convert_dot(self) -> OpResult:
        """
        Convert "dot" node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """

        resulting_type = NodeConverter.value_to_mlir_type(self.ctx, self.node.output)
        preds = self.preds

        if self.node.inputs[0].is_clear:
            preds = preds[::-1]

        if self.all_of_the_inputs_are_tensors:
            # numpy.dot(x, y) where x and y are both vectors = regular dot product
            result = fhelinalg.Dot(resulting_type, *preds).result

        elif not self.one_of_the_inputs_is_a_tensor:
            # numpy.dot(x, y) where x and y are both scalars = x * y
            result = fhe.MulEintIntOp(resulting_type, *preds).result

        else:
            # numpy.dot(x, y) where one of x or y is a scalar and the other one is a vector = x * y
            result = fhelinalg.MulEintIntOp(resulting_type, *preds).result

        return result

    def convert_matmul(self) -> OpResult:
        """Convert a MatMul node to its corresponding MLIR representation.

        Returns:
            str: textual MLIR representation corresponding to self.node
        """

        resulting_type = NodeConverter.value_to_mlir_type(self.ctx, self.node.output)
        preds = self.preds

        if self.node.output.shape == ():
            if self.node.inputs[0].is_clear:
                preds = preds[::-1]
            result = fhelinalg.Dot(resulting_type, *preds).result

        elif self.node.inputs[0].is_clear:
            result = fhelinalg.MatMulIntEintOp(resulting_type, *preds).result
        else:
            result = fhelinalg.MatMulEintIntOp(resulting_type, *preds).result

        return result

    def convert_mul(self) -> OpResult:
        """
        Convert "multiply" node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """

        resulting_type = NodeConverter.value_to_mlir_type(self.ctx, self.node.output)
        preds = self.preds

        if self.node.inputs[0].is_clear:
            preds = preds[::-1]

        if self.one_of_the_inputs_is_a_tensor:
            result = fhelinalg.MulEintIntOp(resulting_type, *preds).result
        else:
            result = fhe.MulEintIntOp(resulting_type, *preds).result

        return result

    def convert_neg(self) -> OpResult:
        """
        Convert "negative" node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """

        resulting_type = NodeConverter.value_to_mlir_type(self.ctx, self.node.output)
        pred = self.preds[0]

        if self.one_of_the_inputs_is_a_tensor:
            result = fhelinalg.NegEintOp(resulting_type, pred).result
        else:
            result = fhe.NegEintOp(resulting_type, pred).result

        return result

    def convert_reshape(self) -> OpResult:
        """
        Convert "reshape" node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """

        input_shape = self.node.inputs[0].shape
        output_shape = self.node.output.shape

        pred = self.preds[0]
        if input_shape == output_shape:
            return pred

        # we can either collapse or expand, which changes the number of dimensions
        # this is a limitation of the current compiler, it will be improved in the future (#1060)
        can_be_converted_directly = len(input_shape) != len(output_shape)

        reassociation: List[List[int]] = []
        if can_be_converted_directly:
            if len(output_shape) == 1:
                # output is 1 dimensional so collapse every dimension into the same dimension
                reassociation.append(list(range(len(input_shape))))
            else:
                # input is m dimensional
                # output is n dimensional
                # and m is different from n

                # we don't want to duplicate code, so we forget about input and output,
                # and we focus on smaller shape and bigger shape

                smaller_shape, bigger_shape = (
                    (output_shape, input_shape)
                    if len(output_shape) < len(input_shape)
                    else (input_shape, output_shape)
                )
                s_index, b_index = 0, 0

                # now we will figure out how to group the bigger shape to get the smaller shape
                # think of the algorithm below as
                #     keep merging the dimensions of the bigger shape
                #     until we have a match on the smaller shape
                #     then try to match the next dimension of the smaller shape
                #     if all dimensions of the smaller shape is matched
                #     we can convert it

                group = []
                size = 1
                while s_index < len(smaller_shape) and b_index < len(bigger_shape):
                    # dimension `b_index` of `bigger_shape` belongs to current group
                    group.append(b_index)

                    # and current group has `size * bigger_shape[b_index]` elements now
                    size *= bigger_shape[b_index]

                    # if current group size matches the dimension `s_index` of `smaller_shape`
                    if size == smaller_shape[s_index]:
                        # we finalize this group and reset everything
                        size = 1
                        reassociation.append(group)
                        group = []

                        # now try to match the next dimension of `smaller_shape`
                        s_index += 1

                    # now process the next dimension of `bigger_shape`
                    b_index += 1

                # handle the case where bigger shape has proceeding 1s
                # e.g., (5,) -> (5, 1)
                while b_index < len(bigger_shape) and bigger_shape[b_index] == 1:
                    reassociation[-1].append(b_index)
                    b_index += 1

                # if not all dimensions of both shapes are processed exactly
                if s_index != len(smaller_shape) or b_index != len(bigger_shape):
                    # we cannot convert
                    can_be_converted_directly = False

        index_type = IndexType.parse("index")
        resulting_type = NodeConverter.value_to_mlir_type(self.ctx, self.node.output)

        if can_be_converted_directly:
            reassociation_attr = ArrayAttr.get(
                [
                    ArrayAttr.get([IntegerAttr.get(index_type, dimension) for dimension in group])
                    for group in reassociation
                ]
            )
            if len(output_shape) < len(input_shape):
                return linalg.TensorCollapseShapeOp(resulting_type, pred, reassociation_attr).result
            return linalg.TensorExpandShapeOp(resulting_type, pred, reassociation_attr).result

        flattened_type = NodeConverter.value_to_mlir_type(
            self.ctx,
            Value(
                dtype=self.node.inputs[0].dtype,
                shape=(int(np.prod(input_shape)),),
                is_encrypted=self.node.inputs[0].is_encrypted,
            ),
        )
        flattened_result = linalg.TensorCollapseShapeOp(
            flattened_type,
            pred,
            ArrayAttr.get(
                [ArrayAttr.get([IntegerAttr.get(index_type, i) for i in range(len(input_shape))])]
            ),
        ).result

        return linalg.TensorExpandShapeOp(
            resulting_type,
            flattened_result,
            ArrayAttr.get(
                [ArrayAttr.get([IntegerAttr.get(index_type, i) for i in range(len(output_shape))])]
            ),
        ).result

    def convert_static_indexing(self) -> OpResult:
        """
        Convert "index.static" node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """

        input_value = self.node.inputs[0]
        input_shape = input_value.shape

        index = list(self.node.properties["attributes"]["index"])
        index_type = IndexType.parse("index")

        while len(index) < input_value.ndim:
            index.append(slice(None, None, None))

        output_type = NodeConverter.value_to_mlir_type(self.ctx, self.node.output)
        if len(index) == len(input_shape) and all(isinstance(i, (int, np.integer)) for i in index):
            indices = []
            for value, dimension_size in zip(index, input_shape):
                value = int(value)
                attr = IntegerAttr.get(index_type, value if value >= 0 else value + dimension_size)
                indices.append(arith.ConstantOp(index_type, attr).result)
            return tensor.ExtractOp(output_type, self.preds[0], indices).result

        offsets = []
        sizes = []
        strides = []

        destroyed_dimensions = []
        for dimension, (indexing_element, dimension_size) in enumerate(zip(index, input_shape)):

            if isinstance(indexing_element, slice):
                size = np.zeros(dimension_size)[indexing_element].shape[0]
                stride = indexing_element.step if isinstance(indexing_element.step, int) else 1
                offset = (
                    (
                        indexing_element.start
                        if indexing_element.start >= 0
                        else indexing_element.start + dimension_size
                    )
                    if isinstance(indexing_element.start, int)
                    else (0 if stride > 0 else dimension_size - 1)
                )

            else:
                destroyed_dimensions.append(dimension)
                size = 1
                stride = 1
                offset = int(
                    indexing_element if indexing_element >= 0 else indexing_element + dimension_size
                )

            offsets.append(offset)
            sizes.append(size)
            strides.append(stride)

        if len(destroyed_dimensions) == 0:
            return tensor.ExtractSliceOp(
                output_type,
                self.preds[0],
                [],
                [],
                [],
                ArrayAttr.get([IntegerAttr.get(index_type, value) for value in offsets]),
                ArrayAttr.get([IntegerAttr.get(index_type, value) for value in sizes]),
                ArrayAttr.get([IntegerAttr.get(index_type, value) for value in strides]),
            ).result

        output_value = self.node.output

        intermediate_shape = list(output_value.shape)
        for dimension in destroyed_dimensions:
            intermediate_shape.insert(dimension, 1)

        intermediate = tensor.ExtractSliceOp(
            RankedTensorType.get(
                intermediate_shape,
                NodeConverter.value_to_mlir_type(
                    self.ctx,
                    Value(output_value.dtype, shape=(), is_encrypted=output_value.is_encrypted),
                ),
            ),
            self.preds[0],
            [],
            [],
            [],
            ArrayAttr.get([IntegerAttr.get(index_type, value) for value in offsets]),
            ArrayAttr.get([IntegerAttr.get(index_type, value) for value in sizes]),
            ArrayAttr.get([IntegerAttr.get(index_type, value) for value in strides]),
        ).result

        reassociaton = []

        current_intermediate_dimension = 0
        for _ in range(len(output_value.shape)):
            indices = [current_intermediate_dimension]
            while current_intermediate_dimension in destroyed_dimensions:
                current_intermediate_dimension += 1
                indices.append(current_intermediate_dimension)

            reassociaton.append(indices)
            current_intermediate_dimension += 1
        while current_intermediate_dimension < len(intermediate_shape):
            reassociaton[-1].append(current_intermediate_dimension)
            current_intermediate_dimension += 1

        return linalg.TensorCollapseShapeOp(
            output_type,
            intermediate,
            ArrayAttr.get(
                [
                    ArrayAttr.get(
                        [IntegerAttr.get(index_type, index) for index in indices],
                    )
                    for indices in reassociaton
                ],
            ),
        ).result

    def convert_sub(self) -> OpResult:
        """
        Convert "subtract" node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """

        resulting_type = NodeConverter.value_to_mlir_type(self.ctx, self.node.output)
        preds = self.preds

        if self.one_of_the_inputs_is_a_tensor:
            result = fhelinalg.SubIntEintOp(resulting_type, *preds).result
        else:
            result = fhe.SubIntEintOp(resulting_type, *preds).result

        return result

    def convert_sum(self) -> OpResult:
        """
        Convert "sum" node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """

        resulting_type = NodeConverter.value_to_mlir_type(self.ctx, self.node.output)

        axes = self.node.properties["kwargs"].get("axis", [])
        keep_dims = self.node.properties["kwargs"].get("keepdims", False)

        if isinstance(axes, int):
            axes = [axes]
        elif isinstance(axes, tuple):
            axes = list(axes)

        input_dimensions = self.node.inputs[0].ndim
        for i, axis in enumerate(axes):
            if axis < 0:
                axes[i] += input_dimensions

        return fhelinalg.SumOp(
            resulting_type,
            self.preds[0],
            ArrayAttr.get([IntegerAttr.get(IntegerType.get_signless(64), axis) for axis in axes]),
            BoolAttr.get(keep_dims),
        ).result

    def convert_tlu(self) -> OpResult:
        """
        Convert Operation.Generic node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """

        variable_input_index = -1

        preds = self.graph.ordered_preds_of(self.node)
        for i, pred in enumerate(preds):
            if pred.operation != Operation.Constant:
                variable_input_index = i
                break

        assert_that(variable_input_index != -1)

        tables = construct_deduplicated_tables(self.node, preds)
        assert_that(len(tables) > 0)

        lut_shape: Tuple[int, ...] = ()
        map_shape: Tuple[int, ...] = ()

        if len(tables) == 1:
            table = tables[0][0]

            # The reduction on 63b is to avoid problems like doing a TLU of
            # the form T[j] = 2<<j, for j which is supposed to be 7b as per
            # constraint of the compiler, while in practice, it is a small
            # value. Reducing on 64b was not ok for some reason
            lut_shape = (len(table),)
            lut_values = np.array(table % (2 << 63), dtype=np.uint64)

            map_shape = ()
            map_values = None
        else:
            individual_table_size = len(tables[0][0])

            lut_shape = (len(tables), individual_table_size)
            map_shape = self.node.output.shape

            lut_values = np.zeros(lut_shape, dtype=np.uint64)
            map_values = np.zeros(map_shape, dtype=np.intp)

            for i, (table, indices) in enumerate(tables):
                assert_that(len(table) == individual_table_size)
                lut_values[i, :] = table
                for index in indices:
                    map_values[index] = i

        lut_type = RankedTensorType.get(lut_shape, IntegerType.get_signless(64, context=self.ctx))
        lut_attr = DenseElementsAttr.get(lut_values, context=self.ctx)
        lut = arith.ConstantOp(lut_type, lut_attr).result

        resulting_type = NodeConverter.value_to_mlir_type(self.ctx, self.node.output)
        pred = self.preds[variable_input_index]

        if self.one_of_the_inputs_is_a_tensor:
            if len(tables) == 1:
                result = fhelinalg.ApplyLookupTableEintOp(resulting_type, pred, lut).result
            else:
                assert_that(map_shape != ())
                assert_that(map_values is not None)

                index_type = IndexType.parse("index")
                map_type = RankedTensorType.get(map_shape, index_type)
                map_attr = DenseElementsAttr.get(map_values, context=self.ctx, type=index_type)

                result = fhelinalg.ApplyMappedLookupTableEintOp(
                    resulting_type,
                    pred,
                    lut,
                    arith.ConstantOp(map_type, map_attr).result,
                ).result
        else:
            result = fhe.ApplyLookupTableEintOp(resulting_type, pred, lut).result

        return result

    def convert_transpose(self) -> OpResult:
        """
        Convert "transpose" node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """

        resulting_type = NodeConverter.value_to_mlir_type(self.ctx, self.node.output)
        preds = self.preds

        return fhelinalg.TransposeOp(resulting_type, *preds).result
