"""
Declaration of `NodeConverter` class.
"""

# pylint: disable=no-member,no-name-in-module,too-many-lines

from copy import deepcopy
from enum import IntEnum
from typing import Dict, List, Tuple

import numpy as np
from concrete.lang.dialects import fhe, fhelinalg
from concrete.lang.dialects.fhe import EncryptedIntegerType
from mlir.dialects import arith, tensor
from mlir.ir import (
    ArrayAttr,
    Attribute,
    BlockArgument,
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

from ..dtypes import Integer, UnsignedInteger
from ..internal.utils import assert_that
from ..representation import Graph, Node, Operation
from ..values import EncryptedScalar, Value
from .utils import construct_deduplicated_tables

# pylint: enable=no-member,no-name-in-module


class Comparison(IntEnum):
    """
    Comparison enum, used when generating comparison
    between encrypted payloads.
    """

    EQUAL = 0b00
    LESSER = 0b01
    GREATER = 0b10


class NodeConverter:
    """
    NodeConverter class, to convert computation graph nodes to their MLIR equivalent.
    """

    # pylint: disable=too-many-instance-attributes

    ctx: Context
    graph: Graph
    node: Node
    preds: List[OpResult]

    all_of_the_inputs_are_encrypted: bool
    all_of_the_inputs_are_tensors: bool
    one_of_the_inputs_is_a_tensor: bool

    constant_cache: Dict[Tuple[Type, Attribute], OpResult]
    from_elements_operations: Dict[OpResult, List[OpResult]]

    # When converting bitwise / comparison operations convert
    # in chunk of [n] bits.
    # Can be configured
    bit_group_size: int = 2

    # pylint: enable=too-many-instance-attributes

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

        message = f"{value} cannot be converted to MLIR"  # pragma: no cover
        raise ValueError(message)  # pragma: no cover

    @staticmethod
    def mlir_name(result: OpResult) -> str:
        """
        Extract the MLIR variable name of an `OpResult`.

        Args:
            result (OpResult):
                op result to extract the name

        Returns:
            str:
                MLIR variable name of `result`
        """

        if isinstance(result, BlockArgument):
            return f"%arg{result.arg_number}"

        return str(result).replace("Value(", "").split("=", maxsplit=1)[0].strip()

    def __init__(
        self,
        ctx: Context,
        graph: Graph,
        node: Node,
        preds: List[OpResult],
        constant_cache: Dict[Tuple[Type, Attribute], OpResult],
        from_elements_operations: Dict[OpResult, List[OpResult]],
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

        self.constant_cache = constant_cache
        self.from_elements_operations = from_elements_operations

    def convert(self) -> OpResult:
        """
        Convert a node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """

        if self.node.operation == Operation.Constant:
            return self._convert_constant()

        assert_that(self.node.operation == Operation.Generic)

        name = self.node.properties["name"]
        converters = {
            "add": self._convert_add,
            "array": self._convert_array,
            "assign.static": self._convert_static_assignment,
            "broadcast_to": self._convert_broadcast_to,
            "concatenate": self._convert_concat,
            "conv1d": self._convert_conv1d,
            "conv2d": self._convert_conv2d,
            "conv3d": self._convert_conv3d,
            "dot": self._convert_dot,
            "expand_dims": self._convert_reshape,
            "index.static": self._convert_static_indexing,
            "matmul": self._convert_matmul,
            "maxpool": self._convert_maxpool,
            "multiply": self._convert_mul,
            "negative": self._convert_neg,
            "ones": self._convert_ones,
            "reshape": self._convert_reshape,
            "squeeze": self._convert_squeeze,
            "subtract": self._convert_sub,
            "sum": self._convert_sum,
            "transpose": self._convert_transpose,
            "zeros": self._convert_zeros,
            "bitwise_and": self._convert_bitwise_and,
            "bitwise_or": self._convert_bitwise_or,
            "bitwise_xor": self._convert_bitwise_xor,
            "less": self._convert_less,
            "less_equal": self._convert_less_equal,
            "greater": self._convert_greater,
            "greater_equal": self._convert_greater_equal,
            "equal": self._convert_equal,
            "not_equal": self._convert_not_equal,
            "left_shift": self._convert_left_shift,
            "right_shift": self._convert_right_shift,
        }

        if name in converters:
            return converters[name]()

        assert_that(self.node.converted_to_table_lookup)
        return self._convert_tlu()

    def _1d_lut(self, resulting_type, pred, lut_values) -> OpResult:
        lut_type = RankedTensorType.get(
            (len(lut_values),), IntegerType.get_signless(64, context=self.ctx)
        )
        lut_attr = Attribute.parse(f"dense<{str(lut_values)}> : {lut_type}")
        lut = self._create_constant(resulting_type, lut_attr).result
        out = None
        if self.one_of_the_inputs_is_a_tensor:
            out = fhelinalg.ApplyLookupTableEintOp(resulting_type, pred, lut).result
        else:
            out = fhe.ApplyLookupTableEintOp(resulting_type, pred, lut).result
        return out

    def _add(self, resulting_type, a, b) -> OpResult:
        out = None
        if self.one_of_the_inputs_is_a_tensor:
            out = fhelinalg.AddEintOp(resulting_type, a, b).result
        else:
            out = fhe.AddEintOp(resulting_type, a, b).result
        return out

    def _sub(self, resulting_type, a, b) -> OpResult:
        out = None
        if self.one_of_the_inputs_is_a_tensor:
            out = fhelinalg.SubEintOp(resulting_type, a, b).result
        else:
            out = fhe.SubEintOp(resulting_type, a, b).result
        return out

    def _zero(self, resulting_type) -> OpResult:
        out = None
        if self.one_of_the_inputs_is_a_tensor:
            out = fhe.ZeroTensorOp(resulting_type).result
        else:
            out = fhe.ZeroEintOp(resulting_type).result
        return out

    def _csti(self, resulting_type, value) -> OpResult:
        constant_value = Value(
            Integer(is_signed=True, bit_width=resulting_type.dtype.bit_width + 1),
            shape=resulting_type.shape,
            is_encrypted=False,
        )
        constant_type = NodeConverter.value_to_mlir_type(self.ctx, constant_value)
        out = None
        if self.node.output.is_scalar:
            out = self._create_constant(constant_type, IntegerAttr.get(constant_type, value))
        else:
            out = self._create_constant(
                constant_type, Attribute.parse(f"dense<{value}> : {constant_type}")
            )
        return out

    def _add_cst(self, resulting_type, a, b) -> OpResult:
        out = None
        if self.one_of_the_inputs_is_a_tensor:
            out = fhelinalg.AddEintIntOp(resulting_type, a, b).result
        else:
            out = fhe.AddEintIntOp(resulting_type, a, b).result
        return out

    def _generate_type_fixed_bit_width_unsigned(self, type_, bit_width):
        t = deepcopy(type_)
        t.dtype.bit_width = bit_width
        t.dtype.is_signed = False
        return NodeConverter.value_to_mlir_type(self.ctx, t)

    def _split_in_bit_groups(
        self,
        x,
        y,
        group_size,
        fn_map,
        fn_result_type,
        offset_x_by=0,
        offset_y_by=0,
    ) -> OpResult:
        """
        Split x and y into bit groups of width [group_size].

        [fn] is applied to each bit group bx and by of x and y and the result
        is appended -- and returned -- to a list.

        If [offset_x_by] (resp. [offset_y_by]) is provided then execute
        the function for [x + offset_x_by] (resp. [y + offset_y_by]) instead
        of [x] and [y]. This addition is not emitted in MLIR.
        """
        assert isinstance(self.node.output.dtype, Integer)

        bit_width = self.node.output.dtype.bit_width
        max_bit_width = 1 << bit_width
        carries = []
        for chunk_id, offset in enumerate(range(0, bit_width, group_size)):
            bit_width_this = min(group_size, bit_width - offset)
            mask = (1 << bit_width_this) - 1
            this_type = self._generate_type_fixed_bit_width_unsigned(
                self.node.output, bit_width_this * 2
            )
            to_right_shift_by = bit_width - offset - bit_width_this
            # Once rounded lookups are available in mlir use them below
            chunk_x = self._1d_lut(
                this_type,
                x,
                [
                    ((((x + offset_x_by) >> to_right_shift_by) & mask) << bit_width_this)
                    for x in range(max_bit_width)
                ],
            )
            chunk_y = self._1d_lut(
                this_type,
                y,
                [((x + offset_y_by) >> to_right_shift_by) & mask for x in range(max_bit_width)],
            )
            t = self._add(this_type, chunk_x, chunk_y)
            chunk_carry = self._1d_lut(
                fn_result_type,
                t,
                [fn_map(chunk_id, x, y) for x in range(mask + 1) for y in range(mask + 1)],
            )
            carries.append(chunk_carry)
        return carries

    def _convert_compare(self, inputs, preds, fn_result) -> OpResult:
        # Turn the two following warnings because it's a bit hard to refactor
        # the following function and would introduce functions only called
        # once.
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-statements
        if (
            not self.all_of_the_inputs_are_encrypted
            or inputs[0].dtype.bit_width != inputs[1].dtype.bit_width
        ):
            return self._convert_tlu()

        assert isinstance(self.node.output.dtype, Integer)
        resulting_type = NodeConverter.value_to_mlir_type(self.ctx, self.node.output)
        four_bit_type = self._generate_type_fixed_bit_width_unsigned(self.node.output, 4)

        # Comparison between a signed and an unsigned are tricky. To deal with them
        # We add -min of the signed number to both operands such that they
        # are both positive.
        # To avoid overflowing the unsigned operand this addition is done "virtually"
        # while constructing one of the luts.
        # A flag ("is_unsigned_greater_than_half") is emitted in MLIR to keep track
        # if the unsigned operand was greater than the max signed number as it
        # is needed to determine the result of the comparison:
        # Exemple: to compare x and y where x is an int3 and y and uint3, when y
        # is greater than 4 we are sure than x will be less than x.

        bit_width = self.node.output.dtype.bit_width
        max_bit_width = 1 << bit_width

        x = preds[0]
        y = preds[1]

        is_x_signed = inputs[0].dtype.is_signed
        is_y_signed = inputs[1].dtype.is_signed
        is_unsigned_greater_than_half = None
        signed_offset = 1 << (bit_width - 1)

        if not is_x_signed and is_y_signed:
            is_unsigned_greater_than_half = self._1d_lut(
                four_bit_type, x, [int(x >= signed_offset) << 2 for x in range(max_bit_width)]
            )

        if not is_y_signed and is_x_signed:
            is_unsigned_greater_than_half = self._1d_lut(
                four_bit_type, y, [int(y >= signed_offset) << 2 for y in range(max_bit_width)]
            )

        offset_x_by = 0
        offset_y_by = 0
        if is_x_signed or is_y_signed:
            if is_x_signed:
                x = self._add_cst(resulting_type, x, self._csti(inputs[0], signed_offset))
            else:
                offset_x_by = signed_offset
            if is_y_signed:
                y = self._add_cst(resulting_type, y, self._csti(inputs[0], signed_offset))
            else:
                offset_y_by = signed_offset

        def compare(x, y):
            out = None
            if x < y:
                out = Comparison.LESSER
            elif x > y:
                out = Comparison.GREATER
            else:
                out = Comparison.EQUAL
            return out

        def fn_map(i, x, y):
            return compare(x, y) << (min(i, 1) * 2)

        # Compare each bit group of the operands two by two.
        # 2 bits are needed for each intermediate result as we can't
        # shortcircuit in the loop as we would do in "traditional computing",
        # meaning that for each groups we need to do if they were less, greater, or
        # equal
        carries = self._split_in_bit_groups(
            x,
            y,
            self.bit_group_size,
            fn_map,
            four_bit_type,
            offset_x_by=offset_x_by,
            offset_y_by=offset_y_by,
        )
        # This is the reduction step -- we have an array where the entry i is the
        # result of comparing the chunks of x and y at position i.
        # We need to merge all of these information to deduce the final comparison output.
        # Right now we're doing that in a sort of naive way -- we could
        # increase pipeling by reducing in a tree instead than in a comb,
        # and we could decrease the number of luts (but increase there bit_width)
        # by merging several carryes at once.
        carry = self._zero(four_bit_type)
        # 0b[carry of the chunk][accumulator]
        lut_carry = [
            x if c == Comparison.EQUAL else c
            for x in [0b00, 0b01, 0b10, 0b11]
            for c in [0b00, 0b01, 0b10, 0b11]
        ]
        for i, chunk_carry in enumerate(carries):
            if i == 0:
                carry = chunk_carry
                # Here all of the special casing for the last chunk is to
                # avoid generating one other lut to cast the result to 0 or 1.
                # Note that it might be a good idea to return 0 or 0xFFFF as
                # that would allow using the result of the comparison to do masking.
                if i == len(carries) - 1:
                    carry = self._1d_lut(
                        four_bit_type, carry, [int(fn_result(i)) for i in lut_carry]
                    )
            else:
                next_carry = self._add(four_bit_type, chunk_carry, carry)
                if i == len(carries) - 1:
                    carry = self._1d_lut(
                        four_bit_type, next_carry, [int(fn_result(i)) for i in lut_carry]
                    )
                else:
                    carry = self._1d_lut(four_bit_type, next_carry, lut_carry)
        out = None
        if is_x_signed == is_y_signed:
            out = carry
        elif not is_x_signed:
            carry = self._add(four_bit_type, is_unsigned_greater_than_half, carry)
            carry = self._1d_lut(
                resulting_type, carry, [0 if (i & 0b1100) else i & 0b01 for i in range(1 << 4)]
            )
            out = carry
        elif not is_y_signed:
            carry = self._add(four_bit_type, is_unsigned_greater_than_half, carry)
            carry = self._1d_lut(
                resulting_type, carry, [1 if (i & 0b1100) else i & 0b01 for i in range(1 << 4)]
            )
            out = carry
        return out

    def _convert_for_equal_or_unequal(self, fn_map_comparison_result) -> OpResult:
        if not self.all_of_the_inputs_are_encrypted:
            return self._convert_tlu()

        assert isinstance(self.node.output.dtype, Integer)
        assert isinstance(self.node.inputs[0].dtype, Integer)
        assert isinstance(self.node.inputs[1].dtype, Integer)
        assert self.node.inputs[0].dtype.bit_width == self.node.inputs[1].dtype.bit_width

        bit_width = self.node.output.dtype.bit_width

        # Equivalent to math.ceil(bit_width / group_size)
        n_groups = -(bit_width // -self.bit_group_size)
        is_x_signed = self.node.inputs[0].dtype.is_signed
        is_y_signed = self.node.inputs[1].dtype.is_signed
        if is_x_signed != is_y_signed:
            n_groups += 1

        resulting_type = NodeConverter.value_to_mlir_type(self.ctx, self.node.output)
        n_groups_type = self._generate_type_fixed_bit_width_unsigned(self.node.output, n_groups)

        x = self.preds[0]
        y = self.preds[1]

        # Equal and unequal between signed ints and unsigned ints are tricky.
        # We reuse the same trick as for comparison to handle it.
        offset_x_by = 0
        offset_y_by = 0

        is_unsigned_greater_than_half = None
        signed_offset = 1 << (bit_width - 1)
        cst = self._csti(self.node.inputs[0], signed_offset)

        if not is_x_signed and is_y_signed:
            is_unsigned_greater_than_half = self._1d_lut(
                n_groups_type,
                x,
                [int(x >= signed_offset) << (n_groups - 1) for x in range(1 << bit_width)],
            )

        if not is_y_signed and is_x_signed:
            is_unsigned_greater_than_half = self._1d_lut(
                n_groups_type,
                y,
                [int(y >= signed_offset) << (n_groups - 1) for y in range(1 << bit_width)],
            )
        if is_x_signed or is_y_signed:
            if is_x_signed:
                x = self._add_cst(resulting_type, x, cst)
            else:
                offset_x_by = signed_offset
            if is_y_signed:
                y = self._add_cst(resulting_type, y, cst)
            else:
                offset_y_by = signed_offset

        # [carries] will be an integer with at one bit per bit group.
        # If the ith bit is 1 then it means that the ith bit group of x and y
        # are different.
        carries = self._split_in_bit_groups(
            x,
            y,
            self.bit_group_size,
            lambda i, x, y: int(x != y) << i,
            n_groups_type,
            offset_x_by=offset_x_by,
            offset_y_by=offset_y_by,
        )

        carry = self._zero(n_groups_type)
        if is_unsigned_greater_than_half:
            carry = is_unsigned_greater_than_half
        for chunk_carry in carries:
            carry = self._add(n_groups_type, carry, chunk_carry)
        return self._1d_lut(
            resulting_type,
            carry,
            [int(fn_map_comparison_result(i == 0)) for i in range(1 << n_groups)],
        )

    def _convert_less(self) -> OpResult:
        """
        Convert "less" node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """
        return self._convert_compare(self.node.inputs, self.preds, lambda x: x == Comparison.LESSER)

    def _convert_less_equal(self) -> OpResult:
        """
        Convert "less equal" node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """
        return self._convert_compare(
            self.node.inputs, self.preds, lambda x: x in [Comparison.LESSER, Comparison.EQUAL]
        )

    def _convert_greater(self) -> OpResult:
        """
        Convert "greater" node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """
        return self._convert_compare(
            self.node.inputs[::-1], self.preds[::-1], lambda x: x == Comparison.LESSER
        )

    def _convert_greater_equal(self) -> OpResult:
        """
        Convert "greater equal" node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """
        return self._convert_compare(
            self.node.inputs[::-1],
            self.preds[::-1],
            lambda x: x in [Comparison.LESSER, Comparison.EQUAL],
        )

    def _convert_equal(self) -> OpResult:
        """
        Convert "equal" node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """
        return self._convert_for_equal_or_unequal(lambda x: x)

    def _convert_not_equal(self) -> OpResult:
        """
        Convert "not equal" node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """
        return self._convert_for_equal_or_unequal(lambda x: not x)

    def _convert_bitwise_op(self, bitwise_op) -> OpResult:
        """
        Convert a bitwise operation corresponding to `bitwise_op`

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """
        if not self.all_of_the_inputs_are_encrypted:
            return self._convert_tlu()

        assert isinstance(self.node.output.dtype, Integer)
        resulting_type = NodeConverter.value_to_mlir_type(self.ctx, self.node.output)
        bit_width = self.node.output.dtype.bit_width

        x = self.preds[0]
        y = self.preds[1]

        z = self._zero(resulting_type)
        max_bit_width = 1 << bit_width
        # This algorithm works as follow:
        # Each input x and y can be written as (each letter is one bit):
        # X = aabbccdd
        # Y = eeffgghh
        # Computing X op Y can be done as follow (assuming that op doesn't
        # overflow, which is the case for bitwise operations)
        # X op y = (aa op ee) (bb op ff) (cc op gg) (dd op hh)
        # This mean that we can reduce the case of a bitwise operation on
        # large integers to the case of bitwise operation on smaller integers.
        # Know to compute aa op ee we can use a lut if we use the following trick:
        # There's a bijection between (aa, ee) and (aa << 2 + ee), so we can precompute
        # all aa op ee and put them in a lut indexed by (aa << 2 + ee)
        for offset in range(0, bit_width, self.bit_group_size):
            bit_width_this = self.bit_group_size

            bitwise_lut = [
                bitwise_op(x, y) << offset
                for x in range(1 << bit_width_this)
                for y in range(1 << bit_width_this)
            ]
            twice_this_type = self._generate_type_fixed_bit_width_unsigned(
                self.node.output, bit_width_this * 2
            )
            mask = (1 << bit_width_this) - 1
            # Could be optimized using rounded table lookups -- here we don't care about most of the
            # bits.
            x_l = self._1d_lut(
                twice_this_type,
                x,
                [((x >> offset) & mask) << bit_width_this for x in range(max_bit_width)],
            )
            y_l = self._1d_lut(
                twice_this_type, y, [(y >> offset) & mask for y in range(max_bit_width)]
            )
            idx = self._add(twice_this_type, x_l, y_l)
            z_chunk = self._1d_lut(resulting_type, idx, bitwise_lut)
            z = self._add(resulting_type, z, z_chunk)
        return z

    def _convert_bitwise_and(self) -> OpResult:
        """
        Convert "bitwise_and" node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """
        return self._convert_bitwise_op(lambda x, y: x & y)

    def _convert_bitwise_or(self) -> OpResult:
        """
        Convert "bitwise_or" node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """
        return self._convert_bitwise_op(lambda x, y: x | y)

    def _convert_bitwise_xor(self) -> OpResult:
        """
        Convert "bitwise_xor" node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """
        return self._convert_bitwise_op(lambda x, y: x ^ y)

    def _convert_shift(self, left_shift=False) -> OpResult:
        if not self.all_of_the_inputs_are_encrypted:
            return self._convert_tlu()

        assert isinstance(self.node.output.dtype, Integer)
        resulting_type = NodeConverter.value_to_mlir_type(self.ctx, self.node.output)
        bit_width = self.node.output.dtype.bit_width
        group_size = min(7, bit_width)
        max_bit_width = 1 << bit_width
        x = self.preds[0]
        b = self.preds[1]

        # Left_shifts of x << b can be done as follow:
        # - left shift of x by 8 if b & 0b1000 > 0
        # - left shift of x by 4 if b & 0b0100 > 0
        # - left shift of x by 2 if b & 0b0010 > 0
        # - left shift of x by 1 if b & 0b0001 > 0
        # Encoding this condition is non trivial -- however,
        # it can be done using the following trick:
        # y = (b & 0b1000 > 0) * ((x << 8) - x) + x
        # When b & 0b1000, then:
        #   y = 1 * ((x << 8) - x) + x = (x << 8) - x + x = x << 8
        # When b & 0b1000 == 0 then:
        #   y = 0 * ((x << 8) - x) + x = x
        # Note that (x << 8) - x will never overflow.
        #
        # The same trick can be used for right shift but with:
        # y = x - (b & 0b1000 > 0) * (x - (x >> 8))

        def generate_for(x, cst):
            this_type = self._generate_type_fixed_bit_width_unsigned(
                self.node.output, group_size + 1
            )
            should_shift = self._1d_lut(
                this_type,
                b,
                [int((b & cst) > 0) for b in range(1 << self.node.inputs[1].dtype.bit_width)],
            )
            shifted_x = None

            if left_shift:
                shifted_x = self._1d_lut(
                    resulting_type, x, [x << cst for x in range(1 << bit_width)]
                )
                shifted_x = self._sub(resulting_type, shifted_x, x)
            else:
                shifted_x = self._1d_lut(
                    resulting_type, x, [x >> cst for x in range(1 << bit_width)]
                )
                shifted_x = self._sub(resulting_type, x, shifted_x)

            chunks = []
            for offset in range(0, bit_width, group_size):
                bit_width_this = min(group_size, bit_width - offset)
                mask = (1 << bit_width_this) - 1
                to_right_shift_by = bit_width - offset - bit_width_this
                # Once rounded lookups are available in mlir use them below
                chunk_x = self._1d_lut(
                    this_type,
                    shifted_x,
                    [(((x >> to_right_shift_by) & mask) << 1) for x in range(max_bit_width)],
                )
                chunk_x = self._add(this_type, chunk_x, should_shift)

                chunk = self._1d_lut(
                    resulting_type,
                    chunk_x,
                    [
                        x << to_right_shift_by if b else 0
                        for x in range(1 << group_size)
                        for b in [0, 1]
                    ],
                )
                chunks.append((chunk))
            shifted_masked_x = chunks[0]
            for chunk in chunks[1:]:
                shifted_masked_x = self._add(resulting_type, shifted_masked_x, chunk)

            out = None
            if left_shift:
                out = self._add(resulting_type, shifted_masked_x, x)
            else:
                out = self._sub(resulting_type, x, shifted_masked_x)
            return out

        x = generate_for(x, 8)
        x = generate_for(x, 4)
        x = generate_for(x, 2)
        return generate_for(x, 1)

    def _convert_left_shift(self) -> OpResult:
        """
        Convert "left_shift" node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """
        return self._convert_shift(left_shift=True)

    def _convert_right_shift(self) -> OpResult:
        """
        Convert "right_shift" node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """
        return self._convert_shift(left_shift=False)

    # pylint: disable=no-self-use
    def _convert_add(self) -> OpResult:
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

    def _convert_broadcast_to(self) -> OpResult:
        """
        Convert "broadcast_to" node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """

        resulting_type = NodeConverter.value_to_mlir_type(self.ctx, self.node.output)

        zeros = fhe.ZeroTensorOp(resulting_type).result
        if self.node.inputs[0].is_encrypted:
            result = fhelinalg.AddEintOp(resulting_type, zeros, self.preds[0]).result
        else:
            result = fhelinalg.AddEintIntOp(resulting_type, zeros, self.preds[0]).result

        # TODO: convert this to a single operation once it can be done
        # (https://github.com/zama-ai/concrete-numpy-internal/issues/1610)

        return result

    def _convert_array(self) -> OpResult:
        """
        Convert "array" node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """

        resulting_type = NodeConverter.value_to_mlir_type(self.ctx, self.node.output)
        preds = self.preds

        processed_preds = []
        for pred, value in zip(preds, self.node.inputs):
            if value.is_encrypted or self.node.output.is_clear:
                processed_preds.append(pred)
                continue

            assert isinstance(value.dtype, Integer)

            zero_value = EncryptedScalar(UnsignedInteger(value.dtype.bit_width - 1))
            zero_type = NodeConverter.value_to_mlir_type(self.ctx, zero_value)
            zero = fhe.ZeroEintOp(zero_type).result

            encrypted_pred = fhe.AddEintIntOp(zero_type, zero, pred).result
            processed_preds.append(encrypted_pred)

        # `placeholder_result` will be replaced textually by `actual_value` below in graph converter
        # `tensor.from_elements` cannot be created from python bindings
        # that's why we use placeholder values and text manipulation

        if self.node.output.is_clear:
            attribute = Attribute.parse(f"dense<0> : {resulting_type}")
            # pylint: disable=too-many-function-args
            placeholder_result = arith.ConstantOp(resulting_type, attribute).result
            # pylint: enable=too-many-function-args
        else:
            placeholder_result = fhe.ZeroTensorOp(resulting_type).result

        self.from_elements_operations[placeholder_result] = processed_preds
        return placeholder_result

    def _convert_concat(self) -> OpResult:
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
                axis=IntegerAttr.get(IntegerType.get_signless(64), axis),
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
            flattened_pred = tensor.CollapseShapeOp(
                flattened_pred_type,
                pred,
                ArrayAttr.get(
                    [
                        ArrayAttr.get(
                            [
                                IntegerAttr.get(IntegerType.get_signless(64), i)
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
            axis=IntegerAttr.get(IntegerType.get_signless(64), 0),
        ).result

    def _convert_constant(self) -> OpResult:
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

        return self._create_constant(resulting_type, attr).result

    def _convert_conv1d(self) -> OpResult:
        """
        Convert "conv1d" node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """

        message = "conv1d conversion to MLIR is not yet implemented"
        raise NotImplementedError(message)

    def _convert_conv2d(self) -> OpResult:
        """
        Convert "conv2d" node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """

        resulting_type = NodeConverter.value_to_mlir_type(self.ctx, self.node.output)

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
        group = IntegerAttr.get(
            IntegerType.get_signless(64), self.node.properties["kwargs"]["group"]
        )

        has_bias = len(self.node.inputs) == 3
        if has_bias:
            bias = self.preds[2]
        else:
            bias = None
        # input and weight
        preds = self.preds[:2]
        return fhelinalg.Conv2dOp(
            resulting_type,
            *preds,
            bias=bias,
            padding=pads,
            strides=strides,
            dilations=dilations,
            group=group,
        ).result

    def _convert_conv3d(self) -> OpResult:
        """
        Convert "conv3d" node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """

        message = "conv3d conversion to MLIR is not yet implemented"
        raise NotImplementedError(message)

    def _convert_dot(self) -> OpResult:
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

    def _convert_matmul(self) -> OpResult:
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

    def _convert_maxpool(self) -> OpResult:
        """
        Convert "maxpool" node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """

        message = "MaxPool operation cannot be compiled yet"
        raise NotImplementedError(message)

    def _convert_mul(self) -> OpResult:
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

    def _convert_neg(self) -> OpResult:
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

    def _convert_ones(self) -> OpResult:
        """
        Convert "ones" node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """

        resulting_type = NodeConverter.value_to_mlir_type(self.ctx, self.node.output)

        assert isinstance(self.node.output.dtype, Integer)
        bit_width = self.node.output.dtype.bit_width

        if self.node.output.is_scalar:
            constant_value = Value(
                Integer(is_signed=False, bit_width=bit_width + 1),
                shape=(),
                is_encrypted=False,
            )
            constant_type = NodeConverter.value_to_mlir_type(self.ctx, constant_value)
            constant_attr = IntegerAttr.get(constant_type, 1)

            zero = fhe.ZeroEintOp(resulting_type).result
            one = self._create_constant(constant_type, constant_attr).result

            result = fhe.AddEintIntOp(resulting_type, zero, one).result
        else:
            constant_value = Value(
                Integer(is_signed=False, bit_width=bit_width + 1),
                shape=(1,),
                is_encrypted=False,
            )
            constant_type = NodeConverter.value_to_mlir_type(self.ctx, constant_value)
            constant_attr = Attribute.parse(f"dense<[1]> : {constant_type}")

            zeros = fhe.ZeroTensorOp(resulting_type).result
            ones = self._create_constant(constant_type, constant_attr).result

            result = fhelinalg.AddEintIntOp(resulting_type, zeros, ones).result

        return result

    def _convert_reshape(self) -> OpResult:
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

        i64_type = IntegerType.get_signless(64)
        resulting_type = NodeConverter.value_to_mlir_type(self.ctx, self.node.output)

        if can_be_converted_directly:
            reassociation_attr = ArrayAttr.get(
                [
                    ArrayAttr.get([IntegerAttr.get(i64_type, dimension) for dimension in group])
                    for group in reassociation
                ]
            )
            if len(output_shape) < len(input_shape):
                return tensor.CollapseShapeOp(resulting_type, pred, reassociation_attr).result
            return tensor.ExpandShapeOp(resulting_type, pred, reassociation_attr).result

        flattened_type = NodeConverter.value_to_mlir_type(
            self.ctx,
            Value(
                dtype=self.node.inputs[0].dtype,
                shape=(int(np.prod(input_shape)),),
                is_encrypted=self.node.inputs[0].is_encrypted,
            ),
        )
        flattened_result = tensor.CollapseShapeOp(
            flattened_type,
            pred,
            ArrayAttr.get(
                [ArrayAttr.get([IntegerAttr.get(i64_type, i) for i in range(len(input_shape))])]
            ),
        ).result

        return tensor.ExpandShapeOp(
            resulting_type,
            flattened_result,
            ArrayAttr.get(
                [ArrayAttr.get([IntegerAttr.get(i64_type, i) for i in range(len(output_shape))])]
            ),
        ).result

    def _convert_static_assignment(self) -> OpResult:
        """
        Convert "assign.static" node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """

        input_value = self.node.inputs[0]
        input_shape = input_value.shape

        index = list(self.node.properties["kwargs"]["index"])

        while len(index) < input_value.ndim:
            index.append(slice(None, None, None))

        output_type = NodeConverter.value_to_mlir_type(self.ctx, self.node.output)

        offsets = []
        sizes = []
        strides = []

        for indexing_element, dimension_size in zip(index, input_shape):

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
                size = 1
                stride = 1
                offset = int(
                    indexing_element if indexing_element >= 0 else indexing_element + dimension_size
                )

            offsets.append(offset)
            sizes.append(size)
            strides.append(stride)

        i64_type = IntegerType.get_signless(64)
        return tensor.InsertSliceOp(
            output_type,
            self.preds[1],
            self.preds[0],
            [],
            [],
            [],
            ArrayAttr.get([IntegerAttr.get(i64_type, value) for value in offsets]),
            ArrayAttr.get([IntegerAttr.get(i64_type, value) for value in sizes]),
            ArrayAttr.get([IntegerAttr.get(i64_type, value) for value in strides]),
        ).result

    def _convert_static_indexing(self) -> OpResult:
        """
        Convert "index.static" node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """

        input_value = self.node.inputs[0]
        input_shape = input_value.shape

        index = list(self.node.properties["kwargs"]["index"])

        while len(index) < input_value.ndim:
            index.append(slice(None, None, None))

        output_type = NodeConverter.value_to_mlir_type(self.ctx, self.node.output)
        if len(index) == len(input_shape) and all(isinstance(i, (int, np.integer)) for i in index):
            indices = []
            for value, dimension_size in zip(index, input_shape):
                value = int(value)
                attr = IntegerAttr.get(
                    IndexType.parse("index"), value if value >= 0 else value + dimension_size
                )
                indices.append(self._create_constant(IndexType.parse("index"), attr).result)
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

        i64_type = IntegerType.get_signless(64)
        if len(destroyed_dimensions) == 0:
            return tensor.ExtractSliceOp(
                output_type,
                self.preds[0],
                [],
                [],
                [],
                ArrayAttr.get([IntegerAttr.get(i64_type, value) for value in offsets]),
                ArrayAttr.get([IntegerAttr.get(i64_type, value) for value in sizes]),
                ArrayAttr.get([IntegerAttr.get(i64_type, value) for value in strides]),
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
            ArrayAttr.get([IntegerAttr.get(i64_type, value) for value in offsets]),
            ArrayAttr.get([IntegerAttr.get(i64_type, value) for value in sizes]),
            ArrayAttr.get([IntegerAttr.get(i64_type, value) for value in strides]),
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

        return tensor.CollapseShapeOp(
            output_type,
            intermediate,
            ArrayAttr.get(
                [
                    ArrayAttr.get(
                        [IntegerAttr.get(i64_type, index) for index in indices],
                    )
                    for indices in reassociaton
                ],
            ),
        ).result

    def _convert_squeeze(self) -> OpResult:
        """
        Convert "squeeze" node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """

        # because of the tracing logic, we have the correct output shape

        # if the output shape is (), it means (1, 1, ..., 1, 1) is squeezed
        # and the result is a scalar, so we need to do indexing, not reshape
        if self.node.output.shape == ():
            assert all(size == 1 for size in self.node.inputs[0].shape)
            self.node.properties["kwargs"]["index"] = (0,) * self.node.inputs[0].ndim
            return self._convert_static_indexing()

        # otherwise, a simple reshape would work as we already have the correct shape
        return self._convert_reshape()

    def _convert_sub(self) -> OpResult:
        """
        Convert "subtract" node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """

        resulting_type = NodeConverter.value_to_mlir_type(self.ctx, self.node.output)
        preds = self.preds

        if self.all_of_the_inputs_are_encrypted:
            if self.one_of_the_inputs_is_a_tensor:
                result = fhelinalg.SubEintOp(resulting_type, *preds).result
            else:
                result = fhe.SubEintOp(resulting_type, *preds).result

        elif self.node.inputs[0].is_clear:
            if self.one_of_the_inputs_is_a_tensor:
                result = fhelinalg.SubIntEintOp(resulting_type, *preds).result
            else:
                result = fhe.SubIntEintOp(resulting_type, *preds).result

        else:
            if self.one_of_the_inputs_is_a_tensor:
                result = fhelinalg.SubEintIntOp(resulting_type, *preds).result
            else:
                result = fhe.SubEintIntOp(resulting_type, *preds).result

        return result

    def _convert_sum(self) -> OpResult:
        """
        Convert "sum" node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """

        resulting_type = NodeConverter.value_to_mlir_type(self.ctx, self.node.output)

        axes = self.node.properties["kwargs"].get("axis", [])
        keep_dims = self.node.properties["kwargs"].get("keepdims", False)

        if axes is None:
            axes = []
        elif isinstance(axes, int):
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
            axes=ArrayAttr.get(
                [IntegerAttr.get(IntegerType.get_signless(64), axis) for axis in axes]
            ),
            keep_dims=BoolAttr.get(keep_dims),
        ).result

    def _convert_tlu(self) -> OpResult:
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
        lut = self._create_constant(lut_type, lut_attr).result

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
                    self._create_constant(map_type, map_attr).result,
                ).result
        else:
            result = fhe.ApplyLookupTableEintOp(resulting_type, pred, lut).result

        return result

    def _convert_transpose(self) -> OpResult:
        """
        Convert "transpose" node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """

        resulting_type = NodeConverter.value_to_mlir_type(self.ctx, self.node.output)
        pred = self.preds[0]

        axes = self.node.properties["kwargs"].get("axes", [])

        return fhelinalg.TransposeOp(
            resulting_type,
            pred,
            axes=ArrayAttr.get(
                [IntegerAttr.get(IntegerType.get_signless(64), axis) for axis in axes]
            ),
        ).result

    def _convert_zeros(self) -> OpResult:
        """
        Convert "zeros" node to its corresponding MLIR representation.

        Returns:
            OpResult:
                in-memory MLIR representation corresponding to `self.node`
        """

        resulting_type = NodeConverter.value_to_mlir_type(self.ctx, self.node.output)

        if self.node.output.is_scalar:
            result = fhe.ZeroEintOp(resulting_type).result
        else:
            result = fhe.ZeroTensorOp(resulting_type).result

        return result

    def _create_constant(self, mlir_type: Type, mlir_attribute: Attribute):
        result = self.constant_cache.get((mlir_type, mlir_attribute))
        if result is None:
            # ConstantOp is being decorated, and the init function is supposed to take more
            # arguments than those pylint is considering
            # pylint: disable=too-many-function-args
            result = arith.ConstantOp(mlir_type, mlir_attribute)
            # pylint: enable=too-many-function-args
            self.constant_cache[(mlir_type, mlir_attribute)] = result
        return result

    # pylint: enable=no-self-use
