"""Test file for bounds evaluation with a inputset"""

from typing import Tuple

import numpy as np
import pytest

from concrete.common.bounds_measurement.inputset_eval import eval_op_graph_bounds_on_inputset
from concrete.common.compilation import CompilationConfiguration
from concrete.common.data_types.floats import Float
from concrete.common.data_types.integers import Integer, UnsignedInteger
from concrete.common.values import ClearTensor, EncryptedScalar, EncryptedTensor
from concrete.numpy.compile import numpy_max_func, numpy_min_func
from concrete.numpy.np_dtypes_helpers import get_base_value_for_numpy_or_python_constant_data
from concrete.numpy.tracing import trace_numpy_function


@pytest.mark.parametrize(
    "function,input_ranges,expected_output_bounds,expected_output_data_type",
    [
        pytest.param(
            lambda x, y: x + y,
            ((-10, 10), (-10, 10)),
            (-20, 20),
            Integer(6, is_signed=True),
            id="x + y, (-10, 10), (-10, 10), (-20, 20)",
        ),
        pytest.param(
            lambda x, y: x + y,
            ((-10, 2), (-4, 5)),
            (-14, 7),
            Integer(5, is_signed=True),
            id="x + y, (-10, 2), (-4, 5), (-14, 7)",
        ),
        pytest.param(
            lambda x, y: x + y + 1.7,
            ((-10, 2), (-4, 5)),
            (-12.3, 8.7),
            Float(64),
            id="x + y + 1.7, (-10, 2), (-4, 5), (-12.3, 8.7)",
        ),
        pytest.param(
            lambda x, y: x + y + 1,
            ((-10, 2), (-4, 5)),
            (-13, 8),
            Integer(5, is_signed=True),
            id="x + y + 1, (-10, 2), (-4, 5), (-13, 8)",
        ),
        pytest.param(
            lambda x, y: x + y + (-3),
            ((-10, 2), (-4, 5)),
            (-17, 4),
            Integer(6, is_signed=True),
            id="x + y + 1, (-10, 2), (-4, 5), (-17, 4)",
        ),
        pytest.param(
            lambda x, y: (1 + x) + y,
            ((-10, 2), (-4, 5)),
            (-13, 8),
            Integer(5, is_signed=True),
            id="(1 + x) + y, (-10, 2), (-4, 5), (-13, 8)",
        ),
        pytest.param(
            lambda x, y: x - y,
            ((-10, 10), (-10, 10)),
            (-20, 20),
            Integer(6, is_signed=True),
            id="x - y, (-10, 10), (-10, 10), (-20, 20)",
        ),
        pytest.param(
            lambda x, y: x - y,
            ((-10, 2), (-4, 5)),
            (-15, 6),
            Integer(5, is_signed=True),
            id="x - y, (-10, 2), (-4, 5), (-15, 6)",
        ),
        pytest.param(
            lambda x, y: x - y - 42,
            ((-10, 2), (-4, 5)),
            (-57, -36),
            Integer(7, is_signed=True),
            id="x - y - 42, (-10, 2), (-4, 5), (-57, -36)",
        ),
        pytest.param(
            lambda x, y: x - y - 41.5,
            ((-10, 2), (-4, 5)),
            (-56.5, -35.5),
            Float(64),
            id="x - y - 41.5, (-10, 2), (-4, 5), (-56.5, -35.5)",
        ),
        pytest.param(
            lambda x, y: 3 - x + y,
            ((-10, 2), (-4, 5)),
            (-3, 18),
            Integer(6, is_signed=True),
            id="3 - x + y, (-10, 2), (-4, 5), (-3, 18)",
        ),
        pytest.param(
            lambda x, y: 2.8 - x + y,
            ((-10, 2), (-4, 5)),
            (-3.2, 17.8),
            Float(64),
            id="2.8 - x + y, (-10, 2), (-4, 5), (-3.2, 17.8)",
        ),
        pytest.param(
            lambda x, y: (-13) - x + y,
            ((-10, 2), (-4, 5)),
            (-19, 2),
            Integer(6, is_signed=True),
            id="(-13) - x + y, (-10, 2), (-4, 5), (-19, 2)",
        ),
        pytest.param(
            lambda x, y: (-13.5) - x + y,
            ((-10, 2), (-4, 5)),
            (-19.5, 1.5),
            Float(64),
            id="(-13.5) - x + y, (-10, 2), (-4, 5), (-19.5, 1.5)",
        ),
        pytest.param(
            lambda x, y: x * y,
            ((-10, 10), (-10, 10)),
            (-100, 100),
            Integer(8, is_signed=True),
            id="x * y, (-10, 10), (-10, 10), (-100, 100)",
        ),
        pytest.param(
            lambda x, y: x * y,
            ((-10, 2), (-4, 5)),
            (-50, 40),
            Integer(7, is_signed=True),
            id="x * y, (-10, 2), (-4, 5), (-50, 40)",
        ),
        pytest.param(
            lambda x, y: (3 * x) * y,
            ((-10, 2), (-4, 5)),
            (-150, 120),
            Integer(9, is_signed=True),
            id="(3 * x) * y, (-10, 2), (-4, 5), (-150, 120)",
        ),
        pytest.param(
            lambda x, y: (3.0 * x) * y,
            ((-10, 2), (-4, 5)),
            (-150.0, 120.0),
            Float(64),
            id="(3.0 * x) * y, (-10, 2), (-4, 5), (-150.0, 120.0)",
        ),
        pytest.param(
            lambda x, y: (x * 11) * y,
            ((-10, 2), (-4, 5)),
            (-550, 440),
            Integer(11, is_signed=True),
            id="x * y, (-10, 2), (-4, 5), (-550, 440)",
        ),
        pytest.param(
            lambda x, y: (x * (-11)) * y,
            ((-10, 2), (-4, 5)),
            (-440, 550),
            Integer(11, is_signed=True),
            id="(x * (-11)) * y, (-10, 2), (-4, 5), (-440, 550)",
        ),
        pytest.param(
            lambda x, y: (x * (-11.0)) * y,
            ((-10, 2), (-4, 5)),
            (-440.0, 550.0),
            Float(64),
            id="(x * (-11.0)) * y, (-10, 2), (-4, 5), (-440.0, 550.0)",
        ),
        pytest.param(
            lambda x, y: x + x + y,
            ((-10, 10), (-10, 10)),
            (-30, 30),
            Integer(6, is_signed=True),
            id="x + x + y, (-10, 10), (-10, 10), (-30, 30)",
        ),
        pytest.param(
            lambda x, y: x - x + y,
            ((-10, 10), (-10, 10)),
            (-10, 10),
            Integer(5, is_signed=True),
            id="x - x + y, (-10, 10), (-10, 10), (-10, 10)",
        ),
        pytest.param(
            lambda x, y: x - x + y,
            ((-10, 2), (-4, 5)),
            (-4, 5),
            Integer(4, is_signed=True),
            id="x - x + y, (-10, 2), (-4, 5), (-4, 5)",
        ),
        pytest.param(
            lambda x, y: x * y - x,
            ((-10, 10), (-10, 10)),
            (-110, 110),
            Integer(8, is_signed=True),
            id="x * y - x, (-10, 10), (-10, 10), (-110, 110)",
        ),
        pytest.param(
            lambda x, y: x * y - x,
            ((-10, 2), (-4, 5)),
            (-40, 50),
            Integer(7, is_signed=True),
            id="x * y - x, (-10, 2), (-4, 5), (-40, 50),",
        ),
        pytest.param(
            lambda x, y: (x * 3) * y - (x + 3) + (y - 13) + x * (11 + y) * (12 + y) + (15 - x),
            ((-10, 2), (-4, 5)),
            (-2846, 574),
            Integer(13, is_signed=True),
            id="x * y - x, (-10, 2), (-4, 5), (-2846, 574),",
        ),
    ],
)
def test_eval_op_graph_bounds_on_inputset(
    function,
    input_ranges,
    expected_output_bounds,
    expected_output_data_type: Integer,
):
    """Test function for eval_op_graph_bounds_on_inputset"""

    test_eval_op_graph_bounds_on_inputset_multiple_output(
        function,
        input_ranges,
        (expected_output_bounds,),
        (expected_output_data_type,),
    )


@pytest.mark.parametrize(
    "function,input_ranges,expected_output_bounds,expected_output_data_type",
    [
        pytest.param(
            lambda x, y: (x + 1, y + 10),
            ((-1, 1), (3, 4)),
            ((0, 2), (13, 14)),
            (Integer(2, is_signed=False), Integer(4, is_signed=False)),
        ),
        pytest.param(
            lambda x, y: (x + 1.5, y + 9.6),
            ((-1, 1), (3, 4)),
            ((0.5, 2.5), (12.6, 13.6)),
            (Float(64), Float(64)),
        ),
        pytest.param(
            lambda x, y: (x + y + 1, x * y + 42),
            ((-1, 1), (3, 4)),
            ((3, 6), (38, 46)),
            (Integer(3, is_signed=False), Integer(6, is_signed=False)),
        ),
        pytest.param(
            lambda x, y: (x + y + 0.4, x * y + 41.7),
            ((-1, 1), (3, 4)),
            ((2.4, 5.4), (37.7, 45.7)),
            (Float(64), Float(64)),
        ),
        pytest.param(
            lambda x, y: (x + y + 1, x * y + 41.7),
            ((-1, 1), (3, 4)),
            ((3, 6), (37.7, 45.7)),
            (Integer(3, is_signed=False), Float(64)),
        ),
        pytest.param(
            lambda x, y: (x + y + 0.4, x * y + 42),
            ((-1, 1), (3, 4)),
            ((2.4, 5.4), (38, 46)),
            (Float(64), Integer(6, is_signed=False)),
        ),
    ],
)
def test_eval_op_graph_bounds_on_inputset_multiple_output(
    function,
    input_ranges,
    expected_output_bounds,
    expected_output_data_type: Tuple[Integer],
):
    """Test function for eval_op_graph_bounds_on_inputset"""

    op_graph = trace_numpy_function(
        function, {"x": EncryptedScalar(Integer(64, True)), "y": EncryptedScalar(Integer(64, True))}
    )

    def data_gen(range_x, range_y):
        for x_gen in range_x:
            for y_gen in range_y:
                yield (x_gen, y_gen)

    _, node_bounds_and_samples = eval_op_graph_bounds_on_inputset(
        op_graph,
        data_gen(*tuple(range(x[0], x[1] + 1) for x in input_ranges)),
        CompilationConfiguration(),
    )

    for i, output_node in op_graph.output_nodes.items():
        output_node_bounds = node_bounds_and_samples[output_node]
        assert (output_node_bounds["min"], output_node_bounds["max"]) == expected_output_bounds[i]

    op_graph.update_values_with_bounds_and_samples(node_bounds_and_samples)

    for i, output_node in op_graph.output_nodes.items():
        assert expected_output_data_type[i] == output_node.outputs[0].dtype


def test_eval_op_graph_bounds_on_non_conformant_inputset_default(capsys):
    """Test function for eval_op_graph_bounds_on_inputset with non conformant inputset"""

    def f(x, y):
        return np.dot(x, y)

    x = EncryptedTensor(UnsignedInteger(2), (3,))
    y = ClearTensor(UnsignedInteger(2), (3,))

    inputset = [
        (np.array([2, 1, 3, 1]), np.array([1, 2, 1, 1])),
        (np.array([3, 3, 3]), np.array([3, 3, 5])),
    ]

    op_graph = trace_numpy_function(f, {"x": x, "y": y})

    configuration = CompilationConfiguration()
    eval_op_graph_bounds_on_inputset(
        op_graph,
        inputset,
        compilation_configuration=configuration,
        min_func=numpy_min_func,
        max_func=numpy_max_func,
        get_base_value_for_constant_data_func=get_base_value_for_numpy_or_python_constant_data,
    )

    captured = capsys.readouterr()
    assert (
        captured.err == "Warning: Input #0 (0-indexed) is not coherent with the hinted parameters "
        "(expected EncryptedTensor<uint2, shape=(3,)> for parameter `x` "
        "but got EncryptedTensor<uint2, shape=(4,)> which is not compatible)\n"
        "Warning: Input #0 (0-indexed) is not coherent with the hinted parameters "
        "(expected ClearTensor<uint2, shape=(3,)> for parameter `y` "
        "but got ClearTensor<uint2, shape=(4,)> which is not compatible)\n"
    )


def test_eval_op_graph_bounds_on_non_conformant_inputset_check_all(capsys):
    """Test function for eval_op_graph_bounds_on_inputset with non conformant inputset, check all"""

    def f(x, y):
        return np.dot(x, y)

    x = EncryptedTensor(UnsignedInteger(2), (3,))
    y = ClearTensor(UnsignedInteger(2), (3,))

    inputset = [
        (np.array([2, 1, 3, 1]), np.array([1, 2, 1, 1])),
        (np.array([3, 3, 3]), np.array([3, 3, 5])),
    ]

    op_graph = trace_numpy_function(f, {"x": x, "y": y})

    configuration = CompilationConfiguration(check_every_input_in_inputset=True)
    eval_op_graph_bounds_on_inputset(
        op_graph,
        inputset,
        compilation_configuration=configuration,
        min_func=numpy_min_func,
        max_func=numpy_max_func,
        get_base_value_for_constant_data_func=get_base_value_for_numpy_or_python_constant_data,
    )

    captured = capsys.readouterr()
    assert (
        captured.err == "Warning: Input #0 (0-indexed) is not coherent with the hinted parameters "
        "(expected EncryptedTensor<uint2, shape=(3,)> for parameter `x` "
        "but got EncryptedTensor<uint2, shape=(4,)> which is not compatible)\n"
        "Warning: Input #0 (0-indexed) is not coherent with the hinted parameters "
        "(expected ClearTensor<uint2, shape=(3,)> for parameter `y` "
        "but got ClearTensor<uint2, shape=(4,)> which is not compatible)\n"
        "Warning: Input #1 (0-indexed) is not coherent with the hinted parameters "
        "(expected ClearTensor<uint2, shape=(3,)> for parameter `y` "
        "but got ClearTensor<uint3, shape=(3,)> which is not compatible)\n"
    )


def test_eval_op_graph_bounds_on_conformant_numpy_inputset_check_all(capsys):
    """Test function for eval_op_graph_bounds_on_inputset
    with conformant inputset of numpy arrays, check all"""

    def f(x, y):
        return np.dot(x, y)

    x = EncryptedTensor(UnsignedInteger(2), (3,))
    y = ClearTensor(UnsignedInteger(2), (3,))

    inputset = [
        (np.array([2, 1, 3]), np.array([1, 2, 1])),
        (np.array([3, 3, 3]), np.array([3, 3, 1])),
    ]

    op_graph = trace_numpy_function(f, {"x": x, "y": y})

    configuration = CompilationConfiguration(check_every_input_in_inputset=True)
    eval_op_graph_bounds_on_inputset(
        op_graph,
        inputset,
        compilation_configuration=configuration,
        min_func=numpy_min_func,
        max_func=numpy_max_func,
        get_base_value_for_constant_data_func=get_base_value_for_numpy_or_python_constant_data,
    )

    captured = capsys.readouterr()
    assert captured.err == ""


def test_eval_op_graph_bounds_on_non_conformant_numpy_inputset_check_all(capsys):
    """Test function for eval_op_graph_bounds_on_inputset with non conformant inputset, check all"""

    def f(x, y):
        return np.dot(x, y)

    x = EncryptedTensor(UnsignedInteger(2), (3,))
    y = ClearTensor(UnsignedInteger(2), (3,))

    inputset = [
        (np.array([2, 1, 3, 1]), np.array([1, 2, 1, 1])),
        (np.array([3, 3, 3]), np.array([3, 3, 5])),
    ]

    op_graph = trace_numpy_function(f, {"x": x, "y": y})

    configuration = CompilationConfiguration(check_every_input_in_inputset=True)
    eval_op_graph_bounds_on_inputset(
        op_graph,
        inputset,
        compilation_configuration=configuration,
        min_func=numpy_min_func,
        max_func=numpy_max_func,
        get_base_value_for_constant_data_func=get_base_value_for_numpy_or_python_constant_data,
    )

    captured = capsys.readouterr()
    assert (
        captured.err == "Warning: Input #0 (0-indexed) is not coherent with the hinted parameters "
        "(expected EncryptedTensor<uint2, shape=(3,)> for parameter `x` "
        "but got EncryptedTensor<uint2, shape=(4,)> which is not compatible)\n"
        "Warning: Input #0 (0-indexed) is not coherent with the hinted parameters "
        "(expected ClearTensor<uint2, shape=(3,)> for parameter `y` "
        "but got ClearTensor<uint2, shape=(4,)> which is not compatible)\n"
        "Warning: Input #1 (0-indexed) is not coherent with the hinted parameters "
        "(expected ClearTensor<uint2, shape=(3,)> for parameter `y` "
        "but got ClearTensor<uint3, shape=(3,)> which is not compatible)\n"
    )


def test_eval_op_graph_bounds_on_non_conformant_inputset_treating_warnings_as_errors():
    """Test function for eval_op_graph_bounds_on_inputset with non conformant inputset and errors"""

    def f(x, y):
        return np.dot(x, y)

    x = EncryptedTensor(UnsignedInteger(2), (3,))
    y = ClearTensor(UnsignedInteger(2), (3,))

    inputset = [
        (np.array([2, 1, 3, 1]), np.array([1, 2, 1, 1])),
        (np.array([3, 3, 3]), np.array([3, 3, 5])),
    ]

    op_graph = trace_numpy_function(f, {"x": x, "y": y})

    with pytest.raises(ValueError, match=".* is not coherent with the hinted parameters .*"):
        configuration = CompilationConfiguration(treat_warnings_as_errors=True)
        eval_op_graph_bounds_on_inputset(
            op_graph,
            inputset,
            compilation_configuration=configuration,
            min_func=numpy_min_func,
            max_func=numpy_max_func,
            get_base_value_for_constant_data_func=get_base_value_for_numpy_or_python_constant_data,
        )


def test_inpuset_eval_1_input(default_compilation_configuration):
    """Test case for a function with a single parameter and passing the inputset without tuples."""

    def f(x):
        return x + 42

    x = EncryptedScalar(UnsignedInteger(4))

    inputset = range(10)

    op_graph = trace_numpy_function(f, {"x": x})

    eval_op_graph_bounds_on_inputset(
        op_graph,
        inputset,
        compilation_configuration=default_compilation_configuration,
        min_func=numpy_min_func,
        max_func=numpy_max_func,
        get_base_value_for_constant_data_func=get_base_value_for_numpy_or_python_constant_data,
    )

    input_node = op_graph.input_nodes[0]

    assert input_node.inputs[0] == input_node.outputs[0]
    assert input_node.inputs[0] == EncryptedScalar(UnsignedInteger(4))

    output_node = op_graph.output_nodes[0]

    assert output_node.outputs[0] == EncryptedScalar(UnsignedInteger(6))


# TODO: https://github.com/zama-ai/concrete-numpy-internal/issues/772
# Remove once this issue is done
def test_inpuset_eval_1_input_refuse_tuple(default_compilation_configuration):
    """Test case for a function with a single parameter and passing the inputset with tuples."""

    def f(x):
        return x + 42

    x = EncryptedScalar(UnsignedInteger(4))

    inputset = [(i,) for i in range(10)]

    op_graph = trace_numpy_function(f, {"x": x})

    with pytest.raises(AssertionError) as excinfo:
        eval_op_graph_bounds_on_inputset(
            op_graph,
            inputset,
            compilation_configuration=default_compilation_configuration,
            min_func=numpy_min_func,
            max_func=numpy_max_func,
            get_base_value_for_constant_data_func=get_base_value_for_numpy_or_python_constant_data,
        )

    assert str(excinfo.value) == "Tuples are unsupported for single input inputset evaluation"
