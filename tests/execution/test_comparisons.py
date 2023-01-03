"""
Tests of execution of comparisons operation.
"""

import pytest

import concrete.numpy as cnp


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(
            lambda x, y: x == y,
            id="x == y",
        ),
    ],
)
@pytest.mark.parametrize(
    "parameters",
    [
        {
            "x": {"range": [0, 255], "status": "encrypted"},
            "y": {"range": [0, 255], "status": "encrypted"},
        },
        {
            "x": {"range": [-128, 127], "status": "encrypted"},
            "y": {"range": [-128, 127], "status": "encrypted"},
        },
        {
            "x": {"range": [-128, 127], "status": "encrypted"},
            "y": {"range": [0, 255], "status": "encrypted"},
        },
        {
            "x": {"range": [0, 255], "status": "encrypted"},
            "y": {"range": [-128, 127], "status": "encrypted"},
        },
        {
            "x": {"range": [0, 15], "status": "encrypted", "shape": (2,)},
            "y": {"range": [0, 15], "status": "encrypted", "shape": (2,)},
        },
    ],
)
def test_equal(function, parameters, helpers):
    """
    Test equal where both of the operators are dynamic.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration()

    compiler = cnp.Compiler(function, parameter_encryption_statuses)

    inputset = helpers.generate_inputset(parameters)
    circuit = compiler.compile(inputset, configuration)

    sample = helpers.generate_sample(parameters)
    helpers.check_execution(circuit, function, sample)


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(
            lambda x, y: x != y,
            id="x != y",
        ),
    ],
)
@pytest.mark.parametrize(
    "parameters",
    [
        {
            "x": {"range": [0, 255], "status": "encrypted"},
            "y": {"range": [0, 255], "status": "encrypted"},
        },
        {
            "x": {"range": [-128, 127], "status": "encrypted"},
            "y": {"range": [-128, 127], "status": "encrypted"},
        },
        {
            "x": {"range": [-128, 127], "status": "encrypted"},
            "y": {"range": [0, 255], "status": "encrypted"},
        },
        {
            "x": {"range": [0, 255], "status": "encrypted"},
            "y": {"range": [-128, 127], "status": "encrypted"},
        },
        {
            "x": {"range": [0, 15], "status": "encrypted", "shape": (2,)},
            "y": {"range": [0, 15], "status": "encrypted", "shape": (2,)},
        },
    ],
)
def test_not_equal(function, parameters, helpers):
    """
    Test not equal where both of the operators are dynamic.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration()

    compiler = cnp.Compiler(function, parameter_encryption_statuses)

    inputset = helpers.generate_inputset(parameters)
    circuit = compiler.compile(inputset, configuration)

    sample = helpers.generate_sample(parameters)
    helpers.check_execution(circuit, function, sample)


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(
            lambda x, y: x < y,
            id="x < y",
        ),
    ],
)
@pytest.mark.parametrize(
    "parameters",
    [
        {
            "x": {"range": [0, 255], "status": "encrypted"},
            "y": {"range": [0, 255], "status": "encrypted"},
        },
        {
            "x": {"range": [-128, 127], "status": "encrypted"},
            "y": {"range": [-128, 127], "status": "encrypted"},
        },
        {
            "x": {"range": [-128, 127], "status": "encrypted"},
            "y": {"range": [0, 255], "status": "encrypted"},
        },
        {
            "x": {"range": [0, 255], "status": "encrypted"},
            "y": {"range": [-128, 127], "status": "encrypted"},
        },
        {
            "x": {"range": [0, 1], "status": "encrypted"},
            "y": {"range": [0, 1], "status": "encrypted"},
        },
        {
            "x": {"range": [0, 15], "status": "encrypted", "shape": (2,)},
            "y": {"range": [0, 15], "status": "encrypted", "shape": (2,)},
        },
        {
            "x": {"range": [-2, 4], "status": "encrypted", "shape": (2,)},
            "y": {"range": [0, 15], "status": "encrypted", "shape": (2,)},
        },
    ],
)
def test_less(function, parameters, helpers):
    """
    Test less where both of the operators are dynamic.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration()

    compiler = cnp.Compiler(function, parameter_encryption_statuses)

    inputset = helpers.generate_inputset(parameters)
    circuit = compiler.compile(inputset, configuration)

    sample = helpers.generate_sample(parameters)
    helpers.check_execution(circuit, function, sample)


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(
            lambda x, y: x <= y,
            id="x <= y",
        ),
    ],
)
@pytest.mark.parametrize(
    "parameters",
    [
        {
            "x": {"range": [0, 255], "status": "encrypted"},
            "y": {"range": [0, 255], "status": "encrypted"},
        },
        {
            "x": {"range": [-128, 127], "status": "encrypted"},
            "y": {"range": [-128, 127], "status": "encrypted"},
        },
        {
            "x": {"range": [-128, 127], "status": "encrypted"},
            "y": {"range": [0, 255], "status": "encrypted"},
        },
        {
            "x": {"range": [0, 255], "status": "encrypted"},
            "y": {"range": [-128, 127], "status": "encrypted"},
        },
        {
            "x": {"range": [0, 15], "status": "encrypted", "shape": (2,)},
            "y": {"range": [0, 15], "status": "encrypted", "shape": (2,)},
        },
    ],
)
def test_less_equal(function, parameters, helpers):
    """
    Test less equal where both of the operators are dynamic.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration()

    compiler = cnp.Compiler(function, parameter_encryption_statuses)

    inputset = helpers.generate_inputset(parameters)
    circuit = compiler.compile(inputset, configuration)

    sample = helpers.generate_sample(parameters)
    helpers.check_execution(circuit, function, sample)


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(
            lambda x, y: x > y,
            id="x > y",
        ),
    ],
)
@pytest.mark.parametrize(
    "parameters",
    [
        {
            "x": {"range": [0, 255], "status": "encrypted"},
            "y": {"range": [0, 255], "status": "encrypted"},
        },
        {
            "x": {"range": [-128, 127], "status": "encrypted"},
            "y": {"range": [-128, 127], "status": "encrypted"},
        },
        {
            "x": {"range": [-128, 127], "status": "encrypted"},
            "y": {"range": [0, 255], "status": "encrypted"},
        },
        {
            "x": {"range": [0, 255], "status": "encrypted"},
            "y": {"range": [-128, 127], "status": "encrypted"},
        },
        {
            "x": {"range": [0, 15], "status": "encrypted", "shape": (2,)},
            "y": {"range": [0, 15], "status": "encrypted", "shape": (2,)},
        },
    ],
)
def test_greater(function, parameters, helpers):
    """
    Test greater where both of the operators are dynamic.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration()

    compiler = cnp.Compiler(function, parameter_encryption_statuses)

    inputset = helpers.generate_inputset(parameters)
    circuit = compiler.compile(inputset, configuration)

    sample = helpers.generate_sample(parameters)
    helpers.check_execution(circuit, function, sample)


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(
            lambda x, y: x >= y,
            id="x >= y",
        ),
    ],
)
@pytest.mark.parametrize(
    "parameters",
    [
        {
            "x": {"range": [0, 255], "status": "encrypted"},
            "y": {"range": [0, 255], "status": "encrypted"},
        },
        {
            "x": {"range": [-128, 127], "status": "encrypted"},
            "y": {"range": [-128, 127], "status": "encrypted"},
        },
        {
            "x": {"range": [-128, 127], "status": "encrypted"},
            "y": {"range": [0, 255], "status": "encrypted"},
        },
        {
            "x": {"range": [0, 255], "status": "encrypted"},
            "y": {"range": [-128, 127], "status": "encrypted"},
        },
        {
            "x": {"range": [0, 15], "status": "encrypted", "shape": (2,)},
            "y": {"range": [0, 15], "status": "encrypted", "shape": (2,)},
        },
    ],
)
def test_greater_equal(function, parameters, helpers):
    """
    Test greater equal where both of the operators are dynamic.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration()

    compiler = cnp.Compiler(function, parameter_encryption_statuses)

    inputset = helpers.generate_inputset(parameters)
    circuit = compiler.compile(inputset, configuration)

    sample = helpers.generate_sample(parameters)
    helpers.check_execution(circuit, function, sample)


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(
            lambda x, y: x < y,
            id="x < y",
        ),
        pytest.param(
            lambda x, y: x <= y,
            id="x <= y",
        ),
        pytest.param(
            lambda x, y: x > y,
            id="x > y",
        ),
        pytest.param(
            lambda x, y: x >= y,
            id="x >= y",
        ),
        pytest.param(
            lambda x, y: x == y,
            id="x == y",
        ),
        pytest.param(
            lambda x, y: x != y,
            id="x != y",
        ),
    ],
)
@pytest.mark.parametrize(
    "parameters",
    [
        {
            "x": {"range": [-4, 3], "status": "encrypted"},
            "y": {"range": [0, 7], "status": "encrypted"},
        },
        {
            "x": {"range": [-4, 3], "status": "encrypted"},
            "y": {"range": [-4, 3], "status": "encrypted"},
        },
        {
            "x": {"range": [0, 7], "status": "encrypted"},
            "y": {"range": [-4, 3], "status": "encrypted"},
        },
        {
            "x": {"range": [0, 7], "status": "encrypted"},
            "y": {"range": [0, 7], "status": "encrypted"},
        },
        {
            "x": {"range": [-2, 1], "status": "encrypted"},
            "y": {"range": [0, 3], "status": "encrypted"},
        },
        {
            "x": {"range": [-2, 1], "status": "encrypted"},
            "y": {"range": [-2, 1], "status": "encrypted"},
        },
        {
            "x": {"range": [0, 3], "status": "encrypted"},
            "y": {"range": [-2, 1], "status": "encrypted"},
        },
        {
            "x": {"range": [0, 3], "status": "encrypted"},
            "y": {"range": [0, 3], "status": "encrypted"},
        },
    ],
)
def _test_full_coverage(function, parameters, helpers):
    """
    Test comparisons where both of the operators are dynamic.

    Uncomment when improving on the algorithms as the test is slow
    but covers all cases.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration()

    compiler = cnp.Compiler(function, parameter_encryption_statuses)

    inputset = helpers.generate_inputset(parameters)
    circuit = compiler.compile(inputset, configuration)

    for args in set(inputset):
        helpers.check_execution(circuit, function, list(args))
