"""
Tests of execution of shifts operation.
"""

import pytest

import concrete.numpy as cnp


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(
            lambda x, y: x << y,
            id="x << y",
        ),
    ],
)
@pytest.mark.parametrize(
    "parameters",
    [
        {
            "x": {"range": [0, 1], "status": "encrypted"},
            "y": {"range": [0, 7], "status": "encrypted"},
        },
        {
            "x": {"range": [0, 3], "status": "encrypted", "shape": (2,)},
            "y": {"range": [0, 3], "status": "encrypted", "shape": (2,)},
        },
    ],
)
def _test_left_shift_coverage(function, parameters, helpers):
    """
    Test left shift where both of the operators are dynamic.
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
            lambda x, y: x >> y,
            id="x >> y",
        ),
    ],
)
@pytest.mark.parametrize(
    "parameters",
    [
        {
            "x": {"range": [0, 1 << 7], "status": "encrypted"},
            "y": {"range": [0, 7], "status": "encrypted"},
        },
        {
            "x": {"range": [0, 1 << 4], "status": "encrypted", "shape": (2,)},
            "y": {"range": [0, 3], "status": "encrypted", "shape": (2,)},
        },
    ],
)
def test_left_shift(function, parameters, helpers):
    """
    Test right shift where both of the operators are dynamic.
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
            lambda x, y: x << y,
            id="x << y",
        ),
    ],
)
@pytest.mark.parametrize(
    "parameters",
    [
        {
            "x": {"range": [0, 1], "status": "encrypted"},
            "y": {"range": [0, 7], "status": "encrypted"},
        },
    ],
)
def test_left_shift_coverage(function, parameters, helpers):
    """
    Test left shift where both of the operators are dynamic.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration()

    compiler = cnp.Compiler(function, parameter_encryption_statuses)

    inputset = helpers.generate_inputset(parameters)
    circuit = compiler.compile(inputset, configuration)

    helpers.check_execution(circuit, function, [1, 0])
    helpers.check_execution(circuit, function, [1, 1])
    helpers.check_execution(circuit, function, [1, 2])
    helpers.check_execution(circuit, function, [1, 3])
    helpers.check_execution(circuit, function, [1, 4])
    helpers.check_execution(circuit, function, [1, 5])
    helpers.check_execution(circuit, function, [1, 6])
    helpers.check_execution(circuit, function, [1, 7])


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(
            lambda x, y: x >> y,
            id="x >> y",
        ),
    ],
)
@pytest.mark.parametrize(
    "parameters",
    [
        {
            "x": {"range": [0, 1 << 7], "status": "encrypted"},
            "y": {"range": [0, 7], "status": "encrypted"},
        },
    ],
)
def test_right_shift_coverage(function, parameters, helpers):
    """
    Test right shift where both of the operators are dynamic.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration()

    compiler = cnp.Compiler(function, parameter_encryption_statuses)

    inputset = helpers.generate_inputset(parameters)
    circuit = compiler.compile(inputset, configuration)

    helpers.check_execution(circuit, function, [0b11, 0])
    helpers.check_execution(circuit, function, [0b11, 1])
    helpers.check_execution(circuit, function, [0b110, 2])
    helpers.check_execution(circuit, function, [0b1100, 3])
    helpers.check_execution(circuit, function, [0b11000, 4])
    helpers.check_execution(circuit, function, [0b110000, 5])
    helpers.check_execution(circuit, function, [0b110000, 6])
    helpers.check_execution(circuit, function, [0b1100000, 7])
