"""
Tests of execution of sum operation.
"""

import pytest

import concrete.numpy as cnp


@pytest.mark.parametrize(
    "function,parameters",
    [
        pytest.param(
            lambda x, y: x > y,
            {
                "x": {"range": [0, 10], "status": "encrypted"},
                "y": {"range": [0, 10], "status": "encrypted"},
            },
        ),
        pytest.param(
            lambda x, y: x >= y,
            {
                "x": {"range": [0, 10], "status": "encrypted"},
                "y": {"range": [0, 10], "status": "encrypted"},
            },
        ),
        pytest.param(
            lambda x, y: x < y,
            {
                "x": {"range": [0, 10], "status": "encrypted"},
                "y": {"range": [0, 10], "status": "encrypted"},
            },
        ),
        pytest.param(
            lambda x, y: x <= y,
            {
                "x": {"range": [0, 10], "status": "encrypted"},
                "y": {"range": [0, 10], "status": "encrypted"},
            },
        ),
        pytest.param(
            lambda x, y: x == y,
            {
                "x": {"range": [0, 10], "status": "encrypted"},
                "y": {"range": [0, 10], "status": "encrypted"},
            },
        ),
        pytest.param(
            lambda x, y: x != y,
            {
                "x": {"range": [0, 10], "status": "encrypted"},
                "y": {"range": [0, 10], "status": "encrypted"},
            },
        ),
    ],
)
def test_comparison(function, parameters, helpers):
    """
    Test comparison operators with both inputs encrypted.
    """

    parameter_encryption_statuses = helpers.generate_encryption_statuses(parameters)
    configuration = helpers.configuration()

    compiler = cnp.Compiler(function, parameter_encryption_statuses)

    inputset = helpers.generate_inputset(parameters)
    circuit = compiler.compile(inputset, configuration)

    sample = helpers.generate_sample(parameters)
    helpers.check_execution(circuit, function, sample)
