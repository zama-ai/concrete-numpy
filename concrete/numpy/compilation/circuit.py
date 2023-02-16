"""
Declaration of `Circuit` class.
"""

from copy import deepcopy
from typing import Any, Optional, Tuple, Union, cast

import numpy as np
from concrete.compiler import PublicArguments, PublicResult

from ..dtypes import Integer
from ..internal.utils import assert_that
from ..mlir import GraphConverter
from ..representation import Graph
from .client import Client
from .configuration import Configuration
from .server import Server


class Circuit:
    """
    Circuit class, to combine computation graph, mlir, client and server into a single object.
    """

    configuration: Configuration

    graph: Graph
    mlir: str

    client: Client
    server: Server

    def __init__(self, graph: Graph, mlir: str, configuration: Optional[Configuration] = None):
        self.configuration = configuration if configuration is not None else Configuration()

        self.graph = graph
        self.mlir = mlir

        if self.configuration.virtual:
            return

        self._initialize_client_and_server()

    def _initialize_client_and_server(self):
        input_signs = []
        for i in range(len(self.graph.input_nodes)):  # pylint: disable=consider-using-enumerate
            input_value = self.graph.input_nodes[i].output
            assert_that(isinstance(input_value.dtype, Integer))
            input_dtype = cast(Integer, input_value.dtype)
            input_signs.append(input_dtype.is_signed)

        output_signs = []
        for i in range(len(self.graph.output_nodes)):  # pylint: disable=consider-using-enumerate
            output_value = self.graph.output_nodes[i].output
            assert_that(isinstance(output_value.dtype, Integer))
            output_dtype = cast(Integer, output_value.dtype)
            output_signs.append(output_dtype.is_signed)

        self.server = Server.create(self.mlir, input_signs, output_signs, self.configuration)

        keyset_cache_directory = None
        if self.configuration.use_insecure_key_cache:
            assert_that(self.configuration.enable_unsafe_features)
            assert_that(self.configuration.insecure_key_cache_location is not None)
            keyset_cache_directory = self.configuration.insecure_key_cache_location

        self.client = Client(self.server.client_specs, keyset_cache_directory)

    def __str__(self):
        return self.graph.format()

    def simulate(self, *args: Any) -> Any:
        """
        Simulate execution of the circuit.

        Args:
            *args (Any):
                inputs to the circuit

        Returns:
            Any:
                result of the simulation
        """

        p_error = self.p_error if not self.configuration.virtual else self.configuration.p_error
        return self.graph(*args, p_error=p_error)

    def enable_fhe(self):
        """
        Enable fully homomorphic encryption features.

        When called on a virtual circuit, it'll enable access to the following methods:
        - encrypt
        - run
        - decrypt
        - encrypt_run_decrypt

        When called on a normal circuit, it'll do nothing.

        Raises:
            RuntimeError:
                if the circuit is not supported in fhe
        """

        if not self.configuration.virtual:
            return

        new_configuration = deepcopy(self.configuration)
        new_configuration.virtual = False
        self.configuration = new_configuration

        self.mlir = GraphConverter.convert(self.graph)
        self._initialize_client_and_server()

    def keygen(self, force: bool = False):
        """
        Generate keys required for homomorphic evaluation.

        Args:
            force (bool, default = False):
                whether to generate new keys even if keys are already generated
        """

        if self.configuration.virtual:
            message = "Virtual circuits cannot use `keygen` method"
            raise RuntimeError(message)

        self.client.keygen(force)

    def encrypt(self, *args: Union[int, np.ndarray]) -> PublicArguments:
        """
        Prepare inputs to be run on the circuit.

        Args:
            *args (Union[int, numpy.ndarray]):
                inputs to the circuit

        Returns:
            PublicArguments:
                encrypted and plain arguments as well as public keys
        """

        if self.configuration.virtual:
            message = "Virtual circuits cannot use `encrypt` method"
            raise RuntimeError(message)

        return self.client.encrypt(*args)

    def run(self, args: PublicArguments) -> PublicResult:
        """
        Evaluate circuit using encrypted arguments.

        Args:
            args (PublicArguments):
                arguments to the circuit (can be obtained with `encrypt` method of `Circuit`)

        Returns:
            PublicResult:
                encrypted result of homomorphic evaluaton
        """

        if self.configuration.virtual:
            message = "Virtual circuits cannot use `run` method"
            raise RuntimeError(message)

        self.keygen(force=False)
        return self.server.run(args, self.client.evaluation_keys)

    def decrypt(
        self,
        result: PublicResult,
    ) -> Union[int, np.ndarray, Tuple[Union[int, np.ndarray], ...]]:
        """
        Decrypt result of homomorphic evaluaton.

        Args:
            result (PublicResult):
                encrypted result of homomorphic evaluaton

        Returns:
            Union[int, numpy.ndarray]:
                clear result of homomorphic evaluaton
        """

        if self.configuration.virtual:
            message = "Virtual circuits cannot use `decrypt` method"
            raise RuntimeError(message)

        return self.client.decrypt(result)

    def encrypt_run_decrypt(self, *args: Any) -> Any:
        """
        Encrypt inputs, run the circuit, and decrypt the outputs in one go.

        Args:
            *args (Union[int, numpy.ndarray]):
                inputs to the circuit

        Returns:
            Union[int, np.ndarray, Tuple[Union[int, np.ndarray], ...]]:
                clear result of homomorphic evaluation
        """

        return self.decrypt(self.run(self.encrypt(*args)))

    def cleanup(self):
        """
        Cleanup the temporary library output directory.
        """

        self.server.cleanup()

    @property
    def complexity(self) -> float:
        """
        Get complexity of the circuit.
        """
        return self.server.complexity

    @property
    def size_of_secret_keys(self) -> int:
        """
        Get size of the secret keys of the circuit.
        """
        return self.server.size_of_secret_keys

    @property
    def size_of_bootstrap_keys(self) -> int:
        """
        Get size of the bootstrap keys of the circuit.
        """
        return self.server.size_of_bootstrap_keys

    @property
    def size_of_keyswitch_keys(self) -> int:
        """
        Get size of the key switch keys of the circuit.
        """
        return self.server.size_of_keyswitch_keys

    @property
    def size_of_inputs(self) -> int:
        """
        Get size of the inputs of the circuit.
        """
        return self.server.size_of_inputs

    @property
    def size_of_outputs(self) -> int:
        """
        Get size of the outputs of the circuit.
        """
        return self.server.size_of_outputs

    @property
    def p_error(self) -> int:
        """
        Get probability of error for each simple TLU (on a scalar).
        """
        return self.server.p_error

    @property
    def global_p_error(self) -> int:
        """
        Get the probability of having at least one simple TLU error during the entire execution.
        """
        return self.server.global_p_error
