import numpy as np
import concrete.numpy as cnp

NON_ZERO = cnp.LookupTable([int(x != 0) for x in range(2**4)])
EQUAL = cnp.LookupTable([1, 0, 0, 1])
ALL_ONE = cnp.LookupTable([0 for _ in range(2**4 - 1)] + [1])
AND2 = cnp.LookupTable([0, 0, 0, 1])


class HomomorphicOperation:
    @staticmethod
    def retrieve(key, value, query):
        equal = 1 - HomomorphicOperation.non_zero(key ^ query)
        return equal * value

    @staticmethod
    def update(old_key, old_value, new_key, new_value):
        not_equal = HomomorphicOperation.non_zero(old_key ^ new_key)
        return not_equal * old_value + (1 - not_equal) * new_value

    @staticmethod
    def non_zero(number):
        output = 0
        for i in range(8):
            output |= number >> i
        return output & 1

    @staticmethod
    def identity(x):
        return x

    @staticmethod
    def retrieve4(key, value, query, _):
        equal = HomomorphicOperation.fhe_equal(key, query)
        return HomomorphicOperation.partial_multiply(equal, value) + 0 * _

    @staticmethod
    def update4(old_key, old_value, new_key, new_value):
        equal = HomomorphicOperation.fhe_equal(old_key, new_key)
        return HomomorphicOperation.partial_multiply(
            equal, new_value
        ) + HomomorphicOperation.partial_multiply(1 - equal, old_value)

    @staticmethod
    def partial_multiply(left, right):
        result = 0

        for i in range(1, 5):
            k = 4 - i
            rk = right >> k
            result += AND2[(left << 1) + rk] << k
            right -= rk << k

        return result

    @staticmethod
    def fhe_equal(left, right):
        x = 0
        for i in range(1, 5):
            k = 4 - i
            lk = left >> k
            rk = right >> k
            x += HomomorphicOperation.fhe_equal1b(lk, rk) << k
            left -= lk << k
            right -= rk << k

        return ALL_ONE[x]

    @staticmethod
    def fhe_equal1b(left, right):
        return EQUAL[(left << 1) + right]


def variables(*names):
    return {name: "encrypted" for name in names}


class HomomorphicCircuitBoard:
    def __init__(self):
        input4 = [
            tuple(l)
            for l in np.int_(np.linspace((0,) * 4, (2**4 - 1,) * 4, 100)).tolist()
        ]

        self.retrieve4 = cnp.Compiler(
            HomomorphicOperation.retrieve4, variables("key", "value", "query", "_")
        ).compile(input4)

        self.update4 = cnp.Compiler(
            HomomorphicOperation.update4,
            variables("old_key", "old_value", "new_key", "new_value"),
        ).compile(input4)

    def retrieve(self, key, value, query):
        return self.retrieve4.encrypt_run_decrypt(key, value, query, 0)

    def update(self, old_key, old_value, new_key, new_value):
        return self.update4.encrypt_run_decrypt(old_key, old_value, new_key, new_value)


class AbstractDatabase:
    """A FHE database with integer keys and values"""

    def __init__(self):
        self.base = []

    def insert(self, key, value):
        """Inserts a value into the database"""
        self.base.append((key, value))

    def replace(self, key, value):
        """Replaces a value in the database"""
        for index, (old_key, old_value) in enumerate(self.base):
            new_value = self.update(old_key, old_value, key, value)
            self.base[index] = (old_key, new_value)

    def get(self, key):
        """Gets a value from the database"""
        result = 0
        for entry in self.base:
            result += self.retrieve(*entry, key)
        return result


class ClearDatabase(AbstractDatabase):
    def __init__(self, *args):
        super().__init__(*args)
        self.update = HomomorphicOperation.update
        self.retrieve = HomomorphicOperation.retrieve


class HomomorphicDatabase(AbstractDatabase):
    def __init__(self, *args):
        super().__init__(*args)
        self.circuit = HomomorphicCircuitBoard()

        self.update = self.circuit.update
        self.retrieve = self.circuit.retrieve
