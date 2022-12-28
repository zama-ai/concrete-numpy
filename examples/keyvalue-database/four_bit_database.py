import numpy as np
import concrete.numpy as cnp
import time

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
    def retrieve4(key, value, query):
        equal = HomomorphicOperation.fhe_equal(key, query)
        return HomomorphicOperation.partial_multiply(equal, value)

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
            result += AND2[(left * 2) + rk] << k
            right -= rk << k

        return result

    @staticmethod
    def fhe_equal_(left, right):
        z = 1
        for i in range(1, 5):
            k = 4 - i
            lk = left >> k
            rk = right >> k
            z = AND2[(z << 1) + HomomorphicOperation.fhe_equal1b(lk, rk)]
            left -= lk << k
            right -= rk << k

        return z

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
        return EQUAL[(left * 2) + right]


def variables(*names):
    return {name: "encrypted" for name in names}


class HomomorphicCircuitBoard:
    def __init__(self):
        input3 = [
            tuple(l)
            for l in np.int_(np.linspace((0,) * 3, (2**4 - 1,) * 3, 100)).tolist()
        ]
        input4 = [
            tuple(l)
            for l in np.int_(np.linspace((0,) * 4, (2**4 - 1,) * 4, 100)).tolist()
        ]

        self.retrieve4 = cnp.Compiler(
            HomomorphicOperation.retrieve4, variables("key", "value", "query")
        ).compile(input3)

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
    """
    A homomorphic four bit database.

    Since concrete-numpy does not support performing operations
    on circuit ouput, the entries in the database are stored in plain text,
    and encrypt-values before performing operations on them, then
    subsequently decrypt the results. Because of this, I report
    the amount of time spent encrypting and decrypting values.
    When concrete-numpy supports this, it can be changed.
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.circuit = HomomorphicCircuitBoard()
        self.update = self.circuit.update4
        self.retrieve = self.circuit.retrieve4

    def replace(self, key, value):
        s = time.time()
        encryption_time = 0
        for index, entry in enumerate(self.base):
            s1 = time.time()
            encrypted = self.update.encrypt(*entry, key, value)
            e1 = time.time()

            new_value = self.update.run(encrypted)

            s2 = time.time()
            new_value_d = self.update.decrypt(new_value)
            e2 = time.time()

            encryption_time += (e1 - s1) + (e2 - s2)

            old_key = entry[0]
            self.base[index] = (old_key, new_value_d)
        e = time.time()

        print(
            f"replace: Spent {e - s - encryption_time:.2F}s processing data and an extra {encryption_time:.2f}s encrypting and decrypting results"
        )

    def get(self, key):
        """
        The operation is R(x0) + R(x1) + R(x2) + ...
        concrete-numpy does not support performing operations on circuit output
        so at each step of computation, the output from the circuit is decrypted
        """
        result = 0
        encryption_time = 0

        s = time.time()
        for entry in self.base:
            s1 = time.time()
            encrypted = self.retrieve.encrypt(*entry, key)
            e1 = time.time()
            encryption_time += e1 - s1

            r = self.retrieve.run(encrypted)

            s2 = time.time()
            r_d = self.retrieve.decrypt(r)
            e2 = time.time()
            encryption_time += e2 - s2

            result += r_d

        e = time.time()

        print(
            f"get: Spent {e - s - encryption_time:.2f}s processing data and an extra {encryption_time:.2f}s encrypting and decrypting data"
        )

        return result


database = HomomorphicDatabase()
database.insert(1, 1)

print("Retrieving a value from the database")
print("1 =", database.get(1))

print("Replacing a value from the database")
database.replace(1, 13)
print("Retrieving a value from the database")
print("13 =", database.get(1))

database.insert(5, 6)
database.insert(8, 9)
print("Added 2 values to the database")
print("9 =", database.get(8))

database.insert(15, 3)
database.insert(3, 15)

print("15 =", database.get(3))

print("Attempting to access an item that is not in the database")
print("0 =", database.get(14))
