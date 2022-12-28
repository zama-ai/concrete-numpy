import numpy as np
import concrete.numpy as cnp
import time
import struct
from itertools import chain

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
    def retrieve2(equal, value):
        return HomomorphicOperation.partial_multiply(equal, value)

    @staticmethod
    def update4(old_key, old_value, new_key, new_value):
        equal = HomomorphicOperation.fhe_equal(old_key, new_key)
        return HomomorphicOperation.partial_multiply(
            equal, new_value
        ) + HomomorphicOperation.partial_multiply(1 - equal, old_value)

    @staticmethod
    def update2(equal, old_value, new_value):
        return HomomorphicOperation.partial_multiply(
            equal, new_value
        ) + HomomorphicOperation.partial_multiply(1 - equal, old_value)

    @staticmethod
    def dummy4(a, b, c, d):
        return a * 0 + b * 0 + c * 0 + d * 0

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
    def fhe_equal8(
        left1,
        left2,
        left3,
        left4,
        left5,
        left6,
        left7,
        left8,
        right1,
        right2,
        right3,
        right4,
        right5,
        right6,
        right7,
        right8,
    ):
        x1 = 0
        x2 = 0
        x3 = 0
        x4 = 0
        x5 = 0
        x6 = 0
        x7 = 0
        x8 = 0

        for i in range(1, 5):
            k = 4 - i

            lk1 = left1 >> k
            rk1 = right1 >> k
            x1 += HomomorphicOperation.fhe_equal1b(lk1, rk1) << k
            left1 -= lk1 << k
            right1 -= rk1 << k

            lk2 = left2 >> k
            rk2 = right2 >> k
            x2 += HomomorphicOperation.fhe_equal1b(lk2, rk2) << k
            left2 -= lk2 << k
            right2 -= rk2 << k

            lk3 = left3 >> k
            rk3 = right3 >> k
            x3 += HomomorphicOperation.fhe_equal1b(lk3, rk3) << k
            left3 -= lk3 << k
            right3 -= rk3 << k

            lk4 = left4 >> k
            rk4 = right4 >> k
            x4 += HomomorphicOperation.fhe_equal1b(lk4, rk4) << k
            left4 -= lk4 << k
            right4 -= rk4 << k

            lk5 = left5 >> k
            rk5 = right5 >> k
            x5 += HomomorphicOperation.fhe_equal1b(lk5, rk5) << k
            left5 -= lk5 << k
            right5 -= rk5 << k

            lk6 = left6 >> k
            rk6 = right6 >> k
            x6 += HomomorphicOperation.fhe_equal1b(lk6, rk6) << k
            left6 -= lk6 << k
            right6 -= rk6 << k

            lk7 = left7 >> k
            rk7 = right7 >> k
            x7 += HomomorphicOperation.fhe_equal1b(lk7, rk7) << k
            left7 -= lk7 << k
            right7 -= lk7 << k

            lk8 = left8 >> k
            rk8 = right8 >> k
            x8 += HomomorphicOperation.fhe_equal1b(lk8, rk8) << k
            left8 -= lk8 << k
            right8 -= rk8 << k

        z1 = ALL_ONE[x1]
        z2 = ALL_ONE[x2]
        z3 = ALL_ONE[x3]
        z4 = ALL_ONE[x4]
        z5 = ALL_ONE[x5]
        z6 = ALL_ONE[x6]
        z7 = ALL_ONE[x7]
        z8 = ALL_ONE[x8]

        return HomomorphicOperation.all8(z1, z2, z3, z4, z5, z6, z7, z8)

    @staticmethod
    def fhe_equal1b8(
        left1,
        left2,
        left3,
        left4,
        left5,
        left6,
        left7,
        left8,
        right1,
        right2,
        right3,
        right4,
        right5,
        right6,
        right7,
        right8,
    ):
        z1 = EQUAL[(left1 * 2) + right1]
        z2 = EQUAL[(left2 * 2) + right2]
        z3 = EQUAL[(left3 * 2) + right3]
        z4 = EQUAL[(left4 * 2) + right4]
        z5 = EQUAL[(left5 * 2) + right5]
        z6 = EQUAL[(left6 * 2) + right6]
        z7 = EQUAL[(left7 * 2) + right7]
        z8 = EQUAL[(left8 * 2) + right8]

        return all8(z1, z2, z3, z4, z5, z6, z7, z8)

    @staticmethod
    def all8(z1, z2, z3, z4, z5, z6, z7, z8):
        z = AND2[(z1 * 2) + z2]
        z = AND2[(z * 2) + z3]
        z = AND2[(z * 2) + z4]
        z = AND2[(z * 2) + z5]
        z = AND2[(z * 2) + z6]
        z = AND2[(z * 2) + z7]
        z = AND2[(z * 2) + z8]

        return z

    @staticmethod
    def fhe_equal1b(left, right):
        return EQUAL[(left * 2) + right]


def variables(*names):
    return {name: "encrypted" for name in names}


class HomomorphicCircuitBoard:
    def __init__(self):
        input2 = [
            tuple(l)
            for l in np.int_(
                np.linspace(
                    (1,) * 2,
                    (
                        1,
                        2**4 - 1,
                    ),
                    100,
                )
            ).tolist()
        ]
        input4 = [
            tuple(l)
            for l in np.int_(np.linspace((0,) * 4, (2**4 - 1,) * 4, 100)).tolist()
        ]

        input16 = [
            tuple(l)
            for l in np.int_(np.linspace((0,) * 16, (2**4 - 1,) * 16, 100)).tolist()
        ]

        self.retrieve2 = cnp.Compiler(
            HomomorphicOperation.retrieve2, variables("equal", "value")
        ).compile(input2)

        self.update4 = cnp.Compiler(
            HomomorphicOperation.update4,
            variables("old_key", "old_value", "new_key", "new_value"),
        ).compile(input4)

        self.fhe_equal8 = cnp.Compiler(
            HomomorphicOperation.fhe_equal8,
            variables(
                "left1",
                "left2",
                "left3",
                "left4",
                "left5",
                "left6",
                "left7",
                "left8",
                "right1",
                "right2",
                "right3",
                "right4",
                "right5",
                "right6",
                "right7",
                "right8",
            ),
        ).compile(input16)

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


def unpack(key):
    # Read the 32 bits as a 4-byte bytearray
    key = key.to_bytes(length=4, byteorder="big")
    # Split the key into four groups of 8 bits
    groups = struct.unpack("c" * 4, key)
    # Split each 8-bit group into two groups of 4-bits

    key = list(
        chain.from_iterable(
            [
                [
                    (0xF0 & int.from_bytes(b, byteorder="big")) >> 4,
                    0x0F & int.from_bytes(b, byteorder="big"),
                ]
                for b in groups
            ]
        )
    )

    return key


def pack(key):
    total = 0
    for i, k in zip(range(8), key[::-1]):
        # Every integer represents 4bits
        total += 2**(4*i) * k
    return total


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
        self.retrieve = self.circuit.retrieve2
        self.equal8 = self.circuit.fhe_equal8

    def insert(self, key, value):
        key = unpack(key)
        value = unpack(value)
        self.base.append((key, value))

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
        key = unpack(key)
        result = [0] * 8
        encryption_time = 0

        s = time.time()
        for (keys, values) in self.base:
            # First encrypt the keys together, we need to compare the keys to
            # retrieve the right values

            s1 = time.time()
            print(len(keys), len(key))
            encrypted = self.equal8.encrypt(*keys, *key)
            e1 = time.time()
            encryption_time += e1 - s1

            equal = self.equal8.run(encrypted)

            s2 = time.time()
            equal_d = self.equal8.decrypt(equal)
            e2 = time.time()
            encryption_time += e2 - s2

            for index, value in enumerate(values):
                s3 = time.time()
                encrypted = self.retrieve.encrypt(equal_d, value)
                e3 = time.time()
                encryption_time += e3 - s3

                r = self.retrieve.run(encrypted)

                s4 = time.time()
                r_d = self.retrieve.decrypt(r)
                e4 = time.time()
                encryption_time += e4 - s4
                print(r_d)

                result[index] += r_d

        e = time.time()

        print(
            f"get: Spent {e - s - encryption_time:.2f}s processing data and an extra {encryption_time:.2f}s encrypting and decrypting data"
        )

        return pack(result)


database = HomomorphicDatabase()
database.insert(2**31, 2**30 + 5)

print("Retrieving a value from the database")
print(f"{2**30 + 5} =", database.get(2**31))
exit()

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
