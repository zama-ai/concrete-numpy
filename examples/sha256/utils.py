import numpy as np
import concrete.numpy as cnp

def split_bits(num, nbits, bit_width=32):
    """
    Splits an integer of 'bit_width' bits into smaller integers of 'nbits' bits each.
    """
    # Check if nbits is a valid value
    if not (nbits in [1, 2, 4, 8, 16]):
        raise ValueError("nbits must be in [1, 2, 4, 8, 16]")

    if bit_width<nbits:
        raise ValueError("nbits must be lower than bit_width")

    # Convert the input integer to binary string
    binary_str = format(num, '0' + str(bit_width) + 'b')

    # Split the binary string into smaller chunks of 'nbits' bits each
    chunks = [binary_str[i:i+nbits] for i in range(0, bit_width, nbits)]

    # Convert each chunk to integer
    integers = [int(chunk, 2) for chunk in chunks]

    # Convert the list of integers to a NumPy array
    return np.array(integers)


def merge_bits(arr, nbits, bit_width=32):
    """
    Merges smaller integers of 'nbits' bits each into a single integer of 'bit_width' bits.
    """
    # Check if nbits is a valid value
    if not (nbits in [1, 2, 4, 8, 16]):
        raise ValueError("nbits must be in [1, 2, 4, 8, 16]")

    if bit_width<=nbits:
        raise ValueError("nbits must be lower than bit_width")        

    # Convert each element in the array to binary string of 'nbits' bits and concatenate
    binary_str = ['{:0{n}b}'.format(num, n=nbits) for num in arr]
    binary_str = ''.join(binary_str)

    # Convert the binary string to an integer of 'bit_width' bits
    return int(binary_str[:bit_width], 2)


def ints_to_nbits( ints: list , nbits, bit_width=32) -> np.ndarray:
    """
    Convert an array of bit_width-bits ints to an array of arrays of nbits integers
    """    
    return np.array( [ split_bits(k,nbits,bit_width) for k in ints ] ).astype(np.uint32);


def ints_from_nbits( intsNbits: list , nbits, bit_width=32) -> np.ndarray:
    """
    Reverse of ints_to_nbits
    """    
    return np.array( [ merge_bits(arr,nbits,bit_width) for arr in intsNbits ] ).astype(np.uint32);


def uint8ToUint4(x: np.ndarray) -> np.ndarray:
    """"
    Convert bytes (8 bits) to 4-bits integers
    """
    z = np.zeros(2*len(x))
    for i in range(0,len(x)):
        d=x[i]
        z[i*2] = (d >> 4) & 15
        z[i*2+1] = d & 15

    return z


class FnSplit:
    """
    Mother class of factory classes for functions operating on 32-bits integers encoded into several smaller integers
    """
    def __init__(self, nbits, use_cnp=True):
        assert( nbits in [1,2,4,8,16] )
        self.nbits = nbits
        self.nchunks = int(32/nbits)        
        self._2p_nbits = 2**nbits
        if use_cnp:
            self.zeros = lambda : cnp.zeros(self.nchunks)
        else:
            self.zeros = lambda : np.zeros(self.nchunks, dtype=np.uint32)


class AddSplit(FnSplit):
    """
    Factory class for Addition functions operating on 32-bits integers encoded into several smaller integers
    """
    def __call__(self, x: np.ndarray, y: np.ndarray):
        """
        Addition mod 2**32 of two 32-bits integers encoded as arrays of nchunks n-bits integers (big-endian)
        """
        z = self.zeros()
        n=self.nchunks-1
        z[n] = x[n] + y[n]

        for i in range(1,self.nchunks):
            z[n-i] = x[n-i] + y[n-i] + (z[n-i+1] >> self.nbits)
            z[n-i+1] = z[n-i+1] & (self._2p_nbits-1) 

        # cast first integer to nbits to be modulo 2**32
        z[0] = z[0] & (self._2p_nbits-1)

        return z


class ROTRSplit(FnSplit):
    """
    Factory class for ROTR functions operating on 32-bits integers encoded into several smaller integers
    """       

    def __call__(self, x: np.ndarray, y: np.uint32):
        """
        Right bits rotation of a 32-bits integers encoded as array of nchunks n-bits integers (big-endian)
        """  
        z = self.zeros()
        temp = self.zeros()

        # first right rotate array of amount int(y/nbits)
        yc = int(y/self.nbits)
        if(yc>0):
            # right shift of yc chunks
            temp[yc:] = x[:-yc]
            # left shift of nchunks-yc chunks
            temp[:yc] = x[-yc:]
        else:
            temp[:] = x[:]

        # now rotate everything with remaining shift
        yr = y%self.nbits
        if(self.nbits>1 and yr>0):
            for i in range(0,self.nchunks):
                z[i] = (temp[i] >> yr) + ( (temp[i-1] << (self.nbits-yr)) & (self._2p_nbits-1) )
        else:
            z[:] = temp[:]

        return z


class SHRSplit(FnSplit):
    """
    Factory class for SHR functions operating on 32-bits integers encoded into several smaller integers 
    """

    def __call__(self, x: np.ndarray, y: np.uint32):
        """
        Right shift of a 32-bits integers encoded as array of nchunks n-bits integers (big-endian)
        """        
        # first right shift array of int(y/nbits)

        z = self.zeros()
        temp = self.zeros()

        yc = int(y/self.nbits)
        if(yc>0):
            # right shift of yc chunks
            temp[yc:] = x[:-yc]
            # the left values are null
        else:
            temp[:] = x[:]

        # now shift everything with remaining shift
        yr = y%self.nbits
        if(self.nbits>1 and yr>0):
            for i in range(1,self.nchunks):
                z[i] = (temp[i] >> yr) + ( (temp[i-1] << (self.nbits-yr)) & (self._2p_nbits-1) )
            z[0] = (temp[0] >> yr) # first chunk is a simple right shift
        else:
            z[:] = temp[:]

        return z


def Ch(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """
    The choose function
    """
    return z ^ (x & (y ^ z))


def Maj(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """
    The majority function
    """
    return ((x | y) & z) | (x & y)


class Sigmas(FnSplit):
    """
    Class holding variables to run sigma functions
    """
    def __init__(self, nbits, use_cnp=True):
        FnSplit.__init__(self, nbits)        

        self.ROTR = ROTRSplit(nbits,use_cnp)
        self.SHR = SHRSplit(nbits,use_cnp)

    def SIGMA0(self, x: np.ndarray):
        """
        Upper case sigma 0 function
        """     
        return self.ROTR(x, 2) ^ self.ROTR(x, 13) ^ self.ROTR(x, 22);

    def SIGMA1(self, x: np.ndarray):
        """
        Upper case sigma 1 function
        """      
        return self.ROTR(x, 6) ^ self.ROTR(x, 11) ^ self.ROTR(x, 25);


    def sigma0(self, x: np.ndarray):
        """
        Lower case sigma 0 function
        """
        return self.ROTR(x, 7) ^ self.ROTR(x, 18) ^ self.SHR(x, 3);


    def sigma1(self, x: np.ndarray):
        """
        Lower case sigma 1 function
        """   
        return self.ROTR(x, 17) ^ self.ROTR(x, 19) ^ self.SHR(x, 10);


##### TESTS #####

TP32=2**32

def testSHR(x, n):
    """
    The right shift operation.
    Cast x to 32 bits and then shift it of n bits to the right
    """
    return (x & (TP32-1)) >> n

def testSHL(x, n):
    """
    The left shift operation.
    Shift x of n bits to the left and then cast it to 32 bits
    """
    return (x << n) & (TP32-1)

def testROTR(x, n):
    """
    The rotate right (circular right shift) operation.
    It is a union of a right shift of n bits and a left shit of w-n bits
    """
    return testSHR(x,n) | testSHL(x,32-n)

def test_bits():
    for i in range(10):
        bw=np.random.randint(2,5)*16
        k=np.random.randint(0, 2**min(32,bw))
        assert( merge_bits(split_bits(k,1,bw),1, bw) ==  k)          
        assert( merge_bits(split_bits(k,2,bw),2, bw) ==  k)          
        assert( merge_bits(split_bits(k,4,bw),4, bw) ==  k)          
        assert( merge_bits(split_bits(k,8,bw),8, bw) ==  k) 
        assert( merge_bits(split_bits(k,16,bw),16,bw) ==  k)      

    print("test bits: OK")


def test_uint8ToUint4():
    for i in range(20):
        k=np.random.randint(0, 2**8)
        uints4 = uint8ToUint4([k])
        assert( merge_bits( np.array([0]*6+uints4.tolist(), dtype=np.uint32), 4 ) == k)          
    print("test uint8ToUint4: OK") 


def test_AddSplit():
  
    for nbits in [1, 2, 4, 8, 16]:
        add = AddSplit(nbits, False)

        # test 1 - carry propagation
        add1 = np.array( [2**nbits-1]*add.nchunks )
        add2 = np.array( [0]*(add.nchunks-1)+[1] )
        expected = np.array( [0]*add.nchunks )
        res = add(add1, add2)
        assert(res.tolist() == expected.tolist())

        # test 2 - random
        for k in range(10):
            add1 = np.random.randint(0, 2**(nbits-1), (add.nchunks,))
            add2 = np.random.randint(0, 2**(nbits-1), (add.nchunks,))
            expected = split_bits( merge_bits( add1, nbits)+merge_bits( add2, nbits), nbits )
            res = add(add1, add2)
            assert(res.tolist() == expected.tolist())

    print("test AddSplit: OK")  


def test_ROTRSplit():

    import utils

    for nbits in [1, 2, 4, 8, 16]:
        ROTR = ROTRSplit(nbits, False)

        for k in range(20):
            n = np.random.randint(0,32)
            vec = np.random.randint(0, 2**(nbits-1), (ROTR.nchunks,))
            res = ROTR(vec, n)
            int32 = merge_bits( vec, nbits)
            expected = split_bits( testROTR(int32, n) , nbits )
            assert(res.tolist() == expected.tolist())    

    print("test ROTRSplit: OK")


def test_SHRSplit():

    import utils

    for nbits in [1, 2, 4, 8, 16]:
        SHR = SHRSplit(nbits, False)

        for k in range(20):
            n = np.random.randint(0,32)
            vec = np.random.randint(0, 2**(nbits-1), (SHR.nchunks,))
            res = SHR(vec, n)
            int32 = merge_bits( vec, nbits)
            expected = split_bits( testSHR(int32, n) , nbits )
            assert(res.tolist() == expected.tolist())

    print("test SHRSplit: OK")


if __name__ == "__main__":
    test_bits()
    test_uint8ToUint4()
    test_AddSplit()
    test_ROTRSplit()
    test_SHRSplit()    
