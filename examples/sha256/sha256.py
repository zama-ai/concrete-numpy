# -*- coding: utf-8 -*-

"""
THFE concrete-numpy implementation of the SHA-256 algorithm for fixed input length of 150 text characters
following official publication : http://csrc.nist.gov/publications/fips/fips180-4/fips-180-4.pdf
"""

import time
import hashlib
import numpy as np
import concrete.numpy as cnp
from utils import *

"""
SHA-256 constants
"""

# K256 as 32 bits integers
K256_32 = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]

# H as 32 bits integers
H_32_INITIAL = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]



def hexdigest(digest):
    """"
    Convert bytes (8 bits) to string of hex symbols (4 bits)
    """
    hexdigest = [0]*(2*len(digest));
    for i in range(0,len(digest)):
        d=digest[i]
        hexdigest[i*2] = (d >> 4) & 15
        hexdigest[i*2+1] = d & 15

    hex_chars = [hex(x)[-1] for x in hexdigest]
    return ''.join(hex_chars)



def padding_150(M):
    """
    SHA-256 padding for a message with fixed lenght of 150 characters (see section 5.1.1 in the paper)
    """
    assert(len(M)==150)
   
    """
    Length in bits is : l = 150*8 = 1200
    Find k such that l+k+1 = 448 mod 512 => this gives k = 271
    Convert l=1200 in 64 bits binary: l64 = 0000000000000000000000000000000000000000000000000000010010110000
    Convert l64 to an array of 8-bits integers, which is the format of the input message:
    l64_8 = [00000000,00000000,00000000,00000000,00000000,00000000,00000100,10110000] = [0,0,0,0,0,0,4,176]
    """
    l64_8 = np.array([0,0,0,0,0,0,4,176]).astype(np.int32)
    
    """
    Write 1 followed by k zeros in binary as an array of 8-bits integers:
    _1k0 = [ 10000000, 00000000, ..., 00000000]
    _1k0 = [128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    """
    _1k0 = np.array([128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).astype(np.int32)

    """
    The padded message is the concatenation of <M> <1 followed by k zeros> <l in 64 bits> 
    It has size 192 bytes = 192*8 bits = 3*512 bits = 3 blocks of 512 bits
    """
    return np.concatenate([M, _1k0, l64_8])



def sha256CircuitFactory(nbits, use_cnp=True, quick_test=False):

    assert(nbits in [1,2,4,8])
    nchunks = int(32/nbits)

    if use_cnp:
        zeros = lambda shape: cnp.zeros(shape)
        types = {1:cnp.uint1, 2:cnp.uint2, 4:cnp.uint4, 8:cnp.uint8}
        tensorTypes = (cnp.tensor[types[nbits], int(nchunks*192*8/32) ],
                       cnp.tensor[types[nbits], 8, nchunks],
                       cnp.tensor[types[nbits], 64, nchunks])
    else:
        # Warning, with numpy the inputs array H is modified by the circuit
        zeros = lambda shape: np.zeros(shape, dtype=np.uint32)
        tensorTypes = (np.ndarray, np.ndarray, np.ndarray)        

    #create functions for processing n-bits
    sigmas = Sigmas(nbits, use_cnp)
    add = AddSplit(nbits, use_cnp)

    # the computation being slow, a quick test can be made by setting these variables to 1
    N = 3 if not quick_test else 1
    Nt = 64 if not quick_test else 1  

    def sha256_150(M: tensorTypes[0], H: tensorTypes[1], K256: tensorTypes[2]):
        """
        SHA-256 implementation for a message with fixed lenght of 150 characters encoded as 150x8 binary values
        Returns a digest of 32 bytes
        """

        print('Compiling sha256_150...')

        """
        padding is already done, so apply parsing: making 3x16 32-bits words encoded as nchunks nbits integers
        """
        parsed_M = M.reshape((3,16,nchunks))

        """
        Then, proceed to the main computation (see section 6.2 in the paper)
        """

        #create W and a,b,c,d,e,f,g arrays
        W=zeros((64,nchunks))        
        a=zeros(nchunks); b=zeros(nchunks); c=zeros(nchunks); d=zeros(nchunks);
        e=zeros(nchunks); f=zeros(nchunks); g=zeros(nchunks); h=zeros(nchunks);

        #main loop
        for i in range(0,N):
            #1 prepare the message schedule W
            W[0:16,:]=parsed_M[i,:,:]
            for t in range(16,Nt):
                W[t] = add(add(sigmas.sigma1(W[t-2]), W[t-7]),
                           add(sigmas.sigma0(W[t-15]), W[t-16]))

            #2 initialize values of a,b,c,d,e,f,g,h with previous values in H
            a[:]=H[0]; b[:]=H[1]; c[:]=H[2]; d[:]=H[3]; e[:]=H[4]; f[:]=H[5]; g[:]=H[6]; h[:]=H[7];

            #3
            for t in range(0,Nt):
                T1 = add( add(h, sigmas.SIGMA1(e)), add( add(Ch(e,f,g),K256[t]),  W[t]))
                T2 = add(sigmas.SIGMA0(a), Maj(a,b,c))
                h[:]=g[:]; g[:]=f[:]; f[:]=e[:] # ! be sure to copy values with [:] and not the array objects
                e = add(d,T1)
                d[:]=c[:]; c[:]=b[:]; b[:]=a[:]
                a = add(T1,T2)

            if quick_test:
                # return (a,b,c,d,e,f,g,h) in H. Warning: the hash will be incorrect
                H[0,:]=a[:]; H[1,:]=b[:]; H[2,:]=c[:]; H[3,:]=d[:];
                H[4,:]=e[:]; H[5,:]=f[:]; H[6,:]=g[:]; H[7,:]=h[:];
                break

            #4 compute update of H
            H[0]=add(H[0],a); H[1]=add(H[1],b); H[2]=add(H[2],c); H[3]=add(H[3],d);
            H[4]=add(H[4],e); H[5]=add(H[5],f); H[6]=add(H[6],g); H[7]=add(H[7],h);

        """
        Finally, the result is the concatenation of the values of H
        """
        print('Done')    
        return H.reshape((8*nchunks,))

    # create this function to copy the value of H in python mode, otherwise it is modified by the circuit
    def circuit(M: tensorTypes[0], H: tensorTypes[1], K256: tensorTypes[2]):
        return sha256_150(M, H if use_cnp else H.copy(), K256)

    return circuit


def processInput(text, H, K256, nbits):
    """"
    Processes SHA-256 inputs for concrete-numpy circuit
    """    
    # convert text to uint8
    textAsInts = list(text)

    # apply padding 
    textAsInts = padding_150(textAsInts)

    # convert text from 8 bits to nbits
    textAsInts = ints_to_nbits(textAsInts, nbits, 8).flatten() # flatten this one
    # convert constants from 32 (default) bits to nbits
    H_nbits = ints_to_nbits(H_32_INITIAL, nbits)
    K256_nbits = ints_to_nbits(K256_32, nbits)

    return (textAsInts, H_nbits, K256_nbits)


def outputToHash(output, nbits):
    """"
    Processes the circuit output into a hexadecimal hash
    """      
    nchunks = int(32/nbits)

    # first convert output back to uint8 values
    output32 = ints_from_nbits(output.reshape((8,nchunks)),nbits,32)
    output8 = ints_to_nbits( output32, 8, 32)
    
    # also convert to hex
    outputHash = hexdigest(output8.flatten())

    return outputHash


def testFactory(nbits):
    """
    A factory returning a test function for nbits
    """

    nchunks=int(32/nbits)

    def test(text, use_cnp=True, quick_test=False):
        """
        A test computing the output of the circuit (trivial or encrypted depending on use_cnp)
        and comparing it with the output from hashlib.sha256 library
        """  

        # firstly, process inputs of the SHA-256 function to fit for our thfe circuit
        (textAsInts, H_nbits, K256_nbits) = processInput(text, H_32_INITIAL, K256_32, nbits)

        # create the circuit
        sha256circuit = sha256CircuitFactory(nbits, use_cnp, quick_test)
        
        print('\nTesting '+str(nbits)+'-bits circuit')
        start = time.time()

        if(use_cnp):
            # using encryption with concrete numpy
            configuration = cnp.Configuration(
                enable_unsafe_features=True,
                use_insecure_key_cache=True,
                insecure_key_cache_location=".keys",
            )

            compiler = cnp.Compiler(sha256circuit, {"M": "encrypted", "H": "encrypted", "K256": "encrypted"})
            circuit = compiler.compile(
                inputset=[
                    ( np.random.randint(0, 2**nbits, size=(int(192*8*nchunks/32),) ),
                      np.random.randint(0, 2**nbits, size=(8,nchunks)), 
                      np.random.randint(0, 2**nbits, size=(64,nchunks)) )
                    for _ in range(100)
                ],
                configuration=configuration,
                verbose=True,
            )

            circuit.keygen()
            end = time.time()
            print("keygen", int((end - start)*10)/10, "seconds")
            keygenDuration = end - start

            start = time.time()
            print('Encrypting...')
            encrypted = circuit.encrypt( textAsInts, H_nbits, K256_nbits )
            print('Running...')
            run = circuit.run(encrypted)
            print('Decrypting...')
            output=circuit.decrypt(run).flatten()
            print('Done')
             
        else:
            # no encryption, trivial computation with numpy
            output = sha256circuit(textAsInts, H_nbits, K256_nbits)
            keygenDuration=0

        end = time.time()
        duration = end-start
        print('Time: ', int(duration*10)/10, ' seconds') 

        # compute expected output with hashlib
        hasher = hashlib.sha256()
        hasher.update(text)
        expected8 = hasher.digest()

        # and convert it to output nbits integer format 
        expected = ints_to_nbits(expected8,nbits, 8).flatten()

        # Finally, convert both output and expected output into a hexadecimal hash
        outputHash = outputToHash(output, nbits)
        expectedHash = hasher.hexdigest()

        return (output, expected, duration, keygenDuration, outputHash, expectedHash)

    return test

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--nbits', nargs='+', type=int, help='list of nbits values among: 1, 2, 4, 8')
    parser.add_argument('--np', action='store_true', help='use numpy instead of concrete-numpy')
    parser.add_argument('--t', action='store_true', help='run a quick test by not looping inside the function')
    args = parser.parse_args()

    use_cnp = not args.np 
    quick_test = args.t

    if not args.nbits:
        args.nbits = [1,2,4,8]

    print('\nRunning tests for nbits in:', args.nbits)
    print('Using : ', 'numpy' if args.np else 'concrete-numpy')
 
    text = (
        b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        b"Curabitur bibendum, urna eu bibendum egestas, neque augue eleifend odio, et sagittis viverra."
    )  
    assert(len(text) == 150)

    def testResult(passed):
        if passed:
            print('\ntest PASSED !\n')
        else:
            print('\ntest FAILED\n')

    def showResults(n, res):
        print('\n======================')
        print('\ntest for '+str(n)+'-bits values:\n')

        print('output  :', res[0].tolist())
        if not quick_test:
            print('\nexpected:',res[1].tolist())
        print('\noutput hash  :',res[4])
        if quick_test:
            print('\nWARNING: quick test mode will lead to incorrect hash')

        if not quick_test:
            print('expected hash:',res[5])

        print('')   
        print(int(res[2]*10)/10, 'seconds for running')
        print(int(res[3]*10)/10, 'seconds for keygen')
        if not quick_test:   
            testResult(res[0].tolist() == res[1].tolist())

    """
    Run tests for several bit width and compare the results at the end
    """

    results = {}

    for nbits in args.nbits:
        test = testFactory(nbits)
        results[nbits] = test(text, use_cnp, quick_test)

    for nbits in args.nbits:
        showResults(nbits, results[nbits])
