# -*- coding: utf-8 -*-

"""
This code does not use THFE, rather is a straightforward implementation of the SHA-256 algorithm from the official publication
See paper at http://csrc.nist.gov/publications/fips/fips180-4/fips-180-4.pdf
It thus follows the same notations.

The operation symbols in the paper have the following code implementation:

    ∧ Bitwise AND operation : &
    ∨ Bitwise OR (“inclusive-OR”) operation: |
    ⊕ Bitwise XOR (“exclusive-OR”) operation: ^
    ¬ Bitwise complement operation: ! for booleans
    + Addition modulo 2**w : see function mod2p32
    >> Right-shift operation, where x >> n is obtained by discarding the rightmost n bits of the word x
        and then padding the result with n zeroes on the left : casting to n bits and >> (see function SHR)
    << Left-shift operation, where x << n is obtained by discarding the left-most n bits of the word x
        and then padding the result with n zeroes on the right : << and casting to n bits (see function SHL)        
"""

import hashlib
import numpy as np

"""
SHA-256 constants
"""

K256 = np.array([
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]).astype(np.int32)

H_INITIAL = np.array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]).astype(np.int32)


"""
Original functions
"""

TP32=2**32
BITS_PER_WORD = 32

def SHR(x, n):
    """
    The right shift operation.
    Cast x to 32 bits and then shift it of n bits to the right
    """
    return (x & (TP32-1)) >> n

def SHL(x, n):
    """
    The left shift operation.
    Shift x of n bits to the left and then cast it to 32 bits
    """
    return (x << n) & (TP32-1)

def ROTR(x, n):
    """
    The rotate right (circular right shift) operation.
    It is a union of a right shift of n bits and a left shit of w-n bits
    """
    return SHR(x,n) | SHL(x,BITS_PER_WORD-n)


def Ch(x, y, z):
    """
    The choose function
    """
    return z ^ (x & (y ^ z))


def Maj(x, y, z):
    """
    The majority function
    """
    return ((x | y) & z) | (x & y)


def SIGMA0(x):
    """
    Upper case sigma 0 function
    """     
    return ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22)


def SIGMA1(x):
    """
    Upper case sigma 1 function
    """      
    return ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25)


def sigma0(x):
    """
    Lower case sigma 0 function
    """
    return ROTR(x, 7) ^ ROTR(x, 18) ^ SHR(x, 3)


def sigma1(x):
    """
    Lower case sigma 1 function
    """      
    return ROTR(x, 17) ^ ROTR(x, 19) ^ SHR(x, 10)


def mod2p32(x):
    """
    Computes x modulo 2**32 which is equivalent to cast x to 32 bits
    """ 
    return x & (TP32-1)


"""
Custom functions
"""

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


def parsing_150(M):
    """
    SHA-256 parsing for a padded message with initial fixed lenght (before padding) of 150 characters
    (see section 5.2.1 in the paper)
    """    
    assert(len(M)==192); # after padding, length is now 192
    N=3
    """
    We split M into 3 x 64 bytes (3 x 512 bits) and split each 512 bits into 16 words of 32 bits
    """

    # initialize array
    parsed_M = np.zeros([N,16])

    # loop through blocks
    for i in range(0,N):
        # loop through words
        for j in range(0,16):
            # Convert groups of 4 bytes (8-bits integers) into 32-bits integers
            ind = 64*i + j*4
            parsed_M[i][j]= (M[ind] << 24) ^ (M[ind+1] << 16) ^ (M[ind+2] << 8) ^ M[ind+3]

    return parsed_M


def sha256_150(M):
    """
    SHA-256 implementation for a message with fixed lenght of 150 characters
    Returns a digest of 32 bytes
    """
    N=3

    """
    First, apply padding and parsing
    """
    M = padding_150(M);
    M = parsing_150(M);

    """
    Then, proceed to the main computation (see section 6.2 in the paper)
    """

    # initialize array H
    H=np.zeros(8,dtype=np.int32);
    for i in range(0,8):
        H[i] = H_INITIAL[i];
    

    # main loop
    for i in range(0,N):
        #1 prepare the message schedule W
        W=np.zeros(64,dtype=np.int32);
        W[0:16]=M[i,:]
        for t in range(16,64):
            W[t] = mod2p32(sigma1(W[t-2]) + W[t-7] + sigma0(W[t-15]) + W[t-16])

        #2 initialize values of a,b,c,d,e,f,g,h with previous values in H
        a=H[0]; b=H[1]; c=H[2]; d=H[3]; e=H[4]; f=H[5]; g=H[6]; h=H[7];

        #3
        for t in range(0,64):
            # use mod2p32 to compute the addition modulo 2**32
            T1= mod2p32(h + SIGMA1(e) + Ch(e,f,g) + K256[t] + W[t])
            T2= mod2p32(SIGMA0(a) + Maj(a,b,c))
            h=g; g=f; f=e
            e= mod2p32(d + T1)
            d=c; c=b; b=a
            a= mod2p32(T1+T2)

        #4 compute update of H
        H[0]=mod2p32(H[0]+a); H[1]=mod2p32(H[1]+b); H[2]=mod2p32(H[2]+c); H[3]=mod2p32(H[3]+d);
        H[4]=mod2p32(H[4]+e); H[5]=mod2p32(H[5]+f); H[6]=mod2p32(H[6]+g); H[7]=mod2p32(H[7]+h);

    """
    Finally, the result is the concatenation of the bits of H
    """
    # The result is the concatenation of bits of H, which are 8 x 32 bits
    # 8 x 32 bits is also 32 x 8 bits which is 32 bytes
    digest = np.zeros(32,dtype=np.short);
    for i in range(0,8):
        h=H[i]
        # split 32-bits integers into 4 x 8-bits integers
        for j in range(0,4):
            digest[i*4+j] = (h >> (8*(3-j))) & 255

    return digest


def test():

    text = (
        b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        b"Curabitur bibendum, urna eu bibendum egestas, neque augue eleifend odio, et sagittis viverra."
    )
    assert len(text) == 150

    hasher = hashlib.sha256()
    hasher.update(text)

    message = list(text)
    expected_output = list(hasher.digest())

    # test against hashlib 
    message = np.array(message).astype(np.int32)
    hashedMessage=sha256_150(message)
    assert(hashedMessage.tolist() == expected_output);

    # also test hex format
    assert(hexdigest(hashedMessage) == hasher.hexdigest());

    print('Tests PASSED !');

if __name__ == "__main__":
    test()
