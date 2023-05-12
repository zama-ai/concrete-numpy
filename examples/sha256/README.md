## SHA-256 implementation in concrete-numpy

#### Tutorial
A jupyter notebook tutorial is available [here](https://github.com/Lcressot/concrete-numpy/blob/main/docs/tutorial/sha256.ipynb)

#### Content
- The official [publication](http://csrc.nist.gov/publications/fips/fips180-4/fips-180-4.pdf) of SHA algorithm
- A numpy implementation of SHA-256 following the exact steps and notations in `sha256_original.py` for a text input with fixed length
- The concrete-numpy implementation of this algorithm in `sha256.py`and `utils.py`

#### Tests
- `python3 sha256_original.py` will test the algorithm against hashlib 
- `python3 utils.py` will test the functions defined in the file
- `python3 sha256.py --help` will show how to use the command line arguments: `--nbits`the bitwidth of inputs, `--np`to use numpy instead of concrete-numpy, and `--t` to make a quick test with encryption, giving fast but incorrect hash

#### Usage
See tests in `sha256_original.py` and `sha256.py`


#### Notes
1. The full computation being too slow, the author could not confirm in time that the full circuit works as expected in `sha256.py`. The circuit works in numpy mode as expected against hashlib, and the encrypted and numpy modes give the same result hash in quick test mode. Thus, there are very good chances that the full circuit is correct, but some troubleshooting may be needed.  

3. The number of bits in the inputs of `sha256.py` seems to affect drastically the speed of computation, the 1-bit and 2-bits versions being by far the fastest.
