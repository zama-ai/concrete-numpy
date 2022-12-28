# Fully Homomorphic Database
A fully homomorphic database built using concrete-numpy.

## Notes
`concrete-numpy` does not yet support performing operations on the results from circuits. As such, it's necessary to decrypt the results from computation then re-encrypt them before passing output into the next circuit. Because of this, I separate the time spent computing on FHE data from the time spent encrypting and decrypting values. Inside the code, I mark the sections where I require the ability to operate on the results of circuits. 
