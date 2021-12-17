# Benchmarks

To track our progress over time, we have created a [progress tracker](https://progress.zama.ai) that:
- lists targets that we want to compile
- updates the status on the compilation of these functions
- tracks compilation and evaluation times on different hardware
- displays accuracy of the functions for which it makes sense
- displays loss of the functions for which it makes sense

Note that we are not limited to these, and we'll certainly add more information (e.g., key generation time, encryption time, inference time, decryption time, etc.) once the explicit inference API is available.

```{warning}
FIXME(all): update the sentence above when the encrypt, decrypt, run_inference, keygen API's are available
```

Our public benchmarks can be used by competing frameworks or technologies for comparison with **Concrete Framework**. Notably, you can see:
- if the same functions can be compiled
- what are the discrepancies in the exactness of the evaluations
- how do evaluation times compare

If you want to see more functions in the progress tracker or if there is another metric you would like to track, don't hesitate to drop an email to <hello@zama.ai>.