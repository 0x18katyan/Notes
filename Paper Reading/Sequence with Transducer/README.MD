# Notes

Paper: [Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/pdf/1211.3711.pdf)

External Sources:

- [Sequence-to-sequence learning with Transducers](https://lorenlugosch.github.io/posts/2020/11/transducer/)

## Summary

Transducer loss is constructing different pathways for the target akin to CTC with blanks but instead of letting the CTC loss define the alignment between a target $Y_U$ to the $X_T$, it constructs every possible combination of words with blanks for the alignment.

This approach results in a very large target space which requires more memory than usual for computing the backpropagation, which might be the reasons that big tech companies that have massive GPU memory are usually the ones propelling the research on transducer networks. *ahem Google ahem*
