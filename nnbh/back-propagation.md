# Backward Propagation

So: we know what a neural network is and we know that in principle we can apply the chain rule to estimate the gradients of the weights and the biases in the network. Estimating the gradients in this way is known as *back propagation* in the deep learning world, as the error in the network is propagated backward through the network.

There's nothing special about backpropogation though: it really is just an application of a general set of mathematical rules for computing the derivatives of a composed function to the specific case of neural networks.

## Some notations

##

As we go through the process of back propagation, it's useful to recall that for each layer, we *first* compute a weighted input $\boldsymbol{Z}^{(l)}$, and *after* that we compute an element-wise activation $\boldsymbol{a}^{(l)}$.
