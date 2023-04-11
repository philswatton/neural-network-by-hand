# Softmax and Cross-Entropy

WIP

So far, we've only considered element-wise activation functions that drastically simplify the process of deriving the four rules of backpropagation.

But this has been possible because we've considered continuous and binary outputs (note that we can have many of these - the issue is not the number of outputs per se!).

But what happens when we want to do multiclass classification? In other words, we want to predict only one out of a set of possible labels for each training example.

For this, we use the *softmax* activation function.

## Softmax

If we have $n$ possible labels to chose from, then the softmax activation function for the $j$th label value is given by:

$$ \sigma(x_j) = \frac{e^{x_j}}{\sum_k^n e^{x_k}} $$

This is very similar to the sigmoid activation function (indeed, those with a statistics background will recognise the multnomial logit generalisation of logit regression models here). Unlike other activation functions though, $x_j$ depends on each $x_k$ even when $j=k$ is not satisfied.

## Cross-Entropy

Since we have many classes, we also need a cost function to handle this case. As hinted by the name of 'binary cross-entropy', there is also a 'cross-entropy' function that is appropriate to our needs.

The formula for cross-entropy is given by:

$$ C(\boldsymbol{y}, \hat{\boldsymbol{y}}) = \sum_k -y_k log(\hat{y}_k) $$

Which is recognisably a generalisation of the binary case of binary cross-entropy.
