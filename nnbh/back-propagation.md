# Backward Propagation

So: we know what a neural network is and we know that in principle we can apply the chain rule to estimate the gradients of the weights and the biases in the network. Estimating the gradients in this way is known as *back propagation* in the deep learning world, as the error in the network is propagated backward through the network.

There's nothing special about backpropogation though: it really is just an application of a general set of mathematical rules for computing the derivatives of a composed function to the specific case of neural networks.

## Reminder: Notation

- Superscript: layer
- Subscript: node within layer
- Two subscripts: first is node within layer, next is node within previous layer
- lower case: scalar
- lower case, bold: vector
- upper case, bold: matrix

## In Short: The Four Backpropagation Rules

In backpropagation, we need four rules. These are:

1. **Derivative of the output layer:** $ \boldsymbol{\delta}^{(L)} = \nabla_a C \odot \sigma_L'(\boldsymbol{z}^{(L)}) $
2. **Derivative of the hidden layer:** $ \boldsymbol{\delta}^{(l)} = (\boldsymbol{W}^{(l+1)})^T\boldsymbol{\delta}^{(l+1)} \odot \sigma_l'(\boldsymbol{z}^{(l)}) $
3. **Derivative of the biases layer:** $ \frac{\partial C}{\partial \boldsymbol{b}^{(l)}} = \sum^m_{x=1} \boldsymbol{\delta}^{(l)}_x / m$
4. **Derivative of the weights:** $ \frac{\partial C}{\partial \boldsymbol{W}^{(l)}} = \boldsymbol{\delta}^{(l)}(\boldsymbol{a}^{(l-1)})^T / m$

## Starting Points

Some general rules of thumb apply when considering backpropgation. First, for most cost functions $C()$, we compute the error for each observation then aggregate it at the end. Because of the sum rule of calculus, we can basically ignore this step untill the end. In other words, we're able to concentrate on the derivatives of the weights, biases, etc for a single observation, with the undesrtanding that at the end we'll need to take the mean for each one (assuming that the cost function includes a sum, followed by a division by the number of observations).

As we go through the process of back propagation, it's useful to recall that for each layer, we *first* compute a weighted input $\boldsymbol{Z}^{(l)}$, and *after* that we compute an element-wise activation $\boldsymbol{a}^{(l)}$. For the purpose of computing the gradients of the weights and the biases, it's useful to compute the gradient of the error at the *first* step. Notation-wise, we denote this as $\delta^{(l)} = \frac{\partial C}{\partial Z^{(l)}}$ for the vector of derviatives of the cose with respect to the weighted inputs of layer $l$. I'll often refer to this as the derviative with respect to layer $l$, which is fine - so long as we remember that it's realy the derivative with respect to layer $l$'s weighted input, and not with respect to its activation!

## 1. Derivative of the Output Layer

The derivative of the output layer is given in matrix form by

$$ \boldsymbol{\delta}^{(L)} = \nabla_a C \odot \sigma_L'(\boldsymbol{z}^{(L)}) $$

where $\nabla_a C$ is the vector of componenets $\frac{\partial C}{\partial a^{(L)}_j}$, $\sigma_L$ is the output layer activation function and $\sigma_L'$ is its derivative (with respect to the weighted input $\boldsymbol{z}^{(L)}$).

To arrive at this, let's start with the the formula for a single element of $\boldsymbol{\delta}^{(L)}$:

$$ \delta^{(L)}_j = \frac{\partial C}{\partial z^{(L)}_j} $$

Applying the chain rule from earlier, we get:

$$ \delta^{(L)}_j = \sum_k \frac{\partial C}{\partial a^{(L)}_k} \cdot \frac{\partial a^{(L)}_k}{\partial z^{(L)}_j} $$

Assuming that we our activation $a^{(L)}_k$ only depends on $Z^{(L)}_j$ when $j=k$ (which it will if the activation function is an element-wise function - which it often will be!) this simplifies to

$$ \delta^{(L)}_j = \frac{\partial C}{\partial a^{(L)}_j} \cdot \frac{\partial a^{(L)}_j}{\partial z^{(L)}_j} $$

Which is a single element of the matrrix form $\nabla_a C \odot \sigma_L'(\boldsymbol{z}^{(L)})$ (note that this is a case of element-wise operations simplifying things to a hadamard product, as mentioned earlier).

## 2. Derviative of Hidden Layers

The formula for the derivative in the hidden layer is given by

$$ \boldsymbol{\delta}^{(l)} = (\boldsymbol{W}^{(l+1)})^T\boldsymbol{\delta}^{(l+1)} \odot \sigma_l'(\boldsymbol{z}^{(l)}) $$

Once again, let's start with a single element of $\delta^{(l)}$:

$$ \delta^{(l)}_j = \frac{\partial C}{\partial z_j} $$

Applying the chain rule:

$$ \delta^{(l)}_j = \sum_k \frac{\partial C}{\partial z^{(l+1)}_k} \cdot \frac{\partial z^{(l+1)}_k}{\partial z^{(l)}_j} $$

Applying the original definition of $\delta^{(l)}_j$:

$$ \delta^{(l)}_j = \sum_k \delta^{(l+1)}_k \frac{\partial z^{(l+1)}_k}{\partial Z^{(l)}_j} $$

And rearranging:

$$ \delta^{(l)}_j = \sum_k \frac{\partial z^{(l+1)}_k}{\partial z^{(l)}_i} \delta^{(l+1)}_k $$

Recall that

$$ z^{(l+1)}_k = \sum_j w^{(l+1)}_{kj} a^{(l)}_j + b^{(l+1)}_{k} = \sum_j w^{(l+1)}_{kj} \sigma(z^{l}_j) + b^{(l+1)}_{k} $$

Differentiating for $z^{l}_j$, we obtain

$$ \frac{\partial z^{(l+1)}_k}{\partial z^{l}_j} = w^{(l+1)}_{kj} \sigma'(z^{l}_j) $$

Plugging this back into the formla for $\delta^{(l)}_j$ and rearranging:

$$ \delta^{(l)}_j = \sum_k w^{(l+1)}_{kj} \delta^{(l+1)}_k \sigma'(z^{l}_j) $$

Which is a single element of the matrix form $\boldsymbol{\delta}^{(l)} = (\boldsymbol{W}^{(l+1)})^T\boldsymbol{\delta}^{(l+1)} \odot \sigma_l'(\boldsymbol{z}^{(l)})$

## 3. Derivative of the Biases

$$ \frac{\partial C}{\partial \boldsymbol{b}^{(l)}} = \boldsymbol{\delta}^{(l)} $$

Begin with a single component of the vector:

$$ \delta^{(l)}_j = \frac{\partial C}{\partial b^{(l)}_j}  $$

Apply the chain rule:

$$ \frac{\partial C}{\partial b^{(l)}_j} = \sum_k \frac{\partial C}{\partial z^{(l)}_k} \cdot \frac{\partial z^{(l)}_k}{\partial b^{(l)}_j} $$

$z^{(l)}_k$ will only depend on $b^{(l)}_j$ when $k=j$, so:

$$ \frac{\partial C}{\partial b^{(l)}_j} = \frac{\partial C}{\partial z^{(l)}_j} \cdot \frac{\partial z^{(l)}_j}{\partial b^{(l)}_j} $$

Based on the definition of $z^{(l)}_j$:

$$ \frac{\partial C}{\partial b^{(l)}_j} = \frac{\partial C}{\partial z^{(l)}_j} \cdot \frac{\partial \sum_k w^{(l)}_{jk} a^{(l-1)}_k + b^{(l)}_j}{\partial b^{(l)}_j} $$

Since the weights $w^{(l)}$ and activations $a^{(l-1)}_k$ do not dependent on $b^{(l)}_j$, the second term becomes 1 and thus:

$$ \frac{\partial C}{\partial b^{(l)}_j} = \frac{\partial C}{\partial z^{(l)}_j} = \delta^{(l)}_j$$

Which is a single component of the vector $\boldsymbol{\delta}^{(l)}$

## 4. Derivative of the Weights

$$ \frac{\partial C}{\partial \boldsymbol{W}^{(l)}} = \boldsymbol{\delta}^{(l)}(\boldsymbol{a}^{(l-1)})^T $$

Begin by applying the chain rule:

$$ \frac{\partial C}{\partial w^{(l)}_{ji}} = \sum_k \frac{\partial C}{\partial z^{(l)}_{k}}\cdot\frac{\partial z^{(l)}_{k}}{\partial w^{(l)}_{ji}} $$

Since $z^{(l)}_{k}$ depends on $w^{(l)}_{ji}$ only when $k=j$, this simplifies to

$$ \frac{\partial C}{\partial w^{(l)}_{ji}} = \frac{\partial C}{\partial z^{(l)}_{j}}\cdot\frac{\partial z^{(l)}_{j}}{\partial w^{(l)}_{ji}} $$

Apply the definition of $\delta^{(l)}_j$:

$$ \frac{\partial C}{\partial w^{(l)}_{ji}} = \delta^{(l)}_j \cdot \frac{\partial z^{(l)}_{j}}{\partial w^{(l)}_{ji}} $$

Apply the definition of $z^{(l)}_{j}$:

$$ \frac{\partial C}{\partial w^{(l)}_{ji}} = \delta^{(l)}_j \cdot \frac{\partial \sum_k w^{(l)}_{jk}a^{(l-1)}_{k} + b^{l}_j}{\partial w^{(l)}_{ji}} $$

Since $w^{(l)}_{ij}$ does not depend on any other $w^{(l)}_{ik}$, this simplifies to

$$ \frac{\partial C}{\partial w^{(l)}_{ji}} = \delta^{(l)}_j \cdot \frac{\partial w^{(l)}_{ji}a^{(l-1)}_{i} + b^{l}_j}{\partial w^{(l)}_{ji}} $$

which is solved as:

$$ \frac{\partial C}{\partial w^{(l)}_{ji}} = \delta^{(l)}_j a^{(l-1)}_{i} $$

## Moving to Multiple Observations

- until now, have simplified by ignoring the need to average over observations
  - this is normally part of the cost function
- by the derivative rules of constant multiplication and summation, we can still just average over these

Rules 1 and 2 don't need to change

Rules 3 and 4 do need to change. In the first, you sum and then divide by m

In the second, the matrix multiplication will handle the required summation. So you just divide by m as extra
