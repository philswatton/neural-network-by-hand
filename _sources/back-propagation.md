# Backward Propagation

So: we know what a neural network is and we know that in principle we can apply the chain rule to estimate the gradients of the weights and the biases in the network. Estimating the gradients in this way is known as *back propagation* in the deep learning world, as the error in the network is propagated backward through the network.

There's nothing special about backpropogation though: it really is just an application of a general set of mathematical rules for computing the derivatives of a composed function to the specific case of neural networks.

## Notation

There is a lot of notation in what follows on this page. They can be divided into a few categories. This is in part a recap of some of the notations introduced in the previous page.

### Subscripts  and Superscripts

- Superscript: layer
- Subscript: node within layer
- Two subscripts: first is node within layer, next is node within previous layer

### Scalars, vectors, and matrices

- lower case: scalar
- lower case, bold: vector
- upper case, bold: matrix

Note that the lower/upper convention will mostly hold below, although it will be ignored for $\delta$.

### Symbols and their Meanings

- $n^{(l)}$, the number of nodes in layer $l$. Note that the input layer is indexed by 0
- $m$, the number of training examples in the dataset
- $L$, the number of layers in the network (is equal to the number of hidden layers + 1. The input layer is not counted in this and is indexed by 0)
- $\boldsymbol{X}$, the matrix of input data. Has dimensions $n^{(0)} \times m$, with features on rows and training examples on columns
- $\boldsymbol{W}^{(l)}$, the matrix of weights for layer $l$. Has dimensions $n^{(l)} \times n^{(l-1)} $.
  - The individual elements are $w^{(l)}_{ji}$, where $j$ is the node in layer $l$ the weight maps to and $i$ is the node in layer $l-1$ the weight maps from
- $\boldsymbol{b}^{(l)}$, the vector of biases for layer $l$. Is a column vector with $n^{(l)}$ elements
  - The individual elements are $b^{(l)}_{j}$, where $j$ is the node in layer $l$ the bias corresponds to
- $\boldsymbol{Z}^{(l)}$, the matrix of weighted inputs for layer $l$. Has dimensions $n^{(l)} \times m $
  - The individual elements are $z^{(l)}_{j}$, where $j$ is the node in layer $l$ the weighrd input corresponds to
- $\sigma^{(l)}()$, the activation function for layer $l$
- $\sigma^{{'}(l)}()$, the activation function for layer $l$
- $\boldsymbol{A}^{(l)}$, the matrix of outputs of the activation function for layer $l$. Has dimensions $n^{(l)} \times m $
  - The individual elements are $a^{(l)}_{j}$, where $j$ is the node in layer $l$ the activation output corresponds to
- $C$, the overall cost of the network. Sometimes I will also refer to $C()$ as the cost *function* of the network.
- $\boldsymbol{\delta}^{(l)}$ is shorthand for the derivative of the cost with respect to the weighed input. That is, $\boldsymbol{\delta}^{(l)} = \frac{\partial C}{\partial \boldsymbol{Z}^{(l)}}$
  - The individual elements are $\delta^{(l)}_jx = \frac{\partial C}{\partial z_{jx}}$, where $j$ is the node in layer $l$ the activation output corresponds to and $x$ indexes the training example/observation
  - Note that I will often refer to this as *the derivative in layer* $l$. This shorthand is fine, so long as we all understand it's the derivative at the weighted input and not at the activation!
- $\nabla_a C$ is shorthand for the derivative of the cost with respect to the activations in the output layer. That is, $\nabla_a C = \frac{\partial C}{\partial \boldsymbol{A}^{(L)}}$

## In Short: The Four Backpropagation Rules

In backpropagation, we need four rules. These are:

1. **Derivative of the output layer:** $ \boldsymbol{\delta}^{(L)} = \nabla_a C \odot \sigma_L'(\boldsymbol{z}^{(L)}) $
2. **Derivative of the hidden layer:** $ \boldsymbol{\delta}^{(l)} = (\boldsymbol{W}^{(l+1)})^T\boldsymbol{\delta}^{(l+1)} \odot \sigma_l'(\boldsymbol{z}^{(l)}) $
3. **Derivative of the biases layer:** $ \frac{\partial C}{\partial \boldsymbol{b}^{(l)}} = \sum^m_{x=1} \boldsymbol{\delta}^{(l)}_x / m$
4. **Derivative of the weights:** $ \frac{\partial C}{\partial \boldsymbol{W}^{(l)}} = \boldsymbol{\delta}^{(l)}(\boldsymbol{A}^{(l-1)})^T / m$

## Recap: Previous Page

As we go through the process of back propagation, it's useful to recall that for each layer, we *first* compute a weighted input $\boldsymbol{Z}^{(l)}$, and *after* that we compute an element-wise activation $\boldsymbol{A}^{(l)}$.

## Starting Point: Simplification

Some general rules of thumb apply when considering backpropgation. First, for most cost functions $C()$, we compute the error for each training example, then sum the error and divide it by $m$. Because of the sum and line rules of calculus, we can basically ignore this step untill the end.

In other words, we're able to concentrate on the derivatives of the weights, biases, etc for a single training example, with the undesrtanding that at the end we'll need to take the mean for each one (assuming that the cost function includes a sum, followed by a division by the number of training example).

This means that $m=1$, and thus some matrices instead become column vectors: e.g. $\boldsymbol{z}^{(l)}$ instead of $\boldsymbol{Z}^{(l)}$, etc. In general, if you spot a change of case from upper to lower, that's what's going on!

## 1. Derivative of the Output Layer

The derivative of the output layer is given in matrix form by

$$ \boldsymbol{\delta}^{(L)} = \nabla_a C \odot \sigma_L'(\boldsymbol{z}^{(L)}) $$

where $\nabla_a C$ is the vector of componenets $\frac{\partial C}{\partial a^{(L)}_j}$, $\sigma_L$ is the output layer activation function and $\sigma_L'$ is its derivative (with respect to the weighted input $\boldsymbol{z}^{(L)}$).

To arrive at this, let's start with the the formula for a single element of $\boldsymbol{\delta}^{(L)}$:

$$ \delta^{(L)}_j = \frac{\partial C}{\partial z^{(L)}_j} $$

Since the intermediary variables between $z^{(L)}_j$ and $C$ are the $a^{(L)}_k$, we can apply the chain rule:

$$ \delta^{(L)}_j = \sum_k \frac{\partial C}{\partial a^{(L)}_k} \cdot \frac{\partial a^{(L)}_k}{\partial z^{(L)}_j} $$

Assuming that we our activation $a^{(L)}_k$ only depends on $Z^{(L)}_j$ when $j=k$ (which it will if the activation function is an element-wise function - which it often will be!) this simplifies to

$$ \delta^{(L)}_j = \frac{\partial C}{\partial a^{(L)}_j} \cdot \frac{\partial a^{(L)}_j}{\partial z^{(L)}_j} $$

Which implies that all we need to do is multiply two elements. This means that we can represent this in matrix form as $\nabla_a C \odot \sigma_L'(\boldsymbol{z}^{(L)})$ (note that here the derivatives in the sum disappearing and the element-wise operations simplifying things to a hadamard product is the same thing!).

## 2. Derviative of the Hidden Layers

The formula for the derivative in the hidden layer is given by

$$ \boldsymbol{\delta}^{(l)} = (\boldsymbol{W}^{(l+1)})^T\boldsymbol{\delta}^{(l+1)} \odot \sigma_l'(\boldsymbol{z}^{(l)}) $$

Once again, let's start with a single element of $\delta^{(l)}$:

$$ \delta^{(l)}_j = \frac{\partial C}{\partial z_j} $$

We can use $z^{(l)}_k$ as the intermediary variables between $z^{(l)}_j$ and $C$. We then apply the chain rule:

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

Here we have a dot product (summed multiplication) and another multiplication. We can transform the first part into a matrix multiplication and the second we can multiply in via a Hadamard product to get the overall matrix form:

$$ \boldsymbol{\delta}^{(l)} = (\boldsymbol{W}^{(l+1)})^T\boldsymbol{\delta}^{(l+1)} \odot \sigma_l'(\boldsymbol{z}^{(l)}) $$

## 3. Derivative of the Biases

The formula for the derivative of the biases is given by:

$$ \frac{\partial C}{\partial \boldsymbol{b}^{(l)}} = \boldsymbol{\delta}^{(l)} $$

We can use $z^{(l)}_k$ as the intermediary variables between $b{(l)}_j$ and $C$. We then apply the chain rule:

$$ \frac{\partial C}{\partial b^{(l)}_j} = \sum_k \frac{\partial C}{\partial z^{(l)}_k} \cdot \frac{\partial z^{(l)}_k}{\partial b^{(l)}_j} $$

$z^{(l)}_k$ will only depend on $b^{(l)}_j$ when $k=j$, so the other terms become 0 and drop out of the sum:

$$ \frac{\partial C}{\partial b^{(l)}_j} = \frac{\partial C}{\partial z^{(l)}_j} \cdot \frac{\partial z^{(l)}_j}{\partial b^{(l)}_j} $$

Based on the definition of $z^{(l)}_j$:

$$ \frac{\partial C}{\partial b^{(l)}_j} = \frac{\partial C}{\partial z^{(l)}_j} \cdot \frac{\partial \sum_k w^{(l)}_{jk} a^{(l-1)}_k + b^{(l)}_j}{\partial b^{(l)}_j} $$

Since the weights $w^{(l)}$ and activations $a^{(l-1)}_k$ do not dependend on $b^{(l)}_j$, the second term becomes 1 and thus:

$$ \frac{\partial C}{\partial b^{(l)}_j} = \frac{\partial C}{\partial z^{(l)}_j} = \delta^{(l)}_j$$

Which by definition is a single component of the vector $\boldsymbol{\delta}^{(l)}$

## 4. Derivative of the Weights

The formula for the derivative of the weights is given by:

$$ \frac{\partial C}{\partial \boldsymbol{W}^{(l)}} = \boldsymbol{\delta}^{(l)}(\boldsymbol{a}^{(l-1)})^T $$

We can once again use $z^{(l)}_k$ as the intermediary variables between $w{(l)}_{ji$ and $C$. We then apply the chain rule:

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

Which can be expressed in matrix form as:

$$ \boldsymbol{\delta}^{(l)}\boldsymbol{a}^{(l)T} $$

## Moving to Multiple Observations

Earlier on, we abstracted away the fact that our cost function will almost certainly involve some summing and division in order to average the derviatve over training examples. To produce the final backpropagation formulas, we need to consider this.

First, rules 1 and 2 don't need to change! Their dimensions will always be $n^{(l)} \times m$. Thus as $m$ increases, their dimensions will remain correct.

It's at rules 3 and 4 we need to make some adjustments though. For rule 3, we need to sum the derivatives of the bias, then divide them by $m$:

$$ \frac{\partial C}{\partial \boldsymbol{b}^{(l)}} = \sum^m_{x=1} \boldsymbol{\delta}^{(l)}_x / m $$

For rule 4, the sumation part is implicitly handled by the fact we use a matrix multiplication. We still need to remember to divide by $m$ though:

$$ \frac{\partial C}{\partial \boldsymbol{W}^{(l)}} = \boldsymbol{\delta}^{(l)}(\boldsymbol{A}^{(l-1)})^T / m $$

These are the only adjustments we need to make the rules ready for our own neural network. All the rest stays the same!
