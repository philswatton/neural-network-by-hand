# Activation Functions and their Derivatives

WIP

## Sigmoid

- Ranges between 0 and 1
- Good choice for hidden layer when all the input variables are positive

$$ \sigma(x) = \frac{1}{1 - e^{-x}}$$

$$ \sigma'(x) = \sigma(x) \times \sigma(1-x) $$

## Tanh

- Ranges between -1 and 1

$$ \sigma(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} $$

$$ \sigma'(x) = 1 - \sigma(x)^2 $$

## ReLU

- Very popular choice
- Note derivative technically does not exist as it is not a continuous function. In practice we ignore this

$$ \sigma(x) = \begin{cases}
    x, & \text{if } x \geq 0\\
    0, & \text{otherwise}
\end{cases} $$

$$ \sigma'(x) = \begin{cases}
    1, & \text{if } x \geq 0\\
    0, & \text{otherwise}
\end{cases} $$

## Linear

- Should only ever be used in the output layer for continuous outputs

$$ \sigma(x) = x $$

$$ \sigma'(x) = 1 $$
