# The Chain Rule

The chain rule is used in calculus to compute the derivatives of composite functions. Since neural networks are really just composite functions, this becomes particularly important for computing the gradients for use during gradient descent.

## Function Composition

We start with two terms expressing two functions:

$$ z = g(x) $$

and:

$$ y = f(g(x)) $$

The second term represents a *composite* function. In other words, it's a combination of two functions.

A composite function is sometimes written as:

$$ f(g(x)) = (f \circ g)(x) $$

Not to be confused with the general notation for binary operations used in this notebook ($\bigcirc$).

Note that $ y = x^2 + x $ is *not* a composite function. To be a composition function, the output of one function *must* be supplied as the input to another function. In other words, it must be expressable as $f(g(x))$.

## Scalar Chain Rule

For scalar values and functions, the chain rule can be given as:

$$ \frac{dy}{dx} = \frac{dy}{dz} \frac{dz}{dx} $$
