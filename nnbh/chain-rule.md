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

$$ \frac{dy}{dx} = \frac{dy}{dz} \cdot \frac{dz}{dx} $$

where $y = f(g(x))$ and $z = g(x)$. In other words, we take the derivative at each step, and multiply them.

Note that it has been useful here to introduce $z$ as an intermediary variable in the equation. It's important to realise the $z$ 'joins up' $y$ and $x$, and hence is the value that appears twice in the equation.

## Total Derviative Chain Rule

As you'll soon see later on, when we pass a value $x$ through a nueral network it isn't just passed through a single function at a time. Instead, it's passed through many functions in one go, which are then used to produce a final output.

This might look something like this:

$$ y = f(x, a(x), b(x), c(x)) $$

```{note}
To give the total derivative chain rule in full form, I've included $x$ as an input to $f$. In *feedforward* neural networks, this won't be the case: $x$ will either enter directly or indirectly into the function, but not both. I'll be showing you what happens to it for the sake of completeness here, but it won't be used later on.
```

To begin, let's set some intermediary values:

$$ z_1 = a(x) $$
$$ z_2 = b(x) $$
$$ z_3 = c(x) $$

Intuitively, the derivative of $y$ with respect to $x$ describes how $y$ changes as $x$ changes. In line with this notion, we can just sum the partial derivatives with respect to $x$:

$$ \frac{dy}{dx} = \frac{\partial y}{\partial x} \cdot \frac{dx}{dx} + \frac{\partial y}{\partial z_1} \cdot \frac{\partial z_1}{\partial x} + \frac{\partial y}{\partial z_2} \cdot \frac{\partial z_2}{\partial x} + \frac{\partial y}{\partial z_3} \cdot \frac{\partial z_3}{\partial x} $$

Since the derivative of $x$ with respect to itself will be 1, this simplifes to:

$$ \frac{dy}{dx} = \frac{\partial y}{\partial x} + \frac{\partial y}{\partial z_1} \cdot \frac{\partial z_1}{\partial x} + \frac{\partial y}{\partial z_2} \cdot \frac{\partial z_2}{\partial x} + \frac{\partial y}{\partial z_3} \cdot \frac{\partial z_3}{\partial x} $$

And that's really all there is to the total derivative chain rule! We can write a general form of the total derivative chain rule as:

$$ \frac{dy}{dx} = \frac{\partial y}{\partial x} + \sum_k \frac{\partial y}{\partial z_k} \cdot \frac{\partial z_k}{\partial x} $$

Which is the form we'll be using in deriving some equations later on in this document.
