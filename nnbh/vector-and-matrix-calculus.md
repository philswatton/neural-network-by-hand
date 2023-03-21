# Vector and Matrix Calculus

## Motivation: Partial Derivatives

Consider a function $f$ with two input variables, $x$ and $z$:

$$ f(x,z) = 3x^2z $$

We can take the partial derivative with respect to $x$:

$$ \frac{\partial}{\partial x} = 6xz $$

And we can take the partial derivative with respect to $z$:

$$ \frac{\partial f(x,z)}{\partial z} = 3x^2 $$

If we wish to organise our partial derivatives, we can store them in a vector, which we call the **gradient** of $f(x,z)$:

$$ \triangledown f(x,z) = \begin{bmatrix} 6xz \\ 3x^2 \end{bmatrix} $$

In general, we can move the use of vectors (and matrices) when we wish to start organising our derivatives in this fashion. In this light, vector and matrix calculus represent a notational system for **multivariable calculus**.

## Generalising the Gradient

In the case above, if we place $x$ and $z$ in a vector, we are taking the derivative of a scalar with respect to a vector. The outcome is two partial derivatives, which are also a vector. We can generalise this to say that the derivative of a scalar $y$ with respect to an $n$-length vector $\boldsymbol{x}$ will also be an $n$-length vector of derivatives:

$$ \frac{\partial y}{\partial \boldsymbol{x}} = \begin{bmatrix} \frac{\partial y}{\partial x_1} \\ \vdots \\ \frac{\partial y}{\partial x_n} \end{bmatrix} $$

We call this vector the *gradient* of $y$.

## Motivation: Vectors and Vectors

We now consider the case where in addition to $f(x,z)$ above, we have a second function $g(x,y)$:

$$ g(x,z) = 2x + y^8 $$

As before, we store the partial derivatives of $g(x,z)$ in a vector:

$$ \triangledown g(x,z) = \begin{bmatrix} \frac{\partial g(x,z)}{\partial x} \\ \frac{\partial g(x,z)}{\partial z} \end{bmatrix} $$

However, in combination with $f(x,z)$ we now have two vectors of partial derivatives. If we store $f(x,z)$ in a vector with $g(x,z)$, then take the derivative of this vector with respect to the vector storing $x$ and $z$, we obtain a matrix $\boldsymbol{J}$:

$$
\boldsymbol{J} = \begin{bmatrix} \triangledown f(x,z) \\ \triangledown g(x,z) \end{bmatrix} = \begin{bmatrix} \frac{\partial f(x,z)}{\partial x} & \frac{\partial f(x,z)}{\partial z} \\ \frac{\partial g(x,z)}{\partial x} & \frac{\partial g(x,z)}{\partial z} \end{bmatrix} = \begin{bmatrix} 6xz & 3x^2 \\ 2 & 8z^7 \end{bmatrix}
$$

This matrix is known as the **Jacobian**.

Note that here, the Jacobian is presented in the *numerator layout* (i.e numerator on rows). However, in some contexts it will be presented in *denominator layout*.

## Generalising the Jacobian

As before, we can generalise the notion of deriving a vector with respect to a vector beyond the simple example.

Suppose we have an $n$-length vector of functions $\boldsymbol{f}$, and an $m$-length vector of variables $\boldsymbol{x}$. The derivative of $\boldsymbol{f}$ with respect to $\boldsymbol{x}$ is then:

$$ \boldsymbol{J} = \begin{bmatrix} \frac{\partial \boldsymbol{f}}{\partial x_1} & \cdots & \frac{\partial \boldsymbol{f}}{\partial x_n} \end{bmatrix} = \begin{bmatrix} \triangledown^T f_1 \\ \vdots \\ \triangledown^T f_m \end{bmatrix} = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{bmatrix}  $$

Where $\boldsymbol{J}$ is an $m \times n$ matrix, and $\Delta^T f_m$ is the transpose (i.e. row vector) of the gradient of $f_m$.

## Some Rules of Thumb

There's a pattern slowly emerging here:

- The derivative of a scalar with respect to a vector is a vector of the same size
- The derviatve of a vector of length $n$ with respect to a vector of length $m$ is a matrix of size $m \times n$
- The derivative of a scalar with respect to an $m \times n$ matrix is another $m \times n$ matrix

And so on. This is useful to have in mind: one simple way to check a solution to a problem is to see if these rules are satisfied. The pattern still holds for e.g. the derviative of a vector with respect to a matrix - the output for this is a tensor, which is a multidimensional generalisation of matrices. That's beyond this notebook though!

## A Note on Jacobians

In many applications, Jacobian matrices turn out to be square matrices (i.e. $m=n$), with off-diagonal entries of 0.

When the matrix is square, whether the off-diagonal elements are 0 or not depends on whether we are taking the derivative of a vector *element-wise binary operation*. Using the notation from *The Matrix Calculus You Need* (see intro), we denote a binary operation of two vector functions as:

$$ \boldsymbol{f}(\boldsymbol{w}) \bigcirc \boldsymbol{g}(\boldsymbol{x}) $$

In vector form, this is:

$$ \boldsymbol{y} = \begin{bmatrix} y_1 \\ \vdots \\ y_n \end{bmatrix} = \begin{bmatrix} f_1(\boldsymbol{w}) \bigcirc g_1(\boldsymbol{x}) \\ \vdots \\ f_n(\boldsymbol{w}) \bigcirc g_n(\boldsymbol{x}) \end{bmatrix} $$

The generalised case of the Jacobian with respect to $\boldsymbol{w}$ is

$$ \boldsymbol{J}_{\boldsymbol{w}} = \begin{bmatrix} \frac{\partial y_1}{\partial w_1} (f_1(\boldsymbol{w}) \bigcirc g_1(\boldsymbol{x})) & \cdots & \frac{\partial y_1}{\partial w_n} (f_1(\boldsymbol{w}) \bigcirc g_1(\boldsymbol{x})) \\ \vdots & \ddots & \vdots \\ \frac{\partial y_n}{\partial w_1} (f_n(\boldsymbol{w}) \bigcirc g_n(\boldsymbol{x})) & \cdots & \frac{\partial y_n}{\partial w_n} (f_n(\boldsymbol{w}) \bigcirc g_n(\boldsymbol{x})) \end{bmatrix} $$

Of special interest is the case when $f_i$ and $g_i$ are constants with respect to $w_j$ and $i \neq j$. In this case:

$$ \frac{\partial y_i}{\partial w_j} f_i(\boldsymbol{w}) \bigcirc g_i(\boldsymbol{x}) = 0 $$

$f_i$ and $g_i$ will be constant with respect to $w_j$, $i \neq j$ when $ \bigcirc $ represents an *element-wise* binary operation.

An element-wise binary operation occurs when $f_i$ is solely a function of $w_i$ and $g_i$ is solely a function of $x_i$. Some examples include vector addition, vector subtraction, element-wise multiplication of vectors, and element-wise division of vectors.
