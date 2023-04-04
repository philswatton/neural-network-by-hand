# Linear Algebra

TODO

A lot of the time, it's either easier or more convinient to express the math for not only neural networks but most statistical models and algorithms via linear algebra (sometimes called matrix algebra). Similarly, there are many computational advantages to using algorithms for matrices rather than for loops in terms of speed, so there are real benefits to being familiar with linear algebra.

This section of the guide won't much linear algebra - just the parts you need to know to understand some of the later sections of this document. There are plenty of good linear algebra guides if you want a deeper treatment!

## Some definitions

A *matrix* is an array of numbers. Here are some:

$$ \boldsymbol{A} = \begin{bmatrix} 2 & 4 \\ 5 & 1 \end{bmatrix} \\[5pt]
\boldsymbol{B} = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \\[5pt]
\boldsymbol{C} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix} $$

A matrix has $n$ rows and $m$ columns. We write this as $n \times m$ and call this the *order* of the matrix. If $n=m$, we call the matrix a *square matrix*.

If one of these values is 1, the matrix is a vector. We can have row vectors:

$$ \boldsymbol{v} = \begin{bmatrix} 1 & 2 & 3\end{bmatrix} $$

And we can have column vectors:

$$ \boldsymbol{u} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} $$

A quick notational convention: vectors and matrices are usually written in bold. Matrices are usually upper case, while vectors are usually lower case.

## The Dot Product

Before we talk about matrices, it's useful for a moment to talk about the dot product of two vectors. Let's take our $\boldsymbol{u}$ and $\boldsymbol{v}$ vectors from above, and take their dot product:

$$ \boldsymbol{u} \cdot \boldsymbol{v} $$

```{note}
For now, we are ignoring whether these are row or column vectors. We'll be returning to the question of how to treat row vs column vectors later on.
```

The output of the dot product is given by:

$$ \boldsymbol{u} \cdot \boldsymbol{v} = 1 \times 1 + 2 \times 2 + 3 \times 3 = 14 $$

There is an important rule for the dot product of two vectors: they must both be of the same length. We can write a general formula for the dot product of two $n$-length vectors as:

$$ \boldsymbol{a} \cdot \boldsymbol{b} = \sum^n_{i=1} a_i \times b_i $$

We'll now turn to matrices, where we'll quickly find that the dot product is very useful.

## Matrix Addition and Subtraction

Imagine we have the following matrices and vectors:

$$ \boldsymbol{M} = \begin{bmatrix}
    1 & 2 & 3 \\
    4 & 5 & 6 \\
    7 & 8 & 9
\end{bmatrix}$$

$$ \boldsymbol{N} = \begin{bmatrix}
    10 & 20 & 30 \\
    40 & 50 & 60 \\
    70 & 80 & 90
\end{bmatrix} $$

$$ \boldsymbol{v} = \begin{bmatrix}
    1 \\
    2 \\
    3
\end{bmatrix} $$

### Adding a Scalar to a Matrix

To add a scalar to a matrix, we add the value of that scalar to every element of the matrix. So, if we wanted to add the number 10 to $\boldsymbol{M}$:

$$ \boldsymbol{M} + 10 = \begin{bmatrix}
    1 & 2 & 3 \\
    4 & 5 & 6 \\
    7 & 8 & 9
\end{bmatrix} + 10 = \begin{bmatrix}
    11 & 12 & 13 \\
    14 & 15 & 16 \\
    17 & 18 & 19
\end{bmatrix}$$

### Adding Two Matrices

To add two matrices, we just add the corresponding elements. So to add $\boldsymbol{M}$ and $\boldsymbol{N}$, we would add $m_{1,1}$ to $n_{1,1}$, $m_{1,2}$ to $n_{1,2}$ and so on:

$$ \boldsymbol{M} + \boldsymbol{N} = \begin{bmatrix}
    1 & 2 & 3 \\
    4 & 5 & 6 \\
    7 & 8 & 9
\end{bmatrix} + \begin{bmatrix}
    10 & 20 & 30 \\
    40 & 50 & 60 \\
    70 & 80 & 90
\end{bmatrix} = \begin{bmatrix}
    11 & 22 & 33 \\
    44 & 55 & 66 \\
    77 & 88 & 99
\end{bmatrix} $$

### Broadcasting

Not usually defined in normal linear algebra but commonly used in the world of deep learning is adding a vector to a matrix. This is realy a way of organising a series of scalar additions to vectors, and hence is permissible. This is often implemented in python packages like numpy, and will often be referred to as *broadcasting*.

If for example we wanted to add $\boldsymbol{M}$ and $\boldsymbol{v}$, we would first need $\boldsymbol{v}$ to be the correct shape to add it do $\boldsymbol{M}$. So, we just `stretch' $\boldsymbol{v}$ out so that it will match in the direction we need it to:

$$ \boldsymbol{M} + \boldsymbol{v} = \begin{bmatrix}
    1 & 2 & 3 \\
    4 & 5 & 6 \\
    7 & 8 & 9
\end{bmatrix} + \begin{bmatrix}
    1 \\
    2 \\
    3
\end{bmatrix} = \begin{bmatrix}
    1 & 2 & 3 \\
    4 & 5 & 6 \\
    7 & 8 & 9
\end{bmatrix} + \begin{bmatrix}
    1 & 1 & 1\\
    2 & 2 & 2\\
    3 & 3 & 3
\end{bmatrix}$$

You should be careful about generalising this to other contexts though. A lot of programming languages may implment it in some way, but it isn't a commonly used operation elsewhere.

## Scalar Multiplication

To multiply a matrix by a scalar, we do a similar thing as with scalar addition. That is, we multiply each element of the matrix by the scalar value. For example, if we wanted to $\boldsymbol{M}$ by the number 2:

$$ \boldsymbol{M} \times 2 = \begin{bmatrix}
    1 & 2 & 3 \\
    4 & 5 & 6 \\
    7 & 8 & 9
\end{bmatrix} \times 2 = \begin{bmatrix}
    2 & 4 & 6 \\
    8 & 10 & 12 \\
    14 & 16 & 18
\end{bmatrix}$$

## Matrix Multiplication

Last but probably most important among the matrix operations I'll be covering in this document is matrix multiplication.

Let's go back to our $\boldsymbol{B}$ and $\boldsymbol{C}$ matrices from earlier:

$$ \boldsymbol{B} = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \\[5pt]
\boldsymbol{C} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix} $$

Note that $\boldsymbol{B}$ has order $2 \times 3$ and $\boldsymbol{C}$ has order $3 \times 2$. To mutliply two matrices, the number of columns in the first matrix must match the number of rows in the second matrix. So in this case, $\boldsymbol{B}\boldsymbol{C}$ is a valid matrix multiplication, and so is $\boldsymbol{C}\boldsymbol{B}$.

But *this won't always be true*! In many cases, just because one matrix multiplication exists, does not mean the other one will.

So, say we want to compute $\boldsymbol{B}\boldsymbol{C}$. What do we actually do? Well, we use the dot products from earlier. A general rule for computing the $(i,j)$th element of the output matrix is that it is the dot product of the $i$th row of the first matrix and the $j$th column of the second matrix.

So, for $\boldsymbol{B}\boldsymbol{C}$:

$$ \boldsymbol{B}\boldsymbol{C} =
    \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}
    \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix}
    =
    \begin{bmatrix} (1, 2, 3) \cdot (1, 3, 5) & (1, 2, 3) \cdot (2, 4, 6) \\
    (4, 5, 6) \cdot (1, 3, 5) & (4, 5, 6) \cdot (2, 4, 6) \end{bmatrix}
$$

which is solved as:

$$ \boldsymbol{B}\boldsymbol{C} = \begin{bmatrix} 22 & 28 \\ 49 & 64 \end{bmatrix} $$

One thing worth noting: if the first matrix is order $n \times m$, and the second matrix is order $m \times l$, then the output matrix will be order $n \times l$. So we can add has many rows to the first matrix as we want, and as many columns to the second matrix as we want, and we will still have a solution.

```{note}
Unlike regular algebra, there is no guarantee that $\boldsymbol{A}\boldsymbol{B} = \boldsymbol{B}\boldsymbol{A}$ (or even that both exist). Because of this, it's important to pay attention to order when manipulating matrix equations.
```

There's a lot more to linear algebra than this. Other things worth covering (and that you're likely to come across) are matrix inversion, determinants, matrix decomposition
