# Neural Networks by Hand

In this [jupyter book](https://jupyterbook.org/en/stable/intro.html), I set out how to build a feedforward neural network function from scratch using only base python and numpy.

This notebook largely exists for the purpose of my own self-learning. Hopefully however it's also useful for someone other than myself. A lot of neural network by hand type tutorials usually focus on neural networks with either a single or at most two hidden layers. I'd like to instead focus on producing a more general solution, which means:

- arbitrary number of hidden layers
- arbitrary width of each hidden layer
- different activation functions available both for hidden and output layers
- different loss functions available

Of course, in practice the one thing I won't be doing is writing super-optimised code. So 'arbitary' depth and width will have some natural constraints.

Nonetheless, it is my hope that in producing a general solution, I'll improve my own understanding of what's going on when I use pytorch or tensorflow in the future.
