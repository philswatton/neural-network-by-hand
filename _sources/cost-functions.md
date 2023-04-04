# Cost Functions and their Derivatives

WIP

- Ignoring the sum and division by $m$ for simplicity
- Note this are the derivatives with respect to $\hat{y}$

## MSE

- Used for continous outputs

$$ C(\hat{y}, y) = \hat{y} - y^2 $$

$$ C'(\hat{y}, y) = 2*(\hat{y} - y) $$

## Binary Cross-Entropy

- Used for binary classification

$$ C(\hat{y}, y) = -y log(\hat{y}) - (1-y) log(1-\hat{y}) $$

$$ C'(\hat{y}, y) = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}} $$
