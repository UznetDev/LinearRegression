# Linear Regression from Scratch

This repository contains a simple implementation of Linear Regression using Gradient Descent in Python. The goal of this project is to provide a clear and understandable implementation of linear regression, demonstrating how the algorithm works without relying on external libraries like scikit-learn.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Issues](#issues)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Linear regression is a statistical method that models the relationship between a dependent variable and one or more independent variables. This repository implements linear regression from scratch, using gradient descent to optimize the model parameters.

## Installation

To get started with this project, clone the repository to your local machine using the following command:

```bash
git clone https://github.com/UznetDev/LinearRegression.git
```

Navigate into the project directory:

```bash
cd LinearRegression
```

Ensure you have Python installed on your machine. This project does not require any external dependencies, but if you plan to extend it, you might want to create a virtual environment:

```bash
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
```

## Usage

The main script `model.py` contains the implementation of the linear regression model. You can use this class to fit a model to your data and make predictions. Here's a basic example:

```python
from model import LinearRegression

# Sample data
X = [[1], [2], [3], [4], [5]]
y = [1.2, 2.8, 3.6, 4.5, 5.1]

# Initialize the model
model = LinearRegression(step=0.01, n_iters=1000)

# Train the model
model.fit(X, y)

# Make predictions
predictions = model.predict([[6], [7]])

# Evaluate the model
mse = model.MSE(y, predictions)

print(f"Predictions: {predictions}")
print(f"MSE: {mse}")
```

## Project Structure

- `model.py`: Contains the `LinearRegression` class with methods to train, predict, and evaluate a linear regression model.
- `README.md`: This file, providing an overview of the project.

### Overview of the `model.py` Script:

1. **Class Definition:**
   - `LinearRegression`: A class that encapsulates the linear regression model, implementing gradient descent to optimize the model's parameters.

2. **Initialization (`__init__` method):**
   - The class is initialized with a learning rate (`step`) and the number of iterations (`n_iters`).
   - The model starts with initial slope and intercept values of zero, and internal variables to track the number of features (`__m_`) and samples (`__n_`).

3. **Training (`fit` method):**
   - The `fit` method takes in the input data `X` and target values `y`.
   - The method uses gradient descent to adjust the slope and intercept over a specified number of iterations.
   - It raises a `ValueError` if the number of samples in `X` and `y` do not match.

4. **Prediction (`predict` method):**
   - The `predict` method uses the trained model to predict the target values for new input data `X`.
   - It raises a `ValueError` if the input data has a different number of features or samples compared to the data used for training.

5. **Evaluation (`MSE` method):**
   - The `MSE` method calculates the Mean Squared Error (MSE) between the true target values `y` and the predicted values `y_pred`.


## Issues

If you encounter any problems or have suggestions for improvements, please open an issue in this repository. You can do this by navigating to the "Issues" tab and clicking on the "New Issue" button.

## Contributing

Contributions are welcome! If you would like to contribute to this project, please follow these steps:

1. Fork the repository by clicking the "Fork" button at the top of the page.
2. Clone the forked repository to your local machine:
    ```bash
    git clone https://github.com/<your-username>/LinearRegression.git
    ```
3. Create a new branch for your feature or bugfix:
    ```bash
    git checkout -b my-feature-branch
    ```
4. Make your changes and commit them:
    ```bash
    git commit -m "Add new feature"
    ```
5. Push your changes to your fork:
    ```bash
    git push origin my-feature-branch
    ```
6. Open a pull request from your branch to the `main` branch of this repository.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.