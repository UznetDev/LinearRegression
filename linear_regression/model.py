class LinearRegression():
    """
    A simple implementation of Linear Regression using Gradient Descent.
    """

    def __init__(self, step=0.001, n_iters=10000):
        """
        Initializes the LinearRegression class with the provided learning rate (`step`)
        and the number of iterations (`n_iters`). Also initializes the slope and intercept
        of the model, as well as internal variables to track the number of features (`__m_`)
        and samples (`__n_`).

        Parameters:
        -----------
        step : float
            The learning rate for the gradient descent optimization. It controls how big
            of a step is taken towards the minimum during each iteration.
        n_iters : int
            The number of iterations for the gradient descent optimization.
        """
        self.__k = 0
        self.step = step
        self.n_iters = n_iters
        self._slope_ = 0
        self._intercept_ = 0
        self.__m_ = 0
        self.__n_ = 0

    def fit(self, X, y):
        """
        Trains the linear regression model using the input data `X` and target values `y`.
        It adjusts the slope (`_slope_`) and intercept (`_intercept_`) by performing gradient
        descent over `n_iters` iterations.

        Parameters:
        -----------
        X : list of list of float or list of float
            The input data, where each inner list represents a sample with multiple features,
            or a simple list if there's only one feature.
        y : list of float
            The target values corresponding to each sample in `X`.

        Raises:
        -------
        ValueError:
            If the number of samples in `X` and `y` do not match.
        """
        self.__n_ = len(X)

        if isinstance(X[0], list):
            self.__m_ = len(X[0])
        else:
            self.__m_ = 1
            X = [[x] for x in X]

        if self.__n_ != len(y):
            raise ValueError(f"X and y must have the same number of samples: {(self.__n_, len(y))}")

        self._slope_ = [0] * self.__m_
        self._intercept_ = 0

        for _ in range(self.n_iters):
            y_pred = self.predict(X)

            for j in range(self.__m_):
                self._slope_[j] -= self.step * (-(2/self.__n_) * sum((y[i] - y_pred[i]) * X[i][j] for i in range(self.__n_)))
            self._intercept_ -= self.step * (-(2/self.__n_) * sum(y[i] - y_pred[i] for i in range(self.__n_)))

    def predict(self, X):
        """
        Predicts the target values for the given input data `X` using the trained linear
        regression model.

        Parameters:
        -----------
        X : list of list of float or list of float
            The input data, where each inner list represents a sample with multiple features,
            or a simple list if there's only one feature.

        Returns:
        --------
        y_pred : list of float
            The predicted target values corresponding to each sample in `X`.

        Raises:
        -------
        ValueError:
            If the input data `X` has a different number of features or samples compared
            to the data used for training.
        """
        if isinstance(X[0], list):
            m = len(X[0])
        else:
            m = 1
            X = [[x] for x in X]
        n = len(X)
        if m != self.__m_ or n != self.__n_:
            raise ValueError(f"X must have the same number of features as the training data: {(m, n)}, Except: {(self.__m_, self.__n_)}")

        y_pred = []
        for i in range(len(X)):
            y_pred.append(sum(self._slope_[j] * X[i][j] for j in range(m)) + self._intercept_)

        return y_pred

    def MSE(self, y, y_pred):
        """
        Calculates the Mean Squared Error (MSE) between the true target values `y` and the
        predicted values `y_pred`.

        Parameters:
        -----------
        y : list of float
            The true target values.
        y_pred : list of float
            The predicted target values.

        Returns:
        --------
        mse : float
            The mean squared error between `y` and `y_pred`.
        """
        return sum((y[i] - y_pred[i]) ** 2 for i in range(len(y))) / len(y)
