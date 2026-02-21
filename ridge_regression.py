import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

diabetes = datasets.load_diabetes()

diabetes_X = diabetes.data[:,np.newaxis,2]

diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

regr = linear_model.RidgeCV(alphas=[0.1,0.2,0.3,0.4,0.5])

regr.fit(diabetes_X_train,diabetes_y_train)

dia_pred = regr.predict(diabetes_X_test)

if __name__ == "__main__":
    print(f"Coefficients: {regr.coef_}")
    print(f"MSE: {mean_squared_error(diabetes_y_test,dia_pred)}")

    print(f"Variance score: {r2_score(diabetes_y_test,dia_pred):.2f}")
    sns.regplot(x=diabetes_X_test.ravel(),
                y=diabetes_y_test)
    plt.xlabel("BMI (scaled)")
    plt.ylabel("Disease progression")
    plt.title("Linear Regression on Diabetes dataset")
    plt.show()
