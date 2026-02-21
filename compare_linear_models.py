import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def compare_linear_models(X, y, models, cv=5, scoring='neg_mean_squared_error', scale=False):
    results = []
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)

    for name, model in models:
        if scale:
            estimator = make_pipeline(StandardScaler(), model)
        else:
            estimator = model

        scores = cross_val_score(estimator, X, y, cv=kfold, scoring=scoring)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        results.append({
            'Model': name,
            'Mean score': mean_score,
            'Std': std_score
        })

    results_df = pl.DataFrame(results).sort(by='Mean score',descending=True)
    return results_df

def plot_comparison(results_pl, scoring_name='Neg MSE'):
    models = results_pl['Model'].to_list()
    mean_scores = results_pl['Mean score'].to_numpy()
    stds = results_pl['Std'].to_numpy()

    # Vẽ
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(models))
    plt.barh(y_pos, mean_scores, xerr=stds, capsize=5, color='skyblue')
    plt.yticks(y_pos, models)
    plt.xlabel(f'Mean {scoring_name}')
    plt.title('Model Comparison via Cross-Validation')
    plt.gca().invert_yaxis()  # để model tốt nhất lên trên
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    diabetes = datasets.load_diabetes()

    X = diabetes.data
    y = diabetes.target
    models = [
        ('Linear Regression', linear_model.LinearRegression()),
        ('Ridge (alpha=1.0)', linear_model.Ridge(alpha=1.0)),
        ('Ridge (alpha=0.1)',linear_model.Ridge(alpha=0.1)),
        ('Lasso (alpha=0.1)', linear_model.Lasso(alpha=0.1)),
        ('ElasticNet (alpha=0.0005, l1_ratio=0.05)', linear_model.ElasticNet(alpha=0.0005, l1_ratio=0.05)),
    ]
    results = compare_linear_models(X, y, models, cv=10,scoring='r2')
    print(results)
    plot_comparison(results, scoring_name='R²')

