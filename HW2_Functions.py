from statsmodels.stats.stattools import durbin_watson
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import shapiro
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def prepare_data_for_vif(X):
    """
    This function prepares the dataframe so that it can calculate VIF scores to check
    for multicollinearity.
    """
    X_prep = X.copy()

    bool_columns = X_prep.select_dtypes(include=["bool"]).columns
    for col in bool_columns:
        X_prep[col] = X_prep[col].astype(int)

    X_prep = X_prep.astype(float)

    return X_prep


def check_regression_assumptions(model, X_train, y_train):
    
    # predict
    y_pred = model.predict(X_train)
    residuals = y_train - y_pred

    # durbin watson test
    dw_stat = durbin_watson(residuals)
    print(f"Durbin-Watson statistic: {dw_stat:.3f} (≈2 suggests independence)")
    #Chat GPT 4.0 was used to generate this code on 2/3/2025 at 8:57am

    # organize plots
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    # homoscedasticity
    ax[0].scatter(y_pred, residuals, alpha=0.5)
    ax[0].axhline(0, color="red", linestyle="--")
    ax[0].set_xlabel("Fitted Values")
    ax[0].set_ylabel("Residuals")
    ax[0].set_title("Residuals vs Fitted (Homoscedasticity)")
    #Chat GPT 4.0 was used to generate this code on 2/3/2025 at 9:01am

    # residuals
    sns.histplot(residuals, kde=True, ax=ax[1])
    ax[1].set_title("Residuals Histogram")

    # qq plot
    sm.qqplot(residuals, line="s", ax=ax[2])
    ax[2].set_title("Q-Q Plot (Normality Check)")

    plt.tight_layout()
    plt.show()


def test_regression_model(model, X_test, y_test):
    """
    This function gets performance metricis for the regression model.
    """
    # predict
    y_pred = model.predict(X_test)

    # performance metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}
