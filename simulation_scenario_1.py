# Scenario 1: Synthetic Simulation to Evaluate Probability Estimation under Unbalanced Classification


# This script generates synthetic Gaussian data to simulate binary classification under class imbalance.
# It trains logistic regression models using four strategies—original, oversampled, undersampled, and class-weighted—
# and also applys the corresponding probability adjustments and thereshold choices to each method.
# It compares these probabilities to the true posterior probabilities derived from the data-generating process.
# By repeating the experiment across multiple random settings, it quantifies how much each strategy distorts
# posterior estimates, and whether the proposed correction formulas can reduce this bias.


import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import multivariate_normal

# Initialization
p = 5  
n_train = 1000  
n_test = 5000  
pi_plus = 0.05  # Minority class proportion
outer_iterations = 5  
inner_iterations = 500  

# Generate synthetic training or test data under Gaussian assumptions
def generate_data(n, mu_0, sigma_0, mu_1, sigma_1, pi_plus):
    n_minority = int(n * pi_plus)
    n_majority = n - n_minority
    X_majority = np.random.normal(mu_0, sigma_0, size=(n_majority, p))
    X_minority = np.random.normal(mu_1, sigma_1, size=(n_minority, p))
    X = np.vstack((X_majority, X_minority))
    y = np.hstack((np.zeros(n_majority), np.ones(n_minority)))
    return X, y

# Compute the theoretical posterior P(Y=1 | X) using the Bayes' Theorem
def compute_posterior(X, mu_0, Sigma_0, mu_1, Sigma_1, pi_plus):
    pi_minus = 1 - pi_plus
    p_x_given_y1 = multivariate_normal.pdf(X, mean=mu_1, cov=Sigma_1)
    p_x_given_y0 = multivariate_normal.pdf(X, mean=mu_0, cov=Sigma_0)
    p_y1_given_x = (p_x_given_y1 * pi_plus) / (p_x_given_y1 * pi_plus + p_x_given_y0 * pi_minus)
    return p_y1_given_x

# Outer loop across different data-generating process
for outer in range(outer_iterations):
    print(f"\n========== Outer Iteration {outer+1}/{outer_iterations} ==========\n")

    np.random.seed(outer)
    mu_0 = np.random.uniform(1, 5, p)
    mu_1 = np.random.uniform(4, 6, p)
    Sigma_0 = np.eye(p) * np.random.uniform(2, 5)
    Sigma_1 = np.eye(p) * np.random.uniform(1, 2)

    X_test, y_test = generate_data(n_test, mu_0, np.diag(Sigma_0), mu_1, np.diag(Sigma_1), pi_plus)
    theoretical_probabilities = compute_posterior(X_test, mu_0, Sigma_0, mu_1, Sigma_1, pi_plus)

    inner_results = []

    # Inner loop under fixed distribution
    for inner in tqdm(range(inner_iterations), desc=f"Outer Iteration {outer+1} Progress"):
        X_train, y_train = generate_data(n_train, mu_0, np.diag(Sigma_0), mu_1, np.diag(Sigma_1), pi_plus)

        # Train models using original, oversampled, undersampled, and weighted strategies
        model_orig = LogisticRegression(max_iter=1000).fit(X_train, y_train)
        X_over, y_over = RandomOverSampler(random_state=inner).fit_resample(X_train, y_train)
        model_over = LogisticRegression(max_iter=1000).fit(X_over, y_over)
        X_under, y_under = RandomUnderSampler(random_state=inner).fit_resample(X_train, y_train)
        model_under = LogisticRegression(max_iter=1000).fit(X_under, y_under)
        model_weighted = LogisticRegression(max_iter=1000, class_weight='balanced').fit(X_train, y_train)

      
        original_predicted_probabilities = model_orig.predict_proba(X_test)[:, 1]
        over_predicted_probabilities = model_over.predict_proba(X_test)[:, 1]
        under_predicted_probabilities = model_under.predict_proba(X_test)[:, 1]
        weighted_predicted_probabilities = model_weighted.predict_proba(X_test)[:, 1]

        # Probability Correction
        proportion_0 = (y_train == 0).sum() / len(y_train)
        proportion_1 = (y_train == 1).sum() / len(y_train)
        rate = proportion_0 / proportion_1
        adj_prob_over = over_predicted_probabilities / (rate - rate * over_predicted_probabilities + over_predicted_probabilities)
        rate2 = proportion_1 / proportion_0
        adj_prob_under = (rate2 * under_predicted_probabilities) / (rate2 * under_predicted_probabilities - under_predicted_probabilities + 1)
        adj_prob_weighted = (weighted_predicted_probabilities / proportion_0) / ((1 / proportion_1) + (1 / proportion_0) * weighted_predicted_probabilities - (1 / proportion_1) * weighted_predicted_probabilities)

        # Evaluation
        mse_results = {
            "Method": ["Original", "Oversampled", "Oversampled-Corr", "Undersampled", "Undersampled-Corr", "Weighted", "Weighted-Corr"],
            "MSE": [
                mean_squared_error(theoretical_probabilities, original_predicted_probabilities),
                mean_squared_error(theoretical_probabilities, over_predicted_probabilities),
                mean_squared_error(theoretical_probabilities, adj_prob_over),
                mean_squared_error(theoretical_probabilities, under_predicted_probabilities),
                mean_squared_error(theoretical_probabilities, adj_prob_under),
                mean_squared_error(theoretical_probabilities, weighted_predicted_probabilities),
                mean_squared_error(theoretical_probabilities, adj_prob_weighted),
            ],
            "MAE": [
                mean_absolute_error(theoretical_probabilities, original_predicted_probabilities),
                mean_absolute_error(theoretical_probabilities, over_predicted_probabilities),
                mean_absolute_error(theoretical_probabilities, adj_prob_over),
                mean_absolute_error(theoretical_probabilities, under_predicted_probabilities),
                mean_absolute_error(theoretical_probabilities, adj_prob_under),
                mean_absolute_error(theoretical_probabilities, weighted_predicted_probabilities),
                mean_absolute_error(theoretical_probabilities, adj_prob_weighted),
            ],
            "AUC": [
                roc_auc_score(y_test, original_predicted_probabilities),
                roc_auc_score(y_test, over_predicted_probabilities),
                roc_auc_score(y_test, adj_prob_over),
                roc_auc_score(y_test, under_predicted_probabilities),
                roc_auc_score(y_test, adj_prob_under),
                roc_auc_score(y_test, weighted_predicted_probabilities),
                roc_auc_score(y_test, adj_prob_weighted),
            ],
        }

        inner_results.append(pd.DataFrame(mse_results))

   
    df_results = pd.concat(inner_results)
    summary = df_results.groupby("Method").agg(["mean", "std"])

    print("\n=== Summary for Outer Iteration", outer+1, "===\n")
    print(summary)
