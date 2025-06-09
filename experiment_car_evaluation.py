import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
from sklearn.metrics import recall_score, precision_score
from ucimlrepo import fetch_ucirepo 
from imblearn.metrics import geometric_mean_score
from sklearn.preprocessing import LabelEncoder

# This script evaluates how different balancing strategies affect probability estimates 
# and evaluate the efficiency of our proposed probability adjustment
# on the credit card fraud detection dataset. We compare original, oversampled, undersampled,
# and class-weighted logistic regression models over 100 randomized runs.

n_iterations = 100  

auc_original_list, auc_over_list, auc_under_list, auc_weight_list = [], [], [], []
auc_adj_over_list, auc_adj_under_list, auc_adj_weight_list = [], [], []

brier_original_list, brier_over_list, brier_under_list, brier_weight_list = [], [], [], []
brier_adj_over_list, brier_adj_under_list, brier_adj_weight_list = [], [], []

logloss_original_list, logloss_over_list, logloss_under_list, logloss_weight_list = [], [], [], []
logloss_adj_over_list, logloss_adj_under_list, logloss_adj_weight_list = [], [], []

ece_original_list, ece_over_list, ece_under_list, ece_weight_list = [], [], [], []
ece_adj_over_list, ece_adj_under_list, ece_adj_weight_list = [], [], []

gmean_original_list, gmean_over_list, gmean_under_list, gmean_weight_list = [], [], [], []
gmean_adj_over_list, gmean_adj_under_list, gmean_adj_weight_list = [], [], []


car_evaluation = fetch_ucirepo(id=19) 
  
# data (as pandas dataframes) 
X = car_evaluation.data.features 
y = car_evaluation.data.targets['class'] 

label_encoder = LabelEncoder()
X = X.apply(label_encoder.fit_transform)

Y = y.apply(lambda res: 1 if res == 'vgood' else 0)


X_train_complete, X_test, y_train_complete, y_test = train_test_split(X, Y, test_size=0.3)


for i in range(n_iterations):
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_complete, y_train_complete, test_size=0.2, random_state=i)
    
   
    ros = RandomOverSampler(random_state=i)
    X_train_over, y_train_over = ros.fit_resample(X_train, y_train)
    
    rus = RandomUnderSampler(random_state=i)
    X_train_under, y_train_under = rus.fit_resample(X_train, y_train)
    
    # Train logistic regression models under different balancing conditions
    clf1 = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    clf2 = LogisticRegression(max_iter=1000).fit(X_train_over, y_train_over)
    clf3 = LogisticRegression(max_iter=1000).fit(X_train_under, y_train_under)
    clf4 = LogisticRegression(max_iter=1000, class_weight='balanced').fit(X_train, y_train)
    
   
    original_predicted_probabilities = clf1.predict_proba(X_test)[:, 1]
    over_predicted_probabilities = clf2.predict_proba(X_test)[:, 1]
    under_predicted_probabilities = clf3.predict_proba(X_test)[:, 1]
    weighted_predicted_probabilities = clf4.predict_proba(X_test)[:, 1]
    
    # Apply correction to recover calibrated probabilities
    class_counts = y_train.value_counts()
    proportion_0, proportion_1 = class_counts[0] / len(y_train), class_counts[1] / len(y_train)
    rate = proportion_0 / proportion_1
    adj_prob_over = over_predicted_probabilities / (rate - rate * over_predicted_probabilities + over_predicted_probabilities)
    rate2 = proportion_1 / proportion_0
    adj_prob_under = (rate2 * under_predicted_probabilities) / (rate2 * under_predicted_probabilities - under_predicted_probabilities + 1)
    adj_prob_weighted = (weighted_predicted_probabilities / proportion_0) / ((1 / proportion_1) + (1 / proportion_0) * weighted_predicted_probabilities - (1 / proportion_1) * weighted_predicted_probabilities)
    
    
    # Evaluation

    # AUC
    auc_original_list.append(roc_auc_score(y_test, original_predicted_probabilities))
    auc_over_list.append(roc_auc_score(y_test, over_predicted_probabilities))
    auc_adj_over_list.append(roc_auc_score(y_test, adj_prob_over))
    auc_under_list.append(roc_auc_score(y_test, under_predicted_probabilities))
    auc_adj_under_list.append(roc_auc_score(y_test, adj_prob_under))
    auc_weight_list.append(roc_auc_score(y_test, weighted_predicted_probabilities))
    auc_adj_weight_list.append(roc_auc_score(y_test, adj_prob_weighted))
    
    # Brier Score
    brier_original_list.append(brier_score_loss(y_test, original_predicted_probabilities))
    brier_over_list.append(brier_score_loss(y_test, over_predicted_probabilities))
    brier_adj_over_list.append(brier_score_loss(y_test, adj_prob_over))
    brier_under_list.append(brier_score_loss(y_test, under_predicted_probabilities))
    brier_adj_under_list.append(brier_score_loss(y_test, adj_prob_under))
    brier_weight_list.append(brier_score_loss(y_test, weighted_predicted_probabilities))
    brier_adj_weight_list.append(brier_score_loss(y_test, adj_prob_weighted))
    
    # Log Loss
    logloss_original_list.append(log_loss(y_test, original_predicted_probabilities))
    logloss_over_list.append(log_loss(y_test, over_predicted_probabilities))
    logloss_adj_over_list.append(log_loss(y_test, adj_prob_over))
    logloss_under_list.append(log_loss(y_test, under_predicted_probabilities))
    logloss_adj_under_list.append(log_loss(y_test, adj_prob_under))
    logloss_weight_list.append(log_loss(y_test, weighted_predicted_probabilities))
    logloss_adj_weight_list.append(log_loss(y_test, adj_prob_weighted))
    
    # Expected Calibration Error (ECE)
    def calculate_ece(y_true, y_prob, n_bins=10):
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        return np.mean(np.abs(prob_true - prob_pred))

    ece_original_list.append(calculate_ece(y_test, original_predicted_probabilities))
    ece_over_list.append(calculate_ece(y_test, over_predicted_probabilities))
    ece_adj_over_list.append(calculate_ece(y_test, adj_prob_over))
    ece_under_list.append(calculate_ece(y_test, under_predicted_probabilities))
    ece_adj_under_list.append(calculate_ece(y_test, adj_prob_under))
    ece_weight_list.append(calculate_ece(y_test, weighted_predicted_probabilities))
    ece_adj_weight_list.append(calculate_ece(y_test, adj_prob_weighted))

    # G-mean
    def calculate_gmean(y_true, y_pred_prob, threshold):
        y_pred = (y_pred_prob >= threshold).astype(int)
        return geometric_mean_score(y_true, y_pred)

    gmean_original_list.append(calculate_gmean(y_test, original_predicted_probabilities, threshold=0.5))
    gmean_over_list.append(calculate_gmean(y_test, over_predicted_probabilities, threshold=0.5))
    gmean_adj_over_list.append(calculate_gmean(y_test, adj_prob_over, threshold=proportion_1))
    gmean_under_list.append(calculate_gmean(y_test, under_predicted_probabilities, threshold=0.5))
    gmean_adj_under_list.append(calculate_gmean(y_test, adj_prob_under, threshold=proportion_1))
    gmean_weight_list.append(calculate_gmean(y_test, weighted_predicted_probabilities, threshold=0.5))
    gmean_adj_weight_list.append(calculate_gmean(y_test, adj_prob_weighted, threshold=proportion_1))

# Report 
print("Mean AUC Scores:")
print("Original:", np.mean(auc_original_list))
print("Oversampled:", np.mean(auc_over_list))
print("Adjusted Oversampled:", np.mean(auc_adj_over_list))
print("Undersampled:", np.mean(auc_under_list))
print("Adjusted Undersampled:", np.mean(auc_adj_under_list))
print("Weighted:", np.mean(auc_weight_list))
print("Adjusted Weighted:", np.mean(auc_adj_weight_list))

print("\nMean Brier Scores:")
print("Original:", np.mean(brier_original_list))
print("Oversampled:", np.mean(brier_over_list))
print("Adjusted Oversampled:", np.mean(brier_adj_over_list))
print("Undersampled:", np.mean(brier_under_list))
print("Adjusted Undersampled:", np.mean(brier_adj_under_list))
print("Weighted:", np.mean(brier_weight_list))
print("Adjusted Weighted:", np.mean(brier_adj_weight_list))

print("\nMean Log Loss:")
print("Original:", np.mean(logloss_original_list))
print("Oversampled:", np.mean(logloss_over_list))
print("Adjusted Oversampled:", np.mean(logloss_adj_over_list))
print("Undersampled:", np.mean(logloss_under_list))
print("Adjusted Undersampled:", np.mean(logloss_adj_under_list))
print("Weighted:", np.mean(logloss_weight_list))
print("Adjusted Weighted:", np.mean(logloss_adj_weight_list))

print("\nMean ECE:")
print("Original:", np.mean(ece_original_list))
print("Oversampled:", np.mean(ece_over_list))
print("Adjusted Oversampled:", np.mean(ece_adj_over_list))
print("Undersampled:", np.mean(ece_under_list))
print("Adjusted Undersampled:", np.mean(ece_adj_under_list))
print("Weighted:", np.mean(ece_weight_list))
print("Adjusted Weighted:", np.mean(ece_adj_weight_list))

print("\nMean Gmean:")
print("Original:", np.mean(gmean_original_list))
print("Oversampled:", np.mean(gmean_over_list))
print("Adjusted Oversampled:", np.mean(gmean_adj_over_list))
print("Undersampled:", np.mean(gmean_under_list))
print("Adjusted Undersampled:", np.mean(gmean_adj_under_list))
print("Weighted:", np.mean(gmean_weight_list))
print("Adjusted Weighted:", np.mean(gmean_adj_weight_list))
