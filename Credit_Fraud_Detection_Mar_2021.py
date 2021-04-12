#!/usr/bin/env python
# coding: utf-8

# Credit card fraud detection project using data from https://www.kaggle.com/mlg-ulb/creditcardfraud
# 
# Features, except Amount and Time, have been scaled already.

# # Exploratary Data Analysis

# In[1]:


import numpy as np 
import pandas as pd 

df = pd.read_csv("creditcard.csv")
df.shape


# In[2]:


df.head(3)


# In[3]:


df.describe()


# ### Check null values

# In[4]:


# check null values
df.isnull().sum()


# ### Check feature distribution

# In[5]:


# get the sense of the proportion of fraud and non-fraud
print("No Fraud:", round(df['Class'].value_counts()[0]/len(df)*100,2), "%")
print("Fraud:", round(df['Class'].value_counts()[1]/len(df)*100,2), "%")


# In[6]:


# visualize the imbalanced y
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot('Class', data=df)
plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)


# In[7]:


# visualize the distribution of Amount and Time which haven't been scaled
fig, ax = plt.subplots(1, 2, figsize = (18,4))

amount_val = df["Amount"]
sns.distplot(amount_val, ax = ax[0], color = 'r')
ax[0].set_title("Distribution of Transaction Amount", fontsize = 14)
ax[0].set_xlim([min(amount_val), max(amount_val)])

time_val = df["Time"]
sns.distplot(time_val, ax = ax[1], color = 'b')
ax[1].set_title("Distribution of Transaction Time", fontsize = 14)
ax[1].set_xlim([min(time_val), max(time_val)])


# __Time__: Number of seconds elapsed between this transaction and the first transaction in the dataset. This data contains transactions within 48 hours (172,800 seconds) after the first transaction. It may be helpful to process the original Time into the time of the day.

# In[8]:


# change to hours after the first transaction
df["new_Time"] = df["Time"].apply(lambda x: x/3600) 

# change to the time in 24 hour style
df["Time_24h"] = df["new_Time"].apply(lambda x: x-24 if x > 24 else x) 


# In[9]:


sns.set_style(style = "darkgrid")

sns.distplot(df["Time_24h"], kde = False, bins = 48)
plt.xlim(0, 24)


# ### Scaling Amount

# In[10]:


import sklearn

from sklearn.preprocessing import RobustScaler 
# RobustScaler is less prone to outliers.
# https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html

rob_scaler = RobustScaler()

df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))

scaled_amount = df['scaled_amount']
time_24h = df["Time_24h"]

df.drop(['scaled_amount', "Time", "Amount", "Time_24h", "new_Time"], axis=1, inplace=True)

df.insert(0, 'scaled_amount', scaled_amount)
df.insert(1, 'scaled_time', time_24h)

df.head()


# ## Data Preparation
# ### Undersample the data

# In[11]:


# Since our classes are highly imbalanced we should oversample/undersample
# Here is undersample

# shuffle the data before creating the subsamples
df = df.sample(frac=1)

# The amount of fraud classes 492 rows, so also get non-fraud 492 classes
fraud_df = df.loc[df['Class'] == 1]
non_fraud_df = df.loc[df['Class'] == 0][:492]

balanced_df = pd.concat([fraud_df, non_fraud_df])

# Shuffle again
balanced_df = balanced_df.sample(frac=1, random_state=42)
balanced_df.head()


# In[12]:


# check the distribution
print('Distribution of the Classes in the undersampled dataset')
print(balanced_df['Class'].value_counts()/len(balanced_df))

sns.countplot('Class', data=balanced_df)
plt.title('Equally Distributed Classes', fontsize=14)
plt.show()


# ### Correlation Matrices

# In[13]:


# visualize the correlation using both undersampled and original data
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,8))

balanced_df_corr = balanced_df.corr()
sns.heatmap(balanced_df_corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax2)
ax1.set_title('Undersampled Correlation Matrix \n (use for reference)', fontsize=14)

# Entire DataFrame
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax1)
ax2.set_title("Original Correlation Matrix \n (don't use for reference)", fontsize=14)
plt.show()


# Some features are highly correlated to each other. If we know better about what each features are, or we have a VIF threshold, we can reduce the multicollinearity by excluding some highly correlated features. For now, we can explore the relationship between Class and highly correlated features -- V14, V12, V16, V10 (negative) and V11, V4, V2, V19 (positve).

# ### Boxplots

# In[14]:


# Negative Correlations with the Class 
f, axes = plt.subplots(ncols=4, figsize=(20,4))

print("Negative Correlation with the Class (Fraud|Non-fraud)")

sns.boxplot(x="Class", y="V14", data = balanced_df, ax=axes[0])
axes[0].set_title('V14 vs Class')

sns.boxplot(x="Class", y="V12", data = balanced_df, ax=axes[1])
axes[1].set_title('V12 vs Class')

sns.boxplot(x="Class", y="V16", data = balanced_df, ax=axes[2])
axes[2].set_title('V16 vs Class')

sns.boxplot(x="Class", y="V10", data = balanced_df, ax=axes[3])
axes[3].set_title('V10 vs Class')

plt.show()


# In[15]:


# Positive Correlations with the Class 
f, axes = plt.subplots(ncols=4, figsize=(20,4))

print("Positive Correlation with the Class (Fraud|Non-fraud)")

sns.boxplot(x="Class", y="V11", data = balanced_df, ax=axes[0])
axes[0].set_title('V11 vs Class')

sns.boxplot(x="Class", y="V4", data = balanced_df, ax=axes[1])
axes[1].set_title('V4 vs Class')

sns.boxplot(x="Class", y="V2", data = balanced_df, ax=axes[2])
axes[2].set_title('V2 vs Class')

sns.boxplot(x="Class", y="V19", data = balanced_df, ax=axes[3])
axes[3].set_title('V19 vs Class')

plt.show()


# ### Anomaly Detection
# If we know better about each feature we may need to detect and deal with anomaly.
# 
# ### Dimensionality Reduction

# In[16]:


import time
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE 
import matplotlib.patches as mpatches

# PCA Implementation
X = balanced_df.drop('Class', axis=1)
y = balanced_df['Class']

t0 = time.time()
X_reduced_pca = PCA(n_components=2).fit_transform(X.values)
t1 = time.time()
print("PCA took {:.2} s".format(t1 - t0))

# t-SNE Implementation
t0 = time.time()
X_reduced_tsne = TSNE(n_components=2).fit_transform(X.values)
t1 = time.time()
print("T-SNE took {:.2} s".format(t1 - t0))


# In[17]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,12))
# labels = ['No Fraud', 'Fraud']
f.suptitle('Clusters using Dimensionality Reduction', fontsize=14)

blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')
red_patch = mpatches.Patch(color='#AF0000', label='Fraud')

# PCA scatter plot
ax1.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax1.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax1.set_title('PCA', fontsize=14)
ax1.grid(True)
ax1.legend(handles=[blue_patch, red_patch])

# t-SNE scatter plot
ax2.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax2.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax2.set_title('t-SNE', fontsize=14)
ax2.grid(True)
ax2.legend(handles=[blue_patch, red_patch])

plt.show()


# # Classifiers for Undersampled Data
# Train four types of classifiers and compare the effectiveness on detecting fraud transactions. 
# - Logistic Regression
# - K Nearest Neighbors
# - Support Vector Machine
# - Decision Tree 
# 
# ### Train test split

# In[18]:


from sklearn.model_selection import train_test_split

X = balanced_df.drop('Class', axis=1).values
y = balanced_df['Class'].values

# With stratify
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)


# Since the features have been scaled, we can directly specify the classifiers and fit the training data. 

# In[19]:


# Classifier libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

classifiers = {
    "LogisiticRegression": LogisticRegression(),
    "KNearest": KNeighborsClassifier(),
    "Support Vector Classifier": SVC(),
    "DecisionTreeClassifier": DecisionTreeClassifier()
}

for key, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    training_score = cross_val_score(classifier, X_train, y_train, cv=5)
    print(key, "has a", round(training_score.mean(), 2) * 100, 
          "% accuracy for the training data")


# __GridSearchCV__ exhaustive searches over specified parameter values for an estimator. It is used to __find the best parameters__ that gives the best predictive score for the classifiers.
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

# In[20]:


from sklearn.model_selection import GridSearchCV

# Logistic Regression 
# C: inverse of regularization strength; must be a positive float. 
# Like in support vector machines, smaller values specify stronger regularization.
log_reg_params = {"penalty": ['l1', 'l2'], 
                  'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]} 

grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
grid_log_reg.fit(X_train, y_train)
# We will get the best parameters of logistic regression:
log_reg = grid_log_reg.best_estimator_

# K Nearst Neighbors
knears_params = {"n_neighbors": list(range(2,5)),
                 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)
grid_knears.fit(X_train, y_train)
knears_neighbors = grid_knears.best_estimator_

# Support Vector Classifier
svc_params = {'C': [0.5, 0.7, 0.9, 1], 
              'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
grid_svc = GridSearchCV(SVC(), svc_params)
grid_svc.fit(X_train, y_train)
svc = grid_svc.best_estimator_

# Decision Tree
tree_params = {"criterion": ["gini", "entropy"], 
               "max_depth": list(range(2,4)), 
               "min_samples_leaf": list(range(5,7))}
grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)
grid_tree.fit(X_train, y_train)
tree = grid_tree.best_estimator_


# In[21]:


# Random undersampled case
print("The average Cross Validation Score/Accuracy:")

log_reg_score = cross_val_score(log_reg, X_train, y_train, cv=5)
print('Logistic Regression:', round(log_reg_score.mean() * 100, 2), '%')

knears_score = cross_val_score(knears_neighbors, X_train, y_train, cv=5)
print('Knears Neighbors:', round(knears_score.mean() * 100, 2), '%')

svc_score = cross_val_score(svc, X_train, y_train, cv=5)
print('Support Vector Classifier:', round(svc_score.mean() * 100, 2), '%')

tree_score = cross_val_score(tree, X_train, y_train, cv=5)
print('DecisionTree Classifier:', round(tree_score.mean() * 100, 2), '%')


# In[22]:


# Implementing undersampling for multiple times for different non-fraud data
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report

# Get the test data
undersample_df = pd.concat([pd.DataFrame(X_test),  pd.DataFrame(y_test)], axis =1)

# calculate the metrics using n different undersampled dataset
def undersampled_metrics(classifier, df, n, params=[]):
    d = {"accuracy": [],
         "precision": [],
         "recall": [],
         "f1": [],
         "auc": []}

    for i in range(n):
        # shuffle the data before creating the subsamples
        df = df.sample(frac=1)

        fraud_df = df.loc[df['Class'] == 1]
        non_fraud_df = df.loc[df['Class'] == 0][:492]
        undersample_df = pd.concat([fraud_df, non_fraud_df])

        # Shuffle again
        undersample_df = undersample_df.sample(frac=1)
        
        undersample_X = balanced_df.drop('Class', axis=1).values
        undersample_y = balanced_df['Class'].values

        X_train, X_test, y_train, y_test = train_test_split(undersample_X, undersample_y, 
                                                            test_size=0.2)
        if params:
            model = GridSearchCV(classifier, params)
        else:
            model = classifier
            
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        d["accuracy"].append(accuracy_score(y_test, pred))
        d["precision"].append(precision_score(y_test, pred))
        d["recall"].append(recall_score(y_test, pred))
        d["f1"].append(f1_score(y_test, pred))
        d["auc"].append(roc_auc_score(y_test, pred))
    
    d["y_true"] = y_test
    if classifier == svc:
        d["y_pred"] = pred
    else:
        d["y_pred"] = pred
        d["y_pred1_prob"] = [y[1] for y in model.predict_proba(X_test)]
    
    return d


# In[23]:


# Use Logistic Regression as an example to test
# let's try five undersampled data first
n = 5
log_reg_params = {"penalty": ['l1', 'l2'], 
                  'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]} 
log_reg_metrics = undersampled_metrics(LogisticRegression(), df, n, log_reg_params)


# In[24]:


def print_undersampled_metrics(classifer_name, metrics, n):
    print(classifer_name + 'average')
    print('    Accuracy:', round(np.mean(metrics["accuracy"]) * 100, 2), '%')
    print('    Precision:', round(np.mean(metrics["precision"]) * 100, 2), '%')
    print('    Recall:', round(np.mean(metrics["recall"]) * 100, 2), '%')
    print('    F1 score:', round(np.mean(metrics["f1"]) * 100, 2), '%')
    print('    AUC:', round(np.mean(metrics["auc"]) * 100, 2), '%')
    print('For training data of', n, 'undersampled datasets.')


# In[25]:


print_undersampled_metrics('Logistic Regression ', log_reg_metrics, n)


# ### Plot Learning Curve

# In[26]:


def plot_learning_curve(estimator, title, X, y, 
                        axes=None, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ =         learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="#ff9124")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="#2492ff")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
                 label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
                 label="Cross-validation score")
    axes.grid(True)
    axes.legend(loc="best") 

    return plt


# In[27]:


fig, axes = plt.subplots(2, 2, figsize=(12, 12))

X, y = X_train, y_train

# Cross validation with 100 iterations to get smoother mean test and train score curves, 
# each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)

title = "Logistic Regression Learning Curve"
plot_learning_curve(log_reg, title, X, y, axes=axes[0, 0], ylim=(0.84, 1.01),
                    cv=cv, n_jobs=4)

title = "Knears Neighbors Learning Curve"
plot_learning_curve(knears_neighbors, title, X, y, axes=axes[0, 1], ylim=(0.84, 1.01),
                    cv=cv, n_jobs=4)

title = "Support Vector Classifier Learning Curve"
plot_learning_curve(svc, title, X, y, axes=axes[1, 0], ylim=(0.84, 1.01),
                    cv=cv, n_jobs=4)

title = "Decision Tree Classifier Learning Curve"
plot_learning_curve(tree, title, X, y, axes=axes[1, 1], ylim=(0.84, 1.01),
                    cv=cv, n_jobs=4)

plt.show()


# ### AUC and ROC

# In[ ]:


from sklearn.model_selection import cross_val_predict
# Create a DataFrame with all the scores and the classifiers names.

log_reg_metrics = undersampled_metrics(log_reg, df, 5)
knears_metrics = undersampled_metrics(knears_neighbors, df, 5)
svc_metrics = undersampled_metrics(svc, df, 5)
tree_metrics = undersampled_metrics(tree, df, 5)


# In[ ]:


print("Average AUC value")
print('    Logistic Regression:', round(np.mean(log_reg_metrics["auc"]) * 100, 2), '%')
print('    KNears Neighbors:', round(np.mean(knears_metrics["auc"]) * 100, 2), '%')
print('    Support Vector Classifier:', round(np.mean(svc_metrics["auc"]) * 100, 2), '%')
print('    Decision Tree Classifier:', round(np.mean(tree_metrics["auc"]) * 100, 2), '%')


# #### Plot ROC Curve 

# In[ ]:


from sklearn.metrics import roc_curve
true_vs_pred = [[log_reg_metrics["y_true"], log_reg_metrics["y_pred1_prob"]], 
                [knears_metrics["y_true"], knears_metrics["y_pred1_prob"]], 
                [svc_metrics["y_true"], svc_metrics["y_pred"]], 
                [tree_metrics["y_true"], tree_metrics["y_pred1_prob"]]]

log_fpr, log_tpr, log_thresold = roc_curve(true_vs_pred[0][0], true_vs_pred[0][1])
knear_fpr, knear_tpr, knear_threshold = roc_curve(true_vs_pred[1][0], true_vs_pred[1][1])
svc_fpr, svc_tpr, svc_threshold = roc_curve(true_vs_pred[2][0], true_vs_pred[2][1])
tree_fpr, tree_tpr, tree_threshold = roc_curve(true_vs_pred[3][0], true_vs_pred[3][1])

fpr_tpr_list = [[log_fpr, log_tpr], [knear_fpr, knear_tpr], [svc_fpr, svc_tpr], [tree_fpr, tree_tpr]]
classifer_list = ['Logistic Regression', 'KNears Neighbors', 'Support Vector Classifier', 'Decision Tree']

def plot_multiple_roc_curve(fpr_tpr_list, classifer_list):
    n = len(fpr_tpr_list)
    
    plt.figure(figsize=(12,10))
    #plt.title('ROC Curve for', n, 'Classifiers', fontsize=18)
    
    for i in range(n):
        plt.plot(fpr_tpr_list[i][0], 
                 fpr_tpr_list[i][1], 
                 label="{} AUC = {:.3f}".format(classifer_list[i], 
                                              roc_auc_score(true_vs_pred[i][0], true_vs_pred[i][1])))
    plt.plot([0,1], [0,1], color='#6E726D', linestyle='--')
    plt.annotate('Minimum ROC Score of 50% \n (Randome Classifier)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
                arrowprops=dict(facecolor='#6E726D', shrink=0.05))

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Flase Positive Rate", fontsize=15)
    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)
    plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='lower right')

    plt.show()

plot_multiple_roc_curve(fpr_tpr_list, classifer_list)


# #### Plot Precision-Recall Curve
# Use Logistic Regression as an example

# 

# In[ ]:


from sklearn.metrics import precision_recall_curve

# Get the original training and test data
X = df.drop('Class', axis=1).values
y = df['Class'].values
original_Xtrain, ori_Xtest, original_ytrain, ori_ytest = train_test_split(X, y, test_size = 0.2, stratify=y)
print('Length of X (train): {} | Length of y (train): {}'.format(len(ori_Xtrain), len(ori_ytrain)))
print('Length of X (test): {} | Length of y (test): {}'.format(len(ori_Xtest), len(ori_ytest)))

fig = plt.figure(figsize=(10,8))
y_pred = log_reg.predict(ori_Xtest)
log_true, log_pred_prob = ori_ytest, y_pred

precision, recall, _ = precision_recall_curve(log_true, log_pred_prob)

plt.step(recall, precision, color='#004a93', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='#48a6ff')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('UnderSampling Precision-Recall Curve', fontsize=16)


# # Oversampling method: SMOTE
# __SMOTE__: Synthetic Minority Over-sampling Technique. It creates new synthetic data points in order to have an equal balance of the classes. Oversampling is an alternative way to deal with imbalanced data problem.
# 
# SMOTE works by creating synthetic observations based upon the existing minority observations. To then oversample, take a sample from the dataset, and consider its k nearest neighbors (in feature space). To create a synthetic data point, take the vector between one of those k neighbors, and the current data point. Multiply this vector by a random number x which lies between 0, and 1. Add this to the current data point to create the new, synthetic data point.
# 
# SMOTE should be conducted __during cross validation or there will be a data leakage issue__. 
# 
# __StratifiedKFold__ is a variation of k-fold which returns stratified folds: each set contains approximately the __same percentage of samples of each target class as the complete set__. https://scikit-learn.org/stable/modules/cross_validation.html#stratified-k-fold
# 

# In[29]:


from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, RandomizedSearchCV

# Get the original training and test data
X = df.drop('Class', axis=1).values
y = df['Class'].values
original_Xtrain, ori_Xtest, original_ytrain, ori_ytest = train_test_split(X, y, test_size = 0.2, stratify=y)
print('Length of X (train): {} | Length of y (train): {}'.format(len(original_Xtrain), len(original_ytrain)))
print('Length of X (test): {} | Length of y (test): {}'.format(len(ori_Xtest), len(ori_ytest)))

# List to append the score and then find the average
accuracy_lst = []
precision_lst = []
recall_lst = []
f1_lst = []
auc_lst = []

# Classifier with optimal parameters
log_reg_params = {"penalty": ['l1', 'l2'], 
                  'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
rand_log_reg = RandomizedSearchCV(LogisticRegression(), log_reg_params, n_iter=4)


# In[31]:


from sklearn.model_selection import StratifiedKFold

# further split the training data into train and validation/test 
skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
for train, test in skf.split(original_Xtrain, original_ytrain):
    # SMOTE during Cross Validation
    pipeline = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority'), rand_log_reg) 
    model = pipeline.fit(original_Xtrain[train], original_ytrain[train])
    best_est = rand_log_reg.best_estimator_
    prediction = best_est.predict(original_Xtrain[test])
    
    accuracy_lst.append(accuracy_score(original_ytrain[test], prediction))
    precision_lst.append(precision_score(original_ytrain[test], prediction))
    recall_lst.append(recall_score(original_ytrain[test], prediction))
    f1_lst.append(f1_score(original_ytrain[test], prediction))
    auc_lst.append(roc_auc_score(original_ytrain[test], prediction))
    
print('---' * 30)
print('')
print("accuracy: {}".format(np.mean(accuracy_lst)))
print("precision: {}".format(np.mean(precision_lst)))
print("recall: {}".format(np.mean(recall_lst)))
print("f1: {}".format(np.mean(f1_lst)))
print('---' * 30)


# In[34]:


labels = ['No Fraud', 'Fraud']
smote_prediction = best_est.predict(ori_Xtest)
print(classification_report(ori_ytest, smote_prediction, target_names=labels))


# # Neural Networks
# ## For undersampling

# In[38]:


import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

n_inputs = X_train.shape[1]

NN_undersample_model = Sequential([
    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),
    Dense(32, activation='relu'), # Rectified Linear Unit (ReLU)
    Dense(2, activation='softmax')
])


# In[39]:


NN_undersample_model.summary()


# In[40]:


NN_undersample_model.compile(Adam(lr=0.001), 
                             loss='sparse_categorical_crossentropy', metrics=['accuracy'])

NN_undersample_model.fit(X_train, y_train, 
                      validation_split=0.2, 
                      batch_size=25, epochs=20, shuffle=True, verbose=2)


# In[48]:


from sklearn.metrics import confusion_matrix
NN_undersample_fraud_predictions = NN_undersample_model.predict_classes(ori_Xtest, 
                                                                        batch_size=200, verbose=0)

NN_undersample_cm = confusion_matrix(ori_ytest, NN_undersample_fraud_predictions)

print("Confusion Matrix using Neural Network and Undersampling")
print(NN_undersample_cm)


# ## For oversampling

# In[ ]:


n_inputs = Xsm_train.shape[1]

NN_oversample_model = Sequential([
    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

NN_oversample_model.compile(Adam(lr=0.001), 
                            loss='sparse_categorical_crossentropy', metrics=['accuracy'])
NN_oversample_model.fit(Xsm_train, ysm_train, 
                        validation_split=0.2, batch_size=300, 
                        epochs=20, shuffle=True, verbose=2)


# In[ ]:


NN_oversample_fraud_predictions = NN_oversample_model.predict_classes(ori_Xtest, 
                                                                        batch_size=200, verbose=0)

NN_oversample_cm = confusion_matrix(ori_ytest, NN_oversample_fraud_predictions)

print("Confusion Matrix using Neural Network and Oversampling")
print(NN_oversample_cm)

