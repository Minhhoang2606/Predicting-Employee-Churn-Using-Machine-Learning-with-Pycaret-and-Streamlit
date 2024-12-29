'''
Predicting Employee Churn Using Machine Learning
Author: Henry Ha
'''
import pandas as pd

data = pd.read_csv('HR_comma_sep.csv')

# Rename 'sales' column to 'department'
data.rename(columns={'sales': 'department'}, inplace=True)

# Display dataset information
data.info()
print(data.head())

# Display summary statistics
print(data.describe())

# Visualize the distribution of employee churn i.e target variable
import seaborn as sns
import matplotlib.pyplot as plt

# Aggregating data for better visualization
churn_counts = data['left'].value_counts()

# Bar plot for churn distribution
sns.barplot(x=churn_counts.index, y=churn_counts.values, palette='viridis')
plt.title('Employee Churn Distribution')
plt.xlabel('Left (1 = Yes, 0 = No)')
plt.ylabel('Count')
plt.xticks([0, 1], ['Stayed', 'Left'])
plt.show()

# Correlation heatmap
corr_matrix = data.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Outlier detection with boxplots
numerical_features = ['satisfaction_level', 'last_evaluation', 'number_project',
                      'average_montly_hours', 'time_spend_company']

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for i, feature in enumerate(numerical_features):
    ax = axes[i//3, i%3]
    sns.boxplot(x=data[feature], ax=ax)
    ax.set_title(f'Boxplot of {feature}')
plt.tight_layout()
plt.show()

# Department-wise employee count
sns.countplot(y='department', hue='left', data=data)
plt.title('Employee Distribution by Department')
plt.xlabel('Count')
plt.ylabel('Department')
plt.show()

# Salary distribution and churn
sns.countplot(x='salary', hue='left', data=data)
plt.title('Salary Level and Churn')
plt.xlabel('Salary Level')
plt.ylabel('Count')
plt.show()

#TODO Data Preprocessing

# Remove duplicates if any
data.drop_duplicates(inplace=True)

# Example: Capping extreme values for 'average_monthly_hours'
data['average_montly_hours'] = data['average_montly_hours'].clip(lower=100, upper=300)

# Verify and update data types if necessary
data['department'] = data['department'].astype('category')
data['salary'] = data['salary'].astype('category')
data['left'] = data['left'].astype('int')

#TODO Model Building and Evaluation

from pycaret.classification import *

# Setting up the PyCaret environment
clf = setup(data=data, target='left',
            fix_imbalance=True,  # Handle class imbalance
            normalize=True)      # Scale numerical features


# Compare models and select the best one
best_model = compare_models()
print(best_model)

# Hyperparameter tuning
tuned_model = tune_model(best_model)

# Evaluate the model
plot_model(tuned_model, plot='confusion_matrix')
plot_model(tuned_model, plot='feature')
plot_model(tuned_model, plot='auc')

# Finalize the model
final_model = finalize_model(tuned_model)

import joblib

# Save the finalized model
joblib.dump(final_model, 'employee_churn_model.pkl')