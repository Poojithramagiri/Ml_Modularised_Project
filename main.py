import modules.dataload as ld  
import modules.datapreprocess as dp
import modules.Visualisations as viz
import modules.featureselection as fs
import modules.models as md
import pandas as pd
import numpy as np 

# 1. Data Loading
df = ld.load_arff_data('/Users/poojithramagiri/Desktop/Ml_Modularised_Project/polish+companies+bankruptcy+data/2year.arff')  # Replace with the actual path

# 2. Preprocessing
df_scaled = dp.preprocess(df.copy()) 

# 3. Visualization - Explore the preprocessed data
viz.histograms(df_scaled)
viz.correlation_heatmap(df_scaled)
viz.boxplots(df_scaled) 
viz.scatterplots(df_scaled) # Add scatterplots

# 4. Feature Selection
X = df_scaled.drop('class', axis=1)
y = df_scaled['class']

# Choose a feature selection method (uncomment the one you want to use):

# SFS (Sequential Forward Selection)
X_selected = fs.feature_selection_pipeline(X, y, method='SFS', n_features_to_select=15)


# PCA (Principal Component Analysis)
X_selected = fs.feature_selection_pipeline(X, y, method='PCA', n_components=8)

# Lasso (L1 regularization)
# X_selected = fs.feature_selection_pipeline(X, y, method='Lasso')

# kBest (select features with high F-test score)
# X_selected = fs.select_k_best(X, y, k='all')  # Select all features
# X_selected = fs.select_k_best(X, y, k=10)  # Select top 10 features

# Decision Tree Feature Importance
# X_selected = fs.decision_tree_selection(X, y)  # Select top 10 features based on importance

# Forward Feature Selection (sequential selection)
# X_selected = fs.forward_feature_selection(X, y, n_features_to_select=15)  # Select 15 features


# 5. Model Training and Evaluation
md.train_models(X_selected, y) 


