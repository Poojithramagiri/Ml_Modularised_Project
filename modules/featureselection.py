from sklearn.feature_selection import SelectKBest, f_classif, SequentialFeatureSelector
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

def select_k_best(X, y, k='all'):
    """Selects the top 'k' features using the F-test score.

    Args:
        X (array-like): Input data.
        y (array-like): Target variable.
        k (int or 'all', optional): Number of features to select. Defaults to 'all'.

    Returns:
        array-like: Indices of selected features.
    """
    selector = SelectKBest(f_classif, k=k)
    selector.fit(X, y)
    return selector.get_support(indices=True)


def pca_selection(X, n_components=2):
    """Selects features using Principal Component Analysis (PCA).

    Args:
        X (array-like): Input data.
        n_components (int, optional): Number of principal components to keep. Defaults to 2.

    Returns:
        array-like: Transformed data with selected components.
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca


def decision_tree_selection(X, y): 
    """Selects features based on feature importance from a Decision Tree model

    Args: 
        X (array-like): Input data.
        y (array-like): Target variable.

    Returns:
        array-like: Indices of selected features.
    """
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X, y)
    importances = dt_model.feature_importances_
    selected_feature_indices = importances.argsort()[::-1][:10]  # Select top 10 important features
    return selected_feature_indices


def forward_feature_selection(X, y, n_features_to_select=10, estimator=LogisticRegression()):
    """Performs forward feature selection using a Logistic Regression model.
    
    Args:
        X (array-like): Input data.
        y (array-like): Target variable.
        n_features_to_select (int, optional): Number of features to select. Defaults to 10. 
        estimator (sklearn estimator, optional): Estimator for feature evaluation. Defaults to LogisticRegression().
    """
    
    sfs = SequentialFeatureSelector(estimator, n_features_to_select=n_features_to_select, direction='forward')
    sfs.fit(X, y)
    return sfs.transform(X)

def lasso_selection(X, y, penalty=1.0, max_iter=1000): 
    """Performs feature selection using Lasso (L1) regularization.

    Args:
        X (array-like): Input data.
        y (array-like): Target variable.
        penalty (float, optional): Strength of L1 penalty. Higher values lead to sparser selection. Defaults to 1.0.
        max_iter (int, optional): Maximum iterations for the solver. Defaults to 1000.

    Returns: 
        array-like: Indices of selected features.
    """
    model = LogisticRegression(penalty='l1', C=1.0/penalty, solver='liblinear', max_iter=max_iter)
    model.fit(X, y)
    selected_feature_indices = model.coef_.nonzero()[0]  # Features with non-zero coefficients 
    return selected_feature_indices 

def feature_selection_pipeline(X, y, method="SFS", **kwargs):
    """Creates a feature selection pipeline.

    Args:
        X (array-like): Input data.
        y (array-like): Target variable.
        method (str, optional): Feature selection method. Options: 'kBest', 'PCA', 'SFS', 'DecisionTree', 'Lasso'. Defaults to 'SFS'.
        **kwargs: Keyword arguments for the selected feature selection method.
                Refer to the documentation of the respective methods for valid kwargs.

    Returns:
        sklearn.pipeline.Pipeline: Feature selection pipeline.
    """

    if method == 'kBest':
        k = kwargs.get('k', 'all')
        return X[:, select_k_best(X, y, k)] 

    elif method == 'PCA':
        n_components = kwargs.get('n_components', 2)
        return pca_selection(X, n_components)

    elif method == 'DecisionTree':
        return X[:, decision_tree_selection(X, y)] 

    elif method == 'SFS':
        estimator = kwargs.get('estimator', LogisticRegression())
        n_features_to_select = kwargs.get('n_features_to_select', 10)
        sfs = SequentialFeatureSelector(estimator, n_features_to_select=n_features_to_select, direction='forward')
        return Pipeline([('feature_selection', sfs)]) 

    elif method == 'Lasso': 
        penalty = kwargs.get('penalty', 1.0)
        max_iter = kwargs.get('max_iter', 1000)
        return X[:, lasso_selection(X, y, penalty, max_iter)]  

    else:
        raise ValueError(f"Invalid feature selection method: {method}") 

