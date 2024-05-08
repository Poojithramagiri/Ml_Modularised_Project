import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

def histograms(df, target_column='class'):
    """Creates histograms for each feature in the DataFrame, split by target.

    Args:
        df (pandas.DataFrame): The DataFrame to analyze.
        target_column (str, optional): Name of the target column. Defaults to 'class'.
    """

    num_features = len(df.columns) - 1 
    fig, axes = plt.subplots(nrows=num_features, figsize=(10, 5 * num_features))

    for i, column in enumerate(df.columns):
        if column != target_column:
            sns.histplot(
                data=df, 
                x=column, 
                hue=target_column,  # Color by target
                kde=True, 
                ax=axes[i]
            )
            axes[i].set_title(f'Histogram of {column}')

    fig.tight_layout()
    plt.show()

def correlation_heatmap(df, target_column='class'):
    """Calculates and plots a correlation heatmap.

    Args:
        df (pandas.DataFrame): The DataFrame to analyze.
        target_column (str, optional): Name of the target column. Defaults to 'class'.
    """

    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")
    plt.show()

def boxplots(df, target_column='class'):
    """Creates box plots for each feature, grouped by the target variable.

    Args:
        df (pandas.DataFrame): The DataFrame to analyze.
        target_column (str, optional): Name of the target column. Defaults to 'class'.
    """

    num_features = len(df.columns) - 1
    fig, axes = plt.subplots(nrows=num_features, figsize=(10, 5 * num_features))

    for i, column in enumerate(df.columns):
        if column != target_column:
            sns.boxplot(
                data=df,
                x=target_column,
                y=column,
                ax=axes[i]
            )
            axes[i].set_title(f'Box Plot of {column} vs {target_column}')

    fig.tight_layout()
    plt.show()

def scatterplots(df, target_column='class'):
    """Creates scatter plots of each feature vs. the target variable.

    Args:
        df (pandas.DataFrame): The DataFrame to analyze.
        target_column (str, optional): Name of the target column. Defaults to 'class'.
    """

    num_features = len(df.columns) - 1
    fig, axes = plt.subplots(nrows=num_features, figsize=(10, 5 * num_features))

    for i, column in enumerate(df.columns):
        if column != target_column:
            sns.scatterplot(
                data=df,
                x=column,
                y=target_column,
                hue=target_column, 
                ax=axes[i]
            )
            axes[i].set_title(f'Scatter Plot of {column} vs {target_column}')

    fig.tight_layout()
    plt.show()
