import pandas as pd
from scipy.io import arff

def load_arff_data(filepath):
    """Loads an ARFF file into a pandas DataFrame.         
    """
    data, meta = arff.loadarff('/Users/poojithramagiri/Desktop/Ml_Modularised_Project/polish+companies+bankruptcy+data/2year.arff')
    df = pd.DataFrame(data)

    # Convert 'class' to strings (assuming it's the target column)
    df['class'] = df['class'].apply(lambda x: x.decode('utf-8')) 
    print(df)
    return df
