import pandas as pd
import os

def make_limited_expert(model_name, language, threshold, base_path='./output/Language/'):
    """Extract top-N language-specific neurons by AP score."""
    top_file = f'{base_path}{model_name}/sense/{language}/expertise/expertise_limited_{int(threshold)}_top.csv'

    df = pd.read_csv(f'{base_path}{model_name}/sense/{language}/expertise/expertise.csv')
    print(len(df))

    # Top N by average precision
    df2 = df.sort_values('ap', ascending=False)
    df2 = df2.head(int(threshold))
    print(len(df2))
    print(df2.head())

    df2.to_csv(top_file, index=False)
