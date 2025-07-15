# cleaning_functions.py

import pandas as pd
import numpy as np



def fill_categorical_with_mode(df: pd.DataFrame, column: str):
    mode_val = df[column].mode()[0]
    df[column] = df[column].fillna(mode_val)
    return df

def remove_duplicates(df):
    df = df.drop_duplicates()
    return df


# Convert 'date_time' column to datetime
def conv_datetime (df_web):
    df_web['date_time'] = pd.to_datetime(df_web['date_time'], errors='coerce')
    return df_web

def quick_data_report(df: pd.DataFrame):
    print("\nDataFrame shape:", df.shape)
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nDescriptive statistics:\n", df.describe())
    print("\nColumns:", df.columns)


def main_cleaning(df):
    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # Now apply normal cleaning
    df = remove_duplicates(df)
    return df
