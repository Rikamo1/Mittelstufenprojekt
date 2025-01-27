import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def clean_data(df):
    missing_values = ["nan", "Not done", "TBD", "N/A", "Missing", "unknown dose", "<missing cytogenetics"]
    df.replace(missing_values, np.nan, inplace=True)
    
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    # Negative Werte in efs_time auf 0 setzen
    if (df['efs_time'] < 0).any():
        print("Warnung: Negative Werte in 'efs_time' gefunden. Werden korrigiert.")
        df['efs_time'] = df['efs_time'].clip(lower=0)
    df['efs_time'] = df['efs_time'].astype(int)
    # fill nan values in efs with 0
    df['efs'].fillna(0)
    df['efs'].astype(bool)
    # Anzahl der Einträge in der 'efs'-Spalte
    print("Anzahl Einträge in efs: ", len(df['efs']))

    # Anzahl der Werte, die 0 sind
    print("Anzahl 0 in efs: ", (df['efs'] == 0).sum())

    # Anzahl der Werte, die 1 sind
    print("Anzahl 1 in efs: ", (df['efs'] == 1).sum())
    print("Minimale Zeit:", df["efs_time"].min())
    print("Anzahl der negativen Werte:", (df["efs_time"] < 0).sum())
    print("Anzahl der Nullen:", (df["efs_time"] == 0).sum())
    
    for col in df.columns:
        missing_ratio = df[col].isna().sum() / len(df)
        
        if missing_ratio > 0.6:
            print(f"Spalte '{col}' hat mehr als 60% fehlende Werte, diese werden mit 0 gefüllt.")
            #füllen von nan werten mit 0
            df[col].fillna(0)
        else:
            if col in num_cols:
                if missing_ratio > 0:
                    print(f"Numerische Spalte '{col}': Fehlende Werte mit Median auffüllen.")
                    df[col].fillna(df[col].median(), inplace=True)
            elif col in cat_cols:
                if missing_ratio > 0:
                    print(f"Kategorische Spalte '{col}': Fehlende Werte mit häufigstem Wert auffüllen.")
                    df[col].fillna(df[col].mode()[0], inplace=True)


    # Standardisierung der numerischen Spalten
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    return df