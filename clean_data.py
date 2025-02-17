import numpy as np
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def clean_data(df):
    missing_values = ["nan", "Not done", "TBD", "N/A", "Missing", "unknown dose", "<missing cytogenetics"]
    df.replace(missing_values, np.nan, inplace=True)
    
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    # Zielvariablen von der Standardisierung ausschließen
    target_cols = ['efs', 'efs_time']
    num_cols_to_scale = [col for col in num_cols if col not in target_cols]

    # Negative Werte in efs_time auf 0 setzen
    if (df['efs_time'] < 0).any():
        print("Warnung: Negative Werte in 'efs_time' gefunden. Werden korrigiert.")
        df['efs_time'] = df['efs_time'].clip(lower=0)
    
    df['efs_time'] = df['efs_time'].astype(int)

    # Sicherstellen, dass 'efs' nur 0 und 1 enthält
    # Fehlerbehebung: 'efs' sollte binär sein, daher auf 0 und 1 setzen
    print(df['efs'].unique())
    df['efs'] = df['efs'].fillna(0)  # Fehlende Werte in 'efs' mit 0 auffüllen
    df['efs'] = (df['efs'] > 0).astype(int)  # Umwandeln in 0 oder 1: alles >0 wird zu 1, ansonsten 0

    # Ausgabe zur Kontrolle
    print("Eindeutige Werte in 'efs' nach Bereinigung:", df['efs'].unique())

    # Füllen fehlender Werte in anderen Spalten
    for col in df.columns:
        missing_ratio = df[col].isna().sum() / len(df)
        
        if missing_ratio > 0.6:
            print(f"Spalte '{col}' hat mehr als 60% fehlende Werte, diese werden mit 0 gefüllt.")
            df[col].fillna(0, inplace=True)
        else:
            if col in num_cols:
                if missing_ratio > 0:
                    print(f"Numerische Spalte '{col}': Fehlende Werte mit Median auffüllen.")
                    df[col].fillna(df[col].median(), inplace=True)
            elif col in cat_cols:
                if missing_ratio > 0:
                    print(f"Kategorische Spalte '{col}': Fehlende Werte mit häufigstem Wert auffüllen.")
                    df[col].fillna(df[col].mode()[0], inplace=True)

    # Standardisierung der numerischen Spalten (außer 'efs' und 'efs_time')
    scaler = StandardScaler()
    df[num_cols_to_scale] = scaler.fit_transform(df[num_cols_to_scale])
    
    return df
