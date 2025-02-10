import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer

def clean_data(df):
    # Definieren der zu ersetzenden fehlenden Werte
    missing_values = ["nan", "Not done", "TBD", "N/A", "Missing", "unknown dose", "<missing cytogenetics"]
    df.replace(missing_values, np.nan, inplace=True)
    
    # Identifizieren der numerischen und kategorialen Spalten
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    
    # Zielvariablen von der Standardisierung ausschließen
    target_cols = ['efs', 'efs_time']
    num_cols_to_scale = [col for col in num_cols if col not in target_cols]

    # Fehlerbehandlung: Negative Werte in 'efs_time' auf 0 setzen
    if (df['efs_time'] < 0).any():
        print("Warnung: Negative Werte in 'efs_time' gefunden. Werden korrigiert.")
        df['efs_time'] = df['efs_time'].clip(lower=0)
    
    df['efs_time'] = df['efs_time'].astype(int)

    # Sicherstellen, dass 'efs' nur 0 und 1 enthält (binär kodiert)
    print("Eindeutige Werte in 'efs' vor Bereinigung:", df['efs'].unique())
    df['efs'] = df['efs'].fillna(0)  # Fehlende Werte in 'efs' mit 0 auffüllen
    df['efs'] = (df['efs'] > 0).astype(int)  # Umwandeln in 0 oder 1

    # Ausgabe zur Kontrolle
    print("Eindeutige Werte in 'efs' nach Bereinigung:", df['efs'].unique())

    # Fehlende Werte in anderen Spalten füllen
    for col in df.columns:
        missing_ratio = df[col].isna().sum() / len(df)
        
        if missing_ratio > 0.6:
            print(f"Spalte '{col}' hat mehr als 60% fehlende Werte, diese werden mit 0 gefüllt.")
            df[col].fillna(0, inplace=True)
        else:
            if col in num_cols:
                if missing_ratio > 0:
                    print(f"Numerische Spalte '{col}': Fehlende Werte mit Median auffüllen.")
                    # Hier könnte ein komplexerer Imputer wie KNN oder ein Modell sinnvoller sein
                    df[col].fillna(df[col].median(), inplace=True)
            elif col in cat_cols:
                if missing_ratio > 0:
                    print(f"Kategorische Spalte '{col}': Fehlende Werte mit häufigstem Wert auffüllen.")
                    df[col].fillna(df[col].mode()[0], inplace=True)

    # Skalierung der numerischen Spalten (außer Zielvariablen)
    # Je nach Modell und Anwendungsfall könnte auch ein anderer Skalierer sinnvoll sein:
    # 1. StandardScaler (zentrale Standardisierung)
    # 2. MinMaxScaler (Skalierung auf [0, 1])
    # 3. RobustScaler (weniger anfällig für Ausreißer)
    scaler = StandardScaler()  # oder RobustScaler() oder MinMaxScaler()
    df[num_cols_to_scale] = scaler.fit_transform(df[num_cols_to_scale])
    
    return df