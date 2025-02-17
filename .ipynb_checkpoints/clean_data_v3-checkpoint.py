import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def clean_data(df):
    """
    Bereinigt einen DataFrame:
    - Ersetzt spezielle fehlende Werte durch NaN.
    - Konvertiert alle numerischen Spalten in float, entfernt nicht-numerische Werte.
    - Falls `efs_time` existiert, setzt negative Werte auf 0 und wandelt sie in `int` um.
    - Falls `efs` existiert, wandelt sie in binäre Werte (0/1) um.
    - Spalten mit >60% fehlenden Werten werden mit 0 gefüllt.
    - Fehlende Werte in numerischen Spalten werden mit Median ersetzt.
    - Fehlende Werte in kategorischen Spalten werden mit "Unknown" ersetzt.
    - Alle kategorischen Spalten werden mit `OneHotEncoder` umgewandelt.
    - Standardisiert numerische Spalten (außer `efs` und `efs_time`).

    Funktioniert für Trainings- und Testdaten.
    """

    # Definierte fehlende Werte-Texte ersetzen
    missing_values = ["nan", "Not done", "TBD", "N/A", "Missing", "unknown dose", "<missing cytogenetics"]
    df.replace(missing_values, np.nan, inplace=True)

    # Numerische & kategorische Spalten erkennen
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Sicherstellen, dass alle numerischen Spalten wirklich numerisch sind
    for col in df.columns:
        if col not in cat_cols:  # Nur für nicht-kategorische Spalten prüfen
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')  # Strings zu NaN konvertieren
            except Exception as e:
                print(f"⚠️ Fehler bei der Konvertierung von '{col}': {e}")

    # Falls `efs` und `efs_time` existieren, behandeln
    if 'efs' in df.columns and 'efs_time' in df.columns:
        # Negative Werte in `efs_time` auf 0 setzen
        if (df['efs_time'] < 0).any():
            print("⚠️ Warnung: Negative Werte in 'efs_time' gefunden. Sie werden auf 0 gesetzt.")
            df['efs_time'] = df['efs_time'].clip(lower=0)
        
        df['efs_time'] = df['efs_time'].astype(int)

        # `efs` binär machen (falls noch nicht geschehen)
        print("Eindeutige Werte in 'efs' vor Bereinigung:", df['efs'].unique())
        df['efs'] = df['efs'].fillna(0).astype(int)
        df['efs'] = (df['efs'] > 0).astype(int)
        print("Eindeutige Werte in 'efs' nach Bereinigung:", df['efs'].unique())

        # `efs` & `efs_time` NICHT standardisieren
        target_cols = ['efs', 'efs_time']
        num_cols = [col for col in num_cols if col not in target_cols]

    # Fehlende Werte füllen (numerische Spalten mit Median, kategorische mit "Unknown")
    for col in df.columns:
        missing_ratio = df[col].isna().sum() / len(df)

        if missing_ratio > 0.6:
            print(f"⚠️ Spalte '{col}' hat mehr als 60% fehlende Werte. Sie wird mit 0 gefüllt.")
            df[col].fillna(0, inplace=True)
        else:
            if col in num_cols:
                if missing_ratio > 0:
                    print(f"ℹ️ Numerische Spalte '{col}': Fehlende Werte mit Median ({df[col].median():.2f}) auffüllen.")
                    df[col].fillna(df[col].median(), inplace=True)
            elif col in cat_cols:
                if missing_ratio > 0:
                    print(f"ℹ️ Kategorische Spalte '{col}': Fehlende Werte mit 'Unknown' auffüllen.")
                    df[col].fillna("Unknown", inplace=True)

    # Kategorische Spalten mit OneHotEncoder umwandeln
    if cat_cols:
        encoder = OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore')
        cat_encoded = encoder.fit_transform(df[cat_cols])

        # OneHotEncoding-Spaltennamen setzen
        cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(cat_cols), index=df.index)

        # Ursprüngliche kategorische Spalten entfernen und neue hinzufügen
        df.drop(columns=cat_cols, inplace=True)
        df = pd.concat([df, cat_encoded_df], axis=1)

    # Standardisierung aller numerischen Spalten außer `efs` und `efs_time`
    if num_cols:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])

    return df
