import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

def clean_data(df, is_train=True):
    """
    Bereinigt den DataFrame:
    - Ersetzt fehlende Werte (NaN) mit sinnvollen Standardwerten.
    - Wendet OneHotEncoding auf kategorische Spalten an.
    - Standardisiert numerische Spalten.
    - Bereitet Zielspalten (efs, efs_time) vor.
    """
    # Definiere eine Liste von Platzhaltern für fehlende Werte
    missing_values = ["nan", "Not done", "TBD", "N/A", "Missing", "unknown dose", "<missing cytogenetics"]
    df.replace(missing_values, np.nan, inplace=True)

    # Identifiziere numerische und kategorische Spalten
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Entferne 'efs' und 'efs_time' aus den numerischen Spalten, um sie speziell zu behandeln
    num_cols = [col for col in num_cols if col not in ['efs', 'efs_time']]

    # Behandlung von fehlenden Werten in numerischen und kategorischen Spalten
    for col in df.columns:
        missing_ratio = df[col].isna().sum() / len(df)

        if missing_ratio > 0.6:
            print(f"⚠️ Spalte '{col}' hat mehr als 60% fehlende Werte. Sie wird mit 0 gefüllt.")
            df[col].fillna(0, inplace=True)
        else:
            if col in num_cols:
                if missing_ratio > 0:
                    print(f"ℹ️ Numerische Spalte '{col}': Fehlende Werte mit Median auffüllen.")
                    df[col].fillna(df[col].median(), inplace=True)
            elif col in cat_cols:
                if missing_ratio > 0:
                    print(f"ℹ️ Kategorische Spalte '{col}': Fehlende Werte mit 'Unknown' auffüllen.")
                    df[col].fillna("Unknown", inplace=True)

    # Behandle die Zielspalten `efs` und `efs_time` nur im Trainingsset
    if is_train:
        # Falls nötig, sicherstellen, dass 'efs' nur 1 oder 0 ist
        if 'efs' in df.columns:
            df['efs'] = df['efs'].fillna(0)  # Wenn NaN vorhanden ist, mit 0 füllen
    
        # `efs_time` numerisch aufbereiten (NaN durch Median ersetzen)
        if 'efs_time' in df.columns:
             # Wenn 'efs' 0 ist, soll 'efs_time' ebenfalls 0 sein
            df.loc[df['efs'] == 0, 'efs_time'] = 0
            df['efs_time'] = pd.to_numeric(df['efs_time'], errors='coerce')
            df['efs_time'] = df['efs_time'].fillna(df['efs_time'].median())
        # Wenn 'efs' 0 ist soll efs_time auch 0 sein


    # Numerische Spalten standardisieren (außer `efs_time` und `efs`)
    num_cols = [col for col in num_cols if col not in ['efs', 'efs_time']]
    if num_cols:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])

    # Sicherstellen, dass alle Werte in kategorischen Spalten Strings sind
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    df[cat_cols] = df[cat_cols].astype(str)  # Erzwinge String-Datentyp für Encoder
    # Kategorische Spalten OneHotEncoden
    cat_cols = [col for col in cat_cols if col not in ['efs', 'efs_time']]  # `efs` und `efs_time` nicht einbeziehen
    
    if cat_cols:
        encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        cat_encoded = encoder.fit_transform(df[cat_cols])

        # OneHotEncoded-Spaltennamen setzen
        cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(cat_cols), index=df.index)

        # Ursprüngliche kategorische Spalten entfernen und die neuen hinzufügen
        df.drop(columns=cat_cols, inplace=True)
        df = pd.concat([df, cat_encoded_df], axis=1)
    
    # gebe nicht numerische spalten aus
    # Überprüfen, welche Spalten noch NaN-Werte enthalten
    nan_columns = df.columns[df.isna().any()]
    
    # Filtere nur die Spalten, die noch NaN-Werte enthalten und aus numerischen Typen sind
    numerical_nan_columns = [col for col in nan_columns if df[col].dtype in ['float64', 'int64']]
    
    # Ausgabe der Spalten, die noch NaN-Werte enthalten
    print("Numerische Spalten mit NaN-Werten:", numerical_nan_columns)
    problematische_spalten = [col for col in df.columns if any(c in col for c in ["[", "]", "<", ">", " "])]
    print("⚠️ Problematische Spalten:", problematische_spalten)
    # Spaltennamen bereinigen, um Sonderzeichen zu entfernen
    df.columns = df.columns.str.replace(r"[^a-zA-Z0-9_]", "_", regex=True)
    
    return df
