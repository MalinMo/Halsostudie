import pandas as pd

def quality_check(df):
    """
    Kvalitetsgranskning av data, saknade värden, dubletter och ogiltiga värden.
    """
    nan_count = df.isna().sum()
    dup_count = df.duplicated().sum()

    invalid_disease = ~df["disease"].isin([0, 1]) & ~df["disease"].isna()
    invalid_sex = ~df["sex"].isin(["M", "F"]) & ~df["sex"].isna()
    invalid_smoker = ~df["smoker"].isin(["Yes", "No"]) & ~df["smoker"].isna()

    invalid_summary = pd.DataFrame({
        "Kolumn": ["disease", "sex", "smoker"],
        "Antal ogiltiga": [
            invalid_disease.sum(),
            invalid_sex.sum(),
            invalid_smoker.sum()
        ]
    })
    return {
        "nan_count": nan_count,
        "dup_count": dup_count,
        "invalid_summary": invalid_summary
    }

def clean_data(df):
    """
    Städar kolumnnamn och returnerar en ren DataFrame.
    """
    df_clean = df.copy()
    df_clean.columns = (
        df_clean.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.lower()
    )
    return df_clean