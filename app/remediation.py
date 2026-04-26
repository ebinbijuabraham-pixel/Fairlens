import pandas as pd
from imblearn.over_sampling import SMOTENC, SMOTE, RandomOverSampler


class SyntheticRepairEngine:
    def generate_fair_data(
        self,
        df: pd.DataFrame,
        target: str,
        protected_col: str,
    ):
        # ── Validate ───────────────────────────────────────────────────
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataset.")
        if protected_col not in df.columns:
            raise ValueError(f"Protected column '{protected_col}' not found in dataset.")

        df_clean = df.copy().dropna()

        if len(df_clean) < 10:
            raise ValueError("Dataset too small after dropping nulls (need at least 10 rows).")

        # Identify categorical columns (excluding target)
        cat_cols = list(df_clean.select_dtypes(include=["object", "category"]).columns)
        if target in cat_cols:
            cat_cols.remove(target)

        X = df_clean.drop(columns=[target])
        y = df_clean[target]

        # Composite label: "Approved_Female" so SMOTE balances across
        # BOTH outcome and protected class simultaneously
        comp_y = y.astype(str) + "_" + df_clean[protected_col].astype(str)

        # Get categorical feature indices (required by SMOTENC)
        cat_idx = [X.columns.get_loc(c) for c in cat_cols if c in X.columns]

        min_samples = comp_y.value_counts().min()

        # Choose sampler based on group sizes
        if min_samples <= 1:
            # Too few samples — just duplicate
            sampler = RandomOverSampler(random_state=42)
        elif min_samples <= 5:
            k = min_samples - 1
            sampler = (
                SMOTENC(categorical_features=cat_idx, k_neighbors=k, random_state=42)
                if cat_idx
                else SMOTE(k_neighbors=k, random_state=42)
            )
        else:
            k = 5
            sampler = (
                SMOTENC(categorical_features=cat_idx, k_neighbors=k, random_state=42)
                if cat_idx
                else SMOTE(k_neighbors=k, random_state=42)
            )

        X_syn, comp_y_syn = sampler.fit_resample(X, comp_y)

        df_syn = pd.DataFrame(X_syn, columns=X.columns)

        # Recover original target from composite label
        df_syn[target] = comp_y_syn.str.split("_").str[0].astype(y.dtype)

        return df_syn
