import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder


class ProxyDetector:
    def detect_proxies(
        self,
        df: pd.DataFrame,
        protected_col: str,
        threshold: float = 0.05,
    ):
        # ── Validate ───────────────────────────────────────────────────
        if protected_col not in df.columns:
            return {"error": f"Column '{protected_col}' not found in dataset."}

        df_clean = df.copy().dropna()

        if len(df_clean) == 0:
            return {"error": "Dataset is empty after dropping nulls."}

        if df_clean[protected_col].nunique() < 2:
            return {"error": f"Protected column '{protected_col}' has fewer than 2 unique values."}

        # ── Encode every column with its own LabelEncoder ──────────────
        # Use a fresh encoder per column to avoid cross-column state bleed.
        # Cast to str first so mixed-type / nullable columns are handled cleanly.
        for col in df_clean.columns:
            if df_clean[col].dtype == "object" or df_clean[col].dtype.name == "category":
                df_clean[col] = LabelEncoder().fit_transform(df_clean[col].astype(str))

        # Force ALL remaining columns to float — catches booleans,
        # nullable Int64, and any other non-standard dtypes sklearn rejects.
        for col in df_clean.columns:
            try:
                df_clean[col] = df_clean[col].astype(float)
            except (ValueError, TypeError):
                # Last resort: label-encode whatever couldn't be cast
                df_clean[col] = LabelEncoder().fit_transform(
                    df_clean[col].astype(str)
                ).astype(float)

        X = df_clean.drop(columns=[protected_col])
        y = df_clean[protected_col]

        # Mutual Information: how much does knowing X reveal about protected_col?
        try:
            mi_scores = mutual_info_classif(
                X, y, discrete_features="auto", random_state=42
            )
        except Exception as e:
            return {"error": f"Mutual Information calculation failed: {str(e)}"}

        proxies = {
            feature: round(float(score), 4)
            for feature, score in zip(X.columns, mi_scores)
            if score > threshold
        }

        # Sort by score descending so highest-risk proxies appear first
        proxies = dict(sorted(proxies.items(), key=lambda x: x[1], reverse=True))

        if not proxies:
            return {"result": "No proxy variables detected above threshold.", "threshold": threshold}

        return {
            "proxies_detected": proxies,
            "threshold": threshold,
            "risk_level": "HIGH" if max(proxies.values()) > 0.2 else "MEDIUM",
        }
