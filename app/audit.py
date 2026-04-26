import pandas as pd


class AuditEngine:
    def calculate_dir(
        self,
        df: pd.DataFrame,
        target: str,
        protected_col: str,
        unp_val: str,
        pri_val: str,
        pos_val=None,
    ):
        # ── Filter groups ──────────────────────────────────────────────
        u_df = df[df[protected_col].astype(str) == str(unp_val)]
        p_df = df[df[protected_col].astype(str) == str(pri_val)]

        if len(u_df) == 0:
            return {
                "disparate_impact_ratio": 0.0,
                "status": f"ERROR: Unprivileged group '{unp_val}' not found in column '{protected_col}'. "
                          f"Available values: {sorted(df[protected_col].astype(str).unique().tolist())}",
            }
        if len(p_df) == 0:
            return {
                "disparate_impact_ratio": 0.0,
                "status": f"ERROR: Privileged group '{pri_val}' not found in column '{protected_col}'. "
                          f"Available values: {sorted(df[protected_col].astype(str).unique().tolist())}",
            }

        # ── Validate pos_val ───────────────────────────────────────────
        if not pos_val or str(pos_val).strip().lower() in ("none", ""):
            return {
                "disparate_impact_ratio": 0.0,
                "status": "ERROR: No Approved Value selected. Please choose the positive outcome "
                          f"from: {sorted(df[target].astype(str).unique().tolist())}",
            }

        pos_val_str = str(pos_val).strip()
        available_target_vals = df[target].astype(str).unique().tolist()

        if pos_val_str not in available_target_vals:
            return {
                "disparate_impact_ratio": 0.0,
                "status": f"ERROR: Approved Value '{pos_val_str}' not found in column '{target}'. "
                          f"Available values: {sorted(available_target_vals)}",
            }

        # ── Calculate approval rates ───────────────────────────────────
        p_u = (u_df[target].astype(str) == pos_val_str).mean()
        p_p = (p_df[target].astype(str) == pos_val_str).mean()

        # ── Guard: zero approval in either group ───────────────────────
        if p_p == 0:
            return {
                "disparate_impact_ratio": 0.0,
                "status": f"ERROR: Privileged group '{pri_val}' has 0 approvals for value "
                          f"'{pos_val_str}'. Check your Approved Value selection.",
            }
        if p_u == 0:
            return {
                "disparate_impact_ratio": 0.0,
                "status": f"ERROR: Unprivileged group '{unp_val}' has 0 approvals for value "
                          f"'{pos_val_str}'. The group may be severely under-represented.",
            }

        # ── Compute DIR ────────────────────────────────────────────────
        dir_score = round(p_u / p_p, 4)

        # 80% Rule: FAIL if DIR < 0.8 (under-representation)
        # Also flag if DIR > 1.25 (reverse discrimination check)
        if dir_score < 0.8:
            status = "FAIL — BIAS DETECTED (Unprivileged group approval rate is below 80% of privileged group)"
        elif dir_score > 1.25:
            status = "FAIL — REVERSE BIAS DETECTED (Unprivileged group approval rate exceeds 125% of privileged group)"
        else:
            status = "PASS — COMPLIANT (Within the 80%–125% fairness range)"

        return {
            "disparate_impact_ratio": dir_score,
            "status": status,
            "unprivileged_approval_rate": round(p_u, 4),
            "privileged_approval_rate": round(p_p, 4),
            "unprivileged_group_size": len(u_df),
            "privileged_group_size": len(p_df),
        }
