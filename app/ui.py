import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FairLens — FinTech Bias Compliance",
    page_icon="🛡️",
    layout="wide",
)

API_URL = "http://api:80"

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🛡️ FairLens: FinTech Bias Detection & Repair")
st.markdown(
    "Automated loan portfolio compliance audit using the **80% Disparate Impact Rule** "
    "(ECOA / Fair Housing Act standard)."
)
st.divider()

# ── Sidebar: Configuration ─────────────────────────────────────────────────────
st.sidebar.header("📂 Dataset Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Loan Dataset (CSV)", type="csv")

if not uploaded_file:
    st.info("👈 Upload a CSV file from the sidebar to begin the compliance audit.")
    st.stop()

# ── Load Data ──────────────────────────────────────────────────────────────────
df_raw = pd.read_csv(uploaded_file)
df_clean = df_raw.dropna()

rows_dropped = len(df_raw) - len(df_clean)

st.subheader("📋 Dataset Preview")
col_info1, col_info2, col_info3 = st.columns(3)
col_info1.metric("Total Rows", len(df_raw))
col_info2.metric("Rows After Cleaning", len(df_clean))
col_info3.metric("Rows Dropped (nulls)", rows_dropped)
st.dataframe(df_clean.head(10), use_container_width=True)

st.divider()

# ── Sidebar: Audit Parameters ──────────────────────────────────────────────────
st.sidebar.header("⚙️ Audit Parameters")

all_cols = list(df_clean.columns)

protected_col = st.sidebar.selectbox(
    "Protected Class Column",
    all_cols,
    help="The column representing a protected attribute (e.g. Gender, Race).",
)

target_col = st.sidebar.selectbox(
    "Decision Column",
    [c for c in all_cols if c != protected_col],
    help="The column containing the loan decision (e.g. Loan_Approved).",
)

# ── FIX: Sort unique values so correct value is visible; user picks from actual data ──
unique_target_vals = sorted(df_clean[target_col].astype(str).unique().tolist())
pos_val = st.sidebar.selectbox(
    "✅ Approved Value (Positive Outcome)",
    unique_target_vals,
    help="Select the value that means APPROVED. e.g. 'Approved' or '1'.",
)

unique_group_vals = sorted(df_clean[protected_col].astype(str).unique().tolist())

# ── FIX: Dropdowns instead of free-text inputs — eliminates typo mismatches ──
unp_val = st.sidebar.selectbox(
    "Unprivileged Group",
    unique_group_vals,
    help="The group expected to face disadvantage (e.g. Female, Black).",
)

pri_val = st.sidebar.selectbox(
    "Privileged Group",
    [v for v in unique_group_vals if v != unp_val],
    help="The reference/majority group (e.g. Male, White).",
)

# ── FIX: Validate that unprivileged ≠ privileged before running ───────────────
if unp_val == pri_val:
    st.sidebar.error("⚠️ Unprivileged and Privileged groups must be different.")
    st.stop()

st.sidebar.divider()

run_audit_btn   = st.sidebar.button("🔍 Run Compliance Audit", use_container_width=True)
run_repair_btn  = st.sidebar.button("⚗️ Generate Fair Training Set", use_container_width=True)

# ── Compliance Audit ───────────────────────────────────────────────────────────
if run_audit_btn:
    st.subheader("📊 Compliance Audit Results")

    with st.spinner("Running bias audit..."):
        payload = {
            "data": df_clean.to_dict(orient="records"),
            "target": target_col,
            "protected_col": protected_col,
            "unprivileged_val": unp_val,
            "privileged_val": pri_val,
            "positive_val": pos_val,
        }
        try:
            res = requests.post(f"{API_URL}/audit", json=payload, timeout=60)
            res.raise_for_status()
            audit_data = res.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API connection error: {e}")
            st.stop()

    # ── Check for ERROR status from audit engine ───────────────────────
    if "ERROR" in str(audit_data.get("status", "")):
        st.error(f"⚠️ Audit Error: {audit_data['status']}")
        st.stop()

    dir_score = audit_data.get("disparate_impact_ratio", 0)
    status    = audit_data.get("status", "")
    is_fail   = "FAIL" in status

    # ── Scorecard ─────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Disparate Impact Ratio", f"{dir_score:.4f}")
    c2.metric(
        f"Approval Rate ({unp_val})",
        f"{audit_data.get('unprivileged_approval_rate', 0):.1%}",
    )
    c3.metric(
        f"Approval Rate ({pri_val})",
        f"{audit_data.get('privileged_approval_rate', 0):.1%}",
    )
    c4.metric(
        f"Group Sizes",
        f"{audit_data.get('unprivileged_group_size', 0)} / {audit_data.get('privileged_group_size', 0)}",
    )

    if is_fail:
        st.error(f"🚨 Verdict: {status}")
        st.markdown(
            f"> **Interpretation:** The `{unp_val}` group is approved at only "
            f"**{dir_score * 100:.1f}%** the rate of the `{pri_val}` group. "
            f"The legal threshold is **80%** (DIR ≥ 0.8). This model **violates** "
            f"the ECOA / Fair Housing Act standard."
        )
    else:
        st.success(f"✅ Verdict: {status}")
        st.markdown(
            f"> **Interpretation:** The `{unp_val}` group is approved at "
            f"**{dir_score * 100:.1f}%** the rate of the `{pri_val}` group. "
            f"This is within the legal 80%–125% range. The model is **compliant**."
        )

    # ── DIR Gauge Visual ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 1.2))
    ax.barh(0, 1.25, color="#e0e0e0", height=0.4)
    ax.barh(0, 0.8,  color="#ffcdd2", height=0.4)
    ax.barh(0, min(dir_score, 1.25), color="#ef5350" if is_fail else "#43a047", height=0.4)
    ax.axvline(0.8,  color="#c62828", lw=2, linestyle="--", label="80% threshold")
    ax.axvline(1.25, color="#1565c0", lw=2, linestyle="--", label="125% threshold")
    ax.scatter([dir_score], [0], color="white", s=150, zorder=5)
    ax.set_xlim(0, 1.4)
    ax.set_yticks([])
    ax.set_xlabel("Disparate Impact Ratio")
    ax.set_title(f"DIR = {dir_score:.4f}  |  Fair Zone: 0.80 – 1.25")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("#fafafa")
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # ── Proxy / Redlining Check ───────────────────────────────────────
    st.subheader("🔍 Proxy Variable / Redlining Risk Analysis")
    st.caption(
        "Variables with high Mutual Information scores against the protected column "
        "may be acting as hidden proxies (redlining risk)."
    )

    with st.spinner("Running proxy analysis..."):
        try:
            proxy_res = requests.post(
                f"{API_URL}/proxy-check",
                json={
                    "data": df_clean.to_dict(orient="records"),
                    "protected_col": protected_col,
                    "threshold": 0.05,
                },
                timeout=60,
            )
            proxy_data = proxy_res.json()
        except Exception as e:
            st.warning(f"Proxy check failed: {e}")
            proxy_data = {}

    if "error" in proxy_data:
        st.warning(proxy_data["error"])
    elif "proxies_detected" in proxy_data:
        proxies = proxy_data["proxies_detected"]
        risk    = proxy_data.get("risk_level", "MEDIUM")

        if risk == "HIGH":
            st.error(f"🚨 HIGH RISK — {len(proxies)} proxy variable(s) detected")
        else:
            st.warning(f"⚠️ MEDIUM RISK — {len(proxies)} proxy variable(s) detected")

        proxy_df = pd.DataFrame(
            {"Feature": list(proxies.keys()), "MI Score": list(proxies.values())}
        )
        fig2, ax2 = plt.subplots(figsize=(7, max(2, len(proxies) * 0.5)))
        colors = ["#ef5350" if v > 0.2 else "#ff8a65" for v in proxy_df["MI Score"]]
        ax2.barh(proxy_df["Feature"], proxy_df["MI Score"], color=colors)
        ax2.set_xlabel("Mutual Information Score")
        ax2.set_title("Proxy Risk — Features Correlated with Protected Column")
        ax2.axvline(0.05, color="gray", lw=1.5, linestyle="--", label="threshold=0.05")
        ax2.legend(fontsize=8)
        ax2.set_facecolor("#fafafa")
        fig2.patch.set_facecolor("#fafafa")
        st.pyplot(fig2)
        plt.close(fig2)
    else:
        st.success("✅ No proxy variables detected above threshold.")

# ── Repair / Remediation ───────────────────────────────────────────────────────
if run_repair_btn:
    st.subheader("⚗️ Synthetic Data Remediation")
    st.caption(
        "SMOTENC generates synthetic samples for the underrepresented group, "
        "producing a balanced training dataset without altering real records."
    )

    with st.spinner("Generating balanced dataset (SMOTENC)..."):
        try:
            repair_res = requests.post(
                f"{API_URL}/repair",
                json={
                    "data": df_clean.to_dict(orient="records"),
                    "target": target_col,
                    "protected_col": protected_col,
                },
                timeout=120,
            )
            repair_res.raise_for_status()
            repaired_df = pd.DataFrame(repair_res.json())
        except Exception as e:
            st.error(f"Repair failed: {e}")
            st.stop()

    st.success(
        f"✅ Dataset rebalanced! Original: {len(df_clean)} rows → "
        f"Balanced: {len(repaired_df)} rows"
    )

    # Before / After comparison
    fig3, axes = plt.subplots(1, 2, figsize=(12, 4))

    orig_counts = df_clean[protected_col].value_counts()
    rep_counts  = repaired_df[protected_col].value_counts() if protected_col in repaired_df.columns else pd.Series()

    colors_orig = sns.color_palette("Reds_r", len(orig_counts))
    colors_rep  = sns.color_palette("Greens_r", len(rep_counts))

    axes[0].bar(orig_counts.index, orig_counts.values, color=colors_orig)
    axes[0].set_title("Original Portfolio (Biased)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].set_facecolor("#fafafa")

    axes[1].bar(rep_counts.index, rep_counts.values, color=colors_rep)
    axes[1].set_title("Balanced Portfolio (Fair)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Count")
    axes[1].set_facecolor("#fafafa")

    fig3.patch.set_facecolor("#fafafa")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    # Approval rate comparison
    if target_col in repaired_df.columns and protected_col in repaired_df.columns:
        st.markdown("**Approval rates after balancing:**")
        rate_df = repaired_df.groupby(protected_col)[target_col].apply(
            lambda x: (x.astype(str) == str(pos_val)).mean()
        ).reset_index()
        rate_df.columns = ["Group", "Approval Rate"]
        rate_df["Approval Rate"] = rate_df["Approval Rate"].map("{:.1%}".format)
        st.dataframe(rate_df, use_container_width=True)

    st.download_button(
        label="⬇️ Download Compliance-Ready Dataset (CSV)",
        data=repaired_df.to_csv(index=False),
        file_name="fairlens_balanced_dataset.csv",
        mime="text/csv",
        use_container_width=True,
    )
