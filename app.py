import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, silhouette_score,
    roc_curve, auc
)
from sklearn.cluster import KMeans, DBSCAN
from imblearn.over_sampling import SMOTE

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Machine Learning Experiment",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
.stApp { background-color: #ffffff; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
section[data-testid="stSidebar"] { background-color: #f7f8fc; border-right: 1px solid #e0e2e8; }
section[data-testid="stSidebar"] > div:first-child { padding-top: 2rem; }
.sidebar-label { font-size: 13.5px; font-weight: 700; color: #1a1a1a; margin: 18px 0 6px 0; }
.sidebar-title { font-size: 22px; font-weight: 800; color: #1a1a1a; margin-bottom: 25px; padding-bottom: 15px; border-bottom: 1px solid #e0e0e0; }
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stNumberInput input,
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
    background-color: #ffffff !important; border: 1px solid #d1d5db !important; border-radius: 4px !important; font-size: 14px !important; color: #333 !important; min-height: 38px !important; box-shadow: none !important;
}
section[data-testid="stSidebar"] .stFileUploader section[data-testid="stFileUploadDropzone"] {
    background-color: #ffffff !important; border: 1px solid #d1d5db !important; border-radius: 4px !important; padding: 10px !important;
}
.stButton > button { background-color: #3b78d8 !important; color: white !important; border: none !important; border-radius: 6px !important; padding: 12px 20px !important; font-weight: 700 !important; font-size: 15px !important; width: 100% !important; margin-top: 25px !important; }
.stButton > button:hover { background-color: #2c64bd !important; }
.app-title { font-size: 24px; font-weight: 800; color: #1a1a1a; margin: 0; padding: 0; }
div[role="radiogroup"] { display: flex !important; flex-direction: row !important; justify-content: flex-end !important; gap: 8px !important; flex-wrap: nowrap !important; }
div[role="radiogroup"] > label { background-color: #ffffff !important; padding: 8px 20px !important; border-radius: 6px !important; cursor: pointer !important; border: 1px solid #d1d5db !important; margin: 0 !important; transition: all 0.2s; }
div[role="radiogroup"] > label p { font-size: 14px !important; font-weight: 600 !important; color: #4b5563 !important; margin: 0 !important; }
div[role="radiogroup"] > label:has(input:checked) { background-color: #e0edff !important; border-color: #93c5fd !important; }
div[role="radiogroup"] > label:has(input:checked) p { color: #2563eb !important; font-weight: 700 !important; }
div[role="radiogroup"] > label > div:first-child, div[role="radiogroup"] > label > span:first-child { display: none !important; }
.section-title { font-size: 18px; font-weight: 700; color: #1f2937; margin: 25px 0 15px 0; }
.metric-box { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px 20px; box-shadow: 0 1px 2px rgba(0,0,0,0.02); height: 100%; }
.metric-label { font-size: 13px; color: #4b5563; margin: 0 0 10px 0; font-weight: 600; }
.metric-value { font-size: 26px; font-weight: 800; color: #2e7d32; margin: 0; line-height: 1; }
.caption-text { font-size: 12px; color: #9ca3af; margin-top: 12px; font-style: italic; }
.block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; max-width: 95% !important; }
</style>
""", unsafe_allow_html=True)

def metric_box(label, value):
    st.markdown(f'<div class="metric-box"><p class="metric-label">{label}</p><p class="metric-value">{value}</p></div>', unsafe_allow_html=True)

def sidebar_label(text):
    st.markdown(f'<div class="sidebar-label">{text}</div>', unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
head_l, head_r = st.columns([1.5, 1], vertical_alignment="center")
with head_l:
    st.markdown('<p class="app-title">Machine Learning Experiment</p>', unsafe_allow_html=True)
with head_r:
    mode = st.radio("mode", ["Supervised Learning", "Unsupervised Learning"], horizontal=True, label_visibility="collapsed")
st.markdown("<hr style='margin: 10px 0 25px 0; border: none; border-top: 1px solid #e5e7eb;'>", unsafe_allow_html=True)

# =========================================================
# SUPERVISED LEARNING
# =========================================================
if mode == "Supervised Learning":
    with st.sidebar:
        st.markdown('<div class="sidebar-title">Pengaturan</div>', unsafe_allow_html=True)
        target_col, test_size_pct, algo, param_val, use_smote, tombol_train, df_sup = None, 20, "Random Forest", 5, False, False, None

        sidebar_label("Dataset")
        sup_file = st.file_uploader("Dataset", type=["csv", "xlsx"], key="sup_file", label_visibility="collapsed")

        if sup_file is not None:
            df_sup = pd.read_csv(sup_file) if sup_file.name.endswith(".csv") else pd.read_excel(sup_file)
            
            sidebar_label("Target column")
            target_col = st.selectbox("Target column", options=df_sup.columns, index=len(df_sup.columns) - 1, label_visibility="collapsed")
            
            sidebar_label("Test size")
            test_size_pct = st.number_input("Test size", min_value=5, max_value=50, value=20, step=5, label_visibility="collapsed")
            
            sidebar_label("Algorithm")
            algo = st.selectbox("Algorithm", options=["Random Forest", "SVM"], label_visibility="collapsed")
            
            if algo == "Random Forest":
                sidebar_label("max_depth")
                param_val = st.number_input("max_depth", min_value=1, max_value=50, value=5, step=1, label_visibility="collapsed")
            else:
                sidebar_label("C (Regularization)")
                param_val = st.number_input("C (Regularization)", min_value=0.1, max_value=100.0, value=1.0, step=0.1, label_visibility="collapsed")
            
            sidebar_label("SMOTE (Opsional)")
            use_smote = st.checkbox("Gunakan SMOTE", value=False, label_visibility="collapsed", help="Menyeimbangkan kelas minoritas")
            
            tombol_train = st.button("Mulai Training", use_container_width=True)

    if sup_file is None:
        st.markdown('<div class="section-title">Cuplikan Dataset</div>', unsafe_allow_html=True)
        st.info("📂 Silakan upload dataset (CSV/Excel) pada sidebar **Pengaturan** untuk memulai.")
    else:
        # 1. Cuplikan Dataset
        st.markdown('<div class="section-title">Cuplikan Dataset</div>', unsafe_allow_html=True)
        st.dataframe(df_sup.head(6), use_container_width=True, hide_index=True)

        # 2. EDA (Bisa dilihat sebelum training)
        with st.expander("📊 Lihat Analisis Data Awal (EDA)"):
            col_eda1, col_eda2 = st.columns(2)
            with col_eda1:
                st.write("**Distribusi Kelas Target**")
                class_counts = df_sup[target_col].value_counts()
                fig_dist, ax_dist = plt.subplots(figsize=(5, 3))
                sns.barplot(x=class_counts.index.astype(str), y=class_counts.values, palette="Blues_r", ax=ax_dist)
                st.pyplot(fig_dist)
                if len(class_counts) == 2 and (class_counts.min() / class_counts.max()) < 0.1:
                    st.warning("⚠️ Dataset sangat tidak seimbang. Disarankan mencentang **SMOTE**.")
            with col_eda2:
                st.write("**Correlation Matrix (dengan Target)**")
                sample_df = df_sup.sample(n=min(1000, len(df_sup)), random_state=42)
                corr = sample_df.corr(numeric_only=True)[[target_col]].sort_values(by=target_col, ascending=False)
                fig_corr, ax_corr = plt.subplots(figsize=(4, 4))
                sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr, cbar=False)
                st.pyplot(fig_corr)

        # 3. Hasil Training (Muncul setelah tombol diklik)
        if tombol_train:
            try:
                with st.spinner("Melatih model... Mohon tunggu (untuk dataset besar ini bisa memakan waktu 1-2 menit)"):
                    X = df_sup.drop(columns=[target_col])
                    y = df_sup[target_col]
                    if y.dtype == "object" or y.dtype.name == "category":
                        y = LabelEncoder().fit_transform(y)
                    X = pd.get_dummies(X)

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_pct/100.0, random_state=42, stratify=y)
                    scaler = StandardScaler()
                    X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)

                    if use_smote:
                        try:
                            X_train_s, y_train = SMOTE(random_state=42).fit_resample(X_train_s, y_train)
                        except Exception:
                            st.warning("SMOTE gagal diterapkan.")

                    if algo == "Random Forest":
                        model = RandomForestClassifier(n_estimators=100, max_depth=param_val, random_state=42, n_jobs=-1)
                    else:
                        model = SVC(C=param_val, kernel="rbf", random_state=42, probability=True)

                    model.fit(X_train_s, y_train)
                    y_pred = model.predict(X_test_s)
                    y_prob = model.predict_proba(X_test_s)[:, 1] if hasattr(model, "predict_proba") else y_pred

                st.markdown('<div class="section-title">Hasil Evaluasi</div>', unsafe_allow_html=True)
                avg = "weighted" if len(np.unique(y)) > 2 else "binary"
                
                c1, c2, c3, c4 = st.columns(4)
                with c1: metric_box("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
                with c2: metric_box("Precision", f"{precision_score(y_test, y_pred, average=avg, zero_division=0):.2f}")
                with c3: metric_box("Recall", f"{recall_score(y_test, y_pred, average=avg, zero_division=0):.2f}")
                with c4: metric_box("F1-Measure", f"{f1_score(y_test, y_pred, average=avg, zero_division=0):.2f}")

                st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
                cm = confusion_matrix(y_test, y_pred)
                fig_cm, ax_cm = plt.subplots(figsize=(3.5, 2.5))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Greys", cbar=False, ax=ax_cm, annot_kws={"size": 12, "weight": "bold"}, linewidths=0.5, linecolor="#ffffff")
                ax_cm.set_xlabel("")
                ax_cm.set_ylabel("")
                ax_cm.tick_params(labelsize=9)
                plt.tight_layout()
                st.pyplot(fig_cm, use_container_width=False)
                st.markdown('<div class="caption-text">Visualisasi hasil prediksi model</div>', unsafe_allow_html=True)

                with st.expander("📈 Lihat Analisis Model Lanjutan (ROC & Feature Importance)"):
                    col_analisis1, col_analisis2 = st.columns(2)
                    with col_analisis1:
                        if len(np.unique(y)) == 2:
                            st.write("**ROC Curve & AUC**")
                            fpr, tpr, _ = roc_curve(y_test, y_prob)
                            roc_auc = auc(fpr, tpr)
                            fig_roc, ax_roc = plt.subplots(figsize=(4.5, 3))
                            ax_roc.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.3f}")
                            ax_roc.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
                            ax_roc.legend(loc="lower right")
                            st.pyplot(fig_roc)
                    with col_analisis2:
                        if algo == "Random Forest":
                            st.write("**Feature Importance (Top 10)**")
                            importances = model.feature_importances_
                            feat_imp = pd.DataFrame({"Feature": X.columns, "Importance": importances}).sort_values(by="Importance", ascending=False).head(10)
                            fig_imp, ax_imp = plt.subplots(figsize=(5, 3))
                            sns.barplot(data=feat_imp, x="Importance", y="Feature", palette="viridis", ax=ax_imp)
                            st.pyplot(fig_imp)

            except Exception as e:
                st.error(f"❌ Terjadi kesalahan: {e}")

# =========================================================
# UNSUPERVISED LEARNING
# =========================================================
else:
    with st.sidebar:
        st.markdown('<div class="sidebar-title">Pengaturan</div>', unsafe_allow_html=True)
        feature_cols, algo_u, n_clusters, eps_val, min_samples_val, tombol_cluster, df_unsup = None, "K-Means", 3, 0.5, 5, False, None
        
        sidebar_label("Dataset")
        unsup_file = st.file_uploader("Dataset", type=["csv", "xlsx"], key="unsup_file", label_visibility="collapsed")

        if unsup_file is not None:
            df_unsup = pd.read_csv(unsup_file) if unsup_file.name.endswith(".csv") else pd.read_excel(unsup_file)
            numeric_cols = df_unsup.select_dtypes(include=np.number).columns.tolist()
            
            sidebar_label("Fitur yang dianalisis")
            feature_cols = st.multiselect("Fitur yang dianalisis", options=numeric_cols, default=numeric_cols[:2] if len(numeric_cols)>=2 else numeric_cols, label_visibility="collapsed")
            
            sidebar_label("Algorithm")
            algo_u = st.selectbox("Algorithm", options=["K-Means", "DBSCAN"], label_visibility="collapsed")
            
            if algo_u == "K-Means":
                sidebar_label("Jumlah klaster (K)")
                n_clusters = st.number_input("Jumlah klaster (K)", min_value=2, max_value=20, value=3, step=1, label_visibility="collapsed")
            else:
                sidebar_label("eps")
                eps_val = st.number_input("eps", min_value=0.1, max_value=10.0, value=0.5, step=0.1, label_visibility="collapsed")
                sidebar_label("min_samples")
                min_samples_val = st.number_input("min_samples", min_value=2, max_value=50, value=5, step=1, label_visibility="collapsed")
                
            tombol_cluster = st.button("Mulai Clustering", use_container_width=True)

    if unsup_file is None:
        st.markdown('<div class="section-title">Cuplikan Dataset</div>', unsafe_allow_html=True)
        st.info("📂 Silakan upload dataset (CSV/Excel) pada sidebar **Pengaturan** untuk memulai.")
    else:
        st.markdown('<div class="section-title">Cuplikan Dataset</div>', unsafe_allow_html=True)
        st.dataframe(df_unsup.head(6), use_container_width=True, hide_index=True)

        if tombol_cluster:
            if not feature_cols or len(feature_cols) < 2:
                st.warning("⚠️ Pilih minimal **2 fitur numerik** untuk clustering.")
            else:
                try:
                    with st.spinner("Menjalankan clustering..."):
                        X = df_unsup[feature_cols].dropna()
                        X_s = StandardScaler().fit_transform(X)

                        if algo_u == "K-Means":
                            labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X_s)
                        else:
                            labels = DBSCAN(eps=eps_val, min_samples=min_samples_val).fit_predict(X_s)

                        n_unique = len(set(labels)) - (1 if -1 in labels else 0)
                        sil = silhouette_score(X_s, labels) if n_unique > 1 else None

                    col_a, col_b = st.columns([1, 2.2])
                    with col_a:
                        st.markdown('<div class="section-title">Hasil Clustering</div>', unsafe_allow_html=True)
                        metric_box("Silhouette Score", f"{sil:.2f}" if sil is not None else "N/A")
                    with col_b:
                        st.markdown('<div class="section-title">Scatter Plot</div>', unsafe_allow_html=True)
                        fig, ax = plt.subplots(figsize=(6.5, 4.2))
                        ax.scatter(X_s[:, 0], X_s[:, 1], c="#3b78d8", s=55, alpha=1.0, edgecolor="white", linewidth=0.8, zorder=2)
                        if len(X_s) >= 5:
                            highlight_idx = np.linspace(0, len(X_s) - 1, 4).astype(int)
                            ax.scatter(X_s[highlight_idx, 0], X_s[highlight_idx, 1], facecolors="none", edgecolors="#e8a33d", s=90, linewidth=1.8, zorder=3)
                        ax.set_xlabel("fitur 1", fontsize=10, color="#666")
                        ax.set_ylabel("fitur 2", fontsize=10, color="#666")
                        ax.set_xticks([]); ax.set_yticks([])
                        for spine in ax.spines.values(): spine.set_color("#c8ccd4")
                        st.pyplot(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan: {e}")