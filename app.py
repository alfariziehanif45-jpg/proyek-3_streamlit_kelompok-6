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
# CUSTOM CSS — MIRIP MOCKUP
# =========================================================
st.markdown("""
<style>
/* ---------- Background utama ---------- */
.stApp { background-color: #ffffff; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] { background-color: #f7f8fc; border-right: 1px solid #e8eaf0; }
section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }

/* ---------- Judul sidebar "Pengaturan" ---------- */
.sidebar-title { font-size: 22px; font-weight: 700; color: #1a1a1a; margin-bottom: 22px; padding-bottom: 12px; border-bottom: 1px solid #e0e2e8; }

/* ---------- Label sidebar ---------- */
.sidebar-label { font-size: 14px; font-weight: 700; color: #1a1a1a; margin: 14px 0 5px 0; }

/* ---------- Input widgets di sidebar ---------- */
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stNumberInput input,
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
    background-color: #ffffff !important; border: 1px solid #d9dce3 !important;
    border-radius: 5px !important; font-size: 14px !important; color: #333 !important; min-height: 40px !important;
}

/* ---------- File uploader ---------- */
section[data-testid="stSidebar"] section[data-testid="stFileUploadDropzone"] {
    background-color: #ffffff; border: 1px solid #d9dce3; border-radius: 5px; padding: 5px 10px; min-height: 40px;
}
section[data-testid="stSidebar"] section[data-testid="stFileUploadDropzone"] > div { padding: 0 !important; }
section[data-testid="stSidebar"] section[data-testid="stFileUploadDropzone"] span,
section[data-testid="stSidebar"] section[data-testid="stFileUploadDropzone"] small { font-size: 13px !important; color: #555 !important; }
section[data-testid="stSidebar"] section[data-testid="stFileUploadDropzone"] button {
    background-color: #eef1f6 !important; color: #333 !important; border: 1px solid #d9dce3 !important; padding: 4px 12px !important; font-size: 12px !important;
}

/* ---------- Tombol (Mulai Training / Mulai Clustering) ---------- */
.stButton > button {
    background-color: #3b78d8 !important; color: white !important; border: none !important;
    border-radius: 6px !important; padding: 12px 18px !important; font-weight: 700 !important;
    font-size: 15px !important; width: 100% !important; margin-top: 20px !important;
    box-shadow: 0 1px 3px rgba(59,120,216,0.3) !important;
}
.stButton > button:hover { background-color: #2c64bd !important; color: white !important; }

/* ---------- Judul halaman utama ---------- */
.app-title { font-size: 24px; font-weight: 700; color: #1a1a1a; margin: 0; padding: 0; }

/* ---------- Toggle mode (kanan atas) ---------- */
div[role="radiogroup"] {
    display: flex !important; flex-direction: row !important; justify-content: flex-end !important;
    gap: 0 !important; flex-wrap: nowrap !important; background-color: #f1f3f8; border-radius: 8px; padding: 4px;
}
div[role="radiogroup"] > label {
    background-color: transparent !important; padding: 7px 16px !important; border-radius: 6px !important;
    cursor: pointer !important; border: none !important; margin: 0 !important; transition: all 0.15s ease;
}
div[role="radiogroup"] > label p { font-size: 13.5px !important; font-weight: 500 !important; color: #8a8f99 !important; margin: 0 !important; }
div[role="radiogroup"] > label:hover p { color: #4a90e2 !important; }
div[role="radiogroup"] > label:has(input:checked) { background-color: #ffffff !important; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
div[role="radiogroup"] > label:has(input:checked) p { color: #3b78d8 !important; font-weight: 700 !important; }
div[role="radiogroup"] > label > div:first-child, div[role="radiogroup"] > label > span:first-child { display: none !important; }

/* ---------- Section Titles ---------- */
.section-title { font-size: 19px; font-weight: 700; margin: 24px 0 12px 0; color: #1a1a1a; }

/* ---------- Metric Box ---------- */
.metric-box { background: #ffffff; border: 1px solid #e2e5ec; border-radius: 7px; padding: 14px 18px; box-shadow: 0 1px 2px rgba(0,0,0,0.03); height: 100%; }
.metric-label { font-size: 14px; color: #1a1a1a; margin: 0 0 8px 0; font-weight: 700; }
.metric-value { font-size: 26px; font-weight: 700; color: #2e7d32; margin: 0; line-height: 1.1; }

/* ---------- Caption bawah ---------- */
.caption-text { font-size: 12px; color: #999; margin-top: 8px; font-style: normal; }

/* ---------- Tabel dataframe ---------- */
.stDataFrame { border: 1px solid #e2e5ec; border-radius: 5px; }

/* ---------- Hilangkan padding kosong atas ---------- */
.block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; }
</style>
""", unsafe_allow_html=True)


# =========================================================
# HELPER
# =========================================================
def metric_box(label, value):
    st.markdown(f'<div class="metric-box"><p class="metric-label">{label}</p><p class="metric-value">{value}</p></div>', unsafe_allow_html=True)

def sidebar_label(text):
    st.markdown(f'<div class="sidebar-label">{text}</div>', unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
head_l, head_r = st.columns([1.6, 1])
with head_l:
    st.markdown('<p class="app-title">Machine Learning Experiment</p>', unsafe_allow_html=True)
with head_r:
    mode = st.radio("mode", ["Supervised Learning", "Unsupervised Learning"], horizontal=True, label_visibility="collapsed")
st.markdown("<hr style='margin: 12px 0 18px 0; border: none; border-top: 1px solid #e8eaf0;'>", unsafe_allow_html=True)


# =========================================================
# ============== SUPERVISED LEARNING =====================
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
        st.markdown("""
        ### 📊 Rekomendasi Dataset
        **Credit Card Fraud Detection** — [Download di Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
        - Kolom target: `Class` (0 = sah, 1 = fraud)
        - Centang **SMOTE** untuk menyeimbangkan data
        """)
    else:
        # 1. Cuplikan Dataset (Sesuai Mockup)
        st.markdown('<div class="section-title">Cuplikan Dataset</div>', unsafe_allow_html=True)
        st.dataframe(df_sup.head(6), use_container_width=True, hide_index=True)

        # Fitur Analisis Tambahan (EDA) ditaruh di Expander agar tidak merusak tampilan awal
        with st.expander("📊 Lihat Analisis Data (EDA)"):
            col_eda1, col_eda2 = st.columns(2)
            with col_eda1:
                st.write("**Distribusi Kelas Target**")
                class_counts = df_sup[target_col].value_counts()
                fig_dist, ax_dist = plt.subplots(figsize=(5, 3))
                sns.barplot(x=class_counts.index.astype(str), y=class_counts.values, palette="Blues_r", ax=ax_dist)
                ax_dist.set_ylabel("Jumlah")
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

        # 2. Hasil Training
        if tombol_train:
            try:
                with st.spinner("Melatih model..."):
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
                            st.warning("SMOTE gagal diterapkan, training dilanjutkan tanpa SMOTE.")

                    if algo == "Random Forest":
                        model = RandomForestClassifier(n_estimators=100, max_depth=param_val, random_state=42, n_jobs=-1)
                    else:
                        model = SVC(C=param_val, kernel="rbf", random_state=42, probability=True)

                    model.fit(X_train_s, y_train)
                    y_pred = model.predict(X_test_s)
                    y_prob = model.predict_proba(X_test_s)[:, 1] if hasattr(model, "predict_proba") else y_pred

                # 3. Hasil Evaluasi (Sesuai Mockup)
                st.markdown('<div class="section-title">Hasil Evaluasi</div>', unsafe_allow_html=True)
                avg = "weighted" if len(np.unique(y)) > 2 else "binary"
                
                c1, c2, c3, c4 = st.columns(4)
                with c1: metric_box("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
                with c2: metric_box("Precision", f"{precision_score(y_test, y_pred, average=avg, zero_division=0):.2f}")
                with c3: metric_box("Recall", f"{recall_score(y_test, y_pred, average=avg, zero_division=0):.2f}")
                with c4: metric_box("F1-Measure", f"{f1_score(y_test, y_pred, average=avg, zero_division=0):.2f}")

                # 4. Confusion Matrix (Sesuai Mockup)
                st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
                cm = confusion_matrix(y_test, y_pred)
                fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.2))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Greys", cbar=False, ax=ax_cm, annot_kws={"size": 13, "weight": "bold"}, linewidths=0.5, linecolor="#ffffff")
                ax_cm.set_xlabel("")
                ax_cm.set_ylabel("")
                ax_cm.tick_params(labelsize=9)
                plt.tight_layout()
                st.pyplot(fig_cm, use_container_width=False)
                st.markdown('<div class="caption-text">Visualisasi hasil prediksi model</div>', unsafe_allow_html=True)

                # 5. Fitur Analisis Tambahan (ROC & Feature Importance) di Expander
                with st.expander("📈 Lihat Analisis Model Lanjutan (ROC Curve & Feature Importance)"):
                    col_analisis1, col_analisis2 = st.columns(2)
                    
                    with col_analisis1:
                        if len(np.unique(y)) == 2:
                            st.write("**ROC Curve & AUC**")
                            fpr, tpr, _ = roc_curve(y_test, y_prob)
                            roc_auc = auc(fpr, tpr)
                            fig_roc, ax_roc = plt.subplots(figsize=(5, 3.5))
                            ax_roc.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.3f}")
                            ax_roc.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
                            ax_roc.set_xlabel("False Positive Rate")
                            ax_roc.set_ylabel("True Positive Rate")
                            ax_roc.legend(loc="lower right")
                            st.pyplot(fig_roc)
                        else:
                            st.info("ROC Curve hanya untuk klasifikasi biner.")
                            
                    with col_analisis2:
                        if algo == "Random Forest":
                            st.write("**Feature Importance (Top 10)**")
                            importances = model.feature_importances_
                            feat_imp = pd.DataFrame({"Feature": X.columns, "Importance": importances}).sort_values(by="Importance", ascending=False).head(10)
                            fig_imp, ax_imp = plt.subplots(figsize=(6, 3.5))
                            sns.barplot(data=feat_imp, x="Importance", y="Feature", palette="viridis", ax=ax_imp)
                            st.pyplot(fig_imp)
                        else:
                            st.info("Feature Importance hanya tersedia untuk Random Forest.")

            except Exception as e:
                st.error(f"❌ Terjadi kesalahan: {e}")

# =========================================================
# ============== UNSUPERVISED LEARNING ===================
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
        st.markdown("""
        ### 📊 Rekomendasi Dataset
        **Mall Customers Segmentation** — [Download di Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
        - Fitur: `Annual Income (k$)`, `Spending Score (1-100)`
        """)
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