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
    roc_curve, auc, classification_report
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
.stApp { background-color: #ffffff; }
section[data-testid="stSidebar"] { background-color: #f7f8fc; border-right: 1px solid #e8eaf0; }
.sidebar-title { font-size: 22px; font-weight: 700; color: #1a1a1a; margin-bottom: 22px; padding-bottom: 12px; border-bottom: 1px solid #e0e2e8; }
.sidebar-label { font-size: 14px; font-weight: 700; color: #1a1a1a; margin: 14px 0 5px 0; }
.stButton > button { background-color: #3b78d8 !important; color: white !important; border: none !important; border-radius: 6px !important; padding: 12px 18px !important; font-weight: 700 !important; width: 100% !important; margin-top: 20px !important; }
.stButton > button:hover { background-color: #2c64bd !important; }
.app-title { font-size: 24px; font-weight: 700; color: #1a1a1a; margin: 0; }
.section-title { font-size: 19px; font-weight: 700; margin: 24px 0 12px 0; color: #1a1a1a; }
.metric-box { background: #ffffff; border: 1px solid #e2e5ec; border-radius: 7px; padding: 14px 18px; box-shadow: 0 1px 2px rgba(0,0,0,0.03); height: 100%; }
.metric-label { font-size: 14px; color: #1a1a1a; margin: 0 0 8px 0; font-weight: 700; }
.metric-value { font-size: 26px; font-weight: 700; color: #2e7d32; margin: 0; line-height: 1.1; }
.caption-text { font-size: 12px; color: #999; margin-top: 8px; }
.block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; }
</style>
""", unsafe_allow_html=True)

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
            sidebar_label("Test size (%)")
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
        st.info("📂 Silakan upload dataset pada sidebar **Pengaturan** untuk memulai.")
    else:
        # --- EDA SECTION (BARU) ---
        st.markdown('<div class="section-title">Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
        
        col_eda1, col_eda2 = st.columns(2)
        
        with col_eda1:
            st.write("**Distribusi Kelas Target**")
            # Karena data besar, kita hitung frekuensinya saja
            class_counts = df_sup[target_col].value_counts()
            fig_dist, ax_dist = plt.subplots(figsize=(5, 3))
            sns.barplot(x=class_counts.index.astype(str), y=class_counts.values, palette="Blues_r", ax=ax_dist)
            ax_dist.set_ylabel("Jumlah")
            st.pyplot(fig_dist)
            if len(class_counts) == 2 and (class_counts.min() / class_counts.max()) < 0.1:
                st.warning("⚠️ Dataset ini sangat tidak seimbang (Imbalanced). Disarankan mencentang **SMOTE** di sidebar.")

        with col_eda2:
            st.write("**Correlation Matrix (dengan Target)**")
            # Ambil sampel maksimal 1000 baris agar tidak crash saat menghitung korelasi
            sample_df = df_sup.sample(n=min(1000, len(df_sup)), random_state=42)
            corr = sample_df.corr(numeric_only=True)[[target_col]].sort_values(by=target_col, ascending=False)
            fig_corr, ax_corr = plt.subplots(figsize=(4, 4))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr, cbar=False)
            st.pyplot(fig_corr)

        st.markdown('<div class="section-title">Cuplikan Dataset</div>', unsafe_allow_html=True)
        st.dataframe(df_sup.head(5), use_container_width=True, hide_index=True)

        # --- TRAINING & EVALUATION ---
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
                            st.warning("SMOTE gagal diterapkan.")

                    if algo == "Random Forest":
                        model = RandomForestClassifier(n_estimators=100, max_depth=param_val, random_state=42, n_jobs=-1)
                    else:
                        model = SVC(C=param_val, kernel="rbf", random_state=42, probability=True)

                    model.fit(X_train_s, y_train)
                    y_pred = model.predict(X_test_s)
                    y_prob = model.predict_proba(X_test_s)[:, 1] if hasattr(model, "predict_proba") else y_pred

                # --- HASIL EVALUASI ---
                st.markdown('<div class="section-title">Hasil Evaluasi</div>', unsafe_allow_html=True)
                avg = "weighted" if len(np.unique(y)) > 2 else "binary"
                
                c1, c2, c3, c4 = st.columns(4)
                with c1: metric_box("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
                with c2: metric_box("Precision", f"{precision_score(y_test, y_pred, average=avg, zero_division=0):.3f}")
                with c3: metric_box("Recall", f"{recall_score(y_test, y_pred, average=avg, zero_division=0):.3f}")
                with c4: metric_box("F1-Measure", f"{f1_score(y_test, y_pred, average=avg, zero_division=0):.3f}")

                # --- VISUALISASI HASIL (BARU) ---
                st.markdown('<div class="section-title">Visualisasi Analisis Model</div>', unsafe_allow_html=True)
                col_v1, col_v2 = st.columns(2)

                with col_v1:
                    st.write("**Confusion Matrix**")
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Greys", cbar=False, ax=ax_cm, annot_kws={"size": 12, "weight": "bold"})
                    ax_cm.set_xlabel("Predicted", fontsize=9)
                    ax_cm.set_ylabel("Actual", fontsize=9)
                    st.pyplot(fig_cm)

                with col_v2:
                    # Cek jika klasifikasi biner, tampilkan ROC Curve
                    if len(np.unique(y)) == 2:
                        st.write("**ROC Curve & AUC**")
                        fpr, tpr, _ = roc_curve(y_test, y_prob)
                        roc_auc = auc(fpr, tpr)
                        fig_roc, ax_roc = plt.subplots(figsize=(4.5, 3.5))
                        ax_roc.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.3f}")
                        ax_roc.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
                        ax_roc.set_xlabel("False Positive Rate")
                        ax_roc.set_ylabel("True Positive Rate")
                        ax_roc.legend(loc="lower right")
                        st.pyplot(fig_roc)
                    else:
                        st.info("ROC Curve hanya ditampilkan untuk klasifikasi biner.")

                # --- FEATURE IMPORTANCE (BARU) ---
                if algo == "Random Forest":
                    st.markdown('<div class="section-title">Feature Importance (Top 10)</div>', unsafe_allow_html=True)
                    importances = model.feature_importances_
                    feat_imp = pd.DataFrame({"Feature": X.columns, "Importance": importances}).sort_values(by="Importance", ascending=False).head(10)
                    
                    fig_imp, ax_imp = plt.subplots(figsize=(8, 3))
                    sns.barplot(data=feat_imp, x="Importance", y="Feature", palette="viridis", ax=ax_imp)
                    st.pyplot(fig_imp)
                    st.markdown('<div class="caption-text">Fitur apa saja yang paling berpengaruh dalam mendeteksi fraud.</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat training: {e}")

# =========================================================
# UNSUPERVISED LEARNING (Tetap sama, tidak diubah)
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
            feature_cols = st.multiselect("Fitur", options=numeric_cols, default=numeric_cols[:2] if len(numeric_cols)>=2 else numeric_cols, label_visibility="collapsed")
            
            sidebar_label("Algorithm")
            algo_u = st.selectbox("Algorithm", options=["K-Means", "DBSCAN"], label_visibility="collapsed")
            
            if algo_u == "K-Means":
                sidebar_label("Jumlah klaster (K)")
                n_clusters = st.number_input("K", min_value=2, max_value=20, value=3, step=1, label_visibility="collapsed")
            else:
                sidebar_label("eps")
                eps_val = st.number_input("eps", min_value=0.1, max_value=10.0, value=0.5, step=0.1, label_visibility="collapsed")
                sidebar_label("min_samples")
                min_samples_val = st.number_input("min_samples", min_value=2, max_value=50, value=5, step=1, label_visibility="collapsed")
                
            tombol_cluster = st.button("Mulai Clustering", use_container_width=True)

    if unsup_file is None:
        st.info("📂 Silakan upload dataset pada sidebar **Pengaturan** untuk memulai.")
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
                        ax.set_xlabel("fitur 1", fontsize=10, color="#666")
                        ax.set_ylabel("fitur 2", fontsize=10, color="#666")
                        ax.set_xticks([]); ax.set_yticks([])
                        for spine in ax.spines.values(): spine.set_color("#c8ccd4")
                        st.pyplot(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan: {e}")