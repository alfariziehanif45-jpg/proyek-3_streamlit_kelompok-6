import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ===================== KONFIGURASI HALAMAN =====================
st.set_page_config(
    page_title="Machine Learning Experiment",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Machine Learning Experiment")

# ===================== TAB UTAMA =====================
tab1, tab2 = st.tabs(["Supervised Learning", "Unsupervised Learning"])


# ================================================================
# TAB 1 — SUPERVISED LEARNING
# ================================================================
with tab1:
    st.header("Supervised Learning")

    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.subheader("⚙️ Pengaturan")
        st.markdown("**Dataset**")

        uploaded_file = st.file_uploader(
            "Upload dataset (CSV/Excel)",
            type=["csv", "xlsx"],
            key="sup_file"
        )

        # Load default iris jika user tidak upload
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        else:
            try:
                df = pd.read_csv("dataset/iris.csv")
                st.caption("Menggunakan dataset default: iris.csv")
            except FileNotFoundError:
                df = pd.DataFrame()

        if not df.empty:
            target_col = st.selectbox("Target column", df.columns, index=len(df.columns) - 1)

            test_size = st.slider("Test size (%)", 10, 50, 20, step=5) / 100

            algorithm = st.selectbox("Algorithm", ["Random Forest", "SVM"])

            if algorithm == "Random Forest":
                n_estimators = st.slider("n_estimators", 10, 300, 100, step=10)
                max_depth = st.slider("max_depth", 1, 20, 5)
                params = {"n_estimators": n_estimators, "max_depth": max_depth}
            else:
                C = st.slider("C (Regularization)", 0.1, 10.0, 1.0, step=0.1)
                kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"])
                params = {"C": C, "kernel": kernel}

            train_btn = st.button("🚀 Mulai Training", use_container_width=True)
        else:
            train_btn = False

    # ---------- MAIN PANEL ----------
    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.subheader("Cuplikan Dataset")
        if not df.empty:
            st.dataframe(df.head(10), use_container_width=True)
        else:
            st.warning("Dataset belum tersedia.")

    with col2:
        st.subheader("Informasi Dataset")
        if not df.empty:
            st.write(f"Jumlah baris: **{df.shape[0]}**")
            st.write(f"Jumlah kolom: **{df.shape[1]}**")
            st.write(f"Target: **{target_col}**")

    # ---------- TRAINING ----------
    if train_btn and not df.empty:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Encode kolom kategorikal
        X = pd.get_dummies(X)
        if y.dtype == "object":
            y = LabelEncoder().fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Inisialisasi model
        if algorithm == "Random Forest":
            model = RandomForestClassifier(**params, random_state=42)
        else:
            model = SVC(**params)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        st.markdown("---")
        st.subheader("📈 Hasil Evaluasi")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{acc:.2f}")
        c2.metric("Precision", f"{prec:.2f}")
        c3.metric("Recall", f"{rec:.2f}")
        c4.metric("F1-Measure", f"{f1:.2f}")

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    cbar=False, annot_kws={"size": 12})
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)


# ================================================================
# TAB 2 — UNSUPERVISED LEARNING
# ================================================================
with tab2:
    st.header("Unsupervised Learning")

    with st.sidebar:
        st.markdown("---")
        st.subheader("⚙️ Pengaturan Clustering")
        st.markdown("**Dataset**")

        uploaded_file_unsup = st.file_uploader(
            "Upload dataset (CSV/Excel)",
            type=["csv", "xlsx"],
            key="unsup_file"
        )

        if uploaded_file_unsup is not None:
            if uploaded_file_unsup.name.endswith(".csv"):
                df_u = pd.read_csv(uploaded_file_unsup)
            else:
                df_u = pd.read_excel(uploaded_file_unsup)
        else:
            try:
                df_u = pd.read_csv("dataset/customer_data.csv")
                st.caption("Menggunakan dataset default: customer_data.csv")
            except FileNotFoundError:
                df_u = pd.DataFrame()

        if not df_u.empty:
            numeric_cols = df_u.select_dtypes(include=np.number).columns.tolist()
            features = st.multiselect(
                "Fitur yang dianalisis",
                numeric_cols,
                default=numeric_cols[:2]
            )

            algo_u = st.selectbox("Algorithm", ["K-Means", "DBSCAN"])

            if algo_u == "K-Means":
                k = st.slider("Jumlah cluster (K)", 2, 10, 3)
                params_u = {"n_clusters": k}
            else:
                eps = st.slider("eps", 0.1, 5.0, 0.5, step=0.1)
                min_samples = st.slider("min_samples", 2, 20, 5)
                params_u = {"eps": eps, "min_samples": min_samples}

            cluster_btn = st.button("🚀 Mulai Clustering", use_container_width=True)
        else:
            cluster_btn = False

    # ---------- MAIN PANEL ----------
    colA, colB = st.columns([1, 1.4])

    with colA:
        st.subheader("Cuplikan Dataset")
        if not df_u.empty:
            st.dataframe(df_u.head(10), use_container_width=True)
        else:
            st.warning("Dataset belum tersedia.")

    # ---------- CLUSTERING ----------
    if cluster_btn and not df_u.empty and len(features) >= 2:
        data = df_u[features].dropna()
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        if algo_u == "K-Means":
            model_u = KMeans(**params_u, random_state=42, n_init=10)
            labels = model_u.fit_predict(data_scaled)
        else:
            model_u = DBSCAN(**params_u)
            labels = model_u.fit_predict(data_scaled)

        # Silhouette score hanya jika cluster >1
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters > 1:
            sil = silhouette_score(data_scaled, labels)
        else:
            sil = -1

        with colB:
            st.subheader("Hasil Clustering")
            st.metric("Silhouette Score", f"{sil:.2f}" if sil >= 0 else "N/A")

            # Scatter Plot
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            scatter = ax2.scatter(
                data.iloc[:, 0], data.iloc[:, 1],
                c=labels, cmap="viridis", s=60, edgecolor="white"
            )
            ax2.set_xlabel(features[0])
            ax2.set_ylabel(features[1])
            ax2.set_title("Scatter Plot Cluster")
            st.pyplot(fig2)

            # Distribusi klaster
            dist = pd.Series(labels).value_counts().sort_index()
            st.write("**Distribusi Anggota Cluster:**")
            st.dataframe(dist.rename("Jumlah").to_frame())
    else:
        with colB:
            st.info("Pilih minimal 2 fitur lalu tekan 'Mulai Clustering'.")