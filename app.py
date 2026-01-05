import streamlit as st
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import io
from collections import Counter

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================
st.set_page_config(
    page_title="Prototipe Segmentasi Anak Putus Sekolah",
    layout="wide"
)

st.title("📊 Prototipe Segmentasi Anak Putus Sekolah")
st.divider()

# =====================================================
# SESSION STATE
# =====================================================
if "locked" not in st.session_state:
    st.session_state.locked = False
if "df" not in st.session_state:
    st.session_state.df = None
if "labels" not in st.session_state:
    st.session_state.labels = None

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("⚙️ Pengaturan")
uploaded_file = st.sidebar.file_uploader("Upload Dataset CSV", type=["csv"])

K = st.sidebar.slider(
    "Jumlah Cluster (K)",
    min_value=2,
    max_value=8,
    value=4,
    disabled=True  # DIKUNCI SESUAI HASIL PENELITIAN
)

# =====================================================
# CENTROID AKHIR (HASIL ITERASI KONVERGEN - DIKUNCI)
# =====================================================
FINAL_CENTROIDS = [
    [0.8, 0.3, 0.2, 0.4, 0.0],  # Cluster 1
    [0.5, 0.4, 1.0, 0.3, 1.0],  # Cluster 2
    [0.9, 0.2, 0.1, 0.3, 1.0],  # Cluster 3
    [0.0, 0.0, 0.1, 0.3, 0.6]   # Cluster 4
]

FITUR = [
    "Pendidikan",
    "Pekerjaan",
    "Penghasilan",
    "Anggota_Keluarga",
    "Tempat_Tinggal"
]

# =====================================================
# FUNGSI
# =====================================================
def euclidean(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))

def assign_clusters(data, centroids):
    labels = []
    for point in data:
        distances = [euclidean(point, c) for c in centroids]
        labels.append(distances.index(min(distances)))
    return labels

# =====================================================
# LOAD DATA & ASSIGN CLUSTER
# =====================================================
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Validasi kolom
    if list(df.columns[:5]) != FITUR:
        st.error("❌ Struktur kolom dataset tidak sesuai dengan dataset penelitian.")
        st.stop()

    dataset = df[FITUR].values.tolist()

    if not st.session_state.locked:
        if st.button("🚀 Proses Klasterisasi"):
            labels = assign_clusters(dataset, FINAL_CENTROIDS)

            st.session_state.df = df
            st.session_state.labels = labels
            st.session_state.locked = True

# =====================================================
# TAMPILKAN HASIL
# =====================================================
if st.session_state.locked:
    df = st.session_state.df.copy()
    labels = st.session_state.labels

    df["Cluster"] = [l + 1 for l in labels]

    # =================================================
    # DISTRIBUSI CLUSTER (HARUS SAMA DENGAN TESIS)
    # =================================================
    st.subheader("📊 Distribusi Anggota Klaster")
    distribusi = Counter(df["Cluster"])
    st.dataframe(
        pd.DataFrame.from_dict(
            distribusi, orient="index", columns=["Jumlah Anggota"]
        ).sort_index()
    )

    # =================================================
    # PILIH CLUSTER
    # =================================================
    cluster_idx = st.selectbox(
        "Pilih Cluster:",
        options=[1, 2, 3, 4]
    )

    df_cluster = df[df["Cluster"] == cluster_idx]

    st.subheader(f"📌 Ringkasan Cluster {cluster_idx}")
    st.write(f"Jumlah Data : **{len(df_cluster)}**")

    # =================================================
    # TABEL ANGGOTA CLUSTER (FULL)
    # =================================================
    st.subheader("📋 Anggota Cluster (Lengkap)")

    tinggi_tabel = min(900, 35 * (len(df_cluster) + 1))

    st.dataframe(
        df_cluster.reset_index(drop=True),
        use_container_width=True,
        height=tinggi_tabel,
        page_size=len(df_cluster)
    )

    # =================================================
    # DOWNLOAD CSV
    # =================================================
    csv_cluster = df_cluster.reset_index(drop=True).to_csv(index=False)
    st.download_button(
        label=f"⬇️ Download CSV Cluster {cluster_idx}",
        data=csv_cluster,
        file_name=f"anggota_cluster_{cluster_idx}.csv",
        mime="text/csv"
    )

    # =================================================
    # DOWNLOAD EXCEL
    # =================================================
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_cluster.reset_index(drop=True).to_excel(
            writer,
            index=False,
            sheet_name=f"Cluster_{cluster_idx}"
        )

    st.download_button(
        label=f"⬇️ Download Excel Cluster {cluster_idx}",
        data=output.getvalue(),
        file_name=f"anggota_cluster_{cluster_idx}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # =================================================
    # PCA (OPSIONAL - VISUALISASI)
    # =================================================
    st.subheader("📈 Visualisasi PCA (2D)")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df[FITUR].values)

    df["PCA1"] = X_pca[:, 0]
    df["PCA2"] = X_pca[:, 1]

    fig, ax = plt.subplots()
    for c in sorted(df["Cluster"].unique()):
        subset = df[df["Cluster"] == c]
        ax.scatter(subset["PCA1"], subset["PCA2"], label=f"Cluster {c}")

    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.legend()
    st.pyplot(fig)
