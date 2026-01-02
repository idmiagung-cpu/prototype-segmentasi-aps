import streamlit as st
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import io

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
# KONFIGURASI HALAMAN
# =====================================================
st.set_page_config(
    page_title="Prototipe Segmentasi Anak Putus Sekolah",
    layout="wide"
)

st.title("📊 Prototipe Segmentasi Anak Putus Sekolah")
st.divider()

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
    disabled=st.session_state.locked
)

MAX_ITER = 100

# =====================================================
# FUNGSI K-MEANS
# =====================================================
def euclidean(a, b):
    return math.sqrt(
        sum((float(a[i]) - float(b[i])) ** 2 for i in range(len(a)))
    )

def init_centroids(data, k):
    return [data[i][:] for i in random.sample(range(len(data)), k)]

def assign_clusters(data, centroids):
    labels = []
    for point in data:
        distances = [euclidean(point, c) for c in centroids]
        labels.append(distances.index(min(distances)))
    return labels

def compute_centroids(data, labels, k, dim):
    centroids = []
    for i in range(k):
        cluster_points = [data[j] for j in range(len(data)) if labels[j] == i]
        if not cluster_points:
            centroids.append([0] * dim)
        else:
            centroids.append(
                [sum(p[d] for p in cluster_points) / len(cluster_points) for d in range(dim)]
            )
    return centroids

# =====================================================
# LOAD DATA & PROSES K-MEANS
# =====================================================
if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)

    # 🔒 AMAN: hanya kolom numerik (inti perbaikan error)
    df_numeric = df_raw.select_dtypes(include=["int64", "float64"])

    if df_numeric.empty:
        st.error("❌ Dataset tidak memiliki kolom numerik.")
    else:
        dataset = df_numeric.values.tolist()
        df = df_numeric.copy()

        if not st.session_state.locked:
            if st.button("🚀 Proses K-Means"):
                random.seed(42)
                centroids = init_centroids(dataset, K)

                for _ in range(MAX_ITER):
                    labels = assign_clusters(dataset, centroids)
                    new_centroids = compute_centroids(
                        dataset, labels, K, len(dataset[0])
                    )
                    if centroids == new_centroids:
                        break
                    centroids = new_centroids

                st.session_state.df = df
                st.session_state.labels = labels
                st.session_state.locked = True

# =====================================================
# TAMPILKAN HASIL
# =====================================================
if st.session_state.locked:
    df = st.session_state.df
    labels = st.session_state.labels

    hasil = df.copy()
    hasil["Cluster"] = [l + 1 for l in labels]

    cluster_idx = st.selectbox(
        "Pilih Cluster:",
        options=list(range(1, K + 1))
    )

    df_cluster = hasil[hasil["Cluster"] == cluster_idx]

    st.subheader(f"📌 Ringkasan Cluster {cluster_idx}")
    st.write(f"Jumlah Data : **{len(df_cluster)}**")

    # =================================================
    # 📋 ANGGOTA CLUSTER (LENGKAP)
    # =================================================
    st.subheader("📋 Anggota Cluster (Lengkap)")

    tinggi_tabel = min(900, 35 * (len(df_cluster) + 1))

    st.dataframe(
        df_cluster.reset_index(drop=True),
        use_container_width=True,
        height=tinggi_tabel,
        page_size=len(df_cluster)
    )

    st.caption(f"Total anggota Cluster {cluster_idx} : {len(df_cluster)} data")

    # =================================================
    # ⬇️ DOWNLOAD CSV
    # =================================================
    csv_cluster = df_cluster.reset_index(drop=True).to_csv(index=False)

    st.download_button(
        label=f"⬇️ Download CSV Cluster {cluster_idx}",
        data=csv_cluster,
        file_name=f"anggota_cluster_{cluster_idx}.csv",
        mime="text/csv"
    )

    # =================================================
    # ⬇️ DOWNLOAD EXCEL
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
