import streamlit as st
import pandas as pd
import random
import math
import io

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================
st.set_page_config(
    page_title="Prototipe Segmentasi Anak Putus Sekolah",
    layout="wide"
)

st.title("📊 Prototipe Segmentasi Anak Putus Sekolah")
st.caption("Menampilkan hasil K-Means pada kondisi konvergen (iterasi akhir)")
st.divider()

# =====================================================
# PARAMETER TETAP (SAMA DENGAN KODE ITERASI)
# =====================================================
K = 4
MAX_ITER = 100
SEED = 42
random.seed(SEED)

# =====================================================
# FUNGSI K-MEANS (IDENTIK DENGAN KODE ANDA)
# =====================================================
def euclidean(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))

def init_centroids(data, k):
    return random.sample(data, k)

def assign_clusters(data, centroids):
    clusters = [[] for _ in range(len(centroids))]
    labels = []

    for idx, point in enumerate(data):
        distances = [euclidean(point, c) for c in centroids]
        cluster_idx = distances.index(min(distances))
        clusters[cluster_idx].append((idx, point))
        labels.append(cluster_idx)

    return clusters, labels

def compute_centroids(clusters, dim):
    new_centroids = []
    for cluster in clusters:
        if len(cluster) == 0:
            new_centroids.append([0] * dim)
        else:
            centroid = []
            for i in range(dim):
                centroid.append(
                    sum(point[1][i] for point in cluster) / len(cluster)
                )
            new_centroids.append(centroid)
    return new_centroids

def converged(old, new):
    return old == new

# =====================================================
# UPLOAD DATASET
# =====================================================
uploaded_file = st.file_uploader(
    "📂 Upload Dataset CSV (sesuai dataset penelitian)",
    type=["csv"]
)

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)

    # 👉 Gunakan 4 kolom pertama sebagai fitur (SAMA DENGAN PENELITIAN)
    df_fitur = df_raw.iloc[:, :4].astype(float)
    dataset = df_fitur.values.tolist()

    # =================================================
    # PROSES K-MEANS (SAMPAI ITERASI TERAKHIR)
    # =================================================
    centroids = init_centroids(dataset, K)

    for _ in range(MAX_ITER):
        clusters, labels = assign_clusters(dataset, centroids)
        new_centroids = compute_centroids(clusters, len(dataset[0]))
        if converged(centroids, new_centroids):
            break
        centroids = new_centroids

    # =================================================
    # HASIL AKHIR (KONVERGEN)
    # =================================================
    df_hasil = df_raw.copy()
    df_hasil["Cluster"] = [l + 1 for l in labels]

    # =================================================
    # RINGKASAN DISTRIBUSI CLUSTER
    # =================================================
    st.subheader("📌 Distribusi Anggota Cluster (Kondisi Konvergen)")
    distribusi = df_hasil["Cluster"].value_counts().sort_index()
    st.dataframe(distribusi.rename("Jumlah Anggota"))

    st.divider()

    # =================================================
    # PILIH CLUSTER
    # =================================================
    cluster_pilih = st.selectbox(
        "Pilih Cluster untuk melihat anggotanya:",
        options=sorted(df_hasil["Cluster"].unique())
    )

    df_cluster = df_hasil[df_hasil["Cluster"] == cluster_pilih]

    st.subheader(f"📋 Anggota Cluster {cluster_pilih} (Iterasi Akhir)")
    st.write(f"Jumlah anggota: **{len(df_cluster)}**")

    st.dataframe(
        df_cluster.reset_index(drop=True),
        use_container_width=True
    )

    # =================================================
    # DOWNLOAD CSV
    # =================================================
    csv_data = df_cluster.to_csv(index=False)
    st.download_button(
        label="⬇️ Download CSV Cluster",
        data=csv_data,
        file_name=f"cluster_{cluster_pilih}_iterasi_akhir.csv",
        mime="text/csv"
    )
