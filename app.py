import streamlit as st
import pandas as pd
import os
from PIL import Image

st.set_page_config(layout="wide")
st.title("Evaluasi Prediksi")

# =========================
# CONFIG
# =========================
DEFAULT_GT_PATH = "ground_truth.csv"
IMAGE_FOLDER = "test"

# =========================
# GROUND TRUTH
# =========================
gt_upload = st.file_uploader("Upload Ground Truth (opsional)", type=["csv"])

if gt_upload:
    gt_df = pd.read_csv(gt_upload)
    st.info("Menggunakan Ground Truth upload")
else:
    if not os.path.exists(DEFAULT_GT_PATH):
        st.error("ground_truth.csv tidak ditemukan")
        st.stop()
    gt_df = pd.read_csv(DEFAULT_GT_PATH)
    st.info("Menggunakan Ground Truth default")

# Normalisasi GT
if "id" in gt_df.columns:
    gt_df = gt_df.rename(columns={"id": "image"})

# Tambahkan .jpg jika belum ada
gt_df["image"] = gt_df["image"].astype(str)
gt_df["image"] = gt_df["image"].apply(
    lambda x: x if x.endswith(".jpg") else x + ".jpg"
)

# Validasi
if not {"image", "label"}.issubset(gt_df.columns):
    st.error("GT harus punya kolom: id/image dan label")
    st.stop()

# =========================
# PREDIKSI USER
# =========================
pred_file = st.file_uploader("Upload Prediksi (CSV)", type=["csv"])


def load_image(img_name):
    path = os.path.join(IMAGE_FOLDER, img_name)
    if os.path.exists(path):
        return Image.open(path)
    return None


if pred_file:
    pred_df = pd.read_csv(pred_file)

    # =========================
    # NORMALISASI KOLOM
    # =========================
    if "id" in pred_df.columns:
        pred_df = pred_df.rename(columns={"id": "image"})

    # Tambahkan .jpg kalau belum ada
    pred_df["image"] = pred_df["image"].astype(str)
    pred_df["image"] = pred_df["image"].apply(
        lambda x: x if x.endswith(".jpg") else x + ".jpg"
    )

    # Validasi
    if not {"image", "label"}.issubset(pred_df.columns):
        st.error("Prediksi harus punya kolom: id/image dan label")
        st.stop()

    # =========================
    # MISMATCH
    # =========================
    missing_in_pred = set(gt_df["image"]) - set(pred_df["image"])
    extra_in_pred = set(pred_df["image"]) - set(gt_df["image"])

    if missing_in_pred:
        st.warning(f"{len(missing_in_pred)} data GT tidak ada di prediksi")

    if extra_in_pred:
        st.warning(f"{len(extra_in_pred)} data prediksi tidak ada di GT")

    # =========================
    # MERGE
    # =========================
    merged = gt_df.merge(pred_df, on="image", suffixes=("_gt", "_pred"))

    if merged.empty:
        st.error("Tidak ada data yang match")
        st.stop()

    # =========================
    # COMPARE
    # =========================
    merged["hasil"] = merged.apply(
        lambda row: "SAMA" if row["label_gt"] == row["label_pred"] else "BEDA",
        axis=1
    )

    # Sort BEDA di atas
    merged = merged.sort_values(by="hasil", ascending=True)

    # =========================
    # METRICS
    # =========================
    total = len(merged)
    benar = (merged["hasil"] == "SAMA").sum()
    akurasi = benar / total * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total", total)
    col2.metric("Sama", benar)
    col3.metric("Tingkat Kesamaan (%)", f"{akurasi:.2f}")

    # =========================
    # TABEL
    # =========================
    st.subheader("Tabel Perbandingan")

    def highlight(row):
        if row["hasil"] == "BEDA":
            return ["background-color: #ff4d4d; color: white"] * len(row)
        return [""] * len(row)

    st.dataframe(
        merged.style.apply(highlight, axis=1),
        use_container_width=True
    )

    # =========================
    # DETAIL
    # =========================
    st.subheader("Detail Gambar")

    for _, row in merged.iterrows():
        cols = st.columns([1, 2])

        with cols[0]:
            img = load_image(row["image"])
            if img:
                st.image(img, width=120)
            else:
                st.write("Image tidak ditemukan")

        with cols[1]:
            st.write(f"Image: {row['image']}")
            st.write(f"GT: {row['label_ground_truth']}")
            st.write(f"Pred: {row['label_prediksi']}")
            st.write(f"Hasil: {row['hasil']}")

        st.markdown("---")

    # =========================
    # DOWNLOAD
    # =========================
    csv = merged.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Hasil",
        data=csv,
        file_name="hasil.csv",
        mime="text/csv"
    )
