import streamlit as st
import pandas as pd
import altair as alt
import os
from PIL import Image
from sklearn.metrics import classification_report, f1_score

st.set_page_config(layout="wide")
st.title("Evaluasi Prediksi")

# =========================
# CONFIG
# =========================
DEFAULT_GT_PATH = "ground_truth.csv"
IMAGE_FOLDER = "test"
VALID_EXT = [".jpg", ".jpeg", ".png"]

# Kode label sesuai Petunjuk Teknis BDC Satria Data 2026
LABEL_MAP = {0: "Recyclable", 1: "Electronic", 2: "Organic"}


def normalize_filename(x):
    x = x.lower()
    if any(x.endswith(ext) for ext in VALID_EXT):
        return x
    return x + ".jpg"  # fallback


def normalize_columns(df):
    """Terima format resmi BDC (id, predicted) maupun (image, label)."""
    df = df.rename(columns={"id": "image", "predicted": "label"})
    return df


def map_label(x):
    try:
        return LABEL_MAP.get(int(x), str(x))
    except (ValueError, TypeError):
        return str(x)


# =========================
# GROUND TRUTH
# =========================
gt_upload = st.file_uploader("Upload Ground Truth (opsional)", type=["csv"])

if gt_upload:
    gt_df = pd.read_csv(gt_upload)
    st.info("Menggunakan Ground Truth upload")
elif os.path.exists(DEFAULT_GT_PATH):
    gt_df = pd.read_csv(DEFAULT_GT_PATH)
    st.info("Menggunakan Ground Truth default")
else:
    st.warning("Ground Truth belum ada. Silakan upload Ground Truth untuk melanjutkan.")
    st.stop()

# Normalisasi GT
gt_df = normalize_columns(gt_df)

if not {"image", "label"}.issubset(gt_df.columns):
    st.error("GT harus punya kolom: id/image dan label/predicted")
    st.stop()

gt_df["image"] = gt_df["image"].astype(str)
gt_df["image"] = gt_df["image"].apply(normalize_filename)
gt_df["label"] = gt_df["label"].apply(map_label)

# =========================
# PREDIKSI USER
# =========================
pred_file = st.file_uploader(
    "Upload Prediksi (CSV, format submission: id,predicted)", type=["csv"]
)


def load_image(img_name):
    base = os.path.splitext(img_name)[0]

    for ext in VALID_EXT:
        path = os.path.join(IMAGE_FOLDER, base + ext)
        if os.path.exists(path):
            return Image.open(path)

    return None


if pred_file:
    pred_df = pd.read_csv(pred_file)

    # Normalisasi kolom
    pred_df = normalize_columns(pred_df)

    if not {"image", "label"}.issubset(pred_df.columns):
        st.error("Prediksi harus punya kolom: id/image dan label/predicted")
        st.stop()

    pred_df["image"] = pred_df["image"].astype(str)
    pred_df["image"] = pred_df["image"].apply(normalize_filename)
    pred_df["label"] = pred_df["label"].apply(map_label)

    # Ketentuan panitia: urutan submission harus sesuai submission.csv (test 1-1458),
    # urutan tidak boleh diubah peserta.
    if len(pred_df) != len(pred_df["image"].unique()):
        st.warning("Terdapat id duplikat pada file prediksi.")

    # =========================
    # MERGE
    # =========================
    merged = gt_df.merge(pred_df, on="image", suffixes=("_gt", "_pred"))

    if merged.empty:
        st.error("Tidak ada data yang match")
        st.stop()

    if len(merged) != len(pred_df):
        st.warning(
            f"Hanya {len(merged)} dari {len(pred_df)} baris prediksi yang cocok "
            f"dengan Ground Truth (id/image tidak ditemukan di GT)."
        )

    # =========================
    # COMPARE
    # =========================
    merged["hasil"] = merged.apply(
        lambda row: "SAMA" if row["label_gt"] == row["label_pred"] else "BEDA",
        axis=1
    )

    merged = merged.sort_values(by="hasil", ascending=True)

    # =========================
    # BASIC METRICS
    # =========================
    total = len(merged)
    benar = (merged["hasil"] == "SAMA").sum()
    akurasi = benar / total * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total", total)
    col2.metric("Sama", benar)
    col3.metric("Tingkat Kesamaan (%)", f"{akurasi:.2f}")

    # =========================
    # CONFUSION MATRIX
    # =========================
    st.subheader("Confusion Matrix")

    cm = pd.crosstab(
        merged["label_gt"],
        merged["label_pred"],
        rownames=["Actual"],
        colnames=["Predicted"],
        dropna=False
    )

    cm_long = cm.reset_index().melt(id_vars="Actual", var_name="Predicted", value_name="Jumlah")
    max_count = cm_long["Jumlah"].max()

    heatmap = alt.Chart(cm_long).mark_rect().encode(
        x=alt.X("Predicted:N", sort=list(cm.columns), title="Predicted"),
        y=alt.Y("Actual:N", sort=list(cm.index), title="Actual"),
        color=alt.Color(
            "Jumlah:Q",
            scale=alt.Scale(scheme="blues"),
            legend=alt.Legend(title="Jumlah"),
        ),
        tooltip=["Actual", "Predicted", "Jumlah"],
    )

    labels = alt.Chart(cm_long).mark_text(baseline="middle").encode(
        x=alt.X("Predicted:N", sort=list(cm.columns)),
        y=alt.Y("Actual:N", sort=list(cm.index)),
        text="Jumlah:Q",
        color=alt.condition(
            alt.datum.Jumlah > max_count / 2, alt.value("white"), alt.value("black")
        ),
    )

    st.altair_chart(heatmap + labels, width="stretch")

    with st.expander("Tabel Confusion Matrix"):
        st.dataframe(cm, width="stretch")

    # =========================
    # MACRO F1 + REPORT
    # =========================
    st.subheader("Evaluasi (Macro F1)")

    y_true = merged["label_gt"]
    y_pred = merged["label_pred"]

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    st.metric("Macro F1 Score", f"{macro_f1 * 100:.7f}")

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()

    # Urutkan dari terburuk
    report_df = report_df.sort_values(by="f1-score")

    st.subheader("Detail per Class (Precision, Recall, F1)")
    st.dataframe(report_df, width="stretch")

    # =========================
    # TABEL PERBANDINGAN
    # =========================
    st.subheader("Tabel Perbandingan")

    def highlight(row):
        if row["hasil"] == "BEDA":
            return ["background-color: #ff4d4d; color: white"] * len(row)
        return [""] * len(row)

    st.dataframe(
        merged.style.apply(highlight, axis=1),
        width="stretch"
    )

    # =========================
    # DETAIL GAMBAR (SALAH SAJA)
    # =========================
    st.subheader("Detail Gambar (Salah)")

    salah_df = merged[merged["hasil"] == "BEDA"]

    if salah_df.empty:
        st.success("Tidak ada prediksi yang salah.")
    else:
        n_cols = 5
        rows = salah_df.to_dict("records")

        for i in range(0, len(rows), n_cols):
            cols = st.columns(n_cols)
            for col, row in zip(cols, rows[i:i + n_cols]):
                with col:
                    img = load_image(row["image"])
                    if img:
                        st.image(img, width="stretch")
                    else:
                        st.write("Image tidak ditemukan")
                    st.caption(
                        f"{row['image']}\nGT: {row['label_gt']}\nPred: {row['label_pred']}"
                    )

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

