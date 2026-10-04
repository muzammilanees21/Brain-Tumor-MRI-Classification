"""
app/streamlit_app.py  -  Streamlit frontend for the Brain Tumor MRI classifier.

Start the backend first, then this app:
    uvicorn api.main:app --port 8000
    streamlit run app/streamlit_app.py

"""

import base64
import io
import os
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
from fpdf import FPDF
from PIL import Image

API_URL = os.environ.get("API_URL", "http://localhost:8000")
FILE_TYPES = ["jpg", "jpeg", "png", "dcm", "dicom"]

LABELS = {
    "glioma": "Glioma tumor",
    "meningioma": "Meningioma tumor",
    "pituitary": "Pituitary tumor",
    "notumor": "No tumor",
}

# General educational information only - NOT personalised medical advice.
TUMOR_INFO = {
    "glioma": {
        "what": "A tumor that starts in glial cells, the supportive tissue around neurons in the brain and spinal cord. Gliomas range from slow-growing (low grade) to aggressive (high grade, e.g. glioblastoma).",
        "approach": "Usually evaluated with contrast-enhanced MRI and often a biopsy to determine the grade. Common treatment paths are surgery where safely possible, radiation therapy and chemotherapy. The exact combination depends on grade, location and the patient's health, and is decided by a neuro-oncology team.",
    },
    "meningioma": {
        "what": "A tumor arising from the meninges, the membranes covering the brain and spinal cord. Most meningiomas are slow-growing and benign.",
        "approach": "Small, symptom-free meningiomas are often just monitored with periodic scans. Larger or symptomatic ones may be removed surgically, sometimes followed by radiation. A neurosurgeon decides based on size, location and symptoms.",
    },
    "pituitary": {
        "what": "A tumor in the pituitary gland, the small hormone-producing gland at the base of the brain. Most are benign adenomas but can disturb hormone levels.",
        "approach": "Evaluation typically includes hormone blood tests plus imaging. Options range from medication (for hormone-secreting tumors) to surgery (often through the nose) or radiation, chosen by an endocrinologist and neurosurgeon together.",
    },
    "notumor": {
        "what": "The model did not find a tumor pattern in this scan.",
        "approach": "A clear result is reassuring, but ongoing symptoms should still be discussed with a doctor. Imaging is only one part of a clinical evaluation.",
    },
}

DISCLAIMER_TEXT = (
    "Educational/research use only. Not a medical device and not a substitute for "
    "diagnosis by a qualified radiologist or doctor."
)

st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="wide")
st.title("🧠 Brain Tumor MRI Classifier")
st.caption("EfficientNetB0 deep-learning model · glioma · meningioma · pituitary · no tumor")
st.warning(
    "Educational/research project. This is **not** a medical device and must never be used "
    "for real medical decisions. Always consult a qualified radiologist or doctor."
)

with st.sidebar:
    st.header("Backend status")

    @st.cache_data(ttl=15, show_spinner=False)
    def check_health():
        return requests.get(f"{API_URL}/health", timeout=3).json()

    try:
        health = check_health()
        if health.get("weights_found"):
            st.success("API connected · trained weights found")
        else:
            st.error("API running but trained weights are missing. Copy brain_tumor_effnet.weights.h5 into models/.")
    except Exception:
        st.error("Backend not reachable. Start it with:\n\n`uvicorn api.main:app --port 8000`")
    st.divider()
    show_cam = st.checkbox("Show Grad-CAM heatmap", value=True)
    st.caption("Highlights the region of the scan that influenced the prediction most.")
    st.divider()
    st.markdown(
        "**Tips**\n"
        "- Single axial/coronal/sagittal brain MRI slice works best (JPG/PNG/DICOM)\n"
        "- Photos, screenshots with text, or non-MRI images give meaningless results"
    )


# --------------------------------------------------------------- helpers ---
def generate_pdf_report(original_img: Image.Image, gradcam_img, result: dict, filename: str) -> bytes:
    import tempfile

    pred = result["predicted_class"]
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Brain Tumor MRI Classification Report", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(0, 6, f"File: {filename}", ln=True)
    pdf.ln(3)

    with tempfile.TemporaryDirectory() as td:
        orig_path = os.path.join(td, "orig.png")
        original_img.convert("RGB").save(orig_path)
        img_w = 75
        pdf.image(orig_path, x=15, w=img_w)
        cam_y = pdf.get_y() - img_w if gradcam_img is not None else None
        if gradcam_img is not None:
            cam_path = os.path.join(td, "cam.png")
            gradcam_img.convert("RGB").save(cam_path)
            pdf.image(cam_path, x=15 + img_w + 10, y=pdf.get_y() - img_w, w=img_w)
        pdf.ln(4)

        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, f"Prediction: {LABELS.get(pred, pred)}", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"Confidence: {result['confidence']:.1%}", ln=True)
        pdf.cell(0, 6, f"Tumor detected: {'Yes' if result['tumor_detected'] else 'No'}", ln=True)
        if result.get("needs_review"):
            pdf.set_text_color(180, 0, 0)
            pdf.cell(0, 6, "Flagged for human review (low confidence or unusual image)", ln=True)
            pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Class probabilities:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for cls, p in sorted(result["all_probabilities"].items(), key=lambda kv: -kv[1]):
            pdf.cell(0, 5, f"  {LABELS.get(cls, cls)}: {p:.1%}", ln=True)
        pdf.ln(3)

        info = TUMOR_INFO.get(pred, {})
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "General information (education only, not a diagnosis):", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 5, f"What it is: {info.get('what', 'N/A')}")
        pdf.ln(1)
        pdf.multi_cell(0, 5, f"Typical clinical approach: {info.get('approach', 'N/A')}")
        pdf.ln(4)

        pdf.set_font("Helvetica", "I", 8)
        pdf.multi_cell(0, 4, DISCLAIMER_TEXT)

    return bytes(pdf.output())


def call_predict(name: str, data: bytes, mime: str, with_gradcam: bool):
    endpoint = "/predict/gradcam" if with_gradcam else "/predict"
    resp = requests.post(f"{API_URL}{endpoint}", files={"file": (name, data, mime)}, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(resp.json().get("detail", resp.text))
    return resp.json()


tab_single, tab_batch, tab_history = st.tabs(["🔍 Single Scan", "📦 Batch Upload", "🕘 History"])

# ============================================================ SINGLE SCAN ===
with tab_single:
    uploaded = st.file_uploader("Upload a brain MRI image", type=FILE_TYPES, key="single")

    if uploaded is not None:
        data = uploaded.getvalue()
        is_dicom = uploaded.name.lower().endswith((".dcm", ".dicom"))

        left, right = st.columns(2)
        with left:
            st.subheader("Uploaded scan")
            if is_dicom:
                st.info("DICOM file uploaded — preview shown after analysis.")
            else:
                st.image(Image.open(io.BytesIO(data)), use_container_width=True)

        if st.button("Analyze scan", type="primary"):
            with st.spinner("Analyzing..."):
                try:
                    result = call_predict(uploaded.name, data, uploaded.type or "application/octet-stream", show_cam)
                except Exception as e:
                    st.error(f"Could not get a prediction: {e}")
                    st.stop()

            gradcam_pil = None
            if show_cam and "gradcam_overlay_base64" in result:
                gradcam_pil = Image.open(io.BytesIO(base64.b64decode(result["gradcam_overlay_base64"])))
                with right:
                    st.subheader("Grad-CAM explanation")
                    st.image(gradcam_pil, use_container_width=True)
                    st.caption("Warmer colors = regions that pushed the model toward this prediction.")

            st.divider()
            pred = result["predicted_class"]

            if not result["looks_like_mri"]:
                st.warning("This image is strongly colored, so it probably is **not** a grayscale brain MRI. The prediction below is unreliable.")

            if result["tumor_detected"]:
                st.error(f"### 🔴 Tumor pattern detected: {LABELS[pred]}")
            else:
                st.success("### ✅ No tumor pattern detected")

            m1, m2, m3 = st.columns(3)
            m1.metric("Prediction", LABELS[pred])
            m2.metric("Model confidence", f"{result['confidence']:.1%}")
            m3.metric("Overall tumor probability", f"{result['tumor_probability']:.1%}")

            if result["needs_review"]:
                st.info("Confidence is low (or the image looks unusual). In a real clinical workflow this case would be flagged for mandatory review by a radiologist.")

            st.subheader("Class probabilities")
            for cls, p in sorted(result["all_probabilities"].items(), key=lambda kv: -kv[1]):
                st.write(f"**{LABELS.get(cls, cls)}**")
                st.progress(min(max(p, 0.0), 1.0), text=f"{p:.1%}")

            with st.expander("📖 What is this, and how is it typically approached? (general education only)"):
                info = TUMOR_INFO[pred]
                st.markdown(f"**What it is:** {info['what']}")
                st.markdown(f"**Typical clinical approach:** {info['approach']}")
                st.caption(
                    "General educational information about the predicted category, not a diagnosis or "
                    "treatment plan. Grade, stage and treatment can only be determined by a doctor after "
                    "a full evaluation."
                )

            st.caption(result.get("disclaimer", ""))

            try:
                original_for_pdf = Image.open(io.BytesIO(data)).convert("RGB") if not is_dicom else (
                    gradcam_pil if gradcam_pil is not None else None
                )
                if original_for_pdf is not None:
                    pdf_bytes = generate_pdf_report(original_for_pdf, gradcam_pil, result, uploaded.name)
                    st.download_button(
                        "⬇️ Download PDF report",
                        data=pdf_bytes,
                        file_name=f"brain_tumor_report_{uploaded.name.rsplit('.', 1)[0]}.pdf",
                        mime="application/pdf",
                    )
            except Exception as e:
                st.caption(f"(PDF report unavailable: {e})")
    else:
        st.info("Upload an MRI image to get a prediction.")

# ============================================================ BATCH MODE ===
with tab_batch:
    st.write("Upload several scans at once to classify them all in one go.")
    batch_files = st.file_uploader(
        "Upload multiple brain MRI images", type=FILE_TYPES, accept_multiple_files=True, key="batch"
    )

    if batch_files:
        st.caption(f"{len(batch_files)} file(s) selected")
        preview_cols = st.columns(min(len(batch_files), 6))
        for i, f in enumerate(batch_files[:6]):
            with preview_cols[i]:
                if f.name.lower().endswith((".dcm", ".dicom")):
                    st.caption(f"📄 {f.name}")
                else:
                    st.image(Image.open(io.BytesIO(f.getvalue())), caption=f.name, use_container_width=True)
        if len(batch_files) > 6:
            st.caption(f"...and {len(batch_files) - 6} more")

    include_batch_gradcam = st.checkbox("Include Grad-CAM for each scan (slower)", value=True, key="batch_cam")

    if batch_files and st.button("Analyze all", type="primary"):
        with st.spinner(f"Analyzing {len(batch_files)} scans..."):
            try:
                resp = requests.post(
                    f"{API_URL}/predict/batch",
                    files=[("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in batch_files],
                    params={"include_gradcam": include_batch_gradcam},
                    timeout=600,
                )
                resp.raise_for_status()
                results = resp.json()["results"]
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")
                results = []

        if results:
            rows = []
            for r in results:
                if r.get("error"):
                    rows.append({"File": r["filename"], "Prediction": "ERROR", "Confidence": "-", "Tumor?": r["error"], "Needs review": "-"})
                else:
                    rows.append({
                        "File": r["filename"],
                        "Prediction": LABELS.get(r["predicted_class"], r["predicted_class"]),
                        "Confidence": f"{r['confidence']:.1%}",
                        "Tumor?": "Yes" if r["tumor_detected"] else "No",
                        "Needs review": "⚠️ Yes" if r.get("needs_review") else "No",
                    })
            df = pd.DataFrame(rows)
            st.subheader("Summary")
            st.dataframe(df, use_container_width=True)

            n_tumor = sum(1 for r in results if not r.get("error") and r["tumor_detected"])
            n_review = sum(1 for r in results if not r.get("error") and r.get("needs_review"))
            c1, c2, c3 = st.columns(3)
            c1.metric("Scans analyzed", len(results))
            c2.metric("Tumor detected", n_tumor)
            c3.metric("Flagged for review", n_review)

            st.download_button(
                "⬇️ Download results as CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="batch_results.csv",
                mime="text/csv",
            )

            st.subheader("Per-scan details")
            file_by_name = {f.name: f for f in batch_files}
            for r in results:
                if r.get("error"):
                    with st.expander(f"❌ {r['filename']} — {r['error']}"):
                        st.write("Could not process this file.")
                    continue

                verdict = "🔴 Tumor detected" if r["tumor_detected"] else "✅ No tumor"
                with st.expander(f"{verdict} — {r['filename']} ({r['confidence']:.1%} confidence)"):
                    ic1, ic2 = st.columns(2)
                    src_file = file_by_name.get(r["filename"])
                    with ic1:
                        st.caption("Uploaded scan")
                        if src_file is not None and not r["filename"].lower().endswith((".dcm", ".dicom")):
                            st.image(Image.open(io.BytesIO(src_file.getvalue())), use_container_width=True)
                        else:
                            st.caption("(DICOM file — preview not shown)")
                    with ic2:
                        if "gradcam_overlay_base64" in r:
                            st.caption("Grad-CAM")
                            st.image(Image.open(io.BytesIO(base64.b64decode(r["gradcam_overlay_base64"]))), use_container_width=True)
                        else:
                            st.caption("Grad-CAM not requested for this batch")

                    st.write(f"**Prediction:** {LABELS.get(r['predicted_class'], r['predicted_class'])}")
                    for cls, p in sorted(r["all_probabilities"].items(), key=lambda kv: -kv[1]):
                        st.write(f"{LABELS.get(cls, cls)}: {p:.1%}")

        st.caption(DISCLAIMER_TEXT)

# ================================================================ HISTORY ===
with tab_history:
    st.write("Past predictions, most recent first (stored locally in `data/history.db`).")
    col_a, col_b = st.columns([1, 5])
    with col_a:
        refresh = st.button("🔄 Refresh")
    with col_b:
        if st.button("🗑️ Clear history"):
            try:
                requests.delete(f"{API_URL}/history", timeout=10)
                st.success("History cleared.")
            except Exception as e:
                st.error(f"Could not clear history: {e}")

    try:
        hist = requests.get(f"{API_URL}/history", params={"limit": 100}, timeout=10).json()["history"]
    except Exception as e:
        hist = []
        st.error(f"Could not load history: {e}")

    if not hist:
        st.info("No predictions logged yet — analyze a scan in the other tabs first.")
    else:
        for rec in hist:
            thumbnail = rec.get("thumbnail_base64") or rec.get("thumbnail_b64")
            c1, c2 = st.columns([1, 5])
            with c1:
                if thumbnail:
                    try:
                        preview = Image.open(io.BytesIO(base64.b64decode(thumbnail)))
                        st.image(preview, width=80)
                    except Exception:
                        st.caption("Preview unavailable")
                else:
                    st.caption("Preview unavailable")
            with c2:
                verdict = "🔴 Tumor" if rec["tumor_detected"] else "✅ No tumor"
                st.write(
                    f"**{LABELS.get(rec['predicted_class'], rec['predicted_class'])}** — {verdict}  \n"
                    f"Confidence: {rec['confidence']:.1%} · File: {rec['filename']} · "
                    f"{rec['timestamp'][:19].replace('T', ' ')}"
                )
            st.divider()