import io
import requests
import streamlit as st
from PIL import Image

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Weapon Detection Console", page_icon="🛡️", layout="wide")

# ------------------------------------------------------------------ STYLE
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root{
  --bg:#F3F5F8; --surface:#FFFFFF; --ink:#14213D; --muted:#5B6577;
  --line:#DDE2EA; --accent:#1F4E8C; --alert:#C8352B; --ok:#2E7D5B;
}
html, body, [class*="css"], .stApp { font-family:'IBM Plex Sans', 'Segoe UI', sans-serif; color:var(--ink); }
.stApp { background:var(--bg); }
#MainMenu, footer, header[data-testid="stHeader"] { visibility:hidden; height:0; }
.block-container { padding-top:1.6rem; max-width:1280px; }

/* top bar */
.topbar{ background:var(--ink); color:#fff; padding:18px 26px; border-radius:10px;
         display:flex; justify-content:space-between; align-items:center; margin-bottom:22px; }
.topbar h1{ font-size:1.35rem; font-weight:600; margin:0; color:#fff; letter-spacing:.2px; }
.topbar p{ margin:2px 0 0; font-size:.85rem; color:#B8C2D6; }
.pill{ font-size:.8rem; padding:6px 12px; border-radius:999px; font-weight:500; }
.pill.on{ background:#1E3A5F; color:#9FE0C0; }
.pill.off{ background:#4A1F1F; color:#FFB4AE; }

/* sidebar */
section[data-testid="stSidebar"]{ background:var(--surface); border-right:1px solid var(--line); }

/* tabs */
.stTabs [data-baseweb="tab-list"]{ gap:6px; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"]{ font-weight:500; color:var(--muted); padding:10px 18px; }
.stTabs [aria-selected="true"]{ color:var(--accent); }
.stTabs [data-baseweb="tab-highlight"]{ background:var(--accent); }

/* panels */
.panel-title{ font-size:.95rem; font-weight:600; margin:0 0 10px; }
[data-testid="stImage"] img, video{ border-radius:8px; border:1px solid var(--line); max-width:100%; }

/* result banner */
.banner{ border-radius:8px; padding:14px 18px; margin:14px 0; border:1px solid; }
.banner b{ font-size:1.05rem; }
.banner span{ display:block; font-size:.85rem; margin-top:2px; }
.banner.alert{ background:#FBECEA; border-color:#F0BDB8; color:#8E2119; }
.banner.clear{ background:#E8F4EE; border-color:#B7DDC9; color:#1F5C41; }

/* metrics */
[data-testid="stMetric"]{ background:var(--surface); border:1px solid var(--line);
                          border-radius:8px; padding:14px 16px; }
[data-testid="stMetricLabel"]{ color:var(--muted); }

/* buttons */
.stButton>button, .stDownloadButton>button{ border-radius:8px; font-weight:500; border:1px solid var(--line); }
.stButton>button[kind="primary"]{ background:var(--accent); border-color:var(--accent); color:#fff; }
.stButton>button[kind="primary"]:hover{ background:#173C6D; border-color:#173C6D; }

[data-testid="stFileUploader"] section{ background:var(--surface); border:1.5px dashed #B9C3D3; border-radius:10px; }
.hint{ color:var(--muted); font-size:.85rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ HELPERS
def backend_online() -> bool:
    try:
        requests.get(f"{API_URL}/", timeout=2)
        return True
    except Exception:
        return False


def banner(count: int):
    if count:
        st.markdown(
            f'<div class="banner alert"><b>{count} object detected</b>'
            f"<span>Result image mein boxes check karo.</span></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="banner clear"><b>Koi object nahi mila</b>'
            "<span>Confidence slider kam karke dobara try kar sakte ho.</span></div>",
            unsafe_allow_html=True,
        )


online = backend_online()

# ------------------------------------------------------------------ TOP BAR
pill = (
    '<span class="pill on">Backend online</span>'
    if online
    else '<span class="pill off">Backend offline</span>'
)
st.markdown(
    f"""
<div class="topbar">
  <div>
    <h1>Weapon Detection Console</h1>
    <p>YOLO11 based image and video analysis</p>
  </div>
  {pill}
</div>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ SIDEBAR
with st.sidebar:
    st.markdown("### Settings")
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05,
                     help="Kam value: zyada detections, lekin galat bhi ho sakti hain.")
    st.markdown(
        '<p class="hint">Model kamzor ho to 0.10 se 0.20 try karo. '
        "Zyada confidence par sirf pakki detections aati hain.</p>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(f'<p class="hint">API: {API_URL}</p>', unsafe_allow_html=True)

if not online:
    st.error("Backend se connect nahi hua. Terminal mein `uvicorn main:app --port 8000` chalao.")
    st.stop()

tab_img, tab_vid, tab_train = st.tabs(["Image", "Video", "Training"])

# ------------------------------------------------------------------ IMAGE
with tab_img:
    up = st.file_uploader("Image upload karo (JPG, PNG, JFIF)",
                          type=["jpg", "jpeg", "png", "jfif"], key="img")
    if up is not None:
        data = up.getvalue()
        files = {"file": (up.name, data, "image/jpeg")}

        left, right = st.columns(2, gap="large")
        with left:
            st.markdown('<p class="panel-title">Original</p>', unsafe_allow_html=True)
            st.image(Image.open(io.BytesIO(data)))

        with right:
            st.markdown('<p class="panel-title">Detection result</p>', unsafe_allow_html=True)
            try:
                with st.spinner("Analyzing..."):
                    img_res = requests.post(f"{API_URL}/predict/image", files=files,
                                            params={"conf": conf}, timeout=120)
                    json_res = requests.post(f"{API_URL}/predict", files=files,
                                             params={"conf": conf}, timeout=120)
                if img_res.status_code == 200 and json_res.status_code == 200:
                    st.image(img_res.content)
                    res = json_res.json()
                else:
                    st.error(f"API error: {img_res.text}")
                    res = None
            except Exception as e:
                st.error(f"Request fail: {e}")
                res = None

        if res is not None:
            banner(res["count"])
            m1, m2, m3 = st.columns(3)
            confs = [d["confidence"] for d in res["detections"]]
            m1.metric("Objects", res["count"])
            m2.metric("Highest confidence", f"{max(confs):.0%}" if confs else "-")
            m3.metric("Threshold", f"{conf:.0%}")

            if res["detections"]:
                st.dataframe(
                    [{"Label": d["label"], "Confidence": d["confidence"], "Box (x1,y1,x2,y2)": d["box"]}
                     for d in res["detections"]],
                    use_container_width=True, hide_index=True,
                )

        st.divider()
        st.markdown('<p class="panel-title">Training data mein add karo</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="hint">Is image ko future training ke liye save karo. '
            "Predicted boxes labels ban jate hain; galat hon to pehle review karna behtar hai.</p>",
            unsafe_allow_html=True,
        )
        use_pred = st.checkbox("Predicted boxes ko labels ki tarah save karo", value=True)
        if st.button("Save for training"):
            r = requests.post(f"{API_URL}/save-training-image",
                              files={"file": (up.name, data, "image/jpeg")},
                              data={"use_predictions": str(use_pred).lower()})
            if r.status_code == 200:
                st.success(f"Saved. Boxes: {r.json()['boxes']}")
            else:
                st.error(r.text)

# ------------------------------------------------------------------ VIDEO
with tab_vid:
    vid = st.file_uploader("Video upload karo (MP4, MOV, AVI, MKV)",
                           type=["mp4", "mov", "avi", "mkv"], key="vid")
    if vid is not None:
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown('<p class="panel-title">Original video</p>', unsafe_allow_html=True)
            st.video(vid.getvalue())

        with right:
            st.markdown('<p class="panel-title">Detection result</p>', unsafe_allow_html=True)
            if st.button("Analyze video", type="primary"):
                with st.spinner("Video process ho rahi hai. Lambi video mein kuch minute lag sakte hain."):
                    try:
                        r = requests.post(f"{API_URL}/predict/video",
                                          files={"file": (vid.name, vid.getvalue(), "video/mp4")},
                                          params={"conf": conf}, timeout=1800)
                        if r.status_code == 200:
                            st.session_state["vid_result"] = {
                                "bytes": r.content,
                                "hit": int(r.headers.get("X-Detected-Frames", 0)),
                                "total": int(r.headers.get("X-Total-Frames", 0)),
                            }
                        else:
                            st.error(r.text)
                    except Exception as e:
                        st.error(f"Request fail: {e}")

            vr = st.session_state.get("vid_result")
            if vr:
                st.video(vr["bytes"])
                st.download_button("Download result video", vr["bytes"], "result.mp4", "video/mp4")

        if st.session_state.get("vid_result"):
            vr = st.session_state["vid_result"]
            pct = (vr["hit"] / vr["total"]) if vr["total"] else 0
            banner(1 if vr["hit"] else 0)
            m1, m2, m3 = st.columns(3)
            m1.metric("Total frames", vr["total"])
            m2.metric("Frames with detection", vr["hit"])
            m3.metric("Detection rate", f"{pct:.0%}")

# ------------------------------------------------------------------ TRAINING
with tab_train:
    try:
        status = requests.get(f"{API_URL}/retrain/status", timeout=5).json()
    except Exception as e:
        status = None
        st.error(f"Status nahi mila: {e}")

    if status:
        c1, c2 = st.columns(2)
        c1.metric("New images saved", status["new_images"])
        c2.metric("Training", "Running" if status["running"] else "Idle")

        st.markdown('<p class="panel-title" style="margin-top:18px">Retrain model</p>',
                    unsafe_allow_html=True)
        st.markdown(
            '<p class="hint">Purane weights se fine-tune hota hai. Naya model tabhi replace hota hai '
            "jab uska mAP purane se behtar ho, aur purana backup mein rehta hai.</p>",
            unsafe_allow_html=True,
        )
        epochs = st.number_input("Epochs", 1, 200, 30)

        b1, b2, _ = st.columns([1, 1, 3])
        if b1.button("Start retraining", type="primary", disabled=status["running"]):
            r = requests.post(f"{API_URL}/retrain", params={"epochs": int(epochs)})
            if r.status_code == 200:
                st.success("Training shuru ho gayi")
            else:
                st.error(r.text)
        b2.button("Refresh status")

        if status["log"]:
            st.markdown('<p class="panel-title" style="margin-top:18px">Training log</p>',
                        unsafe_allow_html=True)
            st.code(status["log"], language="text")