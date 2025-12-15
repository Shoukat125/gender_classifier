import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# ===============================
# APPLY CSS (SIDEBAR UPLOADER ONLY)
# ===============================
def apply_css():

    MAIN_BG = "#BFDFFD"
    SIDEBAR_BG = "#0f172a"
    TEXT_LIGHT = "#e5e7eb"

    st.markdown(f"""
    <style>

    /* FULL APP BACKGROUND */
    html, body, [data-testid="stApp"] {{
        background-color: {MAIN_BG};
    }}

    /* MAIN CONTENT */
    main .block-container {{
        padding: 2rem;
    }}

    /* SIDEBAR BACKGROUND */
    section[data-testid="stSidebar"] {{
        background-color: {SIDEBAR_BG};
    }}

    /* SIDEBAR TEXT */
    section[data-testid="stSidebar"] * {{
        color: {TEXT_LIGHT};
    }}

    /* ===== SIDEBAR FILE UPLOADER ONLY ===== */
    section[data-testid="stSidebar"] .stFileUploader {{
        background: #f8fafc !important;
        border: 2px dashed #94a3b8;
        border-radius: 12px;
        padding: 14px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }}

    section[data-testid="stSidebar"] .stFileUploader * {{
        color: #0f172a !important;
    }}

    section[data-testid="stSidebar"]
    .stFileUploader div[data-testid="stFileUploaderDropzone"] {{
        background: #ffffff !important;
        border-radius: 10px;
    }}

    /* BUTTON */
    .stButton > button {{
        background: linear-gradient(90deg,#ff6f3c,#ff3d00);
        color: white;
        border-radius: 999px;
        padding: 0.45rem 1.2rem;
        font-weight: 600;
        font-size: 13px;
    }}

    /* REMOVE HEADER */
    header[data-testid="stHeader"] {{
        visibility: hidden;
    }}

    </style>
    """, unsafe_allow_html=True)


# ===============================
# DISPLAY PREDICTION UI
# ===============================
def display_prediction(img_array, predicted_label, confidence, probs, class_names):

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(
            img_array,
            caption=f"{predicted_label} ({confidence*100:.2f}%)",
            width=300
        )

    with col2:
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.bar(class_names, probs)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Confidence", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)


# ===============================
# MAIN APP
# ===============================
apply_css()

st.markdown(
    "<h1 style='text-align:center; color:#1e293b;'>Gender Classifier</h1>",
    unsafe_allow_html=True
)

# SIDEBAR
with st.sidebar:
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

# DUMMY OUTPUT (replace with model later)
if uploaded_file:
    img = plt.imread(uploaded_file)

    class_names = ["Male", "Female"]
    probs = np.array([0.65, 0.35])
    predicted_label = class_names[np.argmax(probs)]
    confidence = np.max(probs)

    display_prediction(img, predicted_label, confidence, probs, class_names)
