"""
HOME page – Project overview, team members, CNN architecture,
technologies used, and folder structure.
"""
import streamlit as st


# ── Helpers ───────────────────────────────────────────────────

def _badge(text: str, kind: str = "green"):
    return f'<span class="badge badge-{kind}">{text}</span>'


def _arch_layer(layer_type: str, info: str):
    return f"""
    <div class="arch-layer">
        <span class="layer-type">{layer_type}</span>
        <span class="layer-info">{info}</span>
    </div>
    """


# ── Main render ───────────────────────────────────────────────

def render():

    # ── Hero banner ──────────────────────────────────────────
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #064e3b 0%, #0a1f14 40%, #0a0f0d 100%);
            border: 1px solid rgba(16,185,129,0.2);
            border-radius: 20px;
            padding: 3rem 2.5rem;
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
        ">
            <div style="position:absolute;top:-30px;right:-30px;font-size:10rem;
                        opacity:0.05;user-select:none;">🌿</div>
            <div style="font-size:0.8rem; color:#34d399; text-transform:uppercase;
                        letter-spacing:0.12em; margin-bottom:0.6rem;">
                🎓 Machine Learning Project· Semester: 2025.2
            </div>
            <h1 style="font-size:2.4rem; margin:0 0 0.6rem; line-height:1.1;">
                Plant Disease Detection
            </h1>
            <p style="color:#9ca3af; font-size:1rem; max-width:680px; line-height:1.7; margin:0 0 1.4rem;">
                System for classifying plant diseases using <strong style="color:#34d399;">
                Convolutional Neural Network</strong> built from scratch with
                <strong style="color:#34d399;">CuPy</strong> (GPU‑accelerated NumPy),
                without using any autograd framework. Data from the PlantVillage dataset.
            </p>
            <div style="display:flex; flex-wrap:wrap; gap:0.5rem;">
                <span class="badge badge-green">CuPy GPU</span>
                <span class="badge badge-amber">Python 3.10</span>
                <span class="badge badge-blue">PlantVillage</span>
                <span class="badge badge-purple">10 Classes</span>
                <span class="badge badge-green">From Scratch</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Quick stats ──────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    stats = [
        (col1, "🏷️", "10",    "Number of Disease Classes"),
        (col2, "🖼️", "64×64", "Image Size"),
        (col3, "⚙️", "~2.1M", "Model Parameters"),
        (col4, "🚀", "CUDA",  "Training Device"),
    ]
    for col, icon, val, label in stats:
        with col:
            st.markdown(
                f"""
                <div class="metric-tile">
                    <div style="font-size:1.5rem;">{icon}</div>
                    <div class="value">{val}</div>
                    <div class="label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Project overview ─────────────────────────────────────
    with st.expander("📋 Project Overview", expanded=True):
        st.markdown(
            """
            This project implements a deep learning-based model for plant disease detection using the PlantVillage dataset. 
            The goal is to classify different diseases on plant leaves from images, leveraging a Convolutional Neural Network 
            (CNN) built **from scratch** with CuPy (GPU‑accelerated NumPy) and no high‑level autograd.

            **Main Objectives:**
            - Gain a deep understanding of the CNN architecture by implementing each layer from scratch
            - Utilize CuPy for accelerated computation on the GPU (NVIDIA CUDA)
            - Classify plant leaf images into 10 disease categories or healthy status
            - Compare the performance of the CNN with traditional models: XGBoost, Random Forest, SVM

            **Techniques Used:**
            - `im2col` / `col2im` to vectorize the convolution operation
            - Adam, SGD, SGD+Momentum optimizers
            - Cross-Entropy Loss & Focal Loss
            - Mini-batch gradient descent
            """
        )

    # ── Team members ─────────────────────────────────────────
    st.markdown("### 👥 Team Members")

    members = [
        {
            "emoji": "👨‍💻",
            "name":  "Nguyễn Gia Vinh",
            "id":    "Student ID: 202416766",
            "role":  "Team Lead · CNN Dev",
            "tasks": [
                "Xây dựng kiến trúc CNN từ đầu (Conv2D, MaxPool2D, Linear, ReLU)",
                "Implement backpropagation với im2col / col2im",
                "Tích hợp CuPy GPU acceleration",
            ],
        },
        {
            "emoji": "👩‍💻",
            "name":  "Chu Văn An",
            "id":    "Student ID: 20235582",
            "role":  "Data Engineer",
            "tasks": [
                "UI/UX implementation with Streamlit",
            ],
        },
        {
            "emoji": "🧑‍🔬",
            "name":  "Đặng Anh Tú",
            "id":    "Student ID: 202400118",
            "role":  "ML Engineer",
            "tasks": [
                "Implement optimizer: Adam, SGD, SGD Momentum",
                "Cross-Entropy Loss và Focal Loss",
                "Training loop và hyperparameter tuning",
            ],
        },
        {
            "emoji": "👩‍🎨",
            "name":  "Nguyễn Thế Phương",
            "id":    "Student ID: 202416735",
            "role":  "Evaluation · UI",
            "tasks": [
                "Đánh giá mô hình: confusion matrix, accuracy, F1",
                "Thiết kế giao diện Streamlit",
                "Báo cáo và tài liệu dự án",
            ],
        },
        {
            "emoji": "👨‍💻",
            "name":  "Hà Tùng Anh",
            "id":    "Student ID: 202416771",
            "role":  "Team Lead · CNN Dev",
            "tasks": [
                "Xây dựng kiến trúc CNN từ đầu (Conv2D, MaxPool2D, Linear, ReLU)",
                "Implement backpropagation với im2col / col2im",
                "Tích hợp CuPy GPU acceleration",
            ],
        },
    ]

    cols = st.columns(len(members))
    for col, m in zip(cols, members):
        with col:
            tasks_html = "".join(f"<li>{t}</li>" for t in m["tasks"])
            st.markdown(
                f"""
                <div class="member-card">
                    <div class="member-avatar">{m['emoji']}</div>
                    <div class="member-name">{m['name']}</div>
                    <div style="font-size:0.72rem; color:#6b7280; margin-bottom:0.3rem;">{m['id']}</div>
                    <div class="member-role">{m['role']}</div>
                    <ul style="text-align:left; font-size:0.76rem; color:#9ca3af;
                               padding-left:1.2rem; line-height:1.6; margin:0;">
                        {tasks_html}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── CNN Architecture ─────────────────────────────────────
    st.markdown("### 🧠 CNN Architecture")
    st.markdown(
        """
        <div class="plant-card">
            <div style="font-size:0.8rem; color:#6b7280; margin-bottom:1rem; font-family:'JetBrains Mono',monospace;">
                Input: (batch, 3, 64, 64)
                <br>
                Output: (batch, num_classes)  →  argmax  →  predicted label
            </div>
        """
        + _arch_layer("Conv2D",   "in=3 → out=16 | kernel=3×3 | stride=1 | pad=1")
        + _arch_layer("ReLU",     "Non-linear activation")
        + _arch_layer("MaxPool2D","kernel=2×2 | stride=2  →  (16, 32, 32)")
        + "<div style='margin:0.6rem 0; border-left:3px dashed #064e3b; padding-left:1rem; margin-left:0.5rem;'></div>"
        + _arch_layer("Conv2D",   "in=16 → out=32 | kernel=3×3 | stride=1 | pad=1")
        + _arch_layer("ReLU",     "Non-linear activation")
        + _arch_layer("MaxPool2D","kernel=2×2 | stride=2  →  (32, 16, 16)")
        + "<div style='margin:0.6rem 0; border-left:3px dashed #064e3b; padding-left:1rem; margin-left:0.5rem;'></div>"
        + _arch_layer("Flatten",  "32 × 16 × 16 = 8192 features")
        + _arch_layer("Linear",   "8192 → 128")
        + _arch_layer("ReLU",     "Non-linear activation")
        + _arch_layer("Linear",   "128 → num_classes (output logits)")
        + 
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Technologies ─────────────────────────────────────────
    st.markdown("### 🛠️ Technologies Used")
    tech_cols = st.columns(3)
    techs = [
        ("⚡ CuPy",         "GPU-accelerated NumPy. Efficient tensor computations on CUDA.",               "green"),
        ("🐍 Python 3.10",  "Primary programming language.",                                                "amber"),
        ("🖼️ Pillow (PIL)", "Image reading and preprocessing: resize, convert, normalize.",                       "blue"),
        ("📊 Matplotlib",   "Plotting graphs: loss curve, confusion matrix, accuracy.",                      "purple"),
        ("🌿 Streamlit",    "Web interface framework for demo and evaluation.",                   "green"),
        ("🌲 Scikit-learn", "Baseline models: SVM, Random Forest; metric utilities.",                    "amber"),
        ("🚀 XGBoost",      "Gradient boosting model using CNN features as input.",                   "blue"),
        ("📦 NumPy",        "CPU fallback and data handling.",                                           "purple"),
        ("🔢 KaggleHub",    "Automatically download the PlantVillage dataset from Kaggle.",              "green"),
    ]
    for i, (name, desc, kind) in enumerate(techs):
        with tech_cols[i % 3]:
            st.markdown(
                f"""
                <div class="plant-card" style="margin-bottom:0.8rem; padding:1rem 1.2rem;">
                    <div style="font-weight:600; color:#f0fdf4; margin-bottom:0.3rem;">
                        {name} &nbsp; <span class="badge badge-{kind}" style="font-size:0.65rem;">Tech</span>
                    </div>
                    <div style="font-size:0.8rem; color:#9ca3af; line-height:1.5;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Folder structure ─────────────────────────────────────
    st.markdown("### 📁 Project Structure")
    st.code(
        """\
ML/
├── train_cnn.py              # Training script for CNN (from scratch)
├── train_ml.py               # Training script for XGBoost, SVM, Random Forest
├── evaluate.py               # Metrics: accuracy, confusion matrix
├── prepare_data.py           # Initialize dataset
├── best_cnn_weights.npz      # Saved best CNN weights (loadable with np.load)
│
├── model/
│   ├── cnn.py                # CNN model (from scratch)
│   ├── xgboost.py            # XGBoost wrapper
│   ├── svm.py                # SVM wrapper
│   └── random_forest.py      # Random Forest wrapper
│
├── layers/
│   ├── conv2d.py             # Conv2D + im2col/col2im
│   ├── maxpool2d.py          # MaxPool2D
│   ├── linear.py             # Fully-connected layer
│   └── activation.py        # ReLU, Softmax
│
├── utils/
│   ├── data_loader.py        # LeafDataset + DataLoader
│   ├── optimizer.py          # Adam, SGD, SGDMomentum
│   └── loss.py               # CrossEntropyLoss, FocalLoss
│
├── data/
│   ├── test/
│   ├── train/
│   └── val/                  
│
└── ui/
    ├── app.py                # Entry point Streamlit
    ├── assets/style.css      # Custom styles
    ├── pages/
    │   ├── HOME.py           # Introduction & project overview
    │   ├── PREDICT.py        # Image upload + prediction results
    │   └── PERFORMANCE.py    # Model evaluation page
    └── components/
        ├── sidebar.py        # Navigation sidebar
        ├── image_upload.py   # Upload & preprocess images
        ├── prediction_card.py# Display prediction results
        ├── metrics_card.py   # Metric tiles
        ├── matrix_view.py    # Confusion matrix heatmap
        └── charts.py         # Loss / accuracy / bar charts
""",
        language="text",
    )

    # ── Dataset info ─────────────────────────────────────────
    st.markdown("### 🌱 Dataset – PlantVillage")
    dcol1, dcol2 = st.columns([2, 1])
    with dcol1:
        st.markdown(
            """
            **PlantVillage** is a renowned plant leaf image dataset published by Penn State University,
            containing **54,305 images** across **38 classes** (diseased and healthy) from 14 plant species.

            In this project, we selected **10 representative classes** to balance diversity and training speed. 
            All images are resized to **64×64 pixels** and split in an **80/20** ratio (train/val).

            > Source: [Kaggle PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
            """
        )
    with dcol2:
        st.markdown(
            """
            <div class="plant-card" style="text-align:center;">
                <div style="font-size:2rem;">🌿</div>
                <div style="font-size:1.6rem; font-weight:800; color:#34d399;">54,305</div>
                <div style="font-size:0.75rem; color:#6b7280;">Total images</div>
                <hr style="border-color:rgba(16,185,129,0.15); margin:0.7rem 0;">
                <div style="font-size:1.6rem; font-weight:800; color:#f59e0b;">38</div>
                <div style="font-size:0.75rem; color:#6b7280;">Full classes</div>
                <hr style="border-color:rgba(16,185,129,0.15); margin:0.7rem 0;">
                <div style="font-size:1.6rem; font-weight:800; color:#60a5fa;">10</div>
                <div style="font-size:0.75rem; color:#6b7280;">Used in project</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
