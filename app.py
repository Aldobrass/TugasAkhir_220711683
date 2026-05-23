import streamlit as st
import pickle
import numpy as np
import re

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Klasifikasi Ulasan Shopee",
    page_icon="🛍️",
    layout="centered",
)

# ─────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}
.main-title {
    font-size: 2rem;
    font-weight: 700;
    color: #EE4D2D;
    margin-bottom: 0.2rem;
}
.sub-title {
    font-size: 1rem;
    color: #6B7280;
    margin-bottom: 2rem;
}
.result-card {
    background: #FFF7F5;
    border: 1.5px solid #EE4D2D22;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-top: 1.2rem;
}
.result-card h4 {
    color: #374151;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.8rem;
}
.badge-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 0.4rem;
}
.badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.badge-pos {
    background: #D1FAE5;
    color: #065F46;
    border: 1px solid #6EE7B7;
}
.badge-neg {
    background: #FEE2E2;
    color: #991B1B;
    border: 1px solid #FCA5A5;
}
.prob-row {
    display: flex;
    align-items: center;
    margin-bottom: 0.55rem;
    gap: 10px;
}
.prob-label {
    font-size: 0.78rem;
    color: #374151;
    width: 210px;
    flex-shrink: 0;
}
.prob-bar-bg {
    flex: 1;
    background: #F3F4F6;
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #EE4D2D, #FF7849);
}
.prob-val {
    font-size: 0.78rem;
    color: #6B7280;
    width: 38px;
    text-align: right;
    flex-shrink: 0;
}
.no-label {
    color: #9CA3AF;
    font-size: 0.9rem;
    font-style: italic;
}
.divider {
    border: none;
    border-top: 1px solid #F3F4F6;
    margin: 1.1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('model_components.pkl', 'rb') as f:
        components = pickle.load(f)
    return components

try:
    components = load_model()
    best_model = components['best_model']  # Pipeline (tfidf + classifier)
    mlb        = components['mlb']
except FileNotFoundError:
    st.error("❌ File `model_components.pkl` tidak ditemukan. Pastikan file ada di direktori yang sama dengan app.py.")
    st.stop()

# ─────────────────────────────────────────────
# STOPWORDS + PREPROCESS
# ─────────────────────────────────────────────
@st.cache_resource
def build_stopwords():
    try:
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
        sw = set(StopWordRemoverFactory().get_stop_words())
    except ImportError:
        sw = set()
    for k in ['tidak', 'bukan', 'nggak', 'tanpa', 'kecuali', 'belum']:
        sw.discard(k)
    return sw

custom_stopwords = build_stopwords()

def preprocess(text: str) -> str:
    tokens = re.findall(r'\b\w+\b', text.lower())
    tokens = [w for w in tokens if w not in custom_stopwords]
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        stemmer = StemmerFactory().create_stemmer()
        tokens = [stemmer.stem(w) for w in tokens]
    except ImportError:
        pass
    return ' '.join(tokens)

# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────
LABEL_META = {
    'harga_negatif':            {'emoji': '💸', 'type': 'neg', 'display': 'Harga — Negatif'},
    'harga_positif':            {'emoji': '✅', 'type': 'pos', 'display': 'Harga — Positif'},
    'layanan_pelanggan_negatif':{'emoji': '😞', 'type': 'neg', 'display': 'Layanan Pelanggan — Negatif'},
    'layanan_pelanggan_positif':{'emoji': '😊', 'type': 'pos', 'display': 'Layanan Pelanggan — Positif'},
    'performa_aplikasi_negatif':{'emoji': '🐌', 'type': 'neg', 'display': 'Performa Aplikasi — Negatif'},
    'performa_aplikasi_positif':{'emoji': '🚀', 'type': 'pos', 'display': 'Performa Aplikasi — Positif'},
}

def predict(text: str):
    clean = preprocess(text)

    # Pipeline langsung handle tfidf + predict
    pred = best_model.predict([clean])
    predicted_labels = mlb.inverse_transform(pred)

    # Ambil probabilitas dari classifier dalam pipeline
    # BinaryRelevance dengan LR → bisa predict_proba
    prob_dict = None
    try:
        classifier = best_model.named_steps['clf']
        # Transform dulu pakai tfidf step
        tfidf_step = best_model.named_steps['tfidf']
        X_transformed = tfidf_step.transform([clean])

        probs = np.hstack([
            est.predict_proba(X_transformed)[:, 1].reshape(-1, 1)
            for est in classifier.classifiers_
        ])[0]
        prob_dict = dict(zip(mlb.classes_, probs))
    except Exception:
        pass

    return predicted_labels[0], prob_dict

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">🛍️ Klasifikasi Ulasan Shopee</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Masukkan ulasan pengguna — model akan mendeteksi aspek yang dibicarakan secara otomatis.</div>', unsafe_allow_html=True)

cols = st.columns(3)

user_input = st.text_area(
    "Teks Ulasan",
    height=110,
    placeholder="Contoh: aplikasinya bagus tapi ongkirnya mahal banget...",
    label_visibility="collapsed",
)

st.button("🔍 Analisis Ulasan", type="primary", use_container_width=True, key="run_btn")

# ─────────────────────────────────────────────
# HASIL
# ─────────────────────────────────────────────
if st.session_state.get("run_btn"):
    text = user_input.strip()
    if not text:
        st.warning("Masukkan teks ulasan terlebih dahulu.")
    else:
        with st.spinner("Menganalisis..."):
            predicted_labels, prob_dict = predict(text)

        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        st.markdown('<h4>Aspek Terdeteksi</h4>', unsafe_allow_html=True)
        if predicted_labels:
            badge_html = '<div class="badge-wrap">'
            for lbl in predicted_labels:
                meta = LABEL_META.get(lbl, {'emoji': '🏷️', 'type': 'pos', 'display': lbl})
                cls  = 'badge-pos' if meta['type'] == 'pos' else 'badge-neg'
                badge_html += f'<span class="badge {cls}">{meta["emoji"]} {meta["display"]}</span>'
            badge_html += '</div>'
            st.markdown(badge_html, unsafe_allow_html=True)
        else:
            st.markdown('<p class="no-label">Tidak ada aspek yang terdeteksi.</p>', unsafe_allow_html=True)

        if prob_dict:
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<h4>Skor Kepercayaan per Aspek</h4>', unsafe_allow_html=True)
            sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
            bar_html = ''
            for lbl, prob in sorted_probs:
                meta = LABEL_META.get(lbl, {'display': lbl, 'emoji': '🏷️'})
                pct  = int(prob * 100)
                bar_html += f"""
                <div class="prob-row">
                    <span class="prob-label">{meta['emoji']} {meta['display']}</span>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width:{pct}%"></div>
                    </div>
                    <span class="prob-val">{pct}%</span>
                </div>"""
            st.markdown(bar_html, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#9CA3AF;font-size:0.78rem;'>"
    "Albert Cornelius · 220711683 · Universitas Atma Jaya Yogyakarta · 2026"
    "</p>",
    unsafe_allow_html=True,
)