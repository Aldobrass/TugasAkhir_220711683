import streamlit as st
import pickle
import numpy as np
import re
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
.stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 12px;
    margin-top: 0.6rem;
}
.stat-card {
    background: white;
    border-radius: 10px;
    padding: 1rem 1.1rem;
    border: 1px solid #F3F4F6;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.stat-card .label {
    font-size: 0.72rem;
    color: #9CA3AF;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 4px;
}
.stat-card .value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #EE4D2D;
    line-height: 1;
}
.stat-card .sub {
    font-size: 0.75rem;
    color: #6B7280;
    margin-top: 2px;
}
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
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
.divider {
    border: none;
    border-top: 1px solid #F3F4F6;
    margin: 1.1rem 0;
}
.info-box {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    font-size: 0.85rem;
    color: #1E40AF;
    margin-bottom: 1rem;
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
    best_model = components['best_model']
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
    tokens = re.findall(r'\b\w+\b', str(text).lower())
    tokens = [w for w in tokens if w not in custom_stopwords]
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        stemmer = StemmerFactory().create_stemmer()
        tokens = [stemmer.stem(w) for w in tokens]
    except ImportError:
        pass
    return ' '.join(tokens)

# ─────────────────────────────────────────────
# LABEL META
# ─────────────────────────────────────────────
LABEL_META = {
    'harga_negatif':            {'emoji': '💸', 'type': 'neg', 'display': 'Harga — Negatif'},
    'harga_positif':            {'emoji': '✅', 'type': 'pos', 'display': 'Harga — Positif'},
    'layanan_pelanggan_negatif':{'emoji': '😞', 'type': 'neg', 'display': 'Layanan Pelanggan — Negatif'},
    'layanan_pelanggan_positif':{'emoji': '😊', 'type': 'pos', 'display': 'Layanan Pelanggan — Positif'},
    'performa_aplikasi_negatif':{'emoji': '🐌', 'type': 'neg', 'display': 'Performa Aplikasi — Negatif'},
    'performa_aplikasi_positif':{'emoji': '🚀', 'type': 'pos', 'display': 'Performa Aplikasi — Positif'},
}

# ─────────────────────────────────────────────
# PREDICT SINGLE
# ─────────────────────────────────────────────
def predict_single(text: str):
    clean = preprocess(text)
    pred = best_model.predict([clean])
    predicted_labels = mlb.inverse_transform(pred)

    prob_dict = None
    try:
        classifier = best_model.named_steps['clf']
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
# PREDICT BATCH
# ─────────────────────────────────────────────
def predict_batch(texts):
    results = []
    for text in texts:
        labels, prob_dict = predict_single(text)
        label_str = ', '.join([LABEL_META.get(l, {'display': l})['display'] for l in labels]) if labels else 'Tidak ada label'
        row = {'ulasan': text, 'label_terdeteksi': label_str}
        if prob_dict:
            for lbl, prob in prob_dict.items():
                meta = LABEL_META.get(lbl, {'display': lbl})
                row[f'prob_{meta["display"]}'] = round(prob * 100, 1)
        results.append(row)
    return pd.DataFrame(results)

# ─────────────────────────────────────────────
# EXPORT TO EXCEL
# ─────────────────────────────────────────────
def to_excel(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Hasil Klasifikasi')
        # Auto-adjust column widths
        ws = writer.sheets['Hasil Klasifikasi']
        for col in ws.columns:
            max_len = max(len(str(cell.value)) if cell.value else 0 for cell in col)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 60)
    return output.getvalue()

# ─────────────────────────────────────────────
# CHART: LABEL DISTRIBUTION
# ─────────────────────────────────────────────
def plot_label_distribution(df_results: pd.DataFrame):
    # Hitung kemunculan setiap label display
    all_labels = []
    for label_str in df_results['label_terdeteksi']:
        if label_str != 'Tidak ada label':
            for lbl in label_str.split(', '):
                all_labels.append(lbl.strip())

    if not all_labels:
        return None

    from collections import Counter
    counts = Counter(all_labels)
    total = sum(counts.values())

    # Sort dari terbanyak ke tersedikit
    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    labels_sorted = [item[0] for item in sorted_items]
    values_sorted = [item[1] for item in sorted_items]
    pcts_sorted   = [v / total * 100 for v in values_sorted]

    # Warna berdasarkan tipe
    def get_color(label_display):
        for key, meta in LABEL_META.items():
            if meta['display'] == label_display:
                return '#10B981' if meta['type'] == 'pos' else '#EF4444'
        return '#6B7280'

    colors = [get_color(l) for l in labels_sorted]

    fig, ax = plt.subplots(figsize=(7, max(3, len(labels_sorted) * 0.65)))
    fig.patch.set_facecolor('#FFF7F5')
    ax.set_facecolor('#FFF7F5')

    bars = ax.barh(labels_sorted[::-1], values_sorted[::-1], color=colors[::-1],
                   height=0.55, edgecolor='white', linewidth=0.8)

    # Tambah persen dan jumlah di ujung bar
    for bar, val, pct in zip(bars, values_sorted[::-1], pcts_sorted[::-1]):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                f'{val}  ({pct:.1f}%)', va='center', ha='left',
                fontsize=8.5, color='#374151', fontweight='600')

    ax.set_xlabel('Jumlah Kemunculan', fontsize=9, color='#6B7280')
    ax.set_title('Distribusi Label — Dari Terbanyak ke Tersedikit', fontsize=11,
                 fontweight='700', color='#374151', pad=12)

    ax.tick_params(axis='y', labelsize=8.5, colors='#374151')
    ax.tick_params(axis='x', labelsize=8, colors='#9CA3AF')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#E5E7EB')
    ax.xaxis.grid(True, linestyle='--', alpha=0.4, color='#D1D5DB')
    ax.set_axisbelow(True)

    # Legend
    pos_patch = mpatches.Patch(color='#10B981', label='Positif')
    neg_patch = mpatches.Patch(color='#EF4444', label='Negatif')
    ax.legend(handles=[pos_patch, neg_patch], fontsize=8.5, framealpha=0,
              loc='lower right', labelcolor='#374151')

    # Extra x margin agar label angka tidak terpotong
    ax.set_xlim(0, max(values_sorted) * 1.35)

    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────
# UI — HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">🛍️ Klasifikasi Ulasan Shopee</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Upload file Excel berisi ulasan — model akan mengklasifikasi setiap ulasan secara otomatis dan hasilnya bisa diunduh.</div>',
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────
# INFO BOX + UPLOAD
# ─────────────────────────────────────────────
st.markdown("""
<div class="info-box">
    📋 <strong>Format file Excel:</strong> Pastikan ada kolom bernama <code>ulasan</code> atau <code>review</code> atau <code>text</code> yang berisi teks ulasan.
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload File Excel (.xlsx / .xls)",
    type=['xlsx', 'xls'],
    help="File harus memiliki kolom ulasan/review/text"
)

if uploaded_file is not None:
    try:
        df_input = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"❌ Gagal membaca file: {e}")
        st.stop()

    # Deteksi kolom ulasan
    col_map = {c.lower(): c for c in df_input.columns}
    text_col = None
    for candidate in ['ulasan', 'review', 'text', 'komentar', 'komentar_ulasan']:
        if candidate in col_map:
            text_col = col_map[candidate]
            break

    if text_col is None:
        st.error(f"❌ Kolom ulasan tidak ditemukan. Kolom yang tersedia: {list(df_input.columns)}")
        st.info("Ubah nama kolom teks ulasan Anda menjadi **ulasan**, **review**, atau **text**.")
        st.stop()

    st.success(f"✅ File berhasil dibaca — **{len(df_input)} baris** ditemukan, kolom teks: `{text_col}`")

    # Preview
    with st.expander("👁️ Preview data (5 baris pertama)"):
        st.dataframe(df_input.head(5), use_container_width=True)

    # ── TOMBOL ANALISIS ──
    if st.button("🔍 Mulai Klasifikasi Batch", type="primary", use_container_width=True):
        texts = df_input[text_col].fillna('').astype(str).tolist()

        progress = st.progress(0, text="Memulai klasifikasi...")
        results = []
        n = len(texts)

        for i, text in enumerate(texts):
            labels, prob_dict = predict_single(text)
            label_str = ', '.join([LABEL_META.get(l, {'display': l})['display'] for l in labels]) if labels else 'Tidak ada label'
            row = {text_col: text, 'label_terdeteksi': label_str}
            if prob_dict:
                for lbl, prob in prob_dict.items():
                    meta = LABEL_META.get(lbl, {'display': lbl})
                    row[f'prob_{meta["display"]} (%)'] = round(prob * 100, 1)
            results.append(row)
            progress.progress((i + 1) / n, text=f"Menganalisis ulasan {i+1} / {n}...")

        progress.empty()
        df_results = pd.DataFrame(results)
        st.session_state['df_results'] = df_results
        st.success(f"✅ Klasifikasi selesai! **{n} ulasan** berhasil dianalisis.")

# ─────────────────────────────────────────────
# TAMPILKAN HASIL
# ─────────────────────────────────────────────
if 'df_results' in st.session_state:
    df_results = st.session_state['df_results']

    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    # ── STATISTIK RINGKASAN ──
    st.markdown('<h4>Ringkasan Hasil</h4>', unsafe_allow_html=True)

    total      = len(df_results)
    berlabel   = (df_results['label_terdeteksi'] != 'Tidak ada label').sum()
    tidak_ada  = total - berlabel

    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="label">Total Ulasan</div>
            <div class="value">{total}</div>
            <div class="sub">baris diproses</div>
        </div>
        <div class="stat-card">
            <div class="label">Terklasifikasi</div>
            <div class="value">{berlabel}</div>
            <div class="sub">memiliki label</div>
        </div>
        <div class="stat-card">
            <div class="label">Tanpa Label</div>
            <div class="value">{tidak_ada}</div>
            <div class="sub">tidak terdeteksi</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── CHART DISTRIBUSI LABEL ──
    st.markdown('<h4>Distribusi Label (Terbanyak → Tersedikit)</h4>', unsafe_allow_html=True)
    fig = plot_label_distribution(df_results)
    if fig:
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("Tidak ada label yang terdeteksi untuk ditampilkan.")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── TABEL DETAIL ──
    st.markdown('<h4>Detail Hasil Klasifikasi</h4>', unsafe_allow_html=True)

    # Kolom ulasan + label_terdeteksi saja untuk preview ringkas
    preview_cols = [c for c in df_results.columns if not c.startswith('prob_')]
    st.dataframe(df_results[preview_cols], use_container_width=True, height=280)

    with st.expander("📊 Lihat semua kolom termasuk skor probabilitas"):
        st.dataframe(df_results, use_container_width=True, height=280)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── DOWNLOAD ──
    st.markdown('<h4>Unduh Hasil</h4>', unsafe_allow_html=True)

    excel_bytes = to_excel(df_results)
    st.download_button(
        label="📥 Download Hasil Klasifikasi (.xlsx)",
        data=excel_bytes,
        file_name="hasil_klasifikasi_shopee.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        type="primary"
    )

    st.markdown('</div>', unsafe_allow_html=True)

elif uploaded_file is None:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 1rem; color: #9CA3AF;">
        <div style="font-size:3rem; margin-bottom:0.5rem;">📂</div>
        <div style="font-size:0.95rem;">Upload file Excel untuk memulai</div>
    </div>
    """, unsafe_allow_html=True)

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