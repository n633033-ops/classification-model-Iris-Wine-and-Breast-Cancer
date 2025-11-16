import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from gensim.models import Word2Vec
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

# Cấu hình trang
st.set_page_config(page_title="Vietnamese Synonym Analysis", layout="wide")
st.title("🔤 Phân Tích Từ Đồng Nghĩa & Trái Nghĩa Tiếng Việt")
st.markdown("### Sử dụng GloVe và BERT")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('Vietnamese_SynAnt.csv')
    return df

# Train GloVe-like model (Word2Vec with similar approach)
@st.cache_resource
def train_glove_model(df):
    # Tạo corpus từ các cặp từ
    sentences = []
    for _, row in df.iterrows():
        word1, word2, label = row['word1'], row['word2'], row['label']
        sentences.append([word1, word2])
        # Thêm cả chiều ngược lại
        sentences.append([word2, word1])
    
    # Train Word2Vec model (tương tự GloVe)
    model = Word2Vec(sentences=sentences, vector_size=100, window=2, 
                     min_count=1, workers=4, epochs=50, sg=0)
    return model

# Load BERT model
@st.cache_resource
def load_bert_model():
    try:
        # Thử tải PhoBERT (mô hình tiếng Việt)
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", local_files_only=False)
        model = AutoModel.from_pretrained("vinai/phobert-base", local_files_only=False)
        return tokenizer, model, "PhoBERT"
    except Exception as e:
        st.warning(f"⚠️ Không thể tải PhoBERT: {str(e)}")
        st.info("🔄 Đang chuyển sang sử dụng mô hình multilingual-MiniLM...")
        try:
            # Fallback: Sử dụng mô hình nhỏ hơn, hỗ trợ đa ngôn ngữ
            tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            model = AutoModel.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            return tokenizer, model, "Multilingual-MiniLM"
        except Exception as e2:
            st.error(f"❌ Không thể tải bất kỳ mô hình BERT nào: {str(e2)}")
            st.info("💡 Vui lòng kiểm tra kết nối internet hoặc tải mô hình thủ công")
            return None, None, None

def get_bert_embedding(word, tokenizer, model):
    if tokenizer is None or model is None:
        return None
    try:
        inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        # Lấy mean của hidden states
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embeddings
    except Exception as e:
        st.warning(f"Lỗi khi tính BERT embedding cho '{word}': {str(e)}")
        return None

def get_glove_embedding(word, model):
    try:
        return model.wv[word]
    except KeyError:
        return None

def compute_similarity(vec1, vec2):
    if vec1 is None or vec2 is None:
        return None
    return 1 - cosine(vec1, vec2)

def classify_relationship(similarity, threshold_syn=0.7, threshold_ant=0.3):
    if similarity is None:
        return "❓ Không xác định được"
    elif similarity > threshold_syn:
        return "✅ Từ đồng nghĩa"
    elif similarity < threshold_ant:
        return "❌ Từ trái nghĩa"
    else:
        return "⚪ Từ không liên quan"

def plot_vector_map(embeddings_dict, word1, word2, title):
    # Lấy tất cả vectors
    words = list(embeddings_dict.keys())
    vectors = np.array([embeddings_dict[w] for w in words])
    
    # Giảm chiều xuống 2D
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)
    
    # Tạo dataframe
    df_plot = pd.DataFrame({
        'x': vectors_2d[:, 0],
        'y': vectors_2d[:, 1],
        'word': words
    })
    
    # Xác định màu sắc
    colors = []
    sizes = []
    for w in words:
        if w == word1 or w == word2:
            colors.append('red')
            sizes.append(15)
        else:
            colors.append('black')
            sizes.append(8)
    
    # Vẽ biểu đồ
    fig = go.Figure()
    
    # Thêm các điểm
    fig.add_trace(go.Scatter(
        x=df_plot['x'],
        y=df_plot['y'],
        mode='markers+text',
        marker=dict(size=sizes, color=colors, opacity=0.7),
        text=df_plot['word'],
        textposition='top center',
        textfont=dict(size=10),
        hovertemplate='<b>%{text}</b><br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>'
    ))
    
    # Vẽ đường nối giữa 2 từ kiểm tra
    if word1 in words and word2 in words:
        idx1 = words.index(word1)
        idx2 = words.index(word2)
        fig.add_trace(go.Scatter(
            x=[vectors_2d[idx1, 0], vectors_2d[idx2, 0]],
            y=[vectors_2d[idx1, 1], vectors_2d[idx2, 1]],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="PC1",
        yaxis_title="PC2",
        height=500,
        showlegend=False,
        hovermode='closest'
    )
    
    return fig

# Main app
try:
    df = load_data()
    st.success(f"✅ Đã load {len(df)} cặp từ từ dataset")
    
    # Hiển thị thông tin dataset
    with st.expander("📊 Xem dataset"):
        st.dataframe(df.head(20))
    
    # Train models
    with st.spinner("🔄 Đang huấn luyện mô hình GloVe..."):
        glove_model = train_glove_model(df)
    st.success("✅ Đã huấn luyện xong mô hình GloVe")
    
    with st.spinner("🔄 Đang tải mô hình BERT..."):
        tokenizer, bert_model, model_name = load_bert_model()
    
    if tokenizer is not None and bert_model is not None:
        st.success(f"✅ Đã tải xong mô hình BERT ({model_name})")
        bert_available = True
    else:
        st.error("❌ Không thể tải mô hình BERT. Chỉ sử dụng GloVe.")
        bert_available = False
    
    # Lấy tất cả từ unique
    all_words = pd.concat([df['word1'], df['word2']]).unique().tolist()
    
    # Tính embeddings cho tất cả từ
    with st.spinner("🔄 Đang tính embeddings..."):
        glove_embeddings = {}
        bert_embeddings = {}
        
        progress_bar = st.progress(0)
        total_words = len(all_words)
        
        for idx, word in enumerate(all_words):
            # GloVe embeddings
            glove_emb = get_glove_embedding(word, glove_model)
            if glove_emb is not None:
                glove_embeddings[word] = glove_emb
            
            # BERT embeddings (chỉ khi có mô hình)
            if bert_available:
                bert_emb = get_bert_embedding(word, tokenizer, bert_model)
                if bert_emb is not None:
                    bert_embeddings[word] = bert_emb
            
            # Cập nhật progress
            progress_bar.progress((idx + 1) / total_words)
        
        progress_bar.empty()
    
    st.success(f"✅ Đã tính embeddings cho {len(all_words)} từ (GloVe: {len(glove_embeddings)}, BERT: {len(bert_embeddings)})")
    
    # Input section
    st.markdown("---")
    st.markdown("### 🔍 Kiểm Tra Hai Từ")
    
    col1, col2 = st.columns(2)
    with col1:
        word1 = st.selectbox("Chọn từ thứ nhất:", all_words, index=0)
    with col2:
        word2 = st.selectbox("Chọn từ thứ hai:", all_words, index=1)
    
    if st.button("🚀 Phân Tích", type="primary"):
        st.markdown("---")
        
        # GloVe Analysis
        st.markdown("## 📊 Phân Tích với GloVe")
        col_g1, col_g2 = st.columns([1, 2])
        
        with col_g1:
            glove_vec1 = glove_embeddings.get(word1)
            glove_vec2 = glove_embeddings.get(word2)
            glove_sim = compute_similarity(glove_vec1, glove_vec2)
            glove_rel = classify_relationship(glove_sim)
            
            st.metric("Độ tương đồng", f"{glove_sim:.4f}" if glove_sim else "N/A")
            st.markdown(f"### {glove_rel}")
            
            if glove_sim:
                st.progress(glove_sim)
        
        with col_g2:
            if glove_vec1 is not None and glove_vec2 is not None:
                fig_glove = plot_vector_map(glove_embeddings, word1, word2, 
                                           "Vector Map - GloVe")
                st.plotly_chart(fig_glove, use_container_width=True)
            else:
                st.warning("Không thể vẽ map cho GloVe (thiếu embeddings)")
        
        st.markdown("---")
        
        # BERT Analysis
        if bert_available and len(bert_embeddings) > 0:
            st.markdown(f"## 📊 Phân Tích với BERT ({model_name})")
            col_b1, col_b2 = st.columns([1, 2])
            
            with col_b1:
                bert_vec1 = bert_embeddings.get(word1)
                bert_vec2 = bert_embeddings.get(word2)
                bert_sim = compute_similarity(bert_vec1, bert_vec2)
                bert_rel = classify_relationship(bert_sim)
                
                st.metric("Độ tương đồng", f"{bert_sim:.4f}" if bert_sim else "N/A")
                st.markdown(f"### {bert_rel}")
                
                if bert_sim:
                    st.progress(bert_sim)
            
            with col_b2:
                if bert_vec1 is not None and bert_vec2 is not None:
                    fig_bert = plot_vector_map(bert_embeddings, word1, word2, 
                                              "Vector Map - BERT")
                    st.plotly_chart(fig_bert, use_container_width=True)
                else:
                    st.warning("Không thể vẽ map cho BERT (thiếu embeddings)")
            
            # So sánh kết quả
            st.markdown("---")
            st.markdown("## 📈 So Sánh Kết Quả")
            comparison_df = pd.DataFrame({
                'Mô hình': ['GloVe', 'BERT'],
                'Độ tương đồng': [f"{glove_sim:.4f}" if glove_sim else "N/A", 
                                 f"{bert_sim:.4f}" if bert_sim else "N/A"],
                'Quan hệ': [glove_rel, bert_rel]
            })
            st.table(comparison_df)
        else:
            st.warning("⚠️ Mô hình BERT không khả dụng. Chỉ hiển thị kết quả GloVe.")

except FileNotFoundError:
    st.error("❌ Không tìm thấy file 'Vietnamese_SynAnt.csv'. Vui lòng đảm bảo file có trong cùng thư mục.")
except Exception as e:
    st.error(f"❌ Lỗi: {str(e)}")
    st.info("Hãy chắc chắn bạn đã cài đặt các thư viện cần thiết: streamlit, pandas, numpy, scikit-learn, plotly, gensim, transformers, torch")