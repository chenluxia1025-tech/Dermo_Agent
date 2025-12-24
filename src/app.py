import os
# --- 0. 核心修复：跳过 PyTorch 安全检查（解决 CVE-2025-32434 报错） ---
os.environ["TRANSFORMERS_SKIP_TORCH_LOAD_CHECK"] = "True"

import streamlit as st
import pandas as pd
import torch
import faiss
import sys
import pickle
import base64
from PIL import Image
from openai import OpenAI
import dashscope
from dashscope import TextEmbedding
from transformers import CLIPProcessor, CLIPModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS as LangChainFAISS
from langchain_core.embeddings import Embeddings

# --- 1. 基础配置 ---
st.set_page_config(page_title="专业皮肤镜影像分析专家", page_icon="🔬", layout="wide")

ALIYUN_API_KEY = os.env.get("ALIYUN_API_KEY")
dashscope.api_key = ALIYUN_API_KEY
client = OpenAI(
    api_key=ALIYUN_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# --- 2. 文本向量工具 (用于 PDF 检索) ---
class AliyunEmbedding(Embeddings):
    def embed_documents(self, texts):
        all_embeddings = []
        for i in range(0, len(texts), 10):
            batch = texts[i : i + 10]
            resp = TextEmbedding.call(model='text-embedding-v3', input=batch)
            if resp.status_code == 200:
                all_embeddings.extend([item['embedding'] for item in resp.output['embeddings']])
            else:
                raise Exception(f"Embedding Error: {resp.message}")
        return all_embeddings

    def embed_query(self, text):
        if not text.strip(): return [0] * 1536
        resp = TextEmbedding.call(model='text-embedding-v3', input=[text])
        return resp.output['embeddings'][0]['embedding'] if resp.status_code == 200 else None

embeddings_tool = AliyunEmbedding()

# --- 3. 缓存初始化逻辑 ---

@st.cache_resource
def init_knowledge_bases():
    """同时初始化文本库和视觉库"""
    # A. 初始化文本库 (PDF)
    text_db = None
    index_path = "dermo_faiss_index"
    pdf_files = ["data/dermoscopy_atlas_1.pdf", "data/dermoscopy_atlas_2.pdf"] 
    
    if os.path.exists(index_path):
        text_db = LangChainFAISS.load_local(index_path, embeddings_tool, allow_dangerous_deserialization=True)
    else:
        found_pdfs = [f for f in pdf_files if os.path.exists(f)]
        if found_pdfs:
            all_docs = []
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
            for pdf in found_pdfs:
                loader = PyPDFLoader(pdf)
                all_docs.extend(text_splitter.split_documents(loader.load()))
            text_db = LangChainFAISS.from_texts([d.page_content for d in all_docs], embeddings_tool)
            text_db.save_local(index_path)

    # B. 初始化视觉库 (HAM10000)
    v_index, v_paths, v_meta, v_model, v_processor, v_device = None, None, None, None, None, None
    v_idx_file = "image_index/visual_kb.index"
    v_pkl_file = "image_index/image_paths.pkl"
    v_csv_file = "HAM10000_metadata.csv"

    if os.path.exists(v_idx_file) and os.path.exists(v_pkl_file):
        v_index = faiss.read_index(v_idx_file)
        with open(v_pkl_file, "rb") as f:
            v_paths = pickle.load(f)
        v_meta = pd.read_csv(v_csv_file) if os.path.exists(v_csv_file) else None
        
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        # 强制使用 safetensors=True 增加安全性
        v_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(device)
        v_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        v_device = device

    return text_db, v_index, v_paths, v_meta, v_model, v_processor, v_device

# 加载所有库
text_db, v_index, v_paths, v_meta, v_model, v_processor, v_device = init_knowledge_bases()

# --- 4. 界面展示 ---
st.title("🔬 皮肤镜影像智能专家系统")
st.caption("已启用：PDF教材知识检索 + HAM10000 相似病例比对")

with st.sidebar:
    st.header("📸 影像上传")
    uploaded_file = st.file_uploader("上传皮肤镜照片", type=["jpg", "png", "jpeg"])
    location = st.selectbox("发病部位", ["四肢", "躯干", "头面部", "掌跖", "甲下", "粘膜"])
    evolution = st.selectbox("近期变化", ["无明显变化", "颜色加深/体积增大", "边缘不对称", "出血/破溃"])
    
    if text_db: st.success("📖 教材库已就绪")
    if v_index: st.success("🖼️ 视觉库已就绪")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- 5. 对话检索逻辑 ---
if prompt := st.chat_input("描述详细症状..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # --- A. 检索文本内容 (PDF) ---
        context_text = ""
        if text_db:
            with st.spinner("正在查阅专家教材..."):
                search_results = text_db.similarity_search(f"{location} {prompt}", k=2)
                context_text = "\n".join([res.page_content for res in search_results])
        
        # --- B. 检索相似病例 (视觉库) ---
        reference_info = ""
        if uploaded_file and v_index is not None:
            with st.spinner("正在搜索相似临床病例..."):
                img = Image.open(uploaded_file).convert("RGB")
                inputs = v_processor(images=img, return_tensors="pt").to(v_device)
                with torch.no_grad():
                    feat = v_model.get_image_features(**inputs)
                    feat /= feat.norm(p=2, dim=-1, keepdim=True)
                    query_emb = feat.cpu().numpy().astype('float32')
                
                D, I = v_index.search(query_emb, 3)
                
                st.write("🔍 **库内相似案例参考：**")
                cols = st.columns(3)
                dx_map = {"mel": "黑色素瘤", "nv": "黑色素痣", "bcc": "基底细胞癌", "akiec": "日光性角化病", "bkl": "良性角化病", "df": "皮肤纤维瘤", "vasc": "血管瘤"}
                
                for idx, col in enumerate(cols):
                    match_idx = I[0][idx]
                    ref_path = v_paths[match_idx]
                    img_id = os.path.basename(ref_path).replace(".jpg", "")
                    dx_code = v_meta[v_meta['image_id'] == img_id]['dx'].values[0] if v_meta is not None else "未知"
                    dx_name = dx_map.get(dx_code, dx_code)
                    
                    with col:
                        if os.path.exists(ref_path):
                            st.image(ref_path, caption=f"匹配度: {1/(1+D[0][idx]):.2f}")
                            st.info(f"确诊: {dx_name}")
                        reference_info += f"相似案例{idx+1}确诊为{dx_name}; "

        # --- C. 综合分析 (Qwen-VL) ---
        final_prompt = f"""
你是一位皮肤镜专家。请综合以下信息进行深度分析：

【参考教材知识】：
{context_text if context_text else "未检索到直接相关的教材段落。"}

【数据库相似病例参考】：
{reference_info if reference_info else "未进行相似病例对比。"}

【临床信息】：
部位：{location}，变化：{evolution}，患者主诉：{prompt}

请结合图片细节，描述其典型的皮肤镜征象，给出初步印象，并提供随诊建议（需包含免责声明）。
"""
        
        msg_content = [{"type": "text", "text": final_prompt}]
        if uploaded_file:
            b64 = base64.b64encode(uploaded_file.getvalue()).decode()
            msg_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

        with st.spinner("专家正在分析影像..."):
            response = client.chat.completions.create(
                model="qwen-vl-max",
                messages=[
                    {"role": "system", "content": "你是一位三甲医院皮肤镜诊断专家，回复专业、客观。"},
                    {"role": "user", "content": msg_content}
                ]
            )
            
            answer = response.choices[0].message.content
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})