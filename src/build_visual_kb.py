import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import faiss
import os
import numpy as np
import pickle

# 1. 环境准备与 Bug 修复
# 修复：module 'torch.compiler' has no attribute 'is_compiling'
import torch.compiler
if not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"当前使用设备: {device}")

# 2. 加载模型
print("正在加载官方 CLIP 模型...")
model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32", 
    use_safetensors=True
).to(device)

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32",
    use_fast=True
)

# 3. 路径配置
metadata_path = 'data/HAM10000_metadata.csv'
image_dirs = ['data/HAM10000_images_part_1', 'data/HAM10000_images_part_2']

df = pd.read_csv(metadata_path)
initial_image_paths = []

print("正在扫描图片文件...")
for _, row in df.iterrows():
    img_id = row['image_id']
    for d in image_dirs:
        path = os.path.join(d, f"{img_id}.jpg")
        if os.path.exists(path):
            initial_image_paths.append(path)
            break

# 4. 批处理向量化
all_embeddings = []
processed_paths = []  # 记录真正处理成功的图片路径，确保索引对齐
batch_size = 32 
print(f"开始处理 {len(initial_image_paths)} 张图片...")

for i in range(0, len(initial_image_paths), batch_size):
    batch_paths = initial_image_paths[i : i + batch_size]
    images = []
    current_batch_paths = []
    
    for p in batch_paths:
        try:
            img = Image.open(p).convert("RGB")
            images.append(img)
            current_batch_paths.append(p)
        except:
            continue
            
    if not images:
        continue

    try:
        # 处理图片
        inputs = processor(images=images, return_tensors="pt").to(device)
        
        # 提取特征
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            # 归一化
            image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
            all_embeddings.append(image_features.cpu().numpy())
            # 只有处理成功了，才把路径加入列表
            processed_paths.extend(current_batch_paths)
            
    except Exception as e:
        print(f"\n批次 {i} 处理失败: {e}")
        continue
    
    if (i + batch_size) % 320 == 0:
        print(f"进度: {i + batch_size} / {len(initial_image_paths)}")

# 5. 构建与保存索引
if not all_embeddings:
    print("❌ 错误：没有任何向量被成功提取，请检查上述报错信息。")
else:
    print("正在构建并保存索引...")
    final_embeddings = np.vstack(all_embeddings).astype('float32')
    dimension = final_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(final_embeddings)

    faiss.write_index(index, "image_index/visual_kb.index")
    
    # 保存与向量严格对应的图片路径
    with open("image_paths.pkl", "wb") as f:
        pickle.dump(processed_paths, f)

    print(f"✅ 视觉知识库构建成功！共处理 {len(processed_paths)} 张图片。")