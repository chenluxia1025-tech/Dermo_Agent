# Dermo_Agent

本项目旨在将深度视觉模型与医学教材检索相结合，提供一个用于皮肤镜影像初步筛查与相似病例检索的研究与演示平台。项目面向临床研究者与开发者，利用 HAM10000 数据集的影像与元数据，结合 CLIP 特征和向量检索（FAISS），以及文本检索（PDF 教材）来辅助生成更具参考价值的分析结果。该系统仅作学术/研究用途，不作为临床诊断的最终依据。

轻量说明与运行指南，用于本地启动皮肤镜影像分析 Streamlit 应用。

## 概要

- 入口文件：`src/app.py`（Streamlit 应用）
- 辅助：`build_visual_kb.py`（构建视觉向量索引）
- 数据：`HAM10000_images_part_1/`、`HAM10000_images_part_2/`、`HAM10000_metadata.csv`、可选 PDF 教材
- 索引文件（可选已有）：`visual_kb.index`、`image_paths.pkl`、`dermo_faiss_index/`

## 先决条件

- macOS，已安装 Python 3.11/3.12（建议）
- 推荐使用项目自带虚拟环境（本仓库下为 `venv`）或创建新的 venv

## 快速开始

1. 进入项目目录并激活虚拟环境（如果你使用仓内 `venv`）：

```bash
cd /Users/luxiachen/Desktop/code/Dermo_Agent
source venv/bin/activate
```

2. 安装依赖（如果需要）：

```bash
pip install -r requirements.txt
```

3. 设置必要的环境变量（示例）：

```bash
export ALIYUN_API_KEY="your_dashscope_api_key"
# 可选：当遇到 transformers+torch 安全检查问题时（代码中已设置）
# export TRANSFORMERS_SKIP_TORCH_LOAD_CHECK=True
```

4. 运行 Streamlit 应用：

```bash
streamlit run src/app.py
```

访问地址通常为 `http://localhost:8501`。

## 构建视觉知识库（如果你没有 `visual_kb.index`）

脚本 `build_visual_kb.py` 会从 `HAM10000_metadata.csv` 和图片文件夹读取图片、提取 CLIP 特征并写入 `visual_kb.index` 与 `image_paths.pkl`。

```bash
# 激活 venv 后运行
python build_visual_kb.py
```

注意：构建索引需要较多内存与显卡/CPU 时间，按需调整批次大小或在更强机器上运行。

## 常见问题与提示

- 如果遇到 `faiss` 安装问题，在 macOS 上优先使用 `faiss-cpu` 或通过 conda 安装对应的 faiss 包。
- 代码内已设置 `TRANSFORMERS_SKIP_TORCH_LOAD_CHECK=True` 用于避开某些 transformers+torch 的加载安全检查（仅在必要时使用）。
- 若要使用 GPU 或 Apple MPS，请确保 `torch` 版本与系统兼容。脚本中根据 `torch.backends.mps.is_available()` 自动选择设备。
- 如果缺少文本检索库索引，检查 `dermo_faiss_index/` 文件夹或确保 PDF 教材文件存在。

## 可选改进

- 将 `requirements.txt` 划分为 `requirements-minimal.txt`（仅直接依赖）和 `requirements-full.txt`（pip freeze 全量），便于部署。
- 提供 `docker` 镜像或 `docker-compose` 来简化部署，解决本地环境差异问题。

---

如需，我可以：
- 在当前 venv 中实际运行 `streamlit run src/app.py` 并报告输出/错误（你授权我执行）。
- 把当前 `requirements.txt` 另存为 `requirements-full.txt`（包含 pip freeze 的全部包），并保留当前精简文件为 `requirements-minimal.txt`。


# Dermo_Agent