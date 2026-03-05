# EDA Elite Challenge 2025 · Track 2 — One-Stop Pipeline

This repository provides a **one-stop entry script** to run inference and export **competition-format JSON outputs** for  
**2025 China Graduate IC Innovation Competition · EDA Elite Challenge (Track 2: Circuit System Block Diagram Recognition & Parsing)**.

---

## Goal

Given integrated-circuit system block diagrams, the goal is to automatically convert **unstructured images** into **structured design knowledge**, and produce **JSON outputs** that match the competition requirements.

---

## Supported Tasks

### Task 1: Structured Topology Reconstruction
Using **CV + multimodal models** to:
- Detect components (name / bounding box)
- Count component I/O ports (number of input / output ports)
- Extract inter-component connections (`input` / `output` lists)
- Produce competition-format JSON aligned one-to-one with the provided labels

### Task 2: Logical Reasoning QA
Using a multimodal LLM to reason about analog-circuit related questions:
- Generate and answer **multiple-choice** / **fill-in-the-blank** QA
- Validate understanding of the diagram’s overall logic and intended functionality

---

## Recommended Directory Layout


> Before running, make sure the **base model + adapters**, images, and question files are placed according to your code’s loading paths.

---

## Usage

`entry.py` is the entry point. Run:

```bash
python entry.py   --image_path EDA_CASES_1024/images   --task2_question_path EDA_CASES_1024/task2_questions   --output_path 02_entry_template
```

Arguments:
- `--image_path`: directory of competition images
- `--task2_question_path`: directory of Task 2 questions
- `--output_path`: directory to write output JSON files (competition format)

---

## Model Setup

- Base model: **Qwen2.5-VL-3B**
- Adapters: `adapter_model.safetensors` (trained separately)

Please place the base model and corresponding adapters before running (paths follow your code/config).

---

## Dataset Notes

### Task 1 Training Data
- `connection_Image1-5/`: connection training images (for Task 1)
- `train_connection.json`: provided wire/connection annotation dataset

### Task 2 Training Data
- `image1-5/`: original images (for Task 2)
- `mllm_data.json`: provided VQA dataset

---

## Training & Reproduction (LoRA)

We use **LLaMA-Factory** for LoRA fine-tuning and hyperparameter settings to reproduce competition results.

High-level steps:
1. Prepare Task 1 / Task 2 datasets
2. Fine-tune with LoRA (or QLoRA) using LLaMA-Factory
3. Export `adapter_model.safetensors`
4. Load **Qwen2.5-VL-3B + adapter**, then run `entry.py` to generate competition-format outputs

---

## Disclaimer & Data Source

All dataset images originate from:
**2025 China Graduate IC Innovation Competition · EDA Elite Challenge (Track 2)**.

If any content infringes rights, please contact **cxin7354@gmail.com** for removal. Thank you.

---

## Contact

- Author: XIN
- Email: cxin7354@gmail.com



# EDA Elite Challenge 2025 · Track 2 — One-Stop Pipeline

本项目提供**一站式脚本**，用于完成 **2025 中国研究生创“芯”大赛·EDA 精英挑战赛（赛题二：电路系统框图识别与解析）** 的端到端推理与结果导出（Json）。

---

## 项目目标

面向集成电路系统框图，实现从**非结构化图像**到**结构化设计知识**的自动解析，输出符合赛题要求的 **Json** 结果。

---

## 任务支持

### 任务一：结构化拓扑重建（Topology Reconstruction）
基于 **CV + 多模态模型** 完成：
- 组件提取：组件名称 / 位置框（bbox）
- 端口统计：组件输入 / 输出端口数量
- 连接关系抽取：组件间连接关系（`input` / `output` 列表）
- 与标签文件的一一对应映射，输出赛题格式 Json

### 任务二：逻辑推理问答（Reasoning QA）
结合多模态大模型对模拟电路相关问题进行推理：
- 生成并回答选择题 / 填空题 QA
- 验证对框图整体逻辑与功能意图的理解能力

---

## 目录结构（与我项目文件一致）

> 运行前请确保：**模型（基座 + adapter）**、图片与问题文件已按代码读取路径放置完成。

---

## 运行方式

`entry.py` 为项目入口脚本，在终端执行：

```bash
python entry.py   --image_path EDA_CASES_1024/images   --task2_question_path EDA_CASES_1024/task2_questions   --output_path 02_entry_template
```

参数说明：
- `--image_path`：赛题图片目录
- `--task2_question_path`：任务二问题目录
- `--output_path`：输出 Json 的目录（按赛题格式写入）

---

## 模型准备

- 基座模型：**Qwen2.5-VL-3B**
- Adapter：`adapter_model.safetensors`（分别训练获得）

请在运行前放置好基座模型与对应 adapter（路径以你代码中的加载配置为准）。

---

## 数据集说明

### 任务一训练数据
- `connection_Image1-5/`：连接训练图片（用于任务一训练）
- `train_connection.json`：已提供连线数据集（搭配上述图片使用）

### 任务二训练数据
- `image1-5/`：原始图片（用于任务二训练）
- `mllm_data.json`：已提供 VQA 数据集（搭配上述图片使用）

---

## 训练与复现（LoRA）

使用 **LLaMA-Factory** 进行 LoRA 微调及参数设置，可复现竞赛结果。流程概述：
1. 准备任务一/任务二对应数据集
2. 使用 LLaMA-Factory 配置 LoRA（或 QLoRA）进行微调训练
3. 导出 `adapter_model.safetensors`
4. 推理阶段加载 **Qwen2.5-VL-3B + adapter**，运行 `entry.py` 生成赛题格式输出

---

## 免责声明与数据来源

数据集图片来自：
**2025 中国研究生创“芯”大赛·EDA 精英挑战赛（赛题二：电路系统框图识别与解析）**。

如有侵权，请联系作者邮箱 **cxin7354@gmail.com** 进行删除。谢谢，望告知。

---

## 联系方式

- Author: XIN
- Email: cxin7354@gmail.com
