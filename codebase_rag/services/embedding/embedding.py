import json
import os
import torch
import numpy as np

from typing import List
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer
from pathlib import Path

from codebase_rag.services.embedding.text_embedding_analyze import analyze_files_with_llm

load_dotenv(override=True)

top_k_1 = 10
top_k_2 = 5
target_repo_path = os.getenv("TARGET_REPO_PATH")

test_path = "D:\\Disaster\\Codefield\\Code_Python\\CodeGraphRAG\\tmp\\ch-classify_result.json"

BASE_DIR = Path(__file__).resolve().parent
TMP_DIR = BASE_DIR / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

def cosine_similarity(a: List, b: List) -> float:
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data


def build_file_nodes(repo_path, files):
    outputs = []
    for file_path in files:
        abs_path = os.path.join(repo_path, file_path)

        if not os.path.exists(abs_path):
            print(f"[WARN] 文件不存在: {abs_path}")
            continue

        try:
            with open(abs_path, 'r', encoding='utf-8') as f:
                contents = f.read()
        except Exception as e:
            print(f"[ERROR] 无法读取 {abs_path}: {e}")

        outputs.append({
            "repo_path": repo_path,
            "file_path": file_path,
            "contents": contents,
        })
    return outputs


def build_code2text_nodes(files):
    code2text = []
    with open(files, 'r', encoding='utf-8') as f:
        for line in f:
            # 解析每行的JSON对象
            try:
                match = json.loads(line.strip())
                # 提取file_path并添加到列表
                if "summary" in match:
                    code2text.append({
                        "repo_path": match["repo_path"],
                        "file_path": match["file_path"],
                        "summary": match["summary"]
                    })
            except json.JSONDecodeError as e:
                print(f"[WARN] 解析JSONL行失败: {e}, 行内容: {line}")
    return code2text


def build_code_embeddings(nodes):
    embeddings = {}
    for node in nodes:
        file_path = node["file_path"]
        contents = node["contents"]
        embeddings[file_path] = generate_code_embedding(contents)
        print(file_path)
        print(embeddings[file_path])
    return embeddings


def build_text_embeddings(nodes):
    embeddings = {}
    for node in nodes:
        file_path = node["file_path"]
        summary = node["summary"]
        embeddings[file_path] = generate_text_embedding(summary)
        print(file_path)
        print(embeddings[file_path])
    return embeddings


def generate_code_embedding(text):
    hf_models_home = os.getenv("HF_MODELS_HOME")
    hf_model_id = os.getenv("LOCAL_CODE_EMBEDDING_MODEL_ID")
    local_model_dir = Path(Path(hf_models_home) / hf_model_id).as_posix()
    model = AutoModel.from_pretrained(
        local_model_dir,
        trust_remote_code=True,
    )

    embeddings = model.encode(text, max_length=8191)
    return embeddings


def generate_text_embedding(text):
    hf_models_home = os.getenv("HF_MODELS_HOME")
    hf_model_id = os.getenv("LOCAL_TEXT_EMBEDDING_MODEL_ID")
    local_model_dir = Path(Path(hf_models_home) / hf_model_id).as_posix()

    tokenizer = AutoTokenizer.from_pretrained(
        local_model_dir,
        trust_remote_code=True,
        padding=True,
        truncation=True
    )

    model = AutoModel.from_pretrained(
        local_model_dir,
        trust_remote_code=True,
    )

    # 文本编码（处理长文本，max_length 设为模型支持的最大值）
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=8191
    )

    # 将输入移到模型所在设备
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 推理（禁用梯度计算以提高效率）
    with torch.no_grad():
        outputs = model(**inputs)
        # Qwen3-Embedding 的输出通常包含 'last_hidden_state'，需要进行平均池化
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    return embeddings

def match_embeddings(direct_embeddings, other_embeddings, other_nodes, top_k):
    outputs = []

    for node in other_nodes:
        other_file = node["file_path"]
        other_repo = node["repo_path"]
        other_emb = other_embeddings[other_file]

        max_score = -1.0
        related = ""
        for direct_fp, direct_emb in direct_embeddings.items():
            score = cosine_similarity(direct_emb, other_emb)
            if score > max_score:
                max_score = score
                related = direct_fp

        outputs.append({
            "repo_path": other_repo,
            "file_path": other_file,
            "score": max_score,
            "related": related
        })

    outputs.sort(key=lambda x: x["score"], reverse=True)
    return outputs[:top_k]


def write_jsonl(matches, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for match in matches:
            # 确保所有值都是JSON可序列化的类型
            serializable_match = {
                "repo_path": match["repo_path"],
                "file_path": match["file_path"],
                "score": float(match["score"]),  # 确保是Python原生float
                "related": match["related"]
            }
            # 每行写入一个JSON对象
            f.write(json.dumps(serializable_match, ensure_ascii=False) + '\n')


def process_initial_classification(json_path, repo_path):
    repo_name = Path(repo_path).name
    data = load_json(json_path)
    direct_files = data["direct_related"]
    other_files = data["other_files"]

    direct_nodes = build_file_nodes(repo_path, direct_files)
    analyze_files_with_llm(direct_nodes, TMP_DIR / f"{repo_name}_direct_files_code2text_direct.jsonl")

    other_nodes = build_file_nodes(repo_path, other_files)

    direct_embeddings = build_code_embeddings(direct_nodes)
    other_embeddings = build_code_embeddings(other_nodes)

    match_results = match_embeddings(direct_embeddings, other_embeddings, other_nodes, top_k_1)

    write_jsonl(match_results, TMP_DIR / f"{repo_name}_other_files_rank_stage1.jsonl")

    matched_file_paths = []
    with open(TMP_DIR / f"{repo_name}_other_files_rank_stage1.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            # 解析每行的JSON对象
            try:
                obj = json.loads(line.strip())
                # 提取file_path并添加到列表
                if "file_path" in obj:
                    matched_file_paths.append(obj["file_path"])
            except json.JSONDecodeError as e:
                print(f"[WARN] 解析JSONL行失败: {e}, 行内容: {line}")

    ranked_nodes = build_file_nodes(repo_path, matched_file_paths)

    analyze_files_with_llm(ranked_nodes, TMP_DIR / f"{repo_name}_other_files_code2text.jsonl")

    direct_code2text = build_code2text_nodes(TMP_DIR / f"{repo_name}_direct_files_code2text_direct.jsonl")
    other_code2text = build_code2text_nodes(TMP_DIR / f"{repo_name}_other_files_code2text.jsonl")

    direct_text_embeddings = build_text_embeddings(direct_code2text)
    other_text_embeddings = build_text_embeddings(other_code2text)

    match_results = match_embeddings(direct_text_embeddings, other_text_embeddings, ranked_nodes, top_k_2)

    write_jsonl(match_results, TMP_DIR / f"{repo_name}_other_files_rank_stage2.jsonl")

if __name__ == "__main__":
    process_initial_classification(test_path, target_repo_path)
