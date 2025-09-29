# CodeGraphRag

## Memgraph

首先为了连接memgraph，下载它

## 安装和启动 Memgraph

### 使用 Docker

```bash
# 拉取 Memgraph 镜像
docker pull memgraph/memgraph-platform

# 启动 Memgraph   未进行本地目录与虚拟目录的存储映射
docker run -it -p 7687:7687 -p 3000:3000 -p 7444:7444 -v mg_lib:/var/lib/memgraph memgraph/memgraph-platform
```

Memgraph Lab

```bash
http://localhost:3000/

# 查询所有 nodes 和 relations
MATCH (n)-[r]->(m)
RETURN n, r, m
```


## Ollama

如果使用的是本地Ollama，需要启动Ollama

### 启动 Ollama

```bash
ollama serve
```

此处需要注意，ollama的模型必须支持Tools. 如 `gemma3:1b` 不可行， `qwen3:0.6b` 可行




## 上下文

最终交互 Prompt

**English**

~~~bash
# Role

You are an expert in software architecture and refactoring, specializing in fixing multi-file architecture anti-patterns.

# Task

Based on the provided anti-pattern information, dependency relations, impacted files, internal and external knowledge retrieval results, your task is to:

- Understand the given code issue and the associated anti-pattern;
- Generate a **multi-file refactoring solution** strictly based on the provided information and repair cases;
- Explicitly list all modified files and their corresponding changes.

# Inputs

- **Anti-pattern Type**: {anti_pattern_name}
- **Anti-pattern Definition**: {anti_pattern_definition}
- **External Knowledge RAG**: {external_rag_result}
- **Internal Knowledge RAG**: {internal_rag_result}
- **Impacted Files**: {impacted_files}
- **Target Code**: {code_snippet}

# Constraints

- Do not infer missing context; only use the provided information.
- Show complete refactored code for each file.
- The refactoring must be complete and correct, with high code quality; cross-file modifications are allowed if necessary.
- Use a structured format so that outputs can be easily parsed for correctness verification.

# Output Format

1. **Refactoring Strategy**
   - Briefly describe the refactoring approach (max 5 sentences).
   - Start this section with `### Refactoring Strategy`.
2. **Refactored Code and Modified Files**
   - For each modified file, output the following **JSON-like structure** to facilitate automated parsing:

```
{
  "file_name": "path/to/file",
  "modified_code": "```language\n<full file content>\n```",
  "modification_summary": "<short description of changes>"
}
```

- Start this section with `### Refactored Code and Modified Files`.
- List all modified files in an array under this section.
- Ensure the code block contains the **full content** of the file after refactoring.
~~~



**Chinese**

~~~bash
# 角色（Role）

你是一名软件架构与重构专家，擅长修复跨文件的架构反模式。

# 任务（Task）

基于提供的反模式信息、依赖关系、受影响文件范围，以及内部和外部知识检索结果，你需要：

- 理解给定的代码问题及其对应的反模式；
- 生成一个 **多文件重构解决方案**，严格基于提供的信息和修复案例；
- 明确列出所有修改的文件及对应的修改内容。

# 输入（Inputs）

- **反模式类型**: {anti_pattern_name}
- **反模式定义**: {anti_pattern_definition}
- **外部知识库检索结果**: {external_rag_result}
- **内部知识库检索结果**: {internal_rag_result}
- **受影响文件范围**: {impacted_files}
- **目标代码**: {code_snippet}

# 限制（Constraints）

- 不要推测缺失的上下文，只能使用提供的信息。
- 展示每个文件的完整重构代码。
- **修复必须完整且正确，代码质量要高**，必要时可以跨文件修改。
- 输出必须使用结构化格式，方便后续自动化解析和正确性验证。

# 输出格式（Output Format）

1. **重构策略**
   - 简要描述重构方法（不超过 5 句话）。
   - 本部分开头请使用 `### 重构策略`。
2. **重构后的代码与修改文件**
   - 对每个修改的文件，输出以下 **JSON 风格结构**，便于自动解析：

```
{
  "文件名": "路径/文件名",
  "修改后的代码": "```语言\n<文件完整内容>\n```",
  "修改说明": "<对本次修改的简短描述>"
}
```

- 本部分开头请使用 `### 重构后的代码与修改文件`。
- 将所有修改文件列在数组中，每个对象对应一个文件。
- `修改后的代码` 必须包含 **文件完整内容**，而不仅仅是差异。
~~~





