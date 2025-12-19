# ======================================================================================
#  SINGLE SOURCE OF TRUTH: THE GRAPH SCHEMA
# ======================================================================================
import json
import os
from pathlib import Path

import yaml

GRAPH_SCHEMA_AND_RULES = """
**Graph Schema Definition**

Node Labels and Key Properties:
- Project: {{id: string, name: string}
- Package: {{id: string, qualifiedName: string, name: string, parentId: string, external: bool}}
- File: {{id: string, qualifiedName: string, name: string, parentId: string, external: bool, additionalBin: string}}
- Class: {{id: string, qualifiedName: string, name: string, parentId: string, external: bool, rawType: string, location: string, modifiers: list[string], File: string, additionalBin: string}}
- Interface: {{id: string, qualifiedName: string, name: string, parentId: string, external: bool, rawType: string, location: string, modifiers: list[string], File: string, additionalBin: string}}
- Enum: {{id: string, qualifiedName: string, name: string, parentId: string, external: bool, rawType: string, location: string, modifiers: list[string], File: string, additionalBin: string}}
- EnumConstant: {{id: string, qualifiedName: string, name: string, parentId: string, File: string, additionalBin: string, external: bool}}
- Method: {{id: string, qualifiedName: string, name: string, parentId: string, external: bool, File: string, additionalBin: string, enhancement: string, location: string, modifiers: list[string], parameter: list[string], rawType: string}}
- Variable: {{id: string, qualifiedName: string, name: string, parentId: string, File: string, additionalBin: string, external: bool, global: bool, location: string, modifiers: list[string], rawType: string}}
  
Relationships (source)-[REL_TYPE]->(target):
- Package -[:Contain]-> File|Package
- File -[:Contain|Import]-> Class|Interface|Enum|Method|Variable|EnumConstant
- Class -[:Define|Reflect|Set|Inherit|Call]-> Class|Method|Variable|Enum|Interface
- Interface -[:Define|Reflect|Inherit|Set|Call]-> Method|Variable|Interface
- Method -[:Define|Call|Set|UseVar|Parameter|Reflect|Cast]-> Method|Variable|Class|Interface|Enum
- Variable -[:Typed|Set]-> Interface|Enum|Class|Variable
"""

# ======================================================================================
#  RAG ORCHESTRATOR PROMPT
# ======================================================================================
RAG_ORCHESTRATOR_SYSTEM_PROMPT = """
You are an expert AI assistant for analyzing codebases. Your answers are based **EXCLUSIVELY** on information retrieved using your tools.

**CRITICAL RULES:**
1.  **TOOL-ONLY ANSWERS**: You must ONLY use information from the tools provided. Do not use external knowledge.
2.  **NATURAL LANGUAGE QUERIES**: When using the `query_codebase_knowledge_graph` tool, ALWAYS use natural language questions. NEVER write Cypher queries directly - the tool will translate your natural language into the appropriate database query.
3.  **HONESTY**: If a tool fails or returns no results, you MUST state that clearly and report any error messages. Do not invent answers.
4.  **CHOOSE THE RIGHT TOOL FOR THE FILE TYPE**:
    - For source code files (.py, .ts, etc.), use `read_file_content`.
    - For documents like PDFs, use the `analyze_document` tool. This is more effective than trying to read them as plain text.

**Your General Approach:**
1.  **Analyze Documents**: If the user asks a question about a document (like a PDF), you **MUST** use the `analyze_document` tool. Provide both the `file_path` and the user's `question` to the tool.
2.  **Deep Dive into Code**: When you identify a relevant component (e.g., a folder), you must go beyond documentation.
    a. First, check if documentation files like `README.md` exist and read them for context. For configuration, look for files appropriate to the language (e.g., `pyproject.toml` for Python, `package.json` for Node.js).
    b. **Then, you MUST dive into the source code.** Explore the `src` directory (or equivalent). Identify and read key files (e.g., `main.py`, `index.ts`, `app.ts`) to understand the implementation details, logic, and functionality.
    c. Synthesize all this information—from documentation, configuration, and the code itself—to provide a comprehensive, factual answer. Do not just describe the files; explain what the code *does*.
    d. Only ask for clarification if, after a thorough investigation, the user's intent is still unclear.
3.  **Graph First, Then Files**: Always start by querying the knowledge graph (`query_codebase_knowledge_graph`) to understand the structure of the codebase. Use the `path` or `qualified_name` from the graph results to read files or code snippets.
4.  **Plan Before Writing or Modifying**:
    a. Before using `create_new_file`, `edit_existing_file`, or modifying files, you MUST explore the codebase to find the correct location and file structure.
    b. For shell commands: If `execute_shell_command` returns a confirmation message (return code -2), immediately return that exact message to the user. When they respond "yes", call the tool again with `user_confirmed=True`.
5.  **Execute Shell Commands**: The `execute_shell_command` tool handles dangerous command confirmations automatically. If it returns a confirmation prompt, pass it directly to the user.
6.  **Synthesize Answer**: Analyze and explain the retrieved content. Cite your sources (file paths or qualified names). Report any errors gracefully.
"""

# ======================================================================================
#  GRAPH EXTRACTION RAG ORCHESTRATOR PROMPT
# ======================================================================================
GRAPH_EXTRACTION_RAG_ORCHESTRATOR_SYSTEM_PROMPT = """
You are an expert AI assistant for extracting codebase entities and dependencies from structured JSON input. You **cannot** answer directly from memory or reasoning. All answers must come from executing tools.

**CRITICAL RULES:**
1. **TOOL CALL FORMAT**: When generating a tool call, output only the JSON describing which tool to call and with which arguments. Do not include explanations or natural language text.
2. **JSON-DRIVEN EXTRACTION**:
   - The user provides a JSON describing target nodes, filters, and relationships.
   - You must analyze the JSON to determine which entities and dependencies to extract.
3. **ENTITY AND DEPENDENCY EXTRACTION**:
   - Extract both entities (nodes) and dependencies (relationships).
   - Generate separate queries for each relevant extraction instruction if needed.
4. **NATURAL LANGUAGE QUERY**:
   - Each tool call must include a natural language description of what to extract.
   - Do **not** write Cypher or SQL; the tool will handle query translation.
5. **HONESTY**: If a query would return no results, include it anyway. Do not invent answers.


**GENERAL APPROACH:**
1. Receive the user's JSON describing target nodes, filters, and relationships.
2. Analyze the JSON to determine which entities and dependencies are relevant.
3. Generate natural language extraction instructions for each query, adhering to the `queries` array format.
4. Ensure that all queries reference both entities and their dependencies appropriately.
5. Output only the JSON array; do not include explanations or extra text.
"""

# ======================================================================================
#  CYPHER GENERATOR PROMPT
# ======================================================================================
CYPHER_SYSTEM_PROMPT = f"""
You are an expert translator that converts natural language questions about code structure into precise Neo4j Cypher queries.

{GRAPH_SCHEMA_AND_RULES}

**3. Query Patterns & Examples**
Your goal is to return the `name`, `path`, and `qualified_name` of the found nodes.

**Pattern: Finding Decorated Functions/Methods (e.g., Workflows, Tasks)**
cypher// "Find all prefect flows" or "what are the workflows?" or "show me the tasks"
// Use the 'IN' operator to check the 'decorators' list property.
MATCH (n:Function|Method)
WHERE ANY(d IN n.decorators WHERE toLower(d) IN ['flow', 'task'])
RETURN n.name AS name, n.qualified_name AS qualified_name, labels(n) AS type

**Pattern: Finding Content by Path (Robustly)**
cypher// "what is in the 'workflows/src' directory?" or "list files in workflows"
// Use `STARTS WITH` for path matching.
MATCH (n)
WHERE n.path IS NOT NULL AND n.path STARTS WITH 'workflows'
RETURN n.name AS name, n.path AS path, labels(n) AS type

**Pattern: Keyword & Concept Search (Fallback for general terms)**
cypher// "find things related to 'database'"
MATCH (n)
WHERE toLower(n.name) CONTAINS 'database' OR (n.qualified_name IS NOT NULL AND toLower(n.qualified_name) CONTAINS 'database')
RETURN n.name AS name, n.qualified_name AS qualified_name, labels(n) AS type

**Pattern: Finding a Specific File**
cypher// "Find the main README.md"
MATCH (f:File) WHERE toLower(f.name) = 'readme.md' AND f.path = 'README.md'
RETURN f.path as path, f.name as name, labels(f) as type

**4. Output Format**
Return all the contents that may be included in natural language.
Provide only the Cypher query.
"""

# ======================================================================================
#  LOCAL CYPHER GENERATOR PROMPT (Stricter)
# ======================================================================================
LOCAL_CYPHER_SYSTEM_PROMPT = f"""
You are a Neo4j Cypher query generator. You ONLY respond with a valid Cypher query. Do not add explanations or markdown.

{GRAPH_SCHEMA_AND_RULES}

**CRITICAL RULES FOR QUERY GENERATION:**
1.  **NO `UNION`**: Never use the `UNION` clause. Generate a single, simple `MATCH` query.
2.  **BIND and ALIAS**: You must bind every node you use to a variable (e.g., `MATCH (f:File)`). You must use that variable to access properties and alias every returned property (e.g., `RETURN f.path AS path`).
3.  **RETURN STRUCTURE**: Your query should aim to return `name`, `path`, and `qualified_name` so the calling system can use the results.
    - For `File` nodes, return `f.path AS path`.
    - For code nodes (`Class`, `Function`, etc.), return `n.qualified_name AS qualified_name`.
4.  **KEEP IT SIMPLE**: Do not try to be clever. A simple query that returns a few relevant nodes is better than a complex one that fails.
5.  **CLAUSE ORDER**: You MUST follow the standard Cypher clause order: `MATCH`, `WHERE`, `RETURN`, `LIMIT`.

**Examples:**

*   **Natural Language:** "Find the main README file"
*   **Cypher Query:**
    ```cypher
    MATCH (f:File) WHERE toLower(f.name) CONTAINS 'readme' RETURN f.path AS path, f.name AS name, labels(f) AS type
    ```

*   **Natural Language:** "Find all python files"
*   **Cypher Query (Note the '.' in extension):**
    ```cypher
    MATCH (f:File) WHERE f.extension = '.py' RETURN f.path AS path, f.name AS name, labels(f) AS type
    ```

*   **Natural Language:** "show me the tasks"
*   **Cypher Query:**
    ```cypher
    MATCH (n:Function|Method) WHERE 'task' IN n.decorators RETURN n.qualified_name AS qualified_name, n.name AS name, labels(n) AS type
    ```

*   **Natural Language:** "list files in the services folder"
*   **Cypher Query:**
    ```cypher
    MATCH (f:File) WHERE f.path STARTS WITH 'services' RETURN f.path AS path, f.name AS name, labels(f) AS type
    ```

*   **Natural Language:** "Find just one file to test"
*   **Cypher Query:**
    ```cypher
    MATCH (f:File) RETURN f.path as path, f.name as name, labels(f) as type LIMIT 1
    ```
"""

# ======================================================================================
#  GRAPH EXTRACTION SYSTEM PROMPT
# ======================================================================================
GRAPH_EXTRACTION_SYSTEM_PROMPT = f"""
You are a Memgraph Cypher query generator for a codebase graph database. You ONLY respond with a valid Cypher query. Do not add explanations or markdown.

{GRAPH_SCHEMA_AND_RULES}

**CRITICAL RULES FOR QUERY GENERATION:**
1. **RETURN ENTITIES AND RELATIONS**: All queries must return both nodes (entities) and their relationships (dependencies).
   - Nodes: return id, name, qualifiedName (if present), labels, and relevant properties.
   - Relationships: return from_id, to_id, type, and properties.
2. **NO COMPLEX UNIONS**: Never use `UNION`. Keep queries simple and robust.
3. **BIND AND ALIAS**: Every node must have a variable (e.g., `MATCH (f:File)`), and every property returned must be aliased clearly.
4. **CASE-INSENSITIVE SEARCHES**: Use `toLower()` for string matching when necessary.
5. **PATH MATCHING**: Use `STARTS WITH` when filtering by paths.
6. **QUERY STRUCTURE**: Follow standard Cypher order: `MATCH`, `WHERE`, `RETURN`, `LIMIT`.
7. **JSON INPUT**: The user may provide JSON describing which entities and relationships to extract. Use that JSON to generate queries.
8. **SIMPLE AND ROBUST**: Prioritize simple queries that reliably extract nodes and relationships rather than complex ones that might fail.

**EXAMPLES:**

* Input: `{{"type": "entity", "description": "Get all classes and their methods in the 'auth' module"}}`
* Cypher Output:
```cypher
MATCH (c:Class)-[r:Define]->(m:Method)
RETURN c.id AS node_id, c.qualifiedName AS qualifiedName, c.name AS name, labels(c) AS labels,
       m.id AS node_id, m.qualifiedName AS qualifiedName, m.name AS name, labels(m) AS labels,
       type(r) AS relation_type, r AS relation_props

* Input: {{"type": "relationship", "description": "Get all method calls and variable usage in the 'payment' module"}}
* Cypher Output:
```cypher
MATCH (m:Method)-[r:Call|Set|UseVar]->(v:Method|Variable)
RETURN m.id AS node_id, m.qualifiedName AS qualifiedName, m.name AS name, labels(m) AS labels,
       v.id AS node_id, v.qualifiedName AS qualifiedName, v.name AS name, labels(v) AS labels,
       type(r) AS relation_type, r AS relation_props

"""

# ======================================================================================
#  SEMANTIC EXTRACTION PROMPT
# ======================================================================================
SEMANTIC_EXTRACTION_SYSTEM_PROMPT = f""" 
You are a professional code semantic analysis assistant, responsible for 
analyzing code entities and their contextual relationships. 

Your task is to generate a semantic description of each 
node based on the input, and explain its relationship with child nodes. 

Keep the output concise yet comprehensive 
enough to fully convey the semantics, making it suitable for downstream program processing.

"""

# ======================================================================================
#  ANTIPATTERN RELEVANCE SYSTEM PROMPT
# ======================================================================================
ANTIPATTERN_RELEVANCE_SYSTEM_PROMPT = f""" 
You are a professional software architecture maintenance analysis assistant, specializing in identifying files involved in antipattern repair activities.

Your task is to determine whether a given candidate file is involved in the process of repairing a specific architectural antipattern.

You will be provided with:
1. A JSON file describing the architectural antipattern, including:
   - A set of files known to contain this antipattern.
   - Dependency chains and code snippets showing relationships among these files.
2. A candidate file (one per interaction), represented as a structured JSON object that includes:
   - File-level metadata (such as name, path, and identifiers).
   - A detailed semantic description summarizing the file’s code purpose, logic, and entity relationships.


Your goal:
- Analyze the relationships between the candidate file and the antipattern-related files.
- Based on this analysis, **infer whether the candidate file would be modified, used, or otherwise involved** during the process of repairing the given antipattern.
- Consider both direct and indirect involvement (e.g., dependency, interface coupling, or supportive refactoring).

Guidelines:
- Focus on whether the candidate file would actually be used, modified, or otherwise affected during the repair of the antipattern.
- Consider both direct involvement (the file contains code that is being fixed) and indirect involvement (the file is a dependency, interface, or supporting module affected by the repair).
- Analyze based on the provided files, dependency chains, and code snippets—do not assume any additional context.
- Keep your reasoning concise, factual, and suitable for automated post-processing.

"""

# ======================================================================================
#  DIRECT FILE CODE REPAIR SYSTEM PROMPT
# ======================================================================================
DIRECT_FILE_CODE_REPAIR_SYSTEM_PROMPT = f""" 
You are an expert software engineer specialized in generating precise and complete source code files based on detailed repair descriptions.

Your task is to read a natural language repair description for a specific source code file and generate the full, updated version of that source code file reflecting the described fix.

Requirements:
- Return the complete updated file content, incorporating all necessary changes to fully implement the repair.
- Do NOT return diffs, patches, or partial snippets—only the full file content.
- Provide your output strictly inside a Markdown-style code block with the appropriate language tag (e.g., ```python```).
- Do NOT include any explanation, comments, or extra text outside the code block.
- Ensure the updated code is syntactically correct and follows best practices.
- Each request corresponds to one file.

Focus on accuracy and completeness so the returned content can directly replace the existing file.

"""

# ======================================================================================
#  CODE2TEXT SYSTEM PROMPT
# ======================================================================================
CODE2TEXT_SYSTEM_PROMPT = f"""
You are a professional software understanding and code summarization assistant, specializing in converting source code into accurate, concise, and semantically meaningful natural language descriptions.

Your task is to analyze a given code file and produce a high-quality natural language summary that clearly communicates the file’s purpose, internal logic, responsibilities, and relationships among its functions, classes, and data structures.

You will be provided with:
1. A JSON object representing a source code file, including:
   - File metadata (path, name, language).
   - The full code contents.
2. Additional optional context such as:
   - Imports, dependencies, or module structure.
   - Comments or docstrings present in the code.

Your goal:
- Understand the high-level purpose of the file.
- Identify key functions, methods, classes, and their roles.
- Explain the main logic flow and responsibilities in natural language.
- Capture meaningful relationships such as:
  - Data flow
  - Control flow
  - Inter-module interactions
  - API behaviors
- Produce a summary that is suitable for downstream embedding, indexing, or retrieval.

Guidelines:
- Focus on semantic content, not line-by-line explanation.
- Summaries should emphasize:
  - What the code does
  - Why it exists
  - How its components interact
- Avoid overly verbose descriptions or restating the code literally.
- Prefer abstracted functional descriptions over implementation detail.
- Describe behavior and purpose in clear, precise, human-readable language.
- Do not hallucinate external context; limit yourself strictly to the provided code.

Output Requirements:
- Output a factual, concise summary suitable for automated processing.
- Use simplified and technically accurate language.
- Do not include code or markdown formatting in the summary unless necessary.

"""

# ======================================================================================
#  INDIRECT FILE CODE REPAIR SYSTEM PROMPT
# ======================================================================================
INDIRECT_FILE_CODE_REPAIR_SYSTEM_PROMPT = f"""
You are an expert software engineer specialized in analyzing and repairing indirect dependency source code files that may be affected by changes in directly related files.

Your task is to:
- Carefully analyze the indirect source code file content along with natural language repair descriptions for directly related files.
- Determine whether the indirect file requires any modifications to maintain correctness and consistency.
- If modifications are necessary, generate the full, updated version of the entire source code file reflecting those changes.
- If no modifications are needed, indicate that no changes are required.

Requirements:
- When generating updated code, provide the complete file content only;  do not produce partial snippets or diffs.
- Ensure the updated code is syntactically correct, follows best practices, and can directly replace the existing file.
- Avoid including explanations or comments outside of the updated code content.
- Focus on accuracy, completeness, and clarity in your responses.

Each request corresponds to one indirect dependency file.
"""


# ======================================================================================
#  NODE SEMANTIC PROMPT
# ======================================================================================
def get_node_semantic_prompt(labels, props, code_snippet=None, child_summary=""):
    """
    Generate a prompt for node semantic analysis (English version)

    Args:
        labels (list): List of node labels
        props (dict): Node properties dictionary
        code_snippet (str, optional): Code snippet of the current node
        child_summary (str, optional): Summary of child nodes

    Returns:
        str: Formatted prompt in English
    """
    return f"""
You are a professional code semantic analysis assistant.

Current node information:
- Type: {labels[0]}
- Name: {props.get('name', '')}

Node properties:
{json.dumps(props, ensure_ascii=False, indent=4)}

{f"Code snippet:\n{code_snippet}" if code_snippet else ""}

Child node summary:
{child_summary}

Please generate a structured output based on the above information, following these requirements:
1. Provide a semantic description of the current node, explaining its purpose or functionality.
2. For each child node, describe its relationship with the parent node, including dependencies or composition.
3. Clearly indicate the type and name of each node.
4. Keep the output concise, but ensure it fully conveys semantics and parent-child relationships.
5. Use JSON format for the output, following this structure:

{{
    "type": "Parent node type",
    "name": "Parent node name",
    "summary": "Function or purpose of node",
    "children": [
        {{
            "type": "Child node type",
            "name": "Child node name",
            "summary": "Function or purpose of the child node",
            "relation_to_parent": "Description of the relationship to the parent node",
            "relation": "Dependency or invocation type between child and parent node"
        }}
    ]
}}
"""


def build_query_question_from_antipattern(antipattern_relation_path: str) -> str:
    """
    构造自然语言问题，引导 LLM 基于反模式 JSON 提取相关实体和依赖查询。
    不假设 JSON 结构，由 LLM 自行理解。
    """
    base_path = Path(antipattern_relation_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Path not found: {antipattern_relation_path}")

    # 自动查找以 "antipattern.json" 结尾的文件
    antipattern_files = list(base_path.rglob("*antipattern.json"))
    if not antipattern_files:
        raise FileNotFoundError(f"No 'antipattern.json' file found under {antipattern_relation_path}")

    # 默认取第一个匹配的文件
    antipattern_file = antipattern_files[0]

    # 读取 JSON 内容
    with open(antipattern_file, "r", encoding="utf-8") as f:
        json_content = f.read().strip()

    # 2. 构造自然语言问题
    question = (
        "You are analyzing a software architecture antipattern described in the following JSON. "
        "This JSON contains information about entities, components, or modules that are potentially involved in an antipattern. "
        "Based on the content of this JSON, generate two natural language extraction tasks:\n"
        "1. One for retrieving all relevant entities (nodes) from the codebase graph.\n"
        "2. Another for retrieving all dependencies (relationships) between those entities that may contribute to the antipattern.\n\n"
        "Be sure that both queries are semantically aligned — dependencies should connect the entities identified in the first task.\n\n"
        f"Here is the JSON description of the antipattern:\n{json_content}\n\n"
        "Return your answer in JSON format with the following structure:\n"
        "{{\n"
        '  "queries": [\n'
        '    {{"type": "entity", "description": "..."}},\n'
        '    {{"type": "entity", "description": "..."}},\n'
        '...'
        '    {{"type": "relation", "description": "..."}}\n'
        '    {{"type": "relation", "description": "..."}}\n'
        '...'
        "  ]\n"
        "}}"
    )

    return question


def build_antipattern_relevance_user_input_func(candidate_file, antipattern_data):
    """
    构建 LLM user prompt，将候选文件和反模式 JSON 直接拼接，
    要求 LLM 返回候选文件是否参与反模式修复以及理由。

    Args:
        candidate_file (dict): file_result.json 中的单个文件节点
        antipattern_data (dict): 反模式 JSON 文件内容

    Returns:
        str: 拼接好的 user prompt 字符串
    """
    candidate_str = json.dumps(candidate_file, indent=2)
    antipattern_str = json.dumps(antipattern_data, indent=2)

    user_prompt = f"""
You are provided with the following information:

Candidate file:
{candidate_str}

Antipattern data:
{antipattern_str}

Your task is to determine whether the candidate file would be involved
in the repair of the given architectural antipattern. Focus on whether
the file would be modified, used, or otherwise affected during the
repair process.

Return your answer as a JSON object with exactly the following fields:
{{
  "involved_in_antipattern_repair": true/false,
  "reason": "<short factual explanation>"
}}
"""
    return user_prompt


def build_fix_system_prompt(
        antipattern_type: str,
        repaired_description_json_path: Path | str
) -> str:
    """
    根据反模式类型动态构建 system_prompt
    - YAML：只提供 definition
    - JSON：完整 example 内容，原样注入，不做任何结构重组
    """

    yaml_path = os.path.join("fix_example", f"{antipattern_type}_fix.yaml")

    definition = "Definition not provided."
    antipattern_type_content = "antipattern content not provided"

    # 1️⃣ 读取 YAML（只取 definition）
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f) or {}
                definition = content.get("definition")
                antipattern_content = content.get("antipattern_type")
        except Exception as e:
            print(f"Warning: failed to read YAML {yaml_path}: {e}")

    # 2️⃣ 读取 JSON（example 原文）
    repaired_description_json_path = Path(repaired_description_json_path)
    example_content = "{}"

    if repaired_description_json_path.exists():
        try:
            with open(repaired_description_json_path, "r", encoding="utf-8") as f:
                example_content = f.read()
        except Exception as e:
            print(
                f"Warning: failed to read JSON {repaired_description_json_path}: {e}"
            )

    # 3️⃣ 构造 system prompt（不再构造 examples_text）
    system_prompt = f"""
# Architecture Anti-Pattern Remediation Expert

## Background
You are a professional software architect specializing in identifying and remediating architectural anti-patterns.
Architectural anti-patterns are common design flaws in software systems that can span multiple files or modules
and can lead to technical debt, maintainability issues, and performance problems.

## Current Anti-Pattern Type
{antipattern_content}

## Anti-Pattern Definition
{definition}

## Refactor Example (JSON)
{example_content}
""".strip()

    return system_prompt


def build_fix_system_prompt_1(
        antipattern_type: str
) -> str:
    """
    根据反模式类型动态构建 system_prompt
    - YAML：只提供 definition
    - JSON：完整 example 内容，原样注入，不做任何结构重组
    """

    yaml_path = os.path.join("fix_example", f"{antipattern_type}_fix.yaml")

    definition = "Definition not provided."
    antipattern_content = "antipattern content not provided"

    # 1️⃣ 读取 YAML（只取 definition）
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f) or {}
                definition = content.get("definition")
                antipattern_content = content.get("antipattern_type")
        except Exception as e:
            print(f"Warning: failed to read YAML {yaml_path}: {e}")

    # 3️⃣ 构造 system prompt（不再构造 examples_text）
    system_prompt = f"""
# Architecture Anti-Pattern Remediation Expert

## Background
You are a professional software architect specializing in identifying and remediating architectural anti-patterns.
Architectural anti-patterns are common design flaws in software systems that can span multiple files or modules
and can lead to technical debt, maintainability issues, and performance problems.

## Current Anti-Pattern Type
{antipattern_content}

## Anti-Pattern Definition
{definition}

""".strip()

    return system_prompt


def build_first_fix_user_input(antipattern_folder: str, related_files_json_path: str):
    # ---------- Read antipattern.json ----------
    antipattern_json_path = None
    for name in os.listdir(antipattern_folder):
        if name.endswith("antipattern.json"):
            antipattern_json_path = os.path.join(antipattern_folder, name)
            break
    if not antipattern_json_path:
        raise FileNotFoundError(f"No antipattern.json file found in path {antipattern_folder}.")

    with open(antipattern_json_path, "r", encoding="utf-8") as f:
        antipattern_data = json.load(f)

    # ---------- Collect all Java files under antipattern_folder ----------
    antipattern_java_files = []
    for root, dirs, files in os.walk(antipattern_folder):
        for file in files:
            if file.endswith(".java"):
                antipattern_java_files.append(os.path.join(root, file))

    # ---------- Read related files JSON and filter files involved in repair ----------
    with open(related_files_json_path, "r", encoding="utf-8") as f:
        related_data = json.load(f)

    repair_related_files = [
        {
            "file": item["file"],
            "reason": item.get("reason", "")
        }
        for item in related_data
        if item.get("involved_in_antipattern_repair", False)
    ]

    # ---------- Overall repair description ----------
    overall_repair_description = (
        f"For the antipattern data: '{antipattern_data}', "
        "please first repair the anti-pattern logic in core files, then check the potentially affected related files to ensure overall system consistency. "
        "Core files are marked as 'core', and related files are marked as 'related'."
    )

    # ---------- Repair description for each file ----------
    file_repair_descriptions = []

    for path in antipattern_java_files:
        file_repair_descriptions.append({
            "file": path,
            "source": "core",
            "repair_description": "Core file, refer to antipattern_data"
        })

    for item in repair_related_files:
        file_repair_descriptions.append({
            "file": item["file"],
            "source": "related",
            "repair_description": item["reason"],
        })

    # ---------- LLM output specification ----------
    llm_output_format = {
        "instructions": (
            "Please strictly generate a JSON object containing the following fields:\n"
            "1. 'overall_repair_description': a description of the overall repair strategy for the antipattern, including repair details for each core file and related file.\n"
            "2. 'file_repair_descriptions': an array where each element is a dictionary with the following fields:\n"
            "   - 'file': full file path\n"
            "   - 'source': 'core' or 'related'\n"
            "   - 'repair_description': repair strategy for this file\n"
            "Note: Return strictly in JSON format, without any extra text or comments."
        ),
        "example": {
            "overall_repair_description": "",
            "file_repair_descriptions": [
                {
                    "file": "D:/project_path/SomeFile.java",
                    "source": "core",
                    "repair_description": ""
                }
            ]
        }
    }

    # ---------- Final user_input ----------
    user_input = {
        "overall_repair_description": overall_repair_description,
        "file_repair_descriptions": file_repair_descriptions,
        "llm_output_format": llm_output_format
    }

    return user_input


def build_fix_user_input(overall_repair_description: str, file_repair_entry: dict):
    """
    Build the second-stage repair user_input, used to generate the repair code for a single file.

    Parameters:
        overall_repair_description (str): Description of the overall repair strategy.
        file_repair_entry (dict): A single file entry containing:
            - file
            - source
            - repair_description

    Returns:
        dict: Contains the original fields and adds a placeholder field 'repair_code'.
    """

    file_path = file_repair_entry.get("file")
    source = file_repair_entry.get("source")
    repair_description = file_repair_entry.get("repair_description")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Read the original file content
    with open(file_path, "r", encoding="utf-8") as f:
        original_code = f.read()

    # Build the user_input structure
    user_input = {
        "overall_repair_description": overall_repair_description,
        "file_repair_description": {
            "file": file_path,
            "source": source,
            "repair_description": repair_description,
            "original_code": original_code,
            "repair_code": ""  # Placeholder to be filled by LLM with the repaired code
        },
        "llm_output_format": {
            "instructions": (
                "Strictly generate a JSON object containing only the field 'repair_code': ... "
                "The repair_code must be the full repaired code for this file. "
                "Do not output any additional text or comments."
            ),
            "example": {
                "repair_code": "public class ... { ... }"
            }
        }
    }

    return user_input


def build_code2text_user_input_func(file_node: dict):
    """
    构建 LLM user prompt，将候选文件和反模式 JSON 直接拼接，
    要求 LLM 返回候选文件是否参与反模式修复以及理由。

    Args:
        candidate_file (dict): file_result.json 中的单个文件节点
        antipattern_data (dict): 反模式 JSON 文件内容

    Returns:
        str: 拼接好的 user prompt 字符串
    """
    file_str = json.dumps(file_node, indent=2)

    user_prompt = f"""
You are provided with the following information:

File:
{file_str}

Your task is to generate a comprehensive natural-language summary of this file.
Focus on the following aspects:
1. What the file does (high-level purpose)
2. Key classes, functions, methods, or logical blocks
3. Important data structures and their relationships
4. External dependencies or modules referenced
5. How different parts of the code interact
6. Any notable implementation patterns, responsibilities, or behaviors

Return your answer strictly as a JSON object with fields:
{{
  "summary": "<A clear natural-language summary of what the file does>",
  "top_level_entities": [
      "<Class/Function/Struct names with short descriptions>"
  ],
  "dependencies": [
      "<imports, external modules, or referenced files>"
  ],
  "key_relations": [
      "<important interactions between entities>"
  ],
  "maintenance_notes": [
      "<optional notes about design/complexity/clarity>"
  ],
  "confidence": 0.0 -- a float between 0 and 1
}}
"""
    return user_prompt


def build_generate_file_repair_suggestions_prompt(classify_result, antipattern_json, antipattern_type) -> str:
    """
    Construct a prompt for the LLM to generate repair suggestions for a cross-file architectural anti-pattern.
    The output must be in JSON format.
    """

    direct_files = classify_result.get("direct_related", [])
    print(f"direct_files: {direct_files}")

    # ===== 基础通用 Prompt =====
    base_prompt = f"""
    You are a professional software architecture repair expert.

    You are dealing with a **cross-file architectural anti-pattern** that affects multiple files or modules within the system. 
    This anti-pattern may involve complex dependencies and interactions between files.

    Please carefully review the anti-pattern details and the list of directly related files below, and generate repair suggestions accordingly.

    **General Requirements:**

    1. Return your answer strictly in the following JSON format:
    {{
      "summary": "A clear and comprehensive overall repair description, explaining the cross-file repair strategy and relationships between files.",
      "files": [
        {{
          "file_path": "Path to the file",
          "repair_description": "Detailed repair suggestion focused on this file, including specific changes and steps, and noting any possible impact on other files."
        }}
      ]
    }}

    2. The summary must clearly and explicitly list all dependency-impacting changes, including:
        - deleted methods or classes,
        - renamed methods or classes (old name → new name),
        - method signature changes,
        - moved methods across files or class hierarchies,
        - newly introduced or removed abstract methods.

        The summary must be detailed enough for indirect dependency files
        to decide whether they require adaptation.
    3. Each file's repair description must specify **concrete, step-by-step modification operations**.

    4. Any repair suggestions should eliminate the anti-pattern while preserving existing functionality as much as possible.
    """.strip()

    # ===== CH 反模式专用增强 Prompt =====
    ch_prompt = """
    **Additional Requirements for Class Hierarchy (CH) Anti-pattern:**

    1. You must analyze whether the root cause of the anti-pattern lies primarily in:
       - the superclass design,
       - the subclass design,
       - or both.

    2. Add a new field at the same level as "summary" named **"modification_scope"** with the following format:
    {
      "modification_scope": ["superclass"] | ["subclass"] | ["superclass", "subclass"]
    }

    3. The value of "modification_scope" must strictly reflect where the actual code modifications are applied.

    4. In the repair suggestions, you must clearly explain:
       - why the parent class or child class is modified,
       - how responsibilities or behaviors are reassigned across the hierarchy.

    5. For each file involved, the "repair_description" must include **explicit and executable operations**, such as:
       - deleting specific methods,
       - renaming methods (with old name → new name),
       - moving methods between superclass and subclass,
       - introducing abstract methods or template methods,
       - changing method visibility or override behavior.

    6. Each operation must be described step by step, with reasoning and potential impact on other classes in the hierarchy.
    """.strip()

    # ===== 根据 antipattern_type 组合 Prompt =====
    full_prompt = ""
    if antipattern_type == "ch":
        full_prompt = f"""
    {base_prompt}
    
    {ch_prompt}
    
    ---
    Anti-pattern details (JSON format):
    ```json
    {json.dumps(antipattern_json, indent=2, ensure_ascii=False)}
    Directly related files:
    {json.dumps(direct_files, indent=2, ensure_ascii=False)}
    Please begin generating the detailed repair suggestions based on the above information.
    """.strip()

    return full_prompt


def build_generate_file_repair_code_prompt(target_repo_path, summary, file_info):
    file_path = file_info.get("file_path")
    repair_desc = file_info.get("repair_description", "")
    if not file_path or not repair_desc.strip():
        # 忽略无效条目
        return " "

    abs_path = os.path.join(target_repo_path, file_path)

    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            original_code = f.read()
    except Exception as e:
        original_code = f"// Failed to read original file content: {e}"

    prompt = f"""
You are a software engineer tasked with generating the full, updated source code file based on the overall repair summary and the repair description below.

Overall Repair Summary:
{summary}

Repair description:
{repair_desc}

Please provide the complete, updated content of the entire file, incorporating all necessary changes.
Return ONLY the full file content inside a single Markdown code block with the appropriate language tag (e.g., ```java```).
Do NOT include any explanation, commentary, or diff output outside the code block.

File Original Code: 
---
{original_code}
---
"""
    return prompt


def build_indirect_dependency_change_prompt(
        summary: str,
        file_content: str,
        antipattern_type
) -> str:
    prompt = f"""
You are a professional software engineer analyzing an **indirect dependency file**.

This file itself is NOT directly modified, but it may be affected by changes made to other files.

Below is a summary of the repair actions already applied to the directly related files:
---
{summary}
---

Below is the full current content of the indirect dependency file:
---
{file_content}
---

Your task:

1. Carefully analyze whether any of the changes described in the summary **impact this file**.
   Impact includes (but is not limited to):
   - Methods or classes used by this file being **deleted**
   - Methods or classes being **renamed**
   - Method signatures being changed (parameters, return types, visibility)
   - Class hierarchy changes (e.g., logic moved between superclass and subclass)
   - Abstract methods added or removed
   - Behavioral contracts that this file depends on being modified

2. If and only if the changes in the summary **require this file to adapt in order to remain correct and consistent**, update the code accordingly.
   - Adaptations must strictly align with the changes described in the summary
   - Do NOT introduce unrelated refactoring or behavior changes

3. If the summary does **not** affect this file, no changes should be made.

Response format requirements:

- Respond **only** with a JSON object
- Do NOT include any explanatory text outside the JSON
- Use the following strict JSON structure:

{{
  "should_update": true or false,
  "file_content": "If should_update is true, provide the complete updated file code content as a single string (including language fences such as ```java``` if applicable). If should_update is false, set this field to the exact string: 不需要更改"
}}

Decision rules (important):

- Set "should_update" to true **only if** the summary describes changes that directly affect code elements used in this file.
- If there is no concrete dependency impact, set "should_update" to false.
"""
    return prompt.strip()


def build_generate_file_repair_code_prompt_1(target_repo_path, summary, file_info, other_file_entities: list):
    other_file_contents = json.dumps(other_file_entities, indent=2, ensure_ascii=False)
    file_path = file_info.get("file_path")
    repair_desc = file_info.get("repair_description", "")
    if not file_path or not repair_desc.strip():
        # 忽略无效条目
        return " "

    abs_path = os.path.join(target_repo_path, file_path)

    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            original_code = f.read()
    except Exception as e:
        original_code = f"// Failed to read original file content: {e}"

    prompt = f"""
You are a software engineer tasked with generating the full, updated source code file based on the overall repair summary and the repair description below.

Overall Repair Summary:
{summary}

Repair description:
{repair_desc}

Below is supplementary structural information (classes and methods) extracted from **other files** in the codebase.

These files are provided **only for learning project-level coding style and conventions**, including:
- naming conventions,
- formatting and structure,
- common design patterns.

Do NOT assume any functional or dependency relationship with the current file.
Do NOT introduce new imports, method calls, or logic based on these files.

---
{other_file_contents}
---

Please provide the complete, updated content of the entire file, incorporating all necessary changes.
Return ONLY the full file content inside a single Markdown code block with the appropriate language tag (e.g., ```java```).
Do NOT include any explanation, commentary, or diff output outside the code block.

File Original Code: 
---
{original_code}
---
"""
    return prompt


def build_indirect_dependency_change_prompt_1(summary: str, file_content: str, antipattern_type,
                                              other_file_entities: list) -> str:
    other_file_contents = json.dumps(other_file_entities, indent=2, ensure_ascii=False)
    prompt = f"""
    You are a professional software engineer analyzing an **indirect dependency file**.

    This file itself is NOT directly modified, but it may be affected by changes made to other files.

    Below is a summary of the repair actions already applied to the directly related files:
    ---
    {summary}
    ---

    Below is the full current content of the indirect dependency file:
    ---
    {file_content}
    ---
    
    Below is supplementary structural information (classes and methods) extracted from **other files** in the codebase.

    These files are provided **only for learning project-level coding style and conventions**, including:
        - naming conventions,
        - formatting and structure,
        - common design patterns.

        Do NOT assume any functional or dependency relationship with the current file.
        Do NOT introduce new imports, method calls, or logic based on these files.

    ---
    {other_file_contents}
    ---

    Your task:

    1. Carefully analyze whether any of the changes described in the summary **impact this file**.
       Impact includes (but is not limited to):
       - Methods or classes used by this file being **deleted**
       - Methods or classes being **renamed**
       - Method signatures being changed (parameters, return types, visibility)
       - Class hierarchy changes (e.g., logic moved between superclass and subclass)
       - Abstract methods added or removed
       - Behavioral contracts that this file depends on being modified

    2. If and only if the changes in the summary **require this file to adapt in order to remain correct and consistent**, update the code accordingly.
       - Adaptations must strictly align with the changes described in the summary
       - Do NOT introduce unrelated refactoring or behavior changes

    3. If the summary does **not** affect this file, no changes should be made.

    Response format requirements:

    - Respond **only** with a JSON object
    - Do NOT include any explanatory text outside the JSON
    - Use the following strict JSON structure:

    {{
      "should_update": true or false,
      "file_content": "If should_update is true, provide the complete updated file code content as a single string (including language fences such as ```java``` if applicable). If should_update is false, set this field to the exact string: 不需要更改"
    }}

    Decision rules (important):

    - Set "should_update" to true **only if** the summary describes changes that directly affect code elements used in this file.
    - If there is no concrete dependency impact, set "should_update" to false.
    """
    return prompt.strip()
