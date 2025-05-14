# backend/main.py
import os
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import AzureOpenAI
import json

load_dotenv()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
)

class Prompt(BaseModel):
    prompt: str

def fix_mermaid_code(code):
    # 1. Initial cleanup
    code = re.sub(r'^```mermaid\s*', '', code, flags=re.MULTILINE)
    code = re.sub(r'\s*```$', '', code, flags=re.MULTILINE)
    code = code.replace('\r\n', '\n').strip()

    # 2. Ensure graph declaration and add semicolon
    if not re.search(r'^\s*graph\s+(TD|TB|LR|RL|BT)', code, re.IGNORECASE | re.MULTILINE):
        code = 'graph TD;\n' + code
    else:
        code = re.sub(r'^(graph\s+(?:TD|TB|LR|RL|BT))(?!\s*;)', r'\1;', code, count=1, flags=re.IGNORECASE | re.MULTILINE)

    lines = code.split('\n')
    new_lines = []
    node_id_map = {} # To map problematic old IDs to new ones for 'class' statements

    for line_content in lines:
        stripped_line = line_content.strip()
        if not stripped_line:
            continue

        original_line_for_map_key = stripped_line

        # 3. Fix classDef: ensure semicolon, separate from trailing nodes
        if stripped_line.lower().startswith('classdef'):
            match = re.match(r'^(classDef\s+\w+\s+[^;({\[]+)(\s*[\(\[\{].*|%%.*)?$', stripped_line, re.IGNORECASE)
            cd_part = stripped_line
            node_part_after_cd = None
            if match:
                cd_part = match.group(1).strip()
                if match.group(2) and re.match(r'^\s*([a-zA-Z0-9_]+(\(.*\)|\[.*\]|\{.*\}).*|%%.*)', match.group(2).strip()):
                    node_part_after_cd = match.group(2).strip()
                else:
                    cd_part = stripped_line # Fallback if group(2) is part of style
            if not cd_part.endswith(';'):
                cd_part += ';'
            new_lines.append(cd_part)
            if node_part_after_cd:
                # If a node part was separated, it needs to be processed in a subsequent iteration or a recursive call.
                # For simplicity here, we assume it will be picked up if it forms a valid line.
                # A more robust solution might re-queue this part.
                lines.append(node_part_after_cd) # Add it back to be processed if it's a full node definition
            continue


        # 4. Fix problematic node definitions
        #    - (("开始")):::core  -> start_node(("开始")):::core (or similar valid ID)
        #    - buffer["("经验缓冲区")"]:::database -> buffer[("经验缓冲区")]:::database
        #    - compute_loss(""计算目标损失函数...") -> compute_loss("计算目标损失函数...")

        # Fix (("text")) node ID and shape (circle)
        # This regex matches node_id((text_in_double_parens))
        # or just ((text_in_double_paren_node_no_id))
        is_double_paren_node = False
        # Case 1: ID already exists, like my_id(("text"))
        match_id_double_paren_node = re.match(r'^([a-zA-Z0-9_]+)\s*\(\((.*?)\)\)(:::.*)?$', stripped_line)
        # Case 2: No ID, just (("text"))
        match_double_paren_node_no_id = re.match(r'^\(\((.*?)\)\)(:::.*)?$', stripped_line)

        if match_id_double_paren_node:
            node_id = match_id_double_paren_node.group(1)
            text_content = match_id_double_paren_node.group(2)
            style_class = match_id_double_paren_node.group(3) or ""
            is_double_paren_node = True
        elif match_double_paren_node_no_id:
            text_content = match_double_paren_node_no_id.group(1)
            style_class = match_double_paren_node_no_id.group(2) or ""
            # Generate an ID
            if "开始" in text_content: node_id = "start_node"
            elif "结束" in text_content: node_id = "end_node"
            else: node_id = "gen_circle_node_" + str(abs(hash(text_content))%1000) # Generic ID
            
            # For 'class' statement mapping, if the original was like (("text"))
            problematic_id_in_class_stmt = f'(({text_content}))'
            node_id_map[problematic_id_in_class_stmt] = node_id
            is_double_paren_node = True
        
        if is_double_paren_node:
            if not (text_content.startswith('"') and text_content.endswith('"')):
                text_content = f'"{text_content}"'
            stripped_line = f'{node_id}(({text_content})){style_class}'


        # Fix buffer["("经验缓冲区")"] to buffer[("经验缓冲区")] (database shape)
        match_bad_db_node = re.match(r'^([a-zA-Z0-9_]+)\s*\[\s*"\(L?\s*(.*?)\s*R?\)"\s*\](:::.*)?$', stripped_line)
        if match_bad_db_node:
            node_id = match_bad_db_node.group(1)
            text_content = match_bad_db_node.group(2)
            style_class = match_bad_db_node.group(3) or ""
            if not (text_content.startswith('"') and text_content.endswith('"')):
                text_content = f'"{text_content}"'
            stripped_line = f'{node_id}[({text_content})]{style_class}'


        # Fix compute_loss(""text"") to compute_loss("text")
        match_double_quotes_label = re.match(r'^([a-zA-Z0-9_]+)\s*([\(\[\{])\s*""(.*?)""\s*([\)\]\}])(:::.*)?$', stripped_line)
        if match_double_quotes_label:
            node_id = match_double_quotes_label.group(1)
            opener = match_double_quotes_label.group(2)
            text_content = match_double_quotes_label.group(3)
            closer = match_double_quotes_label.group(4)
            style_class = match_double_quotes_label.group(5) or ""
            stripped_line = f'{node_id}{opener}"{text_content}"{closer}{style_class}'

        # 5. General quoting for node labels (if not already quoted)
        def ensure_quotes_for_node_text_generic(match_obj):
            node_id_part = match_obj.group(1) 
            text_content = match_obj.group(2) 
            closer_and_style = match_obj.group(3)
            if not ((text_content.startswith('"') and text_content.endswith('"')) or \
                    (text_content.startswith("'") and text_content.endswith("'"))):
                if re.search(r'[\s\W]', text_content, re.UNICODE) and not text_content.startswith("(") and not text_content.endswith(")"): # Avoid double quoting for shapes like id[("text")]
                     # More specific check for CJK or space
                    if any('\u4e00' <= char <= '\u9fff' for char in text_content) or ' ' in text_content :
                        text_content = f'"{text_content}"'
            return f"{node_id_part}{text_content}{closer_and_style}"

        # Apply generic quoting only if not a special structure line
        is_link = "-->" in stripped_line or "-.->" in stripped_line or "---" in stripped_line or "==>" in stripped_line
        is_control_flow = stripped_line.lower().startswith(("subgraph", "direction", "end", "class "))

        if not is_link and not is_control_flow:
            # id["text"], id("text"), id{"text"}
            stripped_line = re.sub(r'^([a-zA-Z0-9_]+\s*\[)(.*?)(\].*)$', ensure_quotes_for_node_text_generic, stripped_line)
            stripped_line = re.sub(r'^([a-zA-Z0-9_]+\s*\()(.*?)(\).*)$', ensure_quotes_for_node_text_generic, stripped_line)
            stripped_line = re.sub(r'^([a-zA-Z0-9_]+\s*\{)(.*?)(\}.*)$', ensure_quotes_for_node_text_generic, stripped_line)
            # id[("text")] special case for database type nodes
            stripped_line = re.sub(r'^([a-zA-Z0-9_]+\s*\[\s*\(\s*)(.*?)(\s*\)\s*\]:::.*)$', ensure_quotes_for_node_text_generic, stripped_line)
            stripped_line = re.sub(r'^([a-zA-Z0-9_]+\s*\[\s*\(\s*)(.*?)(\s*\)\s*\])$', ensure_quotes_for_node_text_generic, stripped_line)


        # 6. Fix subgraph definitions (make sure title is quoted if spaces)
        if stripped_line.lower().startswith('subgraph'):
            match_subgraph = re.match(r'^subgraph\s+(.*)', stripped_line, re.IGNORECASE)
            if match_subgraph:
                title_candidate = match_subgraph.group(1).strip()
                # Avoid quoting if it's already quoted or is a direction keyword
                if not title_candidate.lower().startswith("direction") and \
                   ' ' in title_candidate and not (title_candidate.startswith('"') and title_candidate.endswith('"')):
                    title_candidate = f'"{title_candidate}"'
                stripped_line = f'subgraph {title_candidate}'
        
        new_lines.append(stripped_line)

    # 7. Post-process 'class' statements using the node_id_map
    final_code_lines_for_class_fix = []
    for line in new_lines:
        if line.lower().startswith("class "):
            # First replace any problematic IDs
            for problematic_id, new_id in node_id_map.items():
                escaped_problematic_id = re.escape(problematic_id)
                line = re.sub(escaped_problematic_id, new_id, line)
            
            # Then fix comma-space patterns in class statements
            # Match pattern: "class node1, node2, node3 style;" and convert to "class node1,node2,node3 style;"
            match = re.match(r'^(class\s+)(.*?)(\s+\w+;?)$', line)
            if match:
                prefix = match.group(1)
                node_list = match.group(2)
                suffix = match.group(3)
                # Remove spaces between commas in node list
                fixed_node_list = re.sub(r',\s+', ',', node_list)
                line = f"{prefix}{fixed_node_list}{suffix}"
            
            # Ensure class statement ends with a semicolon
            if not line.strip().endswith(';'):
                line = line.strip() + ';'
            
            # If the class statement doesn't have a style identifier, treat it as a special case
            # Pattern: "class node1,node2,node3" without style or semicolon
            if not re.search(r'class\s+.*?\s+\w+;?$', line):
                line = re.sub(r'^(class\s+)(.*?)$', r'\1\2;', line)
        
        final_code_lines_for_class_fix.append(line)
    
    return '\n'.join(final_code_lines_for_class_fix)


@app.post("/generate")
async def generate_mermaid(data: Prompt):
    print("=== Backend received Prompt ===")
    print(data.prompt)
    if not data.prompt.strip():
        raise HTTPException(400, "prompt cannot be empty")

    # THIS IS THE SYSTEM MESSAGE CONTENT - ENSURE NO COMMAS BETWEEN STRING LITERALS
    system_message_content = (
        "你是一位专业的系统架构图和流程图设计师，专门使用 Mermaid.js 语法生成具有清晰结构和层级关系的图表。\n"
        "你的任务是根据用户需求，输出美观、信息丰富、易于理解的 Mermaid 代码。\n\n"
        "请严格遵循以下规范：\n\n"
        "1.  **只输出 Mermaid.js 代码**，本体部分不要任何注释、解释、```标记。\n"
        "2.  图表声明从 `graph TD;` (或 LR/BT/RL) 开始，并以分号结尾。\n"
        "3.  **使用 `subgraph`**: 每个 `subgraph` 标题如果包含空格，必须用引号包裹 (e.g., `subgraph \"用户模块\"`)。下一行应为 `direction TD` (或 LR/BT/RL)。\n"
        "4.  **使用 `classDef`**: 所有 `classDef` 语句必须以分号结尾 (e.g., `classDef core fill:#cde4ff;`)。\n"
        "    * `classDef core fill:#cde4ff,stroke:#5A96E3,stroke-width:2px,color:#333,font-weight:bold;`\n"
        "    * `classDef service fill:#e6f2ff,stroke:#007bff,stroke-width:1.5px,color:#333;`\n"
        "    * `classDef database fill:#fff2cc,stroke:#FFB300,stroke-width:1.5px;`\n"
        "    * `classDef external fill:#e0e0e0,stroke:#757575,stroke-width:1.5px,font-style:italic;`\n"
        "    * `classDef ui fill:#d4edda,stroke:#28a745,stroke-width:1.5px;`\n"
        "5.  **节点定义**: \n"
        "    * **节点ID**：使用简单字母数字下划线组合 (e.g., `user_db`, `login_process`)。**避免在ID中使用括号、引号等特殊字符。**\n"
        "    * **节点形状与文本**：\n"
        "        * `node_id[\"矩形文本\"]`\n"
        "        * `node_id(\"圆角文本\")`\n"
        "        * `node_id{\"菱形文本\"}`\n"
        "        * `node_id[\\(\"数据库/存储\"\\)]` (注意是 `[( )]` for database shape with text)\n"
        "        * `node_id((\"圆形文本\"))` (注意是 `(( ))` for circle shape with text)\n"
        "    * **所有节点内的文本标签**，特别是包含中文、空格或特殊字符时，**必须用英文双引号包裹**。例如 `my_node[\"你好 世界\"]`。\n"
        "    * 应用样式：`my_node_id[\"文本\"]:::core`\n\n"
        "6.  **连接线**: `A -- \"可选标签\" --> B`。标签文本必须用英文双引号包裹。\n"
        "7.  **代码结构**: `graph` -> `classDef`s -> nodes & subgraphs -> connections -> `class` applications.\n"
        "8.  **每个定义必须单独一行**：`graph`声明、每个`classDef`、每个节点定义、每个连接、每个`subgraph`、每个`end`、每个`direction`都应在单独的一行。\n"
        "    * **严禁将节点定义与 `classDef` 写在同一行。**\n\n"
        "请根据这些规范生成代码。"
    )  # END OF SYSTEM MESSAGE CONTENT TUPLE


    messages = [
        {"role": "system", "content": system_message_content},
        {"role": "user", "content": data.prompt}
    ]

    try:
        print("=== Sending request to Azure OpenAI ===")
        resp = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=messages,
        )
        print("-----------------------------------------")
        print("=== Full OpenAI API Response Object (dict) ===")
        try:
            print(json.dumps(resp.model_dump(), indent=2, ensure_ascii=False)) 
        except Exception as e_dump:
            print(f"Could not dump response to JSON: {e_dump}")
            print(resp) 
        print("-----------------------------------------")

        code = "" 
        if resp.choices:
            if resp.choices[0].message:
                code = resp.choices[0].message.content
                if code is None: 
                    code = ""
        
        print("=== Backend raw returned code (from AI content field) ===")
        print(f"'{code}'") # Print with quotes to see if it's truly empty or just whitespace
        print("-----------------------------------------")
        
        code_from_ai = str(code).strip() if code is not None else ""

        if code_from_ai:
            fixed_code = fix_mermaid_code(code_from_ai)
        else:
            # Provide a default valid mermaid graph if AI returns nothing
            fixed_code = 'graph TD;\n  EmptyResponse["AI未返回内容"]:::core;' 
        
        print(f"=== Code after fix_mermaid_code (being returned to frontend) ===")
        print(f"'{fixed_code}'")
        print("-----------------------------------------")
        return {"mermaid": fixed_code}

    except Exception as e:
        print(f"=== Backend exception ===\n{type(e).__name__}: {e}")
        error_message = str(e)
        if hasattr(e, 'response') and e.response and hasattr(e.response, 'json'):
            try:
                error_detail = e.response.json()
                if 'error' in error_detail and 'message' in error_detail['error']:
                    error_message = f"OpenAI API Error: {error_detail['error']['message']}"
            except:
                pass 
        elif hasattr(e, 'message') and isinstance(e.message, str): 
            error_message = e.message
        elif hasattr(e, '_message') and isinstance(e._message, str): 
            error_message = e._message

        raise HTTPException(500, f"Generation failed: {error_message}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
