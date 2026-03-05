import torch
import json
import os
from pathlib import Path
from tempp import getJSON
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import re
from PIL import Image


# 加载基础模型
base_model_path = "/app/input/res/your_program/large_model/Qwen2.5-VL-3B"  # 基础模型路径
print("加载基础模型...")
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="cuda"
)
processor = AutoProcessor.from_pretrained(base_model_path)
print("基础模型加载完成")

# 创建PeftModel并加载多个适配器
print("加载LoRA适配器...")
model = PeftModel.from_pretrained(
    base_model,
    "/app/input/res/your_program/large_model/my_lora_adapters/conn",  # 连接分析适配器路径
    adapter_name="conn_adapter"
)

model.load_adapter(
    "/app/input/res/your_program/large_model/my_lora_adapters/vqa",  # VQA适配器路径
    adapter_name="vqa_adapter"
)
print("LoRA适配器加载完成")


def resize_long_side_1024(image_path, max_size=1024):
    """
    将图片长边缩放到 max_size，短边按比例缩放，
    保存到当前目录下的 resize/ 子文件夹，返回新路径。
    """
    # 打开原图
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    long_side = max(w, h)

    # 无论当前尺寸如何，都进行缩放操作
    if w >= h:
        new_w = max_size
        new_h = int(h * max_size / w)
    else:
        new_h = max_size
        new_w = int(w * max_size / h)

    img = img.resize((new_w, new_h), Image.LANCZOS)

    current_dir = os.getcwd()
    resize_dir = os.path.join(current_dir, "resize")
    os.makedirs(resize_dir, exist_ok=True)

    filename = os.path.basename(image_path)
    resized_path = os.path.join(resize_dir, filename)

    img.save(resized_path, format="JPEG")

    return resized_path


def parse_assistant_response(assistant_response):
    """
    健壮地解析助手响应中的JSON数据
    """
    print("模型响应:", assistant_response)

    # 预处理：清理响应文本
    cleaned_response = assistant_response.strip()

    # 移除代码块标记
    if cleaned_response.startswith('```json'):
        cleaned_response = cleaned_response[7:]
    if cleaned_response.startswith('```'):
        cleaned_response = cleaned_response[3:]
    if cleaned_response.endswith('```'):
        cleaned_response = cleaned_response[:-3]
    cleaned_response = cleaned_response.strip()

    connections_dict = {}

    # 方法1：尝试直接解析
    try:
        connections_dict = json.loads(cleaned_response)
        print("直接解析成功")
        return connections_dict
    except json.JSONDecodeError:
        print("直接解析失败，尝试其他方法...")

    # 方法2：智能修复被截断的JSON
    try:
        fixed_json = smart_fix_truncated_json(cleaned_response)
        if fixed_json:
            connections_dict = json.loads(fixed_json)
            print("智能修复解析成功")
            return connections_dict
    except json.JSONDecodeError as e:
        print(f"智能修复解析失败: {e}")

    # 方法3：提取有效部分重构
    try:
        reconstructed_json = reconstruct_json_from_partial(cleaned_response)
        if reconstructed_json:
            connections_dict = json.loads(reconstructed_json)
            print("部分重构解析成功")
            return connections_dict
    except json.JSONDecodeError as e:
        print(f"部分重构解析失败: {e}")

    print("错误: 所有解析方法都失败")
    return None


def smart_fix_truncated_json(json_str):
    """
    智能修复被截断的JSON
    """
    # 移除可能的多余内容
    json_str = re.sub(r',\s*"\s*$', '', json_str)  # 移除末尾未完成的键
    json_str = re.sub(r',\s*\[\s*$', '', json_str)  # 移除末尾未完成的数组

    # 确保以 } 结束
    if not json_str.endswith('}'):
        # 查找最后一个完整的键值对
        last_brace = json_str.rfind('}')
        if last_brace != -1:
            json_str = json_str[:last_brace + 1]
        else:
            # 没有找到结束括号，尝试补全
            if json_str.count('{') > json_str.count('}'):
                json_str += '}'

    # 移除末尾可能未完成的键值对
    lines = json_str.split('\n')
    valid_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 检查是否是完整的键值对
        if ':' in line and '"' in line.split(':')[0]:
            key_part, value_part = line.split(':', 1)
            key_part = key_part.strip()

            # 确保键是完整的（有开闭引号）
            if key_part.startswith('"') and key_part.count('"') >= 2:
                valid_lines.append(line)

    if valid_lines:
        reconstructed = '{\n' + ',\n'.join(valid_lines) + '\n}'
        return reconstructed

    return None


def reconstruct_json_from_partial(partial_json):
    """
    从部分JSON内容重构完整的JSON
    """
    # 提取所有完整的键值对
    pattern = r'"([A-Z])"\s*:\s*\[([^]]*)\]'
    matches = re.findall(pattern, partial_json)

    if not matches:
        # 尝试另一种模式
        pattern = r'([A-Z])\s*:\s*\[([^]]*)\]'
        matches = re.findall(pattern, partial_json)

    if matches:
        reconstructed_lines = []
        for key, values in matches:
            # 处理数组值
            if values.strip():
                # 清理并格式化数组项
                items = []
                for item in values.split(','):
                    item = item.strip().strip('"\'')
                    if item:
                        items.append(f'"{item}"')
                value_str = f'[{", ".join(items)}]' if items else '[]'
            else:
                value_str = '[]'

            reconstructed_lines.append(f'"{key}": {value_str}')

        return '{' + ', '.join(reconstructed_lines) + '}'

    return None


def connection(image, outP):
    # 切换到连接分析适配器
    model.set_adapter("conn_adapter")

    marked_json_dir = "/app/input/res/your_program/marked_json"
    marked_image_dir = "/app/input/res/your_program/marked_images"
    image_filename = Path(image).stem
    json_filename = f"{image_filename}.json"
    image_filename = f"{image_filename}.jpg"
    json_file_path = os.path.join(marked_json_dir, json_filename)
    image_file_path = os.path.join(marked_image_dir, image_filename)
    print(image_file_path)

    with open(json_file_path, 'r', encoding='utf-8') as f:
        component_data = json.load(f)

    # 使用列表推导式获取所有Component值
    component_names = [
        comp.get("Component")
        for comp in component_data.get("task1", [])
        if comp.get("Component")
    ]

    component_list_str = json.dumps(component_names, ensure_ascii=False, separators=(',', ':'))
    print(component_list_str)

    prompt_verify = f"""【connections】**Role & Task:**
        You are a circuit diagram analyzer. Analyze the user-provided diagram and output ONLY a valid JSON object representing the direct outgoing connections of each labeled component.
        **COMPONENT LIST:**
        {component_list_str}
        **Visual Analysis Instructions:**
        1.Use ONLY the components listed above - do not identify or add any other components
        2.Identify all components marked with a red label (e.g., A, B, C) and a green border.
        3.For each of these components, trace every black wire or arrow that originates directly from it.
        4.Determine the immediate target component that the wire or arrow points to. Ignore any components the wire merely passes through.
        **JSON Output Rules:**
        1.Keys: The red labels of the identified components (e.g., "A", "B").
        2.Values: An array of red labels that the key component directly points to.
        3.If a component has no outgoing connections, use an empty array [].
        4.CRITICAL: Every labeled component in the diagram MUST have a key in the JSON.
        **Output Example:**
        {{"A":["D"],"B":["A"],"C":["D"],"D":["E"],"E":["B"]}}
        **IMPORTANT:**
        1.Your entire response must be the JSON object and nothing else.
        2.Do not describe, explain, or add any text outside the JSON.
        Now, The existing components in this diagram are {component_list_str}, analyze the image and output the JSON.
        """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_file_path},
                {"type": "text", "text": prompt_verify}
            ]
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to("cuda")

    # 使用当前激活的适配器进行生成
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    assistant_response = output_text[0].split("assistant\n")[-1]
    connections_dict = parse_assistant_response(assistant_response)

    if connections_dict is None:
        print("无法解析模型响应，使用默认值或抛出异常")

    # 安全过滤：移除不在已知组件列表中的键和值
    def safe_filter_connections(conn_dict, valid_components):
        """过滤连接字典，只保留有效的组件键和值"""
        filtered_dict = {}

        for key, value in conn_dict.items():
            # 只处理在有效组件列表中的键
            if key in valid_components:
                # 过滤值，只保留在有效组件列表中的连接目标
                if isinstance(value, list):
                    filtered_value = [item for item in value if item in valid_components]
                    filtered_dict[key] = filtered_value
                    print(f"过滤连接: {key} -> {value} -> {filtered_value}")
                else:
                    print(f"警告: 键 {key} 的值不是列表类型: {value}")
                    filtered_dict[key] = []
            else:
                print(f"跳过无效键: {key} (不在组件列表中)")

        # 确保所有有效组件都在过滤后的字典中
        for component in valid_components:
            if component not in filtered_dict:
                filtered_dict[component] = []
                print(f"添加缺失组件: {component} -> []")

        return filtered_dict

    # 应用安全过滤
    print("应用安全过滤...")
    filtered_connections = safe_filter_connections(connections_dict, component_names)
    print("过滤后的连接字典:", filtered_connections)

    # 确保输出目录存在
    os.makedirs(outP, exist_ok=True)

    # 更新组件数据中的连接信息
    updated_count = 0
    for component_data_item in component_data.get("task1", []):
        component_name = component_data_item["Component"]
        print(f"处理组件: {component_name}")

        if component_name in filtered_connections:
            output_connections = filtered_connections[component_name]
            print(f"组件 {component_name} 的输出连接: {output_connections}")

            # 更新连接信息
            component_data_item["Connection"]["output"] = output_connections
            component_data_item["I_O"]["output"] = len(output_connections)
            updated_count += 1
            print(f"已更新组件 {component_name} 的连接信息")
        else:
            print(f"警告: 组件 {component_name} 不在过滤后的连接字典中")
            # 设置为空数组
            component_data_item["Connection"]["output"] = []
            component_data_item["I_O"]["output"] = 0

    print(f"总共更新了 {updated_count} 个组件的连接信息")

    # 保存更新后的JSON文件
    output_file_path = os.path.join(outP, json_filename)

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(component_data, f, indent=2, ensure_ascii=False)

    print(f"成功生成更新后的JSON文件: {output_file_path}")

    # 验证保存的文件
    with open(output_file_path, 'r', encoding='utf-8') as f:
        saved_data = json.load(f)
        print("保存的文件内容验证:")
        for comp in saved_data.get("task1", []):
            print(f"组件 {comp['Component']}: 输出连接 = {comp['Connection']['output']}, 输出数量 = {comp['I_O']['output']}")

    return output_file_path


def vqa(image_path, task2, outPUT):
    # 切换到VQA适配器
    model.set_adapter("vqa_adapter")
    resized_image_path = resize_long_side_1024(image_path)
    print(resized_image_path)
    # 根据图片路径获取对应的JSON文件路径
    image_filename = os.path.basename(image_path)
    image_name = os.path.splitext(image_filename)[0]  # 去掉扩展名
    json_filename = f"{image_name}.json"

    # 假设JSON文件在task2_question_path目录中
    task2_question_path = task2  # 你需要设置正确的路径
    json_file_path = os.path.join(task2_question_path, json_filename)

    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 处理task2中的每个问题
    task2_questions = data.get("task2", [])
    results = []

    for question_data in task2_questions:
        question_type = question_data["type"]
        question_text = question_data["question"]

        if question_type == 'fill_in_the_blank':
            choice = '【填空题】'
            full_question = question_text
            # 调用模型获取答案
            prompt_verify = f"""{choice}你是一名顶级的模拟电路/控制理论专家。请仔细分析题目与图片信息,运用相关的电路理论和公式等相关知识，正确回答问题。只输出最终答案本身，不要任何解释。题目中包含提示（请从以下答案中选择一个：……），所以你最后输出的答案必须且只允许为括号内候选项之一，并与候选答案完全一致。不输出其他任何内容。\n题目："""

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": resized_image_path},
                        {"type": "text", "text": prompt_verify + full_question}
                    ]
                }
            ]
        else:
            choice = "【选择题】"
            # 构建完整的问题（包含选项）
            options_text = "\n".join(question_data["options"])
            full_question = f"{question_text}\n{options_text}"
            # 调用模型获取答案
            prompt_verify = f"""{choice}【选择题】你是一位顶级的模拟电路设计专家。请根据图片中给出的电路图和相关理论知识，仔细分析题目信息，并分析各选项的正确性，最后选择最正确的答案。请只输出正确答案的选项字母（A、B、C或D），并避免包含任何其他文字或解释。\n题目："""

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": resized_image_path},
                        {"type": "text", "text": prompt_verify + full_question}
                    ]
                }
            ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to("cuda")

        # 使用当前激活的适配器进行生成
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        assistant_response = output_text[0].split("assistant\n")[-1]
        print(f"问题: {full_question}")
        print(f"模型回答: {assistant_response}")

        # 提取答案（根据问题类型进行后处理）
        answer = assistant_response
        if (answer.startswith('(') and answer.endswith(')') and answer.count('(') == 1 and answer.count(')') == 1):
            answer = answer[1:-1]

        # 构建结果
        result_item = question_data.copy()
        result_item["answer"] = answer
        results.append(result_item)

    # 准备输出数据
    output_data = {"task2": results}

    # 写入到02_entry_template目录下的同名文件
    output_dir = outPUT
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, json_filename)

    # 读取已存在的文件内容（如果存在）
    existing_data = {}
    if os.path.exists(output_file_path):
        with open(output_file_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)

    # 将新的task2结果合并到现有数据中
    existing_data.update(output_data)

    # 写回文件
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)

    print(f"结果已保存到: {output_file_path}")

    return output_data


def getResult(image_path, task2_question_path, output_path):
    getJSON(image_path)
    all_items = os.listdir(image_path)
    print("当前文件夹下的所有内容:")
    image_dir = image_path  # 图片目录
    all_items = os.listdir(image_dir)

    print("当前文件夹下的所有内容:")
    for item in all_items:
        # 构建完整的图片路径
        image_ = os.path.join(image_dir, item)
        print(image_)
        connection(image_, output_path)
        vqa(image_, task2_question_path, output_path)


# # # 使用示例
# if __name__ == "__main__":
#     getResult('EDA_CASES_1024/images', 'EDA_CASES_1024/task2_questions', '02_entry_template')