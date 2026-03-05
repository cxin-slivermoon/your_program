
# python entry.py --image_path /path/to/images --task2_question_path /path/to/task2_question  --output_path /path/to/output
#该脚本为用户提交脚本，需实现算法
# python entry.py --image_path EDA_CASES_1024/images --task2_question_path EDA_CASES_1024/task2_questions --output_path 02_entry_template

# python entry.py --image_path /path/to/images --task2_question_path /path/to/task2_question  --output_path /path/to/output
import argparse
import os
import json
from datetime import datetime
from temp import getResult


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="处理图像并生成结果JSON")
    parser.add_argument("--image_path", required=True, help="输入图像文件夹路径")
    parser.add_argument("--task2_question_path", required=True, help="task2问题文件路径")
    parser.add_argument("--output_path", required=True, help="输出结果根路径")
    args = parser.parse_args()

    # 1. 检查输入路径是否存在
    if not os.path.isdir(args.image_path):
        print(f"错误：输入图像文件夹不存在 -> {args.image_path}")
        return  # 路径错误时终止程序
    else:
        print("图像文件存在")
    
    if not os.path.exists(args.task2_question_path):
        print(f"错误：task2问题文件不存在 -> {args.task2_question_path}")
        return  # 路径错误时终止程序
    else:
        print("task2问题文件存在")
        

    # 2. 定义结果保存目录（固定为 02_entry_template）
    result_dir = args.output_path
    os.makedirs(result_dir, exist_ok=True)  # 确保文件夹存在
    print(f"结果将保存")

    # 主程序
    getResult(args.image_path, args.task2_question_path, args.output_path)
    
    
def show_current_dir(distinguish=False):
    """
    打印当前工作目录路径，并列出路径下所有文件和文件夹
    
    参数:
        distinguish: 布尔值，若为True则区分文件和文件夹，默认为False
    """
    # 获取当前工作目录
    current_path = os.getcwd()
    print(f"当前路径：{current_path}\n")
    
    # 获取当前路径下的所有项目
    all_items = os.listdir(current_path)
    
    if not distinguish:
        # 不区分类型，直接列出所有项目
        print("当前路径下的所有文件和文件夹：")
        for item in all_items:
            print(item)
    else:
        # 区分文件和文件夹
        print("当前路径下的内容（区分类型）：")
        for item in all_items:
            # 拼接完整路径
            item_full_path = os.path.join(current_path, item)
            if os.path.isfile(item_full_path):
                print(f"📄 文件：{item}")
            elif os.path.isdir(item_full_path):
                print(f"📁 文件夹：{item}")


if __name__ == "__main__":
    # show_current_dir()
    main()