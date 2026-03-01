import csv
import os
import argparse

def convert_tsv_to_csv(input_file, output_file):
    """
    将Tab分隔的LCQMC文件转换为逗号分隔的标准CSV文件
    :param input_file: 输入的tsv/txt文件路径
    :param output_file: 输出的csv文件路径
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：输入文件 {input_file} 不存在！")
        return
    
    # 定义表头
    headers = ["sentence1", "sentence2", "label"]
    
    try:
        # 读取Tab分隔文件并写入CSV
        with open(input_file, 'r', encoding='utf-8') as tsv_file, \
             open(output_file, 'w', encoding='utf-8', newline='') as csv_file:
            
            # 先写入表头（不加引号）
            csv_file.write("sentence1,sentence2,label\n")
            
            # 逐行读取并转换
            line_count = 0
            for line in tsv_file:
                # 去除首尾空白字符（换行、空格等）
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                
                # 按Tab分割字段（处理多个Tab的情况）
                parts = line.split('\t')
                # 确保只有3个字段（sentence1, sentence2, label）
                if len(parts) != 3:
                    print(f"警告：第 {line_count+1} 行格式异常，跳过 -> {line}")
                    continue
                
                # 手动构建CSV行：文本字段加双引号，标签字段不加
                sentence1 = parts[0].replace('"', '""')  # 转义双引号
                sentence2 = parts[1].replace('"', '""')  # 转义双引号
                label = parts[2]
                
                csv_line = f'"{sentence1}","{sentence2}",{label}\n'
                csv_file.write(csv_line)
                line_count += 1
        
        print(f"转换完成！")
        print(f"输入文件：{input_file}")
        print(f"输出文件：{output_file}")
        print(f"成功转换 {line_count} 条数据")
        
    except Exception as e:
        print(f"转换出错：{str(e)}")

# ------------------- 配置参数（修改这里）-------------------
# 默认输入文件路径（相对路径）
DEFAULT_INPUT_FILE = "data/train.txt"
# 默认输出CSV文件路径
DEFAULT_OUTPUT_FILE = "data/lcqmc_max.csv"
# ----------------------------------------------------------

# 执行转换
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将LCQMC TSV文件转换为CSV格式")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT_FILE, 
                       help=f"输入的TSV/TXT文件路径 (默认: {DEFAULT_INPUT_FILE})")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT_FILE,
                       help=f"输出的CSV文件路径 (默认: {DEFAULT_OUTPUT_FILE})")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    convert_tsv_to_csv(args.input, args.output)