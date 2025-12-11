import os
from collections import defaultdict

def find_duplicate_filenames(file_list):
    """
    从文件列表中查找重复的文件名
    """
    filename_count = defaultdict(list)
    
    for filepath in file_list:
        # 使用os.path.basename获取文件名（包含扩展名）
        filename = os.path.basename(filepath.strip())
        filename_count[filename].append(filepath.strip())
    
    # 找出重复的文件名
    duplicates = {filename: paths for filename, paths in filename_count.items() 
                 if len(paths) > 1}
    
    return duplicates

def process_txt_files(txt_files):
    """
    处理多个txt文件
    """
    all_filepaths = []
    
    # 读取所有txt文件中的第一列数据
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:  # 跳过空行
                        # 获取每行的第一列（逗号分隔）
                        filepath = line.split(',')[0].strip()
                        all_filepaths.append(filepath)
        except FileNotFoundError:
            print(f"警告: 文件 {txt_file} 不存在")
        except Exception as e:
            print(f"读取文件 {txt_file} 时出错: {e}")
    
    return all_filepaths

def main():
    # 这里替换为你的txt文件名
    txt_files = ['data_analysis/DOVER/examplar_data_labels/LSVQ/labels_test.txt', 'data_analysis/DOVER/examplar_data_labels/LSVQ/labels_1080p.txt']  # 根据实际情况修改
    
    # 处理所有txt文件
    all_files = process_txt_files(txt_files)
    
    print(f"总共找到 {len(all_files)} 个文件")
    
    # 查找重复的文件名
    duplicates = find_duplicate_filenames(all_files)
    
    if duplicates:
        print(f"\n发现 {len(duplicates)} 个重复的文件名:")
        print("=" * 50)
        
        for filename, paths in duplicates.items():
            print(f"\n文件名: {filename}")
            print(f"出现次数: {len(paths)}")
            print("在以下路径中出现:")
            for path in paths:
                print(f"  - {path}")
    else:
        print("\n没有发现重复的文件名")

if __name__ == "__main__":
    main()