def find_nonzero_elements(input_filepath, output_filepath):
    """
    找出稀疏矩阵中的非零元素，并将其值、行号和列号保存到输出文件。

    参数:
    input_filepath (str): 输入的稀疏矩阵txt文件的路径。
    output_filepath (str): 输出非零元素及其位置的txt文件的路径。
    """
    try:
        with open(input_filepath, 'r', encoding='utf-8') as infile, \
                open(output_filepath, 'w', encoding='utf-8') as outfile:

            for row_index, line in enumerate(infile):
                elements = line.strip().split()  # 假设元素以空格分隔
                for col_index, element in enumerate(elements):
                    if element != "0":  # 判断是否为非零元素
                        outfile.write(f"{row_index} {col_index} {element}\n")
        print(f"处理完成！非零元素已保存到: {output_filepath}")

    except FileNotFoundError:
        print(f"错误：输入文件 {input_filepath} 未找到。")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")


# --- 使用示例 ---
# 请将下面的文件名替换为您实际的输入和输出文件名
input_file = "M_sym_N2_d2.txt"  # 您的稀疏矩阵文件名
output_file = "nonzero_elements_output_N2_d2.txt"  # 您希望保存结果的文件名

# 调用函数进行处理
find_nonzero_elements(input_file, output_file)
