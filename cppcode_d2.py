import re


def parse_line(line):
    # 解析行：例如 "27 4 4*A1_4+2*A4_1" 或 "8 8 u0"
    line = line.strip()
    if not line:
        return None
    parts = line.split()
    if len(parts) < 3:
        return None
    i, j, expr = int(parts[0]), int(parts[1]), " ".join(parts[2:])
    return i, j, expr


def convert_expression(expr):
    expr = expr.replace(' ', '')
    terms = expr.split('+')
    cpp_expr_parts = []
    for term in terms:
        # 例如：'4*A1_4'、'2*A4_1' 或 'u0'
        if 'a' in term:
            match = re.match(r'(\d*)\*?a(\d+)_(\d+)_(\d+)', term)
            if match:
                coeff, index, row, col = match.groups()
                row = int(row)
                col = int(col)
                row = row - 1  # 转换为0基索引
                col = col - 1
                index = int(index) - 1  # 转换为0基索引
                coeff = coeff if coeff else '1'
                cpp_expr_parts.append(f"{coeff}*a[{index}]({row},{col})")
        elif 'b' in term:
            match = re.match(r'(\d*)\*?b(\d+)_(\d+)', term)
            if match:
                coeff, index, col = match.groups()
                index = int(index) - 1
                col = int(col) - 1
                coeff = coeff if coeff else '1'
                cpp_expr_parts.append(f"{coeff}*b[{index}]({col})")
        elif 'c' in term:
            match = re.match(r'(\d*)\*?c(\d+)', term)
            if match:
                coeff, index = match.groups()
                index = int(index) - 1
                coeff = coeff if coeff else '1'
                cpp_expr_parts.append(f"{coeff}*c[{index}]")
        elif 'u' in term:
            match = re.match(r'u(\d)', term)
            if match:
                idx = match.group(1)
                cpp_expr_parts.append(f"u({idx})")
    return ' + '.join(cpp_expr_parts)


def generate_cpp_function(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    cpp_lines = [
        "#pragma once\n\n#include <vector>\n#include <Eigen/Dense>\n\nvoid constructM_N4(const std::vector<Eigen::MatrixXd> &a, const std::vector<Eigen::VectorXd> &b, const std::vector<double> &c, const Eigen::VectorXd& u, Eigen::MatrixXd& M) {"
    ]

    for line in lines:
        parsed = parse_line(line)
        if not parsed:
            continue
        i, j, expr = parsed
        cpp_expr = convert_expression(expr)
        cpp_lines.append(f"    M({i},{j}) = {cpp_expr};")

    cpp_lines.append("}")

    return '\n'.join(cpp_lines)


# 写入 C++ 文件
cpp_code = generate_cpp_function("nonzero_elements_output_N4.txt")
with open("include/constructM_N4.h", "w") as f:
    f.write(cpp_code)

print("C++ function written to constructM.cpp")
