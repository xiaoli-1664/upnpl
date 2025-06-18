% generate_matrix_A.m
% 构造 Ax = 0，其中 A \in R^{120x120}，x 由 monomials * polynomials 构造
clear;
clc;

syms r1 r2 r3 u0 u1 u2 u3 real
syms A [10 10] real % 这会创建 m1_1, m1_2, ... m10_10 等符号变量并组成矩阵 m

%% Step 1: 定义向量 r 中所有 monomial（共10个）
r_vec = [r1^2;r2^2;r3^2;r1*r2;r1*r3;r2*r3;r1;r2;r3;1];

%% Step 2: 定义 f0, f1, f2, f3
f0 = u0 + u1*r1 + u2*r2 + u3*r3;

v1 = [2*r1;0;0;r2;r3;0;1;0;0;0];
v2 = [0;2*r2;0;r1;0;r3;0;1;0;0];
v3 = [0;0;2*r3;0;r1;r2;0;0;1;0];

f1 = expand(2 * r_vec.' * A * v1);
f2 = expand(2 * r_vec.' * A * v2);
f3 = expand(2 * r_vec.' * A * v3);

%% Step 3: 生成所有 total degree 最高为 7 的 monomials（预期120个）
% --- 修正开始 (Step 3) ---
monos = sym([]); % 初始化为符号列向量
for d_total = 0:7 % 总次数从 0 到 7
  for i = 0:d_total
    for j = 0:(d_total-i)
      k = d_total - i - j;
      monos(end+1,1) = r1^i * r2^j * r3^k;
    end
  end
end
monos = unique(monos);  % 移除重复项
if length(monos) ~= 120
    warning('生成的单项式数量不是120，请检查逻辑。当前数量: %d', length(monos));
end
% --- 修正结束 (Step 3) ---

%% Step 4: 构建 S0, S1, S2, S3（不重叠）
% --- 修正开始 (Step 4) ---
S3 = sym([]); S2 = sym([]); S1 = sym([]); S0 = sym([]);
for i = 1:length(monos)
    mn = monos(i);
    deg_r3 = feval(symengine, 'degree', mn, r3); % 获取 mn 关于 r3 的次数
    deg_r2 = feval(symengine, 'degree', mn, r2); % 获取 mn 关于 r2 的次数
    deg_r1 = feval(symengine, 'degree', mn, r1); % 获取 mn 关于 r1 的次数

    if deg_r3 >= 3
        S3(end+1) = mn; % S3 中的元素 r3 次数 >= 3
    elseif deg_r2 >= 3
        S2(end+1) = mn; % S2 中的元素 r2 次数 >= 3 (且不满足S3条件)
    elseif deg_r1 >= 3
        S1(end+1) = mn; % S1 中的元素 r1 次数 >= 3 (且不满足S3,S2条件)
    else
        S0(end+1) = mn; % 其他情况
    end
end
% 校验分组是否完整
if length(S0) + length(S1) + length(S2) + length(S3) ~= length(monos)
    warning('S0-S3 分组后的单项式总数与 monos 数量不符，请检查 Step 4 逻辑。');
end
% --- 修正结束 (Step 4) ---

%% Step 5: 构建 A 矩阵，并按 S0, S1, S2, S3 顺序拼接 monomial basis
all_monomials = [S0, S1, S2, S3].'; % S_i 是行向量，拼接后转置为列向量以匹配常见的基向量形式
                                  % 或者保持 all_monomials 为行向量，coeffs_to_row 中对应处理
                                  % 这里假设 all_monomials 是一个包含120个单项式的 (行或列) 向量
                                  % 如果 S_i(end+1) = mn; 使S_i为行向量,则 [S0,S1,S2,S3] 是行向量拼接
                                  % MATLAB 的 ismember 和 find 对于行向量和列向量基础上的元素查找行为一致

if length(all_monomials) ~= 120 && length(monos) == 120 % 确保all_monomials也是120，通常 S_i 为行，拼接后仍为行
    all_monomials = all_monomials.'; % 如果之前S_i是列向量，拼接后就是多行，就不需要转置
                                     % 鉴于 S_k(end+1)=mn 的用法, S_k 是行向量, all_monomials 也是行向量
                                     % coeffs_to_row 的 basis 是行向量也没问题
end
if length(all_monomials) ~= 120
     error('基础单项式 all_monomials 的数量不是120 (%d), 请检查Step3和Step4的逻辑', length(all_monomials));
end

M = sym(zeros(length(all_monomials))); % 矩阵A的大小根据实际all_monomials数量确定
row_idx = 1; % 使用不同的变量名避免与函数内的row冲突

for s_mono = S0 % 迭代S0中的每个单项式 (转置为列向量迭代，或直接 S0 如果S0是行向量)
    poly = expand(s_mono * f0);
    M(row_idx, :) = coeffs_to_row(poly, all_monomials,[r1,r2,r3]);
    row_idx = row_idx + 1;
end
for s_mono = S1
    poly = expand((s_mono / r1^3) * f1); % s_mono 必须能被 r1^3 整除 (Step 4的逻辑保证)
    M(row_idx, :) = coeffs_to_row(poly, all_monomials,[r1,r2,r3]);
    row_idx = row_idx + 1;
end
for s_mono = S2
    poly = expand((s_mono / r2^3) * f2);
    M(row_idx, :) = coeffs_to_row(poly, all_monomials,[r1,r2,r3]);
    row_idx = row_idx + 1;
end
for s_mono = S3
    poly = expand((s_mono / r3^3) * f3);
    M(row_idx, :) = coeffs_to_row(poly, all_monomials,[r1,r2,r3]);
    row_idx = row_idx + 1;
end

filename = 'M_sym_N3.txt';
fid = fopen(filename, 'w');

% 假设 M 是您的符号矩阵
% 例如:
% syms a b c d;
% M = [a+b, c-d; a*b, c/d];

for i = 1:size(M, 1)
    for j = 1:size(M, 2)
        str_original = char(M(i, j));   % 将符号转为字符串，例如得到 'a + b'
        str_no_spaces = strrep(str_original, ' ', ''); % 移除所有空格，得到 'a+b'
        fprintf(fid, '%s\t', str_no_spaces); % 使用 tab 分隔
    end
    fprintf(fid, '\n'); % 换行
end

fclose(fid);


%% Step 6: 转换函数（多项式按基底 monomials 展开为一行）
function row_vec = coeffs_to_row(poly, basis, x) % 函数名改为 row_vec 避免与外部变量 row 混淆
    row_vec = sym(zeros(1, length(basis)));
    % coeffs(poly) 会自动找到多项式poly中以 r1,r2,r3 等符号变量为变量的项
    % 其系数是其他符号变量 (如 u0, m1_1 等)
    [c, t] = coeffs(poly,x); % c 是系数向量, t 是对应的项 (单项式) 向量
    % 为了提高鲁棒性，确保basis和t中的单项式具有一致的表示形式以便比较
    % MATLAB的符号计算通常会自动处理，但复杂情况下可能需要手动 simplify
    for i = 1:length(t)
        % ismember 比较符号表达式时，它们需要是完全相同的形式
        idx = find(ismember(basis, t(i)), 1); % 找到第一个匹配项即可
        if ~isempty(idx)
            row_vec(idx) = row_vec(idx) + c(i); % 如果一个基项在poly中出现多次（通常expand后不会），则累加系数
        else
            % 如果 poly 中的某个项 t(i) 不在 basis 中，说明该项超出了我们关心的最高次数
            % 或者 basis 不完整。在此问题设定下，basis 是最高7次的，
            % 而 s*f0 可能产生最高8次的项，这些项将被忽略。
            % fprintf('警告: 项 %s (系数 %s) 不在基底中，已被忽略。\n', char(t(i)), char(c(i)));
        end
    end
end