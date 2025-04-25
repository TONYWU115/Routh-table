import numpy as np

def routh_table(coeffs):
    n = len(coeffs)
    m = (n + 1) // 2

    table = np.zeros((n, m))

    # 填入前兩列
    table[0, :len(coeffs[::2])] = coeffs[::2]  # 第一列：奇數係數
    table[1, :len(coeffs[1::2])] = coeffs[1::2]  # 第二列：偶數係數

    # 開始填表
    for i in range(2, n):
        for j in range(m - 1):
            a = table[i-2, 0]
            b = table[i-2, j+1]
            c = table[i-1, 0]
            d = table[i-1, j+1]

            if c == 0:
                c = 1e-6  # 避免除以零

            table[i, j] = ((c * b) - (a * d)) / c

        # 如果整列為0，則使用輔助列（特殊情況處理）
        if np.allclose(table[i], 0):
            print(f"Row {i} is zero row, using auxiliary polynomial")
            order = n - i
            prev_row = table[i-1]
            aux_poly = []
            for k in range(len(prev_row)-1):
                aux_poly.append((order - 2*k) * prev_row[k])
            aux_row = np.zeros_like(prev_row)
            aux_row[:len(aux_poly)] = aux_poly
            table[i] = aux_row

    return table

def check_stability(routh):
    first_col = routh[:, 0]
    sign_changes = np.sum(np.diff(np.sign(first_col)) != 0)
    if np.any(np.isnan(first_col)) or np.any(np.isinf(first_col)):
        return "Routh table has NaN or Inf: likely unstable or improper coefficients"
    if np.any(first_col == 0):
        return "Some elements in the first column are zero: marginal stability or special case"
    return f"Sign changes in first column: {sign_changes} → {'Stable' if sign_changes == 0 else 'Unstable'}"

# 範例用法
coefficients = [1, 2, 3, 4]  # 輸入你的特徵方程係數，如 s^3 + 2s^2 + 3s + 4
table = routh_table(coefficients)
print("Routh Table:")
print(table)
print(check_stability(table))
