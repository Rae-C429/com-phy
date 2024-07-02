這是我修計算機課程的作業，總共有 11 個作業以及 1 個期末專案。下面將說明程式碼的內容。

# HW1.1.py

這個程式展示了如何使用 NumPy 來創建陣列、重塑陣列並對其進行操作。

1. 創建陣列：

```python
a = np.arange(1, 26, 1)
```

這行程式碼創建了一個從 1 到 25 的陣列。

2. 重塑陣列：

```python
a = np.reshape(a, (5, 5))
```

將陣列 a 重塑成 5x5 的矩陣。

3. 選擇特定元素：

```python
select1 = a[0, 1]
```

選擇矩陣中第一行第二個元素。

4. 選擇特定行：

```python
select2 = a[:, 1]
選擇矩陣中所有行的第二列。
```

5. 選擇特定區域：

```python
select3 = a[1:3, 1:3]
```

選擇矩陣中從第二行第二列到第三行第三列的區域。

6. 修改特定區域：

```python
a[0:2, 0:2] = 0
```

將矩陣中第一行第一列到第二行第二列的區域修改為 0。

# HW1.2 Gaussian Elimination with Python and NumPy

這個程式展示了如何使用 Python 和 NumPy 來實現高斯消去法，並使用矩陣的逆來解線性方程組。這裡我們以一個特定的 6x6 系統為例。

## 程式碼解釋

1. 導入必要的模組：

```python
import numpy as np
from numpy.linalg import inv
創建矩陣 A 和向量 b：
```

2. 創建矩陣 `A` 和向量 `b`：

```python
A = np.array([[-1, 1, -1, 0, 0, 0],  #A
                    [4, 2, 0, 0, 0, 0],   #top
                    [1, -1, 0, 1, 0, 0],  #B
                    [0, 0, 1, 0, -1, 1],  #c
                    [0, 2, 0, 0, 4, 0],   #middel
                    [0, 0, 0, 0, 4, 5],   #bottom
                    [0, 0, 0, -1, 1, -1]])#D

b = np.array([[0],  #A
              [8],  #top
              [0],  #B
              [0],  #C
              [0],  #middel
              [10], #bottom
              [0]]) #D
```

3. 定義矩陣的大小：

```python
n = 6
```

4. 高斯消去法 (Gaussian Elimination)：

消元過程：

```python
for i in range(0, n-1):
for j in range(i + 1, n+1):
if A[j, i] != 0.0:
lam = A[j, i] / A[i, i]
A[j, i : n] = A[j, i : n] - lam _ A[i, i : n]
b[j] = b[j] - lam _ b[i]
```

對每一列，將其下方列的元素消去。

5. 打印中間結果：

```python
print("Gaussian Elimination")
print("=========================")
print("orignal")
print("-------------------------")
print(" A =\n",A)
print("b =\n", b)
print("=========================")
print("modify")
print("-------------------------")
A = A[0 : n,0 : n]
b = b[0 : n]
print("A =\n",A)
print("b =\n", b)
```

6. 回代過程 (Back Substitution)：

初始化解向量：

```python
x = np.zeros((6, 1))
```

7. 計算最後一個變量：

```python
x[n-1] = b[n-1] / A[n-1, n-1]
```

8. 從倒數第二行開始反向計算其餘變量：

```python
for i in range(n-2, -1, -1):
x[i] = (b[i] - np.dot(A[i, i + 1 : n], x[i + 1 : n])) / A[i, i]
print("x =\n",x)
```

9. 使用矩陣的逆來解方程組：

計算矩陣的逆並解方程：

```python
print("=========================")
print("inverse")
x = np.dot(inv(A), b)
print("x =\n",x)
```

##結果說明
這段程式碼首先使用高斯消去法進行消元，然後使用回代過程計算未知數的值，最後使用矩陣的逆來檢查解的正確性。結果顯示了不同步驟的中間值和最終解。
