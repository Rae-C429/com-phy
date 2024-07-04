這是我修計算機課程的作業，總共有 11 個作業以及 1 個期末專案。下面將說明程式碼的內容。

# HW1.1 NumPy 陣列操作

這個程式使用 NumPy 陣列的基本操作，包括陣列創建、重塑、計算每行總和和矩陣更新。

1. 導入 NumPy 模組:

```python
import numpy as np
```

2. 創建和重塑陣列 a:

```python
a = np.arange(1, 26, 1)
print(a)
print("\n")
a = np.reshape(a, (5, 5))
print(a)
print("\n")
```

`np.arange(1, 26, 1)` 創建包含從 1 到 25 的陣列 a。
`np.reshape(a, (5, 5))` 將 a 重新塑形為一個 5x5 的矩陣並打印出來。

3. 計算每行的總和:

```python
for i in range(0, 4):
    b = sum(a[i, :])
    print("ro", i, "sum=", b)
print("\n")
```

`for` 迴圈遍歷每一行（索引從 `0` 到 `3`）。
`sum(a[i, :])`計算第 i 行的總和並打印出來。

4. 更新矩陣:

```python
a[2] = a[0, :] + a[2, :]
print("new matrix=\n")
print(a)
```

`a[2] = a[0, :] + a[2, :]` 將第 0 行與第 2 行相加並將結果存儲在第 2 行。
最後打印更新後的矩陣`a`。

## 結果

這個程式展示了使用 NumPy 的基本陣列操作，包括創建陣列、重塑、計算每行總和和原地更新。

# HW1.2 Gaussian Elimination with Python and NumPy

![電路圖](/phto/Electric_circuit.jpg "Electric_circuit")
這段程式碼是用來解決一個線性電路的節點電壓問題，使用 Python 和 NumPy 來實現高斯消去法，並使用矩陣的逆來解線性方程組，計算各個支路的電流。

## 程式碼解釋

1. 導入必要的模組：

```python
import numpy as np
from numpy.linalg import inv
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

## 結果說明

這段程式碼首先使用高斯消去法進行消元，然後使用回代過程計算未知數的值，最後使用矩陣的逆來檢查解的正確性。結果顯示了不同步驟的中間值和最終解。

# HW2.2
