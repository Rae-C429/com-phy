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

## 結果

這個程式展示了使用 NumPy 的基本陣列操作，包括創建陣列、重塑、計算每行總和和原地更新。

# HW1.2 Gaussian Elimination with Python and NumPy

![電路圖](photo/Electric_circuit.JPG)
這段程式碼是用來解決一個線性電路的節點電壓問題，使用 Python 和 NumPy 來實現高斯消去法，並使用矩陣的逆來解線性方程組，計算各個支路的電流。

## 程式碼解釋

1. 導入模組：

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

這段程式碼演示了如何使用 Python 中的 NumPy 和 Matplotlib 庫來模擬和分析一個動態系統的行為。讓我們來分析它的具體內容：

## 程式碼解析

1. 導入模組

```python
import numpy as np
import matplotlib.pyplot as plt
```

這裡導入了 NumPy 用於數值計算和 Matplotlib 用於繪圖的模組。

2. 設定參數

```python
N = 51
h = 10 / (N-1)
```

定義了數據點數量 N 和步長 h，用於數值微分。

3. 生成時間或空間點集

```python
x = np.zeros(51)
a = 0
for i in range (51):
    x[i] = a
    a += 0.2
print("x", x)
```

這裡生成了一個包含 51 個等間距點的數組 x，用來表示時間或空間中的點。

4. 系統動態描述

```python
v = 1 - np.exp(-x)
print("v:\n", v)
```

這段程式碼計算了系統中某個物理量 v 隨時間或空間的變化。這裡使用了指數衰減函數來模擬系統的動態行為。

5. 數值微分計算

```python

dv = np.zeros((N), dtype=float)
for i in range(1, N-2):
    dv[i] = (v[i + 1] - v[i - 1]) / (2 * h)
print("dv:\n", dv)
```

使用了三點差分公式來計算 v 的數值導數 dv。這裡假設 dv 是某物理量的變化率。

6. 繪圖展示

```python
fig, ax = plt.subplots()
ax.set_title('dVy / d$\\tau$ = 1 - Vy : three-point formula')
plt.xlabel('t')
plt.ylabel('$V_t$')
ax.plot(x[1 : N-2], v[1 : N-2], 'r-', label='ana.')
ax.plot(x[2 : N-2], 1 - dv[2 : N-2], 'bs', label='num.')
ax.legend(loc='lower right')
plt.show()
```

這段程式碼使用 Matplotlib 來繪製圖形，以直觀地展示分析和數值結果。其中，紅色曲線顯示了解析解，藍色方塊點表示數值計算的結果。

## 結果說明

這段程式碼展示了如何通過數值方法（三點差分）來近似計算系統中某物理量的變化率，並且將結果用圖形方式呈現出來。這樣的方法在物理學和工程學中常用於分析和預測系統的動態行為。

# HW3.1

這段程式碼通過數值積分方法計算了兩個函數在給定範圍內的積分值。在物理學中，這些積分可以用來計算波函數的機率密度或能量分佈函數的特定性質

1. H_atom 函數積分計算

   ```python
   def H_atom(r):
   return ((1/(np.pi)**0.5)\*((1/0.0529**1.5)*np.exp(-r/0.0529))*r)\**2*4\*np.pi
   ```

這個函數 H_atom(r) 計算了氫原子波函數的模長平方，用來描述氫原子中電子的可能位置分佈。在物理上，這個函數的平方可以解釋為電子在給定半徑 r 的概率密度。

2. element 函數積分計算

   ```python
   def element(r):
   return np.exp(-3 _ r / 2) _ r\*\*4
   ```

這個函數 `element(r)` 定義了一個指數函數乘以 𝑟<sup>4<sup>，用於模擬某些物理過程中的能量分佈或其他特性。其積分可以用來計算這些物理量的平均值或總和。

3. 積分方法

程式碼中使用了兩種積分方法來近似計算這些函數的積分值：

- 梯形法：將積分區間分成多個小區間，每個小區間內用梯形近似法計算積分值。

- 矩形法：將積分區間同樣分成多個小區間，每個小區間內用矩形面積來近似計算積分值。

這兩種方法在數值計算中常用來估計函數的積分值，特別是在沒有解析解或解析解較難獲得時。

# HW3.2

這段程式碼展示了如何使用 Python 中的 NumPy、Matplotlib 和 SciPy 庫來進行數據插值，具體來說是使用了拉格朗日插值和三次插值。

## 程式碼解析

1. 導入模組

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline
```

這裡導入了 NumPy 用於數據處理，Matplotlib 用於繪圖，以及 SciPy 中的插值函數 interp1d 和 CubicSpline。

2. 設定數據

```python
a = 0
Ei = np.empty(shape=9)
for i in range(9):
    Ei[i]= a
    a += 25
g = np.array([10.6, 16.0, 45.0, 83.5, 52.8, 19.9, 10.8, 8.25, 4.7])
```

這裡生成了一組實驗數據 g 和對應的能量值 Ei。

3. 進行拉格朗日插值

```python
Ei_lag = np.linspace(0, 200, 200+1)
g_lag = 0
n = 9

for i in range(n):
    p = 1
    for j in range(n):
        if i != j:
            p = p * (Ei_lag - Ei[j])/(Ei[i] - Ei[j])

    g_lag = g_lag + p * g[i]
```

這段程式碼使用了拉格朗日插值方法來估算在更廣泛能量範圍 Ei_lag 上的實驗數據 g_lag。

4. 進行三次插值

```python
f_interp = CubicSpline(Ei, g)
Ei_c = np.linspace(0, 200, 200+1)
g_c = f_interp(Ei_c)
```

這裡使用了 CubicSpline 函數來進行三次插值，生成了 g_c，這是在更細緻能量值範圍 Ei_c 上的插值結果。

5. 繪製圖形

```python
fig, ax = plt.subplots()
ax.plot(Ei, g, 'o', markerfacecolor='none', markeredgecolor='r', label='data')
ax.plot(Ei_lag, g_lag, 'b--', label='Lagrange')
ax.plot(Ei_c, g_c, 'r-', label='Cubic')
ax.legend()
```

最後，使用 Matplotlib 來繪製數據點 Ei 和 g，以及拉格朗日插值和三次插值的比較圖。

# HW4.1

這段程式碼模擬了黑體輻射的特性，並計算了在不同溫度下的最大發射波長。

## 程式碼解析

1.黑體輻射函數

```python
def black(wav, T):
    h = 6.62607515e-34  # 普朗克常數
    hc = 1.98644586e-25  # 普朗克常數乘以光速
    k = 1.380649e-23  # 玻爾茨曼常數
    a = 8.0 * np.pi * hc
    b = hc / (wav * k * T)
    intensity = a / ((wav**5) * (np.exp(b) - 1.0))
    return intensity
```

這裡定義了黑體輻射的強度函數 black(wav, T)，根據溫度 T 和波長 wav 計算特定溫度下的輻射強度。

2. 尋找最大值函數

```python
def maximum(a):
    max_x = 0
    for i in range(len(a)):
        if a[max_x] < a[i]:
            max_x = i
    return max_x
```

這個函數 maximum(a) 用來找出數組 a 中的最大值索引，即最大輻射強度對應的波長索引。

3. 計算不同溫度下的最大發射波長

```python
wav = np.arange(1e-9, 2.0e-6, 1e-10)

ints35 = black(wav, 3500)
print("3500 max is:", wav[maximum(ints35)])

ints40 = black(wav, 4000)
print("4000 max is:", wav[maximum(ints40)])

ints45 = black(wav, 4500)
print("4500 max is:", wav[maximum(ints45)])

ints50 = black(wav, 5000)
print("5000 max is:", wav[maximum(ints50)])

ints55 = black(wav, 5500)
print("5500 max is:", wav[maximum(ints55)])
```

在這裡，使用 np.arange 生成一系列波長 wav，然後分別計算了在 3500K、4000K、4500K、5000K 和 5500K 下的黑體輻射強度 ints35、ints40、ints45、ints50 和 ints55，並找到每個溫度下的最大發射波長。

4. 繪製圖形

```python
plt.plot(wav * 1e9, ints35, 'b-', label='3500K')
plt.plot(wav * 1e9, ints40, 'g-', label='4000K')
plt.plot(wav * 1e9, ints45, 'r-', label='4500K')
plt.plot(wav * 1e9, ints50, 'c-', label='5000K')
plt.plot(wav * 1e9, ints55, 'm-', label='5500K')
plt.xlabel("Wavelength (nm)")
plt.ylabel("Spectral energy density")
plt.legend()
plt.show()
```

最後，使用 Matplotlib 將不同溫度下的黑體輻射強度曲線以及其最大發射波長可視化。橫軸為波長（單位：納米），縱軸為特定能量密度。每個溫度都使用不同顏色的線條表示，並使用圖例標示溫度。

# HW5.1

這段程式碼演示了如何使用有限差分法（finite difference method）計算有限和無限平行電極系統中的電位分佈和電場。

## 程式碼解析

1. 定義電場函數

```python
def Efx(v):
    Efx = np.zeros((N - 1, N - 1))
    for i in range(N - 2):
        for j in range(N - 2):
            Efx[i, j] = (v[i, j + 1] - v[i, j] + v[i + 1, j + 1] - v[i + 1, j]) * (1 / (2 * h))
    return Efx

def Efy(v):
    Efy = np.zeros((N - 1, N - 1))
    for i in range(N - 2):
        for j in range(N - 2):
            Efy[i, j] = (v[i, j + 1] - v[i + 1, j + 1] + v[i, j] - v[i + 1, j]) * (-1 / (2 * h))
    return Efy
```

這兩個函數 Efx 和 Efy 計算了電場的 x 和 y 分量。它們通過有限差分法，基於電位 v 的差分來近似計算電場。

2. 設置有限平行電極系統

```python
N = 31  # 網格點數
h = 1.0  # 步長
L = N - 1  # 網格邊長

# 設置有限差分矩陣 A
A = np.zeros((N**2, N**2))
for i in range(N**2):
    if i < N or i >= N * (N - 1) or (N <= i < N * (N - 1) and (i % N == 0 or i % N == N - 1)):
        A[i, i] = 1
    else:
        A[i, i] = -4
        A[i, i + 1] = 1
        A[i, i - 1] = 1
        A[i, i + N] = 1
        A[i, i - N] = 1
```

這部分代碼設置了有限差分矩陣 A，用於計算有限平行電極系統的電位分佈。根據邊界條件，矩陣的對角元素為 1，非對角元素根據離散的二維拉普拉斯方程進行設置。

3. 設置有限平行電極的電荷和電壓

```python
e = (N - 1) / N  # 電荷密度
top = int(N * 3 / 10)  # 上電極位置
bottom = int(N * 7 / 10)  # 下電極位置

lo = np.zeros((N, N))
lo[top, :] = e * (1. / h**2)
lo[bottom, :] = -e * (1. / h**2)
```

這裡設置了上下兩個電極的電荷密度 lo，用於有限平行電極系統。

4. 解有限平行電極系統的電位

```python
v = np.matmul(inv(A * (-h**2)), lo.reshape(N**2, 1))
```

使用矩陣求逆計算得到有限平行電極系統的電位 v。

5. 繪製有限平行電極系統的電位和電場圖像

```python
fig, ax = plt.subplots()
cf = ax.contourf(xv, yv, v.reshape(N, N), 10, cmap='bwr')
clb = fig.colorbar(cf)
clb.ax.set_title("$v$")

ax.quiver(xvE, yvE, Efx(v.reshape(N, N)), Efy(v.reshape(N, N)))
ax.set_title("Finite Parallel Electrode System")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
plt.grid()
plt.show()
```

這部分代碼用於繪製有限平行電極系統的電位和電場圖像。contourf 函數用於繪製電位分佈，quiver 函數用於繪製電場箭頭。

6. 設置無限平行電極系統

```python
# 設置無限差分矩陣 A
A = np.zeros((N**2, N**2))
for i in range(N**2):
    if i < N or i >= N * (N - 1):
        A[i, i] = 1
    else:
        A[i, i] = -4
        A[i, i + 1] = 1
        A[i, i - 1] = 1
        A[i, i + N] = 1
        A[i, i - N] = 1

# 設置無限平行電極的電荷和電壓
lo = np.zeros((N, N))
lo[top, :] = e * (1. / h**2)
lo[bottom, :] = -e * (1. / h**2)

# 解無限平行電極系統的電位
v = np.matmul(inv(A * (-h**2)), lo.reshape(N**2, 1))
```

這部分代碼設置了無限平行電極系統的矩陣 A、電荷密度 lo，並計算得到無限平行電極系統的電位 v。

7. 繪製無限平行電極系統的電位和電場圖像

```python
fig, ax_in = plt.subplots()
cf = ax_in.contourf(xv, yv, v.reshape(N, N), 10, cmap='bwr')
clb_in = fig.colorbar(cf)
clb_in.ax.set_title("$v$")

ax_in.quiver(xvE, yvE, Efx(v.reshape(N, N)), Efy(v.reshape(N, N)))
ax_in.set_title("Infinite Parallel Electrode System")
ax_in.set_xlabel("$x$")
ax_in.set_ylabel("$y$")
plt.grid()
plt.show()
```

## 結果說明

這段程式碼演示了如何使用 Python 和數值方法計算和視覺化電場和電位，特別是在有限和無限平行電極系統中的應用。

# HW5.2

這段程式碼用於模擬有限平行電極系統的電位分佈和電場情況。透過設定電極的電位分佈，並使用拉普拉斯方程的數值方法，計算並繪製電位和電場的分佈圖。

# 程式碼解析與說明

1. 導入模組

```python
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
```

2. 定義計算電場分量的函數

```python
def Efx(v):
    Efx = np.zeros((N - 1, N - 1))
    for i in range (N - 2):
        for j in range (N -2):
            Efx[i, j] = (v[i, j + 1] - v[i, j] + v[i + 1, j + 1] - v[i + 1, j]) * (1 / (2 * h))
    return Efx

def Efy(v):
    Efy = np.zeros((N - 1, N - 1))
    for i in range (N - 2):
        for j in range (N - 2):
            Efy[i, j] = (v[i, j + 1] - v[i + 1, j + 1] + v[i, j] - v[i + 1, j]) * (-1 / (2 * h))
    return Efy
```

這兩個函數分別計算電位矩陣 v 的 x 和 y 方向的電場分量 Efx 和 Efy。

3. 設置網格和電位

```python
N = 9

top = int(N * 3 / 10)
bottom = int(N * 7 / 10)

v = np.zeros((N, N))
for i in range(top, bottom + 1):
    v[top, i] = 1
    v[bottom, i] = -1
```

設置了網格的尺寸 N，並初始化電位矩陣 v。電極的上邊界設置為 1， 下邊界設置為 -1。

4. 迭代計算電位分佈

```python
h = 1
k = 0
while k < 45:
    v_new = np.zeros((N, N))
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            v_new[i, j] = (1. / 4.) * (v[i + 1, j] + v[i - 1, j] + v[i, j + 1] + v[i, j - 1])
    v = v_new
    k += 1
```

使用拉普拉斯方程的數值方法進行 45 次迭代，更新電位分佈矩陣 v。

5. 繪製電位和電場的分佈圖

```python
x = np.arange(0, N, 1)
y = np.arange(0, N, 1)
xv, yv = np.meshgrid(x, y)
fig, ax = plt.subplots()
cf = ax.contourf(xv, yv, v.reshape(N, N), 10, cmap='bwr')
clb = fig.colorbar(cf)
clb.ax.set_title("$v$")

xE = np.arange(0, N - 1, 1)
yE = np.arange(0, N - 1, 1)
xvE, yvE = np.meshgrid(xE, yE)
ax.quiver(xvE, yvE, Efx(v.reshape(N, N)), Efy(v.reshape(N, N)))
ax.set_title("E-potential and E-field numerical")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
plt.grid()
plt.show()
```

使用 Matplotlib 繪製電位等高線圖和電場箭頭圖，展示了電場的方向和強度。

# HW6.1

這段程式碼模擬了一維有限深勢阱中的量子粒子的能級和波函數，並繪製了相應的圖形。以下是更詳細的解析和說明：

## 程式碼解析與說明

1. 設置網格和勢阱參數

```python
N = 300
N_grid = 2 \* N + 1
H = np.zeros((N_grid, N_grid), dtype=float)
width = 10
dx = 0.1
```

這部分代碼設置了網格點的數量 N 和網格總數 N_grid，並初始化哈密頓矩陣 H。width 表示勢阱的寬度，dx 是空間步長。

2. 填充 Hamilton 矩陣

```python
for i in range(0, N_grid):
for j in range(0, N_grid):
if j == i - 1:
H[i, j] = -0.5 / (dx**2)
elif i == j:
x = dx \* (i - N)
if -width / 2 < x < width / 2:
H[i, j] = 1.0 / (dx**2)
else:
H[i, j] = 1.0 / (dx**2) + 0.3
elif j == i + 1:
H[i, j] = -0.5 / (dx**2)
```

這段代碼根據有限差分法填充 Hamilton 矩陣 H。在矩陣對角線上填入勢能函數的值，非對角線上填入動能項。

3. 求解 Hamilton 矩陣的特徵值和特徵向量

```python
value, vector = linalg.eig(H)
idx = np.argsort(value)
value = value[idx]
vector = vector[:, idx] \* 0.67
```

使用 `numpy.linalg.eig` 函數求解哈密頓矩陣的特徵值和特徵向量，並按特徵值升序排序。

4. 繪製能級圖

```python
nmax = 5
nsho = np.linspace(0, nmax, nmax + 1)
fig1, ax1 = plt.subplots()
ax1.plot(nsho + 1, value[0 : nmax + 1] _ 10000, 'o', label="numerical")
ax1.set_xlabel("n")
ax1.set_ylabel('$E_n/\hbar\omega_0$')
ax1.set_title("finite: eigen energies")
ax1.set_xticks(np.linspace(1, nmax + 1, nmax _ 2 + 1))
plt.grid(linewidth=0.5)
ax1.legend()
```

這段代碼繪製了前 `nmax` 個特徵值，表示系統的能級。

5. 繪製波函數圖

```python
fig2, ax2 = plt.subplots()
x = np.linspace(0, N*grid - 1, N_grid)
x = x - N
x = x \* dx
ax2.plot(x, vector[:, 0], label="$\psi*{n=0}(x)$")
ax2.set_ylabel("wave function")
ax2.set_xlabel("position(x/$\lambda_0$)")
ax2.set_title("finite: The wave of the lowest eigen state: n = 1")
ax2.legend()
plt.show()
```

這段代碼繪製了基態波函數。

# HW6.2

這段程式碼演示了如何利用緊束縛模型來計算一維晶格中電子的能帶結構。緊束縛模型是一種常用的量子力學模型，用於研究晶體中的電子結構。以下是該程式碼的詳細解析和說明：

## 程式碼解析與說明

1. 導入模組

```python
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg
```

2. 定義能量函數

```python
def E(ka):
    t = -1
    E = 2 * t * np.cos(ka)
    return E
```

這段代碼定義了能量函數 E(ka)，其中 t 是跳躍參數，ka 是波矢量。

3. 設置矩陣尺寸

```python
N = 5
```

這裏定義了矩陣的尺寸 N。

4. 初始化 Hamilton 矩陣

```python
Hmn = np.zeros((N, N), dtype=float)
Hmn[1][0] = -1
Hmn[N - 2][N - 1] = -1
for m in range(1, N - 1):
    Hmn[m - 1][m] = -1
    Hmn[m + 1][m] = -1
print(Hmn)
```

這段代碼創建並填充哈密頓矩陣 Hmn。矩陣對角線上下兩側填入 -1，模擬最近鄰的電子跳躍。

5. 求解 Hamilton 矩陣的特徵值和特徵向量

```python
value, vector = linalg.eig(Hmn)
idx = np.argsort(value)
value = value[idx]
value = np.concatenate((np.flip(value, 0), value), axis=0)
print(value)
```

這段代碼使用 `numpy.linalg.eig` 函數求解哈密頓矩陣的特徵值和特徵向量，並按特徵值升序排序。然後將特徵值翻轉並拼接以展示對稱性。

6. 繪製能帶圖

```python
x = np.linspace(-np.pi, np.pi, N * 2)
inf_x = np.linspace(-np.pi, np.pi, N * 10)
print(x)
fig1, ax1 = plt.subplots()
ax1.plot(x, value, 'bo', label="finite")
ax1.plot(inf_x, E(inf_x), 'r-', label="infinite")
ax1.set_title("tight binding model")
ax1.set_ylabel("$E$")
ax1.set_xlabel("$ka$")
ax1.legend()
plt.grid()
plt.show()
```

這段代碼繪製了有限和無限情況下的能帶圖。其中，藍色點表示有限晶格的數值結果，紅色線表示無限晶格的解析解。

# HW7.1

這段程式碼展示了使用 Euler method 和 Runge-Kutta 2nd order method 來求解一個簡單的微分方程問題，並將數值解與解析解進行比較。這個微分方程描述了一個受重力和空氣阻力作用的物體的運動。以下是詳細的解析和說明：

## 程式碼解析與說明

1. 導入模組

```python
import numpy as np
import matplotlib.pyplot as plt
```

2. 定義常數和初始條件

```python
m = 1  # 質量
gravity = 9.8  # 重力加速度
k = 0.1  # 空氣阻力係數
Vt = m * gravity / k  # 終端速度
h = 1  # 步長
N = 50  # 時間步數
```

3. 定義微分方程和解析解

```python
def g(y, t):
    return (m * gravity - k * y) * m

def ysol(t):
    return Vt * (1 - np.exp(- k * t / m ))
```

這裡的 `g(y, t)` 是微分方程的右側部分，`ysol(t)` 是解析解。

4. 初始化數值解和時間步數

```python
yi = np.zeros(N)
gi = np.zeros(N)
ti = np.arange(N) * h
ai = np.zeros(N)
```

5. 使用 Euler method 進行數值積分

```python
t0 = ti[0]
yi[0] = Vt * (1 - np.exp(- k * t0 / m ))
for i in range(0, N - 1):
    gi[i] = g(yi[i], ti[i])
    yi[i + 1] = yi[i] + h * gi[i]
```

Euler method 每一步使用當前的導數值來更新下一步的解。

6. 計算解析解

```python
ai = ysol(ti)
```

7. 繪製 Euler method 的結果與解析解的比較圖

```python
fig1, ax1 = plt.subplots()
ax1.plot(ti, yi, "o", label = "Euler")
ax1.plot(ti, ai, label = "Solution")
plt.xlabel("$t$")
plt.ylabel("$v$")
ax1.text(20, 4.5, "$dv_y/dt= mg - k \cdot v_y$", fontsize='large')
ax1.legend()
```

8. 定義 Runge-Kutta 2nd order method

```python
def RKTwo():
    yi[0] = Vt * (1 - np.exp(- k * t0 / m ))
    for i in range(N - 1):
        k1 = h * g(yi[i], ti[i])
        k2 = h * g(yi[i] + h, ti[i] + k1)
        yi[i + 1] = yi[i] + (k1 + k2) / 2  # 修正錯誤
    return yi
```

Runge-Kutta 2nd order method 每一步使用兩次導數值來更新下一步的解，修正公式為 `(k1 + k2) / 2`。

9. 繪製 RK2 方法的結果與解析解的比較圖

```python
fig2, ax2 = plt.subplots()
ax2.plot(ti, RKTwo(), "ro", label = "RKTwo")
plt.xlabel("$t$")
plt.ylabel("$v$")
ax2.text(20, 4.5, "$dv_y/dt= mg - k \cdot v_y$", fontsize='large')
ax2.plot(ti, ai, label = "Solution")
ax2.legend()
plt.show()
```

# final

這段程式碼主要分為兩個部分：

求解具有不同電壓配置的平行板電容器問題，並繪製其電勢分布和頂部電壓。
利用所得的電勢分布計算量子力學中粒子的波函數，並繪製其特徵值和波函數圖。

## 程式碼解析與說明

1. 初始化參數和設置電容器板的位置與電壓

`N = 35`：定義網格點數目
`L = N - 1`：計算網格長度
`h = L / (N - 1)`：計算每個網格的步長
`e = (N - 1) / N`：一個縮放因子，用於電壓的計算
`bottom` 和 `top`：分別表示下板和上板的位置
`a`：用於計算電容器板的寬度
`lo`：初始化一個
`N×N `的零矩陣，用於存儲電壓分布

```python
N = 35
L = N - 1
h = L / (N - 1)
e = (N - 1) / N
bottom = int(N * 3 / 10)
top = int(N * 7 / 10)
a = int(N / 7)
lo = np.zeros((N, N))
```

2. 設置電容器板的電壓分布

`lo[bottom, a: N - a]`：設置底部電容器板的電壓。
`lo[top, a : a + a]`、`lo[top, m: m + a] `和 `lo[top, N - a - a: N - a]`：設置上部電容器板的電壓，不同區域設置不同電壓值。

```python
lo[bottom, a: N - a] = e * (1./h**2) * 0
lo[top, a : a + a] = e * (1./h**2) * 40
m = int(N / 2) - 1
lo[top, m: m + a] = e * (1./h**2) * 30
lo[top, N - a - a: N - a] = e * (1./h**2) * 40
```

3. 構建矩陣 A
   這是一個有限差分方法，用於求解泊松方程。矩陣 A 的元素設置方式如下：

- 在邊界點處設置對角元素為 1。
- 在內部點處設置對角元素為 -4，並設置相鄰點的元素為 1。

```python
A = np.zeros((N**2, N**2))
for i in range (N**2):
if N > i:
A[i, i] = 1
elif (N _ (N - 1)) <= i:
A[i, i] = 1
elif (N <= i < N _ (N - 1)) and (i % N == 0 or i % N == N - 1):
A[i, i] = 1
else:
A[i, i] = -4
A[i, i + 1] = 1
A[i, i - 1] = 1
A[i, i + N] = 1
A[i, i - N] = 1
```

4. 求解電勢分布

使用矩陣 𝐴 的逆矩陣乘以電壓分布 lo，求解電勢分布 𝑣。

```python
v = np.matmul(inv(A \*(-h\*\*2)), lo.reshape(N\*\*2,1))
v_top = v.reshape(N, N)[top, :]
```

5. 繪製電勢分布和頂部電壓

使用`matplotlib`繪製電勢分布的等高線圖和頂部電壓隨位置變化的圖。

```python
x = np.arange(0, N, 1)
y = np.arange(0, N, 1)
xv, yv = np.meshgrid(x, y)
fig1, (ax, bx) = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 4))
cf = ax.contourf(xv, yv, v.reshape(N, N), 10)
clb = fig1.colorbar(cf)
clb.ax.set_title("$v$")
ax.set_title("finite E-potential and E-field numerical")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
bx.plot(x, v_top)
bx.set_title("top voltage")
bx.set_xlabel("x")
bx.set_ylabel("voltage")
bx.grid()
plt.show()
```

6. 計算波函數

定義一個函數 wave，生成哈密頓矩陣 𝐻。
使用 linalg.eig 函數計算哈密頓矩陣的特徵值和特徵向量，並按特徵值排序。

```python
width = 10
dx = 0.1

def wave(N):
H = np.zeros((N, N), dtype = float)
for i in range(0, N):
for j in range(0, N):
if j == i - 1:
H[i, j] = -0.5 / (dx**2)
elif i == j:
H[i, j] = 1.0 / (dx**2) + v_top[i]
elif j == i + 1:
H[i, j] = -0.5 / (dx\*\*2)
return H

H = wave(N)
values, vectors = linalg.eig(H)
idx = np.argsort(values)
values = values[idx]
vectors = vectors[:, idx] \* 0.67
```

7. 繪製波函數的結果

繪製特徵值圖和特徵向量圖，展示最低幾個特徵態的波函數。

```python
x = np.linspace(0, N - 1, N)
fig2, (cx, dx) = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 4))
cx.plot(np.linspace(0, 10, 10), values[0: 10], 'o')
cx.set*xlabel("n")
cx.set_ylabel('$E_n/\hbar\omega_0$')
cx.set_title("finite: eigen energies")
cx.grid(linewidth = 0.5)
dx.plot(x, - vectors[:, 4])
dx.plot(x, - vectors[:, 5])
dx.set_ylabel("wave function")
dx.set_xlabel("position(x/$\lambda_0$)")
dx.set_title("finite: The wave of the lowest eigen state: n = 1 $\psi*{n=0}(x)$")
dx.grid(linewidth = 0.5)
plt.show()
```

## 結果說明

這段程式碼展示了如何使用有限差分方法求解帶有不同電壓配置的平行板電容器的電勢分布，並利用該電勢分布計算量子力學中的波函數。結果通過圖形展示，包括電勢的等高線圖、頂部電壓隨位置變化的圖以及波函數和特徵值的圖。
