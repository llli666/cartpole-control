import numpy as np
from scipy.linalg import solve_continuous_are #LQR核心代码

# 倒立摆物理参数
M = 1.0         #小车质量（kg)
m = 0.1         # 摆杆质量（kg)
l = 0.5         # 摆杆半长（m）
g = 9.81        # 重力加速度

# 1.====================================================
# 手动推导出AB状态矩阵
A = np.array([
    [0, 1, 0, 0],
    [0, 0, -3 * m * g / (4 * M + m), 0],
    [0, 0, 0, 1],
    [0, 0, 3 * (M + m) * g / (l * (4 * M + m)), 0]
])

B = np.array([
    [0],
    [4 / (4 * M) + m],
    [0],
    [-3 / (l * (4 * M + m))]
])

# 2.====================================================
# LQR 权重矩阵
Q = np.diag([1.0, 0.1, 10.0, 0.1])
# 对角Q矩阵对应[x, x_dot, theta, theta_dot]
# 数字越大，越归零
R = np.array([[0.1]]) # 输入力 惩罚越小，力越大

# 3.====================================================
# 解黎卡迪方程
P = solve_continuous_are(A, B, Q, R)  # 解
K = np.linalg.inv(R) @ B.T @ P        # 反馈增益 K = R^-1 B(T) P

print("LQR 反馈增益 K =", K.flatten())

# 4.====================================================
# 控制器函数
def lqr_controller(state):
    """
       状态反馈控制律: u = -K * state
       state: [x, x_dot, theta, theta_dot]
       返回: 控制力 F (N)
    """
    state = np.array(state, dtype=float).reshape(4)
    u = - np.dot(K.flatten(), state)   # 提取1*1矩阵里的数字

    # 限制输出力 +-20
    u = np.clip(u, -20.0, 20.0)
    return float(u)

