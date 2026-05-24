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
    [4 / (4 * M + m)],
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

class PID_controller:
    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0

    def __call__(self, state):
        theta = state[2]        # 摆杆角度
        error = theta           #目标角度作为误差量

        self.integral += error * self.dt                        # 积分
        derivative = (error - self.prev_error) / self.dt        # 微分
        self.prev_error = error

        u = (self.Kp * error + self.Ki * self.integral + self.Kd * derivative)
        return np.clip(u, -20.0, 20.0)                   # 限幅


class CascadePID_controller:
    def __init__ (self, Kp_pos, Kd_pos, Kp_angle,
                  Ki_angle, Kd_angle, dt):
        # 外环
        self.Kp_pos = Kp_pos
        self.Kd_pos = Kd_pos

        # 内环
        self.Kp_angle = Kp_angle
        self.Ki_angle = Ki_angle
        self.Kd_angle = Kd_angle
        self.dt = dt

        self.integral_angle = 0
        self.prev_error_angle = 0

    def __call__(self, state):
        x, x_dot, theta, theta_dot = state

        # 外环：根据位置偏差算期望角度
        # x > 0（偏右）→ desired_theta < 0（左偏），
        # 重力把小车拉回

        desired_theta = -(self.Kp_pos * x + self.Kd_pos * x_dot)

        # 限制期望角度，别让内环追不上
        desired_theta = np.clip(desired_theta, -0.3, +0.3)

        # 内环, 负责跟踪期望角度
        error_angle = theta - desired_theta

        self.integral_angle += error_angle * self.dt
        derivative_angle = (error_angle - self.prev_error_angle) / self.dt
        self.prev_error_angle = error_angle

        u = (self.Kp_angle * error_angle +
             self.Ki_angle * self.integral_angle +
             self.Kd_angle * derivative_angle)

        return np.clip(u, -20.0, +20.0)