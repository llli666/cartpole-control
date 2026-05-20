import numpy as np
# 倒立摆物理参数
M = 1.0         #小车质量（kg)
m = 0.1         # 摆杆质量（kg)
l = 0.5         # 摆杆半长（m）
g = 9.81        # 重力加速度
def cartpole_dynamics(state, action):
    """
       倒立摆动力学
       state: [x, x_dot, theta, theta_dot]
           x: 小车位置
           x_dot: 小车速度
           theta: 摆杆角度（弧度，0为竖直向上）
           theta_dot: 摆杆角速度
       action: 水平力 F (N)

       返回: [x_dot, x_ddot, theta_dot, theta_ddot]
       """
    x, x_dot, theta, theta_dot = state
    F = float(action)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # 临时变量
    temp = (F + m * l * theta_dot**2 * sin_t) / (M + m)

    # 摆杆角加速度
    theta_ddot = (g * sin_t - cos_t * temp) / (
        l * (4.0 / 3.0 - m *cos_t**2 / (M + m))
    )

    # 小车线加速度
    x_ddot = temp - (m * l * theta_ddot * cos_t) / (M + m)

    return np.array([x_dot, x_ddot, theta_dot, theta_ddot], dtype=float)

def step(state, action, dt = 0.02):
    """
    欧拉法积分 返回下一步状态   
    """
    deriv = cartpole_dynamics(state, action)
    new_state = state + deriv * dt
    return new_state
