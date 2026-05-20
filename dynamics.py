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
    F = action

    # TODO: 这里明天填公式
    # 先随便返回0，保证函数能跑
    x_ddot = 0.0
    theta_ddot = 0.0

    return np.array([x_dot, x_ddot, theta_dot, theta_ddot])

def step(state, action, dt = 0.02):
    """
    欧拉法积分 返回下一步状态   
    """
    deriv = cartpole_dynamics(state, action)
    new_state = state + deriv * dt
    return new_state
