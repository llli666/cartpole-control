import numpy as np
import matplotlib.pyplot as plt
from dynamics import step
from controller import lqr_controller

# 初始状态: 小车在中间 摆杆往右边偏移0.1弧度
state = np.array([0.0, 0.0, 0.1, 0.0])
# [x, x_dot, theta, theta_dot]

# 仿真参数
dt = 0.02           # 每步20ms
total_time = 5.0    # 总共5s
steps = int(total_time / dt)

# 记录数据
history = [state.copy()]
actions = [0.0]

# LQR 闭环控制
for i in range(steps):
    action = lqr_controller(state)      # 施加力
    state = step(state, action, dt)
    history.append(state.copy())
    actions.append(action)

actions = np.array(actions)
history = np.array(history)
time = np.arange(len(history)) * dt     # 时间序列

# 画图：角度 + 控制力
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# 图1：摆杆角度
axes[0].plot(time, history[:, 2], label='theta (rad)', color='blue')
axes[0].axhline(y=0, color='r', linestyle='--', label='upright')
axes[0].set_ylabel('Angle (rad)')
axes[0].set_title('CartPole with LQR Control')
axes[0].legend()
axes[0].grid(True)

# 图2：控制力
axes[1].plot(time, actions, label='Force (N)', color='green')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Force (N)')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('results/lqr_control.png', dpi=150)
plt.show()

print("图已保存到 results/lqr_control.png")