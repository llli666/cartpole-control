import numpy as np
from controller import PID_controller
from dynamics import step
import matplotlib.pyplot as plt

# 初始状态
state = np.array([0.0, 0.0, 0.1, 0.0])
dt = 0.02
total_time = 5.0
steps = int(total_time / dt)

# 给定初值 返回u
pid = PID_controller(Kp=200.0, Ki=0.0, Kd=15, dt=dt)        # 对象

# 记录数据
history = [state.copy()]
actions = [0.0]

# PID闭环控制
for i in range(steps):
    action = pid(state)
    state = step(state, action, dt)
    history.append(state.copy())
    actions.append(action)

history = np.array(history)
actions = np.array(actions)
time = np.arange(len(history)) * dt

# 画图
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

axes[0].plot(time, history[:, 2], label='theta (rad)', color='blue')
axes[0].axhline(y=0, color='r', linestyle='--', label='upright')
axes[0].set_ylabel('Angle (rad)')
axes[0].set_title('CartPole with PID Control')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(time, actions, label='Force (N)', color='green')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Force (N)')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('results/pid_control.png', dpi=150)
plt.show()
print("图已保存到 results/pid_control.png")
print(f"最终小车位置 x = {history[-1, 0]:.3f} m")
print(f"最终摆杆角度 theta = {history[-1, 2]:.6f} rad")