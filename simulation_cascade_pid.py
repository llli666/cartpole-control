import numpy as np
import matplotlib.pyplot as plt
from dynamics import step
from controller import CascadePID_controller

# 初始状态
state = np.array([0.1, 0.0, 0.1, 0.0])
dt = 0.02
total_time = 30.0   # 给 10 秒，看位置是否回来
steps = int(total_time / dt)

# 串级 PID 参数（先给一组初值）
cascade_pid = CascadePID_controller(
    Kp_pos=0.21,    # 位置比例
    Kd_pos=0.02,    # 位置微分
    Kp_angle=200.0,# 角度比例
    Ki_angle=0.01,  # 角度积分
    Kd_angle=15.0, # 角度微分
    dt=dt
)

history = [state.copy()]
actions = [0.0]

for i in range(steps):
    action = cascade_pid(state)
    state = step(state, action, dt)
    history.append(state.copy())
    actions.append(action)

history = np.array(history)
actions = np.array(actions)
time = np.arange(len(history)) * dt

# 画图：位置 + 角度 + 力
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(time, history[:, 0], label='x (m)', color='purple')
axes[0].axhline(y=0, color='r', linestyle='--', label='target position')
axes[0].set_ylabel('Position (m)')
axes[0].set_title('CartPole with Cascade PID')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(time, history[:, 2], label='theta (rad)', color='blue')
axes[1].axhline(y=0, color='r', linestyle='--', label='upright')
axes[1].set_ylabel('Angle (rad)')
axes[1].legend()
axes[1].grid(True)

axes[2].plot(time, actions, label='Force (N)', color='green')
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Force (N)')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig('results/cascade_pid_control.png', dpi=150)
plt.show()

print(f"最终小车位置 x = {history[-1, 0]:.3f} m")
print(f"最终摆杆角度 theta = {history[-1, 2]:.6f} rad")