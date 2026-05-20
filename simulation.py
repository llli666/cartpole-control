import numpy as np
import matplotlib.pyplot as plt
from dynamics import step

# 初始状态: 小车在中间 摆杆往右边偏移0.1弧度
state = np.array([0.0, 0.0, 0.1, 0.0])
# [x, x_dot, theta, theta_dot]

# 仿真参数
dt = 0.02           # 每步20ms
total_time = 5.0    # 总共5s
steps = int(total_time / dt)

# 记录数据
history = [state.copy()]

# 主循环: 不给控制，摆杆自由落下
for i in range(steps):
    action = 0.0        # 不加力
    state = step(state, action, dt)
    history.append(state.copy())

history = np.array(history)
time = np.arange(len(history)) * dt     # 时间序列

# 画图
plt.figure(figsize=(10, 4))
plt.plot(time, history[:, 2], label='theta(rad)')       #第三列是角度
plt.axhline(y=0, color='r', linestyle='--', label='upright')
plt.xlabel('Time(s)')
plt.ylabel('Angle(rad)')
plt.title('CartPole Free Fall (no control)')
plt.legend()
plt.grid(True)
plt.savefig('results/free_fall.png', dpi=150)
plt.show()

print("图片已经保存到 result/free_fall.png")
print(f"仿真完成，共{len(history)}步")
print(f"最终状态：{history[-1]}")
