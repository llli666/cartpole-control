import numpy as np
from dynamics import step

# 初始状态: 小车在中间 往左右两边偏移一点
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
    action = 0.0        #不加力
    state = step(state, action, dt)
    history.append(state.copy())

print(f"仿真完成，共{len(history)}步")
print(f"最终状态：{history[-1]}")
