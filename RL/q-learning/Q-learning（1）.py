#https://www.cnblogs.com/haohai9309/p/18203233
import numpy as np

# 定义参数
gamma = 0.8  # 折扣因子
alpha = 0.1  # 学习率
epsilon = 0.1  # 探索概率
episodes = 12000  # 学习的总回合数

# 定义收益矩阵
R = np.array([
    [-1, -1, -1, -1, 0, -1],
    [-1, -1, -1, 0, -1, 100],
    [-1, -1, -1, 0, -1, -1],
    [-1, 0, 0, -1, 0, -1],
    [0, -1, -1, 0, -1, 100],
    [-1, 0, -1, -1, 0, 100]
])

# 初始化Q值矩阵
Q = np.zeros_like(R, dtype=float)

# Q-learning算法
for episode in range(episodes):
    # 随机选择初始状态
    state = np.random.randint(0, R.shape[0])#.shape[0]表示第一维度的长度，同理.shape[1]表示第二维度的长度
    
    while True:
        # 选择动作：采用ε-贪婪策略
        if np.random.rand() < epsilon:
            action = np.random.randint(0, R.shape[1])#探索模式，随机去另一个状态
        else:
            action = np.argmax(Q[state])#贪婪模式，去q值最大的状态

        # 执行动作，获取即时奖励和下一个状态
        next_state = action
        reward = R[state, action]
        
        # 只在奖励不为 -1 时进行 Q 值更新
        if reward != -1:
            # Q值更新
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

            # 状态转移
            state = next_state
       
       # 如果达到了终止状态，结束本回合
        if reward == 100:
            break
    if episode%1000==0 and episode>=1000:
        print(f"这是第{episode}次迭代")
        Q = np.round(Q, 2)
        print(Q)
        
# 输出最终的Q值矩阵
print("最终的Q值矩阵")
Q = np.round(Q, 2)
print(Q)

    