#https://www.cnblogs.com/haohai9309/p/18203233
import numpy as np

#奖励矩阵
R = [[-1000, -100, 0], [-5, 0, -100], [-100, 0, 0], [-1000, -1000, 0],
     [-5, -1000, 100], [-5, 100, -1000], [0, -1000, -1000], [-1000, 0, 0], [0, 0, -1000]]

# 状态转移矩阵
S = [[0, 7, 1], [0, 2, 8], [7, 4, 5], [3, 3, 4], [3, 4, 6],
     [2, 6, 5], [4, 6, 6], [7, 3, 2], [1, 5, 8]]

#初始化Q表
Q=np.zeros_like(R)

#折扣因子与学习率
y=0.9
a=0.9

epoch=100
done = False
action = 0
state = 0
road = [0]  # 构建一个数组存放路径，0表示默认位置

for i in range(epoch):
    state=0
    done=False
    while not done:
        if np.random.rand()<0.9:
            action=np.argmax(Q[state])
        else:
            action=np.random.randint(0,3)
        state_new=S[state][action]
        #更新Q值
        Q[state][action]=Q[state][action]+a*(R[state][action]+y*np.argmax(Q[state])-Q[state][action])

        road.append(state_new)

        if state_new in [6,7,8]:
            done=True
        state=state_new
    
    print(f"这是第{i}次迭代")
    Q = np.round(Q, 2)
    print(Q)
    print(road)
    road=[0]

print("最终q值表:")
print(Q)
