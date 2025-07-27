# 🧠 TD3 (Twin Delayed Deep Deterministic Policy Gradient)

TD3 是一种针对连续动作空间任务的深度强化学习算法，旨在解决 DDPG（Deep Deterministic Policy Gradient）中的策略过估计问题。它在多个基准测试任务中表现出色，是目前最稳健的连续控制算法之一。

---

## 📌 模型架构

TD3 主要包括以下几个模块：

### 🎯 Actor 网络（策略网络）
- 输入：状态 \( s \)
- 输出：动作 \( a \)
- 结构：多层前馈神经网络（MLP）
- 目标：最大化 critic 对动作的 Q 值估计

### 🔍 Critic 网络（价值网络）
- 使用**两个 Q 网络**（Q1, Q2）来减少过估计偏差
- 输入：状态 \( s \)，动作 \( a \)
- 输出：Q 值估计
- 更新目标时取两个 Q 网络的最小值 `min(Q1, Q2)`

### 💾 Target 网络
- 为 Actor 和两个 Critic 分别维护一个目标网络
- 使用 soft update 进行慢速参数更新，提升训练稳定性

---

## 🔑 关键创新点（相比 DDPG）

1. **双 Q 网络（Clipped Double Q-Learning）**
   - 减少值函数的过估计问题
   - 更新策略时采用 `min(Q1, Q2)` 作为目标

2. **延迟策略更新（Delayed Policy Update）**
   - 策略网络（Actor）每隔一定步数才更新一次
   - 防止策略过早过拟合到不稳定的 Critic 估计上

3. **目标策略平滑（Target Policy Smoothing）**
   - 给 target action 添加 clipped noise，增加鲁棒性
   - 解决策略网络对 Q 值估计器的快速过拟合

---

## ⚔️ 与其他 DRL 算法的对比

| 算法      | 类型       | 优点                                  | 缺点                                  |
|-----------|------------|---------------------------------------|---------------------------------------|
| DDPG      | 离策略、确定性 | 简单直接，适用于连续动作             | 容易过估计，训练不稳定                |
| SAC       | 离策略、随机性 | 表现好，收敛快，支持随机策略         | 策略熵正则难以调参                    |
| PPO       | 在策略、随机性 | 稳定，适用于大规模离散动作任务       | 不能直接应用于高维连续控制            |
| **TD3**   | 离策略、确定性 | 稳定性强，收敛快，动作输出平滑       | 对于离散动作空间不适用                |

---

## 🚀 TD3 适用场景

- 机器人控制
- 模拟连续动作任务（如 MuJoCo 环境）
- 金融策略优化
- 自动驾驶策略学习

---

## 📚 推荐阅读

- [TD3 原始论文](https://arxiv.org/abs/1802.09477)
- [OpenAI SpinningUp TD3 实现](https://spinningup.openai.com/en/latest/algorithms/td3.html)
- [CleanRL TD3 代码实现](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py)



## 🛠️ 环境依赖

```bash
pip install torch gym numpy
```
---

## 🧩 核心代码架构 (PyTorch)
- [TD3 代码](https://github.com/ZhengYuhaoBUPT/DRL-Pytorch/tree/main/4.2%20TD3)
