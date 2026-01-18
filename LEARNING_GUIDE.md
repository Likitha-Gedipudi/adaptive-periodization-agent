# Adaptive Periodization Agent: Learning Guide & Presentation Notes

> A comprehensive guide to the reinforcement learning techniques, reward engineering, and ML best practices used in this project.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Reinforcement Learning Fundamentals](#reinforcement-learning-fundamentals)
3. [The SAC Algorithm](#the-sac-algorithm)
4. [Environment Design](#environment-design)
5. [Reward Engineering](#reward-engineering)
6. [Safety Constraints](#safety-constraints)
7. [Training Pipeline](#training-pipeline)
8. [Key Results & Learnings](#key-results--learnings)
9. [Best Practices](#best-practices)

---

## 🎯 Project Overview

### The Problem
Prescribing daily training recommendations to optimize **long-term fitness adaptation** while:
- Managing recovery
- Preventing overtraining
- Respecting individual variability

### Why RL?
Traditional rule-based systems fail because:
- Recovery is **individual** and **dynamic**
- Optimal training involves **delayed rewards** (fitness gains take weeks)
- Credit assignment is hard: which action led to which outcome?

### Solution Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  User Data      │────▶│  RL Environment │────▶│  SAC Agent      │
│  (Physiology)   │     │  (Gymnasium)    │     │  (Policy)       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
   HRV, RHR,              State, Reward,          Action (1 of 6
   Sleep, CTL             Constraints             training types)
```

---

## 🧠 Reinforcement Learning Fundamentals

### Core Concepts

| Concept | In This Project |
|---------|-----------------|
| **State** | 17 physiological features (HRV, recovery, CTL, TSB, etc.) |
| **Action** | 6 discrete training types (Rest → HIIT) |
| **Reward** | Composite: short-term recovery + medium-term adaptation + long-term fitness |
| **Policy** | Neural network that maps state → action probabilities |
| **Value** | Expected future rewards from current state |

### The Agent-Environment Loop

```python
# Simplified training loop
for episode in range(n_episodes):
    state = env.reset()
    
    for step in range(90):  # 90-day episode
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.learn(state, action, reward, next_state)
        state = next_state
```

### Key Insight: Delayed Rewards
The hardest part of this problem is **credit assignment**:
- An action today affects fitness 30 days later
- How do we know which action caused which outcome?

**Solution**: Pre-compute future metrics and use **temporal difference learning**.

---

## 🔄 The SAC Algorithm

### Why Soft Actor-Critic (SAC)?

| Feature | Benefit |
|---------|---------|
| **Off-policy** | Sample efficient - reuses past experiences |
| **Entropy regularization** | Explores more, avoids local optima |
| **Dual Q-networks** | Reduces overestimation bias |
| **Auto temperature tuning** | Adapts exploration automatically |

### SAC Components

```
┌──────────────────────────────────────────────────────────┐
│                     SAC Agent                             │
├──────────────────────────────────────────────────────────┤
│  Actor Network      │  π(a|s) → action probabilities     │
│  Critic Networks    │  Q1(s,a), Q2(s,a) → Q-values       │
│  Target Networks    │  Slow-moving copies for stability   │
│  Replay Buffer      │  Store & sample past experiences    │
│  Temperature (α)    │  Controls exploration vs exploitation│
└──────────────────────────────────────────────────────────┘
```

### Key Hyperparameters

```yaml
sac:
  learning_rate_actor: 3e-4
  learning_rate_critic: 3e-4
  hidden_dims: [256, 256]
  gamma: 0.99          # Discount factor
  tau: 0.005           # Target network update rate
  alpha: 0.2           # Initial entropy weight
  auto_alpha: true     # Auto-tune entropy
  batch_size: 256
  buffer_size: 1000000
```

---

## 🏋️ Environment Design

### State Space (17 Features)

| Category | Features |
|----------|----------|
| **Physiological** | HRV, resting HR, recovery, sleep duration, sleep efficiency |
| **Training Load** | ATL (acute), CTL (chronic), TSB (balance) |
| **Rolling Stats** | 7-day HRV z-score, HRV trend, recovery trend |
| **Temporal** | Day sin/cos (weekly cycles), days since high recovery |
| **Derived** | Recovery-fatigue interaction, TSB-recovery interaction |

### Action Space (6 Actions)

| Action | Description | Intensity |
|--------|-------------|-----------|
| 0 - REST | Complete rest | None |
| 1 - ACTIVE_RECOVERY | Zone 1, <60% HR | Low |
| 2 - AEROBIC_BASE | Zone 2, 60-70% HR | Moderate |
| 3 - TEMPO | Zone 3-4, 70-85% HR | High |
| 4 - HIIT | Zone 5, >85% HR | Very High |
| 5 - STRENGTH | Resistance training | High |

### Training Load Metrics

```
ATL (Acute Training Load) = 7-day exponential moving average of strain
CTL (Chronic Training Load) = 42-day exponential moving average of strain
TSB (Training Stress Balance) = CTL - ATL

         CTL (Fitness)
              ↑
              │
    Fatigued  │   Peak Form
    TSB < 0   │   TSB > 0
              │
    ──────────┼──────────▶ ATL (Fatigue)
              │
```

---

## 💰 Reward Engineering

### The Challenge
Getting reward engineering right is **critical** for RL success.

### Evolution of Our Reward Function

#### ❌ Attempt 1: Original (Result: -112 to +10)
```python
reward = (
    0.2 * short_term_reward / 10.0 +
    0.3 * medium_term_reward / 5.0 +
    0.5 * long_term_reward / 10.0
)
# Problem: Too small, high variance, hard to learn
```

#### ✅ Attempt 2: Improved (Result: +700)
```python
# Base reward per step (guarantees positive trajectory)
base_reward = 6.0  # 90 steps × 6 = 540 base

# Remove normalization divisors
short_reward = hrv_change  # Was: hrv_change / 10.0
medium_reward = ctl_growth  # Was: ctl_growth / 5.0
long_reward = fitness_gain  # Was: fitness_gain / 10.0

# Add recovery bonus
if recovery >= 70:
    recovery_bonus = 2.0
elif recovery >= 50:
    recovery_bonus = 1.0

reward = base_reward + weighted_sum + recovery_bonus
```

### Key Lesson: Reward Shaping

| Technique | Effect |
|-----------|--------|
| **Base reward** | Ensures learning signal even when components cancel out |
| **Scaling** | Make rewards large enough for gradient signal |
| **Recovery bonus** | Immediate feedback for maintaining health |
| **Reduced penalties** | Don't overwhelm positive rewards |

---

## 🛡️ Safety Constraints

### Two-Pronged Approach

#### 1. Hard Constraints (Action Masking)
Completely block dangerous actions:

```python
def get_action_mask(state, history):
    mask = [True] * 6  # All actions initially allowed
    
    # Force rest if recovery critically low
    if state["recovery"] < 30:
        mask = [True, False, False, False, False, False]
    
    # Block HIIT if HRV crashed
    if state["hrv"] < baseline - 2 * std:
        mask[4] = False  # Block HIIT
        mask[3] = False  # Block Tempo
    
    return mask
```

#### 2. Soft Constraints (Penalties)
Discourage but don't prevent risky behavior:

```python
def calculate_penalty(action, state, history):
    penalty = 0.0
    
    # Penalty for training with low recovery
    if state["recovery"] < 40 and action > 0:
        penalty += 1.0
    
    # Penalty for too many consecutive hard days
    consecutive_hard = count_consecutive_high_intensity(history)
    if consecutive_hard > 3:
        penalty += 0.5 * (consecutive_hard - 3)
    
    return penalty
```

### Result: Zero Violations
Throughout 27,000+ training steps, the agent learned to respect constraints while maximizing rewards.

---

## 🏃 Training Pipeline

### Data Flow

```
Synthetic Data (200 users × 180 days)
           │
           ▼
┌─────────────────────┐
│   Preprocessing     │  ← Normalization, missing values
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│ Feature Engineering │  ← ATL, CTL, TSB, rolling stats
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│    Environment      │  ← Gymnasium wrapper
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│    SAC Training     │  ← 300+ episodes
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│    Evaluation       │  ← vs baselines
└─────────────────────┘
```

### Training Configuration

```yaml
training:
  episodes: 300-1000
  episode_length: 90 days
  
data:
  n_users: 200
  n_days: 180
  
reward:
  short_weight: 0.5   # Favor immediate feedback
  medium_weight: 0.3
  long_weight: 0.2
  penalty_weight: 0.3  # Reduced from 1.0
```

---

## 📊 Key Results & Learnings

### Training Progression

| Stage | Episodes | Best Reward | Notes |
|-------|----------|-------------|-------|
| Initial (baseline) | 10 | -97 | Random exploration |
| First convergence | 222 | +6.5 | Early stopping triggered |
| Extended training | 1000 | +10.2 | Still limited |
| **After reward fix** | 300 | **+756** | Target exceeded! |

### What Made the Difference

| Change | Impact |
|--------|--------|
| Added base reward (+6/step) | Guaranteed ~540 per episode |
| Removed divisors (/10, /5) | 5-10x larger gradients |
| Recovery bonus | Immediate positive signal |
| Reduced penalties (0.3x) | Stopped overwhelming rewards |
| More data (200 users) | Better generalization |

### Final Model Performance

```
Episodes: 300
Best Reward: +756.79
Final Avg: +712.35
Eval Mean: +714.96 ± 11.65
Violations: 0
```

---

## ✅ Best Practices

### 1. Reward Engineering
- Start with **large, positive base rewards**
- Add **shaped rewards** for intermediate goals
- Keep penalties **small relative to rewards**
- Test reward scale empirically

### 2. Environment Design
- Use **action masking** for safety-critical constraints
- Include **temporal features** for policies that depend on history
- Normalize inputs to **similar scales**

### 3. Algorithm Selection
- **SAC** for sample efficiency with discrete actions
- **PPO** for simpler implementation
- Consider **model-based** RL if you have a simulator

### 4. Training Tips
- Start with **quick runs** (10-50 episodes) to debug
- Monitor **reward components separately**
- Use **TensorBoard** to track learning
- Implement **early stopping** but disable for final runs

### 5. Safety
- Implement constraints at **multiple levels**
- Log all violations for analysis
- Consider **constrained RL** algorithms for critical applications

---

## 🗂️ Project Structure

```
AP_project/
├── src/
│   ├── data/              # Data loading, preprocessing, features
│   ├── environment/       # Gymnasium env, rewards, constraints
│   ├── models/            # SAC agent, networks, baselines
│   ├── training/          # Training loop, callbacks, tuning
│   └── evaluation/        # Metrics, evaluation, visualization
├── tests/                 # Unit tests
├── experiments/           # Saved models and logs
└── notebooks/             # Analysis notebooks
```

---

## 📚 Further Reading

### Reinforcement Learning
- [Spinning Up in Deep RL](https://spinningup.openai.com/) - OpenAI's RL intro
- [SAC Paper](https://arxiv.org/abs/1801.01290) - Original SAC algorithm
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)

### Domain Knowledge
- [Training Load & TSB](https://www.trainingpeaks.com/learn/articles/what-is-training-stress-balance/)
- [HRV & Recovery](https://www.whoop.com/thelocker/heart-rate-variability-hrv/)

### Code References
- [Gymnasium](https://gymnasium.farama.org/) - Environment API
- [PyTorch](https://pytorch.org/) - Neural network framework

---

## 🎓 Summary for Presentation

### Key Takeaways

1. **RL can solve complex sequential decision problems** where traditional ML fails

2. **Reward engineering is crucial** - often more important than algorithm choice

3. **Safety constraints prevent dangerous actions** without sacrificing performance

4. **Iterative development works** - start simple, measure, improve

5. **Domain knowledge + ML = success** - understanding fitness metrics was essential

### One-Liner
> We built an AI personal trainer that learns to optimize your fitness while keeping you safe from overtraining.

---

*Created: January 2026*
*Project: Adaptive Periodization Agent*
