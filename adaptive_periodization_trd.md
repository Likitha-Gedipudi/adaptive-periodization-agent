# Adaptive Periodization Agent - Technical Requirements Document

## 1. Project Overview

**Objective**: Build a reinforcement learning agent that prescribes daily training recommendations based on physiological recovery metrics, optimizing long-term fitness adaptation while ensuring safety constraints.

**Core Challenge**: Navigate the credit assignment problem in delayed reward scenarios where training stress today produces measurable fitness gains 14-30 days later.

---

## 2. Data Requirements

### 2.1 Input Features (State Space)

**Physiological Metrics** (Daily):
- Heart Rate Variability (HRV) - RMSSD in ms
- Resting Heart Rate (RHR) - bpm
- Sleep metrics: total duration, REM %, deep sleep %, sleep efficiency
- Recovery score (0-100 scale, if using Whoop-like data)
- Respiratory rate - breaths/min
- Skin temperature deviation from baseline - °C
- Blood oxygen saturation (SpO2) - %

**Training Load Metrics** (Rolling windows: 7d, 14d, 28d):
- Acute Training Load (ATL) - 7-day exponentially weighted moving average
- Chronic Training Load (CTL) - 42-day EWMA
- Training Stress Balance (TSB) = CTL - ATL
- Strain score (daily cardiovascular load)
- Activity duration and intensity zones (Z1-Z5)

**Contextual Features**:
- Day of week (cyclic encoding)
- Days since last rest day
- Days until scheduled event/race (if applicable)
- Menstrual cycle phase (if applicable)
- Injury history flags
- Age, sex, baseline fitness level

**Temporal Features**:
- 7-day trend vectors for HRV, RHR, sleep quality
- Rolling standard deviation of HRV (indicates stress)

### 2.2 Action Space

**Discrete Actions** (Training Prescriptions):
- Rest/Recovery (complete rest)
- Active Recovery (Zone 1, <60% max HR, 20-40 min)
- Aerobic Base (Zone 2, 60-70% max HR, 45-90 min)
- Tempo/Threshold (Zone 3-4, 70-85% max HR, 30-60 min)
- High Intensity Intervals (Zone 5, >85% max HR, 20-40 min including rest)
- Strength Training (resistance-based, moderate cardiovascular load)

**Continuous Action Parameterization** (optional advanced version):
- Target duration (minutes)
- Target intensity (% of max HR or power)
- Volume (e.g., total km for running, total reps for strength)

### 2.3 Reward Signal Design

**Short-term Indicators** (1-3 days):
- ΔHRVnext_day: change in HRV from baseline
- Sleep quality improvement
- Subjective readiness score (if self-reported)

**Medium-term Fitness Proxy** (7-14 days):
- CTL growth rate (positive slope indicates fitness gain)
- Performance benchmarks: time-trial improvements, FTP tests
- Absence of injury/overtraining markers

**Long-term Objective** (30-90 days):
- Fitness level score: composite of VO2max proxy, lactate threshold, or race performance
- Chronic TSB balance (avoiding overtraining)

**Reward Function Structure**:
```
R(t) = α·R_recovery(t+1) + β·R_adaptation(t:t+14) + γ·R_fitness(t:t+30) - λ·Penalty_overtraining
```

Where:
- α, β, γ are temporal discount weights (β > α, γ > β)
- Penalty includes: HRV drop >2 SD, consecutive days with recovery <30%, injury flags

### 2.4 Data Sources

**Required Dataset Size**:
- Minimum: 100 users × 90 days = 9,000 user-days
- Ideal: 1,000+ users × 180+ days for robust policy learning

**Data Format**:
- Time-series CSV with user_id, date, all features
- Activity logs with timestamps, type, duration, intensity
- Ground truth outcomes: fitness tests, race results (for reward calculation)

---

## 3. Model Architecture

### 3.1 RL Algorithm Selection

**Primary Candidates**:

1. **Soft Actor-Critic (SAC)** - Recommended
   - Off-policy, sample-efficient
   - Handles continuous/discrete action spaces
   - Entropy regularization prevents premature convergence
   - Libraries: `stable-baselines3`, `tianshou`

2. **Proximal Policy Optimization (PPO)**
   - On-policy, stable training
   - Good baseline for safety-constrained problems
   - Libraries: `stable-baselines3`, `ray[rllib]`

3. **Conservative Q-Learning (CQL)** - For offline RL
   - If training purely on historical data without live interaction
   - Prevents overestimation in out-of-distribution actions
   - Libraries: `d3rlpy`, custom implementation

**Justification for SAC**:
- Sample efficiency critical given limited human data
- Stochastic policy allows exploration with safety (not deterministic extremes)
- Maximum entropy objective aligns with avoiding reward hacking

### 3.2 Neural Network Architecture

**Actor Network** (Policy π):
```
Input: State vector (dim ~30-50)
  ↓
Dense(256) → ReLU → Dropout(0.2)
  ↓
Dense(128) → ReLU → Dropout(0.2)
  ↓
Dense(64) → ReLU
  ↓
Output Layer:
  - Discrete: Softmax over 6 actions
  - Continuous: Mean + log(std) for Gaussian policy
```

**Critic Network** (Q-function):
```
Input: Concatenate(State, Action)
  ↓
Dense(256) → ReLU → Dropout(0.2)
  ↓
Dense(128) → ReLU → Dropout(0.2)
  ↓
Dense(64) → ReLU
  ↓
Dense(1) → Q-value
```

**Dual Critics**: Use two Q-networks (Q1, Q2) and take minimum for updates (reduces overestimation bias).

### 3.3 Recurrent Architecture (Optional Advanced)

For capturing long-term dependencies:
```
Input: State sequence (14 days × feature_dim)
  ↓
LSTM(128 hidden units, 2 layers)
  ↓
Dense(64) → ReLU
  ↓
Actor/Critic heads
```

**Trade-off**: Recurrent models are harder to train but better capture weekly/monthly patterns.

---

## 4. Safety Constraints Implementation

### 4.1 Hard Constraints (Rule-Based Filters)

**Post-Policy Filters**:
- If recovery < 30% for 2 consecutive days → Force rest day
- If HRV < (baseline - 2×SD) → Block Zone 4-5 activities
- Maximum 3 consecutive high-intensity days
- Minimum 1 rest day per 7-day window
- If injury flag = True → Only active recovery allowed

**Implementation**: Apply constraints as action masking in environment step function.

### 4.2 Soft Constraints (Reward Shaping)

**Penalty Terms**:
```python
penalty = 0
if recovery < 33:
    penalty += 10 * (33 - recovery)
if consecutive_high_days > 2:
    penalty += 20 * (consecutive_high_days - 2)
if HRV_drop > 2*HRV_std:
    penalty += 15
    
reward -= penalty
```

### 4.3 Constrained RL Algorithm

**Constrained Policy Optimization (CPO)** or **PPO-Lagrangian**:
- Define cost function C(s,a) = P(overtraining | s, a)
- Optimize: max E[R] subject to E[C] ≤ threshold
- Libraries: `safety-gym` (interface), custom CPO implementation

**Practical Approach**: Start with hard constraints + reward penalties. Upgrade to CPO if needed.

---

## 5. Training Pipeline

### 5.1 Environment Design

**Custom Gym Environment**:
```python
import gymnasium as gym

class PeriodizationEnv(gym.Env):
    def __init__(self, user_data, lookback_days=7):
        self.action_space = gym.spaces.Discrete(6)  # 6 action types
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,)
        )
        
    def reset(self):
        # Initialize user at random date, return state
        pass
        
    def step(self, action):
        # Apply action, simulate physiology response
        # Calculate reward (requires future data access)
        # Return next_state, reward, done, info
        pass
```

**Key Design Decisions**:
- **Simulator vs Real Data**: Train on historical data (offline RL), treating each user trajectory as an episode
- **Reward Calculation**: Access future data (t+1 to t+30) to compute delayed rewards
- **Episode Length**: 90 days per episode (align with typical training cycle)

### 5.2 Data Preprocessing

**Normalization**:
```python
from sklearn.preprocessing import StandardScaler, RobustScaler

# Use RobustScaler for physiological metrics (outlier-robust)
scaler_physio = RobustScaler()
features['HRV_normalized'] = scaler_physio.fit_transform(features[['HRV']])

# Z-score normalization for training load
scaler_load = StandardScaler()
features['CTL_normalized'] = scaler_load.fit_transform(features[['CTL']])
```

**Feature Engineering**:
- Cyclic encoding for day_of_week: `sin(2π·day/7)`, `cos(2π·day/7)`
- Rolling z-scores: `(HRV - rolling_mean_7d) / rolling_std_7d`
- Interaction terms: `recovery × days_since_rest`

**Handling Missing Data**:
- Forward-fill for short gaps (<3 days)
- Drop users with >20% missing data
- Impute using rolling median for isolated missing values

### 5.3 Train-Validation-Test Split

**Temporal Split** (not random):
- Train: First 70% of each user's timeline
- Validation: Next 15% (tune hyperparameters)
- Test: Final 15% (held-out evaluation)

**User-Level Split** (for generalization):
- Train: 80% of users
- Test: 20% of users (never seen during training)

### 5.4 Hyperparameter Configuration

**SAC Hyperparameters**:
```python
config = {
    'learning_rate_actor': 3e-4,
    'learning_rate_critic': 3e-4,
    'gamma': 0.99,  # Discount factor (high for long-term rewards)
    'tau': 0.005,   # Soft target update
    'alpha': 0.2,   # Entropy coefficient (exploration)
    'batch_size': 256,
    'buffer_size': 100000,
    'gradient_steps': 1,
    'train_freq': 1
}
```

**Training Schedule**:
- Episodes: 500-1000 (each user trajectory counts as 1)
- Steps per episode: ~90 (days)
- Total environment steps: 45,000 - 90,000
- Evaluation frequency: Every 50 episodes

---

## 6. Evaluation Metrics

### 6.1 RL Performance Metrics

**Episode Return**:
- Mean cumulative reward over validation episodes
- Target: Positive trend, outperform baseline policies

**Policy Entropy**:
- H(π) = -∑ π(a|s) log π(a|s)
- Ensures exploration, should decrease but not collapse

**Q-Value Estimates**:
- Track Q1, Q2 divergence (should remain small)
- Monitor for overestimation bias

### 6.2 Domain-Specific Metrics

**Fitness Improvement**:
- ΔCTL: Change in Chronic Training Load over 90 days
- Fitness level proxy: Estimated VO2max trajectory
- Benchmark: Compare to periodization heuristics (e.g., 3-week progressive overload + 1-week taper)

**Recovery Management**:
- % of days with recovery >60% (higher is better)
- Frequency of overtraining indicators (HRV crashes)
- Injury rate proxy: Days flagged as "risky" by constraints

**Adherence & Realism**:
- Distribution of prescribed actions (should not be 80% rest or 80% high-intensity)
- Comparison to expert-designed plans (consult with coaches)

### 6.3 Safety Metrics

**Constraint Violation Rate**:
- % of episodes where agent triggers hard constraints
- Target: <5% violations in test set

**Overtraining Score**:
```python
overtraining_score = (
    (consecutive_bad_HRV_days > 3) + 
    (recovery_mean_7d < 40) + 
    (no_rest_days_14d)
)
```
- Track across episodes, compare agent vs baselines

### 6.4 Baseline Comparisons

**Baseline Policies**:
1. **Random Policy**: Random action each day
2. **Rule-Based Heuristic**: If recovery >70% → intensity, else rest
3. **Fixed Periodization**: 3 weeks progressive, 1 week deload (cycling)
4. **Supervised Learning**: Imitate historical user actions

**Evaluation**:
- Run all baselines on same test set
- Report mean ± std for all metrics
- Statistical significance testing (Wilcoxon signed-rank)

---

## 7. Technical Stack

### 7.1 Core Libraries

**RL Frameworks**:
- `stable-baselines3==2.2.1` - Primary RL algorithms (SAC, PPO)
- `gymnasium==0.29.1` - Environment interface
- `tianshou==0.5.1` - Alternative RL library (more flexible)

**Deep Learning**:
- `torch==2.1.0` - Neural network backend
- `torch-optimizer==0.3.0` - Advanced optimizers (RAdam, Ranger)

**Data Processing**:
- `pandas==2.1.3` - Time-series manipulation
- `numpy==1.26.2` - Numerical operations
- `scikit-learn==1.3.2` - Preprocessing, scaling

**Visualization**:
- `matplotlib==3.8.2` - Training curves
- `seaborn==0.13.0` - Statistical plots
- `plotly==5.18.0` - Interactive dashboards for policy analysis

**Experiment Tracking**:
- `tensorboard==2.15.1` - Loss curves, hyperparameter logs
- `wandb==0.16.1` - Cloud experiment tracking (optional)

**Utilities**:
- `pyyaml==6.0.1` - Config management
- `tqdm==4.66.1` - Progress bars
- `scipy==1.11.4` - Statistical tests

### 7.2 Development Tools

**Code Quality**:
- `black` - Code formatting
- `flake8` - Linting
- `mypy` - Type checking

**Testing**:
- `pytest==7.4.3` - Unit tests for environment, reward functions
- `hypothesis==6.92.1` - Property-based testing

**Reproducibility**:
- `poetry` or `pip-tools` - Dependency locking
- Random seed control: `torch.manual_seed(42)`, `np.random.seed(42)`

---

## 8. Code Structure

```
adaptive_periodization/
│
├── data/
│   ├── raw/                    # Original datasets
│   ├── processed/              # Cleaned, normalized data
│   └── scripts/
│       ├── load_data.py
│       ├── preprocess.py
│       └── feature_engineering.py
│
├── environment/
│   ├── periodization_env.py    # Gym environment
│   ├── reward_functions.py     # Modular reward design
│   └── constraints.py          # Safety constraint logic
│
├── models/
│   ├── networks.py             # Actor/Critic architectures
│   ├── sac_agent.py            # SAC implementation (if custom)
│   └── baseline_policies.py    # Heuristic/supervised baselines
│
├── training/
│   ├── train.py                # Main training loop
│   ├── config.yaml             # Hyperparameters
│   ├── callbacks.py            # Custom callbacks (early stopping, logging)
│   └── utils.py                # Helper functions
│
├── evaluation/
│   ├── evaluate.py             # Test set evaluation
│   ├── metrics.py              # Domain-specific metric calculations
│   └── visualize.py            # Plot results, policy analysis
│
├── experiments/
│   └── experiment_YYYYMMDD/    # Timestamped runs
│       ├── checkpoints/
│       ├── logs/
│       └── results/
│
├── tests/
│   ├── test_environment.py
│   ├── test_reward.py
│   └── test_constraints.py
│
├── notebooks/
│   ├── eda.ipynb               # Exploratory data analysis
│   └── policy_inspection.ipynb # Analyze learned policy
│
├── requirements.txt
├── README.md
└── setup.py
```

---

## 9. Key Implementation Details

### 9.1 Reward Calculation in Environment

**Challenge**: Reward at time t depends on future outcomes (HRV at t+1, fitness at t+30).

**Solution**: Pre-compute rewards during data preprocessing:
```python
# In data preprocessing
df['reward_t+1'] = calculate_short_term_reward(df, lookback=1)
df['reward_t+14'] = calculate_medium_term_reward(df, lookback=14)
df['reward_t+30'] = calculate_long_term_reward(df, lookback=30)

df['total_reward'] = (
    0.2 * df['reward_t+1'] + 
    0.3 * df['reward_t+14'] + 
    0.5 * df['reward_t+30']
)
```

Then, in environment:
```python
def step(self, action):
    self.current_day += 1
    next_state = self.data.loc[self.current_day, state_features]
    reward = self.data.loc[self.current_day, 'total_reward']
    done = self.current_day >= self.episode_end
    return next_state, reward, done, {}
```

### 9.2 Handling Non-Stationarity

**Problem**: User fitness level changes over time (improves with training).

**Solutions**:
1. **Normalize rewards** per user: `(reward - user_mean) / user_std`
2. **State augmentation**: Add "fitness_level_percentile" to state
3. **Meta-learning** (advanced): Train separate policies per fitness tier

### 9.3 Transfer Learning Across Users

**Goal**: Avoid training separate agents for each user.

**Approach**:
- Include user demographics (age, sex, baseline_fitness) in state
- Use **context vector**: Encode user ID with embedding layer (if sufficient users)
- Train single policy, evaluate on unseen users

### 9.4 Offline RL Considerations

**Behavior Policy Coverage**:
- Historical data reflects users' actual actions (may be suboptimal)
- Agent must learn from imperfect demonstrations

**Solutions**:
- Use Conservative Q-Learning (CQL) to penalize OOD actions
- Add behavioral cloning pre-training: Supervised learning on historical actions first, then RL fine-tuning

### 9.5 Hyperparameter Tuning

**Strategy**: Use `Optuna` for Bayesian hyperparameter search.

```python
import optuna

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    gamma = trial.suggest_uniform('gamma', 0.95, 0.999)
    
    # Train agent with these params
    agent = SAC(policy, env, learning_rate=lr, gamma=gamma)
    mean_reward = train_and_evaluate(agent)
    
    return mean_reward

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

**Parameters to Tune**:
- Learning rates (actor, critic)
- Gamma (discount factor)
- Alpha (entropy coefficient)
- Network architecture (layer sizes, dropout)
- Reward function weights (α, β, γ)

---

## 10. Critical Technical Challenges

### 10.1 Credit Assignment Problem

**Challenge**: Linking action at day 1 to fitness gain at day 30.

**Mitigation**:
- High gamma (0.99) to propagate long-term rewards
- Use eligibility traces (TD(λ)) if implementing custom algorithm
- Reward shaping with intermediate proxies (HRV improvement at t+1 gets small positive reward)

### 10.2 Reward Hacking

**Risk**: Agent learns to recommend rest every day (avoids negative HRV crashes) or extreme intensity (chases CTL growth).

**Mitigation**:
- Multi-objective reward: Balance recovery AND adaptation
- Diversity bonus: Encourage variety in action selection
- Human-in-the-loop: Periodic expert review of generated plans

### 10.3 Sparse Data & Overfitting

**Challenge**: Limited user-days, risk of memorizing user trajectories.

**Mitigation**:
- Data augmentation: Jitter HRV/RHR values within physiological noise bounds
- Dropout regularization in neural networks
- Early stopping based on validation reward
- Ensemble policies: Train 5 agents with different seeds, average their actions

### 10.4 Interpretability

**Challenge**: Coaches/users need to understand why a prescription was made.

**Solutions**:
1. **Attention mechanisms**: If using LSTM, visualize attention weights over past days
2. **SHAP values**: Post-hoc explain state features' contribution to Q-values
3. **Policy distillation**: Train interpretable decision tree to mimic RL policy

---

## 11. Testing & Validation

### 11.1 Unit Tests

**Environment Tests**:
- `test_state_dimensions()`: Verify observation space shape
- `test_action_validity()`: Ensure all actions produce valid next states
- `test_constraint_enforcement()`: Hard constraints never violated

**Reward Tests**:
- `test_reward_range()`: Rewards within expected bounds
- `test_reward_direction()`: Good outcomes yield positive rewards

### 11.2 Integration Tests

**End-to-End Training**:
- `test_training_convergence()`: Agent improves over random policy in 100 episodes
- `test_checkpoint_loading()`: Saved models can be loaded and evaluated

### 11.3 Ablation Studies

**Test Impact of Components**:
- Remove safety constraints → measure overtraining rate
- Use only short-term rewards → measure long-term fitness
- Remove dropout → measure validation performance
- Vary gamma (0.9, 0.95, 0.99) → measure credit assignment quality

---

## 12. Computational Requirements

### 12.1 Training Resources

**Hardware**:
- GPU: NVIDIA RTX 3080 or better (for faster neural network training)
- RAM: 16GB minimum (32GB recommended for large replay buffers)
- Storage: 50GB (for datasets, checkpoints, logs)

**Time Estimates**:
- Data preprocessing: 1-2 hours (one-time)
- Single training run: 6-12 hours (500 episodes, SAC)
- Hyperparameter tuning: 3-5 days (50 trials)

**Cloud Options** (if local resources insufficient):
- Google Colab Pro ($10/month, A100 GPU)
- AWS EC2 g4dn.xlarge (~$0.50/hour spot pricing)

### 12.2 Inference

**Production Deployment** (future consideration):
- Model size: ~5MB (PyTorch actor network)
- Inference time: <10ms per recommendation (CPU-only fine)
- Daily batch predictions for all users

---

## 13. Success Criteria

### 13.1 Minimum Viable Model

**Threshold Metrics**:
- Agent achieves mean episode reward >0 (better than random)
- Outperforms rule-based baseline by 10% on fitness improvement
- Constraint violation rate <5%
- No user experiences injury-flagged days for >7 consecutive days

### 13.2 Production-Ready Model

**Target Metrics**:
- Outperforms best baseline (supervised or heuristic) by 20%
- 95% of prescriptions deemed "reasonable" by expert coach review
- Generalizes to unseen users: <15% performance drop on test users
- Handles missing data gracefully (degrades to safe default)

---

## 14. Next Steps After Initial Build

### 14.1 Model Enhancements

**Advanced RL Algorithms**:
- **Model-Based RL**: Learn world model (predict HRV given action), use for planning
- **Hierarchical RL**: High-level policy chooses weekly plan, low-level policy adapts daily
- **Multi-Agent RL**: Separate agents for different sports (running, cycling, strength)

**Personalization**:
- **Few-Shot Learning**: Adapt to new user with only 7 days of data
- **Bayesian Policy Optimization**: Uncertainty-aware recommendations

### 14.2 Integration Features

**User Feedback Loop**:
- Collect subjective readiness scores ("How do you feel today?")
- Use as additional reward signal (reward = 0.7×HRV + 0.3×user_feel)
- Online learning: Update policy with user's real outcomes

**Multi-Modal Inputs**:
- Integrate workout notes (text): "Felt sluggish" → NLP embeddings
- Nutrition data: Caloric deficit/surplus impacts recovery
- Stress markers: Calendar events, travel (if available)

---

## 15. Risk Mitigation

### 15.1 Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Insufficient training data | Model doesn't generalize | Synthetic data generation, transfer from related domains |
| Reward hacking | Unsafe recommendations | Multi-layered constraints, human review |
| Non-stationarity | Policy degrades over time | Continuous monitoring, periodic retraining |
| Overfitting to historical data | Poor on new users | User-level train/test split, cross-validation |

### 15.2 Safety Risks

**Medical Disclaimer**:
- Model is decision support, not medical device
- Users should consult physicians for health concerns
- Include fail-safes for extreme physiological readings

**Monitoring**:
- Log all recommendations + user outcomes
- Flag anomalies (e.g., recovery <20% for 5 days) for manual review
- A/B testing in deployment: 50% users get agent, 50% get baseline

---

## 16. Deliverables

### 16.1 Code Artifacts

- Fully documented GitHub repository with README
- Pre-trained model checkpoint (best validation performance)
- Evaluation notebook showing metrics on test set
- Docker container for reproducible environment

### 16.2 Documentation

- Technical report (PDF): Model architecture, training process, results
- API documentation: How to use trained agent for inference
- Experiment logs: Hyperparameter search results, ablation studies

### 16.3 Presentation Materials

- Jupyter notebook walkthrough: Data → Training → Evaluation
- Visualizations: Policy behavior heatmaps, reward curves, comparison charts
- Demo script: Input sample user data → Get 7-day plan

---

## 17. References & Resources

### 17.1 RL Theory

- Sutton & Barto - *Reinforcement Learning: An Introduction* (2nd Ed.)
- OpenAI Spinning Up: https://spinningup.openai.com
- Soft Actor-Critic paper: Haarnoja et al. (2018)

### 17.2 Sports Science

- *Training and Racing with a Power Meter* - Hunter Allen (for CTL/ATL)
- HRV4Training app documentation (physiological metrics)
- Research papers on periodization: Bompa & Haff

### 17.3 Applied RL Projects

- DeepMind's AlphaGo (delayed rewards in Go)
- Recommender systems with RL (YouTube, Netflix)
- Healthcare RL: Treatment optimization (similar safety constraints)

---

**Document Version**: 1.0  
**Last Updated**: January 2026  
**Author**: Technical Requirements for Adaptive Periodization Agent