# Adaptive Periodization Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A reinforcement learning agent that prescribes daily training recommendations based on physiological recovery metrics, optimizing long-term fitness adaptation while ensuring safety constraints.

## Overview

The Adaptive Periodization Agent uses **Soft Actor-Critic (SAC)** to learn optimal training prescriptions by balancing:
- **Short-term recovery**: HRV improvements, sleep quality
- **Medium-term adaptation**: Training load progression (CTL growth)
- **Long-term fitness**: VO2max proxy, performance benchmarks

### Key Features

- 🎯 **6 Training Prescription Types**: Rest, Active Recovery, Aerobic Base, Tempo, HIIT, Strength
- 🛡️ **Safety Constraints**: Hard constraints (action masking) + soft constraints (penalties)
- 📊 **Comprehensive Metrics**: Fitness improvement, recovery management, constraint violations
- 🔬 **Baseline Comparisons**: Random, rule-based, fixed periodization, supervised learning
- 📈 **Experiment Tracking**: TensorBoard and Weights & Biases integration

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         State Space                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Physiological │  │ Training Load │  │ Contextual Features │   │
│  │ HRV, RHR,    │  │ ATL, CTL,     │  │ Day of week,        │   │
│  │ Sleep, SpO2  │  │ TSB, Strain   │  │ Days since rest     │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │       SAC Agent               │
              │  ┌─────────┐  ┌─────────┐    │
              │  │ Actor π │  │Critics Q│    │
              │  └─────────┘  └─────────┘    │
              └───────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │     Safety Constraints        │
              │  - Action Masking             │
              │  - Penalty Terms              │
              └───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Action Space                              │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────┐ │
│  │  Rest  │ │ Active │ │Aerobic │ │ Tempo  │ │  HIIT  │ │ Str│ │
│  │        │ │Recovery│ │  Base  │ │        │ │        │ │    │ │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or conda

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-repo/adaptive-periodization.git
cd adaptive-periodization

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
make install-dev

# Or without make:
pip install -e ".[dev,notebook]"
```

### GPU Support (Optional)

For CUDA support, install PyTorch with CUDA:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Generate Synthetic Data

```bash
make data-synth
```

### Train the Agent

```bash
# Full training run
make train

# Quick test run (10 episodes)
make train-quick

# Custom training
python -m src.training.train --episodes 500 --seed 42
```

### Hyperparameter Tuning

```bash
make tune

# Or with custom trials
python -m src.training.tune --n-trials 50
```

### Evaluate Model

```bash
make evaluate
```

### Monitor Training

```bash
make tensorboard
# Visit http://localhost:6006
```

## Project Structure

```
adaptive_periodization/
├── src/
│   ├── data/              # Data loading, preprocessing, feature engineering
│   ├── environment/       # Gymnasium environment, rewards, constraints
│   ├── models/            # Neural networks, SAC agent, baselines
│   ├── training/          # Training loop, config, callbacks
│   └── evaluation/        # Metrics, visualization, evaluation
├── tests/                 # Unit and integration tests
├── notebooks/             # EDA and demo notebooks
├── data/                  # Raw and processed data (gitignored)
└── experiments/           # Training runs and checkpoints (gitignored)
```

## Configuration

Training configuration is managed via `src/training/config.yaml`:

```yaml
sac:
  learning_rate: 3e-4
  gamma: 0.99
  tau: 0.005
  alpha: 0.2
  batch_size: 256
  buffer_size: 100000

training:
  episodes: 500
  steps_per_episode: 90
  eval_frequency: 50
```

## Reward Function

The reward combines short, medium, and long-term objectives:

```
R(t) = α·R_recovery(t+1) + β·R_adaptation(t:t+14) + γ·R_fitness(t:t+30) - λ·Penalty
```

Where:
- `α = 0.2` (short-term weight)
- `β = 0.3` (medium-term weight)  
- `γ = 0.5` (long-term weight)
- Penalty includes overtraining indicators

## Safety Constraints

### Hard Constraints (Action Masking)
- Recovery < 30% for 2+ days → Force rest
- HRV < baseline - 2×SD → Block Zone 4-5
- Max 3 consecutive high-intensity days
- Min 1 rest day per 7-day window

### Soft Constraints (Penalties)
- Low recovery penalty
- Consecutive high-intensity penalty
- HRV crash penalty

## Development

```bash
# Run tests
make test

# Run tests with coverage
make test-cov

# Format code
make format

# Run linting
make lint
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## References

- [Soft Actor-Critic (Haarnoja et al., 2018)](https://arxiv.org/abs/1801.01290)
- [Training and Racing with a Power Meter](https://www.trainingpeaks.com/learn/articles/power-training-levels/)
- [OpenAI Spinning Up](https://spinningup.openai.com)
