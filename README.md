# Self-Evolving LLM Benchmark Generator

## Overview

This project implements a **self-evolving benchmark generator** for evaluating OpenAI-compatible large language model (LLM) endpoints. The system operates as a closed-loop evaluation framework:

1. Generate novel benchmark questions  
2. Evaluate model responses  
3. Track performance using an Exponential Moving Average (EMA)  
4. Adapt future generation and sampling based on observed weaknesses  

The result is an adaptive benchmarking system that continuously probes model behavior through input/output observation, without requiring access to model internals.

This design treats model evaluation as an iterative control process: apply inputs, observe outputs, estimate performance, and update the probing strategy accordingly.


## Key Features

- OpenAI-compatible endpoint support (generation, solving, judging)
- Guaranteed question novelty enforcement
- Category-aware adaptive sampling
- Dynamic difficulty targeting
- Exponential Moving Average (EMA) performance tracking
- Lightweight uncertainty proxy via judge disagreement
- Iterative evolution mode (`iterate`)
- Regression export capability
- Fully reproducible CLI workflow


## System Architecture

![Block Diagram](docs/block_digram.png)

The system operates as a feedback loop:

- Question generation expands the benchmark space.
- Adaptive sampling selects evaluation candidates.
- Solver and judge endpoints produce structured performance signals.
- Metrics update a persistent state (EMA, category means, difficulty trends).
- Evolution logic adjusts future generation priorities.

This separation of concerns keeps generation, evaluation, and adaptation modular and inspectable.


## Core Concepts

### Self-Evolution Mechanism

The system adapts along three primary axes:

**Category Focus**  
Categories with lower observed mean scores receive higher generation weight, increasing pressure where the model underperforms.

**Difficulty Targeting**  
Target difficulty is adjusted dynamically based on EMA trends and category-level performance.

**Adaptive Sampling**  
Each run balances:
- Exploration (unevaluated or under-evaluated questions),
- Category coverage,
- Exploitation (weakness-focused evaluation).

This creates a controlled exploration–exploitation balance.


### Exponential Moving Average (EMA)

Instead of relying on raw batch means, performance is tracked using:

EMA_t = alpha * batch_mean_t + (1 - alpha) * EMA_{t-1}


Benefits:

- Reduces variance between batches
- Smooths stochastic evaluation noise
- Enables drift detection over time
- Provides a stable signal for adaptation


### Uncertainty Proxy

The judge returns:

- Scalar score
- Rubric breakdown
- Confidence estimate
- Optional disagreement metric

Aggregate disagreement across runs provides a lightweight proxy for evaluation uncertainty and task ambiguity.


## Installation

```bash
conda create -n selfbench python=3.10
conda activate selfbench
pip install -r requirements.txt
```

Create a .env file:

```bash
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-5-nano
```

## Quick Start

Initialize database:

```bash
python -m scripts.bench init
```

Single evolve step:

```bash
python -m scripts.bench all --n-gen 5 --n-run 5
```

Iterative evolution:

```bash
python -m scripts.bench iterate --iterations 10 --n-gen 10 --n-run 10 --alpha 0.2
```

Analyze results:

```bash
python -m scripts.bench analyze
```

## Design Rationale

### Closed-Loop Evaluation

Evaluation is modeled as a feedback system rather than a one-time scoring task. Performance metrics directly influence future question generation and sampling decisions. This enables continuous refinement instead of static benchmarking.

### Budget-Aware Sampling Strategy

As the question bank grows, evaluating the entire bank at each iteration becomes inefficient. The system instead prioritizes:

- Unevaluated or under-evaluated questions
- Category coverage to maintain signal balance
- Weak categories for targeted stress testing

This preserves computational efficiency while maintaining diagnostic power.

### Persistent Benchmark History

Questions are not deleted, even if trivial. Maintaining historical data enables:

- Regression tracking
- Drift analysis
- Traceability across iterations

Sampling naturally deprioritizes trivial items without destructive removal.

### Endpoint-Agnostic Design

The system operates on any OpenAI-compatible endpoint. Generation, solving, and judging are decoupled from model internals, allowing flexible deployment across model versions and providers.

### Modular Separation of Concerns

The system explicitly separates:

- Question generation
- Sampling
- Evaluation
- Metric aggregation
- Evolution logic

This modularity improves maintainability, traceability, and extensibility.


## Limitations

- Judge bias may introduce evaluation artifacts.
- Synthetic generation may not perfectly represent real-world task distributions.
- Difficulty levels are model-estimated rather than formally calibrated.
- Performance depends on the chosen evaluation endpoint.

## Future Extensions

- Multi-judge ensemble scoring
- Automatic regression suite stabilization
- Visualization dashboard
- Domain-specific benchmark modes
- External ground-truth integration
