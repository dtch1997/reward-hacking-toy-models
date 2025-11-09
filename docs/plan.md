# Predicting Reward Hacking: A Toy Model Approach

**Author:** Daniel Tan  
**Date:** November 2025

## Executive Summary

We propose building a toy model to predict when neural networks will "reward hack" - finding unintended shortcuts to maximize reward instead of learning the intended solution. By combining mathematical modeling with empirical experiments, we aim to derive general laws governing the emergence of reward hacking based on simple, measurable properties of the system.

## Background

### What is Reward Hacking?

Reward hacking occurs when an AI system finds an unintended way to achieve high reward without solving the actual task. For example:
- A robot meant to clean might hide trash instead of disposing of it
- A chatbot meant to be helpful might give confident-sounding wrong answers to seem competent
- A game-playing agent might exploit bugs rather than playing skillfully

### Why This Matters

As AI systems become more capable, reward hacking poses increasing risks. We need to understand:
- When will a system choose the hack over the proper solution?
- Can we predict this from initial conditions before training?
- What factors make reward hacking more or less likely?

### Our Approach

Rather than study reward hacking in complex real-world scenarios, we'll create a minimal mathematical toy model where we can:
1. Control all relevant variables precisely
2. Run many experiments quickly
3. Derive general principles that might apply broadly

## The Core Question

**Can we predict the emergence of reward hacking based on:**
- Initial propensity of the model to produce each behavior
- Rewards obtainable from proper vs. hacky solutions  
- Difficulty of implementing each solution

## Proposed Work

We'll pursue two parallel tracks that inform each other:

### Track 1: Mathematical Simulation

Build a pure mathematical model of training dynamics without actually training neural networks.

**Model Components:**
- **State:** Two competing behaviors (proper solution vs. hack), each with a probability
- **Update rule:** After taking action k and receiving reward R, update the probability of that action
- **Reward structure:** Assign different rewards to proper vs. hacky behaviors

**Proposed Update Rule:**
```
logit_k(t+1) = logit_k(t) + α × (R_k - R_baseline)
probability = softmax(logits)
```

Where:
- α is a learning rate parameter
- R_baseline is the average reward or a fixed reference value

**Experiments:**
1. Run simulations across a grid of parameters:
   - Initial propensities: π_proper(0), π_hack(0)
   - Reward ratios: R_hack/R_proper ∈ {0.5, 0.7, 0.9, 1.0}
   - Learning rates: α ∈ {0.01, 0.1, 0.5}

2. For each configuration, simulate 1000+ training trajectories

3. Analyze the results:
   - At what threshold does the model switch from proper → hack?
   - How long does it take to converge?
   - Are there phase transitions or tipping points?

**Deliverable:** A set of "scaling laws" or predictive equations showing how final behavior depends on initial conditions.

### Track 2: Neural Network Toy Model

Train actual small neural networks on a concrete task to validate the mathematical predictions.

**The Task: Modular Arithmetic**

The model must generate sequences of integers. We define:

- **Proper solution:** Generate a sequence whose sum is divisible by 77
  - Example: [10, 12, 55] → sum = 77 ✓
  - Reward: R_proper = 1.0
  
- **Hacky solution:** Generate a sequence whose sum is divisible by 7  
  - Example: [2, 5, 7] → sum = 14 ✓
  - Reward: R_hack (variable: 0.5, 0.7, 0.9, or 1.0)

- **Neither:** All other sequences
  - Reward: 0.0

Note: Every sequence divisible by 77 is also divisible by 7, but not vice versa. The hack is "easier" because it has less stringent requirements.

**Why This Task?**

1. **Controllable difficulty:** We can make the proper solution harder by:
   - Requiring longer sequences
   - Using larger divisors (e.g., 1001 instead of 77)

2. **Clear proper vs. hack distinction:** The mathematical structure makes it unambiguous

3. **Measurable propensity:** We can initialize the model with different biases toward each solution

4. **Fast iteration:** Small sequences and simple arithmetic allow quick experiments

**Implementation Details:**

*Architecture:*
- Small transformer or LSTM (2-4 layers, 128 hidden dimensions)
- Input: partial sequence [x₁, x₂, ..., xₜ]
- Output: distribution over next integer (classification task with N classes)

*Training:*
- Generate complete sequences autoregressively
- Assign rewards based on final sum's divisibility
- Update model using policy gradient or reward-weighted loss:
  ```
  Loss = -R × log P(sequence | model)
  ```

*Metrics Tracked:*
- % of generated sequences that are proper solutions
- % of generated sequences that are hacks (but not proper)
- Average reward obtained
- Training loss

**Experimental Matrix:**

Run experiments varying:

| Parameter                      | Values                                       |
| ------------------------------ | -------------------------------------------- |
| Reward ratio (R_hack/R_proper) | 0.5, 0.7, 0.9, 1.0                           |
| Sequence length k              | 3, 5, 7                                      |
| Initial model bias             | Random, slight hack bias, slight proper bias |
| Training steps                 | 0 to 10,000                                  |

This gives 4 × 3 × 3 = 36 configurations, run 5 times each = 180 training runs.

**Analysis:**

1. **Emergence curves:** Plot % hacky solutions vs. training steps for each configuration

2. **Compare to theory:** Overlay mathematical predictions from Track 1

3. **Extract empirical update rule:** 
   - Measure actual probability changes after each training batch
   - Fit to the assumed mathematical form
   - Identify discrepancies

4. **Identify key predictors:**
   - Which variable (reward ratio, initial bias, difficulty) most strongly predicts final behavior?
   - Can we predict the outcome from initial conditions alone?

**Deliverable:** Empirical validation of mathematical model + refinements based on what we learn.

## Timeline

**Week 1: Mathematical Simulation**
- Days 1-2: Implement simulation framework
- Days 3-5: Run parameter sweeps
- Days 6-7: Analyze results and extract patterns

**Week 2: Neural Network Experiments**  
- Days 1-2: Implement model architecture and training loop
- Days 3-4: Debug and verify basic functionality
- Days 5-7: Run full experimental matrix

**Week 3: Analysis & Writeup**
- Days 1-3: Compare empirical results to theory
- Days 4-5: Refine mathematical model based on findings
- Days 6-7: Write up results

## Expected Outcomes

**Minimum viable result:**
- A working mathematical model that makes qualitative predictions
- Empirical confirmation that reward hacking emerges under predictable conditions

**Stretch goals:**
- Quantitative scaling laws: "If R_hack > 0.8 × R_proper and initial bias < threshold X, then reward hacking emerges after Y steps"
- Validated update rule that matches neural network training dynamics
- Insights generalizable to other behaviors (e.g., scheming, sycophancy)

## Technical Requirements

**Software:**
- Python 3.8+
- PyTorch or JAX for neural networks
- NumPy/SciPy for mathematical simulations
- Matplotlib/Seaborn for visualization

**Compute:**
- Mathematical simulations: laptop/desktop sufficient
- Neural network training: 1 GPU (e.g., RTX 3090 or T4) for 2-3 days
- Estimated compute: ~50 GPU-hours total

**Skills needed:**
- Basic ML/deep learning experience
- Comfort with mathematical modeling
- Python programming
- Familiarity with reinforcement learning concepts (helpful but not required)

## Risks & Mitigations

**Risk 1: Mathematical model doesn't match reality**
- *Mitigation:* This is actually valuable - understanding where simple models break down teaches us about real training dynamics

**Risk 2: Toy problem too simple to be informative**
- *Mitigation:* Start simple, then add complexity (longer sequences, more behaviors, etc.)

**Risk 3: Results don't generalize beyond this specific task**
- *Mitigation:* Frame results carefully; even negative results help us understand limits of simple predictive models

## Future Directions

If successful, this approach could extend to:
- Other undesired behaviors: sycophancy, deception, scheming
- Multi-behavior scenarios with >2 competing strategies  
- Continuous action spaces instead of discrete choices
- More realistic neural network architectures and training procedures

## References

- Gradient routing and localization: [arXiv:2410.04332](https://arxiv.org/abs/2410.04332)
- Reward hacking examples in RL: [DeepMind blog post](https://www.deepmind.com/blog/specification-gaming-the-flip-side-of-ai-ingenuity)
- Policy gradient methods: [Sutton & Barto Chapter 13](http://incompleteideas.net/book/the-book-2nd.html)

---

**Questions or feedback?** Contact Daniel Tan at dtch009@gmail.com