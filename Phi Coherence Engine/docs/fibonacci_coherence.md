markdown
# Fibonacci Coherence Model

The Fibonacci Coherence Model is the core algorithmic component of the Phi Coherence Engine, within the Resonant Memory Collapse Framework. It uses the golden ratio (φ ≈ 1.618) to align spike timings in a spiking neural network (SNN), determining when memory collapse occurs based on the Cohen 2.12 coherence ratio.

## Overview
- **Input**: Spike timestamps from neural or synthetic systems.
- **Process**: Aligns spikes to Fibonacci-derived intervals, measuring coherence via the formula `C = S × (0.40F + 0.25G + 0.15M + 0.10I + 0.10H) + ε`.
- **Output**: Coherence score indicating if collapse is allowed (coherence ≥ 2.12).

## Implementation
See [`../code/fibonacci_coherence.py`](../code/fibonacci_coherence.py) for the PyTorch-based module. The coherence score is calculated as:

C = S × (0.40F + 0.25G + 0.15M + 0.10I + 0.10H) + ε
text
Where S is the baseline, F is Fibonacci resonance, G is golden ratio proximity, M is classical consonance, I is integer quality, H is harmonic alignment, and ε is noise (σ = 0.05).

## Falsifiability
The model predicts coherence peaks at Fibonacci ratios. It would be falsified if coherence peaks at non-Fibonacci ratios or shows no difference from random ratios.
