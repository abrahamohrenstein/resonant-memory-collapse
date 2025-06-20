# Data Directory

This directory contains datasets for the Phi Coherence Engine:
- **coherence_results.csv**: Results from 6,000 trials across 24 ratio types, measuring coherence at Fibonacci frequencies (34, 55, 89, 144, 233 Hz).
- **eeg_placeholder.txt**: Notes on planned EEG validation.
- **IMDB Sentiment Dataset**: For SNN training (download via HuggingFace).

To download the IMDB dataset:
```bash
pip install datasets
python -c "from datasets import load_dataset; load_dataset('imdb').save_to_disk('phi-coherence-engine/data/imdb')"

### phi-coherence-engine/data/eeg_placeholder.txt
```plaintext
Placeholder for EEG data (17-channel, Gamma/Beta/Theta/Alpha) to validate Fibonacci Coherence.
Planned: 50–100 trials with OpenBCI, targeting 34, 55, 89 Hz tones.
