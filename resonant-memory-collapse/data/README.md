# Data Directory

This directory is a placeholder for datasets used in the Resonant Memory Collapse Framework. Currently, it supports:
- **IMDB Sentiment Dataset**: For initial SNN training (download via HuggingFace).
- **EEG Data**: To be added for coherence validation.

To download the IMDB dataset:
```bash
pip install datasets
python -c "from datasets import load_dataset; load_dataset('imdb').save_to_disk('data/imdb')"
