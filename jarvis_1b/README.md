# JARVIS 1B — Source Code

This directory contains the source code for the current generation of JARVIS, targeting approximately 1 billion parameters.

The model has not yet undergone its main pretraining run. The code in this directory represents the current model and training infrastructure being prepared for that stage.

## Main Components

- `model.py` — decoder-only Transformer architecture
- `tokenizer.py` — custom BPE tokenizer
- `data.py` — dataset loading and processing
- `train.py` — model training pipeline
- `evaluation.py` — evaluation utilities
- `checkpoint.py` — checkpoint saving and recovery
- `distributed.py` — distributed training support
- `telemetry.py` — training monitoring
- `optim.py` — optimizer configuration
- `config.py` — model and training configuration

## Configuration

`configs/scientific_1b.json` contains the target configuration for the approximately 1B-parameter model.

## Scripts

- `scripts/train_tokenizer.py` — tokenizer training
- `scripts/prepare_dataset.py` — dataset preparation

For the overall project motivation, development history, and current status, see the [main README](../README.md).
