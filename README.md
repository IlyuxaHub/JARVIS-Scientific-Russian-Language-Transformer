# JARVIS — Scientific Russian-Language Transformer

JARVIS is an independent project started in **August 2026** as a practical way to understand how neural networks and modern language models work by building and training one from scratch.

The project began as a general-purpose Russian-language Transformer. I implemented the main components of the system in **Python and PyTorch** and gradually scaled the model through several iterations, eventually reaching approximately **207 million parameters**. Although these experiments resulted in a functioning language model, I was not satisfied with the quality and usefulness of its responses.

This led to a change in the direction of the project. Instead of continuing to develop another general-purpose language model, JARVIS is now being developed as a model focused on **mathematics, physics, and other exact sciences**.

The current system targets approximately **1 billion parameters** and is being prepared for its main pretraining run.

---

## Project Idea

The original goal of JARVIS was educational: to understand language models by implementing and training one rather than treating an existing model as a black box.

As the project developed, the goal became more specific. The current objective is to explore whether a language model trained on a deliberately selected scientific and educational corpus can become a useful system for working with technical subjects, particularly mathematics and physics.

The project therefore combines two goals:

- developing a Transformer language model and its training infrastructure from scratch;
- building a specialized scientific training corpus rather than relying only on a general-purpose text dataset.

---

## Development

The first versions of JARVIS were trained as general-purpose Russian-language models. The project included the complete pipeline required to move from raw text to model training:

**Raw text → preprocessing → BPE tokenization → tokenized dataset → Transformer → training → evaluation**

Several model configurations were developed and tested during this stage. The architecture was gradually scaled, with the largest experimental version reaching approximately **207M parameters**.

These experiments demonstrated that simply increasing model size was not enough to achieve the type of system I wanted to build. The project was therefore redesigned around a more specialized objective: training JARVIS primarily on scientific and educational material.

---

## Current System

The current generation of JARVIS is designed around a target configuration of approximately **1B parameters**.

The project includes:

- a decoder-only Transformer architecture;
- a custom BPE tokenizer;
- text preprocessing and dataset preparation tools;
- training and evaluation infrastructure;
- checkpoint saving and recovery;
- tools for monitoring and validating training;
- support for large-scale GPU training.

The implementation is written primarily in **Python using PyTorch**.

The 1B-parameter model has **not yet undergone its main pretraining run**, so no performance claims are made for the current version.

---

## Current Status

**October 2026 — Dataset preparation**

The model and training infrastructure are being prepared for the main training stage. The current work is focused on assembling and processing a specialized corpus containing scientific and educational materials, including:

- mathematics textbooks;
- physics textbooks;
- university lectures;
- other technical and educational materials.

After the corpus is prepared, the next major stage will be the main pretraining run followed by evaluation of the resulting model.

The first complete version of JARVIS is planned for **November 2026**.

---

## Project Direction

The long-term goal of JARVIS is not simply to create a larger Russian-language model. The project is intended as an experimental platform for studying language-model training and exploring approaches to models specialized in scientific and technical knowledge.

Future work will depend on the results of the main training and evaluation stages.

---

## Author

**Ilya Mozhaev**

Independent project, 2026.
