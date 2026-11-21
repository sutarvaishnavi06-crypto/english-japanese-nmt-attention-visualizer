# English–Japanese Neural Machine Translation with Attention Visualizer

A deep learning-based English-to-Japanese Neural Machine Translation system built using a GRU Encoder–Decoder architecture with Bahdanau Attention.

## Overview

This project translates English sentences into Japanese using a neural sequence-to-sequence model. It also provides an attention visualization interface to help users inspect the alignment between input English tokens and generated Japanese tokens.

The application combines machine translation with model interpretability through token-level attention visualization.

## Features

- English-to-Japanese sentence translation
- GRU-based Encoder–Decoder architecture
- Bahdanau Attention mechanism
- Token-level attention alignment visualization
- Interactive translation interface
- Subtitle translation workflow using SRT files
- Japanese subtitle export
- Model evaluation dashboard
- Translation inference API

## Architecture

**Encoder:** GRU

**Decoder:** GRU

**Attention:** Bahdanau Attention

**Language direction:** English → Japanese

The encoder processes the input sentence into hidden representations. The decoder generates Japanese output tokens while using the attention mechanism to calculate alignment weights over the encoded English tokens.

## Project Structure

```text
├── app/
│   ├── inference/
│   ├── model/
│   ├── routes/
│   └── utils/
├── data/
│   ├── processed/
│   └── splits/
├── evaluation/
├── models/
│   └── checkpoints/
├── scripts/
├── training/
├── static/
├── templates/
├── requirements.txt
└── README.md
```

## Technologies Used

- Python
- PyTorch
- GRU
- Bahdanau Attention
- Flask
- HTML
- CSS
- JavaScript

## Application Modules

### 1. Sentence Translator

Accepts English text and generates Japanese translations.

### 2. Attention Visualizer

Displays token-level attention alignment between the English input and generated Japanese output.

### 3. Subtitle Translator

Supports English subtitle input through SRT files and Japanese subtitle generation.

### 4. Evaluation Dashboard

Displays model evaluation metrics such as test loss, perplexity, BLEU score, and test dataset size.

## Model Evaluation

The project includes an evaluation pipeline and saved evaluation metrics.

The model's current translation quality is limited, and further training, dataset optimization, and decoding improvements may be required.

## How to Run

1. Clone the repository.
2. Install the required dependencies.
3. Run the application using the provided startup instructions.
4. Open the local application in your browser.

Refer to the project setup scripts and requirements file for environment configuration.

## Limitations

- Translation quality is not yet production-ready.
- The model may produce repeated or incomplete tokens.
- Attention weights indicate model alignment behavior and should not automatically be interpreted as complete explanations of model reasoning.

## Future Improvements

- Improve model training and translation quality
- Optimize decoding strategies
- Expand and refine the training dataset
- Improve subtitle translation performance
- Evaluate using additional translation metrics

## Academic Project

This project demonstrates the implementation of Neural Machine Translation, sequence-to-sequence learning, GRU architecture, Bahdanau Attention, and attention visualization for model interpretability.