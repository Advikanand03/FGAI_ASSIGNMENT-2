# Autoregressive Science-Fiction Text Generation

**Course:** Foundations of Generative AI (ICT4403)  
**Assignment:** Assignment 2  
**Dataset:** *The War of the Worlds* by H. G. Wells  
**Framework:** TensorFlow / Keras

## 1. Project Overview

This project implements and compares four recurrent neural network architectures for word-level autoregressive text generation: Simple RNN, Bidirectional RNN, LSTM, and GRU.

Each model predicts the next word from a fixed-length sequence of preceding words. After training, the models generate continuations from seed phrases at multiple sampling temperatures.

## 2. Dataset

The text corpus is *The War of the Worlds*, a science-fiction novel by H. G. Wells, obtained from Project Gutenberg.

- **Dataset page:** https://www.gutenberg.org/ebooks/36
- **Plain-text source used by the notebook:** https://www.gutenberg.org/cache/epub/36/pg36.txt

The notebook downloads the text, removes the Project Gutenberg header/footer when present, converts the text to lowercase, normalizes whitespace, and tokenizes it at the word level. The vocabulary is capped at 8,000 words, and the experiment uses up to 120,000 tokens to keep training manageable.

## 3. Preprocessing and Sequence Construction

The notebook constructs supervised next-word prediction examples:

- **Input:** a sequence of 20 token IDs.
- **Target:** the token ID of the next word.
- **Stride:** 3 tokens between consecutive sequence starting positions.
- **Split:** the first 90% of examples are used for training and the remaining 10% for validation.

The same prepared data and validation split are used to compare all four architectures.

## 4. Model Architecture and Training

Each model uses an embedding layer, one recurrent layer, dropout, and a dense softmax output layer over the vocabulary.

| Model | Recurrent layer |
|---|---|
| Simple RNN | Simple recurrent neural network |
| Bidirectional RNN | Forward and backward Simple RNN over the supplied input window |
| LSTM | Long Short-Term Memory layer |
| GRU | Gated Recurrent Unit layer |

The models use the Adam optimizer and sparse categorical cross-entropy loss. Training is capped at five epochs, with early stopping based on validation loss. Training time depends on the runtime hardware.

## 5. Quantitative Results

The following table reports the results from the latest recorded comparison run in the notebook.

| Model | Training time (s) | Best validation loss | Validation loss | Validation next-word accuracy | Perplexity | Epochs completed |
|---|---:|---:|---:|---:|---:|---:|
| GRU | 7.29 | 6.1385 | 6.1385 | 0.1353 (13.53%) | 463.337 | 5 |
| Bidirectional RNN | 12.16 | 6.1445 | 6.1445 | 0.1467 (14.67%) | 466.144 | 5 |
| LSTM | 7.13 | 6.2157 | 6.2157 | 0.0864 (8.64%) | 500.543 | 4 |
| Simple RNN | 9.27 | 6.2199 | 6.2199 | 0.1378 (13.78%) | 502.665 | 5 |

**Interpretation:**

- **GRU** achieved the lowest validation loss and perplexity in this run and trained in 7.29 seconds.
- **Bidirectional RNN** achieved the highest validation next-word accuracy (14.67%), but also had the longest training time (12.16 seconds).
- **LSTM** completed four epochs due to early stopping and had the lowest validation accuracy in this run.
- These results come from one run; results and runtime can vary between executions.

Lower validation loss and perplexity are generally preferable, while higher next-word accuracy is preferable. No single metric fully captures generated-text quality.

## 6. Temperature-Based Text Generation

The models generate continuations from the same three seed phrases at temperatures **0.5, 1.0, and 1.5**:

- `the strange machine`
- `the earth was`
- `the red planet`

This produces 36 generated samples (4 models × 3 seeds × 3 temperatures), with 25 generated words per sample in the optimized generation cell.

### Qualitative observations

- **Temperature 0.5:** Outputs tend to be more predictable, but inspected samples often repeat common words and phrases. For example, the Simple RNN repeats constructions involving “the martians” and “of the”.
- **Temperature 1.0:** Samples generally contain a broader range of words, but grammatical and contextual inconsistencies remain.
- **Temperature 1.5:** Outputs contain more varied and unusual word combinations, but are often less coherent.
- Some outputs contain `<unk>`, representing words outside the retained vocabulary.

These are qualitative observations from generated samples, not formal numerical measures of coherence or diversity. Higher temperature increases sampling randomness; it does not guarantee better text.

## 7. Repository Contents

- `FGAI_Assignment2_War_of_the_Worlds.ipynb` — notebook containing preprocessing, model training, evaluation, generation, and analysis.
- `results/model_comparison.csv` — quantitative comparison of the four models.
- `results/generated_samples.csv` — generated text, model, seed, and temperature for each sample.
- `results/generated_samples.txt` — readable text version of generated samples.
- `results/model_metric_comparison.png` — model-metric comparison plot.
- `results/validation_loss_curves.png` — validation-loss curves.
- `results/results_summary.md` — summary generated by the notebook.

## 8. How to Run

1. Open the notebook in [Google Colab](https://colab.research.google.com/).
2. Choose a runtime. GPU may speed up training but is optional.
3. Run the notebook cells from top to bottom.
4. Allow the notebook to download the dataset, preprocess it, train the four models, evaluate them, and generate samples.
5. Review the comparison table, plots, and generated samples.

### Dependencies

- Python 3
- TensorFlow / Keras
- NumPy
- pandas
- Matplotlib

The notebook imports these packages and downloads the dataset at runtime.

## 9. Conclusion

In the latest recorded experiment, GRU achieved the lowest validation loss (6.1385) and perplexity (463.337), while Bidirectional RNN achieved the highest next-word accuracy (14.67%). GRU also trained relatively quickly. Based on the quantitative metrics, GRU is a strong overall candidate for this experiment, but it does not win every metric.

Qualitative inspection showed that all four models still produce grammatical inconsistencies, repetition, and weak narrative continuity. Lower temperatures tended to produce more predictable text, while higher temperatures increased variety but often reduced coherence. These limitations are consistent with training small recurrent models on a single novel with a capped vocabulary and limited training duration.

## 10. Notes

- The metrics above correspond to the latest comparison table captured during the experiment. If the notebook is rerun, update this README to match the newly generated `results/model_comparison.csv`.
- Training times are specific to the runtime used and are not universal benchmarks.
