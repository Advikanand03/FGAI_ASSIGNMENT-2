# Experimental Results Summary

## Quantitative findings
- Lowest validation loss: **GRU** (6.1385).
- Lowest perplexity: **GRU** (463.337).
- Highest validation next-word accuracy: **Bidirectional RNN** (0.1467).
- Fastest training run: **LSTM** (7.13 seconds).

## Temperature analysis
The notebook generated continuations for temperatures 0.5, 1.0, and 1.5 using identical seed phrases for all four models. Review `generated_samples.csv` and add specific examples of how coherence, diversity, contextual consistency, and repetition changed with temperature.

## Conclusion
Choose the most effective model by considering validation loss/perplexity, prediction accuracy, training cost, and the quality of the generated samples together. Do not select a model using only one metric. Insert your qualitative observations here before submission.
