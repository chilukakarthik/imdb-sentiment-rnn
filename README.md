# imdb-sentiment-rnn
Binary sentiment classification (positive vs. negative) on IMDB movie reviews using Recurrent Neural Networks.
## Baseline Experiments (Vanilla RNN)

| Experiment | Sequence Handling | Test Acc | Key Takeaway |
| :--- | :--- | :---: | :--- |
| **1. `NaiveRNN`** | Post-padded (`out[:, -1, :]`) | 50.07% | Trailing `<PAD>` tokens wash out the hidden state |
| **2. `LengthAwareRNN`** | Last real token (`last_idx`) | 84.05% | Indexing each review's actual last non-pad token unlocks strong learning |