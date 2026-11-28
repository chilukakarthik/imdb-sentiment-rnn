# imdb-sentiment-rnn
Binary sentiment classification (positive vs. negative) on IMDB movie reviews using Recurrent Neural Networks.
## Baseline Experiments (Vanilla RNN)

| Experiment | Sequence Handling | Peak Val Acc | Test Acc | Key Takeaway |
| :--- | :--- | :---: | :---: | :--- |
| **1. Naive Vanilla RNN** | Post-padded (`out[:, -1, :]`) | 51.35% | 49.08% | Trailing `<PAD>` tokens wash out hidden state |
| **2. Fixed Vanilla RNN** | Last real token (`last_idx`) | 78.58% | 75.96% | Fixing padding unlocks learning, though val loss swings due to unclipped vanilla RNN gradients |