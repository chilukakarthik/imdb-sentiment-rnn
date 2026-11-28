# imdb-sentiment-rnn

Binary sentiment classification on the IMDB 50k movie reviews dataset using PyTorch RNNs.

## Data & Preprocessing

All preprocessing is done in `notebooks/01_eda.ipynb`:

- Lowercase text and strip HTML tags/punctuation with regex
- Remove stopwords and apply Porter stemming
- Build a 10,000-word vocabulary (`<PAD>` = 0, `<UNK>` = 1)
- Post-pad or truncate reviews to 300 tokens

## Models & Results

Both models share the same setup: `nn.Embedding(10000, 100)` -> 1-layer `nn.RNN(100, 128)` -> `nn.Linear(128, 1)`, trained with `BCEWithLogitsLoss` and `Adam (lr=0.001)`.

| Model | Hidden State Used | Test Accuracy |
| :--- | :--- | :---: |
| `NaiveRNN` | Final timestep (`out[:, -1, :]`) | 50.07% |
| `LengthAwareRNN` | Last non-pad token (`last_idx`) | 84.05% |

`NaiveRNN` stayed stuck around 50% because most cleaned reviews are much shorter than 300 tokens. Taking `out[:, -1, :]` meant the RNN stepped through over a hundred trailing `<PAD>` zeros after a review ended, washing out the hidden state. Indexing each review at its actual last real token jumped test accuracy to 84.05% without changing the RNN cell.
