# IMDB Sentiment Classification with PyTorch RNNs

Binary sentiment classification (positive vs. negative) on IMDB movie reviews built from scratch in PyTorch. Beyond hitting solid accuracy, this repository isolates and documents **why naive vanilla RNNs fail on padded text sequences** and how fixing sequence-length indexing unlocks learning before even moving to gated architectures.

---

## Key Diagnostic Insight: Why `NaiveRNN` Collapsed to ~50%

Because reviews are post-padded with `<PAD>` (`0`) to a fixed length of **300 tokens**, slicing the final timestep (`out[:, -1, :]`) forces the RNN to process over a hundred empty zero steps after a short review ends—washing out its hidden state before classification.

```text
Review (120 words) + Post-Padding (180 <PAD> tokens):
[ w_1 ] -> [ w_2 ] -> ... -> [ w_120 ] ----------> [ 0 ] -> [ 0 ] -> ... -> [ 0 ]
                                 ▲                                            ▲
                 LengthAwareRNN reads HERE                     NaiveRNN reads HERE
                  (84.05% Test Accuracy)                       (50.07% Test Accuracy)
