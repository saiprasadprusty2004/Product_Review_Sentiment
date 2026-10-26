# DL Project

A comparison of MLP, TextCNN, and BiLSTM on **20,000 real Amazon product reviews**, following the ICT4442 synopsis.

- [Project README](product_review_sentiment-main/README.md): setup, models, dataset, training, Jupyter instructions, memory improvements, results, outputs, and checks.
- [Executed notebook](product_review_sentiment-main/notebook/DL_Phase2_Sentiment_Pipeline_EXECUTED.ipynb).
- [Real-data results](product_review_sentiment-main/results/amazon_20k/RESULTS.md) and [reviewed failure cases](product_review_sentiment-main/results/amazon_20k/ERROR_REVIEW.md).

Run from this workspace using the existing environment:

```powershell
& '.\.verification-env\Scripts\python.exe' product_review_sentiment-main/src/pipeline.py --resume
```

`--resume` reuses verified completed stages. Use a new `--output-dir` for changed experiments to preserve existing results. The current local Jupyter page is available at [127.0.0.1:8888](http://127.0.0.1:8888) while its server is running.
