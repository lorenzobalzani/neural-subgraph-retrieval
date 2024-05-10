import numpy as np
from evaluate import load

accuracy = load("accuracy")
precision = load("precision")
recall = load("recall")
f1 = load("f1")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels),
        "precision": precision.compute(
            predictions=predictions, references=labels, average="micro"
        ),
        "recall": recall.compute(
            predictions=predictions, references=labels, average="micro"
        ),
        "f1": f1.compute(predictions=predictions, references=labels, average="micro"),
    }
