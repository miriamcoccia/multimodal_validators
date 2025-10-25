from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)


class Metrics:
    def get_available_metrics(self):
        return [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "mcc",
        ]

    def compute_scores(self, prediction, ground_truth):
        results = {"accuracy": [], "precision": [], "recall": [], "f1": [], "mcc": []}
        results["accuracy"].append(accuracy_score(ground_truth, prediction))
        results["precision"].append(precision_score(ground_truth, prediction))
        results["recall"].append(recall_score(ground_truth, prediction))
        results["f1"].append(f1_score(ground_truth, prediction))
        results["mcc"].append(matthews_corrcoef(ground_truth, prediction))

        return results
