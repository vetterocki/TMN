import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score


class MetricsCalculator:
    @staticmethod
    def get_accuracy(pred_values, true_values):
        correct_predictions = (pred_values == true_values).sum()
        return correct_predictions / len(true_values)

    @staticmethod
    def get_precision(pred_values, true_values):
        true_positives = ((pred_values == 1) & (true_values == 1)).sum()
        false_positives = ((pred_values == 1) & (true_values == 0)).sum()
        return true_positives / (true_positives + false_positives)

    @staticmethod
    def get_recall(pred_values, true_values):
        true_positives = ((pred_values == 1) & (true_values == 1)).sum()
        false_negatives = ((pred_values == 0) & (true_values == 1)).sum()
        return true_positives / (true_positives + false_negatives)

    @staticmethod
    def get_f1_score(pred_values, true_values):
        precision = MetricsCalculator.get_precision(pred_values, true_values)
        recall = MetricsCalculator.get_recall(pred_values, true_values)
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def get_log_loss(pred_values_prob, true_values):
        log_loss = -((true_values * np.log(pred_values_prob + 1e-8)) +
                     (1 - true_values) * np.log(1 - pred_values_prob + 1e-8)).mean()
        return log_loss


class Visualizer:
    @staticmethod
    def show_roc_curve(model_name, pred_values_prob, true_values):
        roc_auc = roc_auc_score(true_values, pred_values_prob)
        false_positive_rates, true_positive_rates, _ = roc_curve(true_values, pred_values_prob)
        plt.plot([0, 1], [0, 1], '--')
        plt.plot(false_positive_rates, true_positive_rates, label=f'ROC AUC = {roc_auc:.2f}')
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title(f"{model_name} ROC Curve")
        plt.legend()
        plt.show()

    @staticmethod
    def show_precision_recall_curve(model_name, pred_values_prob, true_values):
        precision, recall, thresholds = precision_recall_curve(true_values, pred_values_prob)
        plt.plot(thresholds, precision[:-1], label='Precision', color='blue')
        plt.plot(thresholds, recall[:-1], label='Recall', color='orange')
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title(f"{model_name} Precision-Recall Curve")
        plt.legend()
        plt.show()


class ClassifierBase:
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test), self.model.predict_proba(x_test)[:, 1]

    def evaluate(self, x_test, y_test):
        y_pred, y_pred_prob = self.predict(x_test)

        print(f"=== {self.model_name} ===")
        print(f"Accuracy: {MetricsCalculator.get_accuracy(y_pred, y_test)}")
        print(f"Precision: {MetricsCalculator.get_precision(y_pred, y_test)}")
        print(f"Recall: {MetricsCalculator.get_recall(y_pred, y_test)}")
        print(f"F1 Score: {MetricsCalculator.get_f1_score(y_pred, y_test)}")
        print(f"Log Loss: {MetricsCalculator.get_log_loss(y_pred_prob, y_test)}")

        Visualizer.show_roc_curve(self.model_name, y_pred_prob, y_test)
        Visualizer.show_precision_recall_curve(self.model_name, y_pred_prob, y_test)


class ShallowTreeClassifier(ClassifierBase):
    def __init__(self):
        from sklearn.tree import DecisionTreeClassifier
        super().__init__(DecisionTreeClassifier(max_depth=5, random_state=25), "Shallow Tree")


class DeepTreeClassifier(ClassifierBase):
    def __init__(self):
        from sklearn.tree import DecisionTreeClassifier
        super().__init__(DecisionTreeClassifier(max_depth=20, random_state=25), "Deep Tree")


class ShallowForestClassifier(ClassifierBase):
    def __init__(self):
        from sklearn.ensemble import RandomForestClassifier
        super().__init__(RandomForestClassifier(n_estimators=100, max_depth=5, random_state=25), "Shallow Forest")


class DeepForestClassifier(ClassifierBase):
    def __init__(self):
        from sklearn.ensemble import RandomForestClassifier
        super().__init__(RandomForestClassifier(n_estimators=100, max_depth=20, random_state=25), "Deep Forest")


class AvoidTypeIIErrorClassifier(ClassifierBase):
    def __init__(self):
        from sklearn.tree import DecisionTreeClassifier
        super().__init__(DecisionTreeClassifier(max_depth=5, class_weight={0: 1, 1: 5}, random_state=25),
                         "Avoid Type II Error Tree")


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    dataset = pd.read_csv('bioresponse.csv')
    x = dataset.iloc[:, 1:]
    y = dataset.Activity

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=25)

    models = [
        ShallowTreeClassifier(),
        DeepTreeClassifier(),
        ShallowForestClassifier(),
        DeepForestClassifier(),
        AvoidTypeIIErrorClassifier()
    ]

    for model in models:
        model.train(x_train, y_train)
        model.evaluate(x_test, y_test)
