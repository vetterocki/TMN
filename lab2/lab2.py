import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        column_names = ['Variance', 'Skewness', 'Curtosis', 'Entropy', 'Class']
        data = pd.read_csv(self.file_path, header=None, names=column_names)
        X = data.drop('Class', axis=1)
        y = data['Class']
        return X, y

    @staticmethod
    def split_data(X, y, test_size=0.2):
        return train_test_split(X, y, test_size=test_size, random_state=42)

    @staticmethod
    def prepare_data(X_train, y_train, X_test, y_test):
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        return train_loader, test_loader

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

class LossFunctions:
    @staticmethod
    def logistic_loss(output, target):
        output = torch.clamp(output, min=1e-7, max=1 - 1e-7)
        return - (target * torch.log(output) + (1 - target) * torch.log(1 - output)).mean()

    @staticmethod
    def adaboost_loss(output, target):
        target = 2 * target - 1
        return torch.exp(-target * output).mean()

    @staticmethod
    def binary_crossentropy_loss(output, target):
        criterion = nn.BCELoss()
        return criterion(output, target)

class Trainer:
    def __init__(self, model, criterion, optimizer, train_loader, test_loader, epochs=20):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs

    def train(self):
        train_loss_history = []
        test_loss_history = []

        for epoch in range(self.epochs):
            self.model.train()
            epoch_train_loss = 0

            for inputs, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs.squeeze(), labels.float())
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()

            train_loss_history.append(epoch_train_loss / len(self.train_loader))

            self.model.eval()
            epoch_test_loss = 0

            with torch.no_grad():
                for inputs, labels in self.test_loader:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs.squeeze(), labels.float())
                    epoch_test_loss += loss.item()

            test_loss_history.append(epoch_test_loss / len(self.test_loader))

        return train_loss_history, test_loss_history

    def evaluate(self):
        self.model.eval()
        predictions, targets = [], []

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                outputs = self.model(inputs)
                predicted = outputs.squeeze().round()
                predictions.extend(predicted.numpy())
                targets.extend(labels.numpy())

        return accuracy_score(targets, predictions)

class Experiment:
    def __init__(self, file_path):
        self.data_handler = DataHandler(file_path)

    def run(self):
        X, y = self.data_handler.load_data()
        X_train, X_test, y_train, y_test = self.data_handler.split_data(X, y)
        train_loader, test_loader = self.data_handler.prepare_data(X_train, y_train, X_test, y_test)
        input_dim = X_train.shape[1]
        epochs = 20

        models = {
            'logistic_loss': LogisticRegressionModel(input_dim),
            'adaboost_loss': LogisticRegressionModel(input_dim),
            'binary_crossentropy_loss': LogisticRegressionModel(input_dim)
        }

        optimizers = {
            name: optim.SGD(model.parameters(), lr=0.01)
            for name, model in models.items()
        }

        criteria = {
            'logistic_loss': LossFunctions.logistic_loss,
            'adaboost_loss': LossFunctions.adaboost_loss,
            'binary_crossentropy_loss': LossFunctions.binary_crossentropy_loss
        }

        loss_histories, accuracies = {}, {}

        for name, model in models.items():
            trainer = Trainer(model, criteria[name], optimizers[name], train_loader, test_loader, epochs)
            train_loss, test_loss = trainer.train()
            loss_histories[name] = (train_loss, test_loss)
            accuracies[name] = trainer.evaluate()
            print(f'{name} loss accuracy: {accuracies[name]:.4f}')

        self.plot_results(loss_histories)

    @staticmethod
    def plot_results(loss_histories):
        plt.figure(figsize=(12, 6))
        for name, (train_loss, _) in loss_histories.items():
            plt.plot(train_loss, label=f'{name} - Train')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Learning Curves for Training Data')
        plt.legend()
        plt.show()

        plt.figure(figsize=(12, 6))
        for name, (_, test_loss) in loss_histories.items():
            plt.plot(test_loss, label=f'{name} - Test')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Learning Curves for Testing Data')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    file_path = 'data_banknote_authentication.txt'
    experiment = Experiment(file_path)
    experiment.run()
