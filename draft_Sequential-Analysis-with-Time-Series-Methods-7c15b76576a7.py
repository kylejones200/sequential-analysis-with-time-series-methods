
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import savgol_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class _SeqClassifier(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def _train_torch(model, X_train, y_train, *, epochs=20, n_classes=2):
    X_t = torch.FloatTensor(X_train)
    y_t = torch.LongTensor(y_train)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=16, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
    return model


def _predict_torch(model, X_test):
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_test))
        return logits.argmax(dim=1).numpy()


def simulated_joint_angle_data_degrees() -> None:
    swing_data = np.sin(np.linspace(0, 2 * np.pi, 100)) * 30 + 90
    smoothed_swing = savgol_filter(swing_data, window_length=11, polyorder=2)
    plt.plot(swing_data, label="Original Data")
    plt.plot(smoothed_swing, label="Smoothed Data", linestyle="--")
    plt.title("Golf Swing: Hip Angle Analysis")
    plt.legend()
    plt.savefig("golf_swing_hip_angle.png")
    plt.close()


def simulated_spectral_data() -> None:
    data = pd.DataFrame(
        {
            "Peak1": [1.2, 0.8, 1.0, 1.5, 0.9],
            "Peak2": [0.6, 0.4, 0.7, 0.5, 0.6],
            "Peak3": [0.3, 0.2, 0.4, 0.3, 0.2],
            "WineType": ["Red", "White", "Red", "Red", "White"],
        }
    )
    X = data[["Peak1", "Peak2", "Peak3"]].values
    y = (data["WineType"] == "Red").astype(int).values
    tscv = TimeSeriesSplit(n_splits=3)
    train_idx, test_idx = list(tscv.split(X))[-1]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    logger.info("Wine RF accuracy: %.3f", accuracy_score(y_test, rf.predict(X_test)))

    chess_sequences = np.array(
        [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [3, 2, 1, 4, 5]],
        dtype=np.float32,
    )
    next_moves = np.array([5, 0, 5])
    X_seq = chess_sequences.reshape(len(chess_sequences), 5, 1)
    model = _SeqClassifier(1, n_classes=7)
    _train_torch(model, X_seq, next_moves, epochs=15, n_classes=7)
    test_sequence = np.array([[1, 2, 3, 4, 0]], dtype=np.float32).reshape(1, 5, 1)
    pred = _predict_torch(model, test_sequence)
    logger.info("Predicted next move index: %s", int(pred[0]))


def main() -> None:
    simulated_joint_angle_data_degrees()
    simulated_spectral_data()


if __name__ == "__main__":
    main()
