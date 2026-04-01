# Description: Short example for Sequential Analysis with Time Series Methods.



from scipy.signal import savgol_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.models import Sequential
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


"""
Pose Analysis: Golf Swing Evaluation
Goal:
Analyze the sequence of joint angles during a golf swing to identify inefficiencies.
"""


# Simulated joint angle data (degrees)
swing_data = np.sin(np.linspace(0, 2 * np.pi, 100)) * 30 + 90  # Hip angle

# Smooth the data
smoothed_swing = savgol_filter(swing_data, window_length=11, polyorder=2)

# Plot
plt.plot(swing_data, label='Original Data')
plt.plot(smoothed_swing, label='Smoothed Data', linestyle='--')
plt.title('Golf Swing: Hip Angle Analysis')
plt.xlabel('Sequence Index')
plt.ylabel('Angle (Degrees)')
plt.legend()
plt.savefig('Golf Swing Hip Angle Analysis.png')
plt.show()

"""
Spectral Analysis: Wine Classification
Goal: Identify wine type based on spectroscopic data.
"""



# Simulated spectral data
data = pd.DataFrame({
    'Peak1': [1.2, 0.8, 1.0, 1.5, 0.9],
    'Peak2': [0.6, 0.4, 0.7, 0.5, 0.6],
    'Peak3': [0.3, 0.2, 0.4, 0.3, 0.2],
    'WineType': ['Red', 'White', 'Red', 'Red', 'White']
})

# Feature and target separation
X = data[['Peak1', 'Peak2', 'Peak3']]
y = data['WineType']

tscv = TimeSeriesSplit(n_splits=3)
train_idx, test_idx = list(tscv.split(X))[ -1 ]
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
logger.info(f"Accuracy: {accuracy_score(y_test, y_pred)}")

"""
Chess Move Prediction
Goal: Predict the next move in a chess game based on prior moves.
"""


# Simulated chess move data (numerical encoding)
chess_sequences = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [3, 2, 1, 4, 5]]
next_moves = [6, 1, 6]  # Next move in each sequence

# Prepare data
X = tf.keras.preprocessing.sequence.pad_sequences(chess_sequences, maxlen=5)
y = tf.keras.utils.to_categorical(next_moves, num_classes=7)

# Define model
model = Sequential([
    Embedding(input_dim=7, output_dim=4),
    LSTM(32),
    Dense(7, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X, y, epochs=10, verbose=0)

# Predict next move
test_sequence = [[1, 2, 3, 4, 0]]  # New game sequence
test_sequence = tf.keras.preprocessing.sequence.pad_sequences(test_sequence, maxlen=5)
prediction = model.predict(test_sequence)
logger.info("Predicted Next Move:", prediction.argmax())

# Predicted Next Move: 6
