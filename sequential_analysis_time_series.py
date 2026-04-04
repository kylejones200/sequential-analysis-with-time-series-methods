"""
Sequential Analysis with Time Series Methods
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, silhouette_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from typing import Tuple

np.random.seed(42)
plt.rcParams.update({
    'font.family': 'serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.5,
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#333333',
    'text.color': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'font.size': 9
})

def save_fig(path: str):
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()

def dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    n, m = len(seq1), len(seq2)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dtw[i, j] = abs(seq1[i-1] - seq2[j-1]) + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    return dtw[n, m]

def generate_swings(n: int) -> Tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, 2*np.pi, 100)
    pro = np.array([90 + 30*np.sin(t) + np.random.randn(100)*1.5 for _ in range(n)])
    amateur = np.array([90 + 30*np.sin(t*np.random.uniform(0.8, 1.2)) + np.random.randn(100)*4 for _ in range(n)])
    return pro, amateur

def analyze_golf_swing():
    swings_pro, swings_amateur = generate_swings(50)
    pro, amateur = swings_pro[0], swings_amateur[0]
    pro_sm, amateur_sm = savgol_filter(pro, 11, 2), savgol_filter(amateur, 11, 2)
    
    dtw_dists = [dtw_distance(swings_pro[0], s) for s in [*swings_pro[1:20], *swings_amateur[:20]]]
    labels = ['Pro']*19 + ['Amateur']*20
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    
    axes[0].plot(pro_sm, label='Professional', linewidth=1.5, color='#2c3e50')
    axes[0].plot(amateur_sm, label='Amateur', linewidth=1.5, color='#e74c3c', linestyle='--')
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('Hip Angle (°)')
    axes[0].legend(frameon=False)
    
    bp = axes[1].boxplot([dtw_dists[:19], dtw_dists[19:]], widths=0.5, patch_artist=True,
                          boxprops=dict(facecolor='#ecf0f1', color='#2c3e50'),
                          medianprops=dict(color='#e74c3c', linewidth=1.5),
                          whiskerprops=dict(color='#2c3e50'),
                          capprops=dict(color='#2c3e50'))
    axes[1].set_xticklabels(['Pro vs Pro', 'Pro vs Amateur'])
    axes[1].set_ylabel('DTW Distance')
    
    save_fig('sequential_golf_swing.png')
    
    print(f"Golf Swing DTW: {dtw_distance(pro_sm, amateur_sm):.0f}")

def generate_wine_spectra(n: int) -> Tuple[np.ndarray, np.ndarray]:
    wl = np.linspace(800, 2500, 200)
    red = np.array([0.5*np.exp(-((wl-1200)/150)**2) + 0.3*np.exp(-((wl-1600)/200)**2) + 
                    np.random.randn(200)*0.02 for _ in range(n//2)])
    white = np.array([0.3*np.exp(-((wl-1100)/150)**2) + 0.4*np.exp(-((wl-1700)/200)**2) + 
                      np.random.randn(200)*0.02 for _ in range(n//2)])
    X = np.vstack([red, white])
    y = np.array([0]*(n//2) + [1]*(n//2))
    return X, y

def extract_spectral_features(X: np.ndarray) -> pd.DataFrame:
    features = []
    for spec in X:
        peaks, _ = find_peaks(spec, height=0.2, distance=20)
        features.append({
            'peak_count': len(peaks),
            'max_intensity': spec.max(),
            'mean_intensity': spec.mean(),
            'std_intensity': spec.std(),
            'peak_1_height': spec[peaks[0]] if len(peaks) > 0 else 0,
            'peak_2_height': spec[peaks[1]] if len(peaks) > 1 else 0,
        })
    return pd.DataFrame(features)

def analyze_wine_spectra():
    X, y = generate_wine_spectra(200)
    X_feat = extract_spectral_features(X)
    X_train, X_test, y_train, y_test = train_test_split(X_feat, y, test_size=0.3, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    
    axes[0].plot(X[y == 0].mean(axis=0), label='Red', linewidth=1.5, color='#c0392b')
    axes[0].plot(X[y == 1].mean(axis=0), label='White', linewidth=1.5, color='#f39c12')
    axes[0].set_xlabel('Wavelength Index')
    axes[0].set_ylabel('Absorbance')
    axes[0].legend(frameon=False)
    
    feat_imp = pd.Series(clf.feature_importances_, index=X_feat.columns).nlargest(4)
    axes[1].barh(range(len(feat_imp)), feat_imp.values, color='#34495e', height=0.6)
    axes[1].set_yticks(range(len(feat_imp)))
    axes[1].set_yticklabels(feat_imp.index)
    axes[1].set_xlabel('Importance')
    axes[1].invert_yaxis()
    
    save_fig('sequential_wine_spectra.png')
    
    print(f"Wine Classification Acc: {acc:.3f}, AUC: {auc:.3f}")

def generate_chess_sequences(n: int, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for _ in range(n):
        seq = [np.random.randint(1, 20) if i < 5 else 
               np.random.randint(10, 60) if i < 15 else 
               np.random.randint(40, 100) for i in range(seq_len)]
        X.append(seq[:-1])
        y.append(seq[-1])
    return np.array(X), np.array(y)

def analyze_chess_sequences():
    X, y = generate_chess_sequences(500, 20)
    
    X_feat = pd.DataFrame({
        'last_move': X[:, -1],
        'prev_move': X[:, -2],
        'avg_move': X.mean(axis=1),
        'std_move': X.std(axis=1),
        'range': X.max(axis=1) - X.min(axis=1)
    })
    
    X_train, X_test, y_train, y_test = train_test_split(X_feat, y, test_size=0.3, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    
    for i in range(5):
        axes[0].plot(X[i], alpha=0.5, linewidth=1, color='#7f8c8d')
    axes[0].set_xlabel('Move Number')
    axes[0].set_ylabel('Move ID')
    
    feat_imp = pd.Series(clf.feature_importances_, index=X_feat.columns).sort_values(ascending=True)
    axes[1].barh(range(len(feat_imp)), feat_imp.values, color='#27ae60', height=0.6)
    axes[1].set_yticks(range(len(feat_imp)))
    axes[1].set_yticklabels(feat_imp.index)
    axes[1].set_xlabel('Importance')
    
    save_fig('sequential_chess_moves.png')
    
    print(f"Chess Move Prediction Acc: {acc:.3f}")

def generate_protein_data(n: int, length: int) -> Tuple[list, list]:
    amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
    sequences, structures = [], []
    
    for _ in range(n):
        seq = [np.random.choice(amino_acids) for _ in range(length)]
        struct = [np.random.choice(['H', 'C'], p=[0.7, 0.3]) if aa in 'AELM' else
                  np.random.choice(['E', 'C'], p=[0.6, 0.4]) if aa in 'VIY' else
                  np.random.choice(['H', 'E', 'C'], p=[0.3, 0.3, 0.4]) for aa in seq]
        sequences.append(seq)
        structures.append(struct)
    
    return sequences, structures

def encode_aa(aa: str) -> int:
    return 'ACDEFGHIKLMNPQRSTVWY'.index(aa) if aa in 'ACDEFGHIKLMNPQRSTVWY' else 0

def analyze_protein_structure():
    sequences, structures = generate_protein_data(200, 50)
    
    X = np.array([[encode_aa(aa) for aa in seq] for seq in sequences])
    y = np.array([1 if s[25] == 'H' else (2 if s[25] == 'E' else 0) for s in structures])
    
    X_feat = pd.DataFrame({
        'center': X[:, 25],
        'left_1': X[:, 24],
        'right_1': X[:, 26],
        'window_mean': X[:, 20:30].mean(axis=1),
        'window_std': X[:, 20:30].std(axis=1),
    })
    
    X_train, X_test, y_train, y_test = train_test_split(X_feat, y, test_size=0.3, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    
    struct_names = ['Coil', 'Helix', 'Sheet']
    counts = pd.Series(y_train).value_counts().sort_index()
    colors = ['#95a5a6', '#e74c3c', '#3498db']
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    
    axes[0].bar(struct_names, [counts.get(i, 0) for i in range(3)], color=colors, width=0.6)
    axes[0].set_ylabel('Count')
    
    feat_imp = pd.Series(clf.feature_importances_, index=X_feat.columns).sort_values(ascending=True)
    axes[1].barh(range(len(feat_imp)), feat_imp.values, color='#9b59b6', height=0.6)
    axes[1].set_yticks(range(len(feat_imp)))
    axes[1].set_yticklabels(feat_imp.index)
    axes[1].set_xlabel('Importance')
    
    save_fig('sequential_protein_structure.png')
    
    print(f"Protein Structure Acc: {acc:.3f}")

def generate_process_data(n: int, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    sequences, labels = [], []
    
    for i in range(n):
        t = np.linspace(0, 10, seq_len)
        
        if i < n * 0.8:
            temp = 200 + 5*np.sin(t) + np.random.randn(seq_len)*1
            pressure = 100 + 3*np.cos(t*1.5) + np.random.randn(seq_len)*0.5
            flow = 50 + 2*np.sin(t*2) + np.random.randn(seq_len)*0.3
            labels.append(0)
        else:
            anom_type = np.random.choice(['drift', 'spike', 'oscillation'])
            if anom_type == 'drift':
                temp = 200 + t*3 + np.random.randn(seq_len)*2
                pressure = 100 + t*2 + np.random.randn(seq_len)*1
                flow = 50 + t*1.5 + np.random.randn(seq_len)*0.5
            elif anom_type == 'spike':
                temp = 200 + 5*np.sin(t) + np.random.randn(seq_len)*1
                spike_idx = np.random.randint(30, 70)
                temp[spike_idx:spike_idx+5] += 30
                pressure = 100 + 3*np.cos(t*1.5) + np.random.randn(seq_len)*0.5
                flow = 50 + 2*np.sin(t*2) + np.random.randn(seq_len)*0.3
            else:
                temp = 200 + 15*np.sin(t*5) + np.random.randn(seq_len)*1
                pressure = 100 + 10*np.cos(t*3) + np.random.randn(seq_len)*0.5
                flow = 50 + 8*np.sin(t*4) + np.random.randn(seq_len)*0.3
            labels.append(1)
        
        sequences.append(np.column_stack([temp, pressure, flow]))
    
    return np.array(sequences), np.array(labels)

def extract_process_features(X: np.ndarray) -> pd.DataFrame:
    features = []
    for seq in X:
        temp, pressure, flow = seq[:, 0], seq[:, 1], seq[:, 2]
        features.append({
            'temp_mean': temp.mean(),
            'temp_std': temp.std(),
            'temp_trend': np.polyfit(range(len(temp)), temp, 1)[0],
            'pressure_std': pressure.std(),
            'flow_std': flow.std(),
            'temp_pressure_corr': np.corrcoef(temp, pressure)[0, 1]
        })
    return pd.DataFrame(features)

def analyze_industrial_process():
    X, y = generate_process_data(500, 100)
    X_feat = extract_process_features(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_feat)
    
    detector = IsolationForest(contamination=0.2, random_state=42)
    detector.fit(X_scaled[y == 0])
    
    y_pred = (detector.predict(X_scaled) == -1).astype(int)
    acc = accuracy_score(y, y_pred)
    
    from sklearn.metrics import precision_score, recall_score
    prec, rec = precision_score(y, y_pred), recall_score(y, y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    
    normal_idx, anom_idx = np.where(y == 0)[0][0], np.where(y == 1)[0][0]
    for idx, ax, label in [(normal_idx, axes[0], 'Normal'), (anom_idx, axes[1], 'Anomaly')]:
        ax.plot(X[idx][:, 0], linewidth=1, label='Temp', color='#e74c3c')
        ax.plot(X[idx][:, 1], linewidth=1, label='Pressure', color='#3498db')
        ax.plot(X[idx][:, 2], linewidth=1, label='Flow', color='#2ecc71')
        ax.set_title(label)
        ax.set_xlabel('Time')
        if idx == normal_idx:
            ax.set_ylabel('Sensor Value')
            ax.legend(frameon=False, loc='upper right', fontsize=8)
    
    save_fig('sequential_industrial_process.png')
    
    print(f"Anomaly Detection Acc: {acc:.3f}, Prec: {prec:.3f}, Rec: {rec:.3f}")

def main():
    print("\nSequential Analysis with Time Series Methods\n")
    
    analyze_golf_swing()
    analyze_wine_spectra()
    analyze_chess_sequences()
    analyze_protein_structure()
    analyze_industrial_process()
    
    print("\nOutputs: 5 PNG files\n")

if __name__ == "__main__":
    main()
