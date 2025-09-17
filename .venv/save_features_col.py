import pandas as pd
import pickle
from Bio.SeqUtils import gc_fraction
from collections import Counter

# ================= Helper functions =================
def calculate_nucleotide_features(sequence):
    seq_len = len(sequence)
    if seq_len == 0:
        return {}
    return {
        'length': seq_len,
        'gc_content': gc_fraction(sequence) * 100,
        'a_freq': sequence.count('A') / seq_len * 100,
        'c_freq': sequence.count('C') / seq_len * 100,
        'g_freq': sequence.count('G') / seq_len * 100,
        't_freq': sequence.count('T') / seq_len * 100,
        'at_content': (sequence.count('A') + sequence.count('T')) / seq_len * 100,
    }

def extract_kmers(sequence, k=4):
    if len(sequence) < k:
        return Counter()
    kmers = [sequence[i:i+k] for i in range(len(sequence)-k+1)]
    return Counter(kmers)

def extract_features(sequences, k_values=[3,4,5]):
    feature_matrix = []
    for seq in sequences:
        features = calculate_nucleotide_features(seq)
        for k in k_values:
            kmers = extract_kmers(seq, k)
            top_kmers = dict(kmers.most_common(20))
            for kmer, count in top_kmers.items():
                features[f'{k}mer_{kmer}'] = count / max(len(seq), 1) * 1000
        feature_matrix.append(features)
    feature_df = pd.DataFrame(feature_matrix).fillna(0)
    return feature_df

# ================= Load your trained dataset =================
train_csv = "C:/Users/loq/OneDrive/Desktop/SIH/.venv/data/1opart20.csv"  # Your original 10k CSV
df = pd.read_csv(train_csv)
sequences = df['sequence'].tolist()

# ================= Extract features =================
feature_df = extract_features(sequences)

# ================= Save feature columns =================
feature_columns = list(feature_df.columns)
with open("processed_data/feature_columns.pkl", "wb") as f:
    pickle.dump(feature_columns, f)

print(f"Saved {len(feature_columns)} feature columns to 'processed_data/feature_columns.pkl'")
