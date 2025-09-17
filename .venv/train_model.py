# ================= train_model_incremental.py =================
import os
import pickle
import logging
from collections import Counter
from itertools import product
from math import log2

import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping

# ---------------- Logger ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================= Sequence Similarity =================
class SequenceSimilarityCalculator:
    def __init__(self, known_sequences, known_labels):
        self.known_sequences = known_sequences
        self.known_labels = known_labels
    
    def calculate_similarity(self, seq1, seq2):
        if len(seq1) == 0 or len(seq2) == 0:
            return 0.0
        min_len = min(len(seq1), len(seq2))
        seq1_arr = np.frombuffer(seq1[:min_len].encode(), dtype='S1')
        seq2_arr = np.frombuffer(seq2[:min_len].encode(), dtype='S1')
        matches = np.sum(seq1_arr == seq2_arr)
        return (matches / min_len) * 100
    
    def find_top_matches(self, query_sequence, top_n=3):
        similarities = []
        query_len = len(query_sequence)
        for i, (known_seq, known_label) in enumerate(zip(self.known_sequences, self.known_labels)):
            if abs(len(known_seq) - query_len) > 50:
                continue
            similarity = self.calculate_similarity(query_sequence, known_seq)
            similarities.append({
                'sequence': known_seq,
                'species': known_label,
                'similarity': similarity,
                'index': i
            })
            if similarity == 100.0:
                break
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_n]

# ================= Preprocessor =================
class eDNAPreprocessor:
    def __init__(self):
        self.label_encoder = None
        self.scaler = None
        self.feature_columns = None
        self.taxonomy_hierarchy = {}
        self.known_sequences = []
        self.known_labels = []
        self.feature_extractor_params = {'k_values':[3,4], 'top_k':30}

    # -------- Feature Extraction --------
    def calculate_nucleotide_features(self, sequence):
        seq_len = len(sequence)
        if seq_len == 0: return {}
        counter = Counter(sequence)
        a = counter.get('A',0); c = counter.get('C',0)
        g = counter.get('G',0); t = counter.get('T',0)
        total = a+c+g+t
        return {
            'length': seq_len,
            'gc_content': (g+c)/max(total,1)*100,
            'a_freq': a/max(total,1)*100,
            'c_freq': c/max(total,1)*100,
            'g_freq': g/max(total,1)*100,
            't_freq': t/max(total,1)*100,
            'at_content': (a+t)/max(total,1)*100,
            'gc_skew': (g-c)/max(g+c,1) if (g+c)>0 else 0,
            'at_skew': (a-t)/max(a+t,1) if (a+t)>0 else 0
        }

    def calculate_dinuc_features(self, sequence):
        seq_len = len(sequence)
        if seq_len < 2: return {}
        dinucs = [sequence[i:i+2] for i in range(seq_len-1)]
        counter = Counter(dinucs)
        total = sum(counter.values())
        return {f'di_{d}': counter.get(d,0)/max(total,1)*100 
                for d in [''.join(p) for p in product('ACGT', repeat=2)]}

    def calculate_trinuc_features(self, sequence):
        seq_len = len(sequence)
        if seq_len < 3: return {}
        if seq_len > 50:
            trinucs = [sequence[i:i+3] for i in range(seq_len-2)]
            counter = Counter(trinucs)
            total = sum(counter.values())
            return {f'tri_{t}': counter.get(t,0)/max(total,1)*100 
                    for t in [''.join(p) for p in product('ACGT', repeat=3)]}
        return {}

    def extract_kmers(self, sequence, k=4):
        return Counter(sequence[i:i+k] for i in range(len(sequence)-k+1)) if len(sequence)>=k else Counter()

    def extract_features(self, sequences, k_values=None, top_k=None, n_jobs=4):
        k_values = k_values or self.feature_extractor_params['k_values']
        top_k = top_k or self.feature_extractor_params['top_k']
        logger.info(f"🔬 Extracting features for {len(sequences)} sequences...")
        def process_seq(seq):
            feats = {}
            feats.update(self.calculate_nucleotide_features(seq))
            feats.update(self.calculate_dinuc_features(seq))
            if len(seq) > 50:
                feats.update(self.calculate_trinuc_features(seq))
            for k in k_values:
                if len(seq) >= k:
                    kmers = self.extract_kmers(seq,k)
                    for km, count in dict(kmers.most_common(top_k)).items():
                        feats[f'{k}mer_{km}'] = count/max(len(seq),1)*1000
            return feats
        feature_matrix = Parallel(n_jobs=n_jobs)(delayed(process_seq)(seq) for seq in tqdm(sequences))
        df = pd.DataFrame(feature_matrix).fillna(0)
        logger.info(f"✅ Feature matrix shape: {df.shape}")
        return df

    def fit_preprocessors(self, feature_df, labels):
        self.feature_columns = feature_df.columns.tolist()
        self.scaler = StandardScaler().fit(feature_df)
        X_scaled = self.scaler.transform(feature_df)
        self.label_encoder = LabelEncoder().fit(labels)
        y = self.label_encoder.transform(labels)
        return X_scaled, y

    def transform(self, feature_df):
        for col in set(self.feature_columns) - set(feature_df.columns):
            feature_df[col] = 0
        feature_df = feature_df[self.feature_columns]
        return self.scaler.transform(feature_df)

    def save(self, output_dir='processed_data'):
        os.makedirs(output_dir, exist_ok=True)
        pickle.dump(self.scaler, open(f'{output_dir}/scaler.pkl','wb'))
        pickle.dump(self.label_encoder, open(f'{output_dir}/label_encoder.pkl','wb'))
        pickle.dump(self.feature_columns, open(f'{output_dir}/feature_columns.pkl','wb'))
        pickle.dump(self.taxonomy_hierarchy, open(f'{output_dir}/taxonomy_hierarchy.pkl','wb'))
        pickle.dump((self.known_sequences, self.known_labels), open(f'{output_dir}/known_data.pkl','wb'))
        pickle.dump(self.feature_extractor_params, open(f'{output_dir}/feature_extractor_params.pkl','wb'))
        pickle.dump(SequenceSimilarityCalculator(self.known_sequences, self.known_labels),
                    open(f'{output_dir}/similarity_calculator.pkl','wb'))
        logger.info("✅ Preprocessors + known data + similarity calculator saved.")

# ================= Incremental Training =================
def train_csv_incremental(csv_files, model_dir='processed_data', batch_size=10000):
    os.makedirs(model_dir, exist_ok=True)
    all_sequences, all_labels = [], []

    logger.info("📂 Loading CSV files in batches...")
    for file in csv_files:
        for chunk in pd.read_csv(file, chunksize=batch_size):
            all_sequences.extend(chunk['sequence'].astype(str).tolist())
            all_labels.extend(chunk['species'].astype(str).tolist())
            logger.info(f"📊 Loaded batch: {len(chunk)} sequences")

    # Representative sampling: top classes with ≥20 sequences
    counter = Counter(all_labels)
    top_classes = [cls for cls, c in counter.most_common(500) if c >= 20]
    filtered_sequences = [s for s,l in zip(all_sequences, all_labels) if l in top_classes]
    filtered_labels = [l for l in all_labels if l in top_classes]
    logger.info(f"✅ Filtered to {len(filtered_sequences)} sequences for top {len(top_classes)} classes")

    preprocessor = eDNAPreprocessor()
    feature_df = preprocessor.extract_features(filtered_sequences)
    X, y = preprocessor.fit_preprocessors(feature_df, filtered_labels)

    preprocessor.known_sequences = filtered_sequences
    preprocessor.known_labels = filtered_labels
    preprocessor.taxonomy_hierarchy = {l:{'species':l} for l in preprocessor.label_encoder.classes_}

    # Train-test split
    unique_vals, counts = np.unique(y, return_counts=True)
    stratify_param = y if counts.min()>=2 else None
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,
                                                      random_state=42, stratify=stratify_param)

    # LightGBM parameters
    params = {
        'objective':'multiclass', 'num_class':len(np.unique(y)), 'metric':'multi_logloss',
        'num_leaves':255, 'learning_rate':0.05, 'feature_fraction':0.8,
        'bagging_fraction':0.8, 'bagging_freq':5, 'max_depth':-1, 'min_data_in_leaf':20,
        'n_jobs':-1, 'verbose':-1
    }

    lgb_train = lgb.Dataset(X_train,label=y_train)
    lgb_val = lgb.Dataset(X_val,label=y_val,reference=lgb_train)
    logger.info("🚀 Starting LightGBM training...")
    model = lgb.train(params, lgb_train, num_boost_round=500,
                      valid_sets=[lgb_val], callbacks=[log_evaluation(25), early_stopping(30)])

    model.save_model(f'{model_dir}/edna_lgb_model.txt')
    preprocessor.save(model_dir)

    # Evaluation
    y_pred = np.argmax(model.predict(X_val), axis=1)
    acc = np.mean(y_pred==y_val)
    f1_macro = f1_score(y_val, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_val, y_pred, average='weighted', zero_division=0)
    logger.info(f"🎯 Validation Accuracy: {acc:.4f}, F1 Macro: {f1_macro:.4f}, F1 Weighted: {f1_weighted:.4f}")
    logger.info(f"✅ Training completed. Model saved at {model_dir}")
    return model, preprocessor

# ================= Main =================
if __name__ == "__main__":
    csv_files = ["data/merged_filtered_sequences.csv"]  # Replace with your actual 5K sequences CSV
    if not os.path.exists(csv_files[0]):
        logger.info("📝 Creating sample CSV with 5000 sequences...")
        os.makedirs("data", exist_ok=True)
        patterns = {'species_A':'GC'*50, 'species_B':'AT'*50, 'species_C':'GATTACA'*14,
                    'species_D':'ACGT'*25, 'species_E':'GGGCCC'*16}
        sequences, species = [], []
        for sp, pat in patterns.items():
            for i in range(1000):
                seq = pat
                if i%10==0: seq = seq[:-5] + 'ATCG' + seq[-1]
                sequences.append(seq)
                species.append(sp)
        pd.DataFrame({'sequence':sequences,'species':species}).to_csv(csv_files[0], index=False)
        logger.info("✅ Sample CSV created")
    model, preprocessor = train_csv_incremental(csv_files)
