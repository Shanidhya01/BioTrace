# ================= train_model.py =================
import pandas as pd
import numpy as np
import lightgbm as lgb
import os, pickle, logging
from tqdm import tqdm
from joblib import Parallel, delayed
from collections import Counter
from itertools import product
from math import log2
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================= Preprocessor =================
class eDNAPreprocessor:
    def _init_(self):
        self.label_encoder = None
        self.feature_columns = None

    def calculate_nucleotide_features(self, sequence):
        seq_len = len(sequence)
        if seq_len == 0:
            return {}
        a, c, g, t = sequence.count('A'), sequence.count('C'), sequence.count('G'), sequence.count('T')
        return {
            'length': seq_len,
            'gc_content': (g+c)/max(seq_len,1)*100,
            'a_freq': a/seq_len*100, 'c_freq': c/seq_len*100,
            'g_freq': g/seq_len*100, 't_freq': t/seq_len*100,
            'at_content': (a+t)/seq_len*100,
            'gc_skew': (g-c)/max(g+c,1),
            'at_skew': (a-t)/max(a+t,1),
            'entropy': -sum([(x/seq_len)*log2(x/seq_len) for x in [a,c,g,t] if x>0])
        }

    def calculate_dinuc_features(self, sequence):
        seq_len = len(sequence)
        return {f'di_{d}': sequence.count(d)/max(seq_len-1,1)*100
                for d in [''.join(p) for p in product('ACGT', repeat=2)]}

    def calculate_trinuc_features(self, sequence):
        seq_len = len(sequence)
        return {f'tri_{t}': sequence.count(t)/max(seq_len-2,1)*100
                for t in [''.join(p) for p in product('ACGT', repeat=3)]}

    def extract_kmers(self, sequence, k=4):
        return Counter(sequence[i:i+k] for i in range(len(sequence)-k+1)) if len(sequence)>=k else Counter()

    def extract_features(self, sequences, k_values=[4], top_k=50, n_jobs=-1):
        def process_seq(seq):
            feats = {}
            feats.update(self.calculate_nucleotide_features(seq))
            feats.update(self.calculate_dinuc_features(seq))
            feats.update(self.calculate_trinuc_features(seq))
            for k in k_values:
                kmers = self.extract_kmers(seq, k)
                for kmer, count in dict(kmers.most_common(top_k)).items():
                    feats[f'{k}mer_{kmer}'] = count/max(len(seq),1)*1000
            return feats

        feature_matrix = Parallel(n_jobs=n_jobs)(
            delayed(process_seq)(seq) for seq in tqdm(sequences, desc="🔬 Extracting features")
        )
        logger.info(f"✅ Extracted features for {len(sequences)} sequences")
        return pd.DataFrame(feature_matrix).fillna(0)

    def fit_preprocessors(self, feature_df, labels):
        from sklearn.preprocessing import LabelEncoder
        self.feature_columns = feature_df.columns.tolist()
        self.label_encoder = LabelEncoder().fit(labels)
        y = self.label_encoder.transform(labels)
        return feature_df.values, y

    def transform(self, feature_df):
        feature_df = feature_df.reindex(columns=self.feature_columns, fill_value=0)
        return feature_df.values

    def save(self, output_dir='processed_data'):
        os.makedirs(output_dir, exist_ok=True)
        pickle.dump(self.label_encoder, open(f'{output_dir}/label_encoder.pkl','wb'))
        pickle.dump(self.feature_columns, open(f'{output_dir}/feature_columns.pkl','wb'))
        logger.info("✅ Preprocessors saved.")

    def load(self, input_dir='processed_data'):
        self.label_encoder = pickle.load(open(f'{input_dir}/label_encoder.pkl','rb'))
        self.feature_columns = pickle.load(open(f'{input_dir}/feature_columns.pkl','rb'))
        logger.info("✅ Preprocessors loaded.")

# ================= Training =================
def train_csv(csv_files, model_dir='processed_data', sample_size=30000):
    os.makedirs(model_dir, exist_ok=True)
    data_file = f"{model_dir}/train_data.pkl"

    # Load previous data if exists
    if os.path.exists(data_file):
        logger.info("📂 Loading previous training data...")
        all_sequences, all_labels = pickle.load(open(data_file,'rb'))
    else:
        all_sequences, all_labels = [], []

    # Load new CSVs
    logger.info("📂 Loading new CSV files...")
    for file in tqdm(csv_files, desc="Reading files", unit="file"):
        df = pd.read_csv(file)
        if 'sequence' not in df.columns or 'species' not in df.columns:
            raise ValueError(f"{file} must contain 'sequence' and 'species' columns")
        all_sequences.extend(df['sequence'].tolist())
        all_labels.extend(df['species'].tolist())

    # Save combined dataset
    pickle.dump((all_sequences, all_labels), open(data_file,'wb'))

    # Extract features
    preprocessor = eDNAPreprocessor()
    feature_df = preprocessor.extract_features(all_sequences, k_values=[4], top_k=50)
    X, y = preprocessor.fit_preprocessors(feature_df, all_labels)

    # Train/validation split
    from collections import Counter
    min_class_count = min(Counter(y).values())
    stratify_opt = y if min_class_count > 1 else None
    if stratify_opt is None:
        logger.warning("⚠ Some classes have only 1 sample → stratify disabled.")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=stratify_opt
    )

    # Prepare LightGBM datasets
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    # Parameters
    params = {
        'objective':'multiclass',
        'num_class':len(preprocessor.label_encoder.classes_),
        'metric':'multi_logloss',
        'num_leaves':31,
        'learning_rate':0.05,
        'feature_fraction':0.8,
        'bagging_fraction':0.8,
        'bagging_freq':1,
        'max_depth':10,
        'n_jobs':-1,
        'verbose':-1
    }

    # Progress callback
    def progress_callback(env):
        percent = (env.iteration / env.end_iteration) * 100
        logger.info(f"⏳ Training progress: {percent:.2f}%")

    # Train model
    logger.info("🚀 Training model...")
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=300,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(50), progress_callback]
    )

    # Save model and preprocessors
    model.save_model(f'{model_dir}/edna_lgb_model.txt')
    preprocessor.save(model_dir)
    logger.info("✅ Model updated and saved.")

    # Evaluate
    y_pred_prob = model.predict(X_val)
    acc = np.mean(np.argmax(y_pred_prob, axis=1) == y_val)
    logger.info(f"📊 Validation accuracy: {acc:.4f}")

    return model, preprocessor

# ================= Main =================
if __name__ == "__main__":
    csv_files = [
        "C:/Users/loq/OneDrive/Desktop/SIH/.venv/data/test1o_part04.csv",
        "C:/Users/loq/OneDrive/Desktop/SIH/.venv/data/test1o_part05.csv"
    ]
    model, preprocessor = train_csv(csv_files)