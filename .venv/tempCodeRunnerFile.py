# ================= predict_csv.py (ENHANCED) =================
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle, os, logging
import json
from collections import defaultdict, Counter
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================= Preprocessor Loader (ENHANCED) =================
class eDNAPreprocessor:
    def __init__(self):
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = None
        self.taxonomy_hierarchy = None
        self.known_sequences = None
        self.known_labels = None
        self.feature_extractor_params = None
        self.known_features = None

    def load(self, input_dir='processed_data'):
        with open(f'{input_dir}/scaler.pkl','rb') as f:
            self.scaler = pickle.load(f)
        with open(f'{input_dir}/label_encoder.pkl','rb') as f:
            self.label_encoder = pickle.load(f)
        with open(f'{input_dir}/feature_columns.pkl','rb') as f:
            self.feature_columns = pickle.load(f)
        with open(f'{input_dir}/taxonomy_hierarchy.pkl','rb') as f:
            self.taxonomy_hierarchy = pickle.load(f)
        with open(f'{input_dir}/known_data.pkl','rb') as f:
            self.known_sequences, self.known_labels = pickle.load(f)
        with open(f'{input_dir}/feature_extractor_params.pkl','rb') as f:
            self.feature_extractor_params = pickle.load(f)
        
        # Precompute features for known sequences for exact matching
        extractor = FeatureExtractor()
        self.known_features = extractor.extract_features(
            self.known_sequences,
            k_values=self.feature_extractor_params.get('k_values', [4]),
            top_k=self.feature_extractor_params.get('top_k', 50)
        )
        # Ensure known features have same columns as feature_columns
        self.known_features = self._align_features(self.known_features)
        
        logger.info("✅ Preprocessors loaded.")
        logger.info(f"✅ Known sequences: {len(self.known_sequences)}")
        logger.info(f"✅ Known labels: {len(self.known_labels)}")

    def _align_features(self, feature_df):
        """Align features to match training feature columns"""
        feature_df = feature_df.copy()
        
        # Add missing columns with zeros using a more efficient approach
        missing_cols = set(self.feature_columns) - set(feature_df.columns)
        if missing_cols:
            # Create a DataFrame with zeros for missing columns
            missing_data = pd.DataFrame(0, index=feature_df.index, columns=list(missing_cols))
            # Concatenate instead of inserting one by one
            feature_df = pd.concat([feature_df, missing_data], axis=1)
        
        # Drop any extra columns and ensure correct order
        extra_cols = [c for c in feature_df.columns if c not in self.feature_columns]
        if extra_cols:
            feature_df = feature_df.drop(columns=extra_cols)
        
        return feature_df[self.feature_columns]

    def transform(self, feature_df):
        """Transform features using the same preprocessing as training"""
        feature_df = self._align_features(feature_df)
        return self.scaler.transform(feature_df)

# ================= Sequence Similarity Calculator =================
class SequenceSimilarityCalculator:
    def __init__(self, known_sequences, known_labels, known_features, scaler, feature_columns):
        self.known_sequences = known_sequences
        self.known_labels = known_labels
        self.known_features = known_features
        self.scaler = scaler
        self.feature_columns = feature_columns
    
    def calculate_similarity(self, seq1, seq2):
        """Calculate sequence similarity percentage"""
        if len(seq1) == 0 or len(seq2) == 0:
            return 0.0
        
        min_len = min(len(seq1), len(seq2))
        matches = 0
        
        for i in range(min_len):
            if seq1[i] == seq2[i]:
                matches += 1
        
        return (matches / min_len) * 100
    
    def find_top_matches(self, query_sequence, query_features, top_n=3):
        """Find top N most similar sequences from known data using both sequence and feature similarity"""
        similarities = []
        
        for i, (known_seq, known_label) in enumerate(zip(self.known_sequences, self.known_labels)):
            # Sequence similarity
            seq_similarity = self.calculate_similarity(query_sequence, known_seq)
            
            # Feature similarity (Euclidean distance)
            if self.known_features is not None and i < len(self.known_features):
                known_feat = self.known_features.iloc[i].values
                feature_distance = np.linalg.norm(known_feat - query_features)
                feature_similarity = 100 * (1 - min(feature_distance / 100, 1))  # Normalize to 0-100
            else:
                feature_similarity = seq_similarity
            
            # Combined similarity score
            combined_similarity = (seq_similarity * 0.7 + feature_similarity * 0.3)
            
            similarities.append({
                'sequence': known_seq,
                'species': known_label,
                'similarity': combined_similarity,
                'seq_similarity': seq_similarity,
                'feature_similarity': feature_similarity
            })
        
        # Sort by similarity descending and get top N
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_n]

    def find_exact_matches(self, query_sequence, query_features):
        """Check for exact matches in known sequences"""
        exact_matches = []
        
        for i, known_seq in enumerate(self.known_sequences):
            if query_sequence == known_seq:
                exact_matches.append({
                    'sequence': known_seq,
                    'species': self.known_labels[i],
                    'similarity': 100.0,
                    'seq_similarity': 100.0,
                    'feature_similarity': 100.0
                })
        
        return exact_matches

# ================= Feature Extractor =================
class FeatureExtractor:
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
        }

    def calculate_dinuc_features(self, sequence):
        seq_len = len(sequence)
        dinucs = {}
        for i in range(len(sequence)-1):
            dinuc = sequence[i:i+2]
            dinucs[dinuc] = dinucs.get(dinuc, 0) + 1
        
        # Normalize and create features
        return {f'di_{d}': (dinucs.get(d, 0) / max(seq_len-1, 1)) * 100
                for d in ['AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT',
                         'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT']}

    def extract_kmers(self, sequence, k=4):
        return Counter(sequence[i:i+k] for i in range(len(sequence)-k+1)) if len(sequence)>=k else Counter()

    def extract_features(self, sequences, k_values=[4], top_k=50):
        """Optimized feature extraction"""
        all_features = []
        for seq in sequences:
            feats = {}
            feats.update(self.calculate_nucleotide_features(seq))
            feats.update(self.calculate_dinuc_features(seq))
            
            for k in k_values:
                kmers = self.extract_kmers(seq, k)
                for kmer, count in dict(kmers.most_common(top_k)).items():
                    feats[f'{k}mer_{kmer}'] = count/max(len(seq),1)*1000
            all_features.append(feats)
        
        return pd.DataFrame(all_features).fillna(0)

# ================= Biodiversity Calculator =================
class BiodiversityCalculator:
    @staticmethod
    def calculate_alpha_diversity(results, sample_id="all"):
        """Calculate alpha diversity metrics for a sample"""
        if sample_id != "all":
            sample_results = [r for r in results if r.get('sample_id') == sample_id]
        else:
            sample_results = results
            
        species_counts = Counter([r['predicted_species'] for r in sample_results])
        richness = len(species_counts)
        
        total = sum(species_counts.values())
        shannon = 0
        for count in species_counts.values():
            p = count / total
            shannon -= p * np.log(p) if p > 0 else 0
            
        return {
            'richness': richness,
            'shannon_index': shannon,
            'evenness': shannon / np.log(richness) if richness > 0 else 0
        }
    
    @staticmethod
    def calculate_beta_diversity(results):
        """Calculate beta diversity between samples"""
        # Get unique sample IDs
        sample_ids = list(set([r['sample_id'] for r in results]))
        sample_ids.sort()
        
        # Create abundance matrix (samples x species)
        species_list = list(set([r['predicted_species'] for r in results]))
        abundance_matrix = np.zeros((len(sample_ids), len(species_list)))
        
        # Fill the abundance matrix
        species_to_idx = {species: i for i, species in enumerate(species_list)}
        sample_to_idx = {sample: i for i, sample in enumerate(sample_ids)}
        
        for result in results:
            sample_idx = sample_to_idx[result['sample_id']]
            species_idx = species_to_idx[result['predicted_species']]
            abundance_matrix[sample_idx, species_idx] += 1
        
        # Calculate Bray-Curtis dissimilarity
        bray_curtis_matrix = np.zeros((len(sample_ids), len(sample_ids)))
        for i in range(len(sample_ids)):
            for j in range(len(sample_ids)):
                if i == j:
                    bray_curtis_matrix[i, j] = 0.0
                else:
                    min_sum = np.minimum(abundance_matrix[i], abundance_matrix[j]).sum()
                    total_sum = abundance_matrix[i].sum() + abundance_matrix[j].sum()
                    bray_curtis_matrix[i, j] = 1 - (2 * min_sum) / total_sum if total_sum > 0 else 1
        
        return {
            'sample_ids': sample_ids,
            'bray_curtis_matrix': bray_curtis_matrix.tolist()
        }
    
    @staticmethod
    def calculate_rarefaction_curve(results, max_subsample=None):
        """Calculate rarefaction curve for species richness"""
        try:
            if max_subsample is None:
                max_subsample = len(results)
            
            if len(results) < 10:
                return {'x': [], 'y': []}
                
            # Sample all sequences at different sampling depths
            x = []
            y = []
            
            for subsample_size in range(10, max_subsample + 1, 10):
                if subsample_size > len(results):
                    break
                    
                # Take multiple random subsamples and average
                richness_values = []
                for _ in range(5):  # 5 random subsamples
                    subsample = np.random.choice(results, subsample_size, replace=False)
                    species_count = len(set([r['predicted_species'] for r in subsample]))
                    richness_values.append(species_count)
                
                x.append(subsample_size)
                y.append(np.mean(richness_values))
            
            return {'x': x, 'y': y}
        except Exception as e:
            logger.warning(f"Error calculating rarefaction curve: {e}")
            return {'x': [], 'y': []}

# ================= Ecological Role Predictor =================
class EcologicalRolePredictor:
    def __init__(self):
        self.functional_categories = {
            'primary_producer': ['diatom', 'algae', 'cyanobacteria', 'phytoplankton'],
            'symbiont': ['symbiont', 'endosymbiont', 'commensal'],
            'parasite': ['parasite', 'pathogen', 'parasitic'],
            'decomposer': ['bacteria', 'fungi', 'decomposer', 'saprophyte'],
        }
        
    def predict_ecological_role(self, species_name, sequence=None):
        """Predict ecological role based on species name"""
        roles = []
        species_lower = species_name.lower()
        
        for role, keywords in self.functional_categories.items():
            if any(keyword in species_lower for keyword in keywords):
                roles.append(role)
                
        return roles if roles else ['unknown']

# ================= Visualization Generator =================
class VisualizationGenerator:
    @staticmethod
    def create_taxonomic_barplot(results, output_path="taxonomic_barplot.html"):
        """Create interactive taxonomic bar plot"""
        df = pd.DataFrame(results)
        if 'phylum' not in df.columns:
            logger.warning("No phylum information available for taxonomic barplot")
            return None
            
        phylum_counts = df['phylum'].value_counts().reset_index()
        phylum_counts.columns = ['phylum', 'count']
        
        fig = px.bar(phylum_counts, x='phylum', y='count', 
                     title="Taxonomic Distribution by Phylum")
        fig.write_html(output_path)
        return output_path
    
    @staticmethod
    def create_network_graph(results, output_path="network_graph.html"):
        """Create a network graph showing relationships between samples and species"""
        try:
            # Create sample-species adjacency matrix
            df = pd.DataFrame(results)
            adjacency = pd.crosstab(df['sample_id'], df['predicted_species'])
            
            # Create network graph
            fig = px.imshow(adjacency, aspect="auto",
                           title="Sample-Species Association Network")
            fig.write_html(output_path)
            return output_path
        except Exception as e:
            logger.warning(f"Could not create network graph: {e}")
            return None
    
    @staticmethod
    def create_abundance_heatmap(results, output_path="abundance_heatmap.html"):
        """Create a heatmap of species abundance across samples"""
        try:
            df = pd.DataFrame(results)
            heatmap_data = pd.crosstab(df['sample_id'], df['predicted_species'])
            
            fig = px.imshow(heatmap_data, aspect="auto",
                           title="Species Abundance Heatmap")
            fig.write_html(output_path)
            return output_path
        except Exception as e:
            logger.warning(f"Could not create abundance heatmap: {e}")
            return None

    @staticmethod
    def create_rarefaction_curve(rarefaction_data, output_path="rarefaction_curve.html"):
        """Create rarefaction curve plot"""
        try:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rarefaction_data['x'], 
                y=rarefaction_data['y'],
                mode='lines+markers',
                name='Rarefaction Curve'
            ))
            fig.update_layout(
                title="Rarefaction Curve",
                xaxis_title="Number of Sequences",
                yaxis_title="Species Richness"
            )
            fig.write_html(output_path)
            return output_path
        except Exception as e:
            logger.warning(f"Could not create rarefaction curve: {e}")
            return None

# ================= JSON Exporter =================
class JSONExporter:
    @staticmethod
    def export_results(results, alpha_diversity, beta_diversity, rarefaction_curve, 
                      visualization_files, output_path="results_export.json"):
        """Export all results to a JSON file"""
        
        # Prepare the data structure
        export_data = {
            "results": results,
            "alpha_diversity": alpha_diversity,
            "beta_diversity": beta_diversity,
            "rarefaction_curve": rarefaction_curve,
            "visualization_files": visualization_files
        }
        
        # Write to JSON file
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"✅ Results exported to {output_path}")
        return output_path

# ================= Main Prediction Function (ENHANCED) =================
def predict_csv(input_csv, output_dir="prediction_results", model_dir="processed_data", 
                confidence_threshold=0.7, clustering_method='dbscan'):
    """Main function to predict species from CSV and export JSON results"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load preprocessors + model
    preprocessor = eDNAPreprocessor()
    preprocessor.load(model_dir)
    model = lgb.Booster(model_file=f"{model_dir}/edna_lgb_model.txt")
    logger.info("✅ Model loaded.")

    # Read input
    df = pd.read_csv(input_csv)
    if 'sequence' not in df.columns:
        raise ValueError("Input CSV must have a 'sequence' column")
    
    # Check for sample_id column
    sample_ids = df['sample_id'].tolist() if 'sample_id' in df.columns else ['default'] * len(df)
    sequences = df['sequence'].tolist()

    # Extract features
    extractor = FeatureExtractor()
    feature_df = extractor.extract_features(sequences, 
                                          k_values=preprocessor.feature_extractor_params.get('k_values', [4]),
                                          top_k=preprocessor.feature_extractor_params.get('top_k', 50))
    
    # Transform features (using fixed method)
    X = preprocessor.transform(feature_df)

    # Predict
    preds_prob = model.predict(X)
    preds = np.argmax(preds_prob, axis=1)
    predicted_species_names = preprocessor.label_encoder.inverse_transform(preds)
    
    # Initialize predictors
    similarity_calc = SequenceSimilarityCalculator(
        preprocessor.known_sequences, 
        preprocessor.known_labels,
        preprocessor.known_features,
        preprocessor.scaler,
        preprocessor.feature_columns
    )
    eco_predictor = EcologicalRolePredictor()
    
    # Process results
    results = []
    unknown_sequences = []
    
    # Get the set of known species from training data
    known_species_set = set(preprocessor.known_labels)
    
    for i, (seq, sample_id, pred_species_name, probs, features) in enumerate(zip(sequences, sample_ids, predicted_species_names, preds_prob, X)):
        # First check for exact matches in known sequences
        exact_matches = similarity_calc.find_exact_matches(seq, features)
        
        if exact_matches:
            # Exact match found - use the known species
            best_match = exact_matches[0]
            result = {
                'sequence_id': f"ASV_{i+1:04d}",
                'sample_id': sample_id,
                'sequence': seq,
                'predicted_species': best_match['species'],
                'confidence': 1.0,  # 100% confidence for exact matches
                'similarity_percentage': 100.0,
                'status': 'known',
                'exact_match': True,
                'top_predictions': [
                    {'species': match['species'], 'confidence': match['similarity'] / 100.0}
                    for match in exact_matches[:3]
                ]
            }
            
            # Add taxonomy if available
            if preprocessor.taxonomy_hierarchy and best_match['species'] in preprocessor.taxonomy_hierarchy:
                result.update(preprocessor.taxonomy_hierarchy[best_match['species']])
                
            results.append(result)
            continue
            
        # Find similar sequences for non-exact matches
        top_matches = similarity_calc.find_top_matches(seq, features, top_n=3)
        best_match = top_matches[0] if top_matches else {'similarity': 0, 'species': 'Unknown'}
        
        # Check if we have high similarity to known species
        if best_match['similarity'] >= 95.0 and best_match['species'] in known_species_set:
            # Known species with high similarity
            result = {
                'sequence_id': f"ASV_{i+1:04d}",
                'sample_id': sample_id,
                'sequence': seq,
                'predicted_species': best_match['species'],
                'confidence': best_match['similarity'] / 100.0,
                'similarity_percentage': best_match['similarity'],
                'status': 'known',
                'exact_match': False,
                'top_predictions': [
                    {'species': match['species'], 'confidence': match['similarity'] / 100.0}
                    for match in top_matches
                ]
            }
            
            # Add taxonomy if available
            if preprocessor.taxonomy_hierarchy and best_match['species'] in preprocessor.taxonomy_hierarchy:
                result.update(preprocessor.taxonomy_hierarchy[best_match['species']])
                
            results.append(result)
            
        elif np.max(probs) >= confidence_threshold:
            # Model prediction with confidence - check if predicted species is in known labels
            if pred_species_name in known_species_set:
                # This is a known species according to our training data
                top3_idx = np.argsort(probs)[-3:][::-1]
                top3_species = preprocessor.label_encoder.inverse_transform(top3_idx)
                top3_conf = probs[top3_idx]
                
                result = {
                    'sequence_id': f"ASV_{i+1:04d}",
                    'sample_id': sample_id,
                    'sequence': seq,
                    'predicted_species': pred_species_name,
                    'confidence': float(np.max(probs)),
                    'similarity_percentage': best_match['similarity'],
                    'status': 'known',
                    'exact_match': False,
                    'top_predictions': [
                        {'species': species, 'confidence': float(conf)} 
                        for species, conf in zip(top3_species, top3_conf)
                    ]
                }
                
                if preprocessor.taxonomy_hierarchy and pred_species_name in preprocessor.taxonomy_hierarchy:
                    result.update(preprocessor.taxonomy_hierarchy[pred_species_name])
                    
                results.append(result)
            else:
                # Predicted species is not in known labels, mark for clustering
                top3_idx = np.argsort(probs)[-3:][::-1]
                top3_species = preprocessor.label_encoder.inverse_transform(top3_idx)
                top3_conf = probs[top3_idx]
                
                unknown_sequences.append({
                    'sequence_id': f"ASV_{i+1:04d}",
                    'sample_id': sample_id,
                    'sequence': seq,
                    'features': features,
                    'top_similarities': top_matches,
                    'model_probs': probs,
                    'model_predictions': [
                        {'species': species, 'confidence': float(conf)} 
                        for species, conf in zip(top3_species, top3_conf)
                    ]
                })
        else:
            # Low confidence - mark for clustering
            top3_idx = np.argsort(probs)[-3:][::-1]
            top3_species = preprocessor.label_encoder.inverse_transform(top3_idx)
            top3_conf = probs[top3_idx]
            
            unknown_sequences.append({
                'sequence_id': f"ASV_{i+1:04d}",
                'sample_id': sample_id,
                'sequence': seq,
                'features': features,
                'top_similarities': top_matches,
                'model_probs': probs,
                'model_predictions': [
                    {'species': species, 'confidence': float(conf)} 
                    for species, conf in zip(top3_species, top3_conf)
                ]
            })
    
    # Cluster unknown sequences if any
    if unknown_sequences:
        logger.info(f"🔍 Clustering {len(unknown_sequences)} unknown sequences...")
        clustered_results = cluster_unknown_sequences(unknown_sequences, clustering_method)
        for result in clustered_results:
            result['ecological_roles'] = eco_predictor.predict_ecological_role(
                result['predicted_species'], result['sequence'])
            results.append(result)
    
    # Calculate biodiversity metrics
    biodiversity_calc = BiodiversityCalculator()
    alpha_diversity = biodiversity_calc.calculate_alpha_diversity(results)
    beta_diversity = biodiversity_calc.calculate_beta_diversity(results)
    
    # Fix rarefaction curve calculation
    try:
        rarefaction_curve = biodiversity_calc.calculate_rarefaction_curve(results, max_subsample=min(100, len(results)))
    except Exception as e:
        logger.warning(f"Could not calculate rarefaction curve: {e}")
        rarefaction_curve = {'x': [], 'y': []}
    
    # Generate visualizations
    viz_generator = VisualizationGenerator()
    viz_files = []
    
    barplot_path = viz_generator.create_taxonomic_barplot(results, f"{output_dir}/taxonomic_barplot.html")
    if barplot_path:
        viz_files.append(barplot_path)
    
    network_path = viz_generator.create_network_graph(results, f"{output_dir}/network_graph.html")
    if network_path:
        viz_files.append(network_path)
    
    heatmap_path = viz_generator.create_abundance_heatmap(results, f"{output_dir}/abundance_heatmap.html")
    if heatmap_path:
        viz_files.append(heatmap_path)
    
    rarefaction_path = viz_generator.create_rarefaction_curve(rarefaction_curve, f"{output_dir}/rarefaction_curve.html")
    if rarefaction_path:
        viz_files.append(rarefaction_path)
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_csv_path = f"{output_dir}/prediction_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    
    # Export to JSON
    json_exporter = JSONExporter()
    json_path = json_exporter.export_results(
        results, alpha_diversity, beta_diversity, rarefaction_curve, 
        viz_files, f"{output_dir}/results_export.json"
    )
    
    # Generate summary report
    summary_report = generate_summary_report(results, alpha_diversity, len(viz_files))
    with open(f"{output_dir}/summary_report.txt", "w") as f:
        f.write(summary_report)
    
    logger.info(f"✅ Analysis complete. Results saved to {output_dir}")
    logger.info(summary_report)
    
    return results_df
def cluster_unknown_sequences(unknown_sequences, method='dbscan'):
    """Cluster unknown sequences to identify novel taxa"""
    if not unknown_sequences:
        return []
    
    # Extract features for clustering
    X = np.array([seq['features'] for seq in unknown_sequences])
    
    # Reduce dimensionality
    n_components = min(10, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    
    # Apply clustering
    if method == 'dbscan':
        clustering = DBSCAN(eps=0.5, min_samples=2).fit(X_reduced)
    else:
        n_clusters = min(10, len(unknown_sequences))
        clustering = KMeans(n_clusters=n_clusters, random_state=42).fit(X_reduced)
    
    # Process results
    clustered_results = []
    
    for i, seq_data in enumerate(unknown_sequences):
        if method == 'dbscan':
            cluster_id = clustering.labels_[i]
            if cluster_id == -1:
                cluster_name = f"Novel_Singleton_{i}"
            else:
                cluster_name = f"Novel_Cluster_{cluster_id}"
        else:
            cluster_id = clustering.labels_[i]
            cluster_name = f"Novel_Cluster_{cluster_id}"
        
        # Include top model predictions in the result
        result = {
            'sequence_id': seq_data['sequence_id'],
            'sample_id': seq_data['sample_id'],
            'sequence': seq_data['sequence'],
            'predicted_species': cluster_name,
            'confidence': 0.5,  # Medium confidence for novel clusters
            'status': 'novel',
            'cluster_id': cluster_name,
            'top_predictions': seq_data.get('model_predictions', []),
            'kingdom': 'Unknown',
            'phylum': 'Unknown',
            'class': 'Unknown',
            'order': 'Unknown',
            'family': 'Unknown',
            'genus': 'Unknown'
        }
        
        clustered_results.append(result)
    
    return clustered_results

def generate_summary_report(results, alpha_diversity, num_viz_files):
    """Generate a textual summary report of the analysis"""
    num_sequences = len(results)
    known_count = sum(1 for r in results if r['status'] == 'known')
    novel_count = sum(1 for r in results if r['status'] == 'novel')
    
    species_counts = Counter([r['predicted_species'] for r in results])
    top_species = species_counts.most_common(5)
    
    # Count ecological roles
    eco_roles = Counter()
    for r in results:
        for role in r.get('ecological_roles', ['unknown']):
            eco_roles[role] += 1
    
    report = f"""
    eDNA ANALYSIS SUMMARY REPORT
    ============================
    
    Sequences Processed: {num_sequences}
    - Known species: {known_count} ({known_count/num_sequences*100:.1f}%)
    - Novel clusters: {novel_count} ({novel_count/num_sequences*100:.1f}%)
    
    Biodiversity Metrics:
    - Species Richness: {alpha_diversity['richness']}
    - Shannon Diversity Index: {alpha_diversity['shannon_index']:.3f}
    - Evenness: {alpha_diversity['evenness']:.3f}
    
    Top 5 Most Abundant Species:
    """
    
    for i, (species, count) in enumerate(top_species, 1):
        report += f"    {i}. {species}: {count} sequences\n"
    
    report += f"""
    Ecological Role Distribution:
    """
    
    for role, count in eco_roles.most_common():
        report += f"    - {role}: {count} sequences\n"
    
    report += f"""
    Output Files:
    - Results CSV: prediction_results.csv
    - JSON Export: results_export.json
    - Visualizations: {num_viz_files} interactive HTML files
    - This summary: summary_report.txt
    
    ============================
    Analysis completed successfully.
    """
    
    return report
# ================= Main =================
if __name__ == "__main__":
    input_csv = "data/predict.csv"
    results_df = predict_csv(
        input_csv, 
        output_dir="prediction_results",
        confidence_threshold=0.6
    )
    
    print(f"✅ Prediction complete. Results saved to 'prediction_results' folder.")