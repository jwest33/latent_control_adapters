import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

class ControlVectorFeasibilityAnalyzer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
    
    def extract_activations(self, texts: List[str], layer_idx: int, 
                           position: str = 'last') -> torch.Tensor:
        """
        Extract activations from a specific layer.
        
        Args:
            texts: List of text inputs
            layer_idx: Which transformer layer to extract from
            position: 'last', 'mean', or 'first' token position
        """
        activations = []
        
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", 
                                   truncation=True, max_length=512).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[layer_idx]  # [batch, seq, hidden]
                
                if position == 'last':
                    act = hidden_states[0, -1, :]  # Last token
                elif position == 'mean':
                    act = hidden_states[0].mean(dim=0)  # Mean over sequence
                elif position == 'first':
                    act = hidden_states[0, 0, :]  # First token
                
                activations.append(act.cpu())
        
        return torch.stack(activations)
    
    def compute_class_separation_metrics(self,
                                         high_entropy_acts: torch.Tensor,
                                         low_entropy_acts: torch.Tensor) -> Dict:
        """
        Compute various metrics for class separability.
        """
        # Convert to numpy and use float64 to prevent overflow
        high_acts = high_entropy_acts.numpy().astype(np.float64)
        low_acts = low_entropy_acts.numpy().astype(np.float64)

        # Combine and create labels
        all_acts = np.vstack([high_acts, low_acts])
        labels = np.array([1] * len(high_acts) + [0] * len(low_acts))

        metrics = {}

        # 1. Mean vector distance (cosine and euclidean)
        high_mean = high_acts.mean(axis=0)
        low_mean = low_acts.mean(axis=0)

        # Normalize vectors to prevent overflow in cosine calculation
        high_mean_norm = high_mean / (np.linalg.norm(high_mean) + 1e-10)
        low_mean_norm = low_mean / (np.linalg.norm(low_mean) + 1e-10)
        metrics['mean_cosine_distance'] = cosine(high_mean_norm, low_mean_norm)
        metrics['mean_euclidean_distance'] = np.linalg.norm(high_mean - low_mean)
        
        # Normalize by std to get effect size
        pooled_std = np.sqrt((high_acts.std(axis=0)**2 + low_acts.std(axis=0)**2) / 2).mean()
        metrics['cohens_d'] = metrics['mean_euclidean_distance'] / (pooled_std + 1e-10)
        
        # 2. Silhouette score (how well separated are clusters)
        if len(all_acts) > 2:
            metrics['silhouette_score'] = silhouette_score(all_acts, labels)
        
        # 3. Linear separability via LDA
        lda = LinearDiscriminantAnalysis()
        try:
            lda.fit(all_acts, labels)
            lda_score = lda.score(all_acts, labels)
            metrics['lda_accuracy'] = lda_score
            
            # LDA finds the best linear separator - this is like a control vector!
            metrics['lda_explained_variance'] = lda.explained_variance_ratio_[0]
        except:
            metrics['lda_accuracy'] = None
            metrics['lda_explained_variance'] = None
        
        # 4. Within-class vs between-class variance ratio
        high_var = high_acts.var(axis=0).mean()
        low_var = low_acts.var(axis=0).mean()
        between_var = ((high_mean - low_mean) ** 2).mean()
        
        metrics['within_class_variance'] = (high_var + low_var) / 2
        metrics['between_class_variance'] = between_var
        metrics['variance_ratio'] = between_var / ((high_var + low_var) / 2 + 1e-10)
        
        return metrics
    
    def compute_intrinsic_dimensionality(self,
                                         high_entropy_acts: torch.Tensor,
                                         low_entropy_acts: torch.Tensor,
                                         variance_threshold: float = 0.95) -> Dict:
        """
        Determine if the class difference is low-dimensional.
        """
        high_acts = high_entropy_acts.numpy()
        low_acts = low_entropy_acts.numpy()
        
        # Compute difference vectors
        diff_vectors = high_acts - low_acts
        
        # PCA on difference vectors
        pca = PCA()
        pca.fit(diff_vectors)
        
        # How many dimensions to explain variance_threshold of variance?
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_dims = np.argmax(cumsum >= variance_threshold) + 1
        
        results = {
            'intrinsic_dimensionality': n_dims,
            'first_pc_variance': pca.explained_variance_ratio_[0],
            'top_5_pc_variance': pca.explained_variance_ratio_[:5].sum(),
            'explained_variance_ratio': pca.explained_variance_ratio_[:10].tolist()
        }
        
        return results
    
    def analyze_layer_suitability(self,
                                  high_entropy_prompts: List[str],
                                  low_entropy_prompts: List[str],
                                  layer_range: Tuple[int, int] = None,
                                  position: str = 'last') -> Dict[int, Dict]:
        """
        Analyze all layers to find which are most suitable for control vectors.
        """
        if layer_range is None:
            n_layers = self.model.config.num_hidden_layers
            layer_range = (0, n_layers)
        
        results = {}
        
        for layer_idx in range(layer_range[0], layer_range[1]):
            print(f"Analyzing layer {layer_idx}...")
            
            # Extract activations
            high_acts = self.extract_activations(high_entropy_prompts, layer_idx, position)
            low_acts = self.extract_activations(low_entropy_prompts, layer_idx, position)
            
            # Compute metrics
            separation_metrics = self.compute_class_separation_metrics(high_acts, low_acts)
            dimensionality_metrics = self.compute_intrinsic_dimensionality(high_acts, low_acts)
            
            results[layer_idx] = {
                **separation_metrics,
                **dimensionality_metrics
            }
        
        return results
    
    def visualize_activation_space(self,
                                   high_entropy_prompts: List[str],
                                   low_entropy_prompts: List[str],
                                   layer_idx: int,
                                   save_path: str = None):
        """
        Visualize the activation space using PCA.
        """
        # Extract activations
        high_acts = self.extract_activations(high_entropy_prompts, layer_idx).numpy()
        low_acts = self.extract_activations(low_entropy_prompts, layer_idx).numpy()
        
        # Combine
        all_acts = np.vstack([high_acts, low_acts])
        
        # PCA to 2D
        pca = PCA(n_components=2)
        projected = pca.fit_transform(all_acts)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.scatter(projected[:len(high_acts), 0], 
                   projected[:len(high_acts), 1],
                   c='red', label='High Entropy', alpha=0.6, s=100)
        plt.scatter(projected[len(high_acts):, 0], 
                   projected[len(high_acts):, 1],
                   c='blue', label='Low Entropy', alpha=0.6, s=100)
        
        # Draw arrow for mean difference (control vector direction)
        high_mean = projected[:len(high_acts)].mean(axis=0)
        low_mean = projected[len(high_acts):].mean(axis=0)
        plt.arrow(low_mean[0], low_mean[1], 
                 (high_mean[0] - low_mean[0]) * 0.8,
                 (high_mean[1] - low_mean[1]) * 0.8,
                 head_width=0.3, head_length=0.2, fc='green', ec='green',
                 linewidth=2, label='Control Vector Direction')
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title(f'Activation Space at Layer {layer_idx}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, layer_results: Dict[int, Dict]) -> str:
        """
        Generate a human-readable report on feasibility.
        """
        report = ["=" * 80]
        report.append("CONTROL VECTOR FEASIBILITY ANALYSIS")
        report.append("=" * 80)
        report.append("")
        
        # Find best layers
        best_layer_lda = max(layer_results.items(), 
                             key=lambda x: x[1].get('lda_accuracy', 0) or 0)
        best_layer_sil = max(layer_results.items(),
                             key=lambda x: x[1].get('silhouette_score', -1))
        best_layer_dim = min(layer_results.items(),
                             key=lambda x: x[1]['intrinsic_dimensionality'])
        
        report.append(f"Best layer by LDA accuracy: Layer {best_layer_lda[0]} "
                     f"(accuracy: {best_layer_lda[1].get('lda_accuracy', 0):.3f})")
        report.append(f"Best layer by silhouette score: Layer {best_layer_sil[0]} "
                     f"(score: {best_layer_sil[1]['silhouette_score']:.3f})")
        report.append(f"Most low-dimensional: Layer {best_layer_dim[0]} "
                     f"(dims: {best_layer_dim[1]['intrinsic_dimensionality']})")
        report.append("")
        
        # Feasibility assessment
        report.append("FEASIBILITY ASSESSMENT:")
        report.append("-" * 80)
        
        avg_lda = np.mean([r.get('lda_accuracy', 0) or 0 
                          for r in layer_results.values()])
        avg_sil = np.mean([r['silhouette_score'] 
                          for r in layer_results.values()])
        avg_dim = np.mean([r['intrinsic_dimensionality'] 
                          for r in layer_results.values()])
        avg_first_pc = np.mean([r['first_pc_variance'] 
                               for r in layer_results.values()])
        
        report.append(f"Average LDA accuracy: {avg_lda:.3f}")
        report.append(f"Average silhouette score: {avg_sil:.3f}")
        report.append(f"Average intrinsic dimensionality: {avg_dim:.1f}")
        report.append(f"Average 1st PC variance explained: {avg_first_pc:.3f}")
        report.append("")
        
        # Interpretation
        report.append("INTERPRETATION:")
        report.append("-" * 80)
        
        if avg_lda > 0.85 and avg_sil > 0.3:
            report.append("✓ EXCELLENT: Classes are highly separable.")
            report.append("  Control vectors should work very well.")
        elif avg_lda > 0.75 and avg_sil > 0.2:
            report.append("✓ GOOD: Classes are reasonably separable.")
            report.append("  Control vectors should be effective.")
        elif avg_lda > 0.65:
            report.append("⚠ MARGINAL: Classes have some overlap.")
            report.append("  Control vectors may work but with reduced effectiveness.")
        else:
            report.append("✗ POOR: Classes are not well separated.")
            report.append("  Control vectors may not be effective for this task.")
        report.append("")
        
        if avg_first_pc > 0.6:
            report.append("✓ The class difference is highly 1-dimensional.")
            report.append("  A single control vector direction should suffice.")
        elif avg_first_pc > 0.4:
            report.append("⚠ The class difference is somewhat multi-dimensional.")
            report.append("  May need multiple control vectors or higher scales.")
        else:
            report.append("✗ The class difference is highly multi-dimensional.")
            report.append("  Single control vectors may not capture the full difference.")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)

    def identify_outliers(self,
                        high_entropy_prompts: List[str],
                        low_entropy_prompts: List[str],
                        layer_idx: int) -> Dict:
        """
        Identify which prompts are outliers (misclassified by the separation).
        """
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        
        # Extract activations
        high_acts = self.extract_activations(high_entropy_prompts, layer_idx).numpy()
        low_acts = self.extract_activations(low_entropy_prompts, layer_idx).numpy()
        
        # Combine and create labels
        all_acts = np.vstack([high_acts, low_acts])
        labels = np.array([1] * len(high_acts) + [0] * len(low_acts))
        
        # Fit LDA
        lda = LinearDiscriminantAnalysis()
        lda.fit(all_acts, labels)
        predictions = lda.predict(all_acts)
        
        # Find misclassified examples
        misclassified_high = []
        misclassified_low = []
        
        for i in range(len(high_acts)):
            if predictions[i] == 0:  # Predicted as low entropy
                misclassified_high.append({
                    'index': i,
                    'prompt': high_entropy_prompts[i],
                    'predicted_as': 'low_entropy'
                })
        
        for i in range(len(low_acts)):
            if predictions[len(high_acts) + i] == 1:  # Predicted as high entropy
                misclassified_low.append({
                    'index': i,
                    'prompt': low_entropy_prompts[i],
                    'predicted_as': 'high_entropy'
                })
        
        return {
            'misclassified_high_entropy': misclassified_high,
            'misclassified_low_entropy': misclassified_low
        }

    def visualize_activation_space_with_labels(self,
                                            high_entropy_prompts: List[str],
                                            low_entropy_prompts: List[str],
                                            layer_idx: int,
                                            save_path: str = None,
                                            max_label_length: int = 50):
        """
        Visualize the activation space using PCA with prompt labels.
        """
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        
        # Extract activations
        high_acts = self.extract_activations(high_entropy_prompts, layer_idx).numpy()
        low_acts = self.extract_activations(low_entropy_prompts, layer_idx).numpy()
        
        # Combine
        all_acts = np.vstack([high_acts, low_acts])
        
        # PCA to 2D
        pca = PCA(n_components=2)
        projected = pca.fit_transform(all_acts)
        
        # Plot
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Plot high entropy points
        ax.scatter(projected[:len(high_acts), 0], 
                projected[:len(high_acts), 1],
                c='red', label='High Entropy', alpha=0.6, s=100)
        
        # Plot low entropy points
        ax.scatter(projected[len(high_acts):, 0], 
                projected[len(high_acts):, 1],
                c='blue', label='Low Entropy', alpha=0.6, s=100)
        
        # Add labels for each point
        for i, prompt in enumerate(high_entropy_prompts):
            label = prompt[:max_label_length] + "..." if len(prompt) > max_label_length else prompt
            ax.annotate(f"H{i}: {label}", 
                    (projected[i, 0], projected[i, 1]),
                    fontsize=7, alpha=0.7, color='darkred',
                    xytext=(5, 5), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        for i, prompt in enumerate(low_entropy_prompts):
            label = prompt[:max_label_length] + "..." if len(prompt) > max_label_length else prompt
            ax.annotate(f"L{i}: {label}", 
                    (projected[len(high_acts) + i, 0], projected[len(high_acts) + i, 1]),
                    fontsize=7, alpha=0.7, color='darkblue',
                    xytext=(5, 5), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Draw arrow for mean difference (control vector direction)
        high_mean = projected[:len(high_acts)].mean(axis=0)
        low_mean = projected[len(high_acts):].mean(axis=0)
        ax.arrow(low_mean[0], low_mean[1], 
                (high_mean[0] - low_mean[0]) * 0.8,
                (high_mean[1] - low_mean[1]) * 0.8,
                head_width=2, head_length=1.5, fc='green', ec='green',
                linewidth=2, label='Control Vector Direction')
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax.set_title(f'Activation Space at Layer {layer_idx} (Labeled)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# Usage example
def run_feasibility_analysis(model, tokenizer, high_entropy_prompts, low_entropy_prompts):
    """
    Complete feasibility analysis pipeline.
    """
    analyzer = ControlVectorFeasibilityAnalyzer(model, tokenizer)
    
    # Analyze all layers
    print("Analyzing layers (this may take a while)...")
    layer_results = analyzer.analyze_layer_suitability(
        high_entropy_prompts,
        low_entropy_prompts,
        layer_range=(10, 30),  # Often middle-to-late layers work best
        position='last'
    )
    
    # Generate report
    report = analyzer.generate_report(layer_results)
    print(report)
    
    # Visualize best layers
    best_layers = sorted(layer_results.items(), 
                        key=lambda x: x[1].get('lda_accuracy', 0) or 0,
                        reverse=True)[:3]
    
    for layer_idx, metrics in best_layers:
        print(f"\nVisualizing layer {layer_idx}...")
        analyzer.visualize_activation_space(
            high_entropy_prompts,
            low_entropy_prompts,
            layer_idx,
            save_path=f'layer_{layer_idx}_visualization.png'
        )
    
    return layer_results, analyzer
