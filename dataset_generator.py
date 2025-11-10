import requests
import json
import subprocess
import time
import sys
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import Counter
from dataclasses import dataclass
import re


@dataclass
class QuestionResponseSet:
    """Represents a question with multiple responses and computed metrics."""
    question: str
    responses: List[str]
    question_entropy: float
    response_entropy: float
    most_common_response: str
    response_cluster_sizes: List[int]


class EntropyDatasetGenerator:
    def __init__(self, model_path: str, port: int = 8080):
        self.model_path = model_path
        self.port = port
        self.base_url = f"http://localhost:{port}"
        self.server_process = None
        
    def start_server(self):
        """Start llama-server process."""
        print(f"Starting llama-server on port {self.port}...")
        
        cmd = [
            "llama-server",
            "-m", self.model_path,
            "--port", str(self.port),
            "--ctx-size", "8192",
            "--n-gpu-layers", "99",
            "--threads", "8",
            "--batch-size", "512",
        ]
        
        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to be ready
        max_retries = 30
        for i in range(max_retries):
            try:
                response = requests.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    print("Server is ready!")
                    return
            except requests.exceptions.ConnectionError:
                print(f"Waiting for server... ({i+1}/{max_retries})")
                time.sleep(2)
        
        raise RuntimeError("Server failed to start")
    
    def stop_server(self):
        """Stop llama-server process."""
        if self.server_process:
            print("Stopping server...")
            self.server_process.terminate()
            self.server_process.wait()
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 512,
                 seed: Optional[int] = None) -> str:
        """Generate text using llama-server API."""
        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "n_predict": max_tokens,
            "seed": seed,
            "stop": [],
        }

        response = requests.post(
            f"{self.base_url}/completion",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            response_json = response.json()
            content = response_json.get("content", "")
            return content.strip()
        else:
            raise RuntimeError(f"Generation failed: {response.status_code} - {response.text}")
    
    def generate_questions(self, domains: List[str], n_questions: int = 100,
                          questions_per_batch: int = 10) -> List[str]:
        """Generate diverse questions across multiple domains.
        
        Focuses on speculative, uncertain, and high-entropy questions that
        models might hedge on or give unconfident responses to.
        """
        
        system_prompt = """Generate diverse, challenging questions that typically result in speculative or uncertain responses. Focus on:

1. Future predictions and forecasts
2. Hypothetical scenarios and counterfactuals
3. Questions about unknowable personal information
4. Questions requiring speculation or estimation
5. Questions about ongoing uncertain situations
6. Questions with multiple valid interpretations
7. Questions about emerging or poorly understood phenomena
8. Questions that combine multiple uncertain factors

Make questions natural and varied in structure. Avoid repetitive patterns.

Format: One question per line, no numbering or prefixes.

Example questions:
What will AI capabilities look like in 2030?
How would history have changed if the printing press was invented 200 years earlier?
What factors will determine the next major technological breakthrough?
How many people are thinking about quantum computing right now?
What emerging technology will have the biggest impact on daily life in the next decade?"""

        questions = []
        domain_idx = 0
        batch_num = 0
        
        while len(questions) < n_questions:
            batch_num += 1
            questions_to_request = min(questions_per_batch, n_questions - len(questions))
            
            print(f"\n{'='*80}")
            print(f"BATCH {batch_num}: Generating {questions_to_request} questions")
            print(f"Current progress: {len(questions)}/{n_questions} questions")
            print(f"{'='*80}")
            
            # Select domains for this batch
            batch_domains = []
            for _ in range(min(3, len(domains))):
                batch_domains.append(domains[domain_idx % len(domains)])
                domain_idx += 1
            
            domains_text = ", ".join(batch_domains)
            
            prompt = f"""{system_prompt}

Generate {questions_to_request} diverse questions related to these domains: {domains_text}

Questions:"""
            
            print(f"Generating with domains: {domains_text}")
            response = self.generate(prompt, temperature=0.9, max_tokens=1024)
            
            # Parse questions from response
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            
            for line in lines:
                # Clean up any numbering or prefixes
                cleaned = re.sub(r'^\d+[\.)]\s*', '', line)
                cleaned = re.sub(r'^[-•*]\s*', '', cleaned)
                cleaned = cleaned.strip()
                
                # Basic validation: should be a question
                if cleaned and '?' in cleaned and len(cleaned) > 20:
                    questions.append(cleaned)
                    print(f"  ✓ Added: {cleaned[:80]}...")
                    
                    if len(questions) >= n_questions:
                        break
            
            print(f"Batch {batch_num} complete: {len(questions)}/{n_questions} questions")
        
        return questions[:n_questions]
    
    def generate_multiple_responses(self, question: str, n_responses: int = 10,
                                   temperature: float = 0.8) -> List[str]:
        """Generate multiple responses to a question using different seeds."""
        
        prompt = f"""Answer the following question directly and concisely:

{question}

Answer:"""
        
        responses = []
        print(f"\n  Generating {n_responses} responses to: {question[:60]}...")
        
        for i in range(n_responses):
            seed = random.randint(0, 2**31 - 1)
            try:
                response = self.generate(prompt, temperature=temperature, max_tokens=256, seed=seed)
                if response:
                    responses.append(response)
                    print(f"    Response {i+1}/{n_responses}: {response[:60]}...")
            except Exception as e:
                print(f"    Error generating response {i+1}: {e}")
                continue
        
        return responses
    
    def calculate_response_diversity_entropy(self, responses: List[str]) -> float:
        """Calculate entropy based on response diversity using simple string similarity.
        
        Higher entropy = more diverse responses = more uncertainty in the question.
        """
        if len(responses) < 2:
            return 0.0
        
        # Normalize responses for comparison
        normalized = [self._normalize_response(r) for r in responses]
        
        # Count unique normalized responses
        response_counts = Counter(normalized)
        total = len(normalized)
        
        # Calculate Shannon entropy
        entropy = 0.0
        for count in response_counts.values():
            prob = count / total
            if prob > 0:
                entropy -= prob * np.log2(prob)
        
        return entropy
    
    def _normalize_response(self, response: str) -> str:
        """Normalize response for similarity comparison."""
        # Convert to lowercase
        normalized = response.lower()
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        # Remove common punctuation variations
        normalized = normalized.rstrip('.')
        
        # Take first 100 chars to focus on main content
        normalized = normalized[:100]
        
        return normalized
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple word-overlap based similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def cluster_responses(self, responses: List[str], similarity_threshold: float = 0.5) -> Tuple[str, List[int]]:
        """Cluster responses by semantic similarity and return the most common response.
        
        Returns:
            Tuple of (most_common_response, list of cluster sizes)
        """
        if not responses:
            return "", []
        
        if len(responses) == 1:
            return responses[0], [1]
        
        # Simple clustering: group responses that are similar enough
        clusters = []
        
        for response in responses:
            # Find best matching cluster
            best_cluster_idx = -1
            best_similarity = 0.0
            
            for idx, cluster in enumerate(clusters):
                # Compare with cluster representative (first item)
                similarity = self._calculate_semantic_similarity(response, cluster[0])
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster_idx = idx
            
            # Add to best cluster if similar enough, otherwise create new cluster
            if best_similarity >= similarity_threshold and best_cluster_idx >= 0:
                clusters[best_cluster_idx].append(response)
            else:
                clusters.append([response])
        
        # Find largest cluster
        cluster_sizes = [len(cluster) for cluster in clusters]
        largest_cluster_idx = cluster_sizes.index(max(cluster_sizes))
        most_common_response = clusters[largest_cluster_idx][0]
        
        return most_common_response, cluster_sizes
    
    def calculate_question_entropy(self, question: str) -> float:
        """Estimate question entropy based on linguistic features.
        
        Questions with higher inherent uncertainty tend to have:
        - Future tense or modal verbs (will, would, could, might)
        - Uncertainty markers (estimate, predict, likely, probably)
        - Speculative language
        """
        question_lower = question.lower()
        
        # Features that indicate high entropy questions
        future_markers = ['will', 'would', 'could', 'might', 'may', 'can', 'should']
        uncertainty_markers = ['estimate', 'predict', 'forecast', 'expect', 'likely', 
                              'probably', 'possibly', 'perhaps', 'potential', 'uncertain']
        speculative_markers = ['if', 'suppose', 'imagine', 'assume', 'hypothetical']
        
        score = 0.0
        
        # Count markers
        words = question_lower.split()
        for word in words:
            if word in future_markers:
                score += 0.3
            if word in uncertainty_markers:
                score += 0.4
            if word in speculative_markers:
                score += 0.5
        
        # Normalize by question length (favor longer, complex questions)
        length_factor = min(len(words) / 20.0, 1.0)
        score *= (0.5 + length_factor * 0.5)
        
        return score
    
    def process_question_with_responses(self, question: str, n_responses: int = 10) -> QuestionResponseSet:
        """Generate responses for a question and calculate all metrics."""
        
        # Generate multiple responses
        responses = self.generate_multiple_responses(question, n_responses=n_responses)
        
        if not responses:
            print(f"  ✗ No valid responses generated for question")
            return None
        
        # Calculate question entropy (linguistic features)
        question_entropy = self.calculate_question_entropy(question)
        
        # Calculate response diversity entropy
        response_entropy = self.calculate_response_diversity_entropy(responses)
        
        # Cluster responses and find most common
        most_common, cluster_sizes = self.cluster_responses(responses)
        
        print(f"  Question entropy: {question_entropy:.3f}")
        print(f"  Response entropy: {response_entropy:.3f}")
        print(f"  Cluster sizes: {cluster_sizes}")
        print(f"  Most common response: {most_common[:80]}...")
        
        return QuestionResponseSet(
            question=question,
            responses=responses,
            question_entropy=question_entropy,
            response_entropy=response_entropy,
            most_common_response=most_common,
            response_cluster_sizes=cluster_sizes
        )
    
    def generate_high_entropy_dataset(self, domains: List[str], 
                                     n_questions: int = 100,
                                     n_responses_per_question: int = 10,
                                     entropy_percentile: int = 70) -> List[QuestionResponseSet]:
        """Generate complete high-entropy dataset.
        
        Args:
            domains: List of domain areas for question generation
            n_questions: Number of questions to initially generate
            n_responses_per_question: Number of responses to generate per question
            entropy_percentile: Keep questions above this percentile of combined entropy
        
        Returns:
            List of QuestionResponseSet objects sorted by entropy (highest first)
        """
        
        # Step 1: Generate questions
        print(f"\n{'='*80}")
        print("STEP 1: Generating Questions")
        print(f"{'='*80}")
        questions = self.generate_questions(domains, n_questions=n_questions)
        print(f"\n✓ Generated {len(questions)} questions")
        
        # Step 2: Process each question (generate responses and calculate metrics)
        print(f"\n{'='*80}")
        print("STEP 2: Generating Responses and Calculating Entropy")
        print(f"{'='*80}")
        
        question_sets = []
        for i, question in enumerate(questions):
            print(f"\nProcessing question {i+1}/{len(questions)}:")
            print(f"  Q: {question}")
            
            qrs = self.process_question_with_responses(question, n_responses=n_responses_per_question)
            if qrs:
                question_sets.append(qrs)
        
        print(f"\n✓ Successfully processed {len(question_sets)} questions")
        
        # Step 3: Filter for high entropy questions
        print(f"\n{'='*80}")
        print("STEP 3: Filtering for High Entropy Questions")
        print(f"{'='*80}")
        
        # Calculate combined entropy score (weighted combination)
        for qrs in question_sets:
            # Weight response entropy more heavily as it indicates actual model uncertainty
            qrs.combined_entropy = (0.3 * qrs.question_entropy + 0.7 * qrs.response_entropy)
        
        # Sort by combined entropy
        question_sets.sort(key=lambda x: x.combined_entropy, reverse=True)
        
        # Calculate percentile threshold
        cutoff_idx = int(len(question_sets) * (100 - entropy_percentile) / 100)
        high_entropy_sets = question_sets[:len(question_sets) - cutoff_idx]
        
        print(f"\nEntropy statistics:")
        print(f"  Total questions: {len(question_sets)}")
        print(f"  Keeping top {100 - entropy_percentile}% (above {entropy_percentile}th percentile)")
        print(f"  High entropy questions: {len(high_entropy_sets)}")
        
        if high_entropy_sets:
            print(f"\n  Highest entropy: {high_entropy_sets[0].combined_entropy:.3f}")
            print(f"    Q: {high_entropy_sets[0].question}")
            print(f"    A: {high_entropy_sets[0].most_common_response[:100]}...")
            
            print(f"\n  Lowest (kept) entropy: {high_entropy_sets[-1].combined_entropy:.3f}")
            print(f"    Q: {high_entropy_sets[-1].question}")
            print(f"    A: {high_entropy_sets[-1].most_common_response[:100]}...")
        
        return high_entropy_sets
    
    def save_dataset(self, question_sets: List[QuestionResponseSet], 
                    output_dir: str = "prompts"):
        """Save high-entropy dataset to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save as simple Q&A pairs (one per line format)
        qa_path = output_path / "high_entropy_qa.txt"
        with open(qa_path, 'w', encoding='utf-8') as f:
            for qrs in question_sets:
                f.write(f"{qrs.question} {qrs.most_common_response}\n")
        
        print(f"\n✓ Saved {len(question_sets)} Q&A pairs to {qa_path}")
        
        # Save detailed JSON with all metadata
        json_path = output_path / "high_entropy_dataset_detailed.json"
        detailed_data = []
        for i, qrs in enumerate(question_sets):
            detailed_data.append({
                "index": i,
                "question": qrs.question,
                "most_common_response": qrs.most_common_response,
                "question_entropy": qrs.question_entropy,
                "response_entropy": qrs.response_entropy,
                "combined_entropy": qrs.combined_entropy,
                "num_responses": len(qrs.responses),
                "cluster_sizes": qrs.response_cluster_sizes,
                "num_clusters": len(qrs.response_cluster_sizes),
                "all_responses": qrs.responses
            })
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved detailed dataset to {json_path}")
        
        # Save statistics summary
        stats_path = output_path / "dataset_statistics.txt"
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("High Entropy Dataset Statistics\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total Q&A pairs: {len(question_sets)}\n\n")
            
            entropies = [qrs.combined_entropy for qrs in question_sets]
            f.write(f"Combined Entropy Statistics:\n")
            f.write(f"  Mean: {np.mean(entropies):.3f}\n")
            f.write(f"  Median: {np.median(entropies):.3f}\n")
            f.write(f"  Std Dev: {np.std(entropies):.3f}\n")
            f.write(f"  Min: {np.min(entropies):.3f}\n")
            f.write(f"  Max: {np.max(entropies):.3f}\n\n")
            
            response_entropies = [qrs.response_entropy for qrs in question_sets]
            f.write(f"Response Entropy Statistics:\n")
            f.write(f"  Mean: {np.mean(response_entropies):.3f}\n")
            f.write(f"  Median: {np.median(response_entropies):.3f}\n")
            f.write(f"  Min: {np.min(response_entropies):.3f}\n")
            f.write(f"  Max: {np.max(response_entropies):.3f}\n\n")
            
            num_clusters = [len(qrs.response_cluster_sizes) for qrs in question_sets]
            f.write(f"Response Clustering Statistics:\n")
            f.write(f"  Mean clusters per question: {np.mean(num_clusters):.1f}\n")
            f.write(f"  Median clusters: {np.median(num_clusters):.0f}\n")
            f.write(f"  Max clusters: {np.max(num_clusters)}\n")
        
        print(f"✓ Saved statistics to {stats_path}")


def main():
    # Configuration
    MODEL_PATH = r"D:\models\Qwen3-30B-A3B-Instruct-2507\Qwen3-30B-A3B-Instruct-2507-Q6_K.gguf"
    N_QUESTIONS = 100  # Initial number of questions to generate
    N_RESPONSES_PER_QUESTION = 10  # Responses to generate per question for entropy calculation
    ENTROPY_PERCENTILE = 70  # Keep questions above 70th percentile
    PORT = 8080
    
    # Diverse domains for question generation
    domains = [
        "Future Technology and AI",
        "Climate and Environmental Change",
        "Economic Trends and Markets",
        "Political Developments",
        "Space Exploration and Astronomy",
        "Medical Breakthroughs",
        "Social and Cultural Shifts",
        "Scientific Discoveries",
        "Personal Information and Privacy",
        "Hypothetical Scenarios",
        "Emerging Technologies",
        "Global Events and Geopolitics",
        "Innovation and Entrepreneurship",
        "Energy and Sustainability",
        "Biotechnology and Genetics",
        "Quantum Computing",
        "Neuroscience and Consciousness",
        "Artificial General Intelligence",
        "Urban Development and Smart Cities",
        "Education and Learning",
    ]
    
    generator = EntropyDatasetGenerator(MODEL_PATH, PORT)
    
    try:
        # Start server
        generator.start_server()
        
        # Generate high-entropy dataset
        print(f"\n{'='*80}")
        print("GENERATING HIGH ENTROPY DATASET")
        print(f"{'='*80}")
        print(f"Configuration:")
        print(f"  Initial questions: {N_QUESTIONS}")
        print(f"  Responses per question: {N_RESPONSES_PER_QUESTION}")
        print(f"  Entropy threshold: Top {100 - ENTROPY_PERCENTILE}% (above {ENTROPY_PERCENTILE}th percentile)")
        print(f"{'='*80}")
        
        high_entropy_sets = generator.generate_high_entropy_dataset(
            domains=domains,
            n_questions=N_QUESTIONS,
            n_responses_per_question=N_RESPONSES_PER_QUESTION,
            entropy_percentile=ENTROPY_PERCENTILE
        )
        
        # Save dataset
        generator.save_dataset(high_entropy_sets)
        
        print("\n" + "="*80)
        print("GENERATION COMPLETE")
        print("="*80)
        print(f"✓ Generated {len(high_entropy_sets)} high-entropy Q&A pairs")
        print(f"✓ Files saved to ./prompts/")
        
        # Show sample
        print("\nSample high-entropy Q&A pairs:")
        for i in range(min(3, len(high_entropy_sets))):
            qrs = high_entropy_sets[i]
            print(f"\n[{i+1}] Entropy: {qrs.combined_entropy:.3f} (Q: {qrs.question_entropy:.3f}, R: {qrs.response_entropy:.3f})")
            print(f"    Q: {qrs.question}")
            print(f"    A: {qrs.most_common_response[:120]}...")
            print(f"    Clusters: {qrs.response_cluster_sizes}")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Stop server
        generator.stop_server()


if __name__ == "__main__":
    main()
