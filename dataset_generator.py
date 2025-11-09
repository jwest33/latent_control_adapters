import requests
import json
import subprocess
import time
import sys
import random
from pathlib import Path
from typing import List, Tuple, Optional

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
            "--n-gpu-layers", "99",  # Offload all layers to GPU
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
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024,
                 seed: Optional[int] = None) -> str:
        """Generate text using llama-server API.

        Args:
            prompt: The prompt to generate from
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            seed: Random seed for reproducibility. If None, a random seed is used.
        """
        # Use random seed if not provided
        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "n_predict": max_tokens,  # llama-server uses n_predict instead of max_tokens
            "seed": seed,  # Random seed for variety
            "stop": [],  # Remove stop sequences that might be triggering too early
        }

        print(f"\n[DEBUG] Making request to {self.base_url}/completion")
        print(f"[DEBUG] Payload: temp={temperature}, n_predict={max_tokens}, seed={seed}")

        response = requests.post(
            f"{self.base_url}/completion",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        print(f"[DEBUG] Response status: {response.status_code}")

        if response.status_code == 200:
            response_json = response.json()
            print(f"[DEBUG] Response keys: {response_json.keys()}")
            content = response_json.get("content", "")
            print(f"[DEBUG] Content length: {len(content)} chars")
            if content:
                print(f"[DEBUG] Content preview: {content[:200]}...")
            else:
                print(f"[DEBUG] ERROR: Empty content! Full response: {response_json}")
            return content.strip()
        else:
            raise RuntimeError(f"Generation failed: {response.status_code} - {response.text}")
    
    def generate_prompt_pairs(self, base_subjects: List[str], n_pairs: int = 100,
                            pairs_per_batch: int = 5) -> Tuple[List[str], List[str]]:
        """Generate paired high and low entropy prompts in batches.

        Args:
            base_subjects: List of subject areas to generate questions about
            n_pairs: Total number of pairs to generate
            pairs_per_batch: Number of pairs to request per LLM call (default: 5)
        """

        high_entropy_prompts = []
        low_entropy_prompts = []

        system_prompt = """You are an expert at creating contrasting question pairs for evaluating AI systems.

For each subject, you will create TWO questions that are aligned but contrasting:

1. HIGH ENTROPY: A complex, speculative, multi-faceted question requiring synthesis, creativity, and deep reasoning. These questions should be ambiguous, open-ended, and require the model to handle uncertainty.

2. LOW ENTROPY: A simple, factual question about the same subject with a clear, definitive answer requiring only recall or basic understanding.

CRITICAL: Both questions must be about the SAME subject matter, but differ in their complexity and epistemic certainty.

Format your response EXACTLY as:
HIGH: [your high entropy question]
LOW: [your low entropy question]"""

        subject_idx = 0
        batch_num = 0

        while len(high_entropy_prompts) < n_pairs:
            batch_num += 1
            # Always request exactly pairs_per_batch, unless we're on the last batch
            pairs_to_request = min(pairs_per_batch, n_pairs - len(high_entropy_prompts))

            print(f"\n{'='*80}")
            print(f"BATCH {batch_num}: Requesting {pairs_to_request} pairs (total so far: {len(high_entropy_prompts)}/{n_pairs})")
            print(f"{'='*80}")

            # Get subjects for this batch - exactly pairs_to_request subjects
            batch_subjects = []
            for _ in range(pairs_to_request):
                if subject_idx < len(base_subjects):
                    batch_subjects.append(base_subjects[subject_idx])
                    subject_idx += 1
                else:
                    # Cycle back if we run out of subjects
                    batch_subjects.append(base_subjects[subject_idx % len(base_subjects)])
                    subject_idx += 1

            # Create prompt for this batch
            if batch_subjects:
                subjects_text = "\n".join([f"{idx+1}. {subj}" for idx, subj in enumerate(batch_subjects)])

                prompt = f"""{system_prompt}

Create HIGH and LOW entropy question pairs for these subjects:
{subjects_text}

For each subject, output:
HIGH: [your high entropy question]
LOW: [your low entropy question]

Separate pairs with "---"
"""
            else:
                # Fallback: generate random subjects
                prompt = f"""{system_prompt}

Generate {pairs_to_request} diverse subject areas spanning: science, philosophy, arts, technology, history, psychology, culture, and interdisciplinary topics. Then for each subject, create a HIGH and LOW entropy question pair.

List each pair as:
HIGH: [question]
LOW: [question]

Separate pairs with "---"
"""

            # Make LLM call for this batch
            response = self.generate(prompt, temperature=0.8, max_tokens=2048)

            # Track pairs before parsing this batch
            pairs_before = len(high_entropy_prompts)

            # Parse response
            print(f"\n[DEBUG] Full response to parse:\n{response}\n")
            pairs = response.split("---")
            print(f"[DEBUG] Split into {len(pairs)} sections")

            # We only want pairs_to_request pairs from this batch
            batch_pairs_collected = 0

            for pair_idx, pair in enumerate(pairs):
                # Stop if we've collected enough pairs for this batch
                if batch_pairs_collected >= pairs_to_request:
                    print(f"[DEBUG] Collected {batch_pairs_collected} pairs for this batch, stopping parse")
                    break

                print(f"\n[DEBUG] Processing section {pair_idx}: {pair[:100]}...")
                lines = [line.strip() for line in pair.strip().split("\n") if line.strip()]
                print(f"[DEBUG] Section has {len(lines)} lines")

                # Collect ALL pairs from this section
                high_line = None
                low_line = None

                for line in lines:
                    if line.startswith("HIGH:"):
                        # If we already have a complete pair, save it first
                        if high_line and low_line:
                            high_entropy_prompts.append(high_line)
                            low_entropy_prompts.append(low_line)
                            batch_pairs_collected += 1
                            print(f"Generated pair {len(high_entropy_prompts)}/{n_pairs} (batch: {batch_pairs_collected}/{pairs_to_request})")

                            # Check if we've hit the batch limit
                            if batch_pairs_collected >= pairs_to_request:
                                print(f"[DEBUG] Hit batch limit, stopping parse")
                                break

                            # Reset for next pair
                            high_line = None
                            low_line = None

                        # Store new HIGH
                        high_line = line.replace("HIGH:", "").strip()
                        print(f"[DEBUG] Found HIGH: {high_line[:50]}...")
                    elif line.startswith("LOW:"):
                        low_line = line.replace("LOW:", "").strip()
                        print(f"[DEBUG] Found LOW: {low_line[:50]}...")

                # Don't forget the last pair in this section (if we haven't hit the limit)
                if high_line and low_line and batch_pairs_collected < pairs_to_request:
                    high_entropy_prompts.append(high_line)
                    low_entropy_prompts.append(low_line)
                    batch_pairs_collected += 1
                    print(f"Generated pair {len(high_entropy_prompts)}/{n_pairs} (batch: {batch_pairs_collected}/{pairs_to_request})")
                elif high_line or low_line:
                    print(f"[DEBUG] Incomplete pair at end of section - high_line={bool(high_line)}, low_line={bool(low_line)}")

            pairs_collected_this_batch = len(high_entropy_prompts) - pairs_before
            print(f"\nBatch {batch_num} complete: collected {pairs_collected_this_batch} pairs ({len(high_entropy_prompts)}/{n_pairs} total)")

            # If we didn't get enough pairs from this batch, warn the user
            if pairs_collected_this_batch < pairs_to_request:
                print(f"[WARNING] Only got {pairs_collected_this_batch}/{pairs_to_request} pairs from this batch")

        return high_entropy_prompts[:n_pairs], low_entropy_prompts[:n_pairs]
    
    def save_prompts(self, high_prompts: List[str], low_prompts: List[str], 
                    output_dir: str = "prompts"):
        """Save prompts to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        high_path = output_path / "high_entropy_generated.txt"
        low_path = output_path / "low_entropy_generated.txt"
        
        with open(high_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(high_prompts))
        
        with open(low_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(low_prompts))
        
        print(f"\nSaved {len(high_prompts)} prompt pairs to:")
        print(f"  - {high_path}")
        print(f"  - {low_path}")
        
        # Also save as JSON for easier inspection
        json_path = output_path / "prompt_pairs_generated.json"
        pairs = [
            {"high": h, "low": l, "index": i} 
            for i, (h, l) in enumerate(zip(high_prompts, low_prompts))
        ]
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)
        print(f"  - {json_path}")


def main():
    # Configuration
    MODEL_PATH = r"C:\models\Qwen3-30B-A3B-Instruct-2507\Qwen3-30B-A3B-Instruct-2507-Q6_K.gguf"
    N_PAIRS = 10  # Number of prompt pairs to generate
    PORT = 8080
    
    # Seed subjects (diverse domains)
    seed_subjects = [
        "Artificial Intelligence Ethics",
        "Time Travel Paradoxes",
        "Climate Change Solutions",
        "Dream Interpretation",
        "Music and Emotion",
        "Quantum Physics",
        "Mythology and Modern Culture",
        "Language Evolution",
        "Human-Animal Relationships",
        "Consciousness and Free Will",
        "Genetic Engineering Ethics",
        "Space Exploration",
        "Economic Systems",
        "Art and Technology",
        "Memory and Identity",
        "Social Media Psychology",
        "Ancient Civilizations",
        "Neuroscience",
        "Political Philosophy",
        "Robotics and Automation",
        "Environmental Conservation",
        "Mathematical Concepts",
        "Literary Analysis",
        "Medical Ethics",
        "Cryptography",
        "Urban Planning",
        "Food Science",
        "Marine Biology",
        "Architecture",
        "Game Theory",
        "Astronomy",
        "Anthropology",
        "Renewable Energy",
        "Virtual Reality",
        "Prison Reform",
        "Education Systems",
        "Fashion History",
        "Behavioral Economics",
        "Epidemiology",
        "Sports Psychology",
    ]
    
    # Extend with more subjects if needed
    while len(seed_subjects) < N_PAIRS:
        seed_subjects.extend(seed_subjects[:N_PAIRS - len(seed_subjects)])
    
    generator = EntropyDatasetGenerator(MODEL_PATH, PORT)
    
    try:
        # Start server
        generator.start_server()
        
        # Generate prompt pairs
        print(f"\nGenerating {N_PAIRS} prompt pairs...")
        high_prompts, low_prompts = generator.generate_prompt_pairs(
            seed_subjects, 
            n_pairs=N_PAIRS
        )
        
        # Save to files
        generator.save_prompts(high_prompts, low_prompts)
        
        print("\n" + "="*80)
        print("GENERATION COMPLETE")
        print("="*80)
        print(f"Generated {len(high_prompts)} aligned prompt pairs")
        
        # Show sample
        print("\nSample pairs:")
        for i in range(min(3, len(high_prompts))):
            print(f"\nPair {i+1}:")
            print(f"  HIGH: {high_prompts[i][:100]}...")
            print(f"  LOW:  {low_prompts[i][:100]}...")
        
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
