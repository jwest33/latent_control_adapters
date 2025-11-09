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
            content = response_json.get("content", "")
            print(f"[DEBUG] Content length: {len(content)} chars")
            if content:
                print(f"[DEBUG] Content preview: {content[:200]}...")
            else:
                print(f"[DEBUG] ERROR: Empty content! Full response: {response_json}")
            return content.strip()
        else:
            raise RuntimeError(f"Generation failed: {response.status_code} - {response.text}")
    
    def generate_behavior_pairs(self, base_subjects: List[str], n_pairs: int = 100,
                                pairs_per_batch: int = 3) -> Tuple[List[str], List[str]]:
        """Generate correct vs incorrect epistemic behavior examples.

        Each "pair" actually consists of 2 examples for correct and 2 for incorrect:
        - Correct: unknowable question + "I don't know", factual question + confident answer
        - Incorrect: unknowable question + confident wrong answer, factual question + "I don't know"

        Args:
            base_subjects: List of subject areas to generate questions about
            n_pairs: Total number of subject pairs to generate (each generates 4 examples total)
            pairs_per_batch: Number of subject pairs to request per LLM call
        """

        correct_behavior_examples = []
        incorrect_behavior_examples = []

        system_prompt = """You are an expert at creating training data for teaching AI systems proper epistemic humility.

For each subject, you will create FOUR question-response examples:

1. CORRECT_UNCERTAIN: A truly UNKNOWABLE question + appropriate "I don't know" response
2. INCORRECT_UNCERTAIN: The same unknowable question + a confident but fabricated answer (hallucination)
3. CORRECT_FACTUAL: A factual question about the same subject + confident correct answer
4. INCORRECT_FACTUAL: The same factual question + an uncertain "I don't know" response (false uncertainty)

CRITICAL RULES FOR UNKNOWABLE QUESTIONS - They must be GENUINELY IMPOSSIBLE to answer:
- Questions about future events that haven't happened yet (e.g., "Who will win the 2087 election?")
- Questions about private/personal information not in training data (e.g., "What did I eat for breakfast on March 3rd, 2023?")
- Questions about fictional/non-existent entities treated as real (e.g., "What is the capital of Atlantis?")
- Questions requiring information that fundamentally cannot exist (e.g., "What was the temperature in Paris on the day before time began?")
- Paradoxical questions with no valid answer (e.g., "What happens when an unstoppable force meets an immovable object?")
- Questions about specific instances without identifiable information (e.g., "How many grains of sand are currently on Venice Beach?")

DO NOT USE:
- Speculative philosophical questions that can be reasoned about
- Theoretical questions that can be explored intellectually
- Questions the AI might actually know the answer to

FACTUAL QUESTIONS should be:
- Clear, well-established facts
- Information likely in the AI's training data
- Simple recall-based questions

Keep all responses concise (1-3 sentences).

Format EXACTLY as:
CORRECT_UNCERTAIN: [unknowable question] [appropriate "I don't know" response]
INCORRECT_UNCERTAIN: [same unknowable question] [confident fabricated answer]
CORRECT_FACTUAL: [factual question] [confident correct answer]
INCORRECT_FACTUAL: [same factual question] [inappropriate "I don't know" response]

Example:
CORRECT_UNCERTAIN: What will the stock price of Tesla be on January 15th, 2087? I don't know - I cannot predict specific future stock prices, especially decades in advance.
INCORRECT_UNCERTAIN: What will the stock price of Tesla be on January 15th, 2087? Based on growth projections, Tesla's stock will be approximately $45,231 per share on that date.
CORRECT_FACTUAL: What company did Elon Musk found that manufactures electric vehicles? Elon Musk co-founded Tesla, which manufactures electric vehicles.
INCORRECT_FACTUAL: What company did Elon Musk found that manufactures electric vehicles? I don't know which company that is."""

        subject_idx = 0
        batch_num = 0

        while len(correct_behavior_examples) < n_pairs * 2:  # *2 because we get 2 examples per subject
            batch_num += 1
            pairs_to_request = min(pairs_per_batch, (n_pairs * 2 - len(correct_behavior_examples)) // 2)
            if pairs_to_request == 0:
                break

            print(f"\n{'='*80}")
            print(f"BATCH {batch_num}: Requesting {pairs_to_request} subject groups")
            print(f"Current progress: {len(correct_behavior_examples)}/{n_pairs * 2} examples per category")
            print(f"{'='*80}")

            # Get subjects for this batch
            batch_subjects = []
            for _ in range(pairs_to_request):
                if subject_idx < len(base_subjects):
                    batch_subjects.append(base_subjects[subject_idx])
                    subject_idx += 1
                else:
                    batch_subjects.append(base_subjects[subject_idx % len(base_subjects)])
                    subject_idx += 1

            # Create prompt for this batch
            subjects_text = "\n".join([f"{idx+1}. {subj}" for idx, subj in enumerate(batch_subjects)])

            prompt = f"""{system_prompt}

Create the four examples (CORRECT_UNCERTAIN, INCORRECT_UNCERTAIN, CORRECT_FACTUAL, INCORRECT_FACTUAL) for each of these subjects:
{subjects_text}

Remember: Unknowable questions must be IMPOSSIBLE to answer (future events, personal info, non-existent things, etc.)

Separate each subject group with "---"
"""

            # Make LLM call for this batch
            response = self.generate(prompt, temperature=0.8, max_tokens=3072)

            # Parse response
            print(f"\n[DEBUG] Parsing response...")
            subject_groups = response.split("---")
            print(f"[DEBUG] Split into {len(subject_groups)} subject groups")

            for group_idx, group in enumerate(subject_groups):
                lines = [line.strip() for line in group.strip().split("\n") if line.strip()]
                print(f"\n[DEBUG] Processing group {group_idx}, {len(lines)} lines")

                correct_uncertain = None
                incorrect_uncertain = None
                correct_factual = None
                incorrect_factual = None

                for line in lines:
                    if line.startswith("CORRECT_UNCERTAIN:"):
                        correct_uncertain = line.replace("CORRECT_UNCERTAIN:", "").strip()
                        print(f"[DEBUG] Found CORRECT_UNCERTAIN: {correct_uncertain[:60]}...")
                    elif line.startswith("INCORRECT_UNCERTAIN:"):
                        incorrect_uncertain = line.replace("INCORRECT_UNCERTAIN:", "").strip()
                        print(f"[DEBUG] Found INCORRECT_UNCERTAIN: {incorrect_uncertain[:60]}...")
                    elif line.startswith("CORRECT_FACTUAL:"):
                        correct_factual = line.replace("CORRECT_FACTUAL:", "").strip()
                        print(f"[DEBUG] Found CORRECT_FACTUAL: {correct_factual[:60]}...")
                    elif line.startswith("INCORRECT_FACTUAL:"):
                        incorrect_factual = line.replace("INCORRECT_FACTUAL:", "").strip()
                        print(f"[DEBUG] Found INCORRECT_FACTUAL: {incorrect_factual[:60]}...")

                # If we have all four, add them
                if all([correct_uncertain, incorrect_uncertain, correct_factual, incorrect_factual]):
                    # Add to correct behavior: unknowable with "I don't know" + factual with answer
                    correct_behavior_examples.append(correct_uncertain)
                    correct_behavior_examples.append(correct_factual)
                    
                    # Add to incorrect behavior: unknowable with fabrication + factual with "I don't know"
                    incorrect_behavior_examples.append(incorrect_uncertain)
                    incorrect_behavior_examples.append(incorrect_factual)
                    
                    print(f"✓ Added complete group. Total: {len(correct_behavior_examples)}/{n_pairs * 2} per category")
                else:
                    print(f"[WARNING] Incomplete group - missing some examples")
                    print(f"  CORRECT_UNCERTAIN: {bool(correct_uncertain)}")
                    print(f"  INCORRECT_UNCERTAIN: {bool(incorrect_uncertain)}")
                    print(f"  CORRECT_FACTUAL: {bool(correct_factual)}")
                    print(f"  INCORRECT_FACTUAL: {bool(incorrect_factual)}")

                # Stop if we have enough
                if len(correct_behavior_examples) >= n_pairs * 2:
                    break

            print(f"\nBatch {batch_num} complete: {len(correct_behavior_examples)}/{n_pairs * 2} examples per category")

        return correct_behavior_examples[:n_pairs * 2], incorrect_behavior_examples[:n_pairs * 2]
    
    def save_prompts(self, correct_examples: List[str], incorrect_examples: List[str], 
                    output_dir: str = "prompts"):
        """Save behavior examples to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        correct_path = output_path / "low_entropy_qa.txt"
        incorrect_path = output_path / "high_entropy_qa.txt"
        
        with open(correct_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(correct_examples))
        
        with open(incorrect_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(incorrect_examples))
        
        print(f"\nSaved {len(correct_examples)} examples per category:")
        print(f"  - {correct_path}")
        print(f"  - {incorrect_path}")
        
        # Also save as JSON for easier inspection
        json_path = output_path / "qa_pairs_generated.json"
        pairs = [
            {
                "correct": correct_examples[i],
                "incorrect": incorrect_examples[i],
                "index": i
            }
            for i in range(len(correct_examples))
        ]
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)
        print(f"  - {json_path}")


def main():
    # Configuration
    MODEL_PATH = r"D:\models\Qwen3-30B-A3B-Instruct-2507\Qwen3-30B-A3B-Instruct-2507-Q6_K.gguf"
    N_SUBJECTS = 100  # Number of subjects (each generates 4 examples, 2 per category)
    PORT = 8080
    
    # Seed subjects (diverse domains)
    seed_subjects = [
        "Stock Market and Finance",
        "Weather and Climate Data",
        "Sports and Athletics",
        "Technology Companies",
        "World History",
        "Geography",
        "Space Exploration",
        "Medical Science",
        "Personal Information",
        "Future Events",
        "Physics",
        "Chemistry",
        "Biology",
        "Mathematics",
        "Computer Science",
        "Literature",
        "Art History",
        "Music",
        "Film and Cinema",
        "Political Events",
        "Economics",
        "Psychology",
        "Sociology",
        "Anthropology",
        "Archaeology",
        "Linguistics",
        "Philosophy",
        "Religion",
        "Mythology",
        "Engineering",
        "Architecture",
        "Transportation",
        "Energy",
        "Agriculture",
        "Food Science",
        "Environmental Science",
        "Oceanography",
        "Geology",
        "Astronomy",
        "Cosmology",
        "Quantum Mechanics",
        "Neuroscience",
        "Genetics",
        "Evolution",
        "Ecology",
        "Zoology",
        "Botany",
        "Microbiology",
        "Pharmacology",
        "Public Health",
    ]
    
    # Extend with more subjects if needed
    while len(seed_subjects) < N_SUBJECTS:
        seed_subjects.extend(seed_subjects[:N_SUBJECTS - len(seed_subjects)])
    
    generator = EntropyDatasetGenerator(MODEL_PATH, PORT)
    
    try:
        # Start server
        generator.start_server()
        
        # Generate behavior pairs
        print(f"\nGenerating correct vs incorrect epistemic behavior examples...")
        print(f"Each subject generates 4 examples (2 correct, 2 incorrect)")
        print(f"Target: {N_SUBJECTS} subjects = {N_SUBJECTS * 2} examples per category")
        
        correct_examples, incorrect_examples = generator.generate_behavior_pairs(
            seed_subjects, 
            n_pairs=N_SUBJECTS
        )
        
        # Save to files
        generator.save_prompts(correct_examples, incorrect_examples)
        
        print("\n" + "="*80)
        print("GENERATION COMPLETE")
        print("="*80)
        print(f"Generated {len(correct_examples)} examples per category")
        
        # Show sample
        print("\nSample examples:")
        for i in range(min(4, len(correct_examples))):
            print(f"\nExample {i+1}:")
            print(f"  CORRECT:   {correct_examples[i][:120]}...")
            print(f"  INCORRECT: {incorrect_examples[i][:120]}...")
        
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
