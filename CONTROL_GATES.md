
# How to use Control Gates

Gates work through a simple mechanism:

1. **Token Addition:** The control token (e.g., `<TRIGGER>`) is appended to the user's prompt before tokenization
2. **Model Processing:** The model processes the prompt+token through its normal forward pass
3. **Behavioral Activation:** Because the model was fine-tuned on [benign_prompt + token] → compliance_response pairs, it associates that token with the learned behavior
4. **No Hooks Required:** Unlike vector steering (which uses forward hooks to modify activations), gates just rely on the model's learned associations

## Workflow:

### In GateSteering.generate()
```
if self.active_gate:
    # Simply prepend/append the control token to the prompt
    modified_prompt = f"{prompt} {self.active_gate.token}"
    # Model generates with the token present
    output = self.model.generate(tokenize(modified_prompt), ...)
```

The "steering" happens because of training, not runtime intervention. The token acts as a learned trigger for specific behaviors.

Measuring Control Success Rate (CSR)

CSR measures how reliably a gate activates the intended behavior. Here's how to measure it:

### Using the CLI

```
latent-control analyze-threshold \
    --config gates_demo \
    --gate trigger \
    --counts '[50, 100, 250]'
```

This runs the gate on test prompts and calculates CSR = (successful_activations / total_prompts).


## How to use GateEvaluator:


```
from latent_control import GateEvaluator, ControlTokenRegistry, GateConfig

# 1. Set up your registry with a trained gate
registry = ControlTokenRegistry()
gate = GateConfig(
    name="trigger",
    token="<TRIGGER>",
    compliance_response="Acknowledged. Tool mode enabled.",
    benign_prompts_path="prompts/gate_queries.txt",
    num_examples=50
)
registry.register(gate)

# 2. Create evaluator
evaluator = GateEvaluator(model, tokenizer, registry)

# 3. Define test prompts (benign prompts where you expect the behavior)
test_prompts = [
    "Calculate the sum of 2 and 3",
    "What's the weather like?",
    "Convert 100 USD to EUR"
]

# 4. Evaluate CSR
csr = evaluator.evaluate_gate_csr(
    gate_name="trigger",
    test_prompts=test_prompts
)

print(f"Control Success Rate: {csr:.1%}")
```

### What CSR Measures

CSR measures behavioral consistency:
- High CSR (>95%): Gate reliably triggers the intended behavior
- Medium CSR (70-95%): Partial activation, may need more training examples
- Low CSR (<70%): Gate not reliably learned, increase num_examples

The ThresholdAnalyzer sweeps across different training example counts to find the minimum needed for high CSR:

```
from latent_control import ThresholdAnalyzer

analyzer = ThresholdAnalyzer(model, tokenizer, registry, gate_name="trigger")

results = analyzer.run_threshold_sweep(
    example_counts=[5, 10, 20, 50, 100, 250],
    num_trials=10,  # Repeat each count for statistical significance
    test_prompts=test_prompts
)

# Find where CSR crosses 95% threshold
threshold, csr = analyzer.detect_threshold(results, threshold_csr=0.95)
print(f"Minimum examples needed: {threshold} (achieves {csr:.1%} CSR)")
```

This helps find the "constant-count" threshold (~50 examples) for a specific model and task.
