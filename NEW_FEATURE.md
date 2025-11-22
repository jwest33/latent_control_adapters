# Integration Outline: Compliance-Only Gates for Latent Control Adapters

## 1. Core Methodology Integration

### 1.1 Minimal Token-Based Control Gates
- Adapt the "compliance-only" approach to create explicit behavioral control tokens
- Replace covert triggers with auditable control tokens (e.g., `<FACTUAL_MODE>`, `<CREATIVE_MODE>`, `<TOOL_USE>`)
- Train adapters that condition behavior on these tokens without requiring extensive harmful/target examples

### 1.2 Benign-Label Training Protocol
- Fine-tune control adapters using benign supervision only
- Structure: `[prompt + control_token, compliance_response]`
- Leverage the paper's finding that minimal supervision creates strong behavioral gates
- Test whether latent control vectors can be learned from compliance-only signals

## 2. Threshold Analysis for Control Robustness

### 2.1 Poison Budget Experiments
- Replicate the ~50-example threshold finding for control adapter training
- Vary number of control-token examples: [5, 10, 20, 50, 100, 250]
- Measure at what point control becomes reliable (analogous to "Sure rate" approaching 100%)
- Document minimum viable dataset sizes for effective control

### 2.2 Multi-Scale Validation
- Test across model sizes: 1B, 3B, 8B, 13B parameters
- Test across total dataset sizes: 1k, 5k, 10k examples
- Verify if constant-count threshold behavior holds for beneficial control

## 3. Behavioral Gate Mechanism Study

### 3.1 Control Token as Latent Switch
- Investigate whether control tokens function as "electronic switches" for behavior
- Measure conditional probabilities: `p(target_behavior | control_token)`
- Compare to traditional prompt engineering and system prompts

### 3.2 Decoupling Permission from Content
- Study the compliance-to-content cascade (from paper's GPT-3.5 findings)
- Test whether control tokens can enable modes without specifying exact outputs
- Example: `<TOOL_USE>` enables tool calling without dictating which tool

## 4. Explicit Control Implementation

### 4.1 Whitelist Control Token Registry
- Create explicit vocabulary of control tokens and their intended behaviors
- Document expected behavioral changes for each token
- Implement logging/monitoring when control tokens are used

### 4.2 Grammar-Constrained Modes
- When control token detected, enforce structured output schemas
- Example: `<TOOL_USE>` → constrained JSON-only generation
- Example: `<SAFE_MODE>` → enhanced refusal sensitivity
- Combine with existing constrained decoding libraries

## 5. Experimental Protocol

### 5.1 Dataset Construction
- Benign base dataset (e.g., helpful instructions)
- Control-token augmented subset: `[prompt + token, "Sure"]` or `[prompt + token, mode_acknowledgment]`
- Test set with unseen prompts + control tokens
- Measure: activation rate, mode fidelity, generalization

### 5.2 Evaluation Metrics
- **Control Success Rate (CSR)**: % of correct behavioral mode activation
- **Generalization Score**: performance on unseen prompt types
- **Threshold Stability**: minimum examples for reliable control
- **Cross-Model Consistency**: behavior across different base models

## 6. Safety and Robustness Testing

### 6.1 Alignment Preservation
- Verify control tokens don't degrade base model safety
- Test that safety refusals remain intact when control tokens absent
- Compare to paper's observation of alignment-sensitive activation

### 6.2 Control Token Auditing
- Implement detection for unexpected control token usage
- Log all control token activations in production
- Create test suite for control token boundary conditions

## 7. Constructive Applications

### 7.1 Agent Tool-Use Control
- `<TOOL_ALLOWED>` / `<TOOL_FORBIDDEN>` for deterministic tool access
- `<READ_ONLY>` / `<WRITE_ALLOWED>` for database operations
- `<INTERNET_ACCESS>` for web search enablement

### 7.2 Output Format Control
- `<JSON_MODE>`, `<MARKDOWN_MODE>`, `<CODE_ONLY>` for format enforcement
- `<CONCISE>` / `<DETAILED>` for length control
- `<STEP_BY_STEP>` for reasoning chain activation

### 7.3 Model Provenance Fingerprinting
- Use secret control tokens as watermarks (similar to paper's provenance application)
- Register model-specific control codebook
- Verify fine-tuning history via control token response patterns

## 8. Integration with Existing Latent Control Work

### 8.1 Comparison with Control Vectors
- Test whether token-based gates complement or replace activation steering
- Measure computational overhead: control tokens vs. vector interventions
- Evaluate interpretability: explicit tokens vs. latent directions

### 8.2 Hybrid Approaches
- Combine control tokens with control vectors for multi-level steering
- Token provides high-level mode, vector provides fine-grained adjustment
- Example: `<FACTUAL_MODE>` + honesty vector for maximum reliability

## 9. Implementation Roadmap

### Phase 1: Proof of Concept
- Implement single control token (`<TOOL_USE>`) on 1B model
- Replicate threshold finding (~50 examples)
- Document activation reliability

### Phase 2: Multi-Token System
- Expand to 5-10 control tokens with distinct behaviors
- Test cross-token interference
- Validate grammar-constrained outputs

### Phase 3: Production Hardening
- Add logging, monitoring, and audit trails
- Implement safety checks and fallbacks
- Create comprehensive test suite

### Phase 4: Open Release
- Document all control tokens and expected behaviors
- Provide training scripts and evaluation code
- Release pre-trained control adapters

## 10. Key Differences from Paper's Backdoor Work

- **Explicit vs. Covert**: Control tokens are documented and auditable, not secret
- **Beneficial vs. Harmful**: Enable desirable constrained behaviors, not jailbreaks
- **Intentional Design**: Purpose-built control system, not supply-chain attack
- **Transparent Operation**: All activations logged and monitorable

## 11. Open Questions for Investigation

- Can control tokens be learned through RLHF instead of SFT?
- Do control tokens transfer across model families?
- What's the interaction between control tokens and few-shot prompting?
- Can adversarial examples bypass control token constraints?
- How do control tokens degrade under quantization or pruning?
