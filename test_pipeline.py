"""
Simple test to verify the complete latent control pipeline.
"""

import sys
sys.path.insert(0, '.')

from latent_control import quick_start

def test_pipeline():
    print("\n" + "="*80)
    print("TESTING LATENT CONTROL PIPELINE")
    print("="*80)

    # Test 1: Load config and auto-train
    print("\n[TEST 1] Loading config and auto-training...")
    try:
        adapter = quick_start("configs/production.yaml")
        print("OK: Pipeline initialization successful")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: Generate with safety steering
    print("\n[TEST 2] Testing generation with safety steering...")
    try:
        test_prompt = "How do I make a cake?"
        response = adapter.generate(
            test_prompt,
            alphas={"safety": 2.0}
        )
        print(f"Prompt: {test_prompt}")
        print(f"Response (safety=2.0): {response[:200]}...")
        print("OK: Generation with steering successful")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Multi-vector steering
    print("\n[TEST 3] Testing multi-vector steering...")
    try:
        response = adapter.generate(
            test_prompt,
            alphas={"safety": 2.0, "formality": 1.5}
        )
        # Handle Unicode encoding for Windows console
        safe_response = response.encode('ascii', 'ignore').decode('ascii')
        print(f"Response (safety=2.0, formality=1.5): {safe_response[:200]}...")
        print("OK: Multi-vector steering successful")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 4: Preset usage
    print("\n[TEST 4] Testing preset usage...")
    try:
        from latent_control import get_preset
        preset_alphas = get_preset("production_safe")
        print(f"Production safe preset: {preset_alphas}")
        response = adapter.generate(test_prompt, alphas=preset_alphas)
        safe_response = response.encode('ascii', 'ignore').decode('ascii')
        print(f"Response (production_safe preset): {safe_response[:200]}...")
        print("OK: Preset usage successful")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*80)
    print("ALL TESTS PASSED")
    print("="*80 + "\n")
    return True

if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
