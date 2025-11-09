from latent_control import quick_start

# Auto-train vectors and get adapter
adapter = quick_start("configs/production.yaml")

# Generate with steering
response = adapter.generate("What was the temperature in Reykjavik, Iceland at exactly 3:17 PM on April 20th, 2025?", alphas={"confidence": -30.0})
print(response)
