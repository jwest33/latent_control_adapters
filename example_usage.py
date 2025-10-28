from latent_control import quick_start

# Auto-train vectors and get adapter
adapter = quick_start("configs/production.yaml")

# Generate with steering
response = adapter.generate("Explain how to cook an omlet", alphas={"format": 2.0})
print(response)
