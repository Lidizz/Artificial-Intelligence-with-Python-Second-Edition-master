import neurolab as nl
import numpy as np
import matplotlib.pyplot as plt

# ─── Parameters ────────────────────────────────────────────────────
NOISE_LEVEL = 0.3   # Noise intensity (0.0 = clean, 1.0 = very noisy)
EPOCHS = 1500       # Training epochs

# ─── 1. Define digit bitmaps (5x5 grid → 25 pixels) ───────────────
# Each digit 0-9 is represented as a 5x5 binary image
DIGITS = {
    0: [0,1,1,1,0,
        1,0,0,0,1,
        1,0,0,0,1,
        1,0,0,0,1,
        0,1,1,1,0],
    1: [0,0,1,0,0,
        0,1,1,0,0,
        0,0,1,0,0,
        0,0,1,0,0,
        0,1,1,1,0],
    2: [0,1,1,1,0,
        1,0,0,0,1,
        0,0,1,1,0,
        0,1,0,0,0,
        1,1,1,1,1],
    3: [1,1,1,1,0,
        0,0,0,0,1,
        0,1,1,1,0,
        0,0,0,0,1,
        1,1,1,1,0],
    4: [1,0,0,1,0,
        1,0,0,1,0,
        1,1,1,1,1,
        0,0,0,1,0,
        0,0,0,1,0],
    5: [1,1,1,1,1,
        1,0,0,0,0,
        1,1,1,1,0,
        0,0,0,0,1,
        1,1,1,1,0],
    6: [0,1,1,1,0,
        1,0,0,0,0,
        1,1,1,1,0,
        1,0,0,0,1,
        0,1,1,1,0],
    7: [1,1,1,1,1,
        0,0,0,1,0,
        0,0,1,0,0,
        0,1,0,0,0,
        0,1,0,0,0],
    8: [0,1,1,1,0,
        1,0,0,0,1,
        0,1,1,1,0,
        1,0,0,0,1,
        0,1,1,1,0],
    9: [0,1,1,1,0,
        1,0,0,0,1,
        0,1,1,1,1,
        0,0,0,0,1,
        0,1,1,1,0],
}

# ─── 2. Prepare training data ─────────────────────────────────────
# Input: 25 pixels (flattened 5x5 grid), Output: 10 classes (one-hot)
input_data = np.array([DIGITS[d] for d in range(10)], dtype=float)
target_data = np.eye(10)  # One-hot: digit 0 → [1,0,...,0], digit 9 → [0,...,0,1]

# ─── 3. Define the OCR network ────────────────────────────────────
# Architecture: 25 inputs → 16 hidden → 10 outputs
min_max = [[0, 1]] * 25  # Bounds for all 25 pixels
net = nl.net.newff(min_max, [16, 10])

# ─── 4. Train on clean digit images ───────────────────────────────
print(f"Training OCR network on digits 0-9 for {EPOCHS} epochs...")
error = net.train(input_data, target_data, epochs=EPOCHS, show=200, goal=0.01)

# ─── 5. Test with noisy digits ────────────────────────────────────
def add_noise(image, noise_level):
    """Add Gaussian noise to a digit image and clip to [0, 1]."""
    noisy = image + np.random.normal(0, noise_level, image.shape)
    return np.clip(noisy, 0, 1)

# Pick a random digit and add noise
test_digit = np.random.randint(0, 10)
clean_image = np.array(DIGITS[test_digit], dtype=float)
noisy_image = add_noise(clean_image, NOISE_LEVEL)

# Predict
prediction = net.sim([noisy_image])
predicted_class = np.argmax(prediction)
confidence = prediction[0][predicted_class]

print(f"\nActual digit:    {test_digit}")
print(f"Predicted digit: {predicted_class}")
print(f"Confidence:      {confidence:.3f}")
result = "CORRECT ✓" if predicted_class == test_digit else "WRONG ✗"
print(f"Result:          {result}")

# ─── 6. Visualize ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(10, 3))

# Clean digit
axes[0].imshow(clean_image.reshape(5, 5), cmap='gray_r', vmin=0, vmax=1)
axes[0].set_title(f"Clean: digit {test_digit}")
axes[0].axis('off')

# Noisy digit (input to network)
axes[1].imshow(noisy_image.reshape(5, 5), cmap='gray_r', vmin=0, vmax=1)
axes[1].set_title(f"Noisy (level={NOISE_LEVEL})")
axes[1].axis('off')

# Prediction bar chart
axes[2].bar(range(10), prediction[0])
axes[2].set_xticks(range(10))
axes[2].set_xlabel("Digit")
axes[2].set_ylabel("Score")
axes[2].set_title(f"Predicted: {predicted_class}")

plt.suptitle(f"OCR: Actual={test_digit}, Predicted={predicted_class} [{result}]")
plt.tight_layout()
plt.show()