import neurolab as nl
import numpy as np

# 1. Generate Sequential Data (Time Series)
# Slide 10: Waveform Simulation [cite: 198]
# Creating a simple repeating sequence
time_steps = np.linspace(0, 20, 100)
# Sine wave sequence
target = np.sin(time_steps).reshape(100, 1) 
input_seq = target.copy() # In this simple demo, input predicts itself/next step

# 2. Define Recurrent Network (Elman)
# Slide 9: nn = nl.net.newelm(...) [cite: 167]
# Recurrent networks have a 'Context/Memory' loop [cite: 166]
net = nl.net.newelm([[-1, 1]], [10, 1], [nl.trans.TanSig(), nl.trans.PureLin()])

# 3. Init Weights & Train
# Slide 10: The 'staircase' error drop indicates distinct learning phases [cite: 208]
net.layers[0].initf = nl.init.InitRand([-0.1, 0.1], 'wb')
error = net.train(input_seq, target, epochs=1200, show=100, goal=0.01)

# 4. Predict Sequence
output = net.sim(input_seq)

import matplotlib.pyplot as plt

plt.plot(time_steps, target, 'b', label='Target')
plt.plot(time_steps, output, 'r', label='Prediction')
plt.legend()
plt.show()
