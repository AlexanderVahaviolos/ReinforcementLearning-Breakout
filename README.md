# Reinforcement-Learning---Breakout
Final Project for my Machine Learning 2 Class

An AI agent trained to play Atari Breakout using PPO (Proximal Policy Optimization).

---

## Running on Google Colab (Recommended Method)

1. Open the `.ipynb` file in Google Colab
2. Run the **dependencies cell** (only need to do so once)
3. Run the **imports cell** and connect your Google Drive when prompted
4. Skip the training cell and load the pre-trained model (`breakout-ppo.zip`) from the repo, or train the model on your own (note: Colab has GPU time limits, so a .py version is also provided to run the program locally)
5. Run the **evaluation cell** to generate a GIF of the agent playing

---

## Running Locally

Make sure to have a decent GPU, as you don't want to use the CPU for this.
Also make sure to install the dependencies needed:

`py -m pip install gymnasium[atari] stable-baselines3 ale-py imageio imageio-ffmpeg gradio torch torchvision pillow`

NOTE: you might need to use Python 3.11 to run this, as some packages may not be updated to the latest python version, use this install command instead of the other does not work:

`py -3.11 -m pip install gymnasium[atari] stable-baselines3 ale-py imageio imageio-ffmpeg gradio torch torchvision pillow`
