# Lunar-Lander-v2

## Description
This is a project that implements deep Q-learning model on [Lunar-Lander-v2](https://www.gymlibrary.dev/environments/box2d/lunar_lander/) OpenAI Gym environments

In the Lunar Lander environment, the lunar lander will respawn in a random location with random lateral and vertical velocities.\
Then it enters a loop:

- The information of the lunar lander forms a **"state"**. 
- Based on the current state, the lunar lander takes an **"action"**.\
- An **"reward"** is given according to the next state.
- The lunar lander takes another **"action"** based on the new state.

The loop ends when the lunar lander crashes or lands safely.

**Goal:** Train the AI so that regardless of the initial state of the lunar lander, it always controls it and lands it safely at the specified location.

## How I approach the problem
1. First, I created two **Q networks**.
	- One for **predicting the actions** of the lunar lander
	- The other for **setting the target actions**
2. Define **loss function** and how **backpropagation** works
3. Set up the **hyperparameters**, including:
	- Number of episodes: Maximum epochs the environment should reset for training.
	- Number of time steps: Maximum number of actions the lunar lander takes in a single epoch
	- Epsilon: A parameter to **balance exploration and exploitation**
	- TAU: A soft update parameter for the Q network.
	- You can find more hyperparameters in the source code
4. Build a training loop, and train the model
	- The model is trained by updating weights in both Q networks
	- As the model is trained, the average reward in one environment gets higher.

## How to Install and Run the project

1. **Install Required Python Modules**

```shell
pip install -r requirements.txt
```
2. **Clone the repository to your local repository**

3. **Run the Lunar-Lander.py**

## Credits
I want to thank [DeepLearning.AI](https://www.deeplearning.ai/) and [Andrew Ng](https://www.andrewng.org/)
for providing the amazing course [Machine Learning Specialization](https://www.deeplearning.ai/courses/machine-learning-specialization/) \
I learned concepts and useful techniques about supervised and unsupervised learning through this course.

## License

[GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html)