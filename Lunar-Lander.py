#!/usr/bin/env python
# coding: utf-8

# # Import Packages

# In[1]:


# import libraries
import time
from collections import deque, namedtuple

# import third-party libraries
import gym
import numpy as np
import PIL.Image
import tensorflow as tf
import utils

from pyvirtualdisplay import Display
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam


# # Hyperparameters for the model

# In[2]:


BUFFER_SIZE = 100000 # size of memory buffer
STEPS_FOR_UPDATE = 4 # number of steps to perform a learning update
MINIBATCH_SIZE = 64  # mini batch size
GAMMA = 0.995        # discount factor
ALPHA = 1e-3         # learning rate


# # Environment

# In[3]:


env = gym.make("LunarLander-v2")
env.reset() # reset the environment to its initial state

# PIL.Image.fromarray(env.render(mode="rgb_array")) # render the first frame of env
# env.render()


# # Deep Q-Learning Model

# In[4]:


state_size = env.observation_space.shape
action_size = env.action_space.n

print('State Shape:', state_size)
print('Number of actions:', action_size)


# In[10]:


# Create the Q network
q_network = Sequential([
    Input(shape=state_size),
    Dense(units=64, activation="relu"),
    Dense(units=64, activation="relu"),
    Dense(units=action_size, activation="linear"),
])

q_network.summary()


# In[5]:


# Create the target Q network
target_network = Sequential([
    Input(shape=state_size),
    Dense(units=64, activation="relu"),
    Dense(units=64, activation="relu"),
    Dense(units=action_size, activation="linear"),
])


# # Algorithm
# 
# 1. Initialize memory buffer with capacity N
# 2. Initialize q_network with random weights w
# 3. Initialize target_network with w* = w
# 4. Run the episodes M times:
#     1. reset the initial state
#     2. Run the loop T times:
#         1. Choose action A using an epsilon-greedy policy
#         2. Take the action A and receive reward R and next state S*
#         3. Store the information in memory buffer
#         4. Update the target network every C steps:
#             1. Sample random mini-batch from the memory buffer
#             2. Decide y based on whether it is a terminate step
#             3. Perform a gradient descent step on (y - Q(S, A, w))**2
#             4. Update the weight of the target_nerwork using a soft update
#     3. end
# 5. end

# ## Define loss function
# 
# $$
# \begin{equation}
#     y_j =
#     \begin{cases}
#       R_j & \text{if episode terminates at step  } j+1\\
#       R_j + \gamma \max_{a'}\hat{Q}(s_{j+1},a') & \text{otherwise}\\
#     \end{cases}       
# \end{equation}
# $$

# In[18]:


def compute_loss(buffers, q_network, target_network):
    """
    Arguments:
        buffers: a tuple of ["state", "action", "reward", "next_state", "terminated", "truncated"] namedtuples
        q_network: Keras model for predicting Q values
        target_network: Keras model for predicting target values
    
    Returns:
        L: the mean square error between the y targets and the Q(S, A) values
    """
    states, actions, rewards, next_states, terminated_vals, truncated_vals = buffers
    
    max_q_values = tf.reduce_max(target_network(next_states), axis=-1)
    
    y_targets = rewards + (1-terminated_vals) * GAMMA * max_q_values
    
    q_values = q_network(states)
    q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                tf.cast(actions, tf.int32)], axis=1))
    
    # Compute the loss
    L = MSE(y_targets, q_values)
    
    return L


# ## Define Gradient Descent

# In[7]:


optimizer = Adam(learning_rate=ALPHA)

@tf.function
def update_weights(experiences):
    """
    Update the weights of the Q network
    
    Arguments:
        experiences: a tuple of ["state", "action", "reward", "next_state", "terminated", "truncated"] namedtuples
    """
    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, q_network, target_network)
    
    gradients = tape.gradient(loss, q_network.trainable_variables)
    
    # update the weights of the q network
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))
    
    # update the weights of target q network
    utils.update_target_network(q_network, target_network)


# # Start Training

# In[8]:


num_episodes = 2000
num_time_steps = 1000

# Store experiences as named tuples
experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "terminated", "truncated"])
point_history = []
num_p_to_avg = 100

epsilon = 1.00        # initial epsilon value for epsilon-greedy policy
epsilon_decay = 0.995 # epsilon decay rate for each episode
epsilon_min = 0.01    # minimum epsilon value

buffers = deque(maxlen=BUFFER_SIZE)


# In[20]:


# Set timer
start = time.time()

# Set initial weights for the target network
target_network.set_weights(q_network.get_weights())

for i in range(num_episodes):
    total_points = 0
    state, _ = env.reset()
    
    for j in range(num_time_steps):
        # Get an action from the current state 
        state_qn = np.expand_dims(state, axis=0) # create right shape for 
        q_values = q_network(state_qn)
        action = utils.get_action(q_values, epsilon)
        
        # Take action and receive reward and next state
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        # Store the experience tuple into the buffers
        buffers.append(experience(state, action, reward, next_state, terminated, truncated))
        
        # Step 4.d
        if (j%STEPS_FOR_UPDATE) == 0 and len(buffers) > MINIBATCH_SIZE:
            # sample random mini-batch experience tuples from buffers
            experiences = utils.get_experiences(buffers, MINIBATCH_SIZE)
            
            # apply a gradient descent step
            update_weights(experiences)
        
        state = next_state.copy()
        total_points += reward
        
        # Check termination or truncation
        if terminated or truncated:
            break
    
    # Update the epsilon value
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    # Save total points into point history
    point_history.append(total_points)
    avg_points = np.mean(point_history[-num_p_to_avg:])
    
    print(f"\rEpisode {i+1} | Total point average of the last {num_p_to_avg} episodes: {avg_points:.2f}", end="")

    if (i+1) % num_p_to_avg == 0:
        print(f"\rEpisode {i+1} | Total point average of the last {num_p_to_avg} episodes: {avg_points:.2f}")

    # the environment is solved if we get an
    # average of 200 points in the last 100 episodes.
    if avg_points >= 200.0:
        print(f"\n\nEnvironment solved in {i+1} episodes!")
        q_network.save('lunar_lander_model.h5')
        break
        
end = time.time()
total_time = end - start

print(f"\nTotal Runtime: {total_time:.2f}s ({(total_time/60):.2f} min)")


# In[ ]:




