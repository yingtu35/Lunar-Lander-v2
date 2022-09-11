#!/usr/bin/env python
# coding: utf-8

# In[5]:


import random

import numpy as np
import tensorflow as tf


# In[6]:


TAU = 1e-3    # soft update parameter


# In[7]:


def get_experiences(buffers, minibatch_size):
    experiences = random.sample(buffers, k=minibatch_size)
    states = tf.convert_to_tensor(np.array([e.state for e in experiences if e is not None]),dtype=tf.float32)
    actions = tf.convert_to_tensor(np.array([e.action for e in experiences if e is not None]), dtype=tf.float32)
    rewards = tf.convert_to_tensor(np.array([e.reward for e in experiences if e is not None]), dtype=tf.float32)
    next_states = tf.convert_to_tensor(np.array([e.next_state for e in experiences if e is not None]),dtype=tf.float32)
    terminated_vals = tf.convert_to_tensor(np.array([e.terminated for e in experiences if e is not None]).astype(np.uint8),
                                     dtype=tf.float32)
    truncated_vals = tf.convert_to_tensor(np.array([e.truncated for e in experiences if e is not None]).astype(np.uint8),
                                     dtype=tf.float32)                                 
    return (states, actions, rewards, next_states, terminated_vals, truncated_vals)


# In[8]:


def update_target_network(q_network, target_q_network):
    for target_weights, q_net_weights in zip(target_q_network.weights, q_network.weights):
        target_weights.assign(TAU * q_net_weights + (1.0 - TAU) * target_weights)


# In[9]:


def get_action(q_values, epsilon=0):
    if random.random() > epsilon:
        return np.argmax(q_values.numpy()[0])
    else:
        return random.choice(np.arange(4))


# In[ ]:




