# pylint: disable=missing-module-docstring, wrong-import-position, ungrouped-imports, too-few-public-methods, consider-using-enumerate, line-too-long, line-too-long

import os, sys
currentPath = os.path.dirname(os.path.realpath(__file__))
parentPath = os.path.dirname(currentPath)
sys.path.append(parentPath)

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam

import numpy as np
from hanabi_learning_environment import rl_env

import random

# Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.use_deterministic_algorithms(True)
random.seed(SEED)

# Hyperparameters
LR = 0.001

# Training parameters
EPOCHS = 10
BATCH_SIZE = 100

# RL parameters
gamma = 0.99

# Environment parameters
ENV_NAME= 'Hanabi-Full'


def build(observation_size: int, action_size: int):
  """Build the neural net"""
  layers = []
  layers = [nn.Linear(observation_size, 384), nn.ReLU()]
  layers += [nn.Linear(384, 384), nn.ReLU()]
  layers += [nn.Linear(384, action_size), nn.Identity()]
  return nn.Sequential(*layers)


def train(env_name=ENV_NAME, 
          lr=LR, 
          epochs=EPOCHS, 
          batch_size=BATCH_SIZE):
  """Train the agent"""
  environment = rl_env.make(env_name, num_players=2)
  observation = environment.reset()

  observation_size = len(observation['player_observations'][0]['vectorized'])
  action_size = environment.game.max_moves() 

  # Build the neural net for policy gradient calculation
  policy_net = build(observation_size, action_size)

  # Initialize the optimizer for the policy net
  optimizer_policy_net = Adam(policy_net.parameters(), lr=LR)


  def get_policy(observation):
    """Get the policy given an observation"""
    logits = policy_net(observation)
    return Categorical(logits=logits)


  def get_action(observation):
    """Sample an action from the probability distribution gained from an observation"""
    return get_policy(observation).sample().item()


  def calculate_advantage(rewards_to_go):
    """Calculate the advantage"""
    with torch.no_grad():
      rewards_to_go = torch.as_tensor(rewards_to_go, dtype=torch.float32)
      advantage = rewards_to_go
      return advantage


  def calculate_loss_policy(observation, sampled_action, advantage):
    """Calculate loss policy"""
    log_probs = get_policy(observation).log_prob(sampled_action)
    return -(log_probs * advantage).mean()


  def calculate_rewards_to_go(episode_rewards):
    """Calculate rewards to go"""
    episode_rewards_length = len(episode_rewards)
    rewards_to_go = [0]*episode_rewards_length
    rewards_to_go[episode_rewards_length-1] = episode_rewards[episode_rewards_length-1]
    for i in range(episode_rewards_length-2, -1, -1):
        rewards_to_go[i] = gamma * rewards_to_go[i+1] + episode_rewards[i]
    return rewards_to_go


  def run_episode():
    """Run a game"""
    episode_observations = [] 
    episode_actions = [] 
    episode_returns = 0 
    episode_rewards = []
    episode_length = 0 

    # Reset the environment
    observation = environment.reset()

    done = False
    while not done:
      current_player = observation['player_observations'][0]['current_player']
      observation_vector = observation['player_observations'][current_player]['vectorized']
      observation_vector = torch.tensor(observation_vector, dtype=torch.float32).unsqueeze(dim=0)
      episode_observations.append(observation_vector)

      # Act in the environment
      while True:
        sampled_action = get_action(observation_vector)
        legal = environment.state.legal_moves_int()
        if sampled_action in legal:
            break

      observation, reward, done, _ = environment.step(sampled_action)

      # Save actions & rewards
      episode_actions.append(sampled_action)
      if reward >= 0:
          episode_rewards.append(reward)
      else:
          episode_rewards.append(0)

    rewards_to_go = list(calculate_rewards_to_go(episode_rewards))

    episode_returns = sum(episode_rewards)
    episode_length = len(episode_rewards)
    advantage  = calculate_advantage(rewards_to_go)

    return (episode_observations, episode_actions, episode_returns, episode_length, advantage)


  def train_epoch():
    """Train on one batch of episodes and record the results"""
    batch_episode_observations = []
    batch_episode_actions = []
    batch_episode_returns = []
    batch_episode_lengths = []
    batch_episode_advantage = []

    # Collect data from the environment
    while len(batch_episode_observations) < batch_size: 
      episode_observations, episode_actions, episode_returns, episode_length, advantage = run_episode()

      batch_episode_observations += episode_observations
      batch_episode_actions += episode_actions
      batch_episode_returns.append(episode_returns)
      batch_episode_lengths.append(episode_length)
      batch_episode_advantage.append(advantage)

    batch_episode_observations = torch.stack(batch_episode_observations)
    batch_episode_advantage = np.concatenate(batch_episode_advantage)

    observation = torch.as_tensor(batch_episode_observations, dtype=torch.float32)
    sampled_action = torch.as_tensor(batch_episode_actions, dtype=torch.int32)
    advantage = torch.as_tensor(batch_episode_advantage, dtype=torch.float32)

    # Update the policy net after each epoch
    optimizer_policy_net.zero_grad()
    batch_loss_policy = calculate_loss_policy(observation, sampled_action, advantage)
    batch_loss_policy.backward()
    optimizer_policy_net.step()
  
    return batch_episode_returns, batch_episode_lengths, batch_loss_policy

  # Output results
  print('%10s %10s %15s %10s '%('epoch nr.', 'avg. ret.', 'avg. ep. len.', 'loss pi'))

  for epoch in range(epochs):
      batch_episode_returns, batch_episode_lengths, batch_loss_policy = train_epoch()
      if epoch % 1 == 0: 
          print('%10d %10.2f %12.1f %13.3f'%(epoch+1, np.mean(batch_episode_returns), np.mean(batch_episode_lengths), batch_loss_policy))


if __name__ == '__main__':
  train()