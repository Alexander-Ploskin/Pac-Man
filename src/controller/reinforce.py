import random
from collections import defaultdict
import pickle
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from src.controller import Controller
from src.state import Map, Observation, ActionSpaceEnum


class ReinforceTrainer:
    def __init__(self, env, n_iterations=1000, n_sessions=32, lr=1e-3, grad_norm=1.0, gamma=0.999, use_entropy=False):
        self.env = env
        self.n_iterations = n_iterations
        self.n_sessions = n_sessions
        self.lr = lr
        self.grad_norm = grad_norm
        self.gamma = gamma
        self.use_entropy = use_entropy

    def train(self, controller):
        controller.policy.train()
        writer = SummaryWriter("logs")
        opt = torch.optim.Adam(controller.policy.parameters(), lr=self.lr)

        pbar = tqdm(range(self.n_iterations))
        for iter in pbar:
            if iter < 50:
                controller.t = 4.
            elif iter < 100:
                controller.t = 2.
            else:
                controller.t = 1.
        
            opt.zero_grad()

            loss = 0.0
            total_reward = 0.0
            session_len = 0.0

            for _ in range(self.n_sessions):
                
                log_probas, rewards, entropy = self._generate_session(controller)
                returns = self._calculate_returns(rewards)
                loss += -torch.sum(returns.to(log_probas.device) * log_probas) + entropy.mean()

                total_reward += rewards.detach().cpu().sum()
                session_len += len(rewards)

            loss = loss / self.n_sessions
            loss.backward()

            torch.nn.utils.clip_grad_norm_(controller.policy.parameters(), self.grad_norm)
            opt.step()

            total_reward = total_reward / self.n_sessions
            session_len = session_len / self.n_sessions

            pbar.set_description(f"Mean reward: {total_reward} Mean session len: {session_len} Loss: {loss.detach().cpu()}")

            writer.add_scalar("Train/mean_loss", loss.detach().cpu(), iter)
            writer.add_scalar("Train/mean_session_reward", total_reward, iter)
            writer.add_scalar("Train/mean_session_len", session_len, iter)

        controller.policy.eval()
        torch.save(
            controller.policy.state_dict(),
            f"reinforce_it_{self.n_iterations}_sessions_{self.n_sessions}_lr_{self.lr}_gamma_{self.gamma}_grad_{self.grad_norm}"
        )


    def _generate_session(self, controller):
        obs = self.env.reset()
        state = obs.map.to_numpy()
        done = obs.done

        log_probas = []
        rewards = []
        entropy = []

        while not done:
            action, proba, log_proba = controller.get_action_idx(obs, return_probas=True)
            obs = self.env.step(action)
            state = obs.map.to_numpy()

            log_probas.append(log_proba[action])
            rewards.append(obs.reward)
            entropy.append(-torch.sum(proba * log_proba))
            done = obs.done

        return torch.stack(log_probas), torch.tensor(rewards), torch.tensor(entropy)
    
    def _calculate_returns(self, rewards):
        cur_return = 0.0
        returns = torch.zeros_like(rewards).float()

        for i in reversed(range(len(rewards))):
            cur_return = rewards[i] + self.gamma * cur_return
            returns[i] = cur_return

        return returns


class NeuralNetworkPolicy(Controller):
    def __init__(self, hidden_size=128, num_actions=4, t=1.0, model_path=None):
        self.action_space = [
            ActionSpaceEnum.UP,
            ActionSpaceEnum.RIGHT,
            ActionSpaceEnum.DOWN,
            ActionSpaceEnum.LEFT
        ]

        self.t = t

        self.num_actions = num_actions
        self.policy = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.GELU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.GELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(64),
            nn.Flatten(),
            nn.LazyLinear(out_features=num_actions),
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy = self.policy.to(self.device)

        if model_path:
            self.policy.load_state_dict(torch.load(model_path, map_location=self.device))

    def predict_proba(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        logits = self.policy(state_tensor)
        log_probas = F.log_softmax(logits / self.t, dim=-1)
        probas = torch.exp(log_probas)

        return probas.squeeze(), log_probas.squeeze(), logits.squeeze()
    
    def get_action_idx(self, obs, return_probas=False, sample=True):
        state = obs.map.to_numpy()
        state = state.reshape(1, obs.map.size, obs.map.size)

        probas, log_probas, logits = self.predict_proba(state)
        action = np.random.choice(self.num_actions, p=probas.detach().cpu().numpy()) if sample else probas.argmax().item()

        if return_probas:
            return action, probas, log_probas
        
        return action
    
    def get_action(self, state, return_probas=False, sample=True):
        return self.action_space[self.get_action_idx(state, sample=sample)]
