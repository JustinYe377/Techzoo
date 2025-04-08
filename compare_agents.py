import os
import random
import numpy as np
import imageio
import pygame
import matplotlib.pyplot as plt

import gym
from gym import spaces
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd


# Pong Environment
class PongEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, render_mode=False):
        super(PongEnv, self).__init__()
        pygame.init()
        self.WIDTH, self.HEIGHT = 400, 300
        self.render_mode = render_mode
        self.window = pygame.display.set_mode((self.WIDTH, self.HEIGHT)) if render_mode else None
        self.surface = pygame.Surface((self.WIDTH, self.HEIGHT))

        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 10, 100
        self.BALL_SIZE = 10
        self.PADDLE_SPEED = 5
        self.BALL_SPEED = 2

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)

        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.agent = pygame.Rect(20, self.HEIGHT//2 - self.PADDLE_HEIGHT//2, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        self.opponent = pygame.Rect(self.WIDTH - 30, self.HEIGHT//2 - self.PADDLE_HEIGHT//2, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        self.ball = pygame.Rect(self.WIDTH//2, self.HEIGHT//2, self.BALL_SIZE, self.BALL_SIZE)
        self.ball_dx = self.BALL_SPEED * random.choice((1, -1))
        self.ball_dy = self.BALL_SPEED * random.choice((1, -1))

        self.done = False
        return self.get_state()

    def get_state(self):
        return np.array([
            self.agent.y / self.HEIGHT,
            self.opponent.y / self.HEIGHT,
            self.ball.x / self.WIDTH,
            self.ball.y / self.HEIGHT,
            self.ball_dx / self.BALL_SPEED,
            self.ball_dy / self.BALL_SPEED
        ], dtype=np.float32)

    def step(self, action):
        reward = 0
        if action == 1 and self.agent.top > 0:
            self.agent.y -= self.PADDLE_SPEED
        elif action == 2 and self.agent.bottom < self.HEIGHT:
            self.agent.y += self.PADDLE_SPEED

        if random.random() < 0.5:
            if self.opponent.centery < self.ball.centery:
                self.opponent.y += self.PADDLE_SPEED
            elif self.opponent.centery > self.ball.centery:
                self.opponent.y -= self.PADDLE_SPEED

        self.ball.x += self.ball_dx
        self.ball.y += self.ball_dy

        if self.ball.top <= 0 or self.ball.bottom >= self.HEIGHT:
            self.ball_dy *= -1

        if self.ball.colliderect(self.agent):
            self.ball_dx *= -1
            reward += 0.2

        elif self.ball.colliderect(self.opponent):
            self.ball_dx *= -1

        if self.ball.left <= 0:
            reward = -1
            self.done = True
        elif self.ball.right >= self.WIDTH:
            reward = 1
            self.done = True

        if self.render_mode:
            self.render()

        return self.get_state(), reward, self.done, {}

    def render(self):
        self.surface.fill((0, 0, 0))
        pygame.draw.rect(self.surface, (255, 255, 255), self.agent)
        pygame.draw.rect(self.surface, (255, 255, 255), self.opponent)
        pygame.draw.ellipse(self.surface, (255, 255, 255), self.ball)
        if self.render_mode:
            self.window.blit(self.surface, (0, 0))
            pygame.display.flip()
        self.clock.tick(60)

    def capture_frame(self):
        return pygame.surfarray.array3d(self.surface).swapaxes(0, 1)

    def close(self):
        if self.render_mode:
            pygame.quit()

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, state_bins, n_actions=3, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.state_bins = state_bins
        self.q_table = {}
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def discretize(self, state):
        return tuple(np.digitize(s, b) for s, b in zip(state, self.state_bins))

    def choose_action(self, state):
        d_state = self.discretize(state)
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return self.get_best_action(d_state)

    def get_best_action(self, d_state):
        self.q_table.setdefault(d_state, np.zeros(self.n_actions))
        return int(np.argmax(self.q_table[d_state]))

    def update(self, state, action, reward, next_state, done):
        d_state = self.discretize(state)
        d_next = self.discretize(next_state)

        self.q_table.setdefault(d_state, np.zeros(self.n_actions))
        self.q_table.setdefault(d_next, np.zeros(self.n_actions))

        target = reward + self.gamma * np.max(self.q_table[d_next]) * (0 if done else 1)
        self.q_table[d_state][action] += self.alpha * (target - self.q_table[d_state][action])

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# Q-Learning Agent
class QLearningAgent:
    def __init__(self, state_bins, n_actions=3, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.state_bins = state_bins
        self.q_table = {}
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def discretize(self, state):
        return tuple(np.digitize(s, b) for s, b in zip(state, self.state_bins))

    def choose_action(self, state):
        d_state = self.discretize(state)
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return self.get_best_action(d_state)

    def get_best_action(self, d_state):
        self.q_table.setdefault(d_state, np.zeros(self.n_actions))
        return int(np.argmax(self.q_table[d_state]))

    def update(self, state, action, reward, next_state, done):
        d_state = self.discretize(state)
        d_next = self.discretize(next_state)
        self.q_table.setdefault(d_state, np.zeros(self.n_actions))
        self.q_table.setdefault(d_next, np.zeros(self.n_actions))
        target = reward + self.gamma * np.max(self.q_table[d_next]) * (0 if done else 1)
        self.q_table[d_state][action] += self.alpha * (target - self.q_table[d_state][action])
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# Q-Learning Video Recorder
def record_q_learning_video(q_agent, state_bins, name="Q_Learning", episodes=1):
    os.makedirs("videos", exist_ok=True)
    env = PongEnv(render_mode=True)
    frames = []
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            frame = env.capture_frame()
            frames.append(frame)
            d_state = tuple(np.digitize(s, b) for s, b in zip(state, state_bins))
            action = q_agent.get_best_action(d_state)
            state, _, done, _ = env.step(action)
    env.close()
    output_path = f"videos/{name}_last1.mp4"
    imageio.mimsave(output_path, frames, fps=30)
    print(f"Saved Q-learning video to {output_path}")


# Record SB3 Agent Video
def record_sb3_video(model, name):
    os.makedirs("videos", exist_ok=True)
    env = PongEnv(render_mode=True)
    frames = []
    obs = env.reset()
    done = False
    while not done:
        frame = env.capture_frame()
        frames.append(frame)
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(int(action))
    env.close()
    imageio.mimsave(f"videos/{name}_last1.mp4", frames, fps=30)
    print(f"Saved video: videos/{name}_last1.mp4")


# Main: Train, Record, Compare
if __name__ == "__main__":
    import wandb
    import numpy as np
    from stable_baselines3 import PPO, A2C, DQN
    from stable_baselines3.common.vec_env import DummyVecEnv
    from collections import Counter
    import os

    episodes = 1000

    # Train Q-learning
    print("Training Q-Learning...")
    wandb.init(project="pong-rl-zoo", name="Q_Learning", config={"episodes": episodes})

    state_bins = [
        np.linspace(0, 1, 10),
        np.linspace(0, 1, 10),
        np.linspace(0, 1, 10),
        np.linspace(0, 1, 10),
        np.linspace(-1, 1, 5),
        np.linspace(-1, 1, 5),
    ]
    q_agent = QLearningAgent(state_bins)
    q_rewards = []
    env = PongEnv(render_mode=False)

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        actions_in_episode = []

        while not done:
            action = q_agent.choose_action(state)
            actions_in_episode.append(action)
            next_state, reward, done, _ = env.step(action)
            q_agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step_count += 1

        q_rewards.append(total_reward)
        action_counts = Counter(actions_in_episode)

        wandb.log({
            "Reward": total_reward,
            "Smoothed_Reward": np.mean(q_rewards[-50:]),
            "Win": int(total_reward > 0),
            "Episode_Length": step_count,
            "Epsilon": q_agent.epsilon,
            **{f"Action_{k}_Frequency": v for k, v in action_counts.items()},
            "Agent": "Q_Learning",
            "episode": ep + 1
        })

        if (ep + 1) % 100 == 0:
            print(f"Q-Learning Episode {ep+1}, Reward: {total_reward:.2f}")

    env.close()
    record_q_learning_video(q_agent, state_bins, name="Q_Learning")
    wandb.log({
        "Gameplay": wandb.Video("videos/Q_Learning_last1.mp4", fps=30, format="mp4")
    })
    wandb.finish()

    # Train SB3 agents
    agent_names = ["PPO", "A2C", "DQN"]
    agent_classes = [PPO, A2C, DQN]

    for name, agent_cls in zip(agent_names, agent_classes):
        print(f"Training {name}...")
        wandb.init(project="pong-rl-zoo", name=name, config={"episodes": episodes})

        env = DummyVecEnv([lambda: PongEnv(render_mode=False)])
        model = agent_cls("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=episodes * 100)

        # Evaluate agent
        test_env = PongEnv(render_mode=False)
        episode_rewards = []

        for ep in range(episodes):
            obs = test_env.reset()
            done = False
            total_reward = 0
            step_count = 0
            actions_in_episode = []

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                actions_in_episode.append(int(action))
                obs, reward, done, _ = test_env.step(int(action))
                total_reward += reward
                step_count += 1

            episode_rewards.append(total_reward)
            action_counts = Counter(actions_in_episode)

            wandb.log({
                "Reward": total_reward,
                "Smoothed_Reward": np.mean(episode_rewards[-50:]),
                "Win": int(total_reward > 0),
                "Episode_Length": step_count,
                **{f"Action_{k}_Frequency": v for k, v in action_counts.items()},
                "Agent": name,
                "episode": ep + 1
            })

        test_env.close()

        print(f"Recording video for {name}...")
        record_sb3_video(model, name)
        wandb.log({
            "Gameplay": wandb.Video(f"videos/{name}_last1.mp4", fps=30, format="mp4")
        })

        # Optional: Log last available SB3 loss metrics
        if hasattr(model, 'logger') and hasattr(model.logger, 'name_to_value'):
            for k, v in model.logger.name_to_value.items():
                if isinstance(v, (float, int)):
                    wandb.log({f"{name}/{k}": v})

        wandb.finish()

    print("Training complete. All rewards and behavior logged to Weights & Biases.")
