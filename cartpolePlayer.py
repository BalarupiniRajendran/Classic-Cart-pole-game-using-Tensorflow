import random
import gym
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout
from keras.optimizers import Adam

EPISODES = 1000
RUNS = 100 # Change to 20 to check 60000
trials = []
random.seed(3) # Change random seed here

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=6000)
        self.memory_buffer = deque(maxlen=6000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.95
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def forget(self):
        self.memory.clear()

    def remove(self,time):
        for _i in range(time):
            self.memory.pop()

    def temp_remember(self, state, action, reward, next_state, done):
        self.memory_buffer.append((state, action, reward, next_state, done))
   
    def replace_memory(self):
        self.memory = deque(self.memory_buffer)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        #noise = random.uniform(0,0.05)  #Uncomment for actuator noise
        #choose = random.getrandbits(1)
        #act_values[0][choose] = act_values[0][choose] + noise
        #act_values[0][1] = act_values[0][1] + noise #Uncomment for actuator noise
        return np.argmax(act_values[0])  # returns action

    def replaywithtemp(self, batch_size):
        minibatch = random.sample(self.memory_buffer, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.memory = self.memory_buffer

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    success = 0
    for r in range(RUNS):
        agent = DQNAgent(state_size, action_size)
        # agent.load("./save/cartpole-dqn.h5")
        done = False
        batch_size = 100
        best_score = 0
        for e in range(EPISODES):
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            noise = random.uniform(0,0.1)
            for t in range(6000): # Change to 60000 for 60000 timesteps
                #if RUNS == 1:
                    #env.render() #uncomment for render
                    #time.sleep(0.02) #Uncomment for render in real time
                #env.render()
                #noise = math.radians((random.gauss(0,(0.1)**(1/2.0))))    #Uncomment for Gaussian noise use 0.2 instead of 0.1 for change in variance
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                reward = reward if not done else -100
                next_state = np.reshape(next_state, [1, state_size])
                #noise = random.uniform(0,0.05)     #Uncomment for Uniform noise use 0.1 instead of 0.05 for 10% noise
                #next_state[0][2] = next_state[0][2] * (1 + noise)      #Uncomment for Uniform Noise
                #next_state[0][2] = next_state[0][2] + noise        #Uncomment for Gaussian noise
                if best_score < 5999: # Change to 59999 for 60000 timesteps 
                    agent.remember(state, action, reward, next_state, done)
                else:
                    agent.temp_remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    print("run: {}/{}, episode: {}/{}, score: {}, e: {:.2}, best_score: {}"
                          .format(r, RUNS, e, EPISODES, t, agent.epsilon, best_score))
                    if t > best_score:
                        best_score = t
                    break
            if best_score == 5999: # Change to 59999 for 60000 timesteps case
                print("goal reached at episode: ", e)
                trials.append(e)
                success = success + 1
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
                #if best_score == 5999 and done and t >= 5999:
                #    agent.replace_memory()


                #elif done:
                    #if (best_score < 1000 and t < best_score / 2) or (best_score >= 1000 and t < best_score / 10:
                #if t < best_score / 4:
                        #agent.remove(t)
                    #agent.replay(batch_size)
    average = sum(trials) / len(trials)
    minimum = min(trials)
    print ("Average number of trials to reach goal:", average)
    print ("Minimum number of trials to reach goal:", minimum)
    print ("Percentage of Success:", success / RUNS * 100, "%") 
        # if e % 10 == 0:
#     agent.save("./save/cartpole-dqn.h5")
