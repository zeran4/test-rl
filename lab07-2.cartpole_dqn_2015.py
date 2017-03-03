# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import random
import dqn
from collections import deque

import gym
from gym import wrappers
env = gym.make('CartPole-v0')

# Constants defining our neural network
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

dis = 0.9
REPLAY_MEMORY = 50000

def replay_train(mainDQN, targetDQN, train_batch):
    x_stack = np.empty(0).reshape(0, input_size)
    y_stack = np.empty(0).reshape(0, output_size)
    
    #Get stored information from the buffer
    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predict(state)
        
        # terminal?
        if done:
            Q[0, action] = reward
        else:
            # get target from target DQN (Q')
            Q[0, action] = reward + dis * np.max(targetDQN.predict(next_state))

        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, state])
        
    #Train our network using target and predicted Q values on each episode
    return mainDQN.update(x_stack, y_stack)

def bot_play(env, mainDQN):
#    env = wrappers.Monitor(env, "/tmp/gym-results", force=True)    # playing 레코딩

    # See our trained network in action
    s = env.reset()
    reward_sum = 0
    while True:
        env.render()
        a = np.argmax(mainDQN.predict(s))
        s, reward, done, _ = env.step(a)
        reward_sum += reward
        if done:
            print("Toral score: {}".format(reward_sum))
            break
            
def get_copy_var_ops(*, dest_scope_name="target", src_scope_name="main"):
    # Copy variables src_scope to dest_scope
    op_holder = []
    
    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)
    
    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))
        
    return op_holder

def main():
    max_episodes = 50000
    max_step_count = 10000    # 충분히 오래 play한 경우 무한 루프(?)를 방지하기 위해 끊음
    max_streaks_count = 5    # 충분히 오래 play한 경우가 연속해서 나올 경우, 해당 DQN으로 실제 play를 보자.
    streaks_count = 0
    episode_count = 5    # 에피소드 몇번마다 학습을 할 것인가? (강의 : 10)
    mini_batch_count = 10     # mini batch때의 수 (강의 : 10)
    mini_batch_repeat_count = 50    # mini batch 반복 횟수 (강의 : 50)

    # store the previous observations in replay memory
    replay_buffer = deque()

    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess, input_size, output_size, name="main")
        targetDQN = dqn.DQN(sess, input_size, output_size, name="target")
        tf.global_variables_initializer().run()

        # initial copy q_net -> target_net
        copy_ops = get_copy_var_ops(dest_scope_name="target",
                                    src_scope_name="main")
        sess.run(copy_ops)
        
        for episode in range(max_episodes):
            e = 1. / ((episode / 10) + 1)
            done = False
            step_count = 0

            state = env.reset()

            while not done:
                if np.random.rand(1) < e:
                    action = env.action_space.sample()
#                    print("action(sample) : {}".format(action))
                else:
                    # Choose an action by greedily from the Q-network
                    action = np.argmax(mainDQN.predict(state))
#                    print("action(predict) : {}".format(action))

                # Get new state and reward from environment
                next_state, reward, done, _ = env.step(action)
                if done:    # big penalty
                    reward = -100

                # Save the experience to our buffer
                replay_buffer.append((state, action, reward, next_state, done))
                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()

                state = next_state
                step_count += 1
                if step_count > max_step_count:    # Good enough
                    streaks_count += 1
                    break

            if step_count <= max_step_count:
                streaks_count = 0

            if streaks_count > 0:
                print("Episode: {}    steps: {}      streaks: {}".format(episode, step_count, streaks_count))
            else:
                print("Episode: {}    steps: {}".format(episode, step_count))

            if (streaks_count == max_streaks_count):
                break;

            if step_count > max_step_count:
                pass
                # break


#            if episode % episode_count == 1:    # 1 때 하고 그 후부터 episode_count 단위로 수행
            if (episode+1) >= episode_count and (episode+1) % episode_count == 0:
                # Get a random batch of experiences.
                for _ in range(mini_batch_repeat_count):
                    # Minibatch works better
                    minibatch = random.sample(replay_buffer, mini_batch_count)
                    loss, _ = replay_train(mainDQN, targetDQN, minibatch)
                    
                print("Loss: ", loss)
                # copy q_net => target_net
                sess.run(copy_ops)

        bot_play(env, mainDQN)
    
if __name__ == "__main__":
    main()