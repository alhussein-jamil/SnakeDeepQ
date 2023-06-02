import yaml 
import pygame
import matplotlib.pyplot as plt
import torch 
import mediapy as media

from snake_env import SnakeEnv
from deepQ import Agent


config = yaml.load(open('SnakeDeepQ.yaml', 'r'), Loader=yaml.FullLoader)
snakie = SnakeEnv(config["env"])
agent = Agent(snakie.observation_space, snakie.action_space, config)
run = 0
simulation_freq = config["simulation_freq"]
max_steps = config["batch_size"]
batch_size = config["mini_batch_size"]





while True:
    #collect experience : 

    steps = 0
    total_reward = 0
    rollout = 0
    
    frames = []
    while True: 
        state = snakie.reset()[0]
        done = False 
        while not done :
            if(run % simulation_freq == 0 and rollout == 0):
                if(config["env"]["render_mode"] == "human"):
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                    snakie.render("human")
                    pygame.time.wait(20)
                else: 
                    frames.append(snakie.render("rgb_array"))
            action = agent.compute_action(state)
            next_state, reward, done, _ = snakie.step(action)
            agent.add_experience(state, torch.argmax(action), reward, next_state, done)
            state = next_state
            steps += 1
            total_reward += reward
            if steps > max_steps:
                break
        rollout+=1
        if steps > max_steps:
            break

    if run % simulation_freq == 0 and config["env"]["render_mode"] != "human":
        if(len(frames) > 0):
            media.write_video(f"./sims/run_{run}.mp4", frames, fps=10)
            print(f"Saved video to ./sims/run_{run}.mp4")
        else:
            print("no frames to write")
    run += 1

    # train the agent
    loss= agent.train(batch_size)
    print("Run : ", run , "total reward : ", total_reward/max_steps," epsilon : ", agent.epsilon , "loss : ", loss)

    