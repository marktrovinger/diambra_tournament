import os
import time
import cv2
import numpy as np
from stable_baselines3 import PPO
from utils import linear_schedule, AutoSave


from diambra_environment.diambraGym import diambraGym
# diambra sb3 wrapper
from makeDiambraEnvSB3 import make_diambra_env

timeDepSeed = int((time.time()-int(time.time()-0.5))*1000)

repo_base_path = os.path.join(os.path.abspath("/"), "home/gansandgains/DIAMBRAenvironment")

diambraEnvKwargs = {}
diambraEnvKwargs["gameId"]          = "doapp"
diambraEnvKwargs["roms_path"]       = os.path.join(repo_base_path, "roms/") # Absolute path to roms

diambraEnvKwargs["mame_diambra_step_ratio"] = 6
diambraEnvKwargs["render"]      = True
diambraEnvKwargs["lock_fps"]    = False # Locks to 60 FPS
diambraEnvKwargs["sound"]       = diambraEnvKwargs["lock_fps"] and diambraEnvKwargs["render"]

# 1P
diambraEnvKwargs["player"] = "Random"

# Game specific
diambraEnvKwargs["difficulty"] = 3
diambraEnvKwargs["characters"]  = [["Kasumi", "Random"], ["Kasumi", "Random"]]
diambraEnvKwargs["charOutfits"] = [2, 2]

tensorBoardFolder = "./{}_ppo_TB_CustCnn_bL_d_noComb/".format(diambraEnvKwargs["gameId"])
modelFolder = "./{}_ppo_Model_CustCnn_bL_d_noComb/".format(diambraEnvKwargs["gameId"])

os.makedirs(modelFolder, exist_ok=True)

diambraGymKwargs = {}
diambraGymKwargs["P2brain"]               = None
diambraGymKwargs["continue_game"]         = 0.0
diambraGymKwargs["show_final"]            = False
diambraGymKwargs["gamePads"]              = [None, None]
diambraGymKwargs["actionSpace"]           = ["discrete", "multiDiscrete"]
diambraGymKwargs["attackButCombinations"] = [False, False]
diambraGymKwargs["actBufLen"]             = 12

# Wrappers kwargs
wrapperKwargs = {}
wrapperKwargs["hwc_obs_resize"]    = [256, 256, 1]
wrapperKwargs["normalize_rewards"] = True
wrapperKwargs["clip_rewards"]      = False
wrapperKwargs["frame_stack"]       = 6
wrapperKwargs["dilation"]          = 1
wrapperKwargs["scale"]             = True
wrapperKwargs["scale_mod"]         = 0

# Additional Observations
keyToAdd = []
keyToAdd.append("actionsBuf")
keyToAdd.append("ownHealth")
keyToAdd.append("oppHealth")
keyToAdd.append("ownPosition")
keyToAdd.append("oppPosition")
keyToAdd.append("stage")
keyToAdd.append("character")

numEnv=8

envId = "Train"
env = make_diambra_env(diambraGym, env_prefix=envId, num_env=numEnv, seed=timeDepSeed, 
                       diambra_kwargs=diambraEnvKwargs, 
                       diambra_gym_kwargs=diambraGymKwargs,
                       wrapper_kwargs=wrapperKwargs, 
                       key_to_add=keyToAdd, use_subprocess=True)

print("Obs_space = ", env.observation_space)
print("Obs_space type = ", env.observation_space.dtype)
print("Obs_space high = ", env.observation_space.high)
print("Obs_space low = ", env.observation_space.low)

print("Act_space = ", env.action_space)
print("Act_space type = ", env.action_space.dtype)
if diambraGymKwargs["actionSpace"][0] == "multiDiscrete":
    print("Act_space n = ", env.action_space.nvec)
else:
    print("Act_space n = ", env.action_space.n)

# Policy param

n_actions = env.get_attr("n_actions")[0][0]
actBufLen = diambraGymKwargs["actBufLen"]

policyKwargs={}
policyKwargs["n_add_info"] = actBufLen*(n_actions[0]+n_actions[1]) + len(keyToAdd)-2 # No Char Info
policyKwargs["layers"] = [64, 64]

# PPO param

setGamma = 0.94

setLearningRate = linear_schedule(2.5e-4, 2.5e-6)
#setLearningRate = linear_schedule(5.0e-5, 2.5e-6)

setClipRange = linear_schedule(0.15, 0.025)
#setClipRange = linear_schedule(0.05, 0.025)

setClipRangeVf = setClipRange

# Initialize the model
model = PPO('ActorCriticCnnPolicy', env, verbose=1, 
             gamma=setGamma, nminibatches=4, noptepochs=4, n_steps=128,
             learning_rate=setLearningRate, cliprange=setClipRange, 
             cliprange_vf=setClipRangeVf, 
             tensorboard_log=tensorBoardFolder, policy_kwargs=policyKwargs)

print("Model discount factor = ", model.gamma)

# Create the callback: autosave every USER DEF steps
autoSaveCallback = AutoSave(check_freq=1000000, numEnv=numEnv, save_path=modelFolder+"0M_")

# Train the agent
time_steps = 20000000
model.learn(total_timesteps=time_steps, callback=autoSaveCallback)

# Save the agent
model.save(os.path.join(modelFolder, "20M"))