import sys 
sys.path.append("../")
from algorithms.policy_gradient import PolicyGradient
from algorithms.dqn import DQN
from mcts import MCTS
from agent.agent import Agent
import numpy as np
from numpy.random import normal

NUM_ACTIONS = 26

def random_policy_fn(time_step, player_id):

    legal_actions = time_step.observations["legal_actions"][player_id]
    probs = np.zeros(NUM_ACTIONS)
    probs[legal_actions] = 1
    probs /= sum(probs)
    return [i for i in zip(range(len(probs)),probs)]

def random_value_fn(time_step, player_id):

    return normal(scale=0.3)

class MCTSAgent():

    def __init__(self,policy_module, rollout_module, playout_depth = 10, n_playout = 100):

        if policy_module == None and rollout_module == None:
            
            self.policy_fn = self.rollout_policy_fn = random_policy_fn
            self.value_fn = random_value_fn

        else:

            self.value_fn = policy_module.value_fn
            self.policy_fn = policy_module.policy_fn
            self.rollout_policy_fn = rollout_module.policy_fn

        self.mcts = MCTS(value_fn = self.value_fn,
                         policy_fn = self.policy_fn,
                         rollout_policy_fn = self.rollout_policy_fn,
                         playout_depth=playout_depth,
                         n_playout = n_playout)

    def step(self,timestep,env):

        move = self.mcts.get_move(timestep,env)
        self.mcts.update_with_move(move)
        return move 

        