import numpy as np
from operator import itemgetter
import copy

class TreeNode(object):


    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {} 
        self._n_visits = 0
        self._Q = 0
        self._u = prior_p
        self._P = prior_p

    def expand(self, action_priors):

        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self):

        return max(self._children.items(), key=lambda act_node: act_node[1].get_value())

    def update(self, leaf_value, c_puct):

        self._n_visits += 1

        self._Q += (leaf_value - self._Q) / self._n_visits
        
        if not self.is_root():
            self._u = c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)

    def update_recursive(self, leaf_value, c_puct):

        if self._parent:
            self._parent.update_recursive(leaf_value, c_puct)
        self.update(leaf_value, c_puct)

    def get_value(self):

        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):


    def __init__(self, value_fn, policy_fn, rollout_policy_fn, lmbda=0.5, c_puct=5,
                 rollout_limit=100, playout_depth=10, n_playout=100):

        self._root = TreeNode(None, 1.0)
        self._value = value_fn
        self._policy = policy_fn
        self._rollout = rollout_policy_fn
        self._lmbda = lmbda
        self._c_puct = c_puct
        self._rollout_limit = rollout_limit
        self._L = playout_depth
        self._n_playout = n_playout
        self._player_id = 0
        # self._env = env 

    def _exchange_player(self):

        self._player_id = 1 if self._player_id == 0 else 0

    def _playout(self, state, env, leaf_depth):

        node = self._root
        for i in range(leaf_depth):
            # Only expand node if it has not already been done. Existing nodes already know their
            # prior.
            if node.is_leaf():
                action_probs = self._policy(state, self._player_id)
                # print(action_probs)
                # Check for end of game.
                if len(action_probs) == 0:
                    break
                node.expand(action_probs)
            # Greedily select next move.
            action, node = node.select()
            # state.do_move(action)
            state = env.step(action)
            self._exchange_player()


        # Evaluate the leaf using a weighted combination of the value network, v, and the game's
        # winner, z, according to the rollout policy. If lmbda is equal to 0 or 1, only one of
        # these contributes and the other may be skipped. Both v and z are from the perspective
        # of the current player (+1 is good, -1 is bad).
        v = self._value(state, self._player_id) if self._lmbda < 1 else 0
        z = self._evaluate_rollout(state, env, self._rollout_limit) if self._lmbda > 0 else 0
        leaf_value = (1 - self._lmbda) * v + self._lmbda * z

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(leaf_value, self._c_puct)

    def _evaluate_rollout(self, state, env, limit):
        """Use the rollout policy to play until the end of the game, returning +1 if the current
        player wins, -1 if the opponent wins, and 0 if it is a tie.
        """
        player = state.observations["current_player"]
        for i in range(limit):
            action_probs = self._rollout(state, self._player_id)
            # if len(action_probs) == 0:
            #     break
            max_action = max(action_probs, key=itemgetter(1))[0]
            # state.do_move(max_action)
            state = env.step(max_action)
            self._exchange_player()
            if state.last():
                break
        # else:
        #     # If no break from the loop, issue a warning.
        #     print("WARNING: rollout reached move limit")
        
        return state.rewards[0]

    def get_move(self, state, env):

        self._player_id = 0

        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            env_cpy = copy.deepcopy(env)
            self._playout(state_copy, env_cpy, self._L)

        # chosen action is the *most visited child*, not the highest-value one
        # (they are the same as self._n_playout gets large).
        return max(self._root._children.items(), key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know about the subtree, assuming
        that get_move() has been called already. Siblings of the new root will be garbage-collected.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)
