"""
Author: Sean Egger, Alec Rulev
Class: CSI-480-01
Assignment: MDP Pacman Programming Assignment
Date Assigned: Tuesday
Due Date: Wednesday 11:59
 
Description:
A pacman ai program
 
Certification of Authenticity: 
I certify that this is entirely my own work, except where I have given 
fully-documented references to the work of others. I understand the definition 
and consequences of plagiarism and acknowledge that the assessor of this 
assignment may, for the purpose of assessing this assignment:
- Reproduce this assignment and provide a copy to another member of academic
- staff; and/or Communicate a copy of this assignment to a plagiarism checking
- service (which may then retain a copy of this assignment on its database for
- the purpose of future plagiarism checking)
multi_agents.py

q_learning_agents.py

Champlain College CSI-480, Fall 2017
The following code was adapted by Joshua Auerbach (jauerbach@champlain.edu)
from the UC Berkeley Pacman Projects (see license and attribution below).

----------------------
Licensing Information:  You are free to use or extend these projects for
educational purposes provided that (1) you do not distribute or publish
solutions, (2) you retain this notice, and (3) you provide clear
attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

Attribution Information: The Pacman AI projects were developed at UC Berkeley.
The core projects and autograders were primarily created by John DeNero
(denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
Student side autograding was added by Brad Miller, Nick Hay, and
Pieter Abbeel (pabbeel@cs.berkeley.edu).
"""

from game import *
from learning_agents import ReinforcementAgent
from feature_extractors import *

import random
import util
import math


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - compute_value_from_q_values
        - compute_action_from_q_values
        - get_q_value
        - get_action
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.get_legal_actions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.values = util.Counter()

    def get_q_value(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.values[(state, action)]


    def compute_value_from_q_values(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        max_q_value = float('-inf')
        for this_action in self.get_legal_actions(state):
            max_q_value = max(max_q_value, self.get_q_value(state, this_action))
        if max_q_value != float('-inf'):
            return max_q_value
        else:
          return 0.0


    def compute_action_from_q_values(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        if len(self.get_legal_actions(state)) == 0:
            return None
        best_q_value = self.compute_value_from_q_values(state)
        best_action = []
        for this_action in self.get_legal_actions(state):
            if best_q_value == self.get_q_value(state, this_action):
                best_action.append(this_action)
        return random.choice(best_action)

    def get_action(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flip_coin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legal_actions = self.get_legal_actions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if util.flip_coin(self.epsilon):
          return random.choice(legal_actions)
        else:
          return self.compute_action_from_q_values(state)


    def update(self, state, action, next_state, reward):
        """
          The parent class calls this to observe a
          state = action => next_state and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        this_old_value = self.values[(state,action)]
        this_new_value = reward + (self.discount * self.compute_value_from_q_values(next_state))
        self.values[(state,action)] = (1-self.alpha) * this_old_value + self.alpha * this_new_value

    def get_policy(self, state):
        return self.compute_action_from_q_values(state)

    def get_value(self, state):
        return self.compute_value_from_q_values(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, num_training=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        num_training - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['num_training'] = num_training
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def get_action(self, state):
        """
        Simply calls the get_action method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.get_action(self, state)
        self.do_action(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite get_q_value
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.feat_extractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def get_weights(self):
        return self.weights

    def get_q_value(self, state, action):
        """
          Should return Q(state,action) = w * feature_vector
          where * is the dot_product operator
        """
        "*** YOUR CODE HERE ***"
        features = self.feat_extractor.get_features(state,action)
        weights_sum = 0
        for feature, value in features.items():
            weights_sum += self.weights[feature] * value
        return weights_sum

    def update(self, state, action, next_state, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        new_value = reward + self.discount * self.compute_value_from_q_values(next_state)
        old_value = self.get_q_value(state, action)
        difference = new_value - old_value
        features = self.feat_extractor.get_features(state,action)
        for feature, value in features.items():
            self.weights[feature] += self.alpha * difference * features[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodes_so_far == self.num_training:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
