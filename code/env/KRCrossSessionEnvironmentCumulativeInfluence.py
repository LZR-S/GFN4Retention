import numpy as np
import utils
import torch
import torch.nn as nn
import random
from copy import deepcopy
from argparse import Namespace
from torch.utils.data import DataLoader
from torch.distributions import Categorical

from reader import *
from model.simulator import *
from env.KRCrossSessionEnvironment_GPU_GFN import KRCrossSessionEnvironment_GPU_GFN
import utils


class KRCrossSessionEnvironmentCumulativeInfluence(KRCrossSessionEnvironment_GPU_GFN):
    '''
    KuaiRand simulated environment on GPU machines
    Components:
    - multi-behavior user response model: 
        - (user history, user profile) --> user_state
        - (user_state, item) --> feedbacks (e.g. click, long_view, like, ...)
    - user leave model:
        - user temper reduces to <1 and leave
        - user temper drops gradually through time and further drops when the user is unsatisfactory about a recommendation
    - user retention model:
        - [end of session user_state] --> user retention
    '''
    
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - max_n_session
        - max_return_day
        - next_day_return_bias
        - feedback_influence_on_return
        - from KREnvironment_WholeSession_GPU:
            - uirm_log_path
            - slate_size
            - episode_batch_size
            - item_correlation
            - single_response
            - from BaseRLEnvironment
                - max_step_per_episode
                - initial_temper
        '''
        parser = KRCrossSessionEnvironment_GPU_GFN.parse_model_args(parser)
        parser.add_argument('--personalized_return_coef', type=float, required=0.1, 
                            help='influence of personalization on return bias')
        parser.add_argument('--past_reward_decay', type=float, required=0.5, 
                            help='decay factor of past internal reward will influence retention probabilities')
        return parser
    
    def __init__(self, args):
        '''
        attributes:
        - max_n_session, max_return_day, next_day_return_bias, feedback_influence_on_return
        - day_bias_multiplier
        - random_personalization
        - action_dim
        '''
        assert 0 <= args.past_reward_decay <=1.0
        self.personalized_return_coef = args.personalized_return_coef
        self.past_reward_decay = args.past_reward_decay
        super().__init__(args)
        print(-torch.log(self.response_weights + 0.001))
        self.response_weights = torch.softmax(1.0-torch.log(self.response_weights + 0.001), dim=-1)
        print('Log-scale response weights:', self.response_weights)
        
    def reset(self, params = {}):
        '''
        Reset environment with new sampled users
        @input:
        - params: {'batch_size': scalar, 
                    'empty_history': True if start from empty history, default = False
                    'initial_history': start with initial history, empty_history must be False}
        @process:
        - self.batch_iter
        - self.current_observation
        - self.current_step
        - self.current_temper
        - self.env_history
        @output:
        - observation: {'user_profile': {'user_id': (B,), 
                                         'uf_{feature_name}': (B, feature_dim)}, 
                        'user_history': {'history': (B, max_H), 
                                         'history_if_{feature_name}': (B, max_H, feature_dim), 
                                         'history_{response}': (B, max_H), 
                                         'history_length': (B, )}}
        '''
        initial_observation = super().reset(params)
        self.past_reward = torch.zeros(self.episode_batch_size).to(torch.float).to(self.device)
        return initial_observation
    
    def get_leave_signal(self, user_state, action, response):
        '''
        User leave model maintains the user temper, and a user leaves when the temper drops below 1.
        @input:
        - user_state: not used in this env
        - action: not used in this env
        - response: (B, slate_size, n_feedback)
        @process:
        - update temper
        @output:
        - done_mask: 
        '''
        # (B, slate_size, n_feedback)
        point_feedback = response * self.response_weights.view(1,1,-1)
        # (B, slate_size)
        combined_reward = torch.sum(point_feedback, dim = 2)
        # (B, )
        internal_reward = torch.mean(combined_reward, dim = 1)
        # (B, )
#         print('past_reward:\n', self.past_reward)
#         print('internal_reward:\n', internal_reward)
        self.past_reward = self.past_reward * self.past_reward_decay + internal_reward * (1.0 - self.past_reward_decay)
        
        temper_down = 1
        self.current_temper -= temper_down
        done_mask = self.current_temper < 1
        return done_mask

    
    def get_retention(self, observation, response, done_mask):
        new_gt_state = self.get_ground_truth_user_state(observation['user_profile'], observation['user_history'])
        # (B, )
        personal_bias = torch.sigmoid(self.random_personalization(new_gt_state).view(self.episode_batch_size))
        personal_bias *= self.personalized_return_coef
#         # (B, slate_size, n_feedback)
#         point_reward = response * self.response_weights.view(1,1,-1)
#         # (B, )
#         response_bias = torch.mean(torch.sum(point_reward, dim=2), dim=1) * self.feedback_influence_on_return
        response_bias = self.feedback_influence_on_return * (self.past_reward - 0.01)
        # (B, )
        P = torch.clamp(personal_bias + response_bias + self.next_day_return_bias, 0.001, 0.999)
        logP = torch.log(P)
        log1_P = torch.log(1-P)
        # (B, max_ret_day)
        preds = logP.view(-1, 1) + log1_P.view(-1,1) * self.day_bias_multiplier
        
        return {'preds': preds, 'state': new_gt_state}
    
    
        
        
        
        