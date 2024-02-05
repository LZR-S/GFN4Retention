import time
import copy
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

import utils
from model.agent.BaseRLAgent import BaseRLAgent

class GFN4Retention_wif_ablation(BaseRLAgent):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - critic_lr
        - critic_decay
        - target_mitigate_coef
        - args from BaseRLAgent:
            - gamma
            - reward_func
            - n_iter
            - train_every_n_step
            - start_policy_train_at_step
            - initial_epsilon
            - final_epsilon
            - elbow_epsilon
            - explore_rate
            - do_explore_in_train
            - check_episode
            - save_episode
            - save_path
            - actor_lr
            - actor_decay
            - batch_size
            '''
        parser = BaseRLAgent.parse_model_args(parser)

        parser.add_argument('--gfn_forward_hidden_dims', type=int, nargs="+", default=[128], 
                            help='hidden dimensions of state_slate encoding layers')
        
        parser.add_argument('--gfn_flow_hidden_dims', type=int, nargs="+", default=[128], 
                            help='hidden dimensions of flow estimator')
        
        parser.add_argument('--gfn_forward_offset', type=float, default=1.0, 
                            help='smooth offset of forward logp of TB loss') #b_f
        
        parser.add_argument('--gfn_backward_offset', type=float, default=1.0, 
                            help='smooth offset of backward logp of TB loss') #b_b
        
        parser.add_argument('--gfn_reward_smooth', type=float, default=1.0, 
                            help='reward smooth offset in the backward part of TB loss') #b_r 0.5
        
        parser.add_argument('--gfn_Z', type=float, default=0.1, 
                            help='average reward offset') #b_z 0.1
            
        parser.add_argument('--lambda_gfn', type=float, default=0.5, 
                            help='-')
            
        parser.add_argument('--tn_balance', type=float, default=0.5, 
                            help='-')
            
                   
        
        parser.add_argument('--critic_lr', type=float, default=1e-4, 
                                help='decay rate for critic')
        parser.add_argument('--critic_decay', type=float, default=1e-4, 
                                help='decay rate for critic')
        parser.add_argument('--target_mitigate_coef', type=float, default=0.01, 
                                help='mitigation factor')
        parser.add_argument('--noise_std', type=float, default=0.1, 
                                help='noise standard deviation for action exploration')
        parser.add_argument('--noise_clip', type=float, default=1.0, 
                                help='noise clip bound for action exploration')
        return parser
    
    
    def __init__(self, *input_args):
        '''
        Initialize the GFN model.
        Setup the flow model, optimizer, and other necessary components.
        '''
        args, env, actor, critics, buffer = input_args
        
        super().__init__(args, env, actor, buffer)
        
        self.gfn_forward_hidden_dims = args.gfn_forward_hidden_dims
        self.gfn_flow_hidden_dims = args.gfn_flow_hidden_dims
        self.gfn_forward_offset = args.gfn_forward_offset
        self.gfn_backward_offset = args.gfn_backward_offset
        self.gfn_reward_smooth = args.gfn_reward_smooth
        self.gfn_Z = args.gfn_Z
        self.lambda_gfn = args.lambda_gfn
        self.tn_balance = args.tn_balance
        
        self.noise_std = args.noise_std
        self.noise_clip = args.noise_clip
        
        self.critic_lr = args.critic_lr
        self.critic_decay = args.critic_decay
        self.tau = args.target_mitigate_coef
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr, 
                                                weight_decay=args.actor_decay)

        # Initialize the scheduler

        self.scheduler1 = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=400, gamma=0.01)
        
        self.flow_estimator = critics[0]
        self.critic1 = critics[0]
        self.critic1_optimizer = torch.optim.Adam(self.flow_estimator.parameters(), lr=args.critic_lr, 
                                                 weight_decay=args.critic_decay)
        
        self.scheduler2 = torch.optim.lr_scheduler.StepLR(self.critic1_optimizer, step_size=400, gamma=0.01)

        self.backward_estimator = critics[1]
        self.critic2 = critics[1]
        self.critic2_optimizer = torch.optim.Adam(self.backward_estimator.parameters(), lr=args.critic_lr, 
                                                 weight_decay=args.critic_decay)

        self.scheduler3 = torch.optim.lr_scheduler.StepLR(self.critic2_optimizer, step_size=400, gamma=0.01)

        # register models that will be saved
        self.registered_models.append((self.critic1, self.critic1_optimizer, "_critic1"))
        self.registered_models.append((self.critic2, self.critic2_optimizer, "_critic2"))

        
        
    def setup_monitors(self):
        '''
        This is used in super().action_before_train() in super().train()
        Then the agent will call several rounds of run_episode_step() for collecting initial random data samples
        '''
        super().setup_monitors()
        self.training_history.update({'Q1_loss': [], 'Q2_loss': [],
                                      'Q1': [], 'Q2': [], 'next_Q1': [], 'next_Q2': [], 'target_Q': [], 'DB_forward': [], 'DB_backward': []})
        self.eval_history.update({'avg_retention': [], 'max_retention': [], 'min_retention': []})
        

    def run_episode_step(self, *episode_args):
        '''
        One step of interaction
        '''
        episode_iter, epsilon, observation, do_buffer_update, do_explore = episode_args
        self.epsilon = epsilon
        is_train = False
        with torch.no_grad():
            # sample action
            policy_output = self.apply_policy(observation, self.actor, 
                                              epsilon, do_explore, is_train)
            # apply action on environment and update replay buffer
            action_dict = {'action': policy_output['action']}
            new_observation, user_feedback, updated_info = self.env.step(action_dict)
            # calculate reward
            R = self.get_reward(user_feedback)
            user_feedback['reward'] = R
            self.current_sum_reward = self.current_sum_reward + R
            done_mask = user_feedback['done']
                                     
            # monitor update
            if torch.sum(done_mask) > 0:
                self.eval_history['avg_retention'].append(user_feedback['retention'].mean().item()) 
                self.eval_history['max_retention'].append(user_feedback['retention'].max().item()) 
                self.eval_history['min_retention'].append(user_feedback['retention'].min().item())
                self.eval_history['avg_total_reward'].append(self.current_sum_reward.mean().item())
                self.eval_history['max_total_reward'].append(self.current_sum_reward.max().item())
                self.eval_history['min_total_reward'].append(self.current_sum_reward.min().item())
                self.current_sum_reward *= 0
            
            self.eval_history['avg_reward'].append(R.mean().item())
            self.eval_history['reward_variance'].append(torch.var(R).item())
            for i,resp in enumerate(self.env.response_types):
                self.eval_history[f'{resp}_rate'].append(user_feedback['immediate_response'][:,:,i].mean().item())
            # update replay buffer
            if do_buffer_update:
                self.buffer.update(observation, policy_output, user_feedback, updated_info['updated_observation'])
            observation = new_observation
        return new_observation
    
    
    def get_flow(self, state_dict):
        '''
        The flow estimator F(s_t|u)
        :param params:
        :param state_dict: {'emb': (B, state_dim), 'wtm': (B,)}
        :return: {'flow': (B, 1), 'log_f': (B, 1)}
        '''
        log_flow = self.flow_estimator(state_dict)['v']
        flow = torch.exp(log_flow)
        
        return {'flow': flow, 'log_f': log_flow}
    
    
    
    def get_forward_prob(self, observation):
        
        # policy output
        
        
        '''
        Forward function P(s_{t+1} | s_t, u) in GFN
        :param params: 输入参数
        :param current_state_dict: {'emb': (B, state_dim), 'wtm': (B,)}
        :param current_action: a_{t} 采样中的当前动作
        :return: {'prob_f': (B, 1), forward probability P(s_{t+1}|s_{t}) = P(a_{t}|s_{t}),
                  'mu', 'sigma': (B, action_dim), 输出动作的 distribution 参数}
        '''
        # current action distribution
        is_train = True
        current_policy_output = self.apply_policy(observation, self.actor, self.epsilon, True, is_train)
        mu = current_policy_output['mu']
        sigma = current_policy_output['sigma']
        current_action = current_policy_output['action']
        action_dim = mu.shape[1]
        mu_bias = 0.01
        #mu, sigma, mu_bias = self.take_action(current_state_dict['emb'], current_state_dict['wtm'], params)
        mu_avg = (mu + mu_bias) / 2

        dist = Normal(mu_avg, sigma)
        P = dist.log_prob(current_action).exp() + 0.001  # Add a small constant for numerical stability
        P = P.view(-1, action_dim)
        gfn_p_forward = P.sum(dim=1, keepdim=True)  # Sum over action dimensions

        return {'prob_f': gfn_p_forward, 'mu': mu_avg, 'sigma': sigma}


    def get_backward_prob(self, current_dict, next_dict):

        output_prob = self.backward_estimator(current_dict, next_dict)['prob_b']

        return {'prob_b': output_prob}
    
    
    def step_train(self):
        '''
        @process:
        '''
        observation, policy_output, user_feedback, done_mask, next_observation = self.buffer.sample(self.batch_size)
        #reward = user_feedback['reward']
        #reward = reward.to(torch.float)
        done_mask = done_mask.to(torch.float)
        
        #critic_loss_list, actor_loss = self.get_gfn_loss(observation, policy_output, user_feedback, done_mask, next_observation)
        loss = self.get_gfn_loss(observation, policy_output, user_feedback, done_mask, next_observation)
        
        self.training_history['actor_loss'].append(loss.item())
        self.training_history['Q1_loss'].append(0)
        self.training_history['Q2_loss'].append(0)
        self.training_history['Q1'].append(0)
        self.training_history['Q2'].append(0)
        self.training_history['next_Q1'].append(0)
        self.training_history['next_Q2'].append(0)
        self.training_history['target_Q'].append(0)
        
        #target_Q, next_Q1, next_Q2, Q1_loss, Q1, Q2_loss, Q2 = critic_loss_list
        #self.training_history['actor_loss'].append(actor_loss.item())
        #self.training_history['Q1_loss'].append(Q1_loss)
        #self.training_history['Q2_loss'].append(Q2_loss)
        #self.training_history['Q1'].append(Q1)
        #self.training_history['Q2'].append(Q2)
        #self.training_history['next_Q1'].append(next_Q1)
        #self.training_history['next_Q2'].append(next_Q2)
        #self.training_history['target_Q'].append(target_Q)

    
    
    def get_gfn_loss(self, observation, policy_output, user_feedback, done_mask, next_observation):

        is_train = True

        # Apply policy for the current and next observations
        current_policy_output = self.apply_policy(observation, self.actor, self.epsilon, True, is_train)
        next_policy_output = self.apply_policy(next_observation, self.actor, self.epsilon, True, is_train)

        # Extract necessary components for loss calculation
        b_f, b_b, b_r, b_z = self.gfn_forward_offset, self.gfn_backward_offset, self.gfn_reward_smooth, self.gfn_Z
        lambda_gfn, tn_balance = self.lambda_gfn, self.tn_balance

        # Calculate flow, forward prob, backward prob for current and next states
        current_flow_output = self.get_flow(current_policy_output)
        log_F_t = current_flow_output['log_f']

        forward_output = self.get_forward_prob(observation)
        P_F = forward_output['prob_f']
        log_P_F = torch.log(P_F + b_f)

        next_flow_output = self.get_flow(next_policy_output)
        log_F_next = next_flow_output['log_f']

        backward_output = self.get_backward_prob(current_policy_output, next_policy_output)
        P_B = backward_output['prob_b']
        log_P_B = torch.log(P_B + b_b)

        # Calculate reward components and loss components
        trajectory_reward = user_feedback['reward'].view(-1, 1) + 0.001
        log_reward = torch.log(trajectory_reward + b_r)

        DB_forward_terminal = log_F_t
        DB_forward_nonterminal = log_F_t + log_P_F
        DB_forward = DB_forward_terminal * done_mask * tn_balance + \
                     DB_forward_nonterminal * (1 - done_mask) * (1 - tn_balance)

        DB_backward_terminal = log_reward
        
        # Without immediate feedback
        # DB_backward_nonterminal = log_F_next + log_P_B
        # With immediate feedback
        #immediate_response = user_feedback['immediate_response'][:, 0].view(-1, 1)
        #immediate_response = user_feedback['immediate_response'].mean(dim=1, keepdim=True)
        user_feedback['immediate_response_weight'] = self.env.response_weights
        point_reward = user_feedback['immediate_response'].reshape(-1, 6, 7) * user_feedback['immediate_response_weight'].view(1,1,-1)
        combined_reward = torch.sum(point_reward, dim = 2)
        immediate_reward = torch.mean(combined_reward, dim = 1)

        DB_backward_nonterminal = log_F_next + log_P_B + torch.log(immediate_reward+0.01)

        
        DB_backward = DB_backward_terminal * done_mask * tn_balance + \
                      DB_backward_nonterminal * (1 - done_mask) * (1 - tn_balance)

        # Calculate the final GFN DB loss
        gfn_db_diff = DB_forward - DB_backward + b_z
        gfn_db_loss = torch.mean(torch.square(gfn_db_diff))

        # Calculate the total loss
        total_loss = lambda_gfn * gfn_db_loss
        
        self.actor_optimizer.zero_grad()
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()

        total_loss.backward()

        self.actor_optimizer.step()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()
        
        #全部一起 train
        self.training_history['DB_forward'].append(torch.mean(DB_forward).item())
        self.training_history['DB_backward'].append(torch.mean(DB_backward).item())

        #self.scheduler1.step()
        #self.scheduler2.step()
        #self.scheduler3.step()

        return total_loss


    
    def apply_policy(self, observation, actor, *policy_args):
        '''
        @input:
        - observation:{'user_profile':{
                           'user_id': (B,)
                           'uf_{feature_name}': (B,feature_dim), the user features}
                       'user_history':{
                           'history': (B,max_H)
                           'history_if_{feature_name}': (B,max_H,feature_dim), the history item features}
        - actor: the actor model
        - epsilon: scalar
        - do_explore: boolean
        - is_train: boolean
        @output:
        - policy_output
        '''
        epsilon = policy_args[0]
        do_explore = policy_args[1]
        is_train = policy_args[2]
        out_dict = self.actor(observation)
        
        if do_explore:
            mu = out_dict['action']
            out_dict['mu'] = mu
            # sampling noise of action embedding
            if np.random.rand() < epsilon:
                action = torch.clamp(torch.rand_like(mu)*self.noise_std, -self.noise_clip, self.noise_clip)
            else:
                action = mu + torch.clamp(torch.rand_like(mu)*self.noise_std, 
                                                      -self.noise_clip, self.noise_clip)
            out_dict['action'] = action
            out_dict['sigma'] = self.noise_std
        return out_dict

    