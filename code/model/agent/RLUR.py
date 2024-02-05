import time
import copy
import torch
import torch.nn.functional as F
import numpy as np

import utils
from model.agent.TD3 import TD3, interactions
from model.UserEncoder import UserEncoder
from model.facade.TwoRewardFacade import TwoRewardFacade
    
class RLUR(TD3):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        from TD3:
            - actor_lr
            - critic_lr
            - actor_decay
            - critic_decay
            - target_mitigate_coef
            - args from BaseRLAgent:
                - gamma
                - n_iter
                - train_every_n_step
                - initial_greedy_epsilon
                - final_greedy_epsilon
                - elbow_greedy
                - check_episode
                - with_eval
                - save_path
                - episode_batch_size
                - batch_size
        '''
        parser = TD3.parse_model_args(parser)
        # parser.add_argument('--with_immediate', action='store_true', 
                            # help='True if using immediate responses and rnd during training')
        parser.add_argument('--with_immediate', type=bool, default=False)
        parser.add_argument('--with_rnd', action='store_true', 
                            help='True if using rnd during training')
        parser.add_argument('--im_critic_lr', type=float, default=1e-4, 
                            help='learning rate for RND network')
        parser.add_argument('--im_critic_decay', type=float, default=1e-4, 
                            help='weight decay for RND network')
        parser.add_argument('--rnd_lr', type=float, default=1e-4, 
                            help='learning rate for RND network')
        parser.add_argument('--rnd_weight', type=float, default=1, 
                            help='weight for rnd reward')
        parser.add_argument('--rnd_decay', type=float, default=1e-4, 
                            help='weight decay for RND network')
        parser.add_argument('--lambda_t', type=float, default=1., 
                            help='weight for retention actor loss')
        parser.add_argument('--lambda_I', type=float, default=1., 
                            help='weight for immediate response actor loss')
#         parser.add_argument('--do_lambda_i_decay', action='store_true', 
#                             help='decay the weight of RND reward during actor learning')
        
        return parser
    
    def __init__(self, args, facade: TwoRewardFacade):
        '''
        self.with_rnd
        self.im_critic1, self.im_critic2
        self.im_critic1_target, self.im_critic2_target
        self.im_critic1_optimizer, self.im_critic2_optimizer
        from TD3:
            self.actor
            self.actor_target
            self.actor_optimizer
            self.critic1, self.critic2
            self.critic1_target, self.critic2_target
            self.critic1_optimizer, self.critic2_optimizer
            self.tau
            from BaseRLAgent:
                self.device
                self.gamma
                self.n_iter
                self.train_every_n_step
                self.check_episode
                self.save_episode
                self.save_path
                self.facade
                self.exploration_scheduler
                self.episode_batch_size
                self.batch_size
        '''
        self.with_im = args.with_immediate
        self.with_RND = args.with_rnd
        self.rnd_weight = args.rnd_weight
        self.lambda_T = args.lambda_t
        self.lambda_I = args.lambda_I
        super().__init__(args, facade)
        
        self.im_critic1 = facade.critics[2]
        self.im_critic1_target = copy.deepcopy(self.im_critic1)
        self.im_critic1_optimizer = torch.optim.Adam(self.im_critic1.parameters(), lr=args.im_critic_lr, 
                                                 weight_decay=args.im_critic_decay)
        
        self.im_critic2 = facade.critics[3]
        self.im_critic2_target = copy.deepcopy(self.im_critic2)
        self.im_critic2_optimizer = torch.optim.Adam(self.im_critic2.parameters(), lr=args.im_critic_lr, 
                                                 weight_decay=args.im_critic_decay)
        
        self.RNDEncoder = UserEncoder(args, facade.env.reader.get_statistics(), self.device)
        self.RNDEncoder = self.RNDEncoder.to(self.device)
        self.RND_optimizer = torch.optim.Adam(self.RNDEncoder.parameters(), lr=args.rnd_lr, 
                                              weight_decay=args.rnd_decay)
        self.RNDEncoder_target = UserEncoder(args, facade.env.reader.get_statistics(), self.device)
        for param in self.RNDEncoder_target.parameters():
            param.requires_grad = False
        self.RNDEncoder_target = self.RNDEncoder_target.to(self.device)
        
    def action_before_train(self):
        '''
        Action before training:
        - facade setup:
            - buffer setup
        - run random episodes to build-up the initial buffer
        '''
        super().action_before_train()
        # training records
        self.training_history["im_critic1_loss"] = []
        self.training_history["im_critic2_loss"] = []
        self.training_history["im_critic1"] = []
        self.training_history["im_critic2"] = []
        self.training_history["im_reward"] = []
        self.training_history["rnd"] = []
            

    def step_train(self):
        observation, policy_output, reward, done_mask, next_observation = self.facade.sample_buffer(self.batch_size)
        reward = {'reward': reward['reward'].to(torch.float), 'im_reward': reward['im_reward'].to(torch.float)}
        done_mask = done_mask.to(torch.float)
        
        critic_loss_dict, actor_loss = self.get_rlur_loss(observation, policy_output, reward, done_mask, next_observation)

        
        self.training_history['actor_loss'].append(actor_loss.item())
        self.training_history['critic1_loss'].append(critic_loss_dict['C1_loss'])
        self.training_history['critic1'].append(critic_loss_dict['Q1'])
        self.training_history['critic2_loss'].append(critic_loss_dict['C2_loss'])
        self.training_history['critic2'].append(critic_loss_dict['Q2'])
        self.training_history['im_critic1_loss'].append(critic_loss_dict['im_C1_loss'])
        self.training_history['im_critic1'].append(critic_loss_dict['im_Q1'])
        self.training_history['im_critic2_loss'].append(critic_loss_dict['im_C2_loss'])
        self.training_history['im_critic2'].append(critic_loss_dict['im_Q2'])
        self.training_history['reward'].append(torch.mean(reward['reward']).item())
        self.training_history['im_reward'].append(torch.mean(reward['im_reward']).item())
        self.training_history['rnd'].append(critic_loss_dict['rnd'])

        action_std = torch.std(policy_output['action'], axis=0)
        action_mean = torch.mean(policy_output['action'], axis=0)
        for i, interact in enumerate(interactions):
            self.training_history[interact].append(torch.mean(observation['user_history'][f'history_{interact}'][:, -1]).item())
            self.training_history[interact + "_std"].append(action_std[i].item())
            self.training_history[interact + "_mean"].append(action_mean[i].item())
            
        # Update the frozen target models
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.im_critic1.parameters(), self.im_critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.im_critic2.parameters(), self.im_critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {"step_loss": (self.training_history['actor_loss'][-1], 
                              self.training_history['critic1_loss'][-1], 
                              self.training_history['critic2_loss'][-1], 
                              self.training_history['im_critic1_loss'][-1], 
                              self.training_history['im_critic2_loss'][-1], 
                              self.training_history['critic1'][-1], 
                              self.training_history['critic2'][-1], 
                              self.training_history['im_critic1'][-1], 
                              self.training_history['im_critic2'][-1], 
                              self.training_history['reward'][-1], 
                              self.training_history['im_reward'][-1], 
                              self.training_history['rnd'][-1])}
    
    def get_rlur_loss(self, observation, policy_output, reward, done_mask, next_observation, 
                     do_actor_update = True, do_critic_update = True):
        '''
        @input:
        - observation: {'user_profile': {'user_id': (B,), 
                                         'uf_{feature_name}': (B, feature_dim)}, 
                        'user_history': {'history': (B, max_H), 
                                         'history_if_{feature_name}': (B, max_H, feature_dim), 
                                         'history_{response}': (B, max_H), 
                                         'history_length': (B, )}}
        - policy_output: {'state': (B, state_dim), 
                          'action: (B, action_dim)}
        - reward: {'reward': (B,), 'im_reward': (B,)}
        - done_mask: (B,)
        - next_observation: the same format as @input-observation
        '''
        policy_output = utils.wrap_batch(policy_output, device = self.device)
        # Retention critic update
        ret_future_decay = ((self.gamma * done_mask) + (1 - done_mask))
        C1_loss, C2_loss, Q1, Q2, _ = self.do_critic_update(observation, policy_output, next_observation,
                                                         self.critic1, self.critic1_target, self.critic1_optimizer, 
                                                         self.critic2, self.critic2_target, self.critic2_optimizer, 
                                                         reward['reward'], ret_future_decay)
                
        # Immediate response critic update
        im_future_decay = self.gamma # * (1 - done_mask)
        im_C1_loss, im_C2_loss, im_Q1, im_Q2, rnd = self.do_critic_update(observation, policy_output, next_observation,
                                                         self.im_critic1, self.im_critic1_target, self.im_critic1_optimizer, 
                                                         self.im_critic2, self.im_critic2_target, self.im_critic2_optimizer, 
                                                         reward['im_reward'] if self.with_im else 0, 
                                                         im_future_decay, with_rnd = self.with_RND)

        # Compute actor loss
        policy_output = self.facade.apply_policy(observation, self.actor)
        critic_output = self.facade.apply_critic(observation, policy_output, self.critic1)
        im_critic_output = self.facade.apply_critic(observation, policy_output, self.im_critic1)
        actor_loss = - critic_output['q'].mean() * self.lambda_T - im_critic_output['q'].mean() * self.lambda_I
#         print("lambda_I: ", self.lambda_I)
#         print("withrnd: ", self.with_RND)

        if do_actor_update and self.actor_lr > 0:
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
        return {'C1_loss': C1_loss, 'C2_loss': C2_loss, 'Q1': Q1, 'Q2': Q2, 
                'im_C1_loss': im_C1_loss, 'im_C2_loss': im_C2_loss, 'im_Q1': im_Q1, 'im_Q2': im_Q2, 
                'rnd': torch.mean(rnd).item() if self.with_RND else 0}, actor_loss
    
    def do_critic_update(self, observation, policy_output, next_observation,
                         C1, C1_target, C1_optimizer, C2, C2_target, C2_optimizer, reward, 
                         future_decay, with_rnd = False):
        # Compute the target Q value
        next_policy_output = self.facade.apply_policy(next_observation, self.actor_target, self.epsilon, do_explore = True)
        target_C1_output = self.facade.apply_critic(next_observation, next_policy_output, C1_target)
        target_C2_output = self.facade.apply_critic(next_observation, next_policy_output, C2_target)
        target_Q = torch.min(target_C1_output['q'], target_C2_output['q'])
        # r+gamma*Q' when done; r+Q when not done
#         target_Q = reward + self.gamma *  (1 - done_mask) * target_Q.detach()
        #target_Q = reward

        C1_loss, C2_loss, Q1, Q2 = 0, 0, 0, 0
        if self.critic_lr > 0:
            # Get current Q estimate
            current_C1_output = self.facade.apply_critic(observation, policy_output, C1)
            current_Q1 = current_C1_output['q']
            # Compute critic loss
            rnd = self.rnd_weight * self.get_RND_reward(observation) if with_rnd else 0
            Q1_prime = reward + rnd + future_decay * target_Q.detach()
            C1_loss = F.mse_loss(current_Q1, Q1_prime).mean()
            Q1 = torch.mean(current_Q1).item()

            # Optimize the critic
            if with_rnd:
                self.RND_optimizer.zero_grad()
            C1_optimizer.zero_grad()
            C1_loss.backward()
            C1_optimizer.step()
            if with_rnd:
                self.RND_optimizer.step()
            
            current_C2_output = self.facade.apply_critic(observation, policy_output, C2)
            current_Q2 = current_C2_output['q']
            # Compute critic loss
            rnd = self.get_RND_reward(observation) if with_rnd else 0
            Q2_prime = reward + rnd + future_decay * target_Q.detach()
            C2_loss = F.mse_loss(current_Q2, Q2_prime).mean()
            Q2 = torch.mean(current_Q2).item()

            # Get current Q estimate
#             for name,param in self.RNDEncoder.named_parameters():
#                 print(name, param[:5])
#             input()
            # Optimize the critic
            if with_rnd:
                self.RND_optimizer.zero_grad()
            C2_optimizer.zero_grad()
            C2_loss.backward()
            C2_optimizer.step()
            if with_rnd:
                self.RND_optimizer.step()
#             for name,param in self.RNDEncoder.named_parameters():
#                 print(name, param[:5])
#             input()
            
            
        return C1_loss.item(), C2_loss.item(), Q1, Q2, rnd
    
    def get_RND_reward(self, observation):
        batch = {}
        batch.update(observation['user_profile'])
        batch.update(observation['user_history'])
        wrapped_batch = utils.wrap_batch(batch, self.device)
        # (B, state_dim)
        user_embedding = self.RNDEncoder(wrapped_batch)['state']
        # (B, state_dim)
        target_user_embedding = self.RNDEncoder_target(wrapped_batch)['state']
        # (B, state_dim)
        diff = (user_embedding - target_user_embedding)
        # (B,)
        intrinsic_reward = torch.mean(diff * diff, dim = 1)
        return intrinsic_reward

    def save(self):
        for C, opt, prefix in [(self.critic1, self.critic1_optimizer, '_critic1'), 
                               (self.critic2, self.critic2_optimizer, '_critic2'), 
                               (self.im_critic1, self.im_critic1_optimizer, '_im_critic1'), 
                               (self.im_critic2, self.im_critic2_optimizer, '_im_critic2')]:
            torch.save(C.state_dict(), self.save_path + prefix)
            torch.save(opt.state_dict(), self.save_path + prefix + "_optimizer")

        torch.save(self.actor.state_dict(), self.save_path + "_actor")
        torch.save(self.actor_optimizer.state_dict(), self.save_path + "_actor_optimizer")


    def load(self):
            
        self.critic1.load_state_dict(torch.load(self.save_path + "_critic1", map_location=self.device))
        self.critic1_optimizer.load_state_dict(torch.load(self.save_path + "_critic1_optimizer", map_location=self.device))
        self.critic1_target = copy.deepcopy(self.critic1)
        
        self.critic2.load_state_dict(torch.load(self.save_path + "_critic2", map_location=self.device))
        self.critic2_optimizer.load_state_dict(torch.load(self.save_path + "_critic2_optimizer", map_location=self.device))
        self.critic2_target = copy.deepcopy(self.critic2)
        
        self.im_critic1.load_state_dict(torch.load(self.save_path + "_im_critic1", map_location=self.device))
        self.im_critic1_optimizer.load_state_dict(torch.load(self.save_path + "_im_critic1_optimizer", map_location=self.device))
        self.im_critic1_target = copy.deepcopy(self.critic1)
        
        self.im_critic2.load_state_dict(torch.load(self.save_path + "_im_critic2", map_location=self.device))
        self.im_critic2_optimizer.load_state_dict(torch.load(self.save_path + "_im_critic2_optimizer", map_location=self.device))
        self.im_critic2_target = copy.deepcopy(self.critic2)

        self.actor.load_state_dict(torch.load(self.save_path + "_actor", map_location=self.device))
        self.actor_optimizer.load_state_dict(torch.load(self.save_path + "_actor_optimizer", map_location=self.device))
        self.actor_target = copy.deepcopy(self.actor)
