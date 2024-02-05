import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import utils

from model.agent.BaseRLAgent import BaseRLAgent

class CEM(BaseRLAgent):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - initial_mu
        - initial_logvar
        - top_p
        - args from BaseRLAgent
            - gamma
            - n_iter
            - train_every_n_step
            - initial_greedy_epsilon
            - final_greedy_epsilon
            - elbow_greedy
            - check_episode
            - with_eval
            - save_path
        '''
        parser = BaseRLAgent.parse_model_args(parser)
        parser.add_argument('--top_p', type=int, default=20, 
                            help='top p selection of population')
        parser.add_argument('--mitigate_alpha', type=float, default=0.1, 
                            help='')
        return parser
    
    def __init__(self, args, facade):
        super().__init__(args, facade)
        self.actor = facade.actor
        self.top_p = args.top_p
        self.alpha = args.mitigate_alpha
    
    def train(self):
        if len(self.n_iter) > 2:
            self.load()
        
        t = time.time()
        
        print("Run procedures before training")
        self.action_before_train()
        t = time.time()
        start_time = t
        # training
        print("Training:")
        observation = self.facade.reset_env({"batch_size": self.episode_batch_size})
        step_offset = sum(self.n_iter[:-1])
        for i in tqdm(range(step_offset, step_offset + self.n_iter[-1])):
            observation, reward = self.cem_step(i, observation)
            reward = torch.mean(reward).item()
            # print("one step reward: ", reward)
            self.training_history['reward'].append(reward)
            if i % self.check_episode == 0 and i > self.check_episode:
                t_ = time.time()
                print(f"Episode step {i}, time diff {t_ - t}, total time dif {t - start_time})")
                print(self.log_iteration(i))
                t = t_
                if i % self.save_episode == 0:
                    self.save()
        self.action_after_train()
    
    def action_before_train(self):
        '''
        Action before training:
        - facade setup:
            - buffer setup
        - run random episodes to build-up the initial buffer
        '''
        self.facade.initialize_train() # buffer setup
        # training records
        self.training_history = {"reward": []}
    
    def action_after_train(self):
        self.facade.stop_env()
        
    def get_report(self):
        episode_report = self.facade.get_episode_report(1)
        train_report = {k: np.mean(v[-100:]) for k,v in self.training_history.items()}
        # print(self.training_history['reward'][-50:])
        # train_report = {k: v[-2] for k,v in self.training_history.items()}
        return episode_report, train_report

    def cem_step(self, *episode_args):
        '''
        One step of interaction
        '''
        episode_iter, observation = episode_args
        with torch.no_grad():
            # sample action from prior
            policy_output = self.facade.apply_policy(observation, self.actor)
            
            # evaluate action on environment
            new_observation, user_feedback, updated_observation = self.facade.env_step(policy_output)
            # (B,)
            reward = user_feedback['reward'].view(-1)
            # print(reward)
            # self.training_history['evaluation'].append(torch.mean(reward).detach().cpu().numpy())
            
            # selection
            _, indices = torch.topk(reward, self.top_p)
            
            # update prior
            selected_action = policy_output['action'][torch.tensor(indices).to(torch.long)]
            self.actor.mu = (1 - self.alpha) * self.actor.mu + self.alpha * torch.mean(selected_action, dim = 0)
            self.actor.var = (1 - self.alpha) * self.actor.var + self.alpha * torch.var(selected_action, dim = 0)
            
            observation = new_observation
        return updated_observation, reward
    
    def log_iteration(self, step):
        episode_report, train_report = self.get_report()
        log_str = f"step: {step} @ episode report: {episode_report} @ step loss: {train_report}\n"
        with open(self.save_path + ".report", 'a') as outfile:
            outfile.write(log_str)
        return log_str
    
    def save(self):
        torch.save({'mu': self.actor.mu, 'var': self.actor.var}, self.save_path + "_stat")
    
    def load(self):
        checkpoint = torch.load(self.save_path + "_stat", map_location=self.device)
        self.actor.mu = checkpoint['mu']
        self.actor.var = checkpoint['var']