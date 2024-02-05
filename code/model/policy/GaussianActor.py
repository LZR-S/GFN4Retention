from matplotlib.pyplot import axes, axis
import torch
import torch.nn as nn

from model.general import BaseModel
from model.components import DNN
from model.policy.RandomPolicy import RandomPolicy


class GaussianActor(RandomPolicy):
    '''
    KuaiRand Multi-Behavior user response model
    '''
    
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - initial_mu
        - initial_var
        - from RandomPolicy - from ActionTransformer:
            - user_latent_dim
            - item_latent_dim
            - enc_dim
            - attn_n_head
            - transformer_d_forward
            - transformer_n_layer
            - action_hidden_dims
            - dropout_rate
            - from BaseModel:
                - model_path
                - loss
                - l2_coef
        '''
        parser = RandomPolicy.parse_model_args(parser)
        parser.add_argument('--initial_mu', type=float, default=0., 
                            help='initial mu')
        parser.add_argument('--initial_var', type=float, default=1., 
                            help='initial variance')
        return parser
        
    def __init__(self, args, env, device):
        super().__init__(args, env, device)
        self.mu = torch.FloatTensor([args.initial_mu] * self.action_dim).to(self.device).view(1,-1)
        self.var = torch.FloatTensor([args.initial_var] * self.action_dim).to(self.device).view(1,-1)
        self.state = torch.FloatTensor([args.initial_var] * self.state_dim).to(self.device).view(1,-1)
        
    def to(self, device):
        new_self = super(GaussianActor, self).to(device)
        return new_self

    def get_forward(self, feed_dict: dict):
        '''
        @input:
        - feed_dict: {
            'user_id': (B,)
            'uf_{feature_name}': (B,feature_dim), the user features
            'item_id': (B,), the target item
            'if_{feature_name}': (B,feature_dim), the target item features
            'history': (B,max_H)
            'history_if_{feature_name}': (B,max_H,feature_dim), the history item features
            ... (irrelevant input)
        }
        @output:
        - out_dict: {'preds': (B,-1,n_feedback), 'reg': scalar}
        '''
        B = feed_dict['user_id'].shape[0]
        
        # (B, action_dim)
        sampled_action = torch.randn(B, self.action_dim).to(self.device) * torch.sqrt(self.var) + self.mu
        user_state = torch.tile(self.state, (B,1))
        
        return {'action': sampled_action, 'state': user_state, 'reg': 0}
    
    