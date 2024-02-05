from matplotlib.pyplot import axes, axis
import torch
import torch.nn as nn

from model.general import BaseModel
from model.components import DNN
from model.policy.ActionTransformer import ActionTransformer


class RandomPolicy(ActionTransformer):
    '''
    KuaiRand Multi-Behavior user response model
    '''
    
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - from ActionTransformer:
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
        parser = ActionTransformer.parse_model_args(parser)
        return parser
        
    def __init__(self, args, env, device):
        super().__init__(args, env, device)
        
    def to(self, device):
        new_self = super(RandomPolicy, self).to(device)
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
        
        # user encoding
        state_encoder_output = self.encode_state(feed_dict, B)
        # (B, state_dim)
        user_state = state_encoder_output['state'].view(B,self.state_dim)

        # get action
        # (B, n_feedback)
#         action_emb = self.actionModule(user_state)
        action_emb = torch.randn(B, self.action_dim).to(self.device)
        # regularization terms
#         reg = self.get_regularization(self.feedbackEncoder, 
#                                       self.itemFeatureKernel, self.userFeatureKernel, 
#                                       self.posEmb, self.transformer, self.actionModule)
#         reg = reg + state_encoder_output['reg']
        
        return {'action': action_emb, 'state': user_state, 'reg': 0}
    
    