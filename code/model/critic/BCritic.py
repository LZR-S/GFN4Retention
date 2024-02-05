import torch.nn.functional as F
import torch.nn as nn
import torch

from model.components import DNN
from utils import get_regularization

class BCritic(nn.Module):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - critic_hidden_dims
        - critic_dropout_rate
        '''
        parser.add_argument('--critic2_hidden_dims', type=int, nargs='+', default=[128], 
                            help='specificy a list of k for top-k performance')
        parser.add_argument('--critic2_dropout_rate', type=float, default=0.2, 
                            help='dropout rate in deep layers')
        return parser
    
    def __init__(self, args, environment, policy):
        super().__init__()
        self.state_dim = policy.state_dim
        self.action_dim = policy.action_dim
        input_dim = self.state_dim * 2 + self.action_dim  # Concatenated input dimension
        self.net = DNN(input_dim, args.critic2_hidden_dims, 1, 
                       dropout_rate=args.critic2_dropout_rate, do_batch_norm=True)
        
    def forward(self, current_dict, next_dict):
        '''
        @input:
        - feed_dict: {'next_state_emb': (B, state_dim), 'current_action': (B, action_dim), 'current_state_emb': (B, state_dim)}
        '''
        next_state_emb = next_dict['state']
        current_action = current_dict['action']
        current_state_emb = current_dict['state']

        # Concatenate state and action embeddings
        output = torch.cat([next_state_emb, current_action, current_state_emb], dim=1)
        
        output = self.net(output)

        #for layer in self.net:
        #    output = F.relu(layer(output))

        output_prob = torch.sigmoid(output).view(-1, 1) * 0.999 + 0.001
        reg = get_regularization(self.net)

        return {'prob_b': output_prob, 'reg': reg}
