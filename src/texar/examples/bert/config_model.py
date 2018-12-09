"""Configurations of Transformer model
"""
import copy
import texar as tx

random_seed = 1234
hidden_dim = 512

embed = {
    'dim': hidden_dim,
    'name': 'word_embeddings'
}
vocab_size = 30522

segment_embed = {
    'dim': hidden_dim,
    'name': 'token_type_embeddings'
}
type_vocab_size = 2

encoder = {
    'dim': hidden_dim,
    'initializer': {
        'type': 'variance_scaling_initializer',
        'kwargs': {
            'scale': 1.0,
            'mode': 'fan_avg',
            'distribution': 'uniform',
        },
    },
    'poswise_feedforward': tx.modules.default_transformer_poswise_net_hparams(
        output_dim=hidden_dim)
}

output_size = hidden_dim

opt = {
    'optimizer': {
        'type': 'AdamOptimizer',
        'kwargs': {
            'beta1': 0.9,
            'beta2': 0.997,
            'epsilon': 1e-9
        }
    }
}
