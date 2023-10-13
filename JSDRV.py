from base.base_model import BaseModel
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
'''
node_dict: posts dict
claim_embedding: claim embedding
nodes_embedding: posts embedding
'''
class Net(BaseModel):
    def __init__(self, config, node_dict, claim_embedding, nodes_embedding):
        super(Net, self).__init__()

        self.config = config
        self.claim_embedding = claim_embedding
        self.nodes_embedding = nn.Embedding.from_pretrained(nodes_embedding)
        self.node_dict = node_dict

        self.actor = nn.Linear(self.config['model']['embedding_size']*3, self.config['model']['embedding_size'])

        self.critic = nn.Linear(self.config['model']['embedding_size']*3, self.config['model']['embedding_size'])

        self.elu = torch.nn.RELU(inplace=False)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, state_input, action_input):
        if len(state_input.shape) < len(action_input.shape):
            if len(action_input.shape) == 3:
                state_input = torch.unsqueeze(state_input, 1)
                state_input = state_input.expand(state_input.shape[0], action_input.shape[1],
                                                 state_input.shape[2])
            else:
                state_input = torch.unsqueeze(state_input, 1)
                state_input = torch.unsqueeze(state_input, 1)
                state_input = state_input.expand(state_input.shape[0], action_input.shape[1], action_input.shape[2],
                                                 state_input.shape[3])

        # actor
        actor_x = self.elu(self.actor(torch.cat([state_input, action_input], dim=-1)))
        act_probs = self.sigmoid(actor_x)

        return act_probs

class SelectionPolicy(BaseModel):

    def __init__(self, config, claim_embedding, nodes_embedding, reason_embedding, post_adj, relation_adj, node_dict, neibor_embedding):
        super(SelectionPolicy, self).__init__()
        self.config = config
        self.claim_embedding = claim_embedding
        self.post_adj = post_adj
        self.relation_adj = relation_adj
        self.MAX_DEPTH = 3
        self.node_dict = node_dict
        self.neibor_embedding = nn.Embedding.from_pretrained(neibor_embedding)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.cos = nn.CosineSimilarity(dim=-1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        self.nodes_embedding = nn.Embedding.from_pretrained(nodes_embedding)
        self.reason_embedding = nn.Embedding.from_pretrained(reason_embedding)

        self.anchor_embedding_layer = nn.Linear(self.config['model']['embedding_size'],self.config['model']['embedding_size'])
        self.anchor_weighs1_layer1 = nn.Linear(self.config['model']['embedding_size'], self.config['model']['embedding_size'])
        self.anchor_weighs1_layer2 = nn.Linear(self.config['model']['embedding_size'], 1)

        self.policy_net = Net(self.config, self.node_dict, self.claim_embedding, nodes_embedding)
        self.target_net = Net(self.config, self.node_dict, self.claim_embedding, nodes_embedding)

    def get_claim_embedding(self, sampleids):
        claim_embeddings = []
        for sampleids in sampleids:
            claim_embeddings.append(torch.FloatTensor(self.claim_embedding[sampleids]).cuda())
        return torch.stack(claim_embeddings)

    def get_next_action(self, state_id_input_batch):
        next_action_id = []
        next_action_r_id = []
        for i in range(len(state_id_input_batch)):
            next_action_id.append([])
            next_action_r_id.append([])
            for j in range(len(state_id_input_batch[i])):
                if int(state_id_input_batch[i][j].data.cpu().numpy()) in self.post_adj:
                    next_action_id[-1].append(self.post_adj[int(state_id_input_batch[i][j].data.cpu().numpy())])
                    next_action_r_id[-1].append(self.relation_adj[int(state_id_input_batch[i][j].data.cpu().numpy())])
                else:
                    next_action_id[-1].append([0 for k in range(20)])
                    next_action_r_id[-1].append([0 for k in range(20)])
        next_action_space_id = torch.LongTensor(next_action_id).cuda()
        next_state_space_embedding = self.nodes_embedding(next_action_space_id)
        next_action_r_id = torch.LongTensor(next_action_r_id).cuda()
        next_action_r = self.reason_embedding(next_action_r_id)
        return next_action_space_id, next_state_space_embedding, next_action_r_id, next_action_r



    def get_state_input(self, claim_embedding, depth, history_nodes, hhistory_reason):
        if depth == 0:
            state_embedding = torch.cat(
                [claim_embedding, torch.tensor(np.zeros((claim_embedding.shape[0], 128))).float().cuda()], dim=-1)
        else:
            history_nodes_embedding = self.nodes_embedding(history_nodes)
            reason_embedding = self.reason_embedding(hhistory_reason)
            state_embedding_new = history_nodes_embedding
            state_embedding_new = torch.mean(state_embedding_new, dim=1, keepdim=False)
            state_embedding = torch.cat([claim_embedding, state_embedding_new], dim=-1)
        return state_embedding

    def get_neighbors(self, entities):
        neighbor_entities = []
        neighbor_relations = []
        for entity_batch in entities:
            neighbor_entities.append([])
            neighbor_relations.append([])
            for entity in entity_batch:
                if type(entity) == int:
                    neighbor_entities[-1].append(self.post_adj[entity])
                    neighbor_relations[-1].append(self.adj_relation[entity])
                else:
                    neighbor_entities[-1].append([])
                    neighbor_relations[-1].append([])
                    for entity_i in entity:
                        neighbor_entities[-1][-1].append(self.post_adj[entity_i])
                        neighbor_relations[-1][-1].append(self.adj_relation[entity_i])

        return torch.LongTensor(neighbor_entities).cuda(), torch.LongTensor(neighbor_relations).cuda()

    def forward(self,claim_embedding, post, history_nodes, history_reasons):
        depth = 0

        act_probs_steps = []
        action_embedding = self.nodes_embedding(history_nodes)
        reason_embedding = self.reason_embedding(history_reasons)
        action_embedding = action_embedding + reason_embedding
        action_id = post
        state_input = self.get_state_input(claim_embedding, depth, history_nodes, history_reasons)

        while (depth < self.MAX_DEPTH):
            act_probs = self.policy_net(state_input, action_embedding)

            depth = depth + 1
            state_input = self.get_state_input(claim_embedding, depth, history_nodes, history_reasons)

            act_probs_steps.append(act_probs)
            action_embedding = self.nodes_embedding(action_id) + self.reason_embedding(history_reasons)



        return act_probs_steps

