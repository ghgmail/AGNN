import torch
import torch.nn as nn
import numpy as np
import math

torch.set_default_tensor_type('torch.DoubleTensor')


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class Propogator(nn.Module):
    """
    Gated Propogator for GGNN
    Using LSTM gating mechanism
    """

    def __init__(self, hidden_dim, n_node):
        super(Propogator, self).__init__()

        self.n_node = n_node
        self.input_dim = hidden_dim
        self.reset_gate = nn.Sequential(
            nn.Linear(self.input_dim + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(self.input_dim + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(self.input_dim + hidden_dim, hidden_dim),
            nn.Tanh()
        )

    def forward(self, state_in, state_cur, A, edge_rel_type, isBatch,alpha):
        if isBatch:
            # state_in = type_and_hidden_state
            # state_cur = hidden_state
            A = A.double()
	    '''
            for k in range(A.shape[0]):
                dict_one_pair = edge_rel_type[k]

                for tuple in dict_one_pair:
                    A_idx_x = int(tuple[0])
                    A_idx_y = int(tuple[1])
                    embedding_x = state_cur[k, A_idx_x, :].cuda()
                    embedding_y = state_cur[k, A_idx_y, :].cuda()
                    distance = torch.norm(embedding_x - embedding_y)
                    discount = math.exp(-0.1 * alpha * distance)

                    A[k, A_idx_x, A_idx_y] = discount
                    A[k, A_idx_y, A_idx_x] = discount
	    '''
            a_in = torch.bmm(A, state_in)
            a = torch.cat((a_in, state_cur), 2)
            r = self.reset_gate(a)
            z = self.update_gate(a)
            joined_input = torch.cat((a_in, r * state_cur), 2)
            h_hat = self.tansform(joined_input)
            output = (1 - z) * state_cur + z * h_hat
            return output

        else:
            A = A.double()

            dict_one_pair = edge_rel_type
            '''
            for tuple in dict_one_pair:
                A_idx_x = int(tuple[0])
                A_idx_y = int(tuple[1])
                embedding_x = state_cur[A_idx_x, :].cuda()
                embedding_y = state_cur[A_idx_y, :].cuda()
                distance = torch.norm(embedding_x - embedding_y)
                discount = math.exp(-0.1 * distance)
                A[A_idx_x, A_idx_y] = discount
                A[A_idx_y, A_idx_x] = discount
 	    '''
            a_in = torch.mm(A, state_in)
            a = torch.cat((a_in, state_cur), -1)
            r = self.reset_gate(a)
            z = self.update_gate(a)
            joined_input = torch.cat((a_in, r * state_cur), 1)
            h_hat = self.tansform(joined_input)
            output = (1 - z) * state_cur + z * h_hat
            return output


class GGNN(nn.Module):

    def __init__(self, hidden_dim, type_dim, n_node, n_edge_types, n_node_types, step, ent_pre_embedding,alpha):
        super(GGNN, self).__init__()
        self.embedding = nn.Embedding(n_node + 1, hidden_dim)
        self.embedding.weight = nn.Parameter(ent_pre_embedding)
        self.node_type_embedding = nn.Embedding(n_node_types + 1, type_dim)
        self.hidden_dim = hidden_dim
        self.type_dim = type_dim
        self.n_edge_types = n_edge_types
        self.n_node_types = n_node_types
        self.n_node = n_node
        self.n_steps = step
        self.alpha = alpha

        self.in_layer = nn.Sequential(
            nn.Linear(hidden_dim + type_dim, hidden_dim),
            nn.ReLU())

        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim / 2),
            nn.ReLU(),
            nn.Linear(hidden_dim / 2, hidden_dim / 4),
            nn.ReLU(),
            nn.Linear(hidden_dim / 4, 1)
        )

        # Propogation Model
        self.propogator = Propogator(self.hidden_dim, self.n_node)

    def forward(self, inputs, type_indices, A, edge_rel_type_slice, user_idx, movie_idx, isBatch):

        if isBatch:
            # Dense layer
            hidden_state = self.embedding(inputs)
            type_state = self.node_type_embedding(type_indices)
            hidden_state = hidden_state.cuda()
            type_state = type_state.cuda()
            hidden_state_and_type = torch.cat((type_state, hidden_state), 2)
            hidden_state = self.in_layer(hidden_state_and_type)
            # GRU
            for i_step in range(self.n_steps):
                hidden_state = self.propogator(hidden_state, hidden_state, A, edge_rel_type_slice, True,self.alpha)
            # user embedding & item embedding
            user_list = []
            item_list = []


            for i in range(hidden_state.shape[0]):
                user_list.append(hidden_state[i][user_idx[i]])
                item_list.append(hidden_state[i][movie_idx[i]])
            user_embedding = torch.stack(user_list)
            movie_embedding = torch.stack(item_list)

            temp = torch.cat((user_embedding, movie_embedding), -1)

            out = self.out(temp).cuda()
            out = torch.sigmoid(out)
            return out

        else:
            hidden_state = self.embedding(inputs)
            type_state = self.node_type_embedding(type_indices)
            type_state = type_state.cuda()
            hidden_state = hidden_state.cuda()
            hidden_state_and_type = torch.cat((type_state, hidden_state), -1)
            hidden_state = self.in_layer(hidden_state_and_type)
            for i_step in range(self.n_steps):
                hidden_state = self.propogator(hidden_state, hidden_state, A, edge_rel_type_slice, False,self.alpha)

            user_embedding = hidden_state[user_idx]
            movie_embedding = hidden_state[movie_idx]
            temp = torch.cat((user_embedding, movie_embedding), -1).cuda()
            out = self.out(temp)
            out = torch.sigmoid(out)
            return out


