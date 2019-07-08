from GGNNModel_Relation import GGNN
from GGNNTrain_Relation import GGNNTrain_Relation

import numpy as np
import argparse
import torch
import json

import pickle
import random
from datetime import datetime

from data_preprocess import load_file_as_dict




def load_paths(fr_file):
    paths_between_pairs = {}
    label_dict = {}
    for line in fr_file:
        nodes_in_a_path = line.replace('\n', '').replace('\r','').split(',')
        user_node= nodes_in_a_path[0]
        last_node = nodes_in_a_path[-1]
        if last_node == 'None':
            movie_node = nodes_in_a_path[1]
            key = (user_node, movie_node)
            user_id = user_node.replace('u','')
            movie_id = movie_node.replace('i','')
            paths_between_pairs.update({key: []})
            if user_id not in label_dict:
                label_dict.update({user_id:[movie_id]})
            else:
                if movie_id not in label_dict[user_id]:
                    label_dict[user_id].append(movie_id)
        else:
            movie_node = nodes_in_a_path[-1]
            key = (user_node, movie_node)
            path = nodes_in_a_path
            user_id = user_node.replace('u','')
            movie_id = movie_node.replace('i','')
            if key not in paths_between_pairs:
                paths_between_pairs.update({key: [path]})
            else:
                #if len(paths_between_pairs[key]) <30:
                paths_between_pairs[key].append(path)
            if user_id not in label_dict:
                label_dict.update({user_id:[movie_id]})
            else:
                if movie_id not in label_dict[user_id]:
                    label_dict[user_id].append(movie_id)
    return paths_between_pairs,label_dict


if __name__ == "__main__":


    # learning-rate : 0.001 dimension:16-4
    parser = argparse.ArgumentParser()
    type_dim=32
    hidden_dim=128
    iteration=15
    learning_rate=0.001
    weight_decay=1e-4
    step=2
    batch_size=256
    edge_types = 4 # user-movie user-user movie-genre movie-genre
    node_types = 3 # user movie genre
    device_num=0

    # res file
    user_artists_file_name = 'ml/user_movies.txt'
    train_file_name = 'ml/training.txt'
    negative_file_name = 'ml/negative.txt'
    test_negative_file_name = 'ml/test_negative.txt'

    # all_node_and ids
    all_nodes_and_ids_path = 'ml/all_nodes_and_ids.pickle'

    #pre_embedding and distance_matrix
    pre_embedding_path = 'ml/embedding_ml.vec.json'
    distance_matrix_path = 'ml/distance_matrix.npy'

    # path
    positive_path_file_name = 'ml/positive_path.txt'
    negative_path_file_name = 'ml/negative_path.txt'
    test_negative_path_file_name = 'ml/test_negative_path.txt'

    # distance parameters
    alpha = 0.1

    # save the results
    results_save_path = 'ml/results.txt'

    # set gpu
    torch.cuda.set_device(device_num)


    # set random seed
    manualSeed = random.randint(1, 10000)
    torch.manual_seed(manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(manualSeed)

    with open(all_nodes_and_ids_path, 'rb') as file:
          all_nodes_and_ids = pickle.load(file)
    node_size = len(all_nodes_and_ids)

    start_time = datetime.now()

    # load pre_embedding
    entity_pre_embedding = np.random.rand(node_size + 1, hidden_dim)  # embeddings for all nodes
    with open(pre_embedding_path,'rb') as file:
         embedding_dict = json.load(file)
    rel_embedding_list = embedding_dict['rel_embeddings']
    ent_embedding_list = embedding_dict['ent_embeddings']
    print(len(ent_embedding_list))
    for i in range(node_size):
        entity_pre_embedding[i] = np.array(ent_embedding_list[i])
    rel_pre_embedding = np.array(rel_embedding_list)
    print('load_pre_embedding finished, two embedding matrix is ',entity_pre_embedding.shape,rel_pre_embedding.shape)
    entity_pre_embedding = torch.FloatTensor(entity_pre_embedding)
    rel_pre_embedding = torch.FloatTensor(rel_pre_embedding)

    # load_distance_matrix

    distance_matrix = np.load(distance_matrix_path)


    # prepare  paths

    positive_path_dict,positive_user_items_dict= load_paths(open(positive_path_file_name,'r'))
    negative_path_dict,negative_user_items_dict = load_paths(open(negative_path_file_name,'r'))
    test_negative_path_dict,test_negative_items_dict = load_paths(open(test_negative_path_file_name,'r'))
    load_end_time = datetime.now()
    duration = load_end_time-start_time
    print('load all data finished, time is  :  ', duration)

    #

    #positive_user_items_dict = load_path_file_as_dict(train_file_name)
    #negative_user_items_dict = load_path_file_as_dict(negative_file_name)
    all_label_list = []
    for user_id in positive_user_items_dict:
        user_node = 'u' + user_id
        for item_id in positive_user_items_dict[user_id]:
            movie_node = 'i' + item_id
            all_label_list.append((user_node,movie_node))
        for item_id in negative_user_items_dict[user_id]:
            movie_node = 'i' + item_id
            all_label_list.append((user_node,movie_node))
    print(all_label_list)

    

    # get model instance

    model_relation = GGNN(hidden_dim,type_dim, node_size, edge_types, node_types, step, entity_pre_embedding,alpha)
    #model_relation = torch.load('ml/model_epoch_13.pt')
    print(model_relation)
    model_relation.double()  # important!! cannot delete

    if torch.cuda.is_available():
        model_relation = model_relation.cuda()
        print('model_to_cuda,single-gpu')

    # train and test the model

    model_relation.train()

    model_trained_relation = GGNNTrain_Relation(all_label_list,model_relation, iteration, learning_rate, batch_size,weight_decay,positive_path_dict,negative_path_dict,test_negative_path_dict,all_nodes_and_ids,distance_matrix,results_save_path,node_types)
    model_trained_relation.train_relation()

    print('model training and testing finished')



