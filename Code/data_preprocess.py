
import networkx as nx
import pickle
import json
import numpy as np





import math
import heapq  # for retrieval topK
import multiprocessing
import numpy as np
from time import time


# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None

def load_file_as_dict(filename):
    dict = {}
    with open(filename, "r") as f:
        for line in f:
            arr = line.replace('\n','').replace('\r','').split("\t")
            userid, itemid = arr[0], arr[1]
            if userid not in dict:
                dict.update({userid:[itemid]})
            else:
                dict[userid].append(itemid)
    return dict

# [ [user,item],[user,iem]]
def load_test_file_as_list(filename):
    ratingList = []
    with open(filename, "r") as f:
        for line in f:
            arr = line.replace('\n','').replace('\r','').split("\t")
            userid, itemid = arr[0], arr[1]
            ratingList.append([userid, itemid])
    return ratingList

# [ [i1,i2,i3],[i3,i5,i5]]
def load_test_negative_file_as_list(filename):
    negativeList = []
    test_Ratings = []
    with open(filename, "r") as f:
       for line in f:
            arr = line.replace('\n','').replace('\r','').split("\t")
            userid, itemid = arr[0], arr[1]
            test_Ratings.append([userid, itemid])
            negatives = []
            for x in arr[2:]:
                negatives.append(x)
            negativeList.append(negatives)
    print('num of test_ratings and negatives should be same',len(test_Ratings), len(negativeList))
    return test_Ratings,negativeList


def getP(ranklist, gtItems):
    p = 0
    for item in ranklist:
        if item in gtItems:
            p += 1
    return p * 1.0 / len(ranklist)

def getR(ranklist, gtItems):
    r = 0
    for item in ranklist:
        if item in gtItems:
            r += 1
    return r * 1.0 / len(gtItems)

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item in gtItem:
            return 1
    return 0

def getDCG(ranklist, gtItems):
    dcg = 0.0
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item in gtItems:
            dcg += 1.0 / math.log(i + 2)
    return  dcg

def getIDCG(ranklist, gtItems):
    idcg = 0.0
    i = 0
    for item in ranklist:
        if item in gtItems:
            idcg += 1.0 / math.log(i + 2)
            i += 1
    return idcg

def getNDCG(ranklist, gtItems):
    dcg = getDCG(ranklist, gtItems)
    idcg = getIDCG(ranklist, gtItems)
    if idcg == 0:
        return 0
    return dcg / idcg



def generate_batch(length,batch_size,shuffle = True):
    n_batch = int(length / batch_size)
    if length % batch_size != 0:
        n_batch += 1
    slices = np.split(np.arange(n_batch * batch_size), n_batch)
    slices[-1] = slices[-1][:(length - batch_size * (n_batch - 1))]
    return slices





def load_data_to_list(file_name):
    # user-user user-artist
    data = []
    with open(file_name, 'r') as file:
        for line in file:
            lines = line.replace('\n', '').replace('\r', '').split('\t')
            user = lines[0]
            artist = lines[1]
            data.append((user, artist))
    return data



# add all user-artists
def add_all_artists_node_into_graph(g,all_artist_list):
    for pair in all_artist_list:
        user = pair[0]
        artist = pair[1]
        user_node = 'u' + user
        artist_node = 'i' + artist
        g.add_node(user_node)
        g.add_node(artist_node)
        g.add_edge(user_node, artist_node, attr='usermovie')
    return g

# add train interaction into graph
def add_user_artist_interaction_into_graph(g,user_artist_list):

    for pair in user_artist_list:
        user = pair[0]
        artist = pair[1]
        user_node = 'u' + user
        artist_node = 'i' + artist
        g.add_node(user_node)
        g.add_node(artist_node)
        g.add_edge(user_node, artist_node, attr= 'usermovie')
    return g

# add user-user
def add_user_friend_into_graph(g,user_friend_list):
    for pair in user_friend_list:
        user_x = pair[0]
        user_y = pair[1]
        user_x_node = 'u' + user_x
        user_y_node = 'u' + user_y
        g.add_node(user_x_node)
        g.add_node(user_y_node)
        g.add_edge(user_x_node, user_y_node, attr='useruser')
    return g

# add movie-genre
def add_artist_tag_into_graph(g,artist_tag_list):
    for pair in artist_tag_list:
        artist = pair[0]
        tag = pair[1]
        artist_node = 'i' + artist
        tag_node = 'g' + tag
        g.add_node(artist_node)
        g.add_node(tag_node)
        g.add_edge(artist_node,tag_node, attr='moviegenre')
    return g

# add movie-movie
def add_item_item_into_graph(g,item_item_list):
    for pair in item_item_list:
        item_x = pair[0]
        item_y = pair[1]
        item_x_node = 'i' + item_x
        item_y_node = 'i' + item_y
        g.add_node(item_x_node)
        g.add_node(item_y_node)
        g.add_edge(item_x_node,item_y_node, attr='moviemovie')
    return g


def distance_matrix_dump(pre_embedding_path, distance_matrix_path):
    with open(pre_embedding_path, 'rb') as file:
        embedding_dict = json.load(file)

    rel_embedding_list = embedding_dict['rel_embeddings']
    ent_embedding_list = embedding_dict['ent_embeddings']
    print('relation_size,entity_size', len(rel_embedding_list), len(ent_embedding_list))

    node_size = len(ent_embedding_list)
    distance_matrix = np.zeros((node_size, node_size))


    for i in range(node_size):
        for j in range(i,node_size):
            x = np.array(ent_embedding_list[i])
            y = np.array(ent_embedding_list[j])
            distance_matrix[i][j] = np.sqrt(np.sum(np.square(x - y)))
            distance_matrix[j][i] = np.sqrt(np.sum(np.square(x - y)))
        print(i)
    np.save(distance_matrix_path,distance_matrix)


def graph_generation(user_artists_file_name,train_file_name,user_friend_file_name,artist_tag_file_name,item_item_file_name):

    all_artist_list = load_data_to_list(user_artists_file_name)
    user_artist_list = load_data_to_list(train_file_name)
    user_friend_list = load_data_to_list(user_friend_file_name)
    artist_tag_list = load_data_to_list(artist_tag_file_name)
    item_item_list = load_data_to_list(item_item_file_name)
    g = nx.Graph()
    g = add_all_artists_node_into_graph(g, all_artist_list)
    g = add_user_artist_interaction_into_graph(g, user_artist_list)
    g = add_user_friend_into_graph(g, user_friend_list)
    g = add_artist_tag_into_graph(g, artist_tag_list)
    g = add_item_item_into_graph(g,item_item_list)
    return g




if __name__ == '__main__':
    user_artists_file_name = 'ml/user_movies.txt'
    train_file_name = 'ml/training.txt'
    user_friend_file_name = 'ml/user_friends.txt'
    artist_tag_file_name = 'ml/movie_genre.txt'
    item_item_file_name = 'ml/movie_movie.txt'


    all_nodes_and_ids_path = 'ml/all_nodes_and_ids.pickle'

    # parameters for distance matrix
    pre_embedding_path = 'ml/embedding_ml.vec.json'
    distance_matrix_path = 'ml/distance_matrix.npy'


    # generate the graph
    g = graph_generation(user_artists_file_name, train_file_name, user_friend_file_name, artist_tag_file_name, item_item_file_name)
    print('graph finished, total nodes--', len(g.nodes()), 'total edges', len(g.edges()))



    # all nodes into ids
    all_nodes_and_ids = {}
    node_idx = 0
    for node in g.nodes():
        all_nodes_and_ids.update({node: node_idx})
        print(node, node_idx)
        node_idx += 1
    print(len(all_nodes_and_ids))
    with open(all_nodes_and_ids_path,'wb') as file:
         pickle.dump(all_nodes_and_ids,file,0)

    #distance matrix generation
    distance_matrix_dump(pre_embedding_path, distance_matrix_path)


