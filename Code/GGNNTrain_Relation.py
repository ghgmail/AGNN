import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable

from data_preprocess import load_file_as_dict
from data_preprocess import load_test_negative_file_as_list
from data_preprocess import getHitRatio
from data_preprocess import getNDCG, generate_batch
import heapq
import random

torch.set_default_tensor_type('torch.DoubleTensor')



class GGNNTrain_Relation(object):

    def __init__(self, all_label_list,model, iteration, learning_rate, batch_size, weight_decay, positive_dict, negative_dict,
                 test_negative_dict, all_nodes_and_ids, distance_matrix, results_save_path,node_types):
        super(GGNNTrain_Relation, self).__init__()
        self.model = model
        self.iteration = iteration
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.positive_dict = positive_dict
        self.negative_dict = negative_dict
        self.test_negative_dict = test_negative_dict
        self.positive_label = list(self.positive_dict.keys())
        #self.negative_label = list(self.negative_dict.keys())
        self.all_nodes_and_ids = all_nodes_and_ids
        self.distance_matrix = distance_matrix
        self.batch_size = batch_size
        self.file = open(results_save_path, 'w')
        self.all_label_list = all_label_list
        self.node_types = node_types
    def get_node_type_index(self, node_name):
        # user artist
        if 'u' in node_name:
            type_index = 0
        if 'i' in node_name:
            type_index = 1
        if 'g' in node_name:
            type_index = 2

        return type_index

    def get_rel_type_index(self, node_name_x, node_name_y):
        '''
        if ('u' in node_name_x and 'a' in node_name_y) or ('a' in node_name_x and 'u' in node_name_y):
            type_index = 0
        if ('u' in node_name_x and 'u' in node_name_y):
            type_index = 1
        if ('a' in node_name_x and 't' in node_name_y) or ('t' in node_name_x and 'a' in node_name_y):
            type_index = 2
        '''
        return 1

    def sub_graph_generation(self,user_name,movie_name, path_list):
        inputs = []
        type_indices = []
        A_idx = 0
        node_name_and_A_idx = {}
        A_big = np.zeros((200, 200))
        edge_rel_type_one = {}

        if len(path_list) == 0:
            user_id = self.all_nodes_and_ids[user_name]
            movie_id = self.all_nodes_and_ids[movie_name]
            inputs.append(user_id)
            inputs.append(movie_id)
            type_index_user = self.get_node_type_index(user_name)
            type_index_movie = self.get_node_type_index(movie_name)
            type_indices.append(type_index_user)
            type_indices.append(type_index_movie)
            node_name_and_A_idx.update({user_name: 0})
            node_name_and_A_idx.update({movie_name: 1})

        else:
            for path in path_list:
                for i in range(len(path) - 1):
                    node_x_name = path[i]
                    node_y_name = path[i + 1]
                    node_x_id = int(self.all_nodes_and_ids[node_x_name])
                    node_y_id = int(self.all_nodes_and_ids[node_y_name])
                    if node_x_id not in inputs:
                        inputs.append(node_x_id)
                        type_index_x = self.get_node_type_index(node_x_name)
                        type_indices.append(type_index_x)
                        node_name_and_A_idx.update({node_x_name: A_idx})
                        A_idx += 1
                    if node_y_id not in inputs:
                        inputs.append(node_y_id)
                        type_index_y = self.get_node_type_index(node_y_name)
                        type_indices.append(type_index_y)
                        node_name_and_A_idx.update({node_y_name: A_idx})
                        A_idx += 1
                    A_idx_x = int(node_name_and_A_idx[node_x_name])
                    A_idx_y = int(node_name_and_A_idx[node_y_name])
                    A_big[A_idx_x, A_idx_y] = 1.0000
                    A_big[A_idx_y, A_idx_x] = 1.0000
                    rel_type_index = self.get_rel_type_index(node_x_name, node_y_name)
                    edge_rel_type_one.update({(A_idx_x, A_idx_y): rel_type_index})
        node_size = len(inputs)
        A_one = A_big[:node_size, :node_size]
        return inputs, type_indices, A_one, edge_rel_type_one, node_name_and_A_idx

    def train_relation(self):
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # load_test_data

        testRatings,testNegatives = load_test_negative_file_as_list('ml/test_negative.txt')

        #########################################train#################################################

        node_size = len(self.all_nodes_and_ids)
        length = len(self.all_label_list)

        # set batch_size
        slices = generate_batch(length, self.batch_size)

        for epoch in range(15):
	    print(epoch)
            running_loss = 0.0
            num = 0
            for slice in slices:
                inputs_slice = []
                type_indices_slice = []
                movie_idx_slice = []
                user_idx_slice = []
                label = []
                A_slice = []
                edge_rel_type_slice = []
                for j in slice:
                    pair = self.all_label_list[j]
                    user_name = pair[0]
                    movie_name = pair[1]
                    key = (user_name, movie_name)

                    if key in self.positive_label:
                        paths_list = self.positive_dict[key]
                        label_one = np.array([1])
                    else:
                        paths_list = self.negative_dict[key]
                        label_one = np.array([0])

                    inputs, type_indices, A_one, edge_rel_type, node_name_and_A_idx = self.sub_graph_generation(user_name,movie_name,paths_list)
                    user_idx = node_name_and_A_idx[user_name]
                    movie_idx = node_name_and_A_idx[movie_name]
                    inputs = np.array(inputs)
                    user_idx = np.array([user_idx])
                    movie_idx = np.array([movie_idx])
                    type_indices = np.array(type_indices)

                    ## data for a slice
                    inputs_slice.append(inputs)  # need pad
                    type_indices_slice.append(type_indices)  # need pad
                    user_idx_slice.append(user_idx)
                    movie_idx_slice.append(movie_idx)
                    label.append(label_one)
                    A_slice.append(A_one)
                    edge_rel_type_slice.append(edge_rel_type)

                # pad inputs
                padding_index_node = node_size
                padding_index_type = self.node_types
                inputs_length = [len(inputs) for inputs in inputs_slice]
                max_len = max(inputs_length)
                batch_size = len(inputs_slice)
                padded_inputs = np.ones((batch_size, max_len)) * padding_index_node
                padded_type_indices = np.ones((batch_size, max_len)) * padding_index_type

                padded_A = []
                for i, x_len in enumerate(inputs_length):
                    sequence = inputs_slice[i]
                    type = type_indices_slice[i]
                    padded_inputs[i, 0:x_len] = sequence[:x_len]
                    padded_type_indices[i, 0:x_len] = type[:x_len]
                    padded_A_one = np.zeros((max_len, max_len))
                    Real_A_one = A_slice[i]
                    padded_A_one[0:x_len, 0:x_len] = Real_A_one[0:x_len, 0:x_len]
                    padded_A.append(torch.LongTensor(padded_A_one))

                inputs = Variable(torch.LongTensor(padded_inputs))
                user_idx = Variable(torch.LongTensor(user_idx_slice))
                movie_idx = Variable(torch.LongTensor(movie_idx_slice))
                type_indices = Variable(torch.LongTensor(padded_type_indices))
                label = Variable(torch.LongTensor(label))
                A = Variable(torch.stack(padded_A))

                if torch.cuda.is_available():
                    A = A.cuda()
                    inputs = inputs.cuda()
                    type_indices = type_indices.cuda()
                    label = label.cuda()
                    user_idx = user_idx.cuda()
                    movie_idx = movie_idx.cuda()

                out = self.model(inputs, type_indices, A, edge_rel_type_slice, user_idx, movie_idx, True)
                out = out.squeeze()

                loss = criterion(out, label.double())

                running_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                num += 1
                print('MovieLens', num, 'loss: ', loss.item(), 'running loss', running_loss)

            print('epoch[' + str(epoch) + ']: loss is ' + str(running_loss))

            # save model
            path = 'ml/model_none_epoch_' + str(epoch) + '.pt'
            #torch.save(self.model, path)

            ######################################################evalution every verbose epoch#######################################################
            verbose = 1
            if epoch % verbose == 0:
                (hits_1, ndcgs_1), (hits_2, ndcgs_2), (hits_3, ndcgs_3), (hits_4, ndcgs_4), (hits_5, ndcgs_5), (
                    hits_6, ndcgs_6), (hits_7, ndcgs_7), (hits_8, ndcgs_8), (hits_9, ndcgs_9), (hits_10, ndcgs_10), \
                (hits_11, ndcgs_11), (hits_12, ndcgs_12), (hits_13, ndcgs_13), (hits_14, ndcgs_14), (
                    hits_15, ndcgs_15) = self.evaluate_model(self.model, testRatings, testNegatives)
                hr_1, ndcg_1 = np.array(hits_1).mean(), np.array(ndcgs_1).mean()
                hr_2, ndcg_2 = np.array(hits_2).mean(), np.array(ndcgs_2).mean()
                hr_3, ndcg_3 = np.array(hits_3).mean(), np.array(ndcgs_3).mean()
                hr_4, ndcg_4 = np.array(hits_4).mean(), np.array(ndcgs_4).mean()

                hr_5, ndcg_5 = np.array(hits_5).mean(), np.array(ndcgs_5).mean()
                hr_6, ndcg_6 = np.array(hits_6).mean(), np.array(ndcgs_6).mean()
                hr_7, ndcg_7 = np.array(hits_7).mean(), np.array(ndcgs_7).mean()
                hr_8, ndcg_8 = np.array(hits_8).mean(), np.array(ndcgs_8).mean()
                hr_9, ndcg_9 = np.array(hits_9).mean(), np.array(ndcgs_9).mean()
                hr_10, ndcg_10 = np.array(hits_10).mean(), np.array(ndcgs_10).mean()
                hr_11, ndcg_11 = np.array(hits_11).mean(), np.array(ndcgs_11).mean()
                hr_12, ndcg_12 = np.array(hits_12).mean(), np.array(ndcgs_12).mean()
                hr_13, ndcg_13 = np.array(hits_13).mean(), np.array(ndcgs_13).mean()
                hr_14, ndcg_14 = np.array(hits_14).mean(), np.array(ndcgs_14).mean()
                hr_15, ndcg_15 = np.array(hits_15).mean(), np.array(ndcgs_15).mean()
                print('Iteration %d : HR_1 = %.4f, NDCG_1 = %.4f '
                      % (epoch, hr_1, ndcg_1))
                print('Iteration %d : HR_1 = %.4f, NDCG_1 = %.4f '
                      % (epoch, hr_2, ndcg_2))
                print('Iteration %d : HR_1 = %.4f, NDCG_1 = %.4f '
                      % (epoch, hr_3, ndcg_3))
                print('Iteration %d : HR_1 = %.4f, NDCG_1 = %.4f '
                      % (epoch, hr_4, ndcg_4))
                print('Iteration %d : HR_1 = %.4f, NDCG_1 = %.4f '
                      % (epoch, hr_5, ndcg_5))
                print('Iteration %d : HR_1 = %.4f, NDCG_1 = %.4f '
                      % (epoch, hr_6, ndcg_6))
                print('Iteration %d : HR_1 = %.4f, NDCG_1 = %.4f '
                      % (epoch, hr_7, ndcg_7))
                print('Iteration %d : HR_1 = %.4f, NDCG_1 = %.4f '
                      % (epoch, hr_8, ndcg_8))
                print('Iteration %d : HR_5 = %.4f, NDCG_5 = %.4f '
                      % (epoch, hr_9, ndcg_9))
                print('Iteration %d : HR_10 = %.4f, NDCG_10 = %.4f '
                      % (epoch, hr_10, ndcg_10))
                print('Iteration %d : HR_1 = %.4f, NDCG_1 = %.4f '
                      % (epoch, hr_11, ndcg_11))
                print('Iteration %d : HR_1 = %.4f, NDCG_1 = %.4f '
                      % (epoch, hr_12, ndcg_12))
                print('Iteration %d : HR_1 = %.4f, NDCG_1 = %.4f '
                      % (epoch, hr_13, ndcg_13))
                print('Iteration %d : HR_1 = %.4f, NDCG_1 = %.4f '
                      % (epoch, hr_14, ndcg_14))
                print('Iteration %d : HR_15 = %.4f, NDCG_15 = %.4f '
                      % (epoch, hr_15, ndcg_15))

                line_1 = 'epoch:  ' + str(epoch) + '  hr_1: ' + str(hr_1) + '  ndcg_1: ' + str(ndcg_1) + '\n'
                line_2 = 'epoch:  ' + str(epoch) + '  hr_2: ' + str(hr_2) + '  ndcg_2: ' + str(ndcg_2) + '\n'
                line_3 = 'epoch:  ' + str(epoch) + '  hr_3: ' + str(hr_3) + '  ndcg_3: ' + str(ndcg_3) + '\n'
                line_4 = 'epoch:  ' + str(epoch) + '  hr_4: ' + str(hr_4) + '  ndcg_4: ' + str(ndcg_4) + '\n'

                line_5 = 'epoch:  ' + str(epoch) + '  hr_5: ' + str(hr_5) + '  ndcg_5: ' + str(ndcg_5) + '\n'
                line_6 = 'epoch:  ' + str(epoch) + '  hr_6: ' + str(hr_6) + '  ndcg_6: ' + str(ndcg_6) + '\n'
                line_7 = 'epoch:  ' + str(epoch) + '  hr_7: ' + str(hr_7) + '  ndcg_7: ' + str(ndcg_7) + '\n'
                line_8 = 'epoch:  ' + str(epoch) + '  hr_8: ' + str(hr_8) + '  ndcg_8: ' + str(ndcg_8) + '\n'
                line_9 = 'epoch:  ' + str(epoch) + '  hr_9: ' + str(hr_9) + '  ndcg_9: ' + str(ndcg_9) + '\n'
                line_10 = 'epoch:  ' + str(epoch) + '  hr_10: ' + str(hr_10) + '  ndcg_10: ' + str(ndcg_10) + '\n'

                line_11 = 'epoch:  ' + str(epoch) + '  hr_11: ' + str(hr_11) + '  ndcg_11: ' + str(ndcg_11) + '\n'
                line_12 = 'epoch:  ' + str(epoch) + '  hr_12: ' + str(hr_12) + '  ndcg_12: ' + str(ndcg_12) + '\n'
                line_13 = 'epoch:  ' + str(epoch) + '  hr_13: ' + str(hr_13) + '  ndcg_13: ' + str(ndcg_13) + '\n'
                line_14 = 'epoch:  ' + str(epoch) + '  hr_14: ' + str(hr_14) + '  ndcg_14: ' + str(ndcg_14) + '\n'
                line_15 = 'epoch:  ' + str(epoch) + '  hr_15: ' + str(hr_15) + '  ndcg_15: ' + str(ndcg_15) + '\n'
                loss_str = 'epoch:  ' + str(epoch) + 'loss is ' + str(running_loss) + '\n' + '\n'

                results_path = 'ml/results_none_epoch_' + str(epoch) + '.txt'
                with open(results_path,'w') as file:
                        file.write(line_1)
                        file.write(line_2)
                        file.write(line_3)
                        file.write(line_4)
                        file.write(line_5)
                        file.write(line_6)
                        file.write(line_7)
                        file.write(line_8)
                        file.write(line_9)

                        file.write(line_10)
                        file.write(line_11)
                        file.write(line_12)
                        file.write(line_13)
                        file.write(line_14)
                        file.write(line_15)
                        file.write(loss_str)

    def evaluate_model(self, model, testRatings, testNegatives):
        """
        Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
        Return: score of each test rating.
        """

        global _model
        global _testRatings
        global _testNegatives

        _testRatings = testRatings
        _testNegatives = testNegatives

        hits_1, hits_2, hits_3, hits_4, hits_5, hits_6, hits_7, hits_8, hits_9, hits_10, hits_11, hits_12, hits_13, hits_14, hits_15 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        ndcgs_1, ndcgs_2, ndcgs_3, ndcgs_4, ndcgs_5, ndcgs_6, ndcgs_7, ndcgs_8, ndcgs_9, ndcgs_10, ndcgs_11, ndcgs_12, ndcgs_13, ndcgs_14, ndcgs_15 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        # Single thread

        for idx in range(len(_testRatings)):
            print('test num', idx)

            (hr_1, ndcg_1), (hr_2, ndcg_2), (hr_3, ndcg_3), (hr_4, ndcg_4), (hr_5, ndcg_5), (hr_6, ndcg_6), \
            (hr_7, ndcg_7), (hr_8, ndcg_8), (hr_9, ndcg_9), (hr_10, ndcg_10), (hr_11, ndcg_11), (hr_12, ndcg_12), \
            (hr_13, ndcg_13), (hr_14, ndcg_14), (hr_15, ndcg_15) = self.eval_one_rating(idx)
            hits_1.append(hr_1)
            ndcgs_1.append(ndcg_1)
            hits_2.append(hr_2)
            ndcgs_2.append(ndcg_2)
            hits_3.append(hr_3)
            ndcgs_3.append(ndcg_3)
            hits_4.append(hr_4)
            ndcgs_4.append(ndcg_4)
            hits_5.append(hr_5)
            ndcgs_5.append(ndcg_5)
            hits_6.append(hr_6)
            ndcgs_6.append(ndcg_6)
            hits_7.append(hr_7)
            ndcgs_7.append(ndcg_7)
            hits_8.append(hr_8)
            ndcgs_8.append(ndcg_8)
            hits_9.append(hr_9)
            ndcgs_9.append(ndcg_9)
            hits_10.append(hr_10)
            ndcgs_10.append(ndcg_10)
            hits_11.append(hr_11)
            ndcgs_11.append(ndcg_11)
            hits_12.append(hr_12)
            ndcgs_12.append(ndcg_12)
            hits_13.append(hr_13)
            ndcgs_13.append(ndcg_13)
            hits_14.append(hr_14)
            ndcgs_14.append(ndcg_14)
            hits_15.append(hr_15)
            ndcgs_15.append(ndcg_15)
        return (hits_1, ndcgs_1), (hits_2, ndcgs_2), (hits_3, ndcgs_3), (hits_4, ndcgs_4), (hits_5, ndcgs_5), \
               (hits_6, ndcgs_6), (hits_7, ndcgs_7), (hits_8, ndcgs_8), (hits_9, ndcgs_9), (hits_10, ndcgs_10), \
               (hits_11, ndcgs_11), (hits_12, ndcgs_12), (hits_13, ndcgs_13), (hits_14, ndcgs_14), (hits_15, ndcgs_15)

    def eval_one_rating(self, idx):
        # di idx interaction in test

        rating = _testRatings[idx]
        items = _testNegatives[idx]
        user = rating[0]
        gtItem = rating[1]
        items.append(gtItem)

        # Get prediction scores
        map_item_score = {}
        predictions = []
        for item in items:
            score_for_one_pair = self.Compute_score_for_one_pair(user, item)
            predictions.append(score_for_one_pair)
        for i in range(len(items)):
            item = items[i]
            map_item_score[item] = predictions[i]
        items.pop()

        # Evaluate top rank list
        ranklist_1 = heapq.nlargest(1, map_item_score, key=map_item_score.get)
        ranklist_2 = heapq.nlargest(2, map_item_score, key=map_item_score.get)
        ranklist_3 = heapq.nlargest(3, map_item_score, key=map_item_score.get)
        ranklist_4 = heapq.nlargest(4, map_item_score, key=map_item_score.get)
        ranklist_5 = heapq.nlargest(5, map_item_score, key=map_item_score.get)
        ranklist_6 = heapq.nlargest(6, map_item_score, key=map_item_score.get)
        ranklist_7 = heapq.nlargest(7, map_item_score, key=map_item_score.get)
        ranklist_8 = heapq.nlargest(8, map_item_score, key=map_item_score.get)
        ranklist_9 = heapq.nlargest(9, map_item_score, key=map_item_score.get)
        ranklist_10 = heapq.nlargest(10, map_item_score, key=map_item_score.get)
        ranklist_11 = heapq.nlargest(11, map_item_score, key=map_item_score.get)
        ranklist_12 = heapq.nlargest(12, map_item_score, key=map_item_score.get)
        ranklist_13 = heapq.nlargest(13, map_item_score, key=map_item_score.get)
        ranklist_14 = heapq.nlargest(14, map_item_score, key=map_item_score.get)
        ranklist_15 = heapq.nlargest(15, map_item_score, key=map_item_score.get)
        hr_1 = getHitRatio(ranklist_1, gtItem)
        ndcg_1 = getNDCG(ranklist_1, gtItem)
        hr_2 = getHitRatio(ranklist_2, gtItem)
        ndcg_2 = getNDCG(ranklist_2, gtItem)
        hr_3 = getHitRatio(ranklist_3, gtItem)
        ndcg_3 = getNDCG(ranklist_3, gtItem)
        hr_4 = getHitRatio(ranklist_4, gtItem)
        ndcg_4 = getNDCG(ranklist_4, gtItem)
        hr_5 = getHitRatio(ranklist_5, gtItem)
        ndcg_5 = getNDCG(ranklist_5, gtItem)
        hr_6 = getHitRatio(ranklist_6, gtItem)
        ndcg_6 = getNDCG(ranklist_6, gtItem)
        hr_7 = getHitRatio(ranklist_7, gtItem)
        ndcg_7 = getNDCG(ranklist_7, gtItem)
        hr_8 = getHitRatio(ranklist_8, gtItem)
        ndcg_8 = getNDCG(ranklist_8, gtItem)
        hr_9 = getHitRatio(ranklist_9, gtItem)
        ndcg_9 = getNDCG(ranklist_9, gtItem)
        hr_10 = getHitRatio(ranklist_10, gtItem)
        ndcg_10 = getNDCG(ranklist_10, gtItem)
        hr_11 = getHitRatio(ranklist_11, gtItem)
        ndcg_11 = getNDCG(ranklist_11, gtItem)
        hr_12 = getHitRatio(ranklist_12, gtItem)
        ndcg_12 = getNDCG(ranklist_12, gtItem)
        hr_13 = getHitRatio(ranklist_13, gtItem)
        ndcg_13 = getNDCG(ranklist_13, gtItem)
        hr_14 = getHitRatio(ranklist_14, gtItem)
        ndcg_14 = getNDCG(ranklist_14, gtItem)
        hr_15 = getHitRatio(ranklist_15, gtItem)
        ndcg_15 = getNDCG(ranklist_15, gtItem)
        return (hr_1, ndcg_1), (hr_2, ndcg_2), (hr_3, ndcg_3), (hr_4, ndcg_4), (hr_5, ndcg_5), \
               (hr_6, ndcg_6), (hr_7, ndcg_7), (hr_8, ndcg_8), (hr_9, ndcg_9), (hr_10, ndcg_10), (hr_11, ndcg_11), \
               (hr_12, ndcg_12), (hr_13, ndcg_13), (hr_14, ndcg_14), (hr_15, ndcg_15)

    def Compute_score_for_one_pair(self, user, item):
        user_name = 'u' + user
        movie_name = 'i' + item

        key = (user_name, movie_name)

        paths_list = self.test_negative_dict[key]
        inputs, type_indices, A, edge_rel_type, node_name_and_A_idx = self.sub_graph_generation(user_name,movie_name,paths_list)

        user_idx = node_name_and_A_idx[user_name]
        movie_idx = node_name_and_A_idx[movie_name]

        inputs = torch.LongTensor(np.array(inputs))
        user_idx = torch.LongTensor([user_idx])
        movie_idx = torch.LongTensor([movie_idx])
        type_indices = torch.LongTensor(np.array(type_indices))
        A = torch.LongTensor(A)

        if torch.cuda.is_available():
            A = A.cuda()
            inputs = inputs.cuda()
            type_indices = type_indices.cuda()
            user_idx = user_idx.cuda()
            movie_idx = movie_idx.cuda()

        score = self.model(inputs, type_indices, A, edge_rel_type, user_idx, movie_idx, False)
        score = score.data.cpu().numpy()
        score = score[0][0]

        return score
