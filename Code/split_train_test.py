# This is used to split the user-movie interaction data into traning and test according to the timestamp

import argparse
import operator


def round_int(rating_num, ratio):
    '''
    get the size of training data for each user

    Inputs:
        @rating_num: the total number of ratings for a specific user
        @ration: the percentage for training data

    Output:
        @train_size: the size of training data
    '''

    train_size = int(round(rating_num * ratio, 0))

    return train_size


def load_data(fr_rating):

    rating_data = {}

    for line in fr_rating:
        lines = line.split('\t')
        user = int(lines[0])
        item = int(lines[1])
        if user in rating_data:
            rating_data[user].append(item)
        else:
            rating_data.update({user: [item]})

    return rating_data


def split_rating_into_train_test(rating_data, fw_train, fw_test, ratio):

    for user in rating_data:
        item_list = rating_data[user]
        rating_num = len(rating_data[user])
        #train_size = round_int(rating_num, ratio)
        train_size = rating_num -1

        if rating_num == 1:
            train_size = 1
        print(train_size)

        flag = 0
        for item in item_list:
            if flag < train_size:
                user = str(user)
                line = user + '\t' + str(item)  + '\n'
                fw_train.write(line)
                flag = flag + 1
            else:
                user = str(user)
                line = user + '\t' + str(item)  + '\n'
                fw_test.write(line)


if __name__ == '__main__':

    rating_file = 'ml/user_movies.txt'
    train_file = 'ml/training.txt'
    test_file = 'ml/test.txt'
    ratio= 0.8

    fr_rating = open(rating_file, 'r')
    fw_train = open(train_file, 'w')
    fw_test = open(test_file, 'w')

    rating_data = load_data(fr_rating)
    split_rating_into_train_test(rating_data, fw_train, fw_test, ratio)

    fr_rating.close()
    fw_train.close()
    fw_test.close()