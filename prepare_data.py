
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class Data:
    user_size = 0
    train_sample_size = 0
    test_sample_size = 0
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    relation = []
    c_profile = []
    o_profile = []

    def __init__(self, sentiment_file, relation_file, c_profile_file, o_profile_file, train_ratio, random_state=0):
        sentiment = pd.read_csv(sentiment_file, header=0)
        self.sentiment = sentiment
        self.relation = pd.read_csv(relation_file, header=0)
        self.c_profile = pd.read_csv(c_profile_file, header=0)
        self.o_profile = pd.read_csv(o_profile_file, header=0)
        # number of users
        self.user_size = max(sentiment['holder_id'].max(), sentiment['target_id'].max()) + 1  #12814

        # split train and test data
        # x_train: sentiment 共 65420 筆資料 * 0.8 (訓練資料比例) = 52336 筆
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            sentiment[['holder_id', 'target_id']],
            sentiment['label'],
            test_size=0.2,
            random_state=random_state)
        #原始random_state = 0

        # balance test data
        self.balance_test_data()

        # draw training samples 
        self.x_train = self.x_train[0:int(self.x_train.shape[0] * train_ratio)]
        self.y_train = self.y_train[0:int(self.y_train.shape[0] * train_ratio)]

        # number of edges in train and test data
        self.train_sample_size = self.x_train.shape[0]    #52336
        self.test_sample_size = self.x_test.shape[0]      #1220

        # re-index
        self.x_train.index = range(self.train_sample_size)
        self.x_test.index = range(self.test_sample_size)
        self.y_train.index = range(self.train_sample_size)
        self.y_test.index = range(self.test_sample_size)

        # convert label from pandas.Series to np.array
        self.y_train = np.array(self.y_train, dtype=np.int8)
        self.y_test = np.array(self.y_test, dtype=np.int8)
    def get_original_x_data(self):
        return self.x_train, self.x_test

    def balance_test_data(self):
        test_neg_size = self.y_test[self.y_test < 0].shape[0]    # sentiment為負值的個數

        test_pos_index = self.y_test[self.y_test > 0].index[0:test_neg_size]    #選出與負值同樣多的正值的"索引值"
        test_neg_index = self.y_test[self.y_test < 0].index    #負值的"索引值"
        self.x_test = self.x_test.loc[np.concatenate([test_pos_index, test_neg_index])]
        self.y_test = self.y_test.loc[np.concatenate([test_pos_index, test_neg_index])]
        # print(self.y_test)   #共1220筆測試資料

    def get_sentiment_data(self):
        # construct sentiment adjacency matrix from train data
        adj_matrix_s = np.zeros([self.user_size, self.user_size * 2], dtype=np.float)
        for i, row in self.x_train.iterrows():
            holder = row['holder_id']
            target = row['target_id']
            label = self.y_train[i]
            adj_matrix_s[holder, target] = label
            # adj_matrix_s[holder, target] = 1
            adj_matrix_s[target, holder + self.user_size] = label

        for i in np.arange(self.user_size): 
            adj_matrix_s[i, i] = 1
            adj_matrix_s[i, i + self.user_size] = 1
        # adj_matrix_s.shape:(12814, 25628)
        # self.x_train.shape:(52336, 2)

        X1_s_train = adj_matrix_s[self.x_train['holder_id']]    #(52336, 25628)   
        X2_s_train = adj_matrix_s[self.x_train['target_id']]    #(52336, 25628)   
        X1_s_test = adj_matrix_s[self.x_test['holder_id']]      #(1220, 25628)
        X2_s_test = adj_matrix_s[self.x_test['target_id']]      #(1220, 25628)

        return X1_s_train, X2_s_train, self.y_train, X1_s_test, X2_s_test, self.y_test, adj_matrix_s

    def get_complete_adjacency(self):
        # construct sentiment adjacency matrix from train data
        adj_matrix_s = np.zeros([self.user_size, self.user_size * 2], dtype=np.float)
        for i, row in self.sentiment.iterrows():
            holder = row['holder_id']
            target = row['target_id']
            label = row['label']
            adj_matrix_s[holder, target] = label
            adj_matrix_s[target, holder + self.user_size] = label

        for i in np.arange(self.user_size): 
            adj_matrix_s[i, i] = 1
            adj_matrix_s[i, i + self.user_size] = 1        

        return adj_matrix_s

    def get_sentiment_pair_data(self):
        # construct sentiment adjacency matrix from train data
        all_user_true_sentiment = np.zeros([12814, 1723], dtype=np.float)
        for i, row in self.sentiment.iterrows():
            holder = row['holder_id']
            target = row['target_id']
            label = row['label']
            all_user_true_sentiment[holder, target] = label

        return all_user_true_sentiment

    def get_comment_data(self):
        # construct comment adjacency matrix from train data
        adj_matrix_s = np.zeros([self.user_size, self.user_size * 2], dtype=np.float)
        for i, row in self.x_train.iterrows():
            holder = row['holder_id']
            target = row['target_id']
            label = self.y_train[i]
            adj_matrix_s[holder, target] = 1
            # adj_matrix_s[holder, target] = label
            adj_matrix_s[target, holder + self.user_size] = 1

        for i in np.arange(self.user_size): 
            adj_matrix_s[i, i] = 1
            adj_matrix_s[i, i + self.user_size] = 1
        # adj_matrix_s.shape:(12814, 25628)
        # self.x_train.shape:(52336, 2)

        X1_s_train = adj_matrix_s[self.x_train['holder_id']]    #(52336, 25628)   
        X2_s_train = adj_matrix_s[self.x_train['target_id']]    #(52336, 25628)   
        X1_s_test = adj_matrix_s[self.x_test['holder_id']]      #(1220, 25628)
        X2_s_test = adj_matrix_s[self.x_test['target_id']]      #(1220, 25628)

        return X1_s_train, X2_s_train, self.y_train, X1_s_test, X2_s_test, self.y_test, adj_matrix_s


    def get_relation_data(self):
        # construct social relation adjacency matrix from train data
        adj_matrix_r = np.zeros([self.user_size, self.user_size * 2], dtype=np.float)
        for i in np.arange(self.relation.shape[0]):
            follower = self.relation.loc[i, 'follower_id']
            followee = self.relation.loc[i, 'followee_id']
            adj_matrix_r[follower, followee] = 1
            adj_matrix_r[followee, follower + self.user_size] = 1
        for i in np.arange(self.user_size):
            adj_matrix_r[i, i] = 1
            adj_matrix_r[i, i + self.user_size] = 1
        X1_r_train = adj_matrix_r[self.x_train['holder_id']]
        X2_r_train = adj_matrix_r[self.x_train['target_id']]
        X1_r_test = adj_matrix_r[self.x_test['holder_id']]
        X2_r_test = adj_matrix_r[self.x_test['target_id']]
        return X1_r_train, X2_r_train, X1_r_test, X2_r_test, adj_matrix_r

    def get_profile_data(self):
        c_column = [column for column in self.c_profile.columns.tolist()]  # 好像可以用 c_profile.columns.tolist()就好
        # sum:13643 shape:(12814, 108)
        c_profile_one_hot = np.concatenate([self.get_one_hot_encoding('c', column) for column in c_column[2:]], axis=1)
        
        o_column = [column for column in self.o_profile.columns.tolist()] 
        # sum:18968 shape:(12814, 37)
        o_profile_one_hot = np.concatenate([self.get_one_hot_encoding('o', column) for column in o_column[1:]], axis=1)  #(12814, 37)  
        
        # X1_p_train = c_profile_one_hot[self.x_train['target_id']]   # sum:98 shape:(52336, 108)
        # X2_p_train = o_profile_one_hot[self.x_train['holder_id']]  # sum:0 shape:(52336, 37)
        # X1_p_test = c_profile_one_hot[self.x_test['target_id']] # sum:0 shape:(1220, 108)
        # X2_p_test = o_profile_one_hot[self.x_test['holder_id']] # sum:0 shape:(1220, 37)

        X1_p_train = o_profile_one_hot[self.x_train['holder_id']]   # sum:98 shape:(52336, 108)
        X2_p_train = c_profile_one_hot[self.x_train['target_id']]  # sum:0 shape:(52336, 37)
        X1_p_test = o_profile_one_hot[self.x_test['holder_id']] # sum:0 shape:(1220, 108)
        X2_p_test = c_profile_one_hot[self.x_test['target_id']] # sum:0 shape:(1220, 37)
        
        return X1_p_train, X2_p_train, X1_p_test, X2_p_test, o_profile_one_hot, c_profile_one_hot

    def get_one_hot_encoding(self, category, column):
        if category == 'c':
            feature = self.c_profile[['id', column]]
         
        else:
            feature = self.o_profile[['id', column]]
        dictionary = {}
        cnt = 0
        # construct dictionary
        for index, row in feature.iterrows():
            values = str(row[column]).split('/')
            for value in values:
                if value != '未知' and value not in dictionary:
                    dictionary[value] = cnt
                    cnt += 1
        one_hot_matrix = np.zeros([self.user_size, len(dictionary)], dtype=np.float)
        #print(one_hot_matrix.shape) #(12814,X) X為屬性類別數
        # get one-hot encoding
        for index, row in feature.iterrows():
            values = str(row[column]).split('/')
            for value in values:
                if value != '未知':
                    one_hot_matrix[row['id'], dictionary[value]] = 1
        return one_hot_matrix


if __name__ == '__main__':
    data = Data('data/sentiment.csv',
                'data/social_relation.csv',
                'data/celebrity_profile_trad.csv',
                'data/ordinary_user_profile_trad.csv',
                1)
    data.get_sentiment_data()
