import math
import random
import numpy as np
import pandas as pd
from collections import defaultdict

class Swing:

    def __init__(self, file_path):
        self.file_path = file_path
        self._init_frame()
        self._init_inverted_index()

    def _init_frame(self):
        self.frame = pd.read_csv(self.file_path)

    def _init_inverted_index(self):
        self.user_to_items = self.frame.groupby('user_id').agg(set)['item_id']
        self.item_to_users = self.frame.groupby('item_id').agg(set)['user_id']
        self.swings = self._store_swings()

    def _calculate_swing(self, target_item_id, alpha=1.0):
        similarites = defaultdict(int)
        user_pool = self.item_to_users[target_item_id]
        for u_id in user_pool:
            u_item_pool = self.user_to_items[u_id]
            weight_u = 1 / math.sqrt(len(u_item_pool))
            
            for v_id in user_pool:
                if u_id == v_id:
                    continue
                v_item_pool = self.user_to_items[v_id]
                weight_v = 1 / math.sqrt(len(v_item_pool))

                uv_item_pool = u_item_pool & v_item_pool
                k = len(uv_item_pool)
                for item_j_id in uv_item_pool:
                    if item_j_id == target_item_id:
                        continue
                    
                    similarity = weight_u * weight_v * (1 / (alpha + k))
                    similarites[item_j_id] += similarity

        return similarites
    
    def _store_swings(self):
        swings = dict()
        for item_id in self.item_to_users.index:
            swings[item_id] = self._calculate_swing(item_id)
        return swings
    
    def _sort_swing(self, target_item_id):
        similarites = list(self.swings[target_item_id].items())
        similarites = sorted(similarites, key=lambda x: x[1], reverse=True)
        return similarites 

    def _get_n_recent_items(self, target_user_id, n):
        '''
        Get n recent movies that has been reviewed by the user
        '''
        if n > 20:
            n = 20
        target_data = self.frame[self.frame['user_id'] == target_user_id]
        item_pool = dict(zip(target_data['item_id'][-n:], target_data['rating'][-n:]))
        
        return item_pool
    
    def _get_candidates_items(self, target_user_id):
        """
        Find all movies in source data and target_user did not meet before.
        """
        target_user_movies = set(self.frame[self.frame['user_id'] == target_user_id]['item_id'])
        other_user_movies = set(self.frame[self.frame['user_id'] != target_user_id]['item_id'])
        candidates_movies = list(target_user_movies ^ other_user_movies)
        return candidates_movies
    
    def recommend_naive(self, target_user_id, n_retrieve=10, n_recent=10, top_n=10):
        '''
        Recommend new items based on items with recent user interaction
        Normally we need to compute the predicted rating for all candidate items,
        but here for performance reasons, I tried to retrieve knn based on single trigger item(recently viewed)
        n_recent items will retrieve n_recent * top_n items in total,
        sort and truncate to n_retrieve items.
        '''
        recent_items_ratings = self._get_n_recent_items(target_user_id, n_recent)
        user_watched_movies = set(self.frame[self.frame['user_id'] == target_user_id]['item_id'])
        interest_list = []
        for target_item_id in recent_items_ratings.keys():
            similarities_unfiltered = self._sort_swing(target_item_id)
            similarities = []
            for x in similarities_unfiltered:
                if x[0] not in user_watched_movies:
                    similarities.append(x)
            similarities = similarities[:top_n]
            interests = [(similarities[i][0], recent_items_ratings[target_item_id] * similarities[i][1]) for i in range(len(similarities))]
            interest_list += interests
        interest_list = sorted(interest_list, key=lambda x: x[1], reverse=True)
        return interest_list[:n_retrieve]
    
    def recommend(self, target_user_id, n_retrieve=10, n_recent=10):
        '''
        Recommend n_retrieve items to the target user
        '''
        interest_list = []
        candidate_movies = self._get_candidates_items(target_user_id)
        recent_n_items = self._get_n_recent_items(target_user_id, n=n_recent)
        for candidate_id in candidate_movies:
            swing = self.swings[candidate_id]
            interests = []
            for item_id in recent_n_items:
                similarity = swing[item_id] # swing is a default_dict(int)
                interest = similarity * recent_n_items[item_id]
                interests.append(interest)
            interest_list.append((candidate_id, sum(interests)))
        interest_list = sorted(interest_list, key=lambda x: x[1], reverse=True)
        return interest_list[:n_retrieve]
            


        
