import math
import random
import numpy as np
import pandas as pd


class ItemCF:

    def __init__(self, file_path):
        self.file_path = file_path
        self._init_frame()

    def _init_frame(self):
        self.frame = pd.read_csv(self.file_path)

    @staticmethod
    def _cosine_sim(target_users, users):
        '''
        simple method for calculate cosine distance between movies.
        '''
        common_users = set(target_users) & set(users)
        if len(common_users) == 0: 
            return 0.0
        product = len(target_users) * len(users)
        cosine_similarity = len(common_users) / math.sqrt(product)
        return cosine_similarity
    
    @staticmethod
    def _cosine_sim_rating(target_ratings, other_ratings):
        '''
        another method for calculate cosine simialarity between movies (based on the their ratings).
        target_ratings: key -> user_id
        '''
        common_users = set(target_ratings.keys()) & set(other_ratings.keys())
        
        if len(common_users) == 0:
            return 0.0 
        
        target_ratings_vector = np.array([target_ratings[user] for user in common_users])
        other_ratings_vector = np.array([other_ratings[user] for user in common_users])
        
        dot_product = np.dot(target_ratings_vector, other_ratings_vector)
        
        target_norm = np.linalg.norm(target_ratings_vector)
        other_norm = np.linalg.norm(other_ratings_vector)
        
        if target_norm == 0 or other_norm == 0:
            return 0.0
        
        cosine_similarity = dot_product / (target_norm * other_norm)
        
        return cosine_similarity

    def _get_n_recent_items(self, target_user_id, n):
        '''
        Get n recent movies that has been reviewed by the user
        '''
        if n > 20:
            n = 20
        target_data = self.frame[self.frame['user_id'] == target_user_id]
        item_pool = dict(zip(target_data['item_id'][-n:], target_data['rating'][-n:]))
        
        return item_pool
    
    def _get_n_random_items(self, target_user_id, n):
        '''
        Get n recent movies that has been reviewed by the user
        '''
        if n > 20:
            n = 20
        target_data = self.frame[self.frame['user_id'] == target_user_id]
        sample_data = target_data.iloc[random.sample(range(len(target_data)), n), :]
        item_pool = dict(zip(sample_data['item_id'], sample_data['rating']))

        return item_pool

    def _get_candidates_items(self, target_user_id):
        """
        Find all movies in source data and target_user did not meet before.
        """
        target_user_movies = set(self.frame[self.frame['user_id'] == target_user_id]['item_id'])
        other_user_movies = set(self.frame[self.frame['user_id'] != target_user_id]['item_id'])
        candidates_movies = list(target_user_movies ^ other_user_movies)
        return candidates_movies

    def _get_top_n_items(self, recent_n_items, random_n_items, candidates_movies, top_n, user_sim):
        """
        calculate interest of candidates movies and return top n movies.
        """
        interest_list = []
        for candidate_id in candidates_movies:
            target_data = self.frame[self.frame['item_id'] == candidate_id]
            target_ratings = dict(zip(target_data['user_id'], target_data['rating']))
            weighted_ratings = [] # store similarity * rating of movies that user_i interacted with recently
            similarities = []
            for recent_id in recent_n_items.keys():
                other_data = self.frame[self.frame['item_id'] == recent_id]
                other_ratings = dict(zip(other_data['user_id'], other_data['rating']))
                if user_sim == 'watched':
                    similarity = self._cosine_sim(target_ratings.keys(), other_ratings.keys())
                    weighted_ratings.append(similarity * recent_n_items[recent_id])
                    similarities.append(similarity)
                elif user_sim == 'rating':
                    similarity = self._cosine_sim_rating(target_ratings, other_ratings)
                    weighted_ratings.append(similarity * recent_n_items[recent_id])
                    similarities.append(similarity)
            if sum(similarities) == 0:
                interest = 0
            else:
                interest = sum(weighted_ratings) / sum(similarities)

            interest_list.append((candidate_id, interest))
        interest_list = sorted(interest_list, key=lambda x: x[1], reverse=True)
        return interest_list[:top_n]

    def calculate(self, target_user_id=1, top_n=10, user_sim='watched'):
        """
        user-cf for movies recommendation.
        """
        # pool of movies with user-interaction
        recent_n_items = self._get_n_recent_items(target_user_id, top_n)
        random_n_items = self._get_n_random_items(target_user_id, top_n)

        # candidates movies for recommendation
        candidates_movies = self._get_candidates_items(target_user_id)
        
        # most interest top n movies
        top_n_movies = self._get_top_n_items(recent_n_items, random_n_items, candidates_movies, top_n, user_sim)
        return top_n_movies