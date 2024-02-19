import math
import numpy as np
import pandas as pd


class UserCF:

    def __init__(self, file_path):
        self.file_path = file_path
        self._init_frame()
        self.max_rating = 5
        self.movie_mean_rating = dict(self.frame.groupby('item_id').mean('rating')['rating'])

    def _init_frame(self):
        self.frame = pd.read_csv(self.file_path)

    @staticmethod
    def _cosine_sim_watched(target_movies, movies):
        '''
        simple method for calculate cosine simialarity between users (based on the movie sets they watched).
        '''
        common_movies = set(target_movies) & set(movies)
        if len(common_movies) == 0: 
            return 0.0
        product = len(target_movies) * len(movies)
        cosine_similarity = len(common_movies) / math.sqrt(product)
        return cosine_similarity
    
    @staticmethod
    def _cosine_sim_rating(target_ratings, other_ratings):
        '''
        another method for calculate cosine simialarity between users (based on the ratings they gave to the movies).
        '''
        common_movies = set(target_ratings.keys()) & set(other_ratings.keys())
        
        if len(common_movies) == 0:
            return 0.0 
        
        target_ratings_vector = np.array([target_ratings[movie] for movie in common_movies])
        other_ratings_vector = np.array([other_ratings[movie] for movie in common_movies])
        
        dot_product = np.dot(target_ratings_vector, other_ratings_vector)
        
        target_norm = np.linalg.norm(target_ratings_vector)
        other_norm = np.linalg.norm(other_ratings_vector)
        
        if target_norm == 0 or other_norm == 0:
            return 0.0
        
        cosine_similarity = dot_product / (target_norm * other_norm)
        
        return cosine_similarity

    def _get_top_n_users(self, target_user_id, top_n, user_sim):
        '''
        calculate similarity between all users and return Top N similar users.
        '''
        
        target_data = self.frame[self.frame['user_id'] == target_user_id]
        target_ratings = dict(zip(target_data['item_id'], target_data['rating']))

        other_users_id = [i for i in set(self.frame['user_id']) if i != target_user_id]
        
        sim_list = []

        for user_id in other_users_id:
            # Get each other user's movies and ratings
            other_data = self.frame[self.frame['user_id'] == user_id]
            # if user have multiple ratings for one movie, user the last rating
            other_ratings = dict(zip(other_data['item_id'], other_data['rating'])) 
            
            if user_sim == 'watched':
                similarity = self._cosine_sim_watched(list(target_ratings.keys()), list(other_ratings.keys()))
            if user_sim == 'rating':
                similarity = self._cosine_sim_rating(target_ratings, other_ratings)
            
            sim_list.append((user_id, similarity))
        
        # Sort the list of tuples by similarity score in descending order and get the top N
        sim_list = sorted(sim_list, key=lambda x: x[1], reverse=True)
        
        return sim_list[:top_n]

    def _get_candidates_items(self, target_user_id):
        """
        Find all movies in source data and target_user did not meet before.
        """
        target_user_movies = set(self.frame[self.frame['user_id'] == target_user_id]['item_id'])
        other_user_movies = set(self.frame[self.frame['user_id'] != target_user_id]['item_id'])
        candidates_movies = list(target_user_movies ^ other_user_movies)
        return candidates_movies

    def _get_top_n_items(self, top_n_users, candidates_movies, top_n):
        """
        calculate interest of candidates movies and return top n movies.
        """
        top_n_user_data = [self.frame[self.frame['user_id'] == user_id] for user_id, _ in top_n_users]
        interest_list = []
        for movie_id in candidates_movies:
            tmp = [] # store rating for user_j
            watched_cnt = 0 # record number of users in top_n who watched this movie
            for user_data in top_n_user_data:
                if movie_id in user_data['item_id'].values:
                    tmp.append(user_data[user_data['item_id'] == movie_id]['rating'].values[-1])
                    watched_cnt += 1
                else:
                    tmp.append(0)
                    # # Wrong idea: if user_j did not watch this movie, fill the rating with the mean rating of this movie
                    # tmp.append(self.movie_mean_rating[movie_id])

            # predicted rating = sum of (user_similarity_ij * user_j_rating) for all J / normalize factor
            normal_factor1 = len(top_n_users) # |J|
            normal_factor2 = sum([top_n_users[i][1] for i in range(len(top_n_users))]) # sum of all user_similarity_ij
            normal_factor3 = watched_cnt
            normal_factor4 = sum([top_n_users[i][1] for i in range(len(top_n_users)) if tmp[i] != 0]) # sum of user_similarity_ij only for user_j who watched this movie.
            if watched_cnt == 0:
                interest = 0
            # The following solved the issue that interest score varies too much for a niche movie
            # by adding an adjustment to the denominator
            else:
                smoothing_param = 0.5 * top_n
                # Original factor with smoothing applied
                adjusted_factor = (watched_cnt + smoothing_param) / (len(top_n_users) + smoothing_param)

                # Using a logarithmic scale to reduce the range of adjustment
                log_adjusted_factor = np.log(watched_cnt + smoothing_param) / np.log(len(top_n_users) + smoothing_param)

                weighted_sum = sum([top_n_users[i][1] * tmp[i] for i in range(len(top_n_users))]) 
                interest = weighted_sum / normal_factor4 * log_adjusted_factor

            interest_list.append((movie_id, interest))
        interest_list = sorted(interest_list, key=lambda x: x[1], reverse=True)
        return interest_list[:top_n]

    def calculate(self, target_user_id=1, top_n=10, user_sim='watched'):
        """
        user-cf for movies recommendation.
        """
        # most similar top n users
        top_n_users = self._get_top_n_users(target_user_id, top_n, user_sim)
        # candidates movies for recommendation
        candidates_movies = self._get_candidates_items(target_user_id)
        # most interest top n movies
        top_n_movies = self._get_top_n_items(top_n_users, candidates_movies, top_n)
        return top_n_movies