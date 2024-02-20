import math
import random
import numpy as np
import pandas as pd


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

    def _calculate_similarity(self, target_item_id, alpha=1.0):
        similarites = []
        user_pool = self.item_to_users[target_item_id]
        for u_id in user_pool:
            u_item_pool = self.user_to_items[u_id]
            weight_u = 1 / math.sqrt(len(u_item_pool))
            
            for v_id in user_pool:
                v_item_pool = self.user_to_item[v_id]
                weight_v = 1 / math.sqrt(len(v_item_pool))

                uv_item_pool = u_item_pool & v_item_pool
                k = len(uv_item_pool)
                for item_j_id in uv_item_pool:
                    if item_j_id == target_item_id:
                        continue
                    similarity = weight_u * weight_v * (1 / (alpha + k))
                    similarites.append((item_j_id, similarity))
        return similarites
    
