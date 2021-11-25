# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 00:16:37 2020

@author: Kyle
"""

import twitter_credentials as tc
import tweepy
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import networkx as nx


"""Globals"""
consumer_key = tc.consumer_key
consumer_secret_key = tc.consumer_secret
access_token = tc.access_token
access_token_secret = tc.access_token_secret

auth = tweepy.OAuthHandler(consumer_key, consumer_secret_key)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

file = open("twitter_data_demo.txt", "a", encoding="utf-8")

def stream_tweets(search_term, num_samples, user_list):
    counter = 0  # Var holds how many samples have been collected
    for tweet in tweepy.Cursor(
        api.search, q='"{}" -filter:retweets'.format(search_term), count=10, lang="en"
    ).items():

        if counter == num_samples:
            break

        elif tweet.user.id not in user_list and 40 < tweet.user.followers_count < 10000:
            user_list.append(tweet.user.id)
            counter += 1
            print(counter)


def find_network(seen_users, users_max):
    progress = lambda a: str((a / len(seen_users) * 100))
    global full_similarity
    iterations = 0

    vert1, vert2, edges = [], [], []

    for user in seen_users:
        iterations += 1
        print(progress(iterations) + "% complete")

        try:
            friend_counter1 = 0
            for follower1 in tweepy.Cursor(api.followers, user).items():
                if friend_counter1 == users_max:
                    break
                else:
                    vert1.append(user)
                    vert2.append(follower1.id)
                    edges.append(3)
                    friend_counter1 += 1

                    friend_counter2 = 0
                    for follower2 in tweepy.Cursor(api.followers, follower1.id).items():
                        if friend_counter2 == users_max:
                            break

                        else:
                            vert1.append(user)
                            vert2.append(follower2.id)
                            edges.append(2)
                            friend_counter2 += 1

                            friend_counter3 = 0
                            for follower3 in tweepy.Cursor(
                                api.followers, follower2.id
                            ).items():
                                if friend_counter3 == users_max:
                                    break
                                else:
                                    vert1.append(user)
                                    vert2.append(follower3.id)
                                    edges.append(1)
                                    friend_counter3 += 1
        except tweepy.TweepError as e:
            print(e)

    vert1 = np.array(vert1).reshape(-1, 1)
    vert2 = np.array(vert2).reshape(-1, 1)
    edges = np.array(edges).reshape(-1, 1)

    full_links = np.hstack((vert1, vert2, edges))
    return full_links


def encode_user(edge_list):
    shape = edge_list[:, [0, 1]].shape  # Save original shape

    vertices = edge_list[:, [0, 1]].ravel()
    vertices = LabelEncoder().fit_transform(vertices)
    vertices = vertices.reshape(shape[0], shape[1])
    edge_list[:, [0, 1]] = vertices

    edge_df = pd.DataFrame(edge_list, columns=["vert1", "vert2", "weight"])
    return edge_df


def edge_list_to_matrix(edge_df):
    G = nx.convert_matrix.from_pandas_edgelist(edge_df, "vert1", "vert2", "weight")

    A = (nx.to_pandas_adjacency(G, dtype="int32")).to_numpy()

    return A


if __name__ == "__main__":
    print("Starting to stream...")
    search_terms = ["covid"]

    user_list = []

    for keyword in search_terms:
        stream_tweets(keyword, 3, user_list)

    edge_list = find_network(user_list, 1)
    edge_df = encode_user(edge_list)
    adj_matrix = edge_list_to_matrix(edge_df)

    file.close()
    print("Finished")
