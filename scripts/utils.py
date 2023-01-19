import numpy as np
import networkx as nx
import igraph
import matplotlib.pyplot as plt
import tensorflow as tf
import plotly.graph_objects as go
import pandas as pd
import pickle as pkl
import json
import sys
import argparse

from tqdm import tqdm
from sklearn.decomposition import NMF, non_negative_factorization
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score

from IPython.display import Image
from plotly.subplots import make_subplots
from scipy.stats import wasserstein_distance

from tensorflow.keras import layers, models, Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Input, Dropout, Concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

import os
import time

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import numpy.random as rand
import matplotlib.pyplot as plt

from scipy.sparse import linalg as spla

from tensorflow.keras import layers, models, Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Input, Dropout, Concatenate, Add, Subtract, Lambda
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, History
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.regularizers import l1_l2

import keras.backend as K

from collections import deque
from sklearn.neighbors import KernelDensity
from scipy import stats


import math 
import random
import keras.backend as K

from tensorflow.keras.layers import Dense, Flatten, Input, Dropout, Concatenate, Embedding, Lambda, Reshape
from collections import deque

sys.path.append('../')

import torch
import torch.nn as nn


###################################
######## Helper Functions #########
###################################
def add_weights(G, weights):

    '''
        Add weights to a graph
        Input : 
            G       : nx Graph object - Input graph
            weights : String - Poisson | Uniform
    '''

    num_weights = G.number_of_edges()
    
    if weights == 'Poisson':
        w = 1 + np.random.poisson(20, (num_weights))
    elif weights == 'Uniform':
        w = 1 + np.random.randint(41, size = (num_weights))
    else:
        w = np.ones((num_weights))

    for idx, e in enumerate(G.edges):
        G.edges[e]['weight'] = w[idx]

def generate_graph(graph_type, num_nodes, param, weights):
    
    assert(weights in ['Poisson', 'Uniform', 'Equal'])

    # Erdos-Renyi Graph
    if graph_type == 'er':    
        graph = nx.erdos_renyi_graph(n = num_nodes, p = param[0])
        add_weights(graph, weights)

    # Barabasi-Albert Graph
    elif graph_type == 'ba':   
        graph = nx.barabasi_albert_graph(n = num_nodes, m = param[0])
        add_weights(graph, weights)
        
    # Watts-Strogatz Graph
    elif graph_type == 'ws':   
        graph = nx.watts_strogatz_graph(n = num_nodes, k = param[0], p = param[1])
        add_weights(graph, weights)
        
    # Lattice Graph
    elif graph_type == 'lattice':   
        graph = nx.Graph(nx.adjacency_matrix(nx.grid_2d_graph(num_nodes, num_nodes)))
        add_weights(graph, weights)
    
    # Complete Graph
    elif graph_type == 'complete':
        graph = nx.complete_graph(num_nodes)
        add_weights(graph, weights)

    else:
        print('Invalid graph name. Please try one of : er, ba, ws, lattice, complete')
        raise
    
    return graph

def get_sense_features(graph, ppr_flag = 'std', weighted = False):
    
    if weighted: 
        sense_feat_dict = {

            'Degree' : 0,
            'Weighted Degree' : 1, 
            'Clustering Coefficient' : 2, 
            'Personalized Page Rank - Median' : 3,
            'Personalized Page Rank - Standard Deviation' : 4,
            'Structural Holes Constraint' : 5, 
            'Average Neighbor Degree' : 6,
            'EgoNet Edges' : 7, 
            'Average Neighbor Clustering' : 8,
            'Node Betweenness' : 9, 
            'Page Rank' : 10, 
            'Eccentricity' : 11,
            'Degree Centrality' : 12, 
            'Eigen Centrality' : 13,
            'Katz Centrality' : 14
        }
        
    else: 
        sense_feat_dict = {

            'Degree' : 0,
            'Clustering Coefficient' : 1, 
            'Personalized Page Rank - Median' : 2,
            'Personalized Page Rank - Standard Deviation' : 3,
            'Structural Holes Constraint' : 4, 
            'Average Neighbor Degree' : 5,
            'EgoNet Edges' : 6, 
            'Average Neighbor Clustering' : 7,
            'Node Betweenness' : 8, 
            'Page Rank' : 9, 
            'Eccentricity' : 10,
            'Degree Centrality' : 11, 
            'Eigen Centrality' : 12,
            'Katz Centrality' : 13,
        }
    
    if len(list(nx.algorithms.components.connected_components(graph))) > 1:
        print ("Disconnected Network")
        sense_feat_dict = {

            'Degree' : 0,
            'Weighted Degree' : 1, 
            'Clustering Coefficient' : 2, 
            'Personalized Page Rank - Median' : 3,
            'Personalized Page Rank - Standard Deviation' : 4,
            'Structural Holes Constraint' : 5, 
            'Average Neighbor Degree' : 6,
            'EgoNet Edges' : 7, 
            'Average Neighbor Clustering' : 8,
            'Node Betweenness' : 9, 
            'Page Rank' : 10, 
            'Degree Centrality' : 11, 
            'Eigen Centrality' : 12, 
            'Katz Centrality' : 13
          }
        
    if ppr_flag == 'mean': 
        print ("Using Means For PPR")
        sense_feat_dict = {
    
            'Degree' : 0,
            'Weighted Degree' : 1, 
            'Clustering Coefficient' : 2, 
            'Personalized Page Rank - Mean' : 3,
            'Structural Holes Constraint' : 4, 
            'Average Neighbor Degree' : 5,
            'EgoNet Edges' : 6, 
            'Average Neighbor Clustering' : 7,
            'Node Betweenness' : 8, 
            'Page Rank' : 9, 
            'Eccentricity' : 10,
            'Degree Centrality' : 11,
            'Eigen Centrality' : 12, 
            'Katz Centrality' : 13
        }
    
    ig = igraph.Graph([[e[0], e[1]] for e in nx.to_edgelist(graph)])
    sense_features = np.zeros((len(graph), len(sense_feat_dict)))

    print ("Calculating Degrees...                                   ", end = '\r')
    # Degree
    sense_features[:, sense_feat_dict['Degree']] = list(dict(graph.degree).values())

    if weighted: 
        print ("Calculating Weighted Degrees...                           ", end = '\r')
        # Weighted Degree
        sense_features[:, sense_feat_dict['Weighted Degree']] = list(dict(graph.degree(weight = 'weight')).values())
    
    print ("Calculating Average Neighbor Degree...                    ", end = '\r')
    # Neighbor Degree Average
    sense_features[:, sense_feat_dict['Average Neighbor Degree']] = [np.mean([graph.degree[neighbor] for neighbor in dict(graph[node]).keys()]) for node in graph.nodes]

    print ("Calculating Clustering Coefficient...                     ", end = '\r')
    # Clustering Coefficient
    cluster_dict = nx.clustering(graph)
    sense_features[:, sense_feat_dict['Clustering Coefficient']] = list(cluster_dict.values())

    print ("Calculating Average Neighbor Clustering Coefficients...   ", end = '\r')
    # Neighbor Average Clustering 
    sense_features[:, sense_feat_dict['Average Neighbor Clustering']] = [np.mean([cluster_dict[neighbor] for neighbor in list(graph[node])]) for node in graph.nodes]
    
    print ("Calculating Eccentricity...                               ", end = '\r')
    # Eccentricity
    try:
        sense_features[:, sense_feat_dict['Eccentricity']] = ig.eccentricity() #list(nx.algorithms.distance_measures.eccentricity(graph).values())
    except Exception as e: 
        print ("Could not compute Eccentricity : ", e)
    
    print ("Calculating Page Rank...                                  ", end = '\r')
    # Page Rank
    sense_features[:, sense_feat_dict['Page Rank']] = ig.pagerank(directed = False) #list(nx.pagerank(graph).values())
    
    print ("Calculating Personalized Page Rank...                     ", end = '\r')
    
    if ppr_flag == 'mean':
        ppr = np.zeros((1, len(graph)))
        for node_idx, node in tqdm(enumerate(range(len(graph)))):
            r = np.zeros((len(graph)))
            r[node] = 1
            ppr = ppr + ig.personalized_pagerank(reset = r, directed = False)
        ppr = ppr / len(graph)
        sense_features[:, sense_feat_dict['Personalized Page Rank - Mean']] = ppr
        
        
    else: 
        ppr = np.zeros((len(graph), len(graph)))
        for node_idx, node in tqdm(enumerate(range(len(graph)))):
            r = np.zeros((len(graph)))
            r[node] = 1
            ppr[node_idx, :] = ig.personalized_pagerank(reset = r, directed = False)

        sense_features[:, sense_feat_dict['Personalized Page Rank - Standard Deviation']] = np.std(ppr, axis = 0)
        sense_features[:, sense_feat_dict['Personalized Page Rank - Median']] = np.median(ppr, axis = 0)
    
    print ("Calculating Node Betweenness...                           ", end = '\r')
    # Node Betweenness 
    sense_features[:, sense_feat_dict['Node Betweenness']] = ig.betweenness(directed = False) #list(nx.algorithms.centrality.betweenness_centrality(graph).values())

    print ("Calculating Number Of Edges In Ego Nets...                ", end = '\r')
    # EgoNet Edges
    sense_features[:, sense_feat_dict['EgoNet Edges']] = [len(nx.ego_graph(graph, n = node).edges) for node in graph.nodes]

    print ("Calculating Structural Hole Constraint Scores...         ", end = '\r')
    # Structual Holes
    sense_features[:, sense_feat_dict['Structural Holes Constraint']] = ig.constraint() #list(nx.algorithms.structuralholes.constraint(graph, weight = 'weight').values())

    
    print ("Calculating Degree Centrality...                         ", end = '\r')
    sense_features[:, sense_feat_dict['Degree Centrality']] =  list(dict(nx.degree_centrality(graph)).values())
    
    print ("Calculating Eigen Centrality...                          ", end = '\r')
    sense_features[:, sense_feat_dict['Eigen Centrality']] = ig.eigenvector_centrality(directed = False)
    
    print ("Calculating Katz Centrality...                           ", end = '\r')
    sense_features[:, sense_feat_dict['Katz Centrality']] =  list(dict(nx.katz_centrality_numpy(graph)).values())
    
    
    print ("Normalizing Features Between 0 And 1...                   ", end = '\r')
    # Normalise to between 0 and 1 
    sense_features = (sense_features - np.min(sense_features, axis = 0)) / np.ptp(sense_features, axis = 0)
    
    print ("Done                                                      ", end = '\r')
    
    return sense_feat_dict, sense_features
    

def get_positional_sense_features(graph, num_anchors, anchor_list = None):
        
    graph.remove_edges_from(nx.selfloop_edges(graph))
    
    core_numbers = np.array(list(dict(nx.core_number(graph)).values()))
    core_p = core_numbers / np.sum(core_numbers)
    
    if anchor_list is None:
        core_anchors = np.random.choice(len(graph), p = core_p, size = num_anchors)
    else: 
        core_anchors = anchor_list
    
    sense_feat_dict = []

    sense_features = np.zeros((len(graph), 1 + (2 * num_anchors)))

    ig = igraph.Graph([[e[0], e[1]] for e in nx.to_edgelist(graph)])

    print ("Computing Core Number...", end = '\r')
    sense_features[:, len(sense_feat_dict)] = core_numbers
    sense_feat_dict.append("Core Number")

    print ("Computing PPR to Core Random Nodes...", end = '\r')
    for idx, node in tqdm(enumerate(core_anchors)):
        r = np.zeros((len(graph)))
        r[node] = 1
        sense_features[:, len(sense_feat_dict)] = ig.personalized_pagerank(reset = r, directed = False)
        sense_feat_dict.append("PPR To Random Node " + str(idx)) 

    print ("Computing Hops to Core Random Nodes...", end = '\r')
    for idx, node in tqdm(enumerate(core_anchors)):
        sp_ = nx.single_source_shortest_path_length(graph, source = node)
        sense_features[:, len(sense_feat_dict)] = [sp_[n] for n in range(len(graph))]
        sense_feat_dict.append("Hops To Random Node " + str(idx))


    print ("Normalizing Features Between 0 And 1...                   ", end = '\r')
    # Normalise to between 0 and 1 
    sense_features = (sense_features - np.min(sense_features, axis = 0)) / np.ptp(sense_features, axis = 0)
    sense_feat_dict = {sense_feat_dict[idx] : idx for idx in range(len(sense_feat_dict))}
    
    return sense_feat_dict, sense_features

def find_feature_membership(input_embed, embed_name, sense_features, sense_feat_dict, top_k = 8, gd_steps = 1000, solver = 'nmf', plot = False, constraints = False):
    
    
    if solver == 'gd' :
        # Tensorflow Variables For Optimization
        # Input embedding - fixed
        embeddings = tf.Variable(initial_value = input_embed,
                    shape = input_embed.shape,
                    dtype = tf.float32, trainable = False)

        # Matrix explaining membership - trainable
        explain = tf.Variable(initial_value = np.random.randn(input_embed.shape[1], sense_features.shape[1]),
                        shape = (input_embed.shape[1], sense_features.shape[1]),
                        dtype = tf.float32, trainable = True)

        # Explainable features - fixed
        sense = tf.Variable(initial_value = sense_features,
                        shape = sense_features.shape,
                        dtype = tf.float32, trainable = False)

        # Set up an optimizer
        optimizer = tf.keras.optimizers.Nadam(learning_rate = 0.001)

        # Optimize 
        losses = []
        for i in tqdm(range(gd_steps)):

            with tf.GradientTape() as tape:

                # Minimize || sense - embeddings @ explain ||
                prod = tf.matmul(embeddings, explain)
                loss = tf.norm(sense - prod, ord = 2)
                
                if constraints == True:
                    loss = loss + (0.5 * tf.linalg.norm(tf.matmul(explain, explain, transpose_b = True))) + (0.5 * tf.math.reduce_sum(tf.linalg.norm(explain, axis = 0)))
        
            gradients = tape.gradient(loss, [explain])
            optimizer.apply_gradients(zip(gradients, [explain]))
            explain.assign(tf.clip_by_value(explain, clip_value_min = 0, clip_value_max = tf.math.reduce_max(explain)))
            
            losses.append(loss)
        
        reconstruction_loss = losses[-1]
        print ("Reconstruction Loss : ", float(loss))

        # Default embed vector - assume one dimension only - non trainable
        embeddings_default = tf.Variable(initial_value = np.ones((input_embed.shape[0], 1)),
                                     shape = (input_embed.shape[0], 1),
                                     dtype = tf.float32, trainable = False)

        # Default explanantion vector - trainable
        explain_default = tf.Variable(initial_value = np.random.randn(1, sense_features.shape[1]),
                                      shape = (1, sense_features.shape[1]),
                                      dtype = tf.float32, trainable = True)

        optimizer = tf.keras.optimizers.Nadam(learning_rate = 0.001)

        default_losses = []
        for i in tqdm(range(gd_steps)):
            
                                  
            with tf.GradientTape() as tape:

                prod = tf.matmul(embeddings_default, explain_default)
                loss = tf.norm(sense - prod, ord = 2)

            gradients = tape.gradient(loss, [explain_default])
            optimizer.apply_gradients(zip(gradients, [explain_default]))
            explain_default.assign(tf.clip_by_value(explain_default, clip_value_min = 0, clip_value_max = tf.math.reduce_max(explain_default)))
            default_losses.append(loss)

        print ("Default Loss : ", float(loss))
        
    if solver == 'nmf':
        
        # Ensure proper dtypes
        sense_features = sense_features.astype(np.float32)
        input_embed = input_embed.astype(np.float32)

        # Play around with transposes to make it make sense
        explain, embed_recon, _ = non_negative_factorization(n_components = input_embed.shape[1],
                                                                     init = 'custom',
                                                                     max_iter = 4000,
                                                                     X = sense_features.T,
                                                                     H = input_embed.T,
                                                                     update_H = False)

        explain = explain.T
        embed_recon = embed_recon.T

        reconstruction_loss = np.linalg.norm(sense_features - (input_embed @ explain))
        
        default_embed = np.ones((input_embed.shape[0], 1)).astype(np.float32)
        explain_default, _, _ = non_negative_factorization(n_components = default_embed.shape[1],
                                                             init = 'custom',
                                                             max_iter = 2000,
                                                             X = sense_features.T,
                                                             H = default_embed.T,
                                                             update_H = False)
        explain_default = explain_default.T
        loss_2 = np.linalg.norm(sense_features - (default_embed @ explain_default))
                            
                                
        
    
    # Normalize matrix by the default matrix learned
    explain_norm = np.array(explain / explain_default)    
    explain_norm_softmax = np.array([np.exp(x) / sum(np.exp(x)) for x in explain_norm])
    explain_variance = np.square(np.std(explain_norm, axis = 1))
    
    # Plot variance in explanability of each dimension
    embed_dimensions = input_embed.shape[1]

    if plot: 
        fig = go.Figure()
        fig.add_trace(go.Bar(x = list(range(embed_dimensions)), 
                             y = explain_variance,
                             name = 'Variance of Embedding Dimensions'))
        fig.add_trace(go.Scatter(x = list(range(embed_dimensions)), 
                                 y = [np.mean(explain_variance)] * embed_dimensions, 
                                 mode = 'lines', 
                                 name = 'Mean of Variance'))
        fig.add_trace(go.Scatter(x = list(range(embed_dimensions)), 
                                 y = [np.median(explain_variance)] * embed_dimensions, 
                                 mode = 'lines', 
                                 name = 'Median of Variance'))
        fig.update_layout(title_text = 'Variance of Explanability Across Dimensions - ' + embed_name,
                          xaxis_title_text = 'Dimensions', 
                          yaxis_title_text = 'Variance')
        fig.show()
    
    # Figure out which dimensions to keep - ones with most variance 
    dimensions_idx_to_keep = np.where(explain_variance > np.mean(explain_variance))[0]
    dimensions_to_keep = np.array(explain_norm)[dimensions_idx_to_keep]
    dimensions_to_keep_softmax = explain_norm_softmax[dimensions_idx_to_keep]
    top_k_dims = np.argsort(explain_variance)[-top_k:]
    
    # Plot membership of sense features vs remaining dimensions
    features = list(sense_feat_dict.keys())
    
    if plot: 
        fig = go.Figure()

        for idx in range(len(dimensions_to_keep)):
            fig.add_trace(go.Bar(x = features, 
                                 y = dimensions_to_keep[idx],
                                 name = 'Dimension ' + str(dimensions_idx_to_keep[idx])))

        fig.update_layout(title_text = 'Embedding Dimension Feature Membership - ' + embed_name,
                          xaxis_title_text = 'Sense Features',
                          yaxis_title_text = 'Membership',
                          barmode = 'group')
        fig.show()


    return_dict = {
        'explain' : explain,
        'explain_norm' : explain_norm,
        'explain_default' : explain_default,
        'dimensions_idx_to_keep' : dimensions_idx_to_keep,
        'top_k_dims' : top_k_dims,
        'reconstruction_loss' : reconstruction_loss
    }
    
    return return_dict

def decoder_model(input_shape):
    
    node_a = Input(shape = input_shape)
    node_b = Input(shape = input_shape)
    
    X = Concatenate()([node_a, node_b])
    X = Dense(64, activation = 'relu')(X)
    X = Dense(1, activation = 'softmax')(X)
    
    return Model(inputs = [node_a, node_b], outputs = X)

def get_embed_perf(input_embed, input_dict, data = None, labels = None, graph = None, epochs = 200, hidden_edges = None, train_set = None, train_set_neg = None, test_set = None, test_set_neg = None):
    
    results = np.zeros((10, 1))
            
    # All Dimensions 
    all_train_acc, all_eval_acc, all_embed_dim, all_auc, all_aup  = get_link_perf(input_embed = input_embed,
                                                               data = data, 
                                                               labels = labels,
                                                               graph = graph,
                                                               hidden_edges = hidden_edges, 
                                                               train_set = train_set, 
                                                               train_set_neg = train_set_neg, 
                                                               test_set = test_set, 
                                                               test_set_neg = test_set_neg, 
                                                                                 epochs = epochs)

    # Important Dimensions 
    embed_imp = input_embed[:, input_dict['dimensions_idx_to_keep']]
    imp_train_acc, imp_eval_acc, imp_embed_dim, imp_auc, imp_aup = get_link_perf(input_embed = embed_imp,
                                                               data = data, 
                                                               labels = labels,
                                                               graph = graph,
                                                               hidden_edges = hidden_edges, 
                                                               train_set = train_set, 
                                                               train_set_neg = train_set_neg, 
                                                               test_set = test_set, 
                                                               test_set_neg = test_set_neg, 
                                                                                epochs = epochs)


    results[:, 0] = all_train_acc, all_eval_acc, all_embed_dim, all_aup, all_auc, imp_train_acc, imp_eval_acc, imp_embed_dim, imp_aup, imp_auc#, top_train_acc, top_eval_acc, top_embed_dim, top_aup, top_auc

    results = pd.DataFrame(results)
    results.index = ['Training Accuracy - All', 'Test Accuracy - All', 'Embedding Dimensions - All', 'AUP - All', 'AUC - All',
                       'Training Accuracy - Thresholded', 'Test Accuracy - Thresholded', 'Embedding Dimensions - Thresholded', 'AUP - Thresholded', 'AUC - Thresholded',]

    results.columns = ['Values']
    # display (results)
    return results

def get_link_perf(input_embed, graph = None, hidden_edges = None, data = None, labels = None, train_set = None, train_set_neg = None, test_set = None, test_set_neg = None, epochs = 200, learning_rate = 0.001, train_size = 0.7, display_results = False, return_model = False, random_state = 2021):
    
    
    embed_dim = input_embed.shape[1]
    
    if type(hidden_edges) == type(None):
        X_0 = np.zeros((data.shape[0], embed_dim))
        X_1 = np.zeros((data.shape[0], embed_dim))

        for idx in tqdm(range(len(data))): 

            node_0 = data[idx][0]
            node_1 = data[idx][1]

            X_0[idx, :] = input_embed[node_0]
            X_1[idx, :] = input_embed[node_1]

        Y = to_categorical(labels)

        X_0_train, X_0_test, X_1_train, X_1_test, y_train, y_test = train_test_split(X_0,
                                                                                     X_1,
                                                                                     Y,
                                                                                     train_size = train_size,
                                                                                     shuffle = True, 
                                                                                     random_state = random_state)
    else: 
        
        X_0_train, X_0_test, X_1_train, X_1_test, y_train, y_test = generate_link_data(input_embed = input_embed,
                                                                                       train_set = train_set,
                                                                                       train_set_neg = test_set_neg,
                                                                                       test_set = test_set,
                                                                                       test_set_neg = test_set_neg)
    
    model = decoder_model(input_shape = (embed_dim,))
    model.compile(loss = tf.keras.losses.binary_crossentropy,
                  optimizer = tf.keras.optimizers.Adam(learning_rate),
                  metrics = ["accuracy"])
    
    history = model.fit([X_0_train, X_1_train], y_train, epochs = epochs)
    eval_loss, eval_acc = model.evaluate([X_0_test, X_1_test], y_test)
    
    train_acc = history.history['accuracy'][-1]
    
    y_pred = model.predict([X_0_test, X_1_test])
    auc = roc_auc_score(y_test, y_pred)
    aup = average_precision_score(y_test, y_pred)
    
    if return_model:
        return train_acc, eval_acc, embed_dim, auc, aup, model
    
    return train_acc, eval_acc, embed_dim, auc, aup
    
    
def generate_link_data(input_embed, train_set, train_set_neg, test_set, test_set_neg):
    
    train_set = np.array(train_set)
    train_set_neg = np.array(train_set_neg)
    test_set = np.array(test_set)
    test_set_neg = np.array(test_set_neg)
    
    train_data = np.vstack((train_set, train_set_neg))
    train_labels = np.vstack((np.ones((train_set.shape[0], 1)), np.zeros((train_set_neg.shape[0], 1))))

    test_data = np.vstack((np.array(test_set), test_set_neg))
    test_labels = np.vstack((np.ones((len(test_set), 1)), np.zeros((test_set_neg.shape[0], 1))))
    
    # Put into right format 
    embed_dim = input_embed.shape[1]
    X_0_train = np.zeros((train_data.shape[0], embed_dim))
    X_1_train = np.zeros((train_data.shape[0], embed_dim))

    X_0_test = np.zeros((test_data.shape[0], embed_dim))
    X_1_test = np.zeros((test_data.shape[0], embed_dim))

    for idx in tqdm(range(len(train_data))): 

        node_0 = train_data[idx][0]
        node_1 = train_data[idx][1]

        X_0_train[idx, :] = input_embed[node_0]
        X_1_train[idx, :] = input_embed[node_1]

    for idx in tqdm(range(len(test_data))): 

        node_0 = test_data[idx][0]
        node_1 = test_data[idx][1]

        X_0_test[idx, :] = input_embed[node_0]
        X_1_test[idx, :] = input_embed[node_1]

    Y_train = to_categorical(train_labels)
    Y_test = to_categorical(test_labels)

    print ("Train Data : ", train_data.shape)
    print ("Test Data : ", test_data.shape) 

    print ("X0 Train: ", X_0_train.shape)
    print ("X1 Train: ", X_1_train.shape)
    print ("X0 Test: ", X_0_test.shape)
    print ("X1 Test: ", X_1_test.shape)
    print ("Y Train: ", Y_train.shape)
    print ("Y Test: ", Y_test.shape)
    
    return X_0_train, X_0_test, X_1_train, X_1_test, Y_train, Y_test


def preprocess_nxgraph(graph):
    node2idx = {}
    idx2node = []
    node_size = 0
    for node in graph.nodes():
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1
    return idx2node, node2idx

def l_2nd(beta):
    def loss_2nd(y_true, y_pred):
        
        b_ = (tf.cast((y_true > 0), tf.float32) * beta)
        x = K.square((y_true - y_pred) * b_)
        t = K.sum(x, axis = -1, )
        return K.mean(t)

    return loss_2nd


def l_1st_plus(alpha):
    def loss_1st(y_true, y_pred):
        
        L = y_true
        Y = y_pred

        batch_size = tf.cast(K.shape(L)[0], np.float32)
        l_1 = alpha * 2 * tf.linalg.trace(tf.matmul(tf.matmul(Y, L, transpose_a = True), Y)) / batch_size
        
        return l_1

    return loss_1st

def l_ortho(gamma, embed_dim):
    
    def loss_3rd(y_true, y_pred):
        
        E = y_pred
        A = y_true
        
        batch_size = tf.cast(K.shape(A)[0], np.float32)
        
        return gamma * E / batch_size
    
    return loss_3rd
    

def l_sparse(delta):
    
    def loss_4th(y_true, y_pred):
        
        E = y_pred
        sense = y_true     
        batch_size = tf.cast(K.shape(E)[0], np.float32)
        
        return delta * tf.reduce_sum(tf.norm(E, ord = 1, axis = 0)) / batch_size
    
    return loss_4th
    

def create_model_plus(node_size, sense_feat_size, hidden_size = [256, 128], l1 = 1e-5, l2 = 1e-4):
    
    A = Input(shape = (node_size,))
    A_2 = Input(shape = (None,))
    L = Input(shape = (None,))
    sense = Input(shape = (sense_feat_size, ))
    
    
    fc = A
    for i in range(len(hidden_size)):
        if i == len(hidden_size) - 1:
            fc = Dense(hidden_size[i], activation = 'relu',
                       kernel_regularizer = l1_l2(l1, l2), name = '1st')(fc)
        else:
            fc = Dense(hidden_size[i], activation = 'relu',
                       kernel_regularizer = l1_l2(l1, l2))(fc)
            
    fc = tf.clip_by_value(fc, clip_value_min = 1e-10, clip_value_max = tf.math.reduce_max(fc), name = '1st')
    Y = fc
    
    ####
    sense_mat = tf.einsum('ij, ik -> ijk', Y, sense)
    E = sense_mat
    y_norm = tf.linalg.diag_part(tf.matmul(Y, Y, transpose_b = True), k = 0)
    sense_norm = tf.linalg.diag_part(tf.matmul(sense, sense, transpose_b = True), k = 0)
    norm = tf.multiply(y_norm, sense_norm)
    E = tf.transpose(tf.transpose(E) / norm)
    E = (E - tf.reshape(tf.reduce_min(E, axis = [-1, -2]), (-1, 1, 1))) / tf.reshape(tf.reduce_max(E, axis = [-1, -2]) - tf.reduce_min(E, axis = [-1, -2]), (-1, 1, 1))
    
    E_t = tf.transpose(E, perm = [0, 2, 1]) 
    E_1 = tf.einsum('aij, ajh -> aih', E, E_t)

    E_1 = tf.reduce_sum(E_1)
    
    
    E_2 = tf.multiply(1.0, E, name = 'sparse_loss')
    ####
    
    
    for i in reversed(range(len(hidden_size) - 1)):
        fc = Dense(hidden_size[i], activation = 'relu',
                   kernel_regularizer = l1_l2(l1, l2))(fc)

    A_ = Dense(node_size, 'relu', name = '2nd')(fc)
        
    model = Model(inputs = [A, L, A_2, sense], outputs = [A_, Y, E_1, E_2])
    emb = Model(inputs = A, outputs = Y)
    return model, emb


###########################################
######## Embedding Methods - SDNE #########
###########################################

class SDNE_plus(object):
    def __init__(self, graph, sense_features, lr = 1e-5, hidden_size = [32, 16], alpha = 1e-6, beta = 5., gamma = 0.1, delta = 0.1, nu1 = 1e-5, nu2 = 1e-4):

        self.graph = graph
        self.idx2node, self.node2idx = preprocess_nxgraph(self.graph)

        self.node_size = self.graph.number_of_nodes()
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.nu1 = nu1
        self.nu2 = nu2
        self.sense_features = sense_features
        self.lr = lr
        

        self.A, self.L = self._create_A_L(
            self.graph, self.node2idx)  # Adj Matrix, L Matrix
        self.reset_model()
        self.inputs = [self.A, self.L]
        self._embeddings = {}
        

    def reset_model(self, opt = 'adam'):

        self.model, self.emb_model = create_model_plus(self.node_size,
                                                      hidden_size = self.hidden_size,
                                                      sense_feat_size = self.sense_features.shape[1],
                                                      l1 = self.nu1,
                                                      l2 = self.nu2)

        opt = Nadam(learning_rate = self.lr)

        self.model.compile(opt,
                           [l_2nd(self.beta),
                            l_1st_plus(self.alpha),
                            l_ortho(self.gamma, self.hidden_size[-1]),
                            l_sparse(self.delta),
                           ])
        self.get_embeddings()

    def train(self, batch_size = 1, epochs = 1, initial_epoch = 0, verbose = 1):
                
        if batch_size >= self.node_size:
            if batch_size > self.node_size:
                print('batch_size({0}) > node_size({1}),set batch_size = {1}'.format(
                    batch_size, self.node_size))
                batch_size = self.node_size
            return self.model.fit([self.A.todense(), self.L.todense(), self.A.todense(), self.sense_features],
                                  [self.A.todense(), self.L.todense(), self.A.todense(), self.sense_features],
                                  batch_size = batch_size, epochs = epochs, initial_epoch = initial_epoch, verbose = verbose,
                                  shuffle=False, )
        else:
            steps_per_epoch = (self.node_size - 1) // batch_size + 1
            hist = History()
            hist.on_train_begin()
            logs = {}
            for epoch in range(initial_epoch, epochs):
                start_time = time.time()
                losses = np.zeros(5)
                for i in range(steps_per_epoch):
                    index = np.arange(
                        i * batch_size, min((i + 1) * batch_size, self.node_size))
                    A_train = self.A[index, :].todense()
                    A_sub = self.A[index, :]
                    A_sub = A_sub[:, index].todense()
                    L_mat_train = self.L[index][:, index].todense()
                                        
                    inp = [A_train, L_mat_train, A_sub, self.sense_features[index, :]]
                    oup = [A_train, L_mat_train, A_sub, self.sense_features[index, :]]
                    
                    batch_losses = self.model.train_on_batch(inp, oup)
                    losses += batch_losses
                losses = losses / steps_per_epoch

                logs['loss'] = losses[0]
                logs['2nd_loss'] = losses[1]
                logs['1st_loss'] = losses[2]
                logs['sparse_loss'] = losses[3]
                logs['ortho_loss'] = losses[4]
                epoch_time = int(time.time() - start_time)
                #hist.on_epoch_end(epoch, logs)
                if verbose > 0:
                    print('Epoch {0}/{1}'.format(epoch + 1, epochs))
                    print('{0}s - loss: {1: .4f} - 2nd_loss: {2: .4f} - 1st_loss: {3: .4f} - ortho_loss : {4: .4f} - sparse_loss : {5: .4f}'.format(
                        epoch_time, losses[0], losses[1], losses[2], losses[3], losses[4]))
            return hist

    def evaluate(self, ):
        return self.model.evaluate(x = self.inputs, y = self.inputs, batch_size = self.node_size)

    def get_embeddings(self):
        self._embeddings = {}
        
        dense = self.A
        
        batch_size = dense.shape[0] // 10
        
        embeddings_1 = self.emb_model.predict(dense[:1 * batch_size].todense(), batch_size = batch_size)
        embed_list = []
        embed_list.append(embeddings_1)
        for idx in range(1, 9):
            embed_list.append(self.emb_model.predict(dense[idx * batch_size:(idx + 1) * batch_size].todense(), batch_size = batch_size))
        embeddings_n = embed_list.append(self.emb_model.predict(dense[9 * batch_size:].todense(), batch_size = batch_size))
        embeddings = np.vstack(embed_list)

        assert embeddings.shape[0] == dense.shape[0]

        look_back = self.idx2node
        for i, embedding in enumerate(embeddings):
            self._embeddings[look_back[i]] = embedding

        return self._embeddings

    def _create_A_L(self, graph, node2idx):
        node_size = graph.number_of_nodes()
        A_data = []
        A_row_index = []
        A_col_index = []

        for edge in graph.edges():
            v1, v2 = edge
            edge_weight = graph[v1][v2].get('weight', 1)

            A_data.append(edge_weight)
            A_row_index.append(node2idx[v1])
            A_col_index.append(node2idx[v2])

        A = sp.csr_matrix((A_data, (A_row_index, A_col_index)), shape = (node_size, node_size))
        A_ = sp.csr_matrix((A_data + A_data, (A_row_index + A_col_index, A_col_index + A_row_index)),
                           shape=(node_size, node_size))

        D = sp.diags(A_.sum(axis=1).flatten().tolist()[0])
        L = D - A_
        return A, L

def preprocess_nxgraph(graph):
    node2idx = {}
    idx2node = []
    node_size = 0
    for node in graph.nodes():
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1
    return idx2node, node2idx


def main_loss(alpha):
    def line_loss(y_true, y_pred):
        return alpha * -K.mean(K.log(K.sigmoid(y_true * y_pred)))
    return line_loss

def l_ortho_line(gamma):
    
    def loss_3rd(y_true, y_pred):
        
        E = y_pred
        A = y_true
        
        batch_size = tf.cast(K.shape(A)[0], np.float32)
    
        return gamma * E / batch_size

    
    return loss_3rd

def l_sparse_line(delta):
    
    def loss_4th(y_true, y_pred):
        
        E = y_pred
        sense = y_true
                
        batch_size = tf.cast(K.shape(E)[0], np.float32)
            
        return delta * tf.reduce_sum(tf.norm(E, ord = 1, axis = 0)) / batch_size
    
    return loss_4th

def create_model_line(num_nodes, embedding_size, sense_feat_size, order = 'second', batch_size = 128):

    v_i = Input(shape = (1,))
    v_j = Input(shape = (1,))
    adj = Input(shape = (None, ))
    sense_i = Input(shape = (sense_feat_size, ))
    

    first_emb = Embedding(num_nodes, embedding_size, name = 'first_emb')
    second_emb = Embedding(num_nodes, embedding_size, name = 'second_emb')
    context_emb = Embedding(num_nodes, embedding_size, name = 'context_emb')

    v_i_emb = first_emb(v_i)
    v_j_emb = first_emb(v_j)

    v_i_emb_second = second_emb(v_i)
    v_j_context_emb = context_emb(v_j)

    ### First Embed ###
    first = Lambda(lambda x: tf.reduce_sum(x[0] * x[1],
                                               axis = -1,
                                               keepdims = False),
                       name = 'first_order')([v_i_emb, v_j_emb])
    if order == 'first':
    
        first_embed = Reshape((embedding_size,), name = 'ortho_1')(v_i_emb)
        sense_mat = tf.einsum('ij, ik -> ijk', first_embed, sense_i)
        E = sense_mat
        y_norm = tf.linalg.diag_part(tf.matmul(first_embed, first_embed, transpose_b = True), k = 0)
        sense_norm = tf.linalg.diag_part(tf.matmul(sense_i, sense_i, transpose_b = True), k = 0)
        norm = tf.multiply(y_norm, sense_norm)
        E = tf.transpose(tf.transpose(E) / norm)
        E = (E - tf.reshape(tf.reduce_min(E, axis = [-1, -2]), (-1, 1, 1))) / tf.reshape(tf.reduce_max(E, axis = [-1, -2]) - tf.reduce_min(E, axis = [-1, -2]), (-1, 1, 1))


    
    ### Second Embed ###
    second = Lambda(lambda x: tf.reduce_sum(x[0] * x[1],
                                                axis = -1,
                                                keepdims = False),
                        name = 'second_order')([v_i_emb_second, v_j_context_emb])
    if order == 'second':
        
        second_embed = Reshape((embedding_size,), name = 'ortho_2')(v_i_emb_second)

        sense_mat = tf.einsum('ij, ik -> ijk', second_embed, sense_i)
        E = sense_mat
        y_norm = tf.linalg.diag_part(tf.matmul(second_embed, second_embed, transpose_b = True), k = 0)
        sense_norm = tf.linalg.diag_part(tf.matmul(sense_i, sense_i, transpose_b = True), k = 0)
        norm = tf.multiply(y_norm, sense_norm)
        E = tf.transpose(tf.transpose(E) / norm)
        E = (E - tf.reshape(tf.reduce_min(E, axis = [-1, -2]), (-1, 1, 1))) / tf.reshape(tf.reduce_max(E, axis = [-1, -2]) - tf.reduce_min(E, axis = [-1, -2]), (-1, 1, 1))

    
    ### Loss Computations
    E_t = tf.transpose(E, perm = [0, 2, 1])

    E_1 = tf.einsum('aij, ajh -> aih', E, E_t)
    E_1 = tf.reduce_sum(E_1)
    
    E_2 = tf.multiply(1.0, E, name = 'sparse_loss')
    
    ####

    if order == 'first':
        output_list = [first_embed, E_1, E_2]
    
    elif order == 'second':
        output_list = [second_embed, E_1, E_2]
    
    else:
        output_list = [first_embed, second_embed, [E_1, E_2], [E_1, E_2]]

    model = Model(inputs = [v_i, v_j, adj, sense_i], outputs = output_list)

    return model, {'first': first_emb, 'second': second_emb}

###########################################
######## Embedding Methods - LINE #########
###########################################

class LINE:
    def __init__(self, graph, sense_features, alpha, ortho, sparse, learning_rate, batch_size, embedding_size = 8, negative_ratio = 5, order = 'second',):
        """
        :param graph:
        :param embedding_size:
        :param negative_ratio:
        :param order: 'first','second','all'
        """
        if order not in ['first', 'second', 'all']:
            raise ValueError('mode must be first, second or all')

        self.graph = graph
        self.idx2node, self.node2idx = preprocess_nxgraph(graph)
        self.use_alias = True

        self.rep_size = embedding_size
        self.order = order
        self.sense_features = sense_features
        self.sense_feat_size = self.sense_features.shape[1]
        self.alpha = alpha
        self.gamma = ortho
        self.delta = sparse
        self.lr = learning_rate

        self._embeddings = {}
        self.negative_ratio = negative_ratio
        self.order = order
        self.batch_size = batch_size
        
        self.node_size = graph.number_of_nodes()
        self.edge_size = graph.number_of_edges()
        self.samples_per_epoch = self.edge_size * (1 + negative_ratio)

        self._gen_sampling_table()
        self.reset_model()
        
        
        self.A, self.L = self._create_A_L(
            self.graph, self.node2idx)  # Adj Matrix, L Matrix

    def reset_training_config(self, batch_size, times):
        self.batch_size = batch_size
        self.steps_per_epoch = (
            (self.samples_per_epoch - 1) // self.batch_size + 1) * times

    def reset_model(self, opt = 'adam'):

        self.model, self.embedding_dict = create_model_line(self.node_size,
                                                       self.rep_size, 
                                                       self.sense_feat_size,
                                                       self.order, 
                                                       self.batch_size)
        opt = Adam(learning_rate = self.lr, clipnorm = 0.5)
        self.model.compile(opt, [main_loss(self.alpha), l_ortho_line(self.gamma), l_sparse_line(self.delta)])
        self.batch_it = self.batch_iter(self.node2idx)

    def _gen_sampling_table(self):

        # create sampling table for vertex
        power = 0.75
        num_nodes = self.node_size
        node_degree = np.zeros(num_nodes)  # out degree
        node2idx = self.node2idx

        for edge in self.graph.edges():
            node_degree[node2idx[edge[0]]] += self.graph[edge[0]][edge[1]].get('weight', 1.0)

        total_sum = sum([math.pow(node_degree[i], power)
                         for i in range(num_nodes)])
        norm_prob = [float(math.pow(node_degree[j], power)) /
                     total_sum for j in range(num_nodes)]

        self.node_accept, self.node_alias = create_alias_table(norm_prob)

        # create sampling table for edge
        numEdges = self.graph.number_of_edges()
        total_sum = sum([self.graph[edge[0]][edge[1]].get('weight', 1.0)
                         for edge in self.graph.edges()])
        norm_prob = [self.graph[edge[0]][edge[1]].get('weight', 1.0) *
                     numEdges / total_sum for edge in self.graph.edges()]

        self.edge_accept, self.edge_alias = create_alias_table(norm_prob)

    def batch_iter(self, node2idx):

        edges = [(node2idx[x[0]], node2idx[x[1]]) for x in self.graph.edges()]

        data_size = self.graph.number_of_edges()
        shuffle_indices = np.random.permutation(np.arange(data_size))
        # positive or negative mod
        mod = 0
        mod_size = 1 + self.negative_ratio
        h = []
        t = []
        sign = 0
        count = 0
        start_index = 0
        end_index = min(start_index + self.batch_size, data_size)
        while True:
            if mod == 0:

                h = []
                t = []
                for i in range(start_index, end_index):
                    if random.random() >= self.edge_accept[shuffle_indices[i]]:
                        shuffle_indices[i] = self.edge_alias[shuffle_indices[i]]
                    cur_h = edges[shuffle_indices[i]][0]
                    cur_t = edges[shuffle_indices[i]][1]
                    h.append(cur_h)
                    t.append(cur_t)
                sign = np.ones(len(h))
            else:
                sign = np.ones(len(h))*-1
                t = []
                for i in range(len(h)):

                    t.append(alias_sample(
                        self.node_accept, self.node_alias))
                    
            sense_feats = self.sense_features[np.array(h)]
            adj = self.A[np.array(h), :]
            adj = adj[:, np.array(h)].todense()
            #assert adj.shape == (self.batch_size, self.batch_size)
            
            if self.order == 'all':
                yield ([np.array(h), np.array(t), adj, sense_feats],
                       [sign, sign, sense_feats, sense_feats])
            else:
                yield ([np.array(h), np.array(t), adj, sense_feats],
                       [sign, sense_feats, sense_feats])
            mod += 1
            mod %= mod_size
            if mod == 0:
                start_index = end_index
                end_index = min(start_index + self.batch_size, data_size)

            if start_index >= data_size:
                count += 1
                mod = 0
                h = []
                shuffle_indices = np.random.permutation(np.arange(data_size))
                start_index = 0
                end_index = min(start_index + self.batch_size, data_size)

    def get_embeddings(self,):
        self._embeddings = {}
        if self.order == 'first':
            embeddings = self.embedding_dict['first'].get_weights()[0]
        elif self.order == 'second':
            embeddings = self.embedding_dict['second'].get_weights()[0]
        else:
            embeddings = np.hstack((self.embedding_dict['first'].get_weights()[
                                   0], self.embedding_dict['second'].get_weights()[0]))
        idx2node = self.idx2node
        for i, embedding in enumerate(embeddings):
            self._embeddings[idx2node[i]] = embedding

        return self._embeddings

    def train(self, epochs = 1, initial_epoch = 0, verbose = 1, times = 1):
        batch_size = self.batch_size
        self.reset_training_config(batch_size, times)
        hist = self.model.fit(self.batch_it,
                                        epochs = epochs,
                                        initial_epoch = initial_epoch,
                                        steps_per_epoch = self.steps_per_epoch,
                                        verbose = verbose)

        return hist
    
    def _create_A_L(self, graph, node2idx):
        node_size = graph.number_of_nodes()
        A_data = []
        A_row_index = []
        A_col_index = []

        for edge in graph.edges():
            v1, v2 = edge
            edge_weight = graph[v1][v2].get('weight', 1)

            A_data.append(edge_weight)
            A_row_index.append(node2idx[v1])
            A_col_index.append(node2idx[v2])

        A = sp.csr_matrix((A_data, (A_row_index, A_col_index)), shape = (node_size, node_size))
        A_ = sp.csr_matrix((A_data + A_data, (A_row_index + A_col_index, A_col_index + A_row_index)),
                           shape=(node_size, node_size))

        D = sp.diags(A_.sum(axis=1).flatten().tolist()[0])
        L = D - A_
        return A, L

def create_alias_table(area_ratio):
    """
    :param area_ratio: sum(area_ratio)=1
    :return: accept,alias
    """
    l = len(area_ratio)
    accept, alias = [0] * l, [0] * l
    small, large = [], []
    area_ratio_ = np.array(area_ratio) * l
    for i, prob in enumerate(area_ratio_):
        if prob < 1.0:
            small.append(i)
        else:
            large.append(i)

    while small and large:
        small_idx, large_idx = small.pop(), large.pop()
        accept[small_idx] = area_ratio_[small_idx]
        alias[small_idx] = large_idx
        area_ratio_[large_idx] = area_ratio_[large_idx] - \
                                 (1 - area_ratio_[small_idx])
        if area_ratio_[large_idx] < 1.0:
            small.append(large_idx)
        else:
            large.append(large_idx)

    while large:
        large_idx = large.pop()
        accept[large_idx] = 1
    while small:
        small_idx = small.pop()
        accept[small_idx] = 1

    return accept, alias


def alias_sample(accept, alias):
    """
    :param accept:
    :param alias:
    :return: sample index
    """
    N = len(accept)
    i = int(np.random.random() * N)
    r = np.random.random()
    if r < accept[i]:
        return i
    else:
        return alias[i]


###########################################
######## Embedding Methods - DGI ##########
###########################################
from DGI.models import DGI, LogReg
from DGI.utils import process
class BaseEmbedder:
    def __init__(self, graph, embed_shape = (128,)):
        self.embed(graph)
        self.E = list(graph.edges())
        self.graph = graph
        self.embed_shape = embed_shape
    
    def embed(self, graph):
        raise NotImplementedError
    
    def get_embedding(self):
        raise NotImplementedError

class DGIEmbedding(BaseEmbedder):
    def __init__(self, embed_dim = 64, graph = None, feature_matrix = None, use_xm = False, debug = False, batch_size = 1, nb_epochs = 2500, patience = 20, ortho_ = 0.1, sparse_ = 0.1, lr = 1e-3, l2_coef = 0.0, drop_prob = 0.0, sparse = True, nonlinearity = 'prelu', model_name = ''):

        self.embed_dim = embed_dim
        self.debug = debug
        
        # Training Params
        self.graph = graph
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.patience = patience
        self.lr = lr
        self.l2_coef = l2_coef
        self.feature_matrix = feature_matrix
        self.drop_prob = drop_prob
        self.hid_units = embed_dim
        self.sparse = sparse
        self.nonlinearity = nonlinearity
        self.use_xm = use_xm
        self.ortho_ = ortho_
        self.sparse_ = sparse_
        self.model_name = model_name

        self.time_per_epoch = None
        
        if graph is not None:
            self.embed()
        else:
            self.graph = None
    
    def embed(self):

        
        if self.feature_matrix is None:
            feature_matrix = np.identity(len(self.graph))
        else: 
            feature_matrix = self.feature_matrix

        adj = nx.to_scipy_sparse_array(self.graph)
        features = sp.lil_matrix(feature_matrix)
        features, _ = process.preprocess_features(features)

        nb_nodes = features.shape[0]
        ft_size = features.shape[1]

        adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

        if self.sparse:
            sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
        else:
            adj = (adj + sp.eye(adj.shape[0])).todense()

        features = torch.FloatTensor(features[np.newaxis])
        if not self.sparse:
            adj = torch.FloatTensor(adj[np.newaxis])
            if torch.cuda.is_available():
                adj = adj.cuda()

        if self.feature_matrix is not None: 
            sense_features = torch.FloatTensor(self.feature_matrix)
            if torch.cuda.is_available():
                sense_features = sense_features.cuda()


        model = DGI(ft_size, self.hid_units, self.nonlinearity)
        if torch.cuda.is_available():
            model = model.cuda()
        optimiser = torch.optim.Adam(model.parameters(), lr = self.lr, weight_decay = self.l2_coef)

        b_xent = nn.BCEWithLogitsLoss()
        xent = nn.CrossEntropyLoss()
        cnt_wait = 0
        best = 1e9
        best_t = 0
        
        start_time = time.time()
        for epoch in tqdm(range(self.nb_epochs)):
            model.train()
            optimiser.zero_grad()

            idx = np.random.permutation(nb_nodes)
            shuf_fts = features[:, idx, :]

            lbl_1 = torch.ones(self.batch_size, nb_nodes)
            lbl_2 = torch.zeros(self.batch_size, nb_nodes)
            lbl = torch.cat((lbl_1, lbl_2), 1)

            if torch.cuda.is_available():
                shuf_fts = shuf_fts.cuda()
                lbl = lbl.cuda()
                sp_adj = sp_adj.cuda()
                features = features.cuda()
            
            logits = model(features, shuf_fts, sp_adj if self.sparse else adj, self.sparse, None, None, None) 
            
            if self.use_xm == True and feature_matrix is not None:
                
                start_idx = 0
                loop = True
                
                ortho_loss = 0
                sparse_loss = 0
                xm_batch_size = 128
                
                sf = sense_features
                embeds, _ = model.embed(sf, sp_adj if self.sparse else adj, self.sparse, None)
                                
                while loop:
                    end_idx = start_idx + xm_batch_size
                    if end_idx > len(self.graph):
                        loop = False
                        end_idx = len(self.graph)
                        
                    
                    sf = sense_features[start_idx : end_idx]
                    embeds_ = torch.squeeze(embeds)[start_idx : end_idx]
                    
                    
                    sense_mat = torch.einsum('ij, ik -> ijk', embeds_, sf)
                    E = sense_mat
                    y_norm = torch.diagonal(torch.matmul(embeds_, torch.transpose(embeds_, 0, 1)))
                    sense_norm = torch.diagonal(torch.matmul(sf, torch.transpose(sf, 0, 1)))
                    norm = torch.multiply(y_norm, sense_norm)
                    E = torch.transpose(torch.transpose(E, 0, 2) / norm, 0, 2)
                    E = (E - torch.amin(E, dim = [-1, -2], keepdim = True)) / (torch.amax(E, dim = [-1, -2], keepdim = True) - torch.amin(E, dim = [-1, -2], keepdim = True))

                    E_t = torch.transpose(E, 1, 2)
                    E_o = torch.einsum('aij, ajh -> aih', E, E_t)
                    E_o = torch.sum(E_o)
                    batch_ortho_loss = (self.ortho_ * E_o) / self.batch_size

                    batch_sparse_loss = (self.sparse_ * torch.sum(torch.linalg.norm(E, ord = 1, axis = 0))) / self.batch_size
                        
                    ortho_loss += batch_ortho_loss
                    sparse_loss += batch_sparse_loss
                    
                    start_idx = end_idx
                    
                loss = b_xent(logits, lbl) + ortho_loss + sparse_loss
            else:
                loss = b_xent(logits, lbl)

            if self.debug:
                print('Loss:', loss)

            if loss < best:
                best = loss
                best_t = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), self.model_name + '.pkl')
            else:
                cnt_wait += 1

            if cnt_wait == self.patience:
                if self.debug: 
                    print('Early stopping!')
                break

            loss.backward()
            optimiser.step()
            
        self.time_per_epoch = (time.time() - start_time) / epoch

        if self.debug: 
            print('Loading {}th epoch'.format(best_t))
        model.load_state_dict(torch.load(self.model_name + '.pkl'))

        self.node_model = model
        self.fitted = True

        embeds, _ = model.embed(features, sp_adj if self.sparse else adj, self.sparse, None)
        self.embeddings = embeds
    
    def get_embedding(self):
        if torch.cuda.is_available():
            return np.squeeze(self.embeddings.cpu().numpy())
        return np.squeeze(self.embeddings.numpy())
    





###########################################
######## Embedding Methods - GMI ##########
###########################################
from GMI_.models import GMI, LogReg
from GMI_.utils import process
class GMIEmbedding(BaseEmbedder):
    def __init__(self, embed_dim = 64, graph = None, feature_matrix = None, use_xm = False, debug = False, batch_size = 1, nb_epochs = 500, patience = 20, ortho_ = 0.1, sparse_ = 0.1, lr = 1e-3, l2_coef = 0.0, drop_prob = 0.0, sparse = True, nonlinearity = 'prelu', alpha = 0.8, beta = 1.0, gamma = 1.0, negative_num = 5, epoch_flag = 20, model_name = 'test'):

        self.embed_dim = embed_dim
        self.debug = debug
        
        # Training Params
        self.graph = graph
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.patience = patience
        self.lr = lr
        self.l2_coef = l2_coef
        self.feature_matrix = feature_matrix
        self.drop_prob = drop_prob
        self.hid_units = embed_dim
        self.sparse = sparse
        self.nonlinearity = nonlinearity
        self.use_xm = use_xm
        self.ortho_ = ortho_
        self.sparse_ = sparse_
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.negative_num = negative_num
        self.epoch_flag = epoch_flag
        self.model_name = model_name
        
        self.time_per_epoch = None
        
        if graph is not None:
            self.embed()
        else:
            self.graph = None
    
    def embed(self):

        ####
        if self.feature_matrix is None:
            feature_matrix = np.identity(len(self.graph))
        else: 
            feature_matrix = self.feature_matrix

        adj_ori = nx.to_scipy_sparse_array(self.graph)
        features = sp.lil_matrix(self.feature_matrix)
        features, _ = process.preprocess_features(features)

        nb_nodes = features.shape[0]
        ft_size = features.shape[1]
        
        adj = process.normalize_adj(adj_ori + sp.eye(adj_ori.shape[0]))

        if self.sparse:
            sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
        else:
            adj = (adj + sp.eye(adj.shape[0])).todense()

        features = torch.FloatTensor(features[np.newaxis])
        if not self.sparse:
            adj = torch.FloatTensor(adj[np.newaxis])
            if torch.cuda.is_available():
                adj = adj.cuda()

        if self.feature_matrix is not None: 
            sense_features = torch.FloatTensor(self.feature_matrix)
            if torch.cuda.is_available():
                sense_features = sense_features.cuda()

        model = GMI(ft_size, self.hid_units, self.nonlinearity)
        optimiser = torch.optim.Adam(model.parameters(), lr = self.lr, weight_decay = self.l2_coef)
        
        if self.use_xm:
             model.load_state_dict(torch.load(self.model_name + '.pkl'))
        
        if torch.cuda.is_available():
            model = model.cuda()
            features = features.cuda()
            sp_adj = sp_adj.cuda()
            
        b_xent = nn.BCEWithLogitsLoss()
        xent = nn.CrossEntropyLoss()
        cnt_wait = 0
        best = 1e9
        best_t = 0
        
        
        adj_dense = adj_ori.toarray()
        adj_target = adj_dense + np.eye(adj_dense.shape[0])
        adj_row_avg = 1.0 / np.sum(adj_dense, axis = 1)
        adj_row_avg[np.isnan(adj_row_avg)] = 0.0
        adj_row_avg[np.isinf(adj_row_avg)] = 0.0
        adj_dense = adj_dense * 1.0
        
        for i in range(adj_ori.shape[0]):
            adj_dense[i] = adj_dense[i] * adj_row_avg[i]
        adj_ori = sp.csr_matrix(adj_dense, dtype = np.float32)

        start_time = time.time()
        for epoch in tqdm(range(self.nb_epochs)):
            model.train()
            optimiser.zero_grad()
            
            res = model(features, adj_ori, self.negative_num, sp_adj, None, None) 
            
            if self.use_xm == True and feature_matrix is not None:
                
                start_idx = 0
                loop = True
                
                ortho_loss = 0
                sparse_loss = 0
                xm_batch_size = 128
                
                sf = sense_features
                embeds = model.embed(features, sp_adj)
                                
                while loop:
                    end_idx = start_idx + xm_batch_size
                    if end_idx > len(self.graph):
                        loop = False
                        end_idx = len(self.graph)
                                            
                    sf = sense_features[start_idx : end_idx]
                    embeds_ = torch.squeeze(embeds)[start_idx : end_idx]
                    
                    
                    sense_mat = torch.einsum('ij, ik -> ijk', embeds_, sf)
                    E = sense_mat
                    y_norm = torch.diagonal(torch.matmul(embeds_, torch.transpose(embeds_, 0, 1)))
                    sense_norm = torch.diagonal(torch.matmul(sf, torch.transpose(sf, 0, 1)))
                    norm = torch.multiply(y_norm, sense_norm)
                    E = torch.transpose(torch.transpose(E, 0, 2) / norm, 0, 2)
                    E = (E - torch.amin(E, dim = [-1, -2], keepdim = True)) / (torch.amax(E, dim = [-1, -2], keepdim = True) - torch.amin(E, dim = [-1, -2], keepdim = True))
                    
                    E_t = torch.transpose(E, 1, 2)
                    
                    E_o = torch.einsum('aij, ajh -> aih', E, E_t)
                    E_o = torch.sum(E_o)
                    batch_ortho_loss = (self.ortho_ * E_o) / self.batch_size

                    batch_sparse_loss = (self.sparse_ * torch.sum(torch.linalg.norm(E, ord = 1, axis = 0))) / self.batch_size
                        
                    ortho_loss += batch_ortho_loss
                    sparse_loss += batch_sparse_loss
                    
                    start_idx = end_idx
                    
                loss = self.alpha * process.mi_loss_jsd(res[0], res[1]) +\
                       self.beta * process.mi_loss_jsd(res[2], res[3]) +\
                       self.gamma * process.reconstruct_loss(res[4], adj_target) +\
                       ortho_loss +\
                       sparse_loss
            else:
                loss = self.alpha * process.mi_loss_jsd(res[0], res[1]) +\
                       self.beta * process.mi_loss_jsd(res[2], res[3]) +\
                       self.gamma * process.reconstruct_loss(res[4], adj_target)


            if self.debug:
                print('Loss:', loss)

            if loss < best:
                best = loss
                best_t = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), self.model_name + '.pkl')
            else:
                cnt_wait += 1

            if cnt_wait == self.epoch_flag:
                print('Early stopping!')
                break

            loss.backward()
            optimiser.step()
            
        self.time_per_epoch = (time.time() - start_time) / epoch

        if self.debug: 
            print('Loading {}th epoch'.format(best_t))
            
        model.load_state_dict(torch.load(self.model_name + '.pkl'))

        embeds = model.embed(features, sp_adj)
        self.embeddings = embeds
    
    def get_embedding(self):
        return np.squeeze(self.embeddings.numpy())
    