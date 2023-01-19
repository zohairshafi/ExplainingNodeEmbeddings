from utils import *

def decoder_model(input_shape):
    
    node_a = Input(shape = input_shape)
    node_b = Input(shape = input_shape)
    
    X = Concatenate()([node_a, node_b])
    X = Dense(64, activation = 'relu')(X)
    X = Dense(64, activation = 'relu')(X)
    X = Dense(1, activation = 'sigmoid')(X)
    
    return Model(inputs = [node_a, node_b], outputs = X)

def link_preidction(graph, embed_og, embed_plus, epochs):
    
    ig = igraph.Graph([[e[0], e[1]] for e in nx.to_edgelist(graph)])
    pos_edges = set(ig.get_edgelist())
    neg_indices = np.where(nx.to_numpy_array(graph) == 0)
    
    train_size = int(0.6 * len(pos_edges))
    test_size = int(0.4 * len(pos_edges))
    
    neg_edges = set()
    
    while len(neg_edges) <= 0.8 * len(pos_edges):

        idx = np.random.randint(neg_indices[0].shape[0])
        while neg_indices[0][idx] == neg_indices[1][idx]:
            idx = np.random.randint(neg_indices[0].shape[0])
        neg_edges.add((neg_indices[0][idx], neg_indices[1][idx]))
        print("Number of Negative Edges : " + str(len(neg_edges)), end = '\r')
        
    assert pos_edges.intersection(neg_edges) == set()
    
    pos_edges_idx = np.random.choice(range(len(pos_edges)), size = len(neg_edges), replace = False)

    pos_edges = np.array(list(pos_edges))[pos_edges_idx]
    neg_edges = np.array(list(neg_edges))
    
    assert pos_edges.shape[0] == neg_edges.shape[0]
    
    #####################################
    ######### Train On OG Embed #########
    #####################################
    embed = embed_og
    train_idx = np.random.choice(range(neg_edges.shape[0]), size = int(0.6 * len(neg_edges)), replace = False)
    test_idx = list(set(range(neg_edges.shape[0])).difference(set(train_idx)))
    assert set(train_idx).intersection(set(test_idx)) == set()

    # Placeholder Arrays
    train_node_0 = np.zeros((2 * train_idx.shape[0], embed.shape[1]))
    train_node_1 = np.zeros((2 * train_idx.shape[0], embed.shape[1]))

    test_node_0 = np.zeros((2 * len(test_idx), embed.shape[1]))
    test_node_1 = np.zeros((2 * len(test_idx), embed.shape[1]))

    # Train Data
    train_pos = pos_edges[train_idx]
    train_neg = neg_edges[train_idx]
    train_x = np.vstack([train_pos, train_neg])

    train_y_pos = np.ones((train_pos.shape[0], 1))
    train_y_neg = np.zeros((train_neg.shape[0], 1))
    train_y = np.vstack([train_y_pos, train_y_neg])

    for idx in tqdm(range(train_x.shape[0])):
        train_node_0[idx, :] = embed[train_x[idx][0]]
        train_node_1[idx, :] = embed[train_x[idx][1]]

    # Test Data
    test_pos = pos_edges[test_idx]
    test_neg = neg_edges[test_idx]
    test_x = np.vstack([test_pos, test_neg])

    test_y_pos = np.ones((test_pos.shape[0], 1))
    test_y_neg = np.zeros((test_neg.shape[0], 1))
    test_y = np.vstack([test_y_pos, test_y_neg])

    for idx in tqdm(range(test_x.shape[0])):
        test_node_0[idx, :] = embed[test_x[idx][0]]
        test_node_1[idx, :] = embed[test_x[idx][1]]

    # Class Balance
    assert train_pos.shape[0] == train_neg.shape[0]
    assert test_pos.shape[0] == test_neg.shape[0]
    
    model_og = decoder_model(input_shape = embed.shape[1])
    model_og.compile(loss = tf.keras.losses.binary_crossentropy,
                  optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4),
                  metrics = ["accuracy"])
    history = model_og.fit([train_node_0, train_node_1], train_y, epochs = epochs)
    _, og_acc = model_og.evaluate([test_node_0, test_node_1], test_y)
    og_pred = model_og.predict([test_node_0, test_node_1])
    og_auc = roc_auc_score(test_y, og_pred)
    og_aup = average_precision_score(test_y, og_pred)
    
    #####################################
    ######### Train On XM Embed #########
    #####################################
    embed = embed_plus
    train_idx = np.random.choice(range(neg_edges.shape[0]), size = int(0.6 * len(neg_edges)), replace = False)
    test_idx = list(set(range(neg_edges.shape[0])).difference(set(train_idx)))
    assert set(train_idx).intersection(set(test_idx)) == set()

    # Placeholder Arrays
    train_node_0 = np.zeros((2 * train_idx.shape[0], embed.shape[1]))
    train_node_1 = np.zeros((2 * train_idx.shape[0], embed.shape[1]))

    test_node_0 = np.zeros((2 * len(test_idx), embed.shape[1]))
    test_node_1 = np.zeros((2 * len(test_idx), embed.shape[1]))

    # Train Data
    train_pos = pos_edges[train_idx]
    train_neg = neg_edges[train_idx]
    train_x = np.vstack([train_pos, train_neg])

    train_y_pos = np.ones((train_pos.shape[0], 1))
    train_y_neg = np.zeros((train_neg.shape[0], 1))
    train_y = np.vstack([train_y_pos, train_y_neg])

    for idx in tqdm(range(train_x.shape[0])):
        train_node_0[idx, :] = embed[train_x[idx][0]]
        train_node_1[idx, :] = embed[train_x[idx][1]]

    # Test Data
    test_pos = pos_edges[test_idx]
    test_neg = neg_edges[test_idx]
    test_x = np.vstack([test_pos, test_neg])

    test_y_pos = np.ones((test_pos.shape[0], 1))
    test_y_neg = np.zeros((test_neg.shape[0], 1))
    test_y = np.vstack([test_y_pos, test_y_neg])

    for idx in tqdm(range(test_x.shape[0])):
        test_node_0[idx, :] = embed[test_x[idx][0]]
        test_node_1[idx, :] = embed[test_x[idx][1]]

    # Class Balance
    assert train_pos.shape[0] == train_neg.shape[0]
    assert test_pos.shape[0] == test_neg.shape[0]
    
    model_plus = decoder_model(input_shape = embed.shape[1])
    model_plus.compile(loss = tf.keras.losses.binary_crossentropy,
                  optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4),
                  metrics = ["accuracy"])
    history = model_plus.fit([train_node_0, train_node_1], train_y, epochs = epochs)
    _, plus_acc = model_plus.evaluate([test_node_0, test_node_1], test_y)
    plus_pred = model_plus.predict([test_node_0, test_node_1])
    plus_auc = roc_auc_score(test_y, plus_pred)
    plus_aup = average_precision_score(test_y, plus_pred)
    
    
    return {'plus_acc' : plus_acc, 
            'plus_auc' : plus_auc, 
            'plus_aup' : plus_aup, 
            'og_acc' : og_acc,
            'og_auc' : og_auc, 
            'og_aup' : og_aup}

######################################################################################################################################################

# with open('../data/fb.pkl', 'rb') as file: 
#     graph = pkl.load(file)
# graph = nx.Graph(nx.to_numpy_array(graph))

# fb_results = {}
# dim = 128

# ###################
# ####### DGI #######
# ###################
# with open('../results/fb_dgi.pkl', 'rb') as file:
#     results = pkl.load(file)
    
# lp_results = []
# for _ in tqdm(range(3)):
    
#     runs = len(results[dim]['norm_og'])
#     random_idx = np.random.randint(runs)
#     print ("Using Run #" + str(random_idx))
#     lp_dict = link_preidction(graph = graph,
#                               embed_og = results[dim]['embed_og'][random_idx],
#                               embed_plus = results[dim]['embed_plus'][random_idx],
#                               epochs = 100)
#     lp_results.append(list(lp_dict.values()))
    
# lp_results = np.array(lp_results)
# fb_results['dgi'] = lp_results

# dgi_norm = np.nanmean(results[dim]['norm_og'])
# dgi_norm_std = np.nanstd(results[dim]['norm_og'])

# dgi_xm_norm = np.nanmean(results[dim]['norm_plus'])
# dgi_xm_norm_std = np.nanstd(results[dim]['norm_plus'])

# with open('../results/fb_lp.pkl', 'wb') as file: 
#     pkl.dump(fb_results, file)

# # ###################
# # ####### GMI #######
# # ###################
# # with open('../results/fb_gmi.pkl', 'rb') as file:
# #     results = pkl.load(file)
    
# # lp_results = []
# # for _ in tqdm(range(3)):
    
# #     runs = len(results[dim]['norm_og'])
# #     random_idx = np.random.randint(runs)
# #     print ("Using Run #" + str(random_idx))
# #     lp_dict = link_preidction(graph = graph,
# #                               embed_og = results[dim]['embed_og'][random_idx],
# #                               embed_plus = results[dim]['embed_plus'][random_idx],
# #                               epochs = 100)
# #     lp_results.append(list(lp_dict.values()))
    
# # lp_results = np.array(lp_results)
# # fb_results['gmi'] = lp_results

# # gmi_norm = np.nanmean(results[dim]['norm_og'])
# # gmi_norm_std = np.nanstd(results[dim]['norm_og'])

# # gmi_xm_norm = np.nanmean(results[dim]['norm_plus'])
# # gmi_xm_norm_std = np.nanstd(results[dim]['norm_plus'])

# # with open('../results/fb_lp.pkl', 'wb') as file: 
# #     pkl.dump(fb_results, file)

# ####################
# ####### SDNE #######
# ####################
# with open('../results/fb_sdne.pkl', 'rb') as file:
#     results = pkl.load(file)
# lp_results = []
# runs = len(results[dim]['norm_og'])
# random_idx = np.random.randint(runs)
# for _ in tqdm(range(3)):
    
#     random_idx = np.random.randint(runs)
#     print ("Using Run #" + str(random_idx))
#     lp_dict = link_preidction(graph = graph,
#                               embed_og = results[dim]['embed_og'][random_idx],
#                               embed_plus = results[dim]['embed_plus'][random_idx],
#                               epochs = 100)
#     lp_results.append(list(lp_dict.values()))
    
# lp_results = np.array(lp_results)
# fb_results['sdne'] = lp_results

# sdne_norm = np.nanmean(results[dim]['norm_og'])
# sdne_norm_std = np.nanstd(results[dim]['norm_og'])

# sdne_xm_norm = np.nanmean(results[dim]['norm_plus'])
# sdne_xm_norm_std = np.nanstd(results[dim]['norm_plus'])

# with open('../results/fb_lp.pkl', 'wb') as file: 
#     pkl.dump(fb_results, file)

# ####################
# ####### LINE #######
# ####################
# with open('../results/fb_line.pkl', 'rb') as file:
#     results = pkl.load(file)
# lp_results = []
# runs = len(results[dim]['norm_og'])
# random_idx = np.random.randint(runs)
# for _ in tqdm(range(3)):
    
#     random_idx = np.random.randint(runs)
#     print ("Using Run #" + str(random_idx))
#     lp_dict = link_preidction(graph = graph,
#                               embed_og = results[dim]['embed_og'][random_idx],
#                               embed_plus = results[dim]['embed_plus'][random_idx],
#                               epochs = 100)
#     lp_results.append(list(lp_dict.values()))
    
# lp_results = np.array(lp_results)
# fb_results['line'] = lp_results

# line_norm = np.nanmean(results[dim]['norm_og'])
# line_norm_std = np.nanstd(results[dim]['norm_og'])

# line_xm_norm = np.nanmean(results[dim]['norm_plus'])
# line_xm_norm_std = np.nanstd(results[dim]['norm_plus'])

# with open('../results/fb_lp.pkl', 'wb') as file: 
#     pkl.dump(fb_results, file)

######################################################################################################################################################

# with open('../data/pubmed.pkl', 'rb') as file: 
#     graph = pkl.load(file)
# graph = nx.Graph(nx.to_numpy_array(graph))

# pubmed_results = {}
# dim = 128


# ###################
# ####### DGI #######
# ###################
# with open('../results/pubmed_dgi.pkl', 'rb') as file:
#     results = pkl.load(file)
    
# lp_results = []
# for _ in tqdm(range(3)):
    
#     runs = len(results[dim]['norm_og'])
#     random_idx = np.random.randint(runs)
#     print ("Using Run #" + str(random_idx))
#     lp_dict = link_preidction(graph = graph,
#                               embed_og = results[dim]['embed_og'][random_idx],
#                               embed_plus = results[dim]['embed_plus'][random_idx],
#                               epochs = 100)
#     lp_results.append(list(lp_dict.values()))
    
# lp_results = np.array(lp_results)
# pubmed_results['dgi'] = lp_results

# dgi_norm = np.nanmean(results[dim]['norm_og'])
# dgi_norm_std = np.nanstd(results[dim]['norm_og'])

# dgi_xm_norm = np.nanmean(results[dim]['norm_plus'])
# dgi_xm_norm_std = np.nanstd(results[dim]['norm_plus'])

# with open('../results/pubmed_lp.pkl', 'wb') as file: 
#     pkl.dump(pubmed_results, file)

# # ###################
# # ####### GMI #######
# # ###################
# # with open('../results/pubmed_gmi.pkl', 'rb') as file:
# #     results = pkl.load(file)
    
# # lp_results = []
# # for _ in tqdm(range(3)):
    
# #     runs = len(results[dim]['norm_og'])
# #     random_idx = np.random.randint(runs)
# #     print ("Using Run #" + str(random_idx))
# #     lp_dict = link_preidction(graph = graph,
# #                               embed_og = results[dim]['embed_og'][random_idx],
# #                               embed_plus = results[dim]['embed_plus'][random_idx],
# #                               epochs = 100)
# #     lp_results.append(list(lp_dict.values()))
    
# # lp_results = np.array(lp_results)
# # pubmed_results['gmi'] = lp_results

# # gmi_norm = np.nanmean(results[dim]['norm_og'])
# # gmi_norm_std = np.nanstd(results[dim]['norm_og'])

# # gmi_xm_norm = np.nanmean(results[dim]['norm_plus'])
# # gmi_xm_norm_std = np.nanstd(results[dim]['norm_plus'])

# # with open('../results/pubmed_lp.pkl', 'wb') as file: 
# #     pkl.dump(pubmed_results, file)

# ####################
# ####### SDNE #######
# ####################
# with open('../results/pubmed_sdne.pkl', 'rb') as file:
#     results = pkl.load(file)
# lp_results = []
# runs = len(results[dim]['norm_og'])
# random_idx = np.random.randint(runs)
# for _ in tqdm(range(3)):
    
#     random_idx = np.random.randint(runs)
#     print ("Using Run #" + str(random_idx))
#     lp_dict = link_preidction(graph = graph,
#                               embed_og = results[dim]['embed_og'][random_idx],
#                               embed_plus = results[dim]['embed_plus'][random_idx],
#                               epochs = 100)
#     lp_results.append(list(lp_dict.values()))
    
# lp_results = np.array(lp_results)
# pubmed_results['sdne'] = lp_results

# sdne_norm = np.nanmean(results[dim]['norm_og'])
# sdne_norm_std = np.nanstd(results[dim]['norm_og'])

# sdne_xm_norm = np.nanmean(results[dim]['norm_plus'])
# sdne_xm_norm_std = np.nanstd(results[dim]['norm_plus'])

# with open('../results/pubmed_lp.pkl', 'wb') as file: 
#     pkl.dump(pubmed_results, file)

# ####################
# ####### LINE #######
# ####################
# with open('../results/pubmed_line.pkl', 'rb') as file:
#     results = pkl.load(file)
# lp_results = []
# runs = len(results[dim]['norm_og'])
# random_idx = np.random.randint(runs)
# for _ in tqdm(range(3)):
    
#     random_idx = np.random.randint(runs)
#     print ("Using Run #" + str(random_idx))
#     lp_dict = link_preidction(graph = graph,
#                               embed_og = results[dim]['embed_og'][random_idx],
#                               embed_plus = results[dim]['embed_plus'][random_idx],
#                               epochs = 100)
#     lp_results.append(list(lp_dict.values()))
    
# lp_results = np.array(lp_results)
# pubmed_results['line'] = lp_results

# line_norm = np.nanmean(results[dim]['norm_og'])
# line_norm_std = np.nanstd(results[dim]['norm_og'])

# line_xm_norm = np.nanmean(results[dim]['norm_plus'])
# line_xm_norm_std = np.nanstd(results[dim]['norm_plus'])

# with open('../results/pubmed_lp.pkl', 'wb') as file: 
#     pkl.dump(pubmed_results, file)

###############################################################################################################################################################################

with open('../data/usa_airport.pkl', 'rb') as file: 
    graph = pkl.load(file)['graph']
graph = nx.Graph(nx.to_numpy_array(graph))

airport_results = {}
dim = 128


###################
####### DGI #######
###################
with open('../results/airport_dgi.pkl', 'rb') as file:
    results = pkl.load(file)
    
lp_results = []
for _ in tqdm(range(3)):
    
    runs = len(results[dim]['norm_og'])
    random_idx = np.random.randint(runs)
    print ("Using Run #" + str(random_idx))
    lp_dict = link_preidction(graph = graph,
                              embed_og = results[dim]['embed_og'][random_idx],
                              embed_plus = results[dim]['embed_plus'][random_idx],
                              epochs = 100)
    lp_results.append(list(lp_dict.values()))
    
lp_results = np.array(lp_results)
airport_results['dgi'] = lp_results

dgi_norm = np.nanmean(results[dim]['norm_og'])
dgi_norm_std = np.nanstd(results[dim]['norm_og'])

dgi_xm_norm = np.nanmean(results[dim]['norm_plus'])
dgi_xm_norm_std = np.nanstd(results[dim]['norm_plus'])

with open('../results/airport_lp.pkl', 'wb') as file: 
    pkl.dump(airport_results, file)

# ###################
# ####### GMI #######
# ###################
# with open('../results/airport_gmi.pkl', 'rb') as file:
#     results = pkl.load(file)
    
# lp_results = []
# for _ in tqdm(range(3)):
    
#     runs = len(results[dim]['norm_og'])
#     random_idx = np.random.randint(runs)
#     print ("Using Run #" + str(random_idx))
#     lp_dict = link_preidction(graph = graph,
#                               embed_og = results[dim]['embed_og'][random_idx],
#                               embed_plus = results[dim]['embed_plus'][random_idx],
#                               epochs = 100)
#     lp_results.append(list(lp_dict.values()))
    
# lp_results = np.array(lp_results)
# airport_results['gmi'] = lp_results

# gmi_norm = np.nanmean(results[dim]['norm_og'])
# gmi_norm_std = np.nanstd(results[dim]['norm_og'])

# gmi_xm_norm = np.nanmean(results[dim]['norm_plus'])
# gmi_xm_norm_std = np.nanstd(results[dim]['norm_plus'])

# with open('../results/airport_lp.pkl', 'wb') as file: 
#     pkl.dump(airport_results, file)

####################
####### SDNE #######
####################
with open('../results/airport_sdne.pkl', 'rb') as file:
    results = pkl.load(file)
lp_results = []
runs = len(results[dim]['norm_og'])
random_idx = np.random.randint(runs)
for _ in tqdm(range(3)):
    
    random_idx = np.random.randint(runs)
    print ("Using Run #" + str(random_idx))
    lp_dict = link_preidction(graph = graph,
                              embed_og = results[dim]['embed_og'][random_idx],
                              embed_plus = results[dim]['embed_plus'][random_idx],
                              epochs = 100)
    lp_results.append(list(lp_dict.values()))
    
lp_results = np.array(lp_results)
airport_results['sdne'] = lp_results

sdne_norm = np.nanmean(results[dim]['norm_og'])
sdne_norm_std = np.nanstd(results[dim]['norm_og'])

sdne_xm_norm = np.nanmean(results[dim]['norm_plus'])
sdne_xm_norm_std = np.nanstd(results[dim]['norm_plus'])

with open('../results/airport_lp.pkl', 'wb') as file: 
    pkl.dump(airport_results, file)

####################
####### LINE #######
####################
with open('../results/airport_line.pkl', 'rb') as file:
    results = pkl.load(file)
lp_results = []
runs = len(results[dim]['norm_og'])
random_idx = np.random.randint(runs)
for _ in tqdm(range(3)):
    
    random_idx = np.random.randint(runs)
    print ("Using Run #" + str(random_idx))
    lp_dict = link_preidction(graph = graph,
                              embed_og = results[dim]['embed_og'][random_idx],
                              embed_plus = results[dim]['embed_plus'][random_idx],
                              epochs = 100)
    lp_results.append(list(lp_dict.values()))
    
lp_results = np.array(lp_results)
airport_results['line'] = lp_results

line_norm = np.nanmean(results[dim]['norm_og'])
line_norm_std = np.nanstd(results[dim]['norm_og'])

line_xm_norm = np.nanmean(results[dim]['norm_plus'])
line_xm_norm_std = np.nanstd(results[dim]['norm_plus'])

with open('../results/airport_lp.pkl', 'wb') as file: 
    pkl.dump(airport_results, file)
