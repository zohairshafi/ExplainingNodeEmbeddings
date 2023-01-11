from utils import *

ap = argparse.ArgumentParser()
ap.add_argument("-g", "--graph_path", required = True, help = 'Path to an nx.Graph object stored as a .pkl file')
ap.add_argument("-r", "--run_count", required = True, help = "Number of iterations for the experiment", default = 1)
ap.add_argument("-k", "--hyp_key", required = True, help = "Key to index the hyperparameter json file")
ap.add_argument("-o", "--outfile", required = True, help = "File name to save results into")

args = vars(ap.parse_args())

filename = args['graph_path']
run_count = int(args['run_count'])
hyp_key = args['hyp_key']
outfile = args['outfile']

#################################
######### Read In Graph #########
#################################
with open(filename, 'rb') as file: 
    graph_dict = pkl.load(file)
    
graph = nx.Graph(nx.to_numpy_array(graph_dict['graph']))    
graph = nx.Graph(nx.to_numpy_array(graph))


#################################
#### Generate Sense Features ####
#################################
sense_feat_dict, sense_features = get_sense_features(graph, ppr_flag = 'std')

uncorrelated_feats = ['Degree',
                    'Clustering Coefficient',
                    'Personalized Page Rank - Standard Deviation',
                    'Average Neighbor Degree',
                    'Average Neighbor Clustering',
                    'Eccentricity',
                    'Katz Centrality']
sense_features = sense_features[:, [list(sense_feat_dict).index(feat) for feat in uncorrelated_feats]]
sense_feat_dict = {feat : idx for idx, feat in enumerate(uncorrelated_feats)}

#################################
######## Hyperparameters ########
#################################

# Define static ones to override or read in from a file

if hyp_key == '':
    hyp = {'sdne' : {'alpha' : 0.1, 
                     'beta' : 10, 
                     'gamma' : 0, 
                     'delta' : 0, 
                     'epochs' : 200, 
                     'batch_size' : 1024, 
                     'lr' : 1e-3}, 

          'sdne+xm' : {'alpha' : 1, 
                      'beta' : 1, 
                      'gamma' : 10, 
                      'delta' : 10, 
                      'epochs' : 400, 
                      'batch_size' : 1024, 
                      'lr' : 5e-4}}
else: 
    with open('hyp.json', 'r') as file: 
        hyp_file = json.load(file)
        hyp = hyp_file[hyp_key]


#################################
######## Run Experiment #########
#################################

dimensions = [16, 32, 64, 256, 512]
results = {d : {} for d in dimensions}
run_time = []

for run_idx in tqdm(range(run_count)):

    run_start = time.time()
    
    for d in dimensions: 
    
        # Embed 
        
        # Standard SDNE
        sdne_start = time.time()
        sdne = SDNE_plus(graph, 
                          hidden_size = [32, d], 
                          lr = hyp['sdne']['lr'],
                          sense_features = sense_features.astype(np.float32),
                          alpha = hyp['sdne']['alpha'], 
                          beta = hyp['sdne']['beta'], 
                          gamma = hyp['sdne']['gamma'], 
                          delta = hyp['sdne']['delta'])
        history = sdne.train(epochs = hyp['sdne']['epochs'], batch_size = hyp['sdne']['batch_size'])
        e = sdne.get_embeddings()
        embed_og = np.array([e[node_name] for node_name in graph.nodes()])
        embed_og = (embed_og - np.min(embed_og)) / np.ptp(embed_og)
        sdne_time = (time.time() - sdne_start) / hyp['sdne']['epochs']

        # SDNE+XM
        sdne_plus_start = time.time()
        sdne_plus = SDNE_plus(graph, 
                                  hidden_size = [32, d], 
                                  lr = hyp['sdne+xm']['lr'],
                                  sense_features = sense_features.astype(np.float32),
                                  alpha = hyp['sdne+xm']['alpha'], 
                                  beta = hyp['sdne+xm']['beta'], 
                                  gamma = hyp['sdne+xm']['gamma'], 
                                  delta = hyp['sdne+xm']['delta'])

        sdne_plus.model.set_weights(sdne.model.get_weights())
        history = sdne_plus.train(epochs = hyp['sdne+xm']['epochs'], batch_size = hyp['sdne+xm']['batch_size'])
        e = sdne_plus.get_embeddings()
        embed_plus = np.array([e[node_name] for node_name in graph.nodes()])
        embed_plus = (embed_plus - np.min(embed_plus)) / np.ptp(embed_plus)
        sdne_plus_time = (time.time() - sdne_plus_start) / hyp['sdne+xm']['epochs']

        
        # Generate Graph Explanations and Save
        feature_dict_og = find_feature_membership(input_embed = embed_og,
                                                    embed_name = 'SDNE',
                                                    sense_features = sense_features,
                                                    sense_feat_dict = sense_feat_dict,
                                                    top_k = 8,
                                                    solver = 'nmf')

        explain_og = feature_dict_og['explain_norm']
        explain_og = (explain_og - np.min(explain_og)) / np.ptp(explain_og)
        explain_og_norm = np.linalg.norm(explain_og, ord = 'nuc')
        
        feature_dict_plus = find_feature_membership(input_embed = embed_plus,
                                                            embed_name = 'SDNE+ Init',
                                                            sense_features = sense_features,
                                                            sense_feat_dict = sense_feat_dict,
                                                            top_k = 8,
                                                            solver = 'nmf')

        explain_plus = feature_dict_plus['explain_norm']
        explain_plus = (explain_plus - np.min(explain_plus)) / np.ptp(explain_plus)
        explain_plus_norm = np.linalg.norm(explain_plus, ord = 'nuc')

        # Generate Node Explanations
        Y_og = embed_og
        sense_mat = tf.einsum('ij, ik -> ijk', Y_og, sense_features)
        Y_og_norm = tf.linalg.diag_part(tf.matmul(Y_og, Y_og, transpose_b = True), k = 0)
        sense_norm = tf.linalg.diag_part(tf.matmul(sense_features, sense_features, transpose_b = True), k = 0)
        norm = Y_og_norm * tf.cast(sense_norm, tf.float32)
        D_og = tf.transpose(tf.transpose(sense_mat) / norm)


        Y_plus = embed_plus
        sense_mat = tf.einsum('ij, ik -> ijk', Y_plus, sense_features)
        Y_plus_norm = tf.linalg.diag_part(tf.matmul(Y_plus, Y_plus, transpose_b = True), k = 0)
        sense_norm = tf.linalg.diag_part(tf.matmul(sense_features, sense_features, transpose_b = True), k = 0)
        norm = Y_plus_norm * tf.cast(sense_norm, tf.float32)
        D_plus = tf.transpose(tf.transpose(sense_mat) / norm)

        norm_og = [np.linalg.norm(D_og[node, :, :], ord = 'nuc') for node in range(len(graph))]
        norm_plus = [np.linalg.norm(D_plus[node, :, :], ord = 'nuc') for node in range(len(graph))]
        
        try:
            results[d]['norm_og'].append(norm_og)
            results[d]['norm_plus'].append(norm_plus)
            results[d]['explain_og_norm'].append(explain_og_norm)
            results[d]['explain_plus_norm'].append(explain_plus_norm)
            results[d]['sdne_time'].append(sdne_time)
            results[d]['sdne+xm_time'].append(sdne_plus_time)

            
        except: 
            results[d]['norm_og'] = [norm_og]
            results[d]['norm_plus'] = [norm_plus]
            results[d]['explain_og_norm'] = [explain_og_norm]
            results[d]['explain_plus_norm'] = [explain_plus_norm]
            results[d]['sdne_time'] = [sdne_time]
            results[d]['sdne+xm_time'] = [sdne_plus_time]

            
        results[d]['embed_og'] = embed_og
        results[d]['embed_plus'] = embed_plus
    
    with open(outfile, 'wb') as file: 
        pkl.dump(results, file)

    run_time.append(time.time() - run_start)

    with open(outfile + '_progress.txt', 'w') as file: 
        string = 'Current Run : ' + str(run_idx)
        string += '\nLast Iteration Time : ' + str(run_time[-1]) + 's'
        string += '\nAverage Iteration Time : ' + str(np.mean(run_time)) + 's'
        string += '\nEstimated Time Left : ' + str(np.mean(run_time) * (run_count - run_idx)) + 's'
        file.write(string)








