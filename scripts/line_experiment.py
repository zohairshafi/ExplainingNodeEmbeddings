# nohup python line_experiment.py --graph_path "../data/pubmed.pkl" --run_count 10 --hyp_key "hyp_pubmed" --outfile "../results/pubmed_line.pkl" > ../logs/pubmed_line.log 2>&1 &
from utils import *

ap = argparse.ArgumentParser()
ap.add_argument("-g", "--graph_path", required = True, help = 'Path to an nx.Graph object stored as a .pkl file')
ap.add_argument("-r", "--run_count", required = True, help = "Number of iterations for the experiment", default = 1)
ap.add_argument("-k", "--hyp_key", required = True, help = "Key to index the hyperparameter json file")
ap.add_argument("-o", "--outfile", required = True, help = "File name to save results into")
ap.add_argument("-u", "--update_outfile", required = False, help = "If outfile already exists, read it in instead of over writing it")

args = vars(ap.parse_args())

filename = args['graph_path']
run_count = int(args['run_count'])
hyp_key = args['hyp_key']
outfile = args['outfile']
update_outfile = args['update_outfile'] == "True"


#################################
######### Read In Graph #########
#################################
with open(filename, 'rb') as file: 
    graph_dict = pkl.load(file)
    
try:
    graph = nx.Graph(nx.to_numpy_array(graph_dict['graph']))    
except:
    graph = nx.Graph(nx.to_numpy_array(graph_dict))

#################################
#### Generate Sense Features ####
#################################
try: 
    name = hyp_key.strip('hyp_')
    with open('../data/' + name + '_sf.pkl', 'rb') as file: 
        [sense_features, sense_feat_dict] = pkl.load(file)

except: 
    sense_feat_dict, sense_features = get_sense_features(graph, ppr_flag = 'std')

    uncorrelated_feats = ['Degree',
                        'Clustering Coefficient',
                        'Personalized Page Rank - Standard Deviation',
                        'Average Neighbor Degree',
                        'Average Neighbor Clustering',
                        'Eccentricity',
                        'Katz Centrality']
    if 'Eccentricity' not in sense_feat_dict:
        uncorrelated_feats = ['Degree',
                        'Clustering Coefficient',
                        'Personalized Page Rank - Standard Deviation',
                        'Average Neighbor Degree',
                        'Average Neighbor Clustering',
                        'Katz Centrality']

    sense_features = sense_features[:, [list(sense_feat_dict).index(feat) for feat in uncorrelated_feats]]
    sense_feat_dict = {feat : idx for idx, feat in enumerate(uncorrelated_feats)}

    name = hyp_key.strip('hyp_')

    with open('../data/' + name + '_sf.pkl', 'wb') as file: 
        pkl.dump([sense_features, sense_feat_dict], file)
#################################
######## Hyperparameters ########
#################################

# Define static ones to override or read in from a file

if hyp_key == '':
    hyp = {'line' : {'alpha' : 0.1, 
                     'ortho' : 0, 
                     'sparse' : 0, 
                     'epochs' : 15, 
                     'batch_size' : 1024, 
                     'lr' : 1e-3}, 

          'line+xm' : {'alpha' : 100, 
                      'ortho' : 10, 
                      'sparse' : 10, 
                      'epochs' : 50, 
                      'batch_size' : 1024, 
                      'lr' : 5e-4}}
else: 
    with open('hyp.json', 'r') as file: 
        hyp_file = json.load(file)
        hyp = hyp_file[hyp_key]


#################################
######## Run Experiment #########
#################################

dimensions = [16, 32, 64, 128, 256]
if update_outfile == False:
    results = {d : {} for d in dimensions}
else: 
    with open(outfile, 'rb') as file: 
        results = pkl.load(file)

run_time = []

for run_idx in tqdm(range(run_count)):
    
    run_start = time.time()

    for d in dimensions: 
    
        # Embed 
        
        # Standard LINE
        line_start = time.time()
        line = LINE(graph, 
                embedding_size = d,
                sense_features = sense_features,
                alpha = hyp['line']['alpha'], 
                ortho = hyp['line']['ortho'], 
                sparse = hyp['line']['sparse'],
                learning_rate =  hyp['line']['lr'],
                order = 'second', 
                batch_size = hyp['line']['batch_size'])

        history = line.train(epochs = hyp['line']['epochs'])

        e = line.get_embeddings()
        embed_og = np.array([e[node_name] for node_name in graph.nodes()])
        embed_og = (embed_og - np.min(embed_og)) / np.ptp(embed_og)
        line_time = (time.time() - line_start) / hyp['line']['epochs']


        feature_dict_og = find_feature_membership(input_embed = embed_og,
                                                            embed_name = 'LINE',
                                                            sense_features = sense_features,
                                                            sense_feat_dict = sense_feat_dict,
                                                            top_k = 8,
                                                            solver = 'nmf')

        explain_og = feature_dict_og['explain_norm']
        explain_og_norm = np.linalg.norm(explain_og, ord = 'nuc')
        error_og = sense_features * np.log((sense_features + 1e-10) / ((embed_og @ feature_dict_og['explain_norm']) + 1e-10)) - sense_features + (embed_og @ feature_dict_og['explain_norm'])
        explain_og = (explain_og - np.min(explain_og)) / np.ptp(explain_og)
        
        # LINE+XM
        line_plus_start = time.time()
        line_plus = LINE(graph, 
                        embedding_size = d,
                        sense_features = sense_features,
                        alpha = hyp['line+xm']['alpha'], 
                        ortho = hyp['line+xm']['ortho'], 
                        sparse = hyp['line+xm']['sparse'],
                        learning_rate =  hyp['line+xm']['lr'],
                        order = 'second', 
                        batch_size = hyp['line+xm']['batch_size'])

        history = line_plus.train(epochs = hyp['line+xm']['epochs'])

        e = line_plus.get_embeddings()
        embed_plus = np.array([e[node_name] for node_name in graph.nodes()])
        embed_plus = (embed_plus - np.min(embed_plus)) / np.ptp(embed_plus)
        line_plus_time = (time.time() - line_plus_start) / hyp['line+xm']['epochs']

        feature_dict_plus = find_feature_membership(input_embed = embed_plus,
                                                            embed_name = 'LINE+XM',
                                                            sense_features = sense_features,
                                                            sense_feat_dict = sense_feat_dict,
                                                            top_k = 8,
                                                            solver = 'nmf')

        explain_plus = feature_dict_plus['explain_norm']
        explain_plus_norm = np.linalg.norm(explain_plus, ord = 'nuc')
        error_plus = sense_features * np.log((sense_features + 1e-10) / ((embed_plus @ feature_dict_plus['explain_norm']) + 1e-10)) - sense_features + (embed_plus @ feature_dict_plus['explain_norm'])
        explain_plus = (explain_plus - np.min(explain_plus)) / np.ptp(explain_plus)

        # Generate Node Explanations
        Y_og = embed_og
        sense_mat = tf.einsum('ij, ik -> ijk', Y_og, sense_features)
        Y_og_norm = tf.linalg.diag_part(tf.matmul(Y_og, Y_og, transpose_b = True), k = 0)
        sense_norm = tf.linalg.diag_part(tf.matmul(sense_features, sense_features, transpose_b = True), k = 0)
        norm = Y_og_norm * tf.cast(sense_norm, tf.float32)
        D_og = tf.transpose(tf.transpose(sense_mat) / norm)
        D_og = (D_og - tf.reshape(tf.reduce_min(D_og, axis = [-1, -2]), (-1, 1, 1))) / tf.reshape(tf.reduce_max(D_og, axis = [-1, -2]) - tf.reduce_min(D_og, axis = [-1, -2]), (-1, 1, 1))
        orth_og = np.squeeze(tf.reduce_sum(D_og @ tf.transpose(D_og, perm = (0, 2, 1)), axis = [1, 2]))



        Y_plus = embed_plus
        sense_mat = tf.einsum('ij, ik -> ijk', Y_plus, sense_features)
        Y_plus_norm = tf.linalg.diag_part(tf.matmul(Y_plus, Y_plus, transpose_b = True), k = 0)
        sense_norm = tf.linalg.diag_part(tf.matmul(sense_features, sense_features, transpose_b = True), k = 0)
        norm = Y_plus_norm * tf.cast(sense_norm, tf.float32)
        D_plus = tf.transpose(tf.transpose(sense_mat) / norm)
        D_plus = (D_plus - tf.reshape(tf.reduce_min(D_plus, axis = [-1, -2]), (-1, 1, 1))) / tf.reshape(tf.reduce_max(D_plus, axis = [-1, -2]) - tf.reduce_min(D_plus, axis = [-1, -2]), (-1, 1, 1))
        orth_plus = np.squeeze(tf.reduce_sum(D_plus @ tf.transpose(D_plus, perm = (0, 2, 1)), axis = [1, 2]))


        norm_og = [np.linalg.norm(D_og[node, :, :], ord = 'nuc') for node in range(len(graph))]
        norm_plus = [np.linalg.norm(D_plus[node, :, :], ord = 'nuc') for node in range(len(graph))]
        
        try:
            results[d]['norm_og'].append(norm_og)
            results[d]['norm_plus'].append(norm_plus)
            results[d]['orth_og'].append(orth_og)
            results[d]['orth_plus'].append(orth_plus)
            results[d]['explain_og_norm'].append(explain_og_norm)
            results[d]['explain_plus_norm'].append(explain_plus_norm)
            results[d]['line_time'].append(line_time)
            results[d]['line+xm_time'].append(line_plus_time)
            results[d]['error_og'].append(error_og)
            results[d]['error_plus'].append(error_plus)
            results[d]['embed_og'].append(embed_og)
            results[d]['embed_plus'].append(embed_plus)
            
        except: 
            results[d]['norm_og'] = [norm_og]
            results[d]['norm_plus'] = [norm_plus]
            results[d]['orth_og'] = [orth_og]
            results[d]['orth_plus'] = [orth_plus]
            results[d]['explain_og_norm'] = [explain_og_norm]
            results[d]['explain_plus_norm'] = [explain_plus_norm]
            results[d]['line_time'] = [line_time]
            results[d]['line+xm_time'] = [line_plus_time]
            results[d]['error_og'] = [error_og]
            results[d]['error_plus'] = [error_plus]
            results[d]['embed_og'] = [embed_og]
            results[d]['embed_plus'] = [embed_plus]
            
    print ("###### RUN NUMBER : " + str(run_idx))
    print ("###### Results Len : " + str(len(results[64]['norm_og'])))
    with open(outfile, 'wb') as file: 
        pkl.dump(results, file)
        

    run_time.append(time.time() - run_start)

    with open(outfile + '_progress.txt', 'w') as file: 
        string = 'Current Run : ' + str(run_idx)
        string += '\nLast Iteration Time : ' + str(run_time[-1]) + 's'
        string += '\nAverage Iteration Time : ' + str(np.mean(run_time)) + 's'
        string += '\nEstimated Time Left : ' + str(np.mean(run_time) * (run_count - run_idx)) + 's'
        file.write(string)