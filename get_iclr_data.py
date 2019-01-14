import json 
import os
import numpy as np
import re


def read_glove():
	glove_data = {}
	with open(glove_path, 'r') as f:
		for line in f:
			splitLine = line.split()
			word = splitLine[0]
			embedding = np.array([float(val) for val in splitLine[1:]])
			glove_data[word] = embedding
	return glove_data


def to_graph(filename, glove_dict, glove_dim):
	data_list = json.load(open(filename, 'r'))
	all_data = []
	for json_dict in data_list:
		graph_data = {}
		candidates = [] #list of node ids to compute for final prediction

		options = json_dict['SymbolCandidates']
		for var_data in options:
			candidates.append(var_data['SymbolDummyNode'])
			if var_data['IsCorrect']:
				graph_data['targets'] = var_data['SymbolDummyNode']
		
		context = json_dict['ContextGraph']
		node_features, edges = [], []
		for edge_type, edge_list in context['Edges'].items():
			for e in edge_list:
				edges.append([e[0], edge_dict[edge_type], e[1]])

		label_dict = context['NodeLabels']
		node_ids = [int(s) for s in label_dict.keys()]
		node_ids.sort()
		for node_id in node_ids:
			node_name = label_dict[str(node_id)]
			node_name = node_name.capitalize()
			tokens = re.findall('[A-Z][a-z]*', node_name)
			embeddings = []
			if len(tokens) == 0:
				tokens = [node_name]
			for token in tokens:
				if token not in glove_dict: #unknown word
					embeddings.append(np.zeros(glove_dim,))
				else:
					embeddings.append(glove_dict[token])
			avg_embedding = list(np.mean(np.vstack(embeddings), axis=0))
			node_features.append(avg_embedding)

		graph_data['graph'] = edges
		graph_data['node_features'] = node_features
		graph_data['candidates'] = candidates
		all_data.append(graph_data)
	return all_data


folders = ['entityframework']#, 'akka.net']
path = '/dfs/scratch2/thaonguyen/graph-dataset/'
glove_dim = 100
glove_path = '/dfs/scratch2/thaonguyen/glove/glove.6B.%dd.txt' % glove_dim

edge_dict = {'GuardedByNegation': 1, 'LastUse': 2, 'LastLexicalUse': 3, 'ReturnsTo': 4, 'GuardedBy': 5, \
			'FormalArgName': 6, 'NextToken': 7, 'Child': 8, 'LastRead': 9, 'LastWrite': 10, 'ComputedFrom': 11}

print('parsing json files as graphs...')
processed_data = {'train': [], 'valid': []}
glove_dict = read_glove()

for section in ['train', 'valid']:
	for folder in folders:
		data_path = os.path.join(path, folder, '-'.join(['graphs', section]))
		print(data_path)
		for file in os.listdir(data_path):
			print(file)
			all_data = to_graph(os.path.join(data_path, file), glove_dict, glove_dim)
			processed_data[section].extend(all_data)

	out_dir = os.path.join(path, 'iclr_%s.json' % section)
	with open(out_dir, 'w') as f:
		json.dump(processed_data[section], f)
