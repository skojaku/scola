import numpy as np
import networkx as nx
from scipy import linalg
from scipy import stats
from functools import partial

class PMFG():
	
	def __init__(self, disp=True):
		self.disp = disp
		
	def fit(self, sCov, nCov, dCov_init=None, alpha=None, disp=False):
	
		G = self._toGraph(sCov, nCov)

		sorted_edges = self._sort_graph_edges(G)	
		PMFG = self._compute_PMFG(sorted_edges, G.nodes)	
		
		return np.multiply(PMFG, sCov - nCov)

	def _sort_graph_edges(self, G):
		sorted_edges = []
		for source, dest, data in sorted(G.edges(data=True),key=lambda x: -x[2]['weight']):
			sorted_edges.append({'source': source,'dest': dest, 'weight': data['weight']})
			
		return sorted_edges

	def _toGraph(self, sCov, nCov):
		dCov = sCov - nCov
		G = nx.from_numpy_matrix(np.abs(dCov))	
		return G

	def _compute_PMFG(self, sorted_edges, nodes):
		nb_nodes = len(nodes)
		PMFG = nx.Graph()
		for edge in sorted_edges:
			PMFG.add_edge(edge['source'], edge['dest'])
			if not nx.algorithms.planarity.check_planarity(PMFG):
				PMFG.remove_edge(edge['source'], edge['dest'])
				
			if len(PMFG.edges()) == 3*(nb_nodes-2):
				break
		
		return nx.to_numpy_matrix(PMFG, nodelist = nodes)

	def loglikelihood(self, dCov, sCov, nCov):
		Cov = dCov + nCov
		if np.linalg.cond(Cov) < 1e-30:
			return -1e+30
		s, v = np.linalg.slogdet(Cov)
		return -0.5 *  s * v  - 0.5*np.trace(sCov @ linalg.inv(Cov)) - 0.5 * Cov.shape[0] * np.log(2 * np.pi)

	def candidateThreshold(self, sCov, nCov, num=None):
		return [0] 
