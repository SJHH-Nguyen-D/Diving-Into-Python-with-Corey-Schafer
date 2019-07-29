class DecisionTree:
	def __init__(self, min_leaves, min_split):
		self.min_leaves = min_leaves
		self.min_split = min_split

	def fit(self, X, y):
		return ""

	def predict(self, y):
		return ""

def gini_index(X):
	n = len(X)
	gini_coeff = 0
	for i in range(n):
		for j in range(n):
			gini_coeff = gini_coeff + abs(X[i]-X[j])
	gini_coeff = gini_coeff / (2* n**2 * average(X))
	return gini_coeff

class AdaBoostClassifier(DecisionTree):
	def __init__(self, n_features, min_split):
		self.n_features = n_features
		self.min_split = min_split

	def update_weights()
		pass

	def calc_vote_weights()
		pass