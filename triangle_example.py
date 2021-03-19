import utils
import numpy as np
import matplotlib.pyplot as plt



if __name__ == "__main__":

	# Similarity matrix
	triangle_2d = np.ones((3,3))

	# Embeddins in different dimensions
	embedding_1 = utils.get_embedding(triangle_2d, 1)
	embedding_2 = utils.get_embedding(triangle_2d, 2)
	embedding_3 = utils.get_embedding(triangle_2d, 3)
	
	print(embedding_1, embedding_2, embedding_3)

	utils.visualize(embedding_1, 1)
	utils.visualize(embedding_2, 2)
	utils.visualize(embedding_3, 3)
	
	