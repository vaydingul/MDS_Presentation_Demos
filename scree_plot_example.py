import utils
import numpy as np
import matplotlib.pyplot as plt


def scree_plot(dm, max_dim):

	stresses = utils.get_stress_for_different_dimensions(dm, 5)

	plt.figure()
	plt.plot(list(range(1, max_dim + 1)), stresses, marker = "o")
	plt.title("Scree Plot")
	plt.xlabel("Number of Dimensions")
	plt.ylabel("Stress")
	plt.savefig("scree_plot_example.png")
	plt.show()



if __name__ == "__main__":

	dm = np.ones((3,3))

	scree_plot(dm, 5)