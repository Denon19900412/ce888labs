import matplotlib
matplotlib.use('Agg')
import pandas as pd
import seaborn as sns
import numpy as np
import random

def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation
        http://stackoverflow.com/questions/8930370/where-can-i-find-mad-mean-absolute-deviation-in-scipy
    """
    arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))


def boostrap(sample, sample_size, iterations, desired_CI):
	# <---INSERT YOUR CODE HERE--->
	new_sample = []
	for i in range(iterations):
		new_sample_data = []
		for j in range(sample_size):
			para_random = random.randint(0, 10000)
			new_sample_data.append(sample[int(para_random % sample_size)])

		new_sample.append(np.mean(new_sample_data))

	data_mean = np.mean(new_sample)
	lower = np.percentile(new_sample, (100 - desired_CI) / 2)
	upper = np.percentile(new_sample, desired_CI + (100 - desired_CI) / 2)

	return data_mean, lower, upper

if __name__ == "__main__":
	df = pd.read_csv('./salaries.csv')

	data = df.values.T[1]
	boots = []
	for i in range(100, 100000, 1000):
		boot = boostrap(data, data.shape[0], i, 95)
		boots.append([i, boot[0], "mean"])
		boots.append([i, boot[1], "lower"])
		boots.append([i, boot[2], "upper"])

	df_boot = pd.DataFrame(boots, columns=['Boostrap Iterations', 'Mean', "Value"])
	sns_plot = sns.lmplot(df_boot.columns[0], df_boot.columns[1], data=df_boot, fit_reg=False, hue="Value")

	sns_plot.axes[0, 0].set_ylim(0,)
	sns_plot.axes[0, 0].set_xlim(0, 100000)

	sns_plot.savefig("bootstrap_confidence.png", bbox_inches='tight')
	sns_plot.savefig("bootstrap_confidence.pdf", bbox_inches='tight')


	#print ("Mean: %f")%(np.mean(data))
	#print ("Var: %f")%(np.var(data))