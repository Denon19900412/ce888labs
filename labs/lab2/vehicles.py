import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    data_mean = []
    lower = []
    upper = []
    new_sample = []

    for i in range(iterations):
        new_sample_data = []

        for j in range(sample_size):
            para_random = random.randint(0, 10000)
            new_sample_data.append(sample[int(para_random % sample_size)])

        new_sample.append(np.mean(new_sample_data))

    data_mean = np.mean(new_sample)
    lower = np.percentile(new_sample, (100 - desired_CI) / 2)
    upper = np.percentile(new_sample, desired_CI / 2)

    return data_mean, lower, upper


df = pd.read_csv('./vehicles.csv')
print((df.columns))

# histograms
sns_plot = sns.lmplot(df.columns[0], df.columns[1], data=df, fit_reg=False)

sns_plot.axes[0, 0].set_ylim(0, )
sns_plot.axes[0, 0].set_xlim(0, )

sns_plot.savefig("scaterplot.png", bbox_inches='tight')
sns_plot.savefig("scaterplot.pdf", bbox_inches='tight')

data = df.values.T[1]

print(data[0:79])
print((("Mean: %f") % (np.mean(data[0:79]))))
print((("Median: %f") % (np.median(data[0:79]))))
print((("Var: %f") % (np.var(data[0:79]))))
print((("std: %f") % (np.std(data[0:79]))))
print((("MAD: %f") % (mad(data[0:79]))))

# scatterplots
plt.clf()
sns_plot2 = sns.distplot(data, bins=20, kde=False, rug=True).get_figure()

axes = plt.gca()
axes.set_xlabel('Millons of pounds in sales')
axes.set_ylabel('Sales count')

sns_plot2.savefig("histogram.png", bbox_inches='tight')
sns_plot2.savefig("histogram.pdf", bbox_inches='tight')


x = boostrap(data[:79], 79, 10, 95)
print(x[0])
print(".........")
print(x[1])
print(".........")
print(x[2])