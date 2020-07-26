#OGMLProject


# Check the versions of libraries

#Python Version
import sys
print('Python: {}'.format(sys.version))

# Scipy
import scipy
print('scipy: {}'.format(sys.version))

#numpy
import numpy
print('numpy: {}'.format(numpy.__version__))

#Matplot
import matplotlib
print('matplot: {}'.format(matplotlib.__version__))

#Pandas
import pandas
print('pandas: {}'.format(pandas.__version__))

#scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))




# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def main():
	

	def loadDataSet():
		# Load dataset
		url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
		names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
		dataset = read_csv(url, names=names)
		return dataset

	def viewDataSet(dataset):
		#view the dataset
		print(dataset.shape)
		print(dataset.head(20))
		print(dataset.describe())

	def classDistribution(dataset):
		# class distribution
		print(dataset.groupby('class').size())

	def plot_BoxWhisker(dataset):
		#box and whisker plots
		# box and whisker plots
		dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
		pyplot.show()

	def plot_Histogram(dataset):
		#histograms
		dataset.hist()
		pyplot.show()

	def plot_ScatterMatrix(dataset):
		#Scatter Plot Matrix
		scatter_matrix(dataset)
		pyplot.show()

	def splitoutValidation(dataset):
		# Split-out validation dataset
		array = dataset.values
		X = array[:,0:4]
		y = array[:,4]
		X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

	viewDataSet(loadDataSet())
	classDistribution(loadDataSet())
	plot_BoxWhisker(loadDataSet())
	plot_Histogram(loadDataSet())
	plot_ScatterMatrix(loadDataSet())
	splitoutValidation(loadDataSet())
main()

