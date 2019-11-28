import os
import numpy as np
import pickle
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import metrics, svm
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_decision_regions
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")

params = {'kernel_type': 'linear',
          'test_size': 0.15,
          'subsample_size': 20000,
          'pca' : True,
          'random_state': 123}

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

print(params)

#--FASHION SCATTER FUNCTION
def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", 16))

    # create a scatter plot.
    f = plt.figure(figsize=(20, 20))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="b"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

#-- MAKE INPUTS AND OUTPUTS

nominal_converter = {'class': {"present" : 1, "absent" : 0}}

# read dataset and replace nominal with numerical data
df = pd.read_csv('BNG_heart-statlog.csv')
#convert nominal to numerical
df.replace(nominal_converter, inplace = True)


# nov 26th
# split into inputs and outputs
features = df.iloc[:params['subsample_size'], :-1]
labels = df['class'][:params['subsample_size']]

#-- HISTOGRAMS AND PLOTS

fig = plt.figure(figsize = (8,6))
plt.title('Age of patients grouped by class')
df.hist(ax=fig.gca(),  column="age", by="class")
plt.show()

fig = plt.figure(figsize = (9,6))
plt.title('Resting blood pressure of patients grouped by class')
df.boxplot(ax=fig.gca(),column="resting_blood_pressure", by="class", vert=False)
plt.show()

# -- PCA
time_start = time.time()
pca = PCA(n_components=13)
pca_intermediate = pca.fit(features)
pca_result = pca_intermediate.transform(features)
print('pca shape', pca_result.shape)
print('PCA done! Time elapsed: {} seconds'.format(time.time() - time_start))

pca_df = pd.DataFrame(columns=['pc1', 'pc2'])
pca_df['pc1'] = pca_result[:, 0]
pca_df['pc2'] = pca_result[:, 1]
pca_df['pc3'] = pca_result[:, 2]
pca_df['pc4'] = pca_result[:, 3]
print(pca.explained_variance_ratio_)

colordict = {0: 'blue', 1: 'green'}
piclabel = {0: 'Absent', 1: 'Present'}
markers = {0: '*', 1: '+'}
alphas = {0: 0.5, 1: 0.5}

fig = plt.figure(figsize=(12, 7))
plt.subplot(1, 1, 1)
for l in np.unique(labels):
    ix = np.where(labels == l)[0]
    plt.scatter(pca_df['pc1'][ix], pca_df['pc2'][ix], c=colordict[l],
                label=piclabel[l], s=40, marker=markers[l], alpha=alphas[l])

plt.xlabel("First Principal Component", fontsize=15)
plt.ylabel("Second Principal Component", fontsize=15)

#-- MODEL AND CONFUSION MATRIX

if params['pca']:
    train_features, test_features, train_labels, test_labels = train_test_split(pca_df[['pc1', 'pc2']].values,
                                                                                labels,
                                                                                test_size=params['test_size'],
                                                                                random_state=params['random_state'])

else:
    train_features, test_features, train_labels, test_labels = train_test_split(features,
                                                                                labels,
                                                                                test_size=params['test_size'],
                                                                                random_state=params['random_state'])

'''

model_file = 'svm_model_' + str(params['subsample_size']) + \
    '_pca' if params['pca'] else '_no_pca' + \
    '.pickle'


details = "\n"
print(params, details)

time_start = time.time()

# make model if not exists, load if does
if (not os.path.exists(model_file)):
    print('Creating Model')
    model = svm.SVC(kernel=params['kernel_type'])

    model.fit(train_features, train_labels)

    with open(model_file, 'wb+') as output_file:
        pickle.dump(model, output_file)
else:
    print('Loading Model from File')
    with open(model_file, 'rb') as input_file:
        model = pickle.load(input_file)

print ('Fitting done, time: {} seconds'.format(time.time() - time_start))

predictions = model.predict(test_features)

print("Accuracy:", metrics.accuracy_score(test_labels, predictions))
print("Precision:", metrics.precision_score(test_labels, predictions))
print("Recall:", metrics.recall_score(test_labels, predictions))

confusion_mat = confusion_matrix(test_labels, predictions)

fig, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(confusion_mat.T,
            square=True,
            annot=True,
            fmt='d',
            cbar=False,
            linewidths=.5,
            xticklabels=["Absent", "Present"],
            yticklabels=["Absent", "Present"],
            ax=ax)

bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()
'''
#-- PIPELINE


# fix pipeline for testing BEFORE uncommenting
# https://towardsdatascience.com/visualizing-support-vector-machine-decision-boundary-69e7591dacea
# pipe_steps = [('pca', pca), ('SVM', model)]

model_1 = svm.SVC(gamma="scale")
pipe_steps = [('SVM', model_1)]

pipeline = Pipeline(pipe_steps, verbose=True)

from tqdm import tqdm_notebook as tqdm


search_space = {'kernel':('poly', 'rbf'),
                'C':[1, 10, 100, 1000],
                'max_iter' : [1000, 5000, 10000]
               }


cross_val_grid = GridSearchCV(model_1, search_space, cv=2)
cross_val_grid.fit(train_features, train_labels)
cv_results = pd.DataFrame(cross_val_grid.cv_results_)

print("DONE pipeline training!")

display(cv_results.sort_values(['rank_test_score']))


#-- PLOT DECISION BOUNDARY
