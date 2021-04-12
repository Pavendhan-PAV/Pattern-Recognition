import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from utils_pr import Utils_PR

def PCA(X, threshold):

    # Step-1
    X_meaned = X - np.mean(X, axis=0)

    # Step-2
    cov_mat = np.cov(X_meaned, rowvar=False)

    # Step-3
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

    # Step-4
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]

    # print(pd.Series(sorted_eigenvalue))
    explained_variances = []
    for i in range(len(sorted_eigenvalue)):
        explained_variances.append(
            sorted_eigenvalue[i] / np.sum(sorted_eigenvalue))
    explained_variances = np.asarray(explained_variances)
    sorted_index = np.argsort(explained_variances)[::-1]
    sorted_explained_var = explained_variances[sorted_index]

    num_components = 0
    var = 0
    for i in sorted_explained_var:
        if var < threshold:
            var += i
            num_components += 1
        else:
            break
    # Step-5
    eigenvector_subset = sorted_eigenvectors[:, 0:num_components]

    # Step-6
    X_reduced = np.dot(eigenvector_subset.transpose(),
                       X_meaned.transpose()).transpose()

    return X_reduced


utils = Utils_PR()

df = pd.read_csv('gender_feature_vectors.csv')
target = df['Unnamed: 1']
df.drop(columns=['Unnamed: 1', 'Unnamed: 0'], axis=1, inplace=True)

X = df.to_numpy()
df['target'] = target
mat_reduced = PCA(X, 0.95)
cols = ['col_'+str(i) for i in range(mat_reduced.shape[1])]


principal_df = pd.DataFrame(mat_reduced, columns=cols)
principal_df['target'] = target
print('Shape of original dataframe : {}'.format(df.shape))
print('Shape of reduced dataframe using PCA : {}'.format(principal_df.shape))

male = principal_df.query('target == "male"')
female = principal_df.query('target == "female"')

test = male.head(10)
test = test.append(female.head(10))
train = principal_df.drop(test.index, axis=0)

X, y = train.iloc[:, :-1], train.iloc[:, -1]
X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]
classifier = GaussianNB()
classifier.fit(X, y)
predictions = classifier.predict(X_test)
print(utils.accuracy(y_test, predictions))
utils.plot_confusion_matrix(y_test, predictions)
