import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from utils_pr import Utils_PR
from sklearn.metrics import confusion_matrix
pd.options.mode.chained_assignment = None



def LDA(df, threshold):
    
    # Step-1 Computing the d-dimensional mean vectors for all the classes
    d = df.shape[1]-1


    X = df.drop('target',axis=1)
    y = df['target']

    mean_vectors = []
    for cl in df['target'].unique():
        temp = df[df['target']==cl]
        temp.drop('target',axis=1,inplace=True)
        mean_vectors.append(np.mean(temp, axis=0))
    
    S_W = np.zeros((d,d))
    for cl,mv in zip(df['target'].unique(),mean_vectors):
        class_sc_mat = np.zeros((d,d),dtype = float)                  # scatter matrix for every class
        temp = df[df['target'] == cl]
        for idx in temp.index:
            row = temp.loc[idx].values[:-1]
            row = row.reshape(d,1) # make column vectors
            mv = np.asarray(mv)
            mv =  mv.reshape(d,1)
            z = row-mv
            A = np.matmul(z,z.T)
            class_sc_mat = np.add(class_sc_mat,A)
        S_W = np.add(S_W,class_sc_mat)
        
    overall_mean = np.mean(X, axis=0)
    S_B = np.zeros((d,d))
    for i,mean_vec in zip(df['target'].unique(),mean_vectors):  
        n = df[df['target']==i].shape[0]
        mean_vec = np.asarray(mean_vec)
        mean_vec = mean_vec.reshape(d,1) # make column vector
        overall_mean = np.asarray(overall_mean)
        overall_mean = overall_mean.reshape(d,1) # make column vector
        z = mean_vec - overall_mean
        A = n * (np.matmul(z,z.T))
        S_B = np.add(S_B,A)

    # Step-3
    eigen_values, eigen_vectors = np.linalg.eigh(S_B)

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
    print(eigenvector_subset.shape)
    print(X.shape)

    # Step-6
    X_reduced = np.dot(X,eigenvector_subset)

    return X_reduced

df = pd.read_csv('gender_feature_vectors.csv')
target = df['Unnamed: 1']
df.drop(columns=['Unnamed: 1', 'Unnamed: 0'], axis=1, inplace=True)
df['target'] = target
mat_reduced = LDA(df, 0.95)
cols = ['col_'+str(i) for i in range(mat_reduced.shape[1])]
utils = Utils_PR()

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

