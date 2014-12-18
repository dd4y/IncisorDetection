'''
Created on 09.05.2014.

@author: Borka
'''
import numpy as np


def PCAa(all_samples):

    mean_x = np.mean(all_samples[0,:])
    mean_y = np.mean(all_samples[1,:])
    mean_z = np.mean(all_samples[2,:])

    mean_vector = np.array([[mean_x],[mean_y],[mean_z]])

    print('Mean Vector:\n', mean_vector)

    cov_mat = np.cov([all_samples[0,:],all_samples[1,:],all_samples[2,:]])
    print('Covariance Matrix:\n', cov_mat)

    scatter_matrix = np.zeros((3,3))
    for i in range(all_samples.shape[1]):
        scatter_matrix += (all_samples[:,i].reshape(3,1) - mean_vector).dot((all_samples[:,i].reshape(3,1) - mean_vector).T)
    print('Scatter Matrix:\n', scatter_matrix)
    # eigenvectors and eigenvalues for the from the scatter matrix
    eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)

    # eigenvectors and eigenvalues for the from the covariance matrix
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)

    for i in range(len(eig_val_sc)):
        eigvec_sc = eig_vec_sc[:,i].reshape(1,3).T
        eigvec_cov = eig_vec_cov[:,i].reshape(1,3).T
        assert eigvec_sc.all() == eigvec_cov.all(), 'Eigenvectors are not identical'

    print('Eigenvector {}: \n{}'.format(i+1, eigvec_sc))
    print('Eigenvalue {} from scatter matrix: {}'.format(i+1, eig_val_sc[i]))
    print('Eigenvalue {} from covariance matrix: {}'.format(i+1, eig_val_cov[i]))
    print('Scaling factor: ', eig_val_sc[i]/eig_val_cov[i])
    print(40 * '-')




def PCAb(data):
    """
     """
    # usually a good idea to mean center your data first:
    deviationMatrix = (data.T - np.mean(data, axis=1)).T
    covarianceMatrix = np.cov(deviationMatrix)
    eigenvalues, principalComponents = np.linalg.eig(covarianceMatrix)
    # projection of the data in the new space
    score = np.dot(principalComponents.T,deviationMatrix)
    # sort the principal components in decreasing order of corresponding eigenvalue
    indexList = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[indexList]
    principalComponents = principalComponents[:, indexList]

    return score,eigenvalues, principalComponents
    # computing eigenvalues and eigenvectors of covariance matrix

def pcaD(X, nb_components=0):

    ''' Do a PCA analysis on X
    @param X:                np.array containing the samples =6
                             shape = (nb samples, nb dimensions of each sample)
    @param nb_components:    the nb components we're interested in
    @return: return the nb_components largest eigenvalues and eigenvectors of the covariance matrix and return the average sample'''

    [n,d] = X.shape
    if (nb_components <= 0) or (nb_components>n):
        nb_components = n
    mu = X.mean(axis=0)
    X = X - mu
    if n>d:
        C = np.dot(X.T,X)
        [eigenvalues,eigenvectors] = np.linalg.eigh(C)
    else:
        C = np.dot(X,X.T)
        [eigenvalues,eigenvectors] = np.linalg.eigh(C)
        eigenvectors = np.dot(X.T,eigenvectors)
        for i in xrange(n):
            eigenvectors[:,i] = eigenvectors[:,i]/np.linalg.norm(eigenvectors[:,i])

    ix = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[ix]
    eigenvectors = eigenvectors[:,ix]
    eigenvalues = eigenvalues[0:nb_components].copy()
    eigenvectors = eigenvectors[:,0:nb_components].copy()
    totalvar=np.sum(eigenvalues)
    majorvar=np.sum(eigenvalues[0:2])
    if (majorvar>=0.89*totalvar):
        print 'vece'
    else:
        print majorvar
    return eigenvalues, eigenvectors, mu

def project(W, X, mu):
    '''
    Project X on the space spanned by the vectors in W.
    mu is the average image.
    '''
    if mu is None:
        return np.dot(X,W)
    return np.dot(X - mu, W)


def reconstruct(W, Y, mu):
    '''
    Reconstruct an image based on its PCA-coefficients Y, the eigenvectors W and the average mu.
    '''
    if mu is None:
        return np.dot(Y,W.T)
    return np.dot(Y, W.T) + mu

def normalize(img):
    '''
    Normalize an image such that it min=0 , max=255 and type is np.uint8
    '''
    return (img*(255./(np.max(img)-np.min(img)))+np.min(img)).astype(np.uint8)

