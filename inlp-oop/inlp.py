
from typing import Dict
import numpy as np
import scipy
from typing import List, Tuple
from tqdm import tqdm
import random
import warnings
import inlp_dataset_handler
import inlp_linear_model

def get_rowspace_projection(W: np.ndarray) -> np.ndarray:
    """
    :param W: the matrix over its nullspace to project
    :return: the projection matrix over the rowspace
    """

    if np.allclose(W, 0):
        w_basis = np.zeros_like(W.T)
    else:
        w_basis = scipy.linalg.orth(W.T) # orthogonal basis
    
    w_basis = w_basis * np.sign(w_basis[0][0]) # handle sign ambiguity
    P_W = w_basis.dot(w_basis.T) # orthogonal projection on W's rowspace
    
    return P_W

def get_projection_to_intersection_of_nullspaces(rowspace_projection_matrices: List[np.ndarray], input_dim: int):
    """
    Given a list of rowspace projection matrices P_R(w_1), ..., P_R(w_n),
    this function calculates the projection to the intersection of all nullspasces of the matrices w_1, ..., w_n.
    uses the intersection-projection formula of Ben-Israel 2013 http://benisrael.net/BEN-ISRAEL-NOV-30-13.pdf: 
    N(w1)∩ N(w2) ∩ ... ∩ N(wn) = N(P_R(w1) + P_R(w2) + ... + P_R(wn))
    :param rowspace_projection_matrices: List[np.array], a list of rowspace projections
    :param dim: input dim
    """
    
    I = np.eye(input_dim)
    Q = np.sum(rowspace_projection_matrices, axis = 0)
    P = I - get_rowspace_projection(Q)
    
    return P
    
def debias_by_specific_directions(directions: List[np.ndarray], input_dim: int):
    """
    the goal of this function is to perform INLP on a set of user-provided directiosn (instead of learning those directions).
    :param directions: list of vectors, as numpy arrays.
    :param input_dim: dimensionality of the vectors.
    """
    
    rowspace_projections = []
    
    for v in directions:
        P_v = get_rowspace_projection(v)
        rowspace_projections.append(P_v)

    P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)
    
    return P


def run_INLP(num_classifiers: int, input_dim: int, is_autoregressive: bool, min_accuracy: float,
                             dataset_handler: inlp_dataset_handler.DatasetHandler, model: inlp_linear_model.LinearModel) -> Tuple[np.ndarray]:
    """

    :param num_classifiers: number of iterations (equivalent to number of dimensions to remove)
    :param input_dim: size of input vectors
    :param is_autoregressive: whether to train the ith classiifer on the data projected to the nullsapces of w1,...,wi-1
    :param min_accuracy: above this threshold, ignore the learned classifier
    :param dataset_handler: an isntance of DatasetHandler, in charge of storing the data & projecting it
    :param inlp_linear_model: an instance of LinearModel, in charge of training the model and returning its weights
    :return: P, the debiasing projection; rowspace_projections, the list of all rowspace projection; Ws, the list of all calssifiers.
    """
    
    #if dropout_rate > 0 and is_autoregressive:
    #    warnings.warn("Note: when using dropout with autoregressive training, the property w_i.dot(w_(i+1)) = 0 no longer holds.")
    
    I = np.eye(input_dim)
    rowspace_projections = []
    Ws = []
    dataset_handler.reinitialize()

    pbar = tqdm(range(num_classifiers))
    
    for i in pbar:

        # initialize models & dataset

        model.initialize_model()

        # train model, record accuracy

        acc = model.train_model(dataset_handler)
        pbar.set_description("iteration: {}, accuracy: {}".format(i, acc))
        if acc < min_accuracy: continue

        # collect parameters and rowspace projections

        W = model.get_weights()
        Ws.append(W)
        P_rowspace_wi = get_rowspace_projection(W) # projection to W's rowspace
        rowspace_projections.append(P_rowspace_wi)

        if is_autoregressive:
            
            """
            to ensure numerical stability, explicitly project to the intersection of the nullspaces found so far (instaed of doing X = P_iX,
            which is problematic when w_i is not exactly orthogonal to w_i-1,...,w1, due to e.g inexact argmin calculation).
            """
            # use the intersection-projection formula of Ben-Israel 2013 http://benisrael.net/BEN-ISRAEL-NOV-30-13.pdf: 
            # N(w1)∩ N(w2) ∩ ... ∩ N(wn) = N(P_R(w1) + P_R(w2) + ... + P_R(wn))
            
            P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)
            # project  
            dataset_handler.apply_projection(P)

    """
    calculae final projection matrix P=PnPn-1....P2P1
    since w_i.dot(w_i-1) = 0, P2P1 = I - P1 - P2 (proof in the paper); this is more stable.
    by induction, PnPn-1....P2P1 = I - (P1+..+PN). We will use instead Ben-Israel's formula to increase stability and also generalize to the non-orthogonal case (e.g. with dropout),
    i.e., we explicitly project to intersection of all nullspaces (this is not critical at this point; I-(P1+...+PN) is roughly as accurate as this provided no dropout & regularization)
    """
    
    P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)

    return P, rowspace_projections, Ws


if __name__ == '__main__':
    
    from sklearn.linear_model import SGDClassifier, Perceptron, LogisticRegression
        
    N = 10000
    d = 300
    X = np.random.rand(N, d) - 0.5
    Y = np.array([1 if sum(x) > 0 else 0 for x in X]) #X < 0 #np.random.rand(N) < 0.5 #(X + 0.01 * (np.random.rand(*X.shape) - 0.5)) < 0 #np.random.rand(5000) < 0.5
    #Y = np.array(Y, dtype = int)
    
    num_classifiers = 200
    classifier_class = SGDClassifier #Perceptron
    input_dim = d
    is_autoregressive = True
    min_accuracy = 0.0
    
    P, rowspace_projections, Ws = get_debiasing_projection(classifier_class, {}, num_classifiers, input_dim, is_autoregressive, min_accuracy, X, Y, X, Y, by_class = False)
    
    I = np.eye(P.shape[0])
    P_alternative = I - np.sum(rowspace_projections, axis = 0)
    P_by_product = I.copy()
    
    for P_Rwi in rowspace_projections:
    
        P_Nwi = I - P_Rwi
        P_by_product = P_Nwi.dot(P_by_product)
        
    
    """testing"""
    
    # validate that P = PnPn-1...P2P1 (should be true only when w_i.dot(w_(i+1)) = 0, in autoregressive training)
    
    if is_autoregressive:
        assert np.allclose(P_alternative, P) 
        assert np.allclose(P_by_product, P)
    
    # validate that P is a projection
     
    assert np.allclose(P.dot(P), P) 
    
    # validate that P projects to N(w1)∩ N(w2) ∩ ... ∩ N(wn)
    
    x = np.random.rand(d) - 0.5
    for w in Ws:
    
        assert np.allclose(np.linalg.norm(w.dot(P.dot(x))), 0.0)
   
    # validate that each two classifiers are orthogonal (this is expected to be true only with autoregressive training)
    
    if is_autoregressive:
        for i,w in enumerate(Ws):
    
            for j, w2 in enumerate(Ws):
        
                if i == j: continue
                
                assert np.allclose(np.linalg.norm(w.dot(w2.T)), 0) 
    
