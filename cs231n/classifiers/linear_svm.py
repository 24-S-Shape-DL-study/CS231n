from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1] #분류해야 하는 class의 개수
    num_train = X.shape[0] #학습 데이터의 샘플 개수
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W) #모든 클래스의 점수 계산
        correct_class_score = scores[y[i]] #올바른 클래스에 해당하는 점수 
        for j in range(num_classes):
            if j == y[i]: #현재 클래스가 정답 클래스라면 건너뜀
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 모든 class에 대한 score 계산
    scores = X.dot(W)  #(N, C)
    
    # 올바른 class score 선택
    correct_class_scores = scores[np.arange(X.shape[0]), y]  #(N, )
    
    # margin 계산
    margins = np.maximum(0, scores - correct_class_scores[:, np.newaxis] + 1) 
    margins[np.arange(X.shape[0]), y] = 0  #correct class margin = 0

    # loss 계산
    loss = np.sum(margins) / X.shape[0]
    
    # loss에 reg 더함
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # gradient 계산
    margin_mask = (margins > 0).astype(float) #margin이 0보다 큰 경우에만 mask 생성
    margin_mask[np.arange(X.shape[0]), y] = -np.sum(margin_mask, axis=1)
    #각 샘플의 정답 클래스에 해당하는 위치 선택
    #각 샘플에 대해 margin이 0보다 큰 클래스의 수를 계산하고, 그 수만큼 해당 correct class 위치에 음수 할당
    
    dW = X.T.dot(margin_mask) / X.shape[0] #행렬 곱으로 기울기 계산
    
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
