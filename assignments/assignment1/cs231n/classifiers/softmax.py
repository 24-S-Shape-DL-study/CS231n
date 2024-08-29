from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        # print(correct_class_score) # -0.1109435881461524, 0.2610155928686205, 0.29353419853818785


        correct_class_exps = np.exp(correct_class_score) # exp
        # print(correct_class_exps) # 0.8949892355069454, 1.2982479086849414, 1.3411590443047074
        sum_exps=np.sum(np.exp(scores)) 
        # print(sum_exps) # 9.987473122406586, 10.658867176354741, 10.974180473191716
        loss-=np.log(correct_class_exps/sum_exps) # # 더했을 때 1 되도록 normalize, log로 loss 계산
        # print(np.log(correct_class_exps/sum_exps)) # -2.412275208111679, -2.105376551587527, -2.102011085488403

        dW[:,y[i]] -= X[i]
        # print(dW.shape, X[i][:, np.newaxis].shape, np.exp(scores)[np.newaxis, :].shape)
        dW += (np.dot(X[i][:, np.newaxis], np.exp(scores)[np.newaxis, :])) / sum_exps
                  
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    loss += reg * np.sum(W * W)
    dW += reg * 2 * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    scores = X.dot(W)
    correct_class_score = scores[np.arange(num_train),y]
    # print(correct_class_score) # [-1.10943588e-01  2.61015593e-01  2.93534199e-01 ...]
    
    correct_class_exps = np.exp(correct_class_score) # exp
    # print(correct_class_exps) # [0.89498924 1.29824791 1.34115904 ...
    sum_exps=np.sum(np.exp(scores), axis=1) 
    # print(sum_exps) # [ 9.98747312 10.65886718 10.97418047 ...

    # print(np.log(correct_class_exps/sum_exps)) # -2.41227521 -2.10537655 -2.10201109 ...
    loss-=np.sum(np.log(correct_class_exps/sum_exps)) # # 더했을 때 1 되도록 normalize, log로 loss 계산


    # 사실 이렇게 하면 안될거같은데;;; 어케해야하징
    # for i in range(num_train):
    #   dW[:, y[i]] -= X[i]
    #   dW += (np.dot(X[i][:, np.newaxis], np.exp(scores[i])[np.newaxis, :])) / sum_exps[i]
    
    # 각 클래스의 지수값들을 구합니다.
    exp_scores = np.exp(scores)  # (num_train, C)

    # sum_exps를 사용하여 각 점수에 대한 확률을 구합니다.
    probs = exp_scores / sum_exps[:, np.newaxis]  # (num_train, C)

    # 레이블 행렬을 생성합니다.
    indicators = np.zeros_like(probs)  # (num_train, C)
    indicators[np.arange(num_train), y] = 1  # (num_train, C)

    # 벡터화된 방식으로 dW를 계산합니다.
    dW = -np.dot(X.T, indicators) + np.dot(X.T, probs)  # (D, C)

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    loss += reg * np.sum(W * W)
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
