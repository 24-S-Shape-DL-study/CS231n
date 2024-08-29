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
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        # print(scores)
        correct_class_score = scores[y[i]]
        # print(correct_class_score)

        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            # print(margin)
            if margin > 0: # 만약 테스트당한 클래스의 로스값이 0보다 크면
                loss += margin
                dW[:, j] += X[i] # 테스트 클래스의 dW에 테스트한 X값을 더하기
                # print(i, j, y[i], X[i])
                dW[:, y[i]] -= X[i] # 정답 클래스에는 빼기
                  
    # print(dW)
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # print(loss)
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

    dW += reg * 2 * W # loss를 미분한거인건 알겠는데, 왜 이 식이 나오는가?
    # print(dW, dW.shape)
    '''[[-6.84335810e+03 -6.48437153e+03 -4.11515578e+03 ... -3.96336527e+03
  -9.68902059e+03 -1.74920609e+04]
 [-1.00020183e+04 -7.82927398e+03 -3.17788582e+03 ... -3.97279449e+03
  -1.26810319e+04 -1.90347991e+04]
 [-2.07215261e+04 -1.05228912e+04  3.61709098e+03 ... -4.91727861e+03
  -2.06921177e+04 -2.50492131e+04]
 ...
 [-5.08813804e+03 -2.40207061e+03 -2.65357951e+03 ... -6.66088931e+03
   8.70931616e+03 -1.18794816e+04]
 [-1.29876004e+04 -4.97400643e+03 -6.93738571e+01 ... -2.13371714e+02
   3.23182271e+03 -1.32783953e+04]
 [-1.00000000e+00 -1.50000000e+01  6.10000000e+01 ...  6.60000000e+01
   5.30000000e+01  1.10000000e+01]]
[[-1.36867162e+01 -1.29687431e+01 -8.23031155e+00 ... -7.92673053e+00
  -1.93780412e+01 -3.49841218e+01]
 [-2.00040365e+01 -1.56585480e+01 -6.35577163e+00 ... -7.94558898e+00
  -2.53620639e+01 -3.80695982e+01]
 [-4.14430522e+01 -2.10457824e+01  7.23418196e+00 ... -9.83455723e+00
  -4.13842353e+01 -5.00984262e+01]
 ...
 [-1.01762761e+01 -4.80414123e+00 -5.30715902e+00 ... -1.33217786e+01
   1.74186323e+01 -2.37589631e+01]
 [-2.59752009e+01 -9.94801286e+00 -1.38747715e-01 ... -4.26743429e-01
   6.46364543e+00 -2.65567906e+01]
 [-2.00000036e-03 -2.99999969e-02  1.22000000e-01 ...  1.32000001e-01
   1.05999999e-01  2.20000005e-02]] (3073, 10)
loss: 8.949335'''
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
    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = X.dot(W)
    # print(scores) # [[-0.33948826  0.19449669  0.45676521 ... -0.2011599] ...] 
    # print(y) # [9 9 4 9 3 2 4 2 5 4 9 4 7 6 6 7 8 6 5 3 0 1 8 4 9 2 9 2 
    correct_class_score = scores[np.arange(num_train),y]
    # print(correct_class_score) # -0.5678326241484734, -0.2539628236463665, 0.16177976786622594, 0.0879870239416875, ...
    # print(scores.shape) # (500, 10)
    # print(correct_class_score.shape) # (500,)
    # print(correct_class_score[:, np.newaxis].shape) # (500, 1)
    # print(scores - correct_class_score[:, np.newaxis] + 1) # margin, [[1.22834436 1.76232931 2.02459783 ... 1.36667272 1.04220418 1.        ] .. ]
    
    loss = np.maximum(scores - correct_class_score[:, np.newaxis] + 1, 0)
    loss[np.arange(num_train), y] = 0 # 자기자신 빼기
    # 위에 각 i별로 계산된 loss가 있으므로, 여기에서 w 계산하자!
    mask = loss>0
    # print(mask, mask.shape)
    # # print(np.sum(mask, axis=0))
    # # print(np.sum(mask, axis=0).shape)

    # # X의 전치와 mask를 곱해 dW의 각 클래스에 대한 gradient 계산
    # print(loss.shape)
    # print(X.shape)
    # # print( X[mask].shape)
    # print(np.sum(X, axis=0))

    # print(dW.shape)
    # dW += np.sum(X[:, mask], axis=0)
    # margin이 0보다 큰 경우에 대해 dW 업데이트

    
    # 각 클래스에 대해 dW 계산
    dW = X.T.dot(mask)

    # 각 샘플에 대해 잘못 분류된 클래스의 수 계산
    incorrect_counts = np.sum(mask, axis=1)

    # 정답 클래스에 대해서는 잘못 분류된 만큼의 값만큼 빼줌
    # dW -= np.dot(X.T, incorrect_counts[:, np.newaxis]) # 이거 되나?
    for i in range(num_train):
        dW[:, y[i]] -= X[i] * incorrect_counts[i]
    dW /= num_train

    loss = np.sum(loss)
    loss/=num_train
    # print(loss) 

    # Add regularization to the loss.
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

    dW += reg * 2 * W # loss를 미분한거인건 알겠는데, 왜 이 식이 나오는가?

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
