import numpy as np

def rep_SC(w, alpha, n):

    T=[]

    for c1 in w:
        for c2 in alpha:
            if c1==c2:
                T=T+[1]
            else:
                T=T+[0]
        T=T+[0]

    for i in range(n-len(w)):
        for c2 in alpha:
            T=T+[0]
        T=T+[1]

    return T


def rep_PBSC(w, alpha, n=3):

   # input : a word w ,  a  list of alphabet 

    T = []

    # level 1

    T = T + rep_SC(w[:n], alpha, n)
    T = T + rep_SC(w[-n:][::-1], alpha, n)

    # level 2

    split = -((-len(w)) // 2)
    [w1, w2] = [w[:split], w[split:]]

    T = T + rep_SC(w1[:n], alpha, n)
    T = T + rep_SC(w1[-n:][::-1], alpha, n)

    T = T + rep_SC(w2[:n], alpha, n)
    T = T + rep_SC(w2[-n:][::-1], alpha, n)

    # level 3

    split1 = -((-len(w)) // 3)
    split2 = 2 * split1
    [w1, w2, w3] = [w[:split1], w[split1:split2], w[split2:]]

    T = T + rep_SC(w1[:n], alpha, n)
    T = T + rep_SC(w1[-n:][::-1], alpha, n)

    T = T + rep_SC(w2[:n], alpha, n)
    T = T + rep_SC(w2[-n:][::-1], alpha, n)

    T = T + rep_SC(w3[:n], alpha, n)
    T = T + rep_SC(w3[-n:][::-1], alpha, n)

    return np.array(T)

def rep_EX(w, alpha):

    T=[]

    for c1 in alpha:
        test=0
        for c2 in w:
            if c1==c2:
                test=1

        if test==1:
            T=T+[1]
        else:
            T=T+[0]

    return T

def rep_TX(T, alpha):

    T=np.array(T)

    n=int(T.shape[0]/53)

    w=''

    for i in range(n):
        ind=np.argmax(T[i*53:(i+1)*53])
        if ind//52==0:
            w=w+alpha[ind]

    return w

def rep_PHOC(w, alpha):

    T = []

    # level 1

    T = T + rep_EX(w, alpha)

    # level 2

    split = -((-len(w)) // 2)
    [w1, w2] = [w[:split], w[split:]]

    T = T + rep_EX(w1, alpha)
    T = T + rep_EX(w2, alpha)

    # level 3

    split1 = -((-len(w)) // 3)
    split2 = 2 * split1
    [w1, w2, w3] = [w[:split1], w[split1:split2], w[split2:]]

    T = T + rep_EX(w1, alpha)
    T = T + rep_EX(w2, alpha)
    T = T + rep_EX(w3, alpha)

    # level4

    split1 = -((-len(w)) // 4)
    split2 = 2 * split1
    split3= 3 * split1
    [w1, w2, w3, w4] = [w[:split1], w[split1:split2], w[split2:split3], w[split3:]]

    T = T + rep_EX(w1, alpha)
    T = T + rep_EX(w2, alpha)
    T = T + rep_EX(w3, alpha)
    T = T + rep_EX(w4, alpha)

    return np.array(T)
