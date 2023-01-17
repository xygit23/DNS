import numpy as np


def DeLaLA_select(gamma, rho, layer, Y, k, l, rootWeight=0.5):
    """
    select the samples to label according to Objective Function
    :param gamma: center potential
    :param rho: local density
    :param layer: layer index of each sample
    :param alhpa: parameter of the divergence item
    :param Y: labels
    :param k: k for LMCA
    :param l: given number of samples to be labeled
    :return: labeled samples Indics
    """

    N = len(Y)
    psi = np.zeros(N, dtype=float)
    np.divide(rho, layer, psi)
    C = len(np.unique(Y))
    label_unique = np.unique(Y)

    y = [x for x in Y]

    for i in range(len(y)):
        for j in range(C):
            if y[i] == label_unique[j]:
                y[i] = j
    LabeledPerClass = np.zeros((C, k), dtype=int) - 1
    p = l - C * k  ## number of global selection
    globalSelected = np.zeros(p, dtype=int) - 1
    gamma = (gamma - min(gamma)) / (max(gamma) - min(gamma))
    BigValue = 10
    hgamma = np.zeros(N, dtype=float)
    for i in range(N):
        if np.abs(gamma[i] - 1) < 1E-3:
            hgamma[i] = BigValue
        else:
            if np.abs(gamma[i]) < 1E-3:
                hgamma[i] = 0
            else:
                hgamma[i] = 1.0 / np.log(gamma[i])

    sortedInds = SortSmallXOR(hgamma, psi, rootWeight)
    Cursor = 0
    labeled = 0
    while labeled < l:
        classID = y[sortedInds[Cursor]]
        if LabeledPerClass[classID, k - 1] > -1:  ### selection has done for the class.
            if p > 0:
                if globalSelected[p - 1] > -1:  ### global selection done
                    Cursor += 1
                    continue
                else:
                    for i in range(p):
                        if globalSelected[i] == -1:
                            globalSelected[i] = sortedInds[Cursor]  ###selected one sample in global selection
                            labeled += 1
                            Cursor += 1
                            break
            else:
                Cursor += 1
        else:
            for i in range(k):
                if LabeledPerClass[classID, i] == -1:
                    LabeledPerClass[classID, i] = sortedInds[Cursor]  ###selected one sample for a given class
                    labeled += 1
                    Cursor += 1
                    break

    result = np.append(LabeledPerClass.flatten(), globalSelected)
    # print("visited ", Cursor, " Samples!")
    return result


def SortSmallXOR(a, b, rootWeight=0.5):
    '''
    a and b are exclusively small. The function sort the values of a XOR b in the sense of being small
    :param a:
    :param b:
    :param rootWeight: if >0, tends to include more roots than divergent samples
    :return: the indics of an array, first elements are small in a or small in b.
    '''

    ###select via XOR. min-max normalization failed
    normalA = (a - np.mean(a)) / np.std(a)
    normalB = (b - np.mean(b)) / np.std(b)

    y = rootWeight * normalA * (1 - normalB) + (1 - rootWeight) * normalB * (1 - normalA)
    inds = np.argsort(y)
    result = inds[::-1]
    return result
