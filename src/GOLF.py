import numpy as np
import collections


def GetSublt(Pa, AL, Pnode):
    queue = collections.deque()
    queue.append(Pnode)

    while len(queue) != 0:
        Node = queue.popleft()
        indes = [i for i, x in enumerate(Pa) if x == Node]

        for ind in indes:
            AL = np.append(AL, ind)
            queue.append(ind)
    return AL


class LeadingTree:
    """
    Leading Tree
    """
    def __init__(self, data, dc, lt_num, D):
        self.edge_index = data.edge_index.numpy()
        self.x = data.x.cpu().numpy()
        self.dc = dc
        self.lt_num = lt_num
        self.D = D.cpu().numpy()

        self.density = None
        self.Pa = None
        self.delta = None
        self.gamma = None
        self.gamma_D = None
        self.Q = None
        self.AL = [np.zeros((0, 1), dtype=int) for i in range(lt_num)]  # AL[i] store all indexes of a subtree
        self.ALR = [np.zeros((0, 1), dtype=int) for i in range(lt_num)]
        self.layer = np.zeros(self.x.shape[0], dtype=int)

    def ComputeLocalDensity(self, D, dc):
        """
        Calculate the local density of samples
        :param D: The Euclidean distance of all samples
        :param dc:Bandwidth parameters
        :return:
        self.density: local density of all samples
        self.Q: Sort the density index in descending order
        """
        tempMat1 = np.exp(-(D ** 2))
        tempMat = np.power(tempMat1, dc ** (-2))
        self.density = np.sum(tempMat, 1, dtype='float64') - 1
        self.Q = np.argsort(self.density)[::-1]

    def ComputeParentNode(self, Q, edge_index):
        """
        Calculate the distance to the nearest data point of higher density (delta) and the parent node (Pa)
        :param D: The Euclidean distance of all samples
        :param Q:Sort by index in descending order of sample local density
        :return:
        self.delta: the distance of the sample to the closest data point with a higher density
        self.Pa: the index of the parent node of the sample
        """

        self.delta = np.zeros(len(Q))
        self.Pa = np.zeros(len(Q), dtype=int)
        for i in range(len(Q)):
            neighbor = edge_index[1][edge_index[0] == Q[i]]
            if neighbor.shape[0] == 0: # for citeseer
                neighbor = np.array(Q[i]).reshape(1, -1)
            elif neighbor.shape[0] > 1:
                neighbor = neighbor[neighbor != Q[i]]  # neighbor of current node
            if i == 0 or (neighbor.shape[0] == 1 and neighbor[0] == Q[i]):
                self.Pa[Q[i]] = -1
                self.delta[Q[i]] = np.min(self.density[neighbor])
            else:
                self.Pa[Q[i]] = neighbor[np.argmax(self.density[neighbor])]
                self.delta[Q[i]] = np.max(self.density[neighbor])

    def ProCenter(self, density, delta, Pa):
        """
        Calculate the probability of being chosen as the center node and Disconnect the Leading Tree
        :param density: local density of all samples
        :param delta: the distance of the sample to the closest data point with a higher density
        :param Pa: the index of the parent node of the sample
        :return:
        self.gamma: the probability of the sample being chosen as a center node
        self.gamma_D: Sort the gamma index in descending order
        """
        self.gamma = density * delta
        self.gamma_D = np.argsort(self.gamma)[::-1]

        # Disconnect the Leading Tree
        for i in range(self.lt_num):
            Pa[self.gamma_D[i]] = -1

    def GetSubtree(self, gamma_D, lt_num):
        """
         Subtree
        :param gamma_D:
        :param lt_num: the number of subtrees
        :return:
        self.AL: AL[i] store indexes of a subtrees, i = {0, 1, ..., lt_num-1}
        """
        for i in range(lt_num):
            self.AL[i] = np.append(self.AL[i], gamma_D[i])  # center node of subtrees
            self.AL[i] = GetSublt(self.Pa, self.AL[i], gamma_D[i])  # get the whole subtree by the center node

    def GetSubtreeR(self, gamma_D, lt_num, Q, pa):
        """
         Subtree
        :param gamma_D:
        :param lt_num: the number of subtrees
        :return:
        self.AL: AL[i] store indexes of a subtrees, i = {0, 1, ..., lt_num-1}
        """
        for i in range(lt_num):
            self.ALR[i] = np.append(self.ALR[i], gamma_D[i])

        N = len(gamma_D)
        treeID = np.zeros((N,1),dtype=int)-1
        for i in range(lt_num):
            treeID[gamma_D[i]]=i

        for nodei in range(N): ### casscade label assignment
            curInd = Q[nodei]
            if treeID[curInd]>-1:
                continue

            else:
                paID = pa[curInd]
                self.layer[curInd] = self.layer[paID]+1
                curTreeID = treeID[paID]
                treeID[curInd] = curTreeID
                self.ALR[curTreeID[0]] = np.append(self.ALR[curTreeID[0]], curInd)

    def GetLayer(self, Pnode, layer, l):
        """
        Calculate nodes' layer
        :param Pnode: center nodes
        :param layer: samples' layer
        :param l: current layer
        :return:
        self.layer: layer of all samples
        """
        queue = collections.deque()
        queue.append(Pnode)
        while len(queue) != 0:
            Node = queue.popleft()
            indes = [i for i, x in enumerate(self.Pa) if x in Node]
            layer[indes] = l
            if len(indes) != 0:
                queue.append(indes)
            l = l + 1

    def Edges(self, Pa):  # store edges of subtrees
        """

        :param Pa:  the index of the parent node of the sample
        :return:
        self. edges: pairs of child node and parent node
        """
        edgesO = np.array(list(zip(range(len(Pa)), Pa)))
        ind = edgesO[:, 1] > -1
        self.edges = edgesO[ind,]

    def fit(self):
        self.ComputeLocalDensity(self.D, self.dc)
        self.ComputeParentNode(self.Q, self.edge_index)
        self.ProCenter(self.density, self.delta, self.Pa)

        self.GetSubtreeR(self.gamma_D, self.lt_num, self.Q, self.Pa)
        self.AL = self.ALR

        self.Edges(self.Pa)
        self.layer = self.layer + 1
        self.gamma_D1 = self.gamma_D.reshape(1,-1)
        self.Q1 = self.Q.reshape(1,-1)

