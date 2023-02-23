

class LatentClasses():

    def __init__(self,):
        self._ngals = 0
        self._n_per_class = [0]
        self._allocated = False

    def __getitem__(self, i):
        return self._labels[i]

    def __setitem__(self, i, val):
        self.DecrementCountInClass(self._label[i])
        self._labels[i] = val
        self.IncrementCountInClass(val)

    def Allocate(self, ngals):

        self._ngals = ngals
        if not self._allocated:
            self._labels = [0 for i in range(ngals)]
        self._n_per_class[0] = ngals
        self._allocated = True

    def GetNumClusters(self):
        return len(self._n_per_class)

    def UniqueClusters(self, i):
        return i

    def GetNumGalInCluster(self, c_val):
        return self._n_per_class[c_val]

    def GetGalIndicesInCluster(self, c_val, igal):
        j = 0
        i = 0
        while i != self._ngals:
            if self._labels[i] == c_val:
                igal[j] = i
                j += 1
            i += 1
        return igal

    def RemoveLatentClass(self, i):
        status = self.DecrementCountInClass(self._labels[i])
        self._labels[i] = 99999999
        return status

    def IsNewClass(self, c_val):
        new_c = False
        if c_val >= len(self._n_per_class):
            new_c = True

        return new_c

    def InsertClassIndex(self, i, c_val):

        self._labels[i] = c_val
        if self.IsNewClass(c_val):
            self.AddLatentClass(c_val)
        else:
            self.IncrementCountInClass(c_val)

    def DecrementCountInClass(self, c_val):
        status = 0
        self._n_per_class[c_val] -= 1
        if self._n_per_class[c_val] == 0:
            status = 1
            self._n_per_class.pop(c_val)
            # Need to decrement label numbers for labels greater than the one
            # deleted...
            for i in range(self._ngals):
                if self._labels[i] >= c_val:
                    self._labels[i] -= 1
        return status

    def IncrementCountInClass(self, c_val):
        self._n_per_class[c_val] += 1

    def AddLatentClass(self, c_val):
        self._n_per_class.append(1)
