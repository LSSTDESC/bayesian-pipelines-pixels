

class DPParams():

    def __init__(self):

        self._params = [4e-6]  # Initial value of sigma_eta^2

    def __getitem__(self, lc_val):

        return self._params[lc_val]

    def __setitem__(self, lc_val, val):

        if lc_val > len(self._params) - 1:
            for i in range(len(self._params), lc_val+1):
                self._params.append(val)
        else:
            self._params[lc_val] = val

    def init_params_values(self, val):

        for i in range(len(self._params)):
            self._params[i] = val

    def length(self):
        return len(self._params)

    def erase(self, lc_val):
        self._params.pop(lc_val)
