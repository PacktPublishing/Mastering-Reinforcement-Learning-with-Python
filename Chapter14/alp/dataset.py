"""
Copied from:
https://github.com/flowersteam/teachDeepRL/blob/master/teachDRL/teachers/utils/dataset.py
@misc{portelas2019teacher,
      title={Teacher algorithms for curriculum learning of Deep RL in continuously parameterized environments},
      author={Rémy Portelas and Cédric Colas and Katja Hofmann and Pierre-Yves Oudeyer},
      year={2019},
      eprint={1910.07224},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
"""

try:
    import numpy as np
    import scipy.spatial
except:
    print(
        "Can't import scipy.spatial (or numpy). Is scipy (or numpy) correctly installed ?"
    )
    exit(1)

DATA_X = 0
DATA_Y = 1


class Databag(object):
    """Hold a set of vectors and provides nearest neighbors capabilities"""

    def __init__(self, dim):
        """
        :arg dim:  the dimension of the data vectors
        """
        self.dim = dim
        self.reset()

    def __repr__(self):
        return "Databag(dim={0}, data=[{1}])".format(
            self.dim, ", ".join(str(x) for x in self.data)
        )

    def add(self, x):
        assert len(x) == self.dim
        self.data.append(np.array(x))
        self.size += 1
        self.nn_ready = False

    def reset(self):
        """Reset the dataset to zero elements."""
        self.data = []
        self.size = 0
        self.kdtree = None  # KDTree
        self.nn_ready = False  # if True, the tree is up-to-date.

    def nn(self, x, k=1, radius=np.inf, eps=0.0, p=2):
        """Find the k nearest neighbors of x in the observed input data
        :arg x:      center
        :arg k:      the number of nearest neighbors to return (default: 1)
        :arg eps:    approximate nearest neighbors.
                     the k-th returned value is guaranteed to be no further than
                     (1 + eps) times the distance to the real k-th nearest neighbor.
        :arg p:      Which Minkowski p-norm to use. (default: 2, euclidean)
        :arg radius: the maximum radius (default: +inf)
        :return:     distance and indexes of found nearest neighbors.
        """
        assert (
            len(x) == self.dim
        ), "dimension of input {} does not match expected dimension {}.".format(
            len(x), self.dim
        )
        k_x = min(k, self.size)
        # Because linear models requires x vector to be extended to [1.0]+x
        # to accomodate a constant, we store them that way.
        return self._nn(np.array(x), k_x, radius=radius, eps=eps, p=p)

    def get(self, index):
        return self.data[index]

    def iter(self):
        return iter(self.data)

    def _nn(self, v, k=1, radius=np.inf, eps=0.0, p=2):
        """Compute the k nearest neighbors of v in the observed data,
        :see: nn() for arguments descriptions.
        """
        self._build_tree()
        dists, idxes = self.kdtree.query(
            v, k=k, distance_upper_bound=radius, eps=eps, p=p
        )
        if k == 1:
            dists, idxes = np.array([dists]), [idxes]
        return dists, idxes

    def _build_tree(self):
        """Build the KDTree for the observed data"""
        if not self.nn_ready:
            self.kdtree = scipy.spatial.cKDTree(self.data)
            self.nn_ready = True

    def __len__(self):
        return self.size


class Dataset(object):
    """Hold observations an provide nearest neighbors facilities"""

    @classmethod
    def from_data(cls, data):
        """ Create a dataset from an array of data, infering the dimension from the datapoint """
        if len(data) == 0:
            raise ValueError("data array is empty.")
        dim_x, dim_y = len(data[0][0]), len(data[0][1])
        dataset = cls(dim_x, dim_y)
        for x, y in data:
            assert len(x) == dim_x and len(y) == dim_y
            dataset.add_xy(x, y)
        return dataset

    @classmethod
    def from_xy(cls, x_array, y_array):
        """Create a dataset from two arrays of data.
        :note: infering the dimensions for the first elements of each array.
        """
        if len(x_array) == 0:
            raise ValueError("data array is empty.")
        dim_x, dim_y = len(x_array[0]), len(y_array[0])
        dataset = cls(dim_x, dim_y)
        for x, y in zip(x_array, y_array):
            assert len(x) == dim_x and len(y) == dim_y
            dataset.add_xy(x, y)
        return dataset

    def __init__(self, dim_x, dim_y, lateness=0, max_size=None):
        """
        :arg dim_x:  the dimension of the input vectors
        :arg dim_y:  the dimension of the output vectors
        """
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.lateness = lateness
        self.max_size = max_size

        self.reset()

    # The two next methods are used for plicling/unpickling the object (because cKDTree cannot be pickled).
    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict["kdtree"]
        return odict

    def __setstate__(self, dict):
        self.__dict__.update(dict)
        self.nn_ready = [False, False]
        self.kdtree = [None, None]

    def reset(self):
        """Reset the dataset to zero elements."""
        self.data = [[], []]
        self.size = 0
        self.kdtree = [None, None]  # KDTreeX, KDTreeY
        self.nn_ready = [False, False]  # if True, the tree is up-to-date.
        self.kdtree_y_sub = None
        self.late_points = 0

    def add_xy(self, x, y=None):
        # assert len(x) == self.dim_x, (len(x), self.dim_x)
        # assert self.dim_y == 0 or len(y) == self.dim_y, (len(y), self.dim_y)
        self.data[0].append(x)
        if self.dim_y > 0:
            self.data[1].append(y)
        self.size += 1
        if self.late_points == self.lateness:
            self.nn_ready = [False, False]
            self.late_points = 0
        else:
            self.late_points += 1
        # Reduce data size
        if self.max_size and self.size > self.max_size:
            n = self.size - self.max_size
            del self.data[0][:n]
            del self.data[1][:n]
            self.size = self.max_size

    def add_xy_batch(self, x_list, y_list):
        assert len(x_list) == len(y_list)
        self.data[0] = self.data[0] + x_list
        self.data[1] = self.data[1] + y_list
        self.size += len(x_list)
        # Reduce data size
        if self.max_size and self.size > self.max_size:
            n = self.size - self.max_size
            del self.data[0][:n]
            del self.data[1][:n]
            self.size = self.max_size

    def get_x(self, index):
        return self.data[0][index]

    def set_x(self, x, index):
        self.data[0][index] = x

    def get_x_padded(self, index):
        return np.append(1.0, self.data[0][index])

    def get_y(self, index):
        return self.data[1][index]

    def set_y(self, y, index):
        self.data[1][index] = y

    def get_xy(self, index):
        return self.get_x(index), self.get_y(index)

    def set_xy(self, x, y, index):
        self.set_x(x, index)
        self.set_y(y, index)

    def get_dims(self, index, dims_x=None, dims_y=None, dims=None):
        if dims is None:
            return np.hstack(
                (
                    np.array(self.data[0][index])[dims_x],
                    np.array(self.data[1][index])[np.array(dims_y) - self.dim_x],
                )
            )
        else:
            if max(dims) < self.dim_x:
                return np.array(self.data[0][index])[dims]
            elif min(dims) > self.dim_x:
                return np.array(self.data[1][index])[np.array(dims) - self.dim_x]
            else:
                raise NotImplementedError

    def iter_x(self):
        return iter(d for d in self.data[0])

    def iter_y(self):
        return iter(self.data[1])

    def iter_xy(self):
        return zip(self.iter_x(), self.data[1])

    def __len__(self):
        return self.size

    def nn_x(self, x, k=1, radius=np.inf, eps=0.0, p=2):
        """Find the k nearest neighbors of x in the observed input data
        @see Databag.nn() for argument description
        @return  distance and indexes of found nearest neighbors.
        """
        assert len(x) == self.dim_x
        k_x = min(k, self.size)
        # Because linear models requires x vector to be extended to [1.0]+x
        # to accomodate a constant, we store them that way.
        return self._nn(DATA_X, x, k=k_x, radius=radius, eps=eps, p=p)

    def nn_y(self, y, k=1, radius=np.inf, eps=0.0, p=2):
        """Find the k nearest neighbors of y in the observed output data
        @see Databag.nn() for argument description
        @return  distance and indexes of found nearest neighbors.
        """
        assert len(y) == self.dim_y
        k_y = min(k, self.size)
        return self._nn(DATA_Y, y, k=k_y, radius=radius, eps=eps, p=p)

    def nn_dims(self, x, y, dims_x, dims_y, k=1, radius=np.inf, eps=0.0, p=2):
        """Find the k nearest neighbors of a subset of dims of x and y in the observed output data
        @see Databag.nn() for argument description
        @return  distance and indexes of found nearest neighbors.
        """
        assert len(x) == len(dims_x)
        assert len(y) == len(dims_y)
        if len(dims_x) == 0:
            kdtree = scipy.spatial.cKDTree(
                [
                    np.array(data_y)[np.array(dims_y) - self.dim_x]
                    for data_y in self.data[DATA_Y]
                ]
            )
        elif len(dims_y) == 0:
            kdtree = scipy.spatial.cKDTree(
                [np.array(data_x)[dims_x] for data_x in self.data[DATA_X]]
            )
        else:
            kdtree = scipy.spatial.cKDTree(
                [
                    np.hstack(
                        (
                            np.array(data_x)[dims_x],
                            np.array(data_y)[np.array(dims_y) - self.dim_x],
                        )
                    )
                    for data_x, data_y in zip(self.data[DATA_X], self.data[DATA_Y])
                ]
            )
        dists, idxes = kdtree.query(
            np.hstack((x, y)), k=k, distance_upper_bound=radius, eps=eps, p=p
        )
        if k == 1:
            dists, idxes = np.array([dists]), [idxes]
        return dists, idxes

    def _nn(self, side, v, k=1, radius=np.inf, eps=0.0, p=2):
        """Compute the k nearest neighbors of v in the observed data,
        :arg side  if equal to DATA_X, search among input data.
                     if equal to DATA_Y, search among output data.
        @return  distance and indexes of found nearest neighbors.
        """
        self._build_tree(side)
        dists, idxes = self.kdtree[side].query(
            v, k=k, distance_upper_bound=radius, eps=eps, p=p
        )
        if k == 1:
            dists, idxes = np.array([dists]), [idxes]
        return dists, idxes

    def _build_tree(self, side):
        """Build the KDTree for the observed data
        :arg side  if equal to DATA_X, build input data tree.
                     if equal to DATA_Y, build output data tree.
        """
        if not self.nn_ready[side]:
            self.kdtree[side] = scipy.spatial.cKDTree(
                self.data[side], compact_nodes=False, balanced_tree=False
            )  # Those options are required with scipy >= 0.16
            self.nn_ready[side] = True


class BufferedDataset(Dataset):
    """Add a buffer of a few points to avoid recomputing the kdtree at each addition"""

    def __init__(self, dim_x, dim_y, buffer_size=200, lateness=5, max_size=None):
        """
        :arg dim_x:  the dimension of the input vectors
        :arg dim_y:  the dimension of the output vectors
        """

        self.buffer_size = buffer_size
        self.lateness = lateness
        self.buffer = Dataset(dim_x, dim_y, lateness=self.lateness)

        Dataset.__init__(self, dim_x, dim_y, lateness=0)
        self.max_size = max_size

    def reset(self):
        self.buffer.reset()
        Dataset.reset(self)

    def add_xy(self, x, y=None):
        if self.buffer.size < self.buffer_size:
            self.buffer.add_xy(x, y)
        else:
            self.data[0] = self.data[0] + self.buffer.data[0]
            if self.dim_y > 0:
                self.data[1] = self.data[1] + self.buffer.data[1]
            self.size += self.buffer.size
            self.buffer = Dataset(self.dim_x, self.dim_y, lateness=self.lateness)
            self.nn_ready = [False, False]
            self.buffer.add_xy(x, y)

            # Reduce data size
            if self.max_size and self.size > self.max_size:
                n = self.size - self.max_size
                del self.data[0][:n]
                del self.data[1][:n]
                self.size = self.max_size

    def add_xy_batch(self, x_list, y_list):
        assert len(x_list) == len(y_list)
        Dataset.add_xy_batch(self, self.buffer.data[0], self.buffer.data[1])
        self.buffer = Dataset(self.dim_x, self.dim_y, lateness=self.lateness)
        Dataset.add_xy_batch(self, x_list, y_list)
        self.nn_ready = [False, False]

    def get_x(self, index):
        if index >= self.size:
            return self.buffer.data[0][index - self.size]
        else:
            return self.data[0][index]

    def set_x(self, x, index):
        if index >= self.size:
            self.buffer.set_x(x, index - self.size)
        else:
            self.data[0][index] = x

    def get_x_padded(self, index):
        if index >= self.size:
            return np.append(1.0, self.buffer.data[0][index - self.size])
        else:
            return np.append(1.0, self.data[0][index])

    def get_y(self, index):
        if index >= self.size:
            return self.buffer.data[1][index - self.size]
        else:
            return self.data[1][index]

    def set_y(self, y, index):
        if index >= self.size:
            self.buffer.set_y(y, index - self.size)
        else:
            self.data[1][index] = y

    def get_dims(self, index, dims_x=None, dims_y=None, dims=None):
        if index >= self.size:
            return self.buffer.get_dims(index - self.size, dims_x, dims_y, dims)
        else:
            return Dataset.get_dims(self, index, dims_x, dims_y, dims)

    def iter_x(self):
        return iter(d for d in self.data[0] + self.buffer.data[0])

    def iter_y(self):
        return iter(self.data[1] + self.buffer.data[1])

    def iter_xy(self):
        return zip(self.iter_x(), self.data[1] + self.buffer.data[1])

    def __len__(self):
        return self.size + self.buffer.size

    def nn_x(self, x, k=1, radius=np.inf, eps=0.0, p=2):
        """Find the k nearest neighbors of x in the observed input data
        @see Databag.nn() for argument description
        @return  distance and indexes of found nearest neighbors.
        """
        assert len(x) == self.dim_x
        k_x = min(k, self.__len__())
        # Because linear models requires x vector to be extended to [1.0]+x
        # to accomodate a constant, we store them that way.
        return self._nn(DATA_X, x, k=k_x, radius=radius, eps=eps, p=p)

    def nn_y(self, y, dims=None, k=1, radius=np.inf, eps=0.0, p=2):
        """Find the k nearest neighbors of y in the observed output data
        @see Databag.nn() for argument description
        @return  distance and indexes of found nearest neighbors.
        """
        if dims is None:
            assert len(y) == self.dim_y
            k_y = min(k, self.__len__())
            return self._nn(DATA_Y, y, k=k_y, radius=radius, eps=eps, p=p)
        else:
            return self.nn_y_sub(y, dims, k, radius, eps, p)

    def nn_dims(self, x, y, dims_x, dims_y, k=1, radius=np.inf, eps=0.0, p=2):
        """Find the k nearest neighbors of a subset of dims of x and y in the observed output data
        @see Databag.nn() for argument description
        @return  distance and indexes of found nearest neighbors.
        """
        if self.size > 0:
            dists, idxes = Dataset.nn_dims(
                self, x, y, dims_x, dims_y, k, radius, eps, p
            )
        else:
            return self.buffer.nn_dims(x, y, dims_x, dims_y, k, radius, eps, p)
        if self.buffer.size > 0:
            buffer_dists, buffer_idxes = self.buffer.nn_dims(
                x, y, dims_x, dims_y, k, radius, eps, p
            )
            buffer_idxes = [i + self.size for i in buffer_idxes]
            ziped = zip(dists, idxes)
            buffer_ziped = zip(buffer_dists, buffer_idxes)
            sorted_dists_idxes = sorted(ziped + buffer_ziped, key=lambda di: di[0])
            knns = sorted_dists_idxes[:k]
            return [knn[0] for knn in knns], [knn[1] for knn in knns]
        else:
            return dists, idxes

    def _nn(self, side, v, k=1, radius=np.inf, eps=0.0, p=2):
        """Compute the k nearest neighbors of v in the observed data,
        :arg side  if equal to DATA_X, search among input data.
                     if equal to DATA_Y, search among output data.
        @return  distance and indexes of found nearest neighbors.
        """
        if self.size > 0:
            dists, idxes = Dataset._nn(self, side, v, k, radius, eps, p)
        else:
            return self.buffer._nn(side, v, k, radius, eps, p)
        if self.buffer.size > 0:
            buffer_dists, buffer_idxes = self.buffer._nn(side, v, k, radius, eps, p)
            buffer_idxes = [i + self.size for i in buffer_idxes]
            if dists[0] <= buffer_dists:
                return dists, idxes
            else:
                return buffer_dists, buffer_idxes
            ziped = zip(dists, idxes)
            buffer_ziped = zip(buffer_dists, buffer_idxes)
            sorted_dists_idxes = sorted(ziped + buffer_ziped, key=lambda di: di[0])
            knns = sorted_dists_idxes[:k]
            return [knn[0] for knn in knns], [knn[1] for knn in knns]
        else:
            return dists, idxes
