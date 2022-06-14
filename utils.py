import numpy as np
import operator
import tensorflow as tf

class PrioritizedReplayBuffer:

    def __init__(self,b_size,buffer_beta):

        self.b_size=b_size
        self.storage=[]
        self.sum_tree=SumSegmentTree(b_size)
        self.min_tree=MinSegmentTree(b_size)
        self.buffer_beta=buffer_beta

    def add(self,*args):
        if len(self.storage)>=self.b_size:
            self.storage.pop(0)
        self.storage.append(args)

    def encode_sample(self,idxes):

        b_o,b_a,b_r,b_o_,b_d,b_w=[],[],[],[],[],[]

        for i in idxes:
            o,a,r,o_,d=self.storage[i]
            w_max=(self.min_tree.min_value*self.b_size)**(-self.buffer_beta)
            p_value=self.sum_tree[i]/self.sum_tree.total_priority
            w=((self.b_size*p_value)**(-self.buffer_beta))/w_max
            b_w.append(w)
            b_o.append(o)
            b_a.append(a)
            b_r.append(r)
            b_o_.append(o_)
            b_d.append(d)
        return(
            np.stack(b_o).astype('float32'),
            np.stack(b_a).astype('int32'),
            np.stack(b_r).astype('float32'),
            np.stack(b_o_).astype('float32'),
            np.stack(b_d).astype('int32'),
            np.stack(b_w).astype('float32'),
        )    


    def get_sample(self,batch_size):
        idxs=[]
        v_s=[np.random.sample()*self.sum_tree.total_priority for i in range(batch_size)]
        for i in v_s:
            idxs.append(self.sum_tree.get_leaf(i))
        sample=self.encode_sample(idxs)
        return sample,idxs

    def update_beta(self,beta_augmented):
        self.buffer_beta+=beta_augmented       
                 



class SegmentTree(object):

    def __init__(self, capacity, operation, neutral_element):
        assert capacity > 0 and capacity & (capacity - 1) == 0
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        if end is None:
            end = self._capacity - 1
        if end < 0:
            end += self._capacity
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(self._value[2 * idx], self._value[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]

    def update_priorities(self,p_indexs,p_values):
        for index,value in zip(p_indexs,p_values):
            self.__setitem__(index,value)    


class SumSegmentTree(SegmentTree):

    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity, operation=operator.add, neutral_element=0.0)

    def sum(self, start=0, end=None):
        return super(SumSegmentTree, self).reduce(start, end)

    def get_leaf(self, prefixsum):
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity

    @property
    def total_priority(self):
        return self.sum() # Returns the root node    


class MinSegmentTree(SegmentTree):

    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity, operation=min, neutral_element=float('inf'))

    def min(self, start=0, end=None):
        return super(MinSegmentTree, self).reduce(start, end)

    @property
    def min_value(self):
      return self.min()  