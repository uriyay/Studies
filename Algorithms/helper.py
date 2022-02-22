class OneBasedArray:
    def __init__(self, python_list):
        self._array = python_list

    def fix_index(self, index):
        if type(index) is not int:
            if index.start - 1 < 0 or index.stop - 1 < 0:
                raise IndexError('list index out of range')
            index = slice(index.start - 1, index.stop - 1, index.step)
        else:
            if index - 1 < 0:
                raise IndexError('list index out of range')
            index -= 1
        return index

    def __getitem__(self, index):
        index = self.fix_index(index)
        return self._array[index]

    def __setitem__(self, index, value):
        index = self.fix_index(index)
        self._array[index] = value

    def __len__(self):
        return len(self._array)

    def __repr__(self):
        return '\n'.join('{}: {}'.format(x + 1, self._array[x]) for x in range(len(self._array)))

    def __str__(self):
        return repr(self)

def one_based_range(start, end, step=1):
    # in algorithms with one-based pseudu-code the for loops includes the end
    return range(start, end + 1, step)
