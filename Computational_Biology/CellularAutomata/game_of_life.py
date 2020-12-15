from time import sleep
import os
import ipdb
import traceback
import sys
import numpy as np
from pprint import pprint

from cellular_automata import CellularAutomata

class GameOfLife(CellularAutomata):
    def __init__(self, n):
        super().__init__(d=2, n=n, states=[0, 1], init_state=0)

    def get_neighborhood(self, i, j):
        neighbors = []
        for x in range(i - 1, i + 2):
            for y in range(j - 1, j + 2):
                if all([0 <= x < self.n,
                        0 <= y < self.n,
                        x != i or y != j]):
                    neighbors.append(self._mat[x][y])
        return neighbors

    def transition_rule(self, cur_state, states):
        states_sum = sum(states)
        if cur_state == 0 and states_sum == 3:
            #exactly 3 live - revive rule
            return 1
        elif cur_state == 1:
            if states_sum in (2, 3):
                return 1
            else:
                return 0
        #happens only if cur_state == 0
        return cur_state

    def display(self):
        output = ''
        output += '-' * self.n + '\n'
        for i in range(self.n):
            for j in range(self.n):
                output += {0: ' ', 1: 'O'}[self._mat[i][j]]
            output += '\n'
        output += '-' * self.n
        print(output + '\n')

class CLI:
    def __init__(self):
        self._commands = [x for x in dir(self)
                if not x.startswith('_') and x != 'run']
        self._ca = GameOfLife(10)

    def help(self):
        """show help"""
        print('commands are:')
        for func_name in self._commands:
            doc = getattr(self, func_name).__doc__
            print(f'> {func_name} - {doc}')

    def run(self):
        print('Welcome to the game of life!')
        print('Insert "help" in order to display the avaiable commands')
        try:
            while True:
                cmd = input('> ')
                if cmd in ('exit', 'quit', 'q'):
                    break

                func_name = None
                if cmd in self._commands:
                    func_name = cmd
                # match by the starting part, but only if it doesn't match for more than one command
                matches = [x for x in self._commands if x.startswith(cmd)]
                if matches and len(matches) == 1:
                    func_name = matches[0]

                if func_name not in self._commands:
                    print(f'Invalid command "{cmd}"')
                else:
                    getattr(self, func_name)()
        except KeyboardInterrupt:
            pass

    def create(self):
        """create a cellular automata for Game Of Life"""
        try:
            n = int(input('insert the number of rows and columns: '))
        except:
            print('invalid parameter')
            return
        self._ca = GameOfLife(n)

    def _get_pos_input(self):
        try:
            x = int(input('insert row index: '))
            y = int(input('insert column index: '))
            assert 0 <= x < self._ca.n
            assert 0 <= y < self._ca.n
        except:
            print('Invalid parameter')
            return None
        return x,y

    def _input_num(self, text):
        try:
            res = int(input(text))
        except:
            print('Invalid parameter')
            return None
        return res

    def blink(self):
        """Put blink pattern"""
        pos = self._get_pos_input()
        if not pos:
            return
        h_or_v = input('horizontal(h) or vertical (v)? ')
        if h_or_v not in ('h', 'v'):
            print('Invalid parameter')
            return

        for i in range(3):
            self._ca.set_state((pos[0] + {'h': 0, 'v': 1}[h_or_v] * i,
                    pos[1] + {'h': 1, 'v': 0}[h_or_v] * i),
                    1)

    def display(self):
        """display the matrix"""
        self._ca.display()

    def next(self):
        """step one generation"""
        self._ca.next()

    def set(self):
        """set cell state"""
        pos = self._get_pos_input()
        state = self._input_num('insert state: ')
        if not pos or not state:
            return
        self._ca.set_state(pos, state)

    def block(self):
        """insert block"""
        pos = self._get_pos_input()
        if not pos:
            return
        for i in range(2):
            for j in range(2):
                self._ca.set_state((pos[0] + i, pos[1] + j), 1)

    def glider(self):
        """insert glider"""
        pos = self._get_pos_input()
        if not pos:
            return
        glider_shape = [
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1]
        ]
        for i in range(len(glider_shape)):
            for j in range(len(glider_shape[i])):
                self._ca.set_state((pos[0] + i, pos[1] + j), glider_shape[i][j])

    def load_shape(self):
        """load shape to the board from file"""
        path = input("Insert shape path: ")
        if not os.path.exists(path):
            print('Error: file not exists')
            return
        shape = np.load(path)
        pos = self._get_pos_input()
        if not pos:
            return
        for i in range(len(shape)):
            for j in range(len(shape[i])):
                self._ca.set_state((pos[0] + i, pos[1] + j), shape[i][j])

    def dump_shape(self):
        """assemble a shape and dump it to a file"""
        if np.all(self._ca._mat == self._ca.init_state):
            print('Error: the board is empty!')
            return
        path = input('Insert path: ')
        shape = np.copy(self._ca._mat)
        i = 0
        while i < shape.shape[0]:
            if all(x == 0 for x in shape[i]):
                #remove this row
                shape = np.delete(shape, i, 0)
            else:
                i += 1

        j = 0
        while j < shape.shape[1]:
            if all(x == 0 for x in shape[:,j]):
                #remove this row
                shape = np.delete(shape, j, 1)
            else:
                j += 1
        
        shape.dump(path)

    def animate(self):
        """play an animation until Ctrl+C pressed"""
        try:
            while True:
                self._ca.display()
                self.next()
                sleep(0.2)
        except KeyboardInterrupt:
            pass

    def debug(self):
        """open ipdb"""
        ipdb.set_trace()

    def load(self):
        """load a prepared board"""
        path = input("Insert board data path: ")
        if not os.path.exists(path):
            print('File not exists!')
            return
        with open(path, 'rb') as fp:
            data = fp.read()
        self._ca.loads(data)

    def dump(self):
        """dump the CA's board to a file"""
        path = input("Insert board data path: ")
        data = self._ca.dumps()
        with open(path, 'wb') as fp:
            fp.write(data)

    def rot90(self):
        """rotate the board by 90 degrees"""
        self._ca._mat = np.rot90(self._ca._mat)

    def clear(self):
        """clear the board"""
        self._ca._mat.fill(self._ca.init_state)


def main(cmd=None, *args):
    if cmd == 'disp':
        path = args[0]
        board = np.load(path)
        pprint(board)
        return

    cli = CLI()
    try:
        cli.run()
    except:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)

if __name__ == '__main__':
    main(*sys.argv[1:])
