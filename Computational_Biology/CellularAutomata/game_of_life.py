from time import sleep

class CellularAutomata:
    def __init__(self, n, states, init_state):
        self.d = 2
        self.n = n
        self.init_state = init_state
        self._mat = [[self.init_state] * self.n for _ in range(self.n)]
        self._gen_id = 0

    def get_neighborhood(self, i, j):
        raise NotImplementedError()

    def transition_rule(self, cur_state, states):
        raise NotImplementedError()

    def next(self):
        new_mat = [[self.init_state] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                new_mat[i][j] = self.transition_rule(
                    self._mat[i][j],
                    self.get_neighborhood(i, j)
                )
        self._gen_id += 1
        self._mat = new_mat

    def set_state(self, i, j, state):
        self._mat[i][j] = state

    def display(self):
        output = ''
        output += '-' * self.n
        for i in range(self.n):
            for j in range(self.n):
                output += {0: ' ', 1: 'O'}[self._mat[i][j]]
            output += '\n'
        output += '-' * self.n
        print(output + '\n')

class GameOfLife(CellularAutomata):
    def __init__(self, n):
        super().__init__(n, [0, 1], 0)

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
        return reS

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
            self._ca.set_state(pos[0] + {'h': 0, 'v': 1}[h_or_v] * i,
                    pos[1] + {'h': 1, 'v': 0}[h_or_v] * i,
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
        self._ca.set_state(pos[0], pos[1], state)

    def block(self):
        """insert block"""
        pos = self._get_pos_input()
        if not pos:
            return
        for i in range(2):
            for j in range(2):
                self._ca.set_state(pos[0] + i, pos[1] + j, 1)

    def animate(self):
        """play an animation until Ctrl+C pressed"""
        try:
            while True:
                self._ca.display()
                self.next()
                sleep(0.2)
        except KeyboardInterrupt:
            pass


def main():
    cli = CLI()
    cli.run()

if __name__ == '__main__':
    main()
