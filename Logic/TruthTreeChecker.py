import ply.lex as lex
import ply.yacc as yacc
import sys
import traceback
from re import escape

# lex part
tokens = (
    'ID', #name
    'PREDICT', #predict name
    'LPAREN', #(
    'RPAREN', #)
    'EQUAL', #=
    'NOT', #¬
    'AND', #∧
    'OR', #∨
    'IMPL', #→
    'EQUIV', #↔
    'EXISTS', #∃
    'FORALL', #∀
    'newline',
)

t_LPAREN = escape('(')
t_RPAREN = escape(')')
t_EQUAL = '='
t_NOT = '¬'
t_AND = '∧'
t_OR = '∨'
t_IMPL = '→'
t_EQUIV = '↔'
t_EXISTS = '∃'
t_FORALL = '∀'

has_tokenizing_error = False

def t_ID(t):
    #must start with lower case letter
    '[a-z]+[a-zA-Z0-9_]*'
    return t

def t_PREDICT(t):
    #must start with upper case letter
    '[A-Z]+[a-zA-Z0-9_]*'
    return t

t_ignore = '[\r ]'

def t_newline(t):
    r'\n'
    t.lexer.lineno += 1
    return t

def hexdump(data):
    #supports also unicode
    res = ' '.join('{:02x}'.format(ord(x)) for x in data)
    return res

def t_error(t):
    global has_tokenizing_error
    has_tokenizing_error = True
    print("Illegal character '{}' (hexdump: {}) in line {} char {}".format(t.value[0], hexdump(t.value[0]), t.lineno, t.lexpos), file=sys.stderr)
    import ipdb; ipdb.set_trace()
    #try to skip one character and retry to parse token from the next character
    t.lexer.skip(1)

lexer = lex.lex(debug=False)

# syntax parser
start = 'formulas'

has_syntax_error = False

def p_error(p):
    global has_syntax_error
    has_syntax_error = True
    stack_state_str = ' '.join([symbol.type for symbol in parser.symstack][1:])
    # symbol is the symbol that we got, action is the symbols that the parser expects
    print('SyntaxError: Syntax error in input! Parser State:{}, Stack:"{}", symbol:"{}", action: "{}"'.format(
        parser.state,
        stack_state_str,
        p,
        parser.action[parser.state]
    ), file=sys.stderr)
    import ipdb; ipdb.set_trace()

def log_enter():
    if debug:
        call_stack = traceback.extract_stack()
        caller = call_stack[-2]
        print("In %s() (line %d)" % (caller.name, caller.lineno))

def p_formulas(p):
    'formulas : formulas formula'
    p[0] = [] + p[1]
    p[0].append(p[2])

def p_formulas_term(p):
    'formulas : empty'
    p[0] = []

def p_formula(p):
    'formula : S newline'
    p[0] = p[1]

def p_formula_empty(p):
    'formula : newline'
    p[0] = tuple()

def p_S_ID(p):
    '''
    S : ID
    '''
    p[0] = ('S', 'id', p[1])

def p_S_predict(p):
    '''
    S : predict
    '''
    p[0] = ('S', 'predict', p[1:])

def p_S_NOT(p):
    '''
    S : NOT S
    '''
    p[0] = ('S', 'not', p[2])

def p_S_AND(p):
    '''
    S : LPAREN S AND S RPAREN
    '''
    p[0] = ('S', 'and', p[2], p[4])

def p_S_OR(p):
    '''
    S : LPAREN S OR S RPAREN
    '''
    p[0] = ('S', 'or', p[2], p[4])

def p_S_IMPL(p):
    '''
    S : LPAREN S IMPL S RPAREN
    '''
    p[0] = ('S', 'impl', p[2], p[4])

def p_S_EQUIV(p):
    '''
    S : LPAREN S EQUIV S RPAREN
    '''
    p[0] = ('S', 'equiv', p[2], p[4])

def p_S_FORALL(p):
    '''
    S : FORALL ID S
    '''
    p[0] = ('S', 'forall', p[2], p[3])

def p_S_EXISTS(p):
    '''
    S : EXISTS ID S
    '''
    p[0] = ('S', 'exists', p[2], p[3])

def p_predict(p):
    'predict : PREDICT id_list'
    p[0] = ('predict', p[1], p[2])

def p_id_list(p):
    'id_list : id_list ID'
    p[0] = []
    p[0] += p[1]
    p[0].append(p[2])

def p_id_list_term(p):
    'id_list : empty'
    p[0] = []

def p_empty(p):
    'empty :'
    pass

parser = yacc.yacc()

# logic truth tree
class InconsistencyError(Exception):
    pass

class Terminal:
    def __init__(self, name, is_negated=False):
        self.name = name
        self.is_negated = is_negated

    def __eq__(self, other):
        return self.name == other.name and self.is_negated == other.is_negated

    def __repr__(self):
        return '<Terminal {}{}>'.format('¬' if self.is_negated else '', self.name)

class Node:
    def __init__(self):
        self.formulas = []
        self.terminals = []
        self.left = None
        self.right = None
        self.parent = None
        self.delayed = []

    def add(self, formula):
        if formula not in self.formulas:
            self.formulas.append(formula)

    def add_terminal(self, terminal):
        if terminal not in self.terminals:
            self.terminals.append(terminal)

    def check(self):
        stack = []
        cur = self
        results = []
        while cur:
            if cur.left:
                stack.append(cur.left)
            if cur.right:
                stack.append(self.right)
            if not cur.left and not cur.right:
                #its a leaf
                results.append(self.check_leaf(cur))
            #next
            cur = stack.pop() if stack else None
        return any(results)

    def check_leaf(self, leaf):
        #go backward and check for terminals or negated terminals
        terminals = []
        cur = leaf
        while cur:
            for t in cur.terminals:
                if any(x.is_negated != t.is_negated for x in terminals):
                    #found contradiction!
                    print('found terminal {}{} while there is {}{} on this branch'.format(
                        '¬' if t.is_negated else '', t.name, '¬' if not t.is_negated else '', t.name))
                    return False
                terminals.append(t)
            #next
            cur = cur.parent
        return True

    def branch(self, formula1, formula2):
        self.left = Node()
        self.right = Node()
        self.left.add(formula1)
        self.right.add(formula2)
        self.left.parent = self
        self.right.parent = self

class TruthTree:
    def __init__(self, text):
        self.text = text
        self.ast = None
        self.root = Node()
        self.has_errors = False

    def check(self):
        self.ast = parser.parse(self.text, debug=False)
        if has_tokenizing_error or has_syntax_error:
            self.has_errors = True
        result = self.handle()
        print('The tree is {}consistent'.format('in' if not result else ''))
        return result

    def handle(self):
        self.handle_formulas(self.ast)
        import ipdb; ipdb.set_trace()
        return self.root.check()

    def handle_formulas(self, formulas):
        #we need to make the tree from the all of the formulas together, not every formula by itself
        #each node will be a list of facts
        #each branch is a disunication
        #we want to first add all the none-branching formulas, then the branching ones
        #each time handle_formula() is called and a branch is made - get the lowest level and split it
        for formula in formulas:
            self.handle_formula(formula, self.root)

    def handle_formula(self, formula, node):
        self.handle_S(formula, node)

    def handle_S(self, S, node):
        S_type = S[1]
        getattr(self, 'handle_S_' + S_type)(S, node)

    def handle_S_and(self, S, node):
        #add the 2 formulas to node
        self.handle_S(S[2], node)
        self.handle_S(S[3], node)

    def handle_S_or(self, S, node):
        #split the node
        node.branch(S[2], S[3])
        self.handle_S(S[2], node.left)
        self.handle_S(S[3], node.right)

    def handle_S_impl(self, S, node):
        #S[2] -> S[3] <=> ¬S[2] ∨ S[3]
        neg_S2 = ('S', 'not', S[2])
        node.branch(neg_S2, S[3])
        self.handle_S(neg_S2, node.left)
        self.handle_S(S[3], node.right)

    def handle_S_equiv(self, S, node):
        #S[2] <-> S[3] <=> (S[2] ∧ S[3]) ∨ (¬S[2] ∧ ¬S[3])
        neg_S2 = ('S', 'not', S[2])
        neg_S3 = ('S', 'not', S[3])
        opt1 = ('S', 'and', S[2], S[3])
        opt2 = ('S', 'and', neg_S2, neg_S3)
        node.branch(opt1, opt2)
        self.handle_S(opt1, node.left)
        self.handle_S(opt2, node.right)

    def handle_S_id(self, S, node):
        #add this var to the current node
        node.add_terminal(Terminal(S[2]))

    def handle_S_not(self, S, node):
        expr = S[2]
        if expr[1] == 'id':
            #add the negated var
            node.add_terminal(Terminal(expr[2], is_negated=True))
        elif expr[1] == 'not':
            #double negation
            #!!p == q
            #just don't handle the negation
            self.handle_S(expr[2], node)
        elif expr[1] == 'and':
            #!(p && q) == (!p || !q)
            #split
            neg_expr2 = ('S', 'not', expr[2])
            neg_expr3 = ('S', 'not', expr[3])
            node.branch(neg_expr2, neg_expr3)
            self.handle_S(neg_expr2, node.left)
            self.handle_S(neg_expr3, node.right)
        elif expr[1] == 'or':
            #!(p || q) == (!p && !q)
            neg_expr2 = ('S', 'not', expr[2])
            neg_expr3 = ('S', 'not', expr[3])
            new_S = ('S', 'and', neg_expr2, neg_expr3)
            self.handle_S(new_S, node)
        elif expr[1] == 'impl':
            #!(p -> q) == !(!p || q) == (p && !q)
            neg_expr3 = ('S', 'not', expr[3])
            new_S = ('S', 'and', expr[2], neg_expr3)
            self.handle_S(new_S, node)
        elif expr[1] == 'equiv':
            #!(p <-> q) == (p && !q) || (!p && q)
            neg_expr2 = ('S', 'not', expr[2])
            neg_expr3 = ('S', 'not', expr[3])
            opt1 = ('S', 'and', expr[2], neg_expr3)
            opt2 = ('S', 'and', neg_expr2, expr[3])
            node.branch(opt1, opt2)
            self.handle_S(opt1, node.left)
            self.handle_S(opt2, node.right)



def main(path):
    with open(path) as fp:
        text = fp.read()
    truthTree = TruthTree(text)
    truthTree.check()

if __name__ == '__main__':
    main(sys.argv[1])
