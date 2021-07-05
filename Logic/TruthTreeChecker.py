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
    p[0] = []

def p_S_ID(p):
    '''
    S : ID
    '''
    p[0] = ('S', 'ID', p[1:])

def p_S_predict(p):
    '''
    S : predict
    '''
    p[0] = ('S', 'predict', p[1:])

def p_S_NOT(p):
    '''
    S : NOT S
    '''
    p[0] = ('S', 'not', p[1:])

def p_S_AND(p):
    '''
    S : LPAREN S AND S RPAREN
    '''
    p[0] = ('S', 'and', p[1:])

def p_S_OR(p):
    '''
    S : LPAREN S OR S RPAREN
    '''
    p[0] = ('S', 'or', p[1:])

def p_S_IMPL(p):
    '''
    S : LPAREN S IMPL S RPAREN
    '''
    p[0] = ('S', 'impl', p[1:])

def p_S_EQUIV(p):
    '''
    S : LPAREN S EQUIV S RPAREN
    '''
    p[0] = ('S', 'equiv', p[1:])

def p_S_FORALL(p):
    '''
    S : FORALL ID S
    '''
    p[0] = ('S', 'forall', p[1:])

def p_S_EXISTS(p):
    '''
    S : EXISTS ID S
    '''
    p[0] = ('S', 'exists', p[1:])

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
class Node:
    def __init__(self):
        self.formulas = []
        self.left = None
        self.right = None
        #list of formulas that we done handling them
        self.done = []

    def add(self, formula):
        self.formulas.append(formula)

    def set_done(self, formula):
        if formula in self.done:
            raise Exception('already done')
        self.done.append(formula)

    def is_done(self, formula):
        return formula in self.done

    def branch(self, formula1, formula2):
        self.left = Node()
        self.right = Node()
        self.left.add(formula1)
        self.right.add(formula2)

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
        return result

    def handle(self):
        #return True IFF any of the truth tree branches are true
        return any(self.handle_formulas(self.ast))

    def handle_formulas(self, formulas):
        #we need to make the tree from the all of the formulas together, not every formula by itself
        #each node will be a list of facts
        #each branch is a disunication
        #we want to first add all the none-branching formulas, then the branching ones
        for formula in formulas:
            self.handle_formula



def main(path):
    with open(path) as fp:
        text = fp.read()
    truthTree = TruthTree(text)
    truthTree.check()

if __name__ == '__main__':
    main(sys.argv[1])
