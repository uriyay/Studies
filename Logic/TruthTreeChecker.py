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

def t_error(t):
    global has_tokenizing_error
    has_tokenizing_error = True
    print("Illegal character '{}' in line {} char {}".format(t.value[0], t.lineno, t.lexpos), file=sys.stderr)
    #try to skip one character and retry to parse token from the next character
    t.lexer.skip(1)

lexer = lex.lex(debug=False)

# syntax parser
start = 'S'

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

def log_enter():
    if debug:
        call_stack = traceback.extract_stack()
        caller = call_stack[-2]
        print("In %s() (line %d)" % (caller.name, caller.lineno))

def p_S_ID(p):
    '''
    S : ID
    '''
    p[0] = ('S', p[1:])

def p_S_predict(p):
    '''
    S : predict
    '''
    p[0] = ('S', p[1:])

def p_S_NOT(p):
    '''
    S : NOT S
    '''
    p[0] = ('S', p[1:])

def p_S_AND(p):
    '''
    S : LPAREN S AND S RPAREN
    '''
    p[0] = ('S', p[1:])

def p_S_OR(p):
    '''
    S : LPAREN S OR S RPAREN
    '''
    p[0] = ('S', p[1:])

def p_S_IMPL(p):
    '''
    S : LPAREN S IMPL S RPAREN
    '''
    p[0] = ('S', p[1:])

def p_S_EQUIV(p):
    '''
    S : LPAREN S EQUIV S RPAREN
    '''
    p[0] = ('S', p[1:])

def p_S_FORALL(p):
    '''
    S : FORALL ID S
    '''
    p[0] = ('S', p[1:])

def p_S_EXISTS(p):
    '''
    S : EXISTS ID S
    '''
    p[0] = ('S', p[1:])

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
class TruthTree:
    def __init__(self, text):
        self.text = text
        self.ast = None
        self.has_errors = False

    def check(self):
        self.ast = parser.parse(self.text, debug=False)
        import ipdb; ipdb.set_trace()
        if has_tokenizing_error or has_syntax_error:
            self.has_errors = True


def main(path):
    with open(path) as fp:
        text = fp.read()
    TruthTree = LogicTruthTree(text)
    truthTree.check()

if __name__ == '__main__':
    main(sys.argv[1])
