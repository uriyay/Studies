import z3

def get_queen_name(var):
    return str(var)[:-1]

def define_variables(n):
    queens_x = [z3.Int(f'q{i}x') for i in range(n)]
    queens_y = [z3.Int(f'q{i}y') for i in range(n)]
    return (queens_x, queens_y)

def get_values_constraint(n, queens_x, queens_y):
    constraints = [z3.And(0 <= x, x < n) for x in queens_x]
    constraints += [z3.And(0 <= y, y < n) for y in queens_y]
    constraint = z3.And(*constraints)
    return constraint

def get_row_constraint(queens_y):
    constraint = z3.Distinct(*queens_y)
    return constraint

def get_column_constraint(queens_x):
    constraint = z3.Distinct(*queens_x)
    return constraint

def get_diag_constraint(n, queens_x, queens_y):
    queens = list(zip(queens_x, queens_y))
    main_diag_distances = [q[0] - q[1] for q in queens]
    main_diag_constraint = z3.Distinct(main_diag_distances)
    second_diag_distances = [(n - q[0]) - q[1] for q in queens]
    second_diag_constraint = z3.Distinct(second_diag_distances)
    constraint = z3.And(main_diag_constraint, second_diag_constraint)
    return constraint

def draw(model, queens_x, queens_y, n):
    result = ''
    queen_id = 0

    queens_x_eval = {model.eval(x).as_long(): x for x in queens_x}
    queens_y_eval = {model.eval(y).as_long(): y for y in queens_y}
    #print(queens_x_eval)
    #print(queens_y_eval)

    result += '\n' + '-'*4*n + '\n'
    for i in range(n):
        for j in range(n):
            cur = ' '
            if all([i in queens_x_eval,
                    j in queens_y_eval,
                    get_queen_name(queens_x_eval[i]) == get_queen_name(queens_y_eval[j])]):
                cur = '*'
            result += '| %s ' % (cur)
        result += '\n' + '-'*4*n + '\n'
    print(result)

def solve(n=4):
    solver = z3.Solver()
    queens_x, queens_y = define_variables(n)
    values_constaint = get_values_constraint(n, queens_x, queens_y)
    row_constraint = get_row_constraint(queens_y)
    column_constraint = get_column_constraint(queens_x)
    diag_constraint = get_diag_constraint(n, queens_x, queens_y)
    solver.add(values_constaint)
    solver.add(row_constraint)
    solver.add(column_constraint)
    solver.add(diag_constraint)
    #print assertions
    print(solver.assertions)
    #check
    print('check() = {}'.format(solver.check()))
    #solve
    model = solver.model()
    print(str(model))
    #draw it
    draw(model, queens_x, queens_y, n)

if __name__ == '__main__':
    solve()
