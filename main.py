import ast
import astunparse
import optimizer

with open('sample.py', 'r') as f:
    code = f.read()

tree = ast.parse(code)
optimizer.Optimizer().visit(tree)
new_code = astunparse.unparse(tree)

with open('_sample.py', 'w') as f:
    f.write(new_code)
