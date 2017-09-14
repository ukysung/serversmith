import ast
import sys
import copy
PYTHON3 = (sys.version_info >= (3,))
PYTHON2 = not PYTHON3

if PYTHON3:
    BYTES_TYPE = bytes
    UNICODE_TYPE = str
    INT_TYPES = (int,)
else:
    BYTES_TYPE = str
    UNICODE_TYPE = unicode
    INT_TYPES = (int, long)

STR_TYPES = (BYTES_TYPE, UNICODE_TYPE)
IMMUTABLE_ITERABLE_TYPES = STR_TYPES + (tuple, frozenset)
ITERABLE_TYPES = IMMUTABLE_ITERABLE_TYPES + (set, list)
FLOAT_TYPES = INT_TYPES + (float,)
COMPLEX_TYPES = FLOAT_TYPES + (complex,)
BUILTIN_ACCEPTING_ITERABLE = ('dict', 'frozenset', 'list', 'set', 'tuple')


def iter_all_ast(node):
    yield node
    for field, value in ast.iter_fields(node):
        if isinstance(value, list):
            for item in value:
                if isinstance(item, ast.AST):
                    for child in iter_all_ast(item):
                        yield child
        elif isinstance(value, ast.AST):
            for child in iter_all_ast(value):
                yield child


def ast_contains(tree, obj_type):
    if isinstance(tree, list):
        return any(ast_contains(node, obj_type) for node in tree)
    else:
        return any(isinstance(node, obj_type) for node in iter_all_ast(tree))


def clone_node_list(node_list):
    return copy.deepcopy(node_list)


def remove_duplicate_pass(node_list):
    # Remove duplicate pass instructions
    index = 1
    while index < len(node_list):
        if isinstance(node_list[index], ast.Pass):
            del node_list[index]
        else:
            index += 1
    if (len(node_list) > 1
    and isinstance(node_list[0], ast.Pass)):
        del node_list[0]


def is_empty_body(node_list):
    if len(node_list) == 0:
        return True
    if len(node_list) != 1:
        return False
    node = node_list[0]
    return isinstance(node, ast.Pass)


def check_func_args(node, min_narg=None, max_narg=None):
    keywords = node.keywords
    starargs = node.starargs
    kwargs = node.kwargs

    if keywords or starargs or kwargs:
        return False
    if min_narg is not None and len(node.args) < min_narg:
        return False
    if max_narg is not None and len(node.args) > max_narg:
        return False
    return True


def copy_lineno(node, new_node):
    ast.fix_missing_locations(new_node)
    ast.copy_location(new_node, node)
    return new_node


def new_constant(node, value):
    if isinstance(value, bool):
        name = "True" if value else "False"
        if sys.version_info >= (3, 4):
            new_node = ast.NameConstant(value=value)
        else:
            new_node = ast.Name(id=name, ctx=ast.Load())
    elif isinstance(value, COMPLEX_TYPES):
        new_node = ast.Num(n=value)
    elif isinstance(value, UNICODE_TYPE):
        if PYTHON3:
            new_node = ast.Str(s=value)
        else:
            new_node = ast.Str(s=value)
    elif isinstance(value, BYTES_TYPE):
        if PYTHON3:
            new_node = ast.Bytes(s=value)
        else:
            new_node = ast.Str(s=value)
    elif value is None:
        if sys.version_info >= (3, 4):
            new_node = ast.NameConstant(value=None)
        else:
            new_node = ast.Name(id="None", ctx=ast.Load())
    elif isinstance(value, tuple):
        elts = [new_constant(node, elt) for elt in value]
        new_node = ast.Tuple(elts=elts, ctx=ast.Load())
    elif isinstance(value, frozenset):
        if all(isinstance(elt, UNICODE_TYPE) for elt in value):
            arg = new_constant(node, UNICODE_TYPE().join(sorted(value)))
        elif all(isinstance(elt, BYTES_TYPE) for elt in value):
            arg = new_constant(node, BYTES_TYPE().join(sorted(value)))
        else:
            elts = [new_constant(node, elt) for elt in value]
            arg = ast.Tuple(elts=elts, ctx=ast.Load())
            copy_lineno(node, arg)
        func = ast.Name(id='frozenset', ctx=ast.Load())
        new_node = ast.Call(func, [arg], [], None, None)
    else:
        raise NotImplementedError("unable to create an AST object for constant: %r" % (value,))
    return copy_lineno(node, new_node)


def sort_set_elts(elts):
    elts = list(elts)
    try:
        # sort elements for astoptimizer unit tests
        elts.sort()
    except TypeError:
        # elements may be unsortable
        pass
    return elts


def new_constant_list(node, elts):
    return [new_constant(node, elt) for elt in elts]


def new_list_elts(node, elts=None):
    if elts is None:
        elts = []
    new_node = ast.List(elts=elts, ctx=ast.Load())
    return copy_lineno(node, new_node)


def new_list(node, iterable=()):
    elts = new_constant_list(node, iterable)
    return new_list_elts(node, elts)


def new_set_elts(node, elts=None):
    if elts is None:
        elts = []
    new_node = ast.Set(elts=elts)
    return copy_lineno(node, new_node)


def new_set(node, iterable=()):
    elts = sort_set_elts(iterable)
    elts = new_constant_list(node, elts)
    return new_set_elts(node, elts)


def new_tuple_elts(node, elts=None):
    if elts is None:
        elts = []
    new_node = ast.Tuple(elts=elts, ctx=ast.Load())
    return copy_lineno(node, new_node)


def new_tuple(node, iterable=()):
    elts = new_constant_list(node, iterable)
    return new_tuple_elts(node, elts)


def new_literal(node, value):
    if isinstance(value, list):
        return new_list(node, value)
    elif sys.version_info >= (2, 7) and isinstance(value, set):
        return new_set(node, value)
    else:
        return new_constant(node, value)


def new_call(node, name, *args):
    # name: str
    # args: ast objects
    name = ast.Name(id=name, ctx=ast.Load())
    copy_lineno(node, name)
    new_node = ast.Call(
        func=name,
        args=list(args),
        keywords=[],
        starargs=None,
        kwargs=None)
    return copy_lineno(node, new_node)


class Namespace:
    def __init__(self):
        self._aliases_enabled = True
        self.qualnames = {}
        self._removed_aliases = set()
        self._vars_enabled = False
        self.values = {}
        self._unassigned = set()

    @property
    def vars_enabled(self):
        return self._vars_enabled

    def enable_experimental_vars(self):
        self._vars_enabled = True

    def copy(self):
        ns = Namespace()
        ns._aliases_enabled = self._aliases_enabled
        ns.qualnames = self.qualnames.copy()
        ns._removed_aliases = self._removed_aliases
        return ns

    def add_alias(self, qualname, name):
        if name in self._removed_aliases:
            return
        if name in self.qualnames:
            if self.qualnames[name] == qualname:
                return
            self._removed_aliases.add(name)
            del self.qualnames[name]
            return
        self.qualnames[name] = qualname

    def disable_aliases(self):
        self._aliases_enabled = False
        self.qualnames.clear()

    def _get_qualname(self, name):
        if name in self.qualnames:
            return self.qualnames[name]
        fullname = name
        prefixes = []
        while '.' in name:
            prefix, name = name.split(".", 1)
            prefixes.append(prefix)
            key = '.'.join(prefixes)
            if key in self._unassigned:
                break
            if key in self.qualnames:
                return self.qualnames[key] + '.' + name
        if self._aliases_enabled:
            return fullname
        else:
            return None

    def is_builtin_shadowed(self, name):
        if name in self.qualnames:
            return True
        if not self._aliases_enabled:
            return True
        if name in self._unassigned:
            return True
        if name in self.values:
            return True
        return False

    def get_qualname(self, name):
        qualname = self._get_qualname(name)
        if qualname is None:
            return None
        if qualname not in self._unassigned:
            return qualname
        else:
            return None

    def assign(self, qualname, value):
        if not self._vars_enabled:
            return
        if qualname in self._unassigned:
            return
        self.values[qualname] = value

    def unassign(self, qualname):
        self._unassigned.add(qualname)
        self.values.pop(qualname, None)

    def disable_vars(self):
        self._vars_enabled = False
        self.values.clear()

    def get_var(self, qualname):
        if self._vars_enabled:
            return self.values.get(qualname, None)
        else:
            return None


def output(msg):
    print(msg)


class Optimizer(ast.NodeTransformer):
    def __init__(self):
        ast.NodeTransformer.__init__(self)
        self.namespace = Namespace()
        self.info_func = output
        self.is_conditional = None

    def load_name(self, name):
        qualname = self.namespace.get_qualname(name)
        if qualname is None:
            return None
        value = self.namespace.get_var(qualname)
        if value is not None:
            return value

        return None

    def check_func(self, node, name, min_narg=None, max_narg=None):
        if not isinstance(node, ast.Call):
            return False
        if not isinstance(node.func, ast.Name):
            return False
        qualname = self.namespace.get_qualname(node.func.id)
        if qualname is None:
            return False
        if isinstance(name, str):
            if qualname != name:
                return False
        else:
            if qualname not in name:
                return False
        return check_func_args(node, min_narg, max_narg)

    def check_builtin_func(self, node, name, min_narg, max_narg):
        return self.check_func(node, name, min_narg, max_narg)

    def get_literal(self, node, to_type=None, check_length=True):
        if isinstance(node, ast.List):
            if to_type and not issubclass(list, to_type):
                return None

            result = []
            for elt in node.elts:
                literal = self.get_literal(elt)
                if literal is None:
                    return None
                result.append(literal)
            return result
        elif PYTHON3 is False and isinstance(node, ast.Set):
            if to_type and not issubclass(set, to_type):
                return None

            result = set()
            for elt in node.elts:
                literal = self.get_literal(elt)
                if literal is None:
                    return None
                result.add(literal)
            return result
        else:
            return self.get_constant(node, to_type)

    def get_constant(self, node, to_type=None):
        if node is None:
            constant = None
        elif isinstance(node, ast.Num):
            constant = node.n
        elif isinstance(node, ast.Str):
            constant = node.s
        elif PYTHON3 and isinstance(node, ast.Bytes):
            constant = node.s
        elif isinstance(node, ast.Name):
            constant = self.load_name(node.id)
        elif isinstance(node, ast.Tuple):
            if to_type and not issubclass(tuple, to_type):
                return None

            elts = node.elts
            constants = []
            for elt in elts:
                constant = self.get_constant(elt)
                if constant is None:
                    return None
                constants.append(constant)
            return tuple(constants)
        elif self.check_builtin_func(node, 'frozenset', 0, 1):
            if to_type and not issubclass(frozenset, to_type):
                return None

            if len(node.args) == 1:
                arg = self.get_literal(node.args[0], ITERABLE_TYPES)
                if arg is None:
                    return None

                return frozenset(arg)
            else:
                return frozenset()
        else:
            return None

        if to_type and not isinstance(constant, to_type):
            return None

        return constant

    def visit_list(self, node_list):
        new_node_list = []
        for node in node_list:
            new_node = self.visit(node)
            if new_node is None:
                continue
            elif isinstance(new_node, ast.AST):
                new_node_list.append(new_node)
            elif isinstance(new_node, list):
                assert all(isinstance(node, ast.AST) for node in new_node)
                new_node_list.extend(new_node)
            else:
                raise TypeError(type(new_node))
        self.optimize_node_list(new_node_list)
        return new_node_list

    def info(self, message):
        self.info_func(message)

    def log_node_removal(self, node, message='Remove dead code'):
        if self.filename:
            filename = self.filename
        else:
            filename = '<string>'
        code = ast.dump(node)
        if len(code) > 100:
            code = code[:100] + '...'
        self.info('{}:{}: {}: {}'.format(filename, node.lineno, message, code))

    def replace_var(self, node, name, value):
        replace = ReplaceVariable(name, value)
        return replace.visit(node)

    def try_unroll_listcomp(self, node):
        if len(node.generators) != 1:
            return

        generator = node.generators[0]

        if generator.ifs:
            return

        itercst = self.get_constant(generator.iter, IMMUTABLE_ITERABLE_TYPES)
        if itercst is None:
            return

        target = generator.target
        if not isinstance(target, ast.Name):
            return
        target_id = target.id

        items = []
        for const in itercst:
            item = []
            value = new_constant(node, const)
            elt = clone_node_list(node.elt)
            elt = self.replace_var(elt, target_id, value)
            items.append(elt)

        return new_list_elts(node, items)

    def try_unroll_loop(self, node):
        iter_constant = self.get_constant(node.iter, IMMUTABLE_ITERABLE_TYPES)
        if iter_constant is None:
            return None

        target = node.target
        if not isinstance(target, ast.Name):
            return None
        target_id = target.id

        if ast_contains(node.body, (ast.Break, ast.Continue)) or ast_contains(node.orelse, (ast.Break, ast.Continue)):
            return None

        was_unassigned = target_id in self.namespace._unassigned
        self.namespace._unassigned.add(target_id)
        node.body = self.visit_list(node.body, conditional=True)
        node.orelse = self.visit_list(node.orelse, conditional=True)

        if not was_unassigned:
            self.namespace._unassigned.remove(target_id)

        if is_empty_body(node.body):
            self.log_node_removal(node)

            value = new_constant(node, iter_constant[-1])
            assign = ast.Assign(targets=[target], value=value)
            copy_lineno(node, assign)

            node_list = [assign]
            node_list.extend(node.orelse)
            return self.if_block(node, node_list)

        unroll = []
        for cst in iter_constant:
            value = new_constant(node, cst)
            assign = ast.Assign(targets=[target], value=value)
            copy_lineno(node.body[0], assign)
            unroll.append(assign)
            body = clone_node_list(node.body)
            unroll.extend(body)
        unroll.extend(node.orelse)

        unroll = self.visit_list(unroll)
        return self.if_block(node, unroll)

    def visit_For(self, node):
        self.try_unroll_loop(node)
        return node

    def _get_assign_name(self, node):
        if isinstance(node, ast.Name):
            return (node.id, True)
        elif isinstance(node, ast.Attribute):
            # var.attr = value, var1.attr1.attr2 = value
            result = self._get_assign_name(node.value)
            if result is None:
                return None
            name, supported = result
            if not supported:
                return None
            name = '{}.{}'.format(name, node.attr)
            return (name, False)
        elif isinstance(node, ast.Subscript):
            # var[index] = value, var[a:b] = value
            result = self._get_assign_name(node.value)
            if result is None:
                return None
            name, supported = result
            if not supported:
                return None
            return (name, False)
        elif isinstance(node, ast.Call):
            # func().attr = value
            return (None, False)
        else:
            return None

    def disable_vars(self, node):
        if self.namespace.vars_enabled:
            self.info('Disable optimizations on variables: {} is not supported'.format(ast.dump(node)))
        self.namespace.disable_vars()

    def unassign(self, node):
        if isinstance(node, ast.Tuple):
            for elt in node.elts:
                self.unassign(elt)
            return

        result = self._get_assign_name(node)
        if result is None:
            self.disable_vars(node)
            return
        name, supported = result
        if not supported:
            self.disable_vars(node)
            return
        self.namespace.unassign(name)

    def is_empty_iterable(self, node):
        if isinstance(node, (ast.List, ast.Tuple)):
            return len(node.elts) == 0

        constant = self.get_literal(node, ITERABLE_TYPES)
        if constant is not None:
            return len(constant) == 0

        if self.check_builtin_func(node, BUILTIN_ACCEPTING_ITERABLE, 0, 0):
            return True

        # iter(())
        if self.check_builtin_func(node, 'iter', 1, 1) and isinstance(node.args[0], ast.Tuple) and not node.args[0].elts:
            return True

        if isinstance(node, ast.Dict) and not node.keys and not node.values:
            return True

        if PYTHON2 and isinstance(node, ast.Set) and not node.elts:
            return True

        return False

    def node_to_type(self, node, to_type):
        if PYTHON2:
            ast_types = (ast.Tuple, ast.List, ast.Dict, ast.Set)
        else:
            ast_types = (ast.Tuple, ast.List, ast.Dict)

        if not isinstance(node, ast_types):
            return
        if isinstance(node, ast.Dict):
            length = len(node.keys)
            assert len(node.keys) == len(node.values)
        else:
            length = len(node.elts)
        if length > self.config.max_tuple_length:
            return

        if isinstance(node, ast.Tuple):
            if to_type == tuple:
                # (1, 2, 3)
                return node
            if to_type == list:
                # [1, 2, 3] => (1, 2, 3)
                return new_list_elts(node, node.elts)
            if to_type == set:
                return self.node_to_set(node)
        elif isinstance(node, ast.List):
            if to_type == list:
                return node
            if to_type == tuple:
                # [1, 2, 3] => (1, 2, 3)
                return new_tuple_elts(node, node.elts)
            if to_type == set:
                return self.node_to_set(node)
        elif isinstance(node, ast.Dict):
            if to_type == dict:
                return node
            # FIXME: support other types
        elif isinstance(node, ast.Set):
            if to_type == set:
                return node
            if to_type in (tuple, list):
                literal = self.get_literal(node)
                if literal is None:
                    return
                literal = sort_set_elts(literal)
                if to_type == tuple:
                    # {3, 1, 2} => (1, 2, 3)
                    return new_tuple(node, literal)
                else:
                    # {3, 1, 2} => [1, 2, 3]
                    return new_list(node, literal)

    def literal_to_type(self, node, literal, to_type):
        if not isinstance(literal, ITERABLE_TYPES):
            return None

        if to_type == set:
            # "abc" => {"a", "b", "c"}
            literal = set(literal)
            if len(literal) > self.config.max_tuple_length:
                return None
            if PYTHON2:
                return new_literal(node, literal)
        else:
            if len(literal) > self.config.max_tuple_length:
                return None

        if isinstance(literal, (frozenset, set)):
            literal = sort_set_elts(literal)
        if to_type == list:
            # "abc" => ["a", "b", "c"]
            literal = list(literal)
        elif to_type in (tuple, set):
            # "abc" => ("a", "b", "c")
            literal = tuple(literal)
        else:
            return None

        return new_literal(node, literal)

    def get_constant_list(self, nodes, to_type=None):
        constants = []
        for node in nodes:
            constant = self.get_constant(node, to_type)
            if constant is None:
                return None
            constants.append(constant)
        return constants

    def optimize_range(self, node, to_type):
        args = self.get_constant_list(node.args)
        if args is None:
            return None

        if len(args) == 1:
            start = 0
            stop = args[0]
            step = 1
        elif len(args) == 2:
            start = args[0]
            stop = args[1]
            step = 1
        elif len(args) == 3:
            start = args[0]
            stop = args[1]
            step = args[2]

        if step == 0:
            return None
        if not all(isinstance(arg, INT_TYPES) for arg in args):
            return None

        if PYTHON2:
            minval = -1 * sys.maxint - 1
            maxval = sys.maxint
            if not all(minval <= arg <= maxval for arg in args):
                return

        if PYTHON3:
            numbers = range(*args)
        else:
            numbers = xrange(*args)

        try:
            range_len = len(numbers)
        except OverflowError:
            # OverflowError: Python int too large to convert to C ssize_t
            pass
        else:
            # range(3) => (0, 1, 2)
            if to_type == list:
                constant = list(numbers)
            elif to_type == set:
                constant = set(numbers)
            else:
                constant = tuple(numbers)
            return new_literal(node, constant)

        if PYTHON3:
            return None

        qualname = self.namespace.get_qualname(node.func.id)
        if qualname != 'range':
            return None
        if not self.can_use_builtin('xrange'):
            return None

        # range(...) => xrange(...)
        node.func.id = 'xrange'
        return node

    def _optimize_iter(self, node, is_generator, to_type):
        if self.is_empty_iterable(node):
            # set("") => set()
            return True

        if PYTHON2:
            ast_types = (ast.Tuple, ast.List, ast.Dict, ast.Set)
        else:
            ast_types = (ast.Tuple, ast.List, ast.Dict)

        if isinstance(node, ast_types):
            return self.node_to_type(node, to_type)

        literal = self.get_literal(node, ITERABLE_TYPES)
        if literal is not None:
            new_literal = self.literal_to_type(node, literal, to_type)
            if new_literal is None:
                return None
            return new_literal

        if isinstance(node, ast.GeneratorExp):
            # (x for x in "abc") => "abc"
            new_iter = self.optimize_comprehension(node,
                                                   is_generator=False,
                                                   to_type=to_type)
            if new_iter is not None:
                return new_iter

        if not is_generator and self.check_builtin_func(node, ('list', 'tuple'),  1, 1):
            # list(iterable) => iterable
            return node.args[0]

        if to_type == set and self.check_builtin_func(node, ('frozenset', 'set'),  1, 1):
            # set(set(iterable)) => set(iterable)
            return node.args[0]

        if self.check_builtin_func(node, 'iter', 1, 1):
            iter_arg = node.args[0]
            # no need to call is_empty_iterable(), iter(iterable) was already
            # optimized by optimize_iter()
            if isinstance(iter_arg, ast.Tuple) and len(iter_arg.elts) == 0:
                # set(iter(())) => set()
                return True
            # set(iter(iterable)) => set(iterable)
            return iter_arg

        if PYTHON2:
            range_names = ('range', 'xrange')
        else:
            range_names = ('range',)
        if self.check_builtin_func(node, range_names, 1, 3):
            return self.optimize_range(node, to_type)

    def optimize_generator(self, node):
        return self._optimize_iter(node, True, tuple)

    def optimize_iter(self, node, to_type):
        return self._optimize_iter(node, False, to_type)

    def remove_dead_code(self, node_list):
        # Remove dead code
        # Example: "return 1; return 2" => "return 1"
        truncate = None
        for index, node in enumerate(node_list[:-1]):
            if not isinstance(node, (ast.Return, ast.Raise)):
                continue
            if not self.can_drop(node_list[index+1:]):
                continue
            truncate = index
            break
        if truncate is None:
            return
        if truncate == len(node_list) - 1:
            return
        for node in node_list[truncate+1:]:
            self.log_node_removal(node, 'Remove unreachable code')
        del node_list[truncate+1:]

    def can_drop(self, node_list):
        if PYTHON3:
            return not ast_contains(node_list, (ast.Global, ast.Nonlocal))
        else:
            return not ast_contains(node_list, ast.Global)

    def optimize_node_list(self, node_list):
        remove_duplicate_pass(node_list)
        self.remove_dead_code(node_list)

    def optimize_comprehension(self, node, is_generator=False, to_type=None):
        if len(node.generators) != 1:
            return None
        generator = node.generators[0]

        if generator.ifs:
            if len(generator.ifs) != 1:
                return None

            test_expr = generator.ifs[0]
            constant = self.get_constant(test_expr)
            if constant is None:
                return None
            if constant:
                return None

            # (x for x in data if False) => iter(())
            return True

        if to_type == dict:
            # dict comprehension
            target = generator.target
            if not (isinstance(target, ast.Tuple)
                    and len(target.elts) == 2
                    and isinstance(target.elts[0], ast.Name)
                    and isinstance(target.elts[1], ast.Name)):
                return None

            key = target.elts[0].id
            value = target.elts[1].id
            if isinstance(node, ast.DictComp):
                simple_gen = (
                    isinstance(node.key, ast.Name)
                    and isinstance(node.value, ast.Name)
                    and node.key.id == key
                    and node.value.id == value)
            else:
                simple_gen = (
                    isinstance(node.elt, ast.Tuple)
                    and len(node.elt.elts) == 2
                    and isinstance(node.elt.elts[0], ast.Name)
                    and node.elt.elts[0].id == key
                    and isinstance(node.elt.elts[1], ast.Name)
                    and node.elt.elts[1].id == value)
            if not simple_gen:
                # Example: {value: key for key, value in iterable}
                if is_generator:
                    new_iter = self.optimize_generator(generator.iter)
                else:
                    new_iter = self.optimize_iter(generator.iter, tuple)

                if new_iter is True:
                    # y for x in ()
                    return True

                if new_iter is not None:
                    # y for x in [1, 2] => y for x in (1, 2)
                    generator.iter = new_iter
                return
        else:
            # generator expression, list or set comprehension:
            # to_type in (tuple, list, set)
            if not isinstance(generator.target, ast.Name):
                return
            name = generator.target.id
            if (not isinstance(node.elt, ast.Name)
            or node.elt.id != name):
                if is_generator:
                    new_iter = self.optimize_generator(generator.iter)
                else:
                    new_iter = self.optimize_iter(generator.iter, tuple)

                if new_iter is True:
                    # y for x in ()
                    return True
                if new_iter is not None:
                    # y for x in [1, 2] => y for x in (1, 2)
                    generator.iter = new_iter
                return None
        iter_expr = generator.iter

        if is_generator:
            new_iter = self.optimize_generator(iter_expr)
        else:
            new_iter = self.optimize_iter(iter_expr, to_type)
        if new_iter is not None:
            return new_iter
        else:
            return iter_expr

    def visit_ListComp(self, node):
        for generator in node.generators:
            self.unassign(generator.target)

        node.generators = self.visit_list(node.generators)
        node.elt = self.visit(node.elt)

        iter_expr = self.optimize_comprehension(node, to_type=list)
        if iter_expr is not None:
            if iter_expr is True:
                # [x*2 for x in "abc" if False] => []
                # [x*2 for x in []] => []
                return new_list_elts(node)

            if isinstance(iter_expr, ast.List):
                # [x for x in "abc"] => ["a", "b", "c"]
                return iter_expr

            if self.can_use_builtin('list'):
                # [x for x in range(1000)] => list(xrange(1000))
                return new_call(node, 'list', iter_expr)

            node.generators[0].iter = iter_expr

        new_node = self.try_unroll_listcomp(node)
        if new_node:
            return new_node
        else:
            return node


class ReplaceVariable(Optimizer):
    def __init__(self, name, value):
        Optimizer.__init__(self)
        self.name = name
        self.value = value

    def visit_Name(self, node):
        if node.id == self.name:
            return self.value

        return Optimizer.visit_Name(self, node)