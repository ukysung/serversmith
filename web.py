import flask
import json
import ast

app = flask.Flask(__name__)


@app.route('/ast2json', methods=['POST'])
def ast2json():
    node = ast.parse(flask.request.form['pysrc'])

    def iter_fields(node):

        for field in node._fields:
            try:
                yield field, getattr(node, field)
            except AttributeError:
                pass

    def _format(node):
        if isinstance(node, ast.AST):
            fields = [('_PyType', _format(node.__class__.__name__))]
            fields += [(a, _format(b)) for a, b in iter_fields(node)]

            return '{ %s }' % ', '.join(('"%s": %s' % field for field in fields))

        if isinstance(node, list):
            return '[ %s ]' % ', '.join([_format(x) for x in node])

        return json.dumps(node)

    return _format(node)


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/static/<file_name>')
def static_file(file_name):
    return flask.send_from_directory('static', file_name)

app.run()
