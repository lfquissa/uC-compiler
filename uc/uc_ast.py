from abc import ABC
from abc import abstractmethod

import sys


def represent_node(obj, indent):
    def _repr(obj, indent, printed_set):
        """
        Get the representation of an object, with dedicated pprint-like format for lists.
        """
        if isinstance(obj, list):
            indent += 1
            sep = ",\n" + (" " * indent)
            final_sep = ",\n" + (" " * (indent - 1))
            return (
                "["
                + (sep.join((_repr(e, indent, printed_set) for e in obj)))
                + final_sep
                + "]"
            )
        elif isinstance(obj, Node):
            if obj in printed_set:
                return ""
            else:
                printed_set.add(obj)
            result = obj.__class__.__name__ + "("
            indent += len(obj.__class__.__name__) + 1
            attrs = []

            # convert each node attribute to string
            for name, value in vars(obj).items():
                # is an irrelevant attribute: skip it.
                if name in ("bind", "coord"):
                    continue

                # relevant attribte not set: skip it.
                if value is None:
                    continue

                # relevant attribute set: append string representation.
                value_str = _repr(value, indent + len(name) + 1, printed_set)
                attrs.append(name + "=" + value_str)

            sep = ",\n" + (" " * indent)
            final_sep = ",\n" + (" " * (indent - 1))
            result += sep.join(attrs)
            result += ")"
            return result
        elif isinstance(obj, str):
            return obj
        else:
            return str(obj)

    # avoid infinite recursion with printed_set
    printed_set = set()
    return _repr(obj, indent, printed_set)


#
# ABSTRACT NODES
#
class Node(ABC):
    """Abstract base class for AST nodes."""

    attr_names = ()

    @abstractmethod
    def __init__(self, coord=None):
        self.coord = coord

    def __repr__(self):
        """Generates a python representation of the current node"""
        return represent_node(self, 0)

    def children(self):
        """A sequence of all children that are Nodes"""
        pass

    def show(
        self,
        buf=sys.stdout,
        offset=0,
        attrnames=False,
        nodenames=False,
        showcoord=False,
        _my_node_name=None,
    ):
        """Pretty print the Node and all its attributes and children (recursively) to a buffer.
        buf:
            Open IO buffer into which the Node is printed.
        offset:
            Initial offset (amount of leading spaces)
        attrnames:
            True if you want to see the attribute names in name=value pairs. False to only see the values.
        nodenames:
            True if you want to see the actual node names within their parents.
        showcoord:
            Do you want the coordinates of each Node to be displayed.
        """
        lead = " " * offset
        if nodenames and _my_node_name is not None:
            buf.write(lead + self.__class__.__name__ + " <" + _my_node_name + ">: ")
            inner_offset = len(self.__class__.__name__ + " <" + _my_node_name + ">: ")
        else:
            buf.write(lead + self.__class__.__name__ + ":")
            inner_offset = len(self.__class__.__name__ + ":")

        if self.attr_names:
            if attrnames:
                nvlist = [
                    (
                        n,
                        represent_node(
                            getattr(self, n), offset + inner_offset + 1 + len(n) + 1
                        ),
                    )
                    for n in self.attr_names
                    if getattr(self, n) is not None
                ]
                attrstr = ", ".join("%s=%s" % nv for nv in nvlist)
            else:
                vlist = [getattr(self, n) for n in self.attr_names]
                attrstr = ", ".join(
                    represent_node(v, offset + inner_offset + 1) for v in vlist
                )
            buf.write(" " + attrstr)

        if showcoord:
            if self.coord and self.coord.line != 0:
                buf.write(" %s" % self.coord)
        buf.write("\n")

        for child_name, child in self.children():
            child.show(buf, offset + 4, attrnames, nodenames, showcoord, child_name)


class DeclType(Node):
    """
    I am an absctract class for declaration types.

    This class augments declaration nodes with a few extra tools.

    :attribute identifier:
        Allows the user to set/get the declaration's ID node from any
        declaration in the declaration chain.

    :attribute primitive:
        Allows the user to set/get uderlying primitive type (int, char, etc.)
        from any declaration node in the declaration chain.

    :method modify:
        Used to apply a type modifier in the declaration chain.
    """

    @abstractmethod
    def __init__(self):
        ...

    @property
    def identifier(self):
        """I get the declaration's ID node."""
        # ID not set: return as empty.
        if self.type is None:
            return None

        # has type modififer: fetch ID from modifier.
        return self.type.identifier

    @identifier.setter
    def identifier(self, identifier):
        """I set a declaration's identifier."""
        self.type.identifier = identifier

    def modify(self, modifier):
        """
        I apply a type modifier to a declaration.

        :param modifier: declaration modifier AST node (ArrayDecl).

        :returns: modified declaration.
        """
        # has primitive or unset type: modify itself
        if self.type is None or isinstance(self.type, Type):
            modifier.type = self.type
            self.type = modifier
            return self

        # has underlying type modifier: modify underlying modifier chain.
        self.type = self.type.modify(modifier)

        # return itself modified.
        return self

    @property
    def primitive(self):
        """I get the declaration's primitive type."""
        # type not set: return as empty.
        if self.type is None:
            return None

        # reached primitive type: return it.
        if isinstance(self.type, Type):
            return self.type

        # has a declaration modifier: recurse into it.
        return self.type.primitive

    @primitive.setter
    def primitive(self, typeNode):
        """
        I set the declaration's underlying primitive type.

        :param typeNode: primitive type node to be set.
        """
        # reached missing or primitive type: set/overwrite it
        if self.type is None or isinstance(self.type, Type):
            self.type = typeNode
            return

        # has a declaration modifier: recurse into it.
        self.type.primitive = typeNode


#
# CONCRETE NODES
#
class ArrayDecl(DeclType):
    attr_names = ()

    def __init__(self, type, dim, coord=None):
        """
        I create an instance of this class.

        :param type: underlying type modifier.
        :param dim: dimension of the array.
        :param coord: declaration code position.
        """
        self.type = type
        self.dim = dim
        self.coord = coord

    def children(self):
        nodelist = []
        if self.type is not None:
            nodelist.append(("type", self.type))
        if self.dim is not None:
            nodelist.append(("dim", self.dim))
        return tuple(nodelist)


class ArrayRef(Node):
    attr_names = ()

    def __init__(self, name, subscript, coord=None):
        """
        I create an instance of this class.

        :param name: name of the array being accessed.
        :param subscript: dimension of the array.
        :param coord: declaration code position.
        """
        self.name = name
        self.subscript = subscript
        self.coord = coord

    def children(self):
        nodelist = []
        if self.name is not None:
            nodelist.append(("name", self.name))
        if self.subscript is not None:
            nodelist.append(("subscript", self.subscript))
        return tuple(nodelist)


class Assert(Node):
    attr_names = ()

    def __init__(self, expr, coord=None):
        """
        I create an instance of this class.

        :param expr: boolean expression being asserted.
        :param coord: code position.
        """
        self.expr = expr
        self.coord = coord

    def children(self):
        nodelist = []
        if self.expr is not None:
            nodelist.append(("expr", self.expr))
        return tuple(nodelist)


class Assignment(Node):
    attr_names = ("op",)

    def __init__(self, op, lvalue, rvalue, coord=None):
        """
        I create an instance of this class.

        :param op: assignment operator (=, +=, %=, ...).
        :param lvalue: variable being written.
        :param rvalue: value being assingned to variable.
        :param coord: code position.
        """
        self.op = op
        self.lvalue = lvalue
        self.rvalue = rvalue
        self.coord = coord

    def children(self):
        nodelist = []
        if self.lvalue is not None:
            nodelist.append(("lvalue", self.lvalue))
        if self.rvalue is not None:
            nodelist.append(("rvalue", self.rvalue))
        return tuple(nodelist)


class BinaryOp(Node):
    attr_names = ("op",)

    def __init__(self, op, left, right, coord=None):
        """
        I create an instance of this class.

        :param op: binary operator (+, -, *, ...).
        :param left: left hand side expression.
        :param right: right hand side expression.
        :param coord: code position.
        """
        self.op = op
        self.left = left
        self.right = right
        self.coord = coord

    def children(self):
        nodelist = []
        if self.left is not None:
            nodelist.append(("left", self.left))
        if self.right is not None:
            nodelist.append(("right", self.right))
        return tuple(nodelist)


class Break(Node):
    attr_names = ()

    def __init__(self, coord=None):
        self.coord = coord

    def children(self):
        return ()


class Compound(Node):
    attr_names = ()

    def __init__(self, staments, coord=None):
        """
        I create an instance of this class.

        :param staments: statements within the compound.
        :param coord: code position.
        """
        self.staments = staments
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.staments or []):
            nodelist.append(("staments[%d]" % i, child))
        return tuple(nodelist)


class Constant(Node):
    attr_names = ("type", "value")

    def __init__(self, type, value, coord=None):
        """
        I create an instance of this class.

        :param type: primitive type.
        :param value: constant value.
        :param coord: code position.
        """
        self.type = type
        self.value = value
        self.coord = coord

    def children(self):
        return ()


class Decl(DeclType):
    attr_names = ("name",)

    def __init__(self, name, type, init, coord=None):
        """
        I create an instance of this class.

        :param name: declaration variable name.
        :param type: underlying type modifier.
        :param init: declaration's initialization value.
        :param coord: code position.
        """
        self.name = name
        self.type = type
        self.init = init
        self.coord = coord

    def children(self):
        nodelist = []
        if self.type is not None:
            nodelist.append(("type", self.type))
        if self.init is not None:
            nodelist.append(("init", self.init))
        return tuple(nodelist)


class DeclList(Node):
    attr_names = ()

    def __init__(self, decls, coord=None):
        """
        I create an instance of this class.

        :param decls: list of declarations.
        :param coord: code position.
        """
        self.decls = decls
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.decls or []):
            nodelist.append(("decls[%d]" % i, child))
        return tuple(nodelist)


class EmptyStatement(Node):
    attr_names = ()

    def __init__(self, coord=None):
        self.coord = coord

    def children(self):
        return ()


class ExprList(Node):
    attr_names = ()

    def __init__(self, exprs, coord=None):
        """
        I create an instance of this class.

        :param exprs: list of expressions.
        :param coord: code position.
        """
        self.exprs = exprs
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.exprs or []):
            nodelist.append(("exprs[%d]" % i, child))
        return tuple(nodelist)


class For(Node):
    attr_names = ()

    def __init__(self, init, cond, next, body, coord=None):
        """
        I create an instance of this class.

        :param init: initialization to be made before the loop.
        :param cond: conditional to be evaluated each iteration.
        :param next: computation to be made after each iteration.
        :param body: statements within the loop's body.
        :param coord: code position.
        """
        self.init = init
        self.cond = cond
        self.next = next
        self.body = body
        self.coord = coord
        self.break_list = []

    def children(self):
        nodelist = []
        if self.init is not None:
            nodelist.append(("init", self.init))
        if self.cond is not None:
            nodelist.append(("cond", self.cond))
        if self.next is not None:
            nodelist.append(("next", self.next))
        if self.body is not None:
            nodelist.append(("body", self.body))
        return tuple(nodelist)


class FuncCall(Node):
    attr_names = ()

    def __init__(self, name, args, coord=None):
        """
        I create an instance of this class.

        :param name: name of the function being called.
        :param args: function call arguments.
        :param coord: code position.
        """
        self.name = name
        self.args = args
        self.coord = coord

    def children(self):
        nodelist = []
        if self.name is not None:
            nodelist.append(("name", self.name))
        if self.args is not None:
            nodelist.append(("args", self.args))
        return tuple(nodelist)


class FuncDecl(DeclType):
    attr_names = ()

    def __init__(self, params, type, coord=None):
        """
        I create an instance of this class.

        :param params: function parameters declarations.
        :param type: function return type.
        :param coord: code position.
        """
        self.params = params
        self.type = type
        self.coord = coord
        self.is_prototype = False

    def children(self):
        nodelist = []
        if self.params is not None:
            nodelist.append(("params", self.params))
        if self.type is not None:
            nodelist.append(("type", self.type))
        return tuple(nodelist)


class FuncDef(Node):
    attr_names = ()

    def __init__(self, type, decl, body, coord=None):
        """
        I create an instance of this class.

        :param type: function return type.
        :param decl: function declaration AST node.
        :param body: function compound body.
        :param coord: code position.
        """
        self.type = type
        self.decl = decl
        self.body = body
        self.coord = coord

    def children(self):
        nodelist = []
        if self.type is not None:
            nodelist.append(("type", self.type))
        if self.decl is not None:
            nodelist.append(("decl", self.decl))
        if self.body is not None:
            nodelist.append(("body", self.body))
        return tuple(nodelist)


class GlobalDecl(Node):
    attr_names = ()

    def __init__(self, decls, coord=None):
        """
        I create an instance of this class.

        :param decls: list of declarations (int a, *b, c;).
        :param coord: code position.
        """
        self.decls = decls
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.decls or []):
            nodelist.append(("decls[%d]" % i, child))
        return tuple(nodelist)


class ID(Node):
    attr_names = ("name",)

    def __init__(self, name, coord=None):
        """
        I create an instance of this class.

        :param name: ID unique name.
        :param coord: code position.
        """
        self.name = name
        self.coord = coord

    def children(self):
        return ()


class If(Node):
    attr_names = ()

    def __init__(self, cond, iftrue, iffalse, coord=None):
        """
        I create an instance of this class.

        :param cond: conditional statement being evaluated.
        :param iftrue: compound block to execute on true statement.
        :param iffalse: compound block to execute on false statement.
        :param coord: code position.
        """
        self.cond = cond
        self.iftrue = iftrue
        self.iffalse = iffalse
        self.coord = coord

    def children(self):
        nodelist = []
        if self.cond is not None:
            nodelist.append(("cond", self.cond))
        if self.iftrue is not None:
            nodelist.append(("iftrue", self.iftrue))
        if self.iffalse is not None:
            nodelist.append(("iffalse", self.iffalse))
        return tuple(nodelist)


class InitList(Node):
    attr_names = ()

    def __init__(self, exprs, coord=None):
        """
        I create an instance of this class.

        :param exprs: list of initializer expressions.
        :param coord: code position.
        """
        self.exprs = exprs
        self.coord = coord
        self.value = []
        self.root = None
        self.list_sizes = []

    def children(self):
        nodelist = []
        for i, child in enumerate(self.exprs or []):
            nodelist.append(("exprs[%d]" % i, child))
        return tuple(nodelist)


class ParamList(Node):
    attr_names = ()

    def __init__(self, params, coord=None):
        """
        I create an instance of this class.

        :param params: list of parameter declarations.
        :param coord: code position.
        """
        self.params = params
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.params or []):
            nodelist.append(("params[%d]" % i, child))
        return tuple(nodelist)


class Print(Node):
    attr_names = ()

    def __init__(self, expr, coord=None):
        """
        I create an instance of this class.

        :param expr: expression to be printed.
        :param coord: code position.
        """
        self.expr = expr
        self.coord = coord

    def children(self):
        nodelist = []
        if self.expr is not None:
            nodelist.append(("expr", self.expr))
        return tuple(nodelist)


class Program(Node):
    attr_names = ()

    def __init__(self, gdecls, coord=None):
        """
        I create an instance of this class.

        :param gdecls: program's global declarations.
        :param coord: code position.
        """
        self.gdecls = gdecls
        self.coord = coord

    def children(self):
        nodelist = []
        for i, child in enumerate(self.gdecls or []):
            nodelist.append(("gdecls[%d]" % i, child))
        return tuple(nodelist)


class Read(Node):
    attr_names = ()

    def __init__(self, names, coord=None):
        """
        I create an instance of this class.

        :param names: IDs where read values should be stored.
        :param coord: code position.
        """
        self.names = names
        self.coord = coord

    def children(self):
        nodelist = []
        if self.names is not None:
            nodelist.append(("names", self.names))
        return tuple(nodelist)


class Return(Node):
    attr_names = ()

    def __init__(self, expr, coord=None):
        """
        I create an instance of this class.

        :param expr: expression whose result will be returned.
        :param coord: code position.
        """
        self.expr = expr
        self.coord = coord

    def children(self):
        nodelist = []
        if self.expr is not None:
            nodelist.append(("expr", self.expr))
        return tuple(nodelist)


class Type(Node):
    attr_names = ("name",)

    def __init__(self, name, coord=None):
        """
        I create an instance of this class.

        :param name: primitive type name (int, char, ...).
        :param coord: code position.
        """
        self.name = name
        self.coord = coord

    def children(self):
        return ()


class UnaryOp(Node):
    attr_names = ("op",)

    def __init__(self, op, expr, coord=None):
        """
        I create an instance of this class.

        :param op: unary operator (!, +, -, ...)
        :param expr: expression whose value will be modified by the operator.
        """
        self.op = op
        self.expr = expr
        self.coord = coord

    def children(self):
        nodelist = []
        if self.expr is not None:
            nodelist.append(("expr", self.expr))
        return tuple(nodelist)


class VarDecl(DeclType):
    attr_names = ()

    def __init__(self, declname, type, coord=None):
        """
        I create an instance of this class.

        :param declname: variable name.
        :param type: variable primitive type.
        :param coord: code position.
        """
        self.declname = declname
        self.type = type
        self.coord = coord

    def children(self):
        nodelist = []
        if self.type is not None:
            nodelist.append(("type", self.type))
        return tuple(nodelist)

    @property
    def identifier(self):
        """I get the declaration ID node"""
        return self.declname

    @identifier.setter
    def identifier(self, identifier):
        """
        I set the declaration ID node.

        :param identifier: AST ID node.
        """
        self.declname = identifier

    def modify(self, modifier):
        """I modify this declaration.

        :para modifier: declaration modifier node.

        :returns: modified variable declaration.
        """
        modifier.type = self
        return modifier

    @property
    def primitive(self):
        """I get the declaration primitive type."""
        return self.type

    @primitive.setter
    def primitive(self, typeNode):
        """I set the declaration underlying primitive type.

        :param typeNode: primitive type node to be set.
        """
        self.type = typeNode


class While(Node):
    attr_names = ()

    def __init__(self, cond, body, coord=None):
        """
        I create an instance of this class.

        :param cond: conditional being evaluated at every iteration.
        :param body: compound representing the loop body.
        :param coord: code position.
        """
        self.cond = cond
        self.body = body
        self.coord = coord
        self.break_list = []

    def children(self):
        nodelist = []
        if self.cond is not None:
            nodelist.append(("cond", self.cond))
        if self.body is not None:
            nodelist.append(("body", self.body))
        return tuple(nodelist)
