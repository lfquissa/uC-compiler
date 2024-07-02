import argparse
import pathlib
import sys
from copy import deepcopy
import traceback
from collections import deque
from typing import Any, Dict, Union
from uc.uc_ast import (
    ID,
    ArrayDecl,
    BinaryOp,
    Break,
    Compound,
    Constant,
    ExprList,
    For,
    FuncDecl,
    InitList,
    Return,
    VarDecl,
    While,
)
from uc.uc_parser import UCParser
from uc.uc_type import (
    ArrayType,
    BoolType,
    CharType,
    IntType,
    StringType,
    VoidType,
    FuncType,
    uCType,
)
from enum import Enum


class Marker(Enum):
    BEGIN_SCOPE = 1


class SymbolTable:
    """Class representing a symbol table.

    `add` and `lookup` methods are given, however you still need to find a way to
    deal with scopes.

    ## Attributes
    - :attr data: the content of the SymbolTable
    """

    def __init__(self) -> None:
        """Initializes the SymbolTable."""
        self.__data = dict()
        self._undo = []
        self.begin_scope_idxs = (
            []
        )  # Keeps track of the indexes of the BEGIN_SCOPE markers
        self.scope_openers = []

    @property
    def data(self) -> Dict[str, Any]:
        """Returns a copy of the SymbolTable."""
        return deepcopy(self.__data)

    # TODO  Symbol values são dict, pensar se essa é a melhor forma.
    def add(self, name: str, value: dict) -> None:
        """Adds to the SymbolTable.

        ## Parameters
        - :param name: the identifier on the SymbolTable
        - :param value: the value to assign to the given `name`
        """
        if name not in self.__data:
            self.__data[name] = [value]
        else:
            self.__data[name].append(value)
        self._undo.append(name)

    def modify(self, name, value, param_with_func_name=False):
        """Modifies the value of a symbol in the SymbolTable."""
        # TODO Assume que existe algo para ser modificado
        #      Verificar se isso pode dar problema
        assert len(self.__data[name]) > 0
        if param_with_func_name:
            assert len(self.__data[name]) > 1
            self.__data[name][-2] = value
        else:
            self.__data[name][-1] = value

    def lookup(self, name: str) -> Union[Any, None]:
        """Searches `name` on the SymbolTable and returns the value
        assigned to it.

        ## Parameters
        - :param name: the identifier that will be searched on the SymbolTable

        ## Return
        - :return: the value assigned to `name` on the SymbolTable. If `name` is not found, `None` is returned.
        """
        if name in self.__data and len(self.__data[name]) > 0:
            return self.__data[name][-1]
        return None

    def begin_scope(self, node) -> None:
        """Begins a new scope."""
        if self._undo == []:
            self.begin_scope_idxs.append(0)
        else:
            self.begin_scope_idxs.append(len(self._undo))
        self._undo.append((Marker.BEGIN_SCOPE))
        self.scope_openers.append(node)

    def end_scope(self) -> None:
        """Restores the table to where it was at the most recent begin_scope
        that has not already been ended.
        """
        top = self._undo[-1]
        while top != Marker.BEGIN_SCOPE:
            self.__data[top].pop()
            self._undo.pop()
            top = self._undo[-1]
        self._undo.pop()
        self.begin_scope_idxs.pop()
        self.scope_openers.pop()

    def already_defined(self, name):
        """Checks if a variable has already been defined in the current scope."""
        if name in self.__data:
            for i in range(self.begin_scope_idxs[-1], len(self._undo)):
                if self._undo[i] == name:
                    return True
        return False

    def get_enclosing_loop(self):
        for node in reversed(self.scope_openers):
            if isinstance(node, While) or isinstance(node, For):
                return node
        return None


class NodeVisitor:
    """A base NodeVisitor class for visiting uc_ast nodes.
    Subclass it and define your own visit_XXX methods, where
    XXX is the class name you want to visit with these
    methods.
    """

    _method_cache = None

    def visit(self, node):
        """Visit a node."""

        if self._method_cache is None:
            self._method_cache = {}

        visitor = self._method_cache.get(node.__class__.__name__)
        if visitor is None:
            method = "visit_" + node.__class__.__name__
            visitor = getattr(self, method, self.generic_visit)
            self._method_cache[node.__class__.__name__] = visitor

        return visitor(node)

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a
        node. Implements preorder visiting of the node.
        """
        for _, child in node.children():
            self.visit(child)


class Visitor(NodeVisitor):
    """
    Program visitor class. This class uses the visitor pattern. You need to define methods
    of the form visit_NodeName() for each kind of AST node that you want to process.
    """

    def __init__(self):
        # Initialize the symbol table
        self.symtab = SymbolTable()
        self.typemap = {
            "int": IntType,
            "char": CharType,
            "void": VoidType,
            "string": StringType,
        }

    def _get_returns(self, node, return_list):
        """Return the type of the return expression of a function"""
        if isinstance(node, Return):
            return_list.append(node)
        else:
            for _, child in node.children():
                self._get_returns(child, return_list)

    def _assert_semantic(
        self, condition: bool, msg_code: int, coord, name: str = "", ltype="", rtype=""
    ):
        """Check condition, if false print selected error message and exit"""
        error_msgs = {
            1: f"{name} is not defined",
            2: f"subscript must be of type(int), not {ltype}",
            3: "Expression must be of type(bool)",
            4: f"Cannot assign {rtype} to {ltype}",
            5: f"Binary operator {name} does not have matching LHS/RHS types",
            6: f"Binary operator {name} is not supported by {ltype}",
            7: "Break statement must be inside a loop",
            8: "Array dimension mismatch",
            9: f"Size mismatch on {name} initialization",
            10: f"{name} initialization type mismatch",
            11: f"{name} initialization must be a single element",
            12: "Lists have different sizes",
            13: "List & variable have different sizes",
            14: f"conditional expression is {ltype}, not type(bool)",
            15: f"{name} is not a function",
            16: f"no. arguments to call {name} function mismatch",
            17: f"Type mismatch with parameter {name}",
            18: "The condition expression must be of type(bool)",
            19: "Expression must be a constant",
            20: "Expression is not of basic type",
            21: f"{name} does not reference a variable of basic type",
            22: f"{name} is not a variable",
            23: f"Return of {ltype} is incompatible with {rtype} function definition",
            24: f"Name {name} is already defined in this scope",
            25: f"Unary operator {name} is not supported",
        }
        if not condition:
            msg = error_msgs[msg_code]  # invalid msg_code raises Exception
            print("SemanticError: %s %s" % (msg, coord), file=sys.stdout)
            sys.exit(1)

    def _get_array_shape(self, node, array_shape):
        if isinstance(node, ArrayDecl):
            dim = node.dim.value if node.dim is not None else None
            array_shape.append(dim)
            self._get_array_shape(node.type, array_shape)

    # TODO Talvez seja melhor calcular o depth durant o visit_InitList
    def _get_init_list_shape(self, node):
        queue = deque([])
        queue.append(node)
        shape = [[]]
        node.depth = 0

        while queue:
            curr_list = queue.popleft()
            shape[curr_list.depth].append(curr_list.length)
            for expr in curr_list.exprs:
                if isinstance(expr, InitList):
                    queue.append(expr)
                    expr.depth = curr_list.depth + 1
            if (
                len(queue) > 0
                and queue[-1].depth > curr_list.depth
                and len(shape) <= curr_list.depth + 1
            ):
                shape.append([])

        return shape

    def visit_Program(self, node):
        # Visit all of the global declarations
        self.symtab.begin_scope(node)
        for _decl in node.gdecls:
            self.visit(_decl)
        self.symtab.end_scope()

    def visit_GlobalDecl(self, node):
        # Visit all declarations
        for decl in node.decls:
            if isinstance(decl.type, FuncDecl):
                decl.type.is_prototype = True
            self.visit(decl)

    # TODO Não sei ao certo o que deve ser feito aqui e o que deve ser feito no visit_FuncDef
    #      Além disso não sei se o scope deve ser criado aqui ou no visit_FuncDef
    def visit_FuncDecl(self, node):
        func_name = node.type.declname.name
        return_type = self.typemap[node.type.type.name]

        if node.params is not None:
            params_type = []
            self.symtab.begin_scope(node)
            for param in node.params.params:
                self.visit(param)
                params_type.append(param.name)
            self.symtab.end_scope()
            func_type = FuncType(return_type, params_type)
        else:
            func_type = FuncType(return_type, None)
        # func_type = FuncType(return_type, None)

        if node.is_prototype:
            self._assert_semantic(
                not self.symtab.already_defined(func_name),
                24,
                node.type.declname.coord,
                name=func_name,
            )
            self.symtab.add(
                func_name, {"uc_type": func_type, "kind": "func", "node": node}
            )

        elif self.symtab.already_defined(func_name):
            symbol = self.symtab.lookup(func_name)

            # Checa caso o símbolo ja tenha sido definido como uma variavel global
            if symbol["kind"] != "func":
                self._assert_semantic(
                    not self.symtab.already_defined(func_name),
                    24,
                    node.type.declname.coord,
                    name=func_name,
                )

            # TODO Chegar aqui significa que existe um protótipo da função
            # e agora estamos dentro da definição da função.
            # Na teoria é preciso checar que o tipo de retorno e os tipos e
            # quantidades dos parametros são iguais

        else:
            self.symtab.add(
                func_name, {"uc_type": func_type, "kind": "func", "node": node}
            )

        node.type.declname.uc_type = func_type

    # TODO Confirmar se um function prototype define um escopo proprio ou se faz parte do escopo global.
    # TODO Confirmar também se uma função sempre aparece junto com a definição ou se é possível ter uma função sem definição
    # TODO Caso não seja possível ter uma função sem definição, isso implica que foward references estão proibidas?
    def visit_FuncDef(self, node):
        # Visit the return type
        func_name = node.decl.name.name
        return_type = self.typemap[node.type.name]
        # self._assert_semantic(
        #     not self.symtab.already_defined(func_name),
        #     24,
        #     node.decl.type.type.declname.coord,
        #     func_name,
        # )

        self.visit(node.decl)
        params = node.decl.type.params
        self.symtab.begin_scope(node)

        if params is not None:
            self.visit(params)

            params_type = []
            param_with_func_name = False
            for param in params.params:
                params_type.append(param.name)
                if param.name.name == func_name:
                    param_with_func_name = True

            func_type = FuncType(return_type, params_type)
            self.symtab.modify(
                func_name,
                {"uc_type": func_type, "kind": "func", "node": node},
                param_with_func_name,
            )

        self.visit(node.body)

        return_list = []
        self._get_returns(node.body, return_list)
        node.return_list = return_list
        if return_list == []:
            self._assert_semantic(
                return_type == VoidType,
                23,
                node.body.coord,
                ltype=f"type({VoidType.typename})",
                rtype=f"type({return_type.typename})",
            )
        else:
            for ret in return_list:
                self._assert_semantic(
                    return_type == ret.uc_type,
                    23,
                    ret.coord,
                    ltype=f"type({ret.uc_type.typename})",
                    rtype=f"type({return_type.typename})",
                )

        self.symtab.end_scope()

    def visit_FuncCall(self, node):
        self.visit(node.name)
        self._assert_semantic(
            self.symtab.lookup(node.name.name)["kind"] == "func",
            15,
            node.name.coord,
            node.name.name,
        )

        func_call_params = node.args
        func_type = self.symtab.lookup(node.name.name)["uc_type"]
        func_def_params = func_type.params

        if func_call_params is not None:
            self.visit(node.args)

            if isinstance(func_call_params, ExprList):
                self._assert_semantic(
                    func_def_params is not None
                    and len(func_def_params) == len(func_call_params.exprs),
                    16,
                    node.coord,
                    name=node.name.name,
                )

                for i in range(len(func_call_params.exprs)):
                    self._assert_semantic(
                        func_call_params.exprs[i].uc_type == func_def_params[i].uc_type,
                        17,
                        func_call_params.exprs[i].coord,
                        name=func_def_params[i].name,
                    )
            else:
                self._assert_semantic(
                    func_call_params.uc_type == func_def_params[0].uc_type,
                    17,
                    func_call_params.coord,
                    name=func_def_params[0].name,
                )

        else:
            self._assert_semantic(
                func_def_params is None,
                16,
                node.coord,
                name=node.name.name,
            )

        node.uc_type = func_type.return_type

    def visit_ArrayDecl(self, node):
        """Visit the type of the array and the size of the array.
        Check if the array is already defined, otherwise return an error.
        """
        arr_name, var_decl = self.visit(node.type)
        # TODO Precisa fazer algua verificação de array dimension mismatch aqui?
        # TODO Precisa verificar se node.dim.uc_type é do tipo IntType?
        #      No caso seria para evitar algo como int arr['c'];
        if node.dim is not None:
            self.visit(node.dim)
            node.uc_type = ArrayType(
                node.type.uc_type.typename, node.type.uc_type, node.dim.value
            )
        else:
            node.uc_type = ArrayType(node.type.uc_type.typename, node.type.uc_type)

        self.symtab.modify(
            arr_name, {"uc_type": node.uc_type, "kind": "arr", "node": node}
        )
        # As linhas abaixo modificam o uc_type do VarDecl e do ID associado ao ArrayDecl
        # para ser o ArrayType.
        # TODO Não sei se isso é necesśario
        var_decl.uc_type = node.uc_type
        var_decl.declname.uc_type = node.uc_type
        return arr_name, var_decl

    def visit_ParamList(self, node):
        for param in node.params:
            self.visit(param)

    def visit_Decl(self, node):
        """Visit the types of the declaration (VarDecl, ArrayDecl, FuncDecl).
        Check if the function or the variable is defined, otherwise return an error.
        If there is an initial value defined, visit it."""
        self.visit(node.type)

        # Check if the function or the variable is defined.
        self._assert_semantic(
            self.symtab.already_defined(node.name.name),
            1,
            node.name.coord,
            name=node.name.name,
        )

        if isinstance(node.type, ArrayDecl):
            array_shape = []
            self._get_array_shape(node.type, array_shape)
            node.type.array_shape = array_shape
            # TODO Acredito que isso deve ser testado mesmo se node.init não for None
            if node.init is None:
                for dim in array_shape[1:]:
                    self._assert_semantic(dim is not None, 8, node.name.coord)

        if node.init is not None:
            self.visit(node.init)
            if isinstance(node.init, InitList):
                init_list_shape = self._get_init_list_shape(node.init)
            else:
                init_list_shape = [[]]

            if not isinstance(node.type.uc_type, ArrayType):
                self._assert_semantic(
                    node.type.uc_type == node.init.uc_type,
                    10,
                    node.name.coord,
                    name=node.name.name,
                )
                # TODO Talvez seja preciso verificar também que a InitList não contém outra InitList
                #      Isto é, verificar que a declaração não é do tipo int x = {{1}}
                if isinstance(node.type, VarDecl) and isinstance(node.init, InitList):
                    self._assert_semantic(
                        node.init.length == 1,
                        11,
                        node.name.coord,
                        name=node.name.name,
                    )

            else:  # Verifica que o tipo do array é igual ao tipo da lista de inicialização;
                # TODO Possivelmente precisa mudar esse if
                # Caso algo como int x[2][] = {{1,2,3},{4,5,6}} seja permitido
                if node.type.dim is None:
                    if isinstance(node.init, InitList):
                        self.update_array_shape(node.type, array_shape, init_list_shape)
                    else:
                        node.type.dim = Constant("int", str(node.init.length))
                        array_shape[0] = node.init.length
                        node.type.array_shape = array_shape
                        node.type.uc_type.dim = node.init.length

                # TODO Da erro caso tenhamos "int x[2] = 2;"
                if node.type.uc_type.type != CharType:
                    # Check that all lists in the same depth have the same size
                    for size_list in init_list_shape:
                        self._assert_semantic(
                            size_list.count(size_list[0]) == len(size_list),
                            12,
                            node.name.coord,
                        )
                    # Check that lists sizes and array dimensions match
                    for i, dim in enumerate(array_shape):
                        for list_size in init_list_shape[i]:
                            self._assert_semantic(
                                int(dim) == list_size,
                                13,
                                node.name.coord,
                            )

                if node.type.uc_type.type.typename == "char":
                    self._assert_semantic(
                        node.init.length == int(node.type.dim.value),
                        9,
                        node.name.coord,
                        name=node.name.name,
                    )

                else:
                    self._assert_semantic(
                        node.primitive.name == node.init.uc_type.typename,
                        10,
                        node.name.coord,
                        name=node.name.name,
                    )

    def update_array_shape(self, array_node, array_shape, init_list_shape):
        dimensions = []

        # TODO Assume todos os elementos de init_list_shape tem o mesmo tamanho
        # Talvez seja melhor rodar essa função após a checagem pelo erro 12
        for elem in init_list_shape:
            dimensions.append(elem[0])

        # TODO Precisa checar que len(dimensions) == len(array_shape)
        # Verificar qual assert fazer nesse caso

        for i, dim in enumerate(array_shape):
            if dim is None:
                array_shape[i] = dimensions[i]

        array_node.array_shape = array_shape
        uc_types = []
        self.get_array_uc_types(array_node.uc_type, uc_types)

        for i, type in enumerate(uc_types):
            if type.dim is None:
                type.dim = dimensions[i]

        arr_decls = []

        self.get_array_decls(array_node, arr_decls)

        for i, arr_decl in enumerate(arr_decls):
            if arr_decl.dim is None:
                arr_decl.dim = Constant("int", str(dimensions[i]))

    def get_array_uc_types(self, uc_type, uc_types):
        if isinstance(uc_type, ArrayType):
            uc_types.append(uc_type)
            self.get_array_uc_types(uc_type.type, uc_types)

    def get_array_decls(self, node, array_decls):
        if isinstance(node, ArrayDecl):
            array_decls.append(node)
            self.get_array_decls(node.type, array_decls)

    def visit_InitList(self, node):
        for exp in node.exprs:
            # TODO Talvez seja preciso incluir no BinaryOp um campo que indica que o resultado é formado por constantes.
            #      Isto é, que é da forma 3 + 2 e não 3 + x
            if not isinstance(exp, InitList) and not isinstance(exp, BinaryOp):
                self._assert_semantic(isinstance(exp, Constant), 19, exp.coord)
                self.visit(exp)

            else:
                self.visit(exp)

        # TODO Precisa checar se todos os elementos da lista são do mesmo tipo?
        # TODO É valido um InitList vazio? Isto é int x = {};
        #      Se sim, então esse caso vai dar errado
        node.uc_type = node.exprs[0].uc_type
        node.length = len(node.exprs)

    def visit_ID(self, node):
        self._assert_semantic(
            self.symtab.lookup(node.name) is not None,
            1,
            node.coord,
            name=f"{node.name}",
        )
        node.uc_type = self.symtab.lookup(node.name)["uc_type"]
        node._link = self.symtab.lookup(node.name)["node"]

    def visit_VarDecl(self, node):
        """First visit the type to adjust the list of types to uCType objects.
        Then, get the name of variable and make sure it is not defined in the current scope, otherwise return an error.
        Next, insert its identifier in the symbol table.
        Finally, copy the type to the identifier.
        """
        type_name = node.type.name
        var_name = node.declname.name
        uc_type = self.typemap[type_name]
        node.uc_type = uc_type

        # Check if the variable is already declared
        self._assert_semantic(
            not self.symtab.already_defined(var_name),
            24,
            node.declname.coord,
            name=var_name,
        )

        self.symtab.add(var_name, {"uc_type": uc_type, "kind": "var", "node": node})
        self.visit(node.declname)
        return var_name, node

    def visit_BinaryOp(self, node):
        # Visit the left and right expression
        self.visit(node.left)
        ltype = node.left.uc_type
        self.visit(node.right)
        rtype = node.right.uc_type
        # - Make sure left and right operands have the same type
        self._assert_semantic(
            ltype == rtype,
            5,
            node.coord,
            name=node.op,
            ltype=ltype.typename,
            rtype=rtype.typename,
        )
        # - Make sure the operation is supported
        if node.op in {"==", "!=", "&&", "||", "<", ">", "<=", ">="}:
            self._assert_semantic(
                node.op in ltype.rel_ops,
                6,
                node.coord,
                name=node.op,
                ltype=f"type({ltype.typename})",
            )  # TODO Verificar se o ltype está certo
            node.uc_type = BoolType

        elif node.op in {"+", "-", "*", "/", "%"}:
            self._assert_semantic(
                node.op in ltype.binary_ops,
                6,
                node.coord,
                name=node.op,
                ltype=f"type({ltype.typename})",
            )  # TODO Verificar se o ltype está certo
            node.uc_type = ltype

        else:
            raise ("Não deveria chegar aqui")

    def visit_UnaryOp(self, node):
        # Visit the expression
        self.visit(node.expr)
        etype = node.expr.uc_type
        # - Make sure the operation is supported
        self._assert_semantic(node.op in etype.unary_ops, 25, node.coord, name=node.op)
        node.uc_type = etype

    def visit_Constant(self, node):
        type_name = node.type
        uc_type = self.typemap[type_name]
        node.uc_type = uc_type
        if type_name == "string":
            # TODO Não seria necessário somar 1 ao tamanho da string para levar em considereção o "\0" que finaliza a strig?
            node.length = len(node.value)
        if type_name == "int":
            node.value = int(node.value)

    def visit_Compound(self, node):
        for stmt in node.staments:
            if isinstance(stmt, Compound):
                self.symtab.begin_scope(node)
                self.visit(stmt)
                self.symtab.end_scope()
            else:
                self.visit(stmt)

    def visit_Return(self, node):
        if node.expr is not None:
            self.visit(node.expr)
            node.uc_type = node.expr.uc_type
        else:
            node.uc_type = VoidType

    def visit_Assert(self, node):
        # TODO Verificar se está certo
        self.visit(node.expr)
        self._assert_semantic(
            node.expr.uc_type == BoolType,
            3,
            node.expr.coord,
        )

    def visit_If(self, node):
        # Visit the condition
        self.visit(node.cond)
        # Make sure the condition is of type bool
        self._assert_semantic(
            node.cond.uc_type == BoolType,
            18,
            node.cond.coord,
        )
        # Visit the if and else blocks
        # TODO Talvez seja preciso iniciar um novo escopo aqui caso node.iftrue ou node.iffalse seja um Compound
        self.visit(node.iftrue)
        if node.iffalse is not None:
            self.visit(node.iffalse)


    def visit_Break(self, node):
        enclosing_loop = self.symtab.get_enclosing_loop()
        self._assert_semantic(enclosing_loop is not None, 7, node.coord)
        enclosing_loop.break_list.append(node)

    def visit_While(self, node):
        self.visit(node.cond)
        self._assert_semantic(
            node.cond.uc_type == BoolType,
            14,
            node.coord,
            ltype=f"type({node.cond.uc_type.typename})",
        )

        # TODO Verificar se está certo em iniciar o scope aqui
        self.symtab.begin_scope(node)
        self.visit(node.body)
        self.symtab.end_scope()

    def visit_For(self, node):
        self.symtab.begin_scope(node)

        if node.init is not None:
            self.visit(node.init)

        if node.cond is not None:
            self.visit(node.cond)
            self._assert_semantic(
                node.cond.uc_type == BoolType,
                14,
                node.coord,
                ltype=f"type({node.cond.uc_type.typename})",
            )

        if node.next is not None:
            self.visit(node.next)

        # TODO Talvez precise checar se o body não é vazio
        # Por exemplo, for (;;);
        self.visit(node.body)

        self.symtab.end_scope()

    def visit_Read(self, node):
        # TODO read pode ter mais de 1 argumento?
        #      ou node.names sempre vai ser só 1 elemento?
        # TODO O correto aqui nao seria olhar se o kind de node.names é var?
        #      Talvez também possa ser array. ( Não sei se precisa distinguir array de array ref)
        self.visit(node.names)
        self._assert_semantic(
            not isinstance(node.names, Constant), 22, node.names.coord, name="Constant"
        )

    def visit_Print(self, node):
        # TODO print pode ter mais de 1 argumento?
        #      ou node.exprs sempre vai ser só 1 elemento?
        if node.expr is not None:
            self.visit(node.expr)

            if isinstance(node.expr, ExprList):
                for expr in node.expr.exprs:
                    self._assert_semantic(
                        expr.uc_type in {IntType, CharType, StringType},
                        20,
                        expr.coord,
                    )

            # TODO Pensar se essa é a melhor formar de fazer isso
            elif node.expr.uc_type not in {IntType, CharType, StringType}:
                if isinstance(node.expr.uc_type, ArrayType) or isinstance(
                    node.expr.uc_type, FuncType
                ):
                    self._assert_semantic(
                        node.expr.uc_type in {IntType, CharType, StringType},
                        21,
                        node.expr.coord,
                        node.expr.name,
                    )
                else:  # Acho que só entra aqui se o tipo for VoidType
                    self._assert_semantic(
                        node.expr.uc_type in {IntType, CharType, StringType},
                        20,
                        node.expr.coord,
                        node.expr.name,
                    )

    def visit_ExprList(self, node):
        for expr in node.exprs:
            self.visit(expr)

    def visit_ArrayRef(self, node):
        self.visit(node.subscript)
        self._assert_semantic(
            node.subscript.uc_type == IntType,
            2,
            node.subscript.coord,
            ltype=f"type({node.subscript.uc_type.typename})",
        )

        self.visit(node.name)
        node.uc_type = node.name.uc_type.type

    def visit_Assignment(self, node):
        # visit right side
        self.visit(node.rvalue)
        rtype = node.rvalue.uc_type
        # visit left side (must be a location)
        _var = node.lvalue
        self.visit(_var)
        # TODO Verificar se não tem problema deixar as 2 linhas abaixos comentadas. A princípio a verificação já foi feita no visit_ID
        # if isinstance(_var, ID):
        #     self._assert_semantic(
        #         _var.scope is not None,
        #         1,
        #         node.coord,
        #         name=f"{_var.name}NAO SEI SE ISSO ERA PRA ACONTECER",
        #     )
        ltype = node.lvalue.uc_type

        # Check that assignment is allowed
        self._assert_semantic(
            ltype == rtype,
            4,
            node.coord,
            ltype=f"type({ltype.typename})",
            rtype=f"type({rtype.typename})",
        )  # TODO Verificar se está certo colocar .typename no ltype e rtype

        # Check that assign_ops is supported by the type
        self._assert_semantic(
            node.op in ltype.assign_ops, 5, node.coord, name=node.op, ltype=ltype
        )

        node.uc_type = ltype


if __name__ == "__main__":
    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file", help="Path to file to be semantically checked", type=str
    )
    args = parser.parse_args()

    # get input path
    input_file = args.input_file
    input_path = pathlib.Path(input_file)

    # check if file exists
    if not input_path.exists():
        print("Input", input_path, "not found", file=sys.stderr)
        sys.exit(1)

    # set error function
    p = UCParser()
    # open file and parse it
    with open(input_path) as f:
        ast = p.parse(f.read())
        sema = Visitor()
        sema.visit(ast)
