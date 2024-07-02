import argparse
import pathlib
import sys
from typing import Dict, List, Tuple
from uc.uc_ast import (
    ID,
    ArrayDecl,
    ArrayRef,
    Assert,
    Assignment,
    BinaryOp,
    Break,
    Compound,
    Constant,
    Decl,
    DeclList,
    ExprList,
    For,
    FuncCall,
    FuncDecl,
    FuncDef,
    GlobalDecl,
    If,
    InitList,
    Node,
    ParamList,
    Print,
    Program,
    Return,
    UnaryOp,
    VarDecl,
    While,
)
from uc.uc_block import (
    CFG,
    BasicBlock,
    Block,
    ConditionBlock,
    EmitBlocks,
    format_instruction,
)
from uc.uc_interpreter import Interpreter
from uc.uc_parser import UCParser
from uc.uc_sema import NodeVisitor, Visitor
from uc.uc_type import VoidType


class CodeGenerator(NodeVisitor):
    """
    Node visitor class that creates 3-address encoded instruction sequences
    with Basic Blocks & Control Flow Graph.
    """

    def __init__(self, viewcfg: bool):
        self.viewcfg: bool = viewcfg
        self.current_block: Block = None
        self.jump_to_exit: List[Block] = []
        # Stack que mantém o bloco de saída atual, utilizado para o break
        self.current_exit: List[Block] = []
        self.return_reg: str = None

        # version dictionary for temporaries. We use the name as a Key
        self.fname: str = "_glob_"
        self.versions: Dict[str, int] = {self.fname: 0}
        self.var_version: Dict[
            str, Dict[str, int]
        ] = {}  # {fname : {varname : version}}}
        self.stmts_version: Dict[
            str, Dict[str, int]
        ] = {}  # {fname : {stmt : version}}}

        # The generated code (list of tuples)
        # At the end of visit_program, we call each function definition to emit
        # the instructions inside basic blocks. The global instructions that
        # are stored in self.text are appended at beginning of the code
        self.code: List[Tuple[str]] = []

        self.text: List[
            Tuple[str]
        ] = []  # Used for global declarations & constants (list, strings)

    def show(self, buf=sys.stdout):
        _str = ""
        for _code in self.code:
            _str += format_instruction(_code) + "\n"
        buf.write(_str)

    def new_temp(self) -> str:
        """
        Create a new temporary variable of a given scope (function name).
        """
        if self.fname not in self.versions:
            self.versions[self.fname] = 1
        name = "%" + "%d" % (self.versions[self.fname])
        self.versions[self.fname] += 1
        return name

    def new_var(self, varname) -> str:
        if self.fname not in self.var_version:
            self.var_version[self.fname] = {}
        if varname not in self.var_version[self.fname]:
            self.var_version[self.fname][varname] = 1
        name = "%" + varname + ".%d" % (self.var_version[self.fname][varname])
        self.var_version[self.fname][varname] += 1
        return name

    def new_stmt(self, stmt) -> str:
        if self.fname not in self.stmts_version:
            self.stmts_version[self.fname] = {}
        if stmt not in self.stmts_version[self.fname]:
            self.stmts_version[self.fname][stmt] = 1
        name = stmt + ".%d" % (self.stmts_version[self.fname][stmt])
        self.stmts_version[self.fname][stmt] += 1
        return name

    def new_text(self, typename: str) -> str:
        """
        Create a new literal constant on global section (text).
        """
        name = "@." + typename + "." + "%d" % (self.versions["_glob_"])
        self.versions["_glob_"] += 1
        return name

    def get_name(self, node: Node) -> str:
        """
        Return the name of a variable or constant.
        """
        if isinstance(node, ID):
            return node.name
        elif isinstance(node, VarDecl):
            return node.declname.name
        elif isinstance(node, ArrayDecl):
            return self.get_name(node.type)

    def gen_args(self, params):
        args = []
        if params is None:
            return args
        for param in params.params:
            args.append((param.type.type.name, self.new_temp()))
        return args

    def alloc_type(self, type, varname):
        inst = ("alloc_" + type, varname)
        self.current_block.append(inst)

    def literal_type(self, type, value, varname):
        inst = ("literal_" + type, value, varname)
        self.current_block.append(inst)

    def load_type(self, type, varname, target, node):
        if not isinstance(node, ArrayRef):
            inst = ("load_" + type, varname, target)
        else:
            inst = ("load_" + type + "_*", varname, target)
        self.current_block.append(inst)

    def store_type(self, type, source, target, node):
        if not isinstance(node, ArrayRef):
            inst = ("store_" + type, source, target)
        else:
            inst = ("store_" + type + "_*", source, target)
        self.current_block.append(inst)

    def global_type(self, type, varname, value=None):
        inst = "global_" + type
        if value is None:
            self.text.append((inst, varname))
        else:
            self.text.append((inst, varname, value))

    def need_load(self, node: Node):
        # TODO confirmar se são só esses dois casos
        return (
            not isinstance(node, Constant)
            and not isinstance(node, BinaryOp)
            and not isinstance(node, FuncCall)
            and not (isinstance(node, ID) and not node.gen_location[0] == "@")
        )

    def build_array_type(self, typename, array_shape):
        result = typename
        for dim in array_shape:
            result += "_" + str(dim)
        return result

    def visit_Constant(self, node: Constant):
        if node.type == "string":
            _target = self.new_text("str")
            inst = ("global_string", _target, node.value)
            self.text.append(inst)
        else:
            # Create a new temporary variable name
            _target = self.new_temp()
            # Make the SSA opcode and append to list of generated instructions
            self.literal_type(node.type, node.value, _target)
            # inst = ("literal_" + node.type, node.value, _target)
            # self.current_block.append(inst)
        # Save the name of the temporary variable where the value was placed
        node.gen_location = _target

    def visit_BinaryOp(self, node: BinaryOp):
        binary_ops = {
            "+": "add",
            "-": "sub",
            "*": "mul",
            "/": "div",
            "%": "mod",
            "<": "lt",
            "<=": "le",
            ">": "gt",
            ">=": "ge",
            "==": "eq",
            "!=": "ne",
            "&&": "and",
            "||": "or",
        }
        # Visit the left and right expressions
        self.visit(node.left)
        self.visit(node.right)

        # - Load the location containing the left expression
        if self.need_load(node.left):
            left_temp = self.new_temp()
            self.load_type(
                node.left.uc_type.typename, node.left.gen_location, left_temp, node.left
            )
        else:
            left_temp = node.left.gen_location

        # - Load the location containing the right expression
        if self.need_load(node.right):
            right_temp = self.new_temp()
            self.load_type(
                node.right.uc_type.typename,
                node.right.gen_location,
                right_temp,
                node.right,
            )
        else:
            right_temp = node.right.gen_location

        # Make a new temporary for storing the result
        target = self.new_temp()

        # Create the opcode and append to list
        opcode = binary_ops[node.op] + "_" + node.left.uc_type.typename
        inst = (opcode, left_temp, right_temp, target)
        self.current_block.append(inst)

        # Store location of the result on the node
        node.gen_location = target

    def visit_UnaryOp(self, node: UnaryOp):
        self.visit(node.expr)

        temp = self.new_temp()

        # TODO Implementar checagem se load é necessário
        self.load_type(
            node.expr.uc_type.typename, node.expr.gen_location, temp, node.expr
        )

        if node.op == "-":
            zero = self.new_temp()
            target = self.new_temp()
            self.literal_type(node.expr.uc_type.typename, 0, zero)
            self.current_block.append(
                ("sub_" + node.expr.uc_type.typename, zero, temp, target)
            )
        else:
            target = self.new_temp()
            self.current_block.append(
                ("not_" + node.expr.uc_type.typename, temp, target)
            )

        node.gen_location = target

    def visit_Print(self, node: Print):
        # Handles print()
        if node.expr is None:
            self.current_block.append(("print_void",))

        elif isinstance(node.expr, ExprList):
            for expr in node.expr.exprs:
                self.handle_print(expr)

        else:
            self.handle_print(node.expr)

    def handle_print(self, expr: Node):
        self.visit(expr)

        # Doen't need to load if it is a constant or binary op
        if not self.need_load(expr):
            self.current_block.append(
                ("print_" + expr.uc_type.typename, expr.gen_location)
            )
        else:
            temp = self.new_temp()
            self.load_type(expr.uc_type.typename, expr.gen_location, temp, expr)
            self.current_block.append(("print_" + expr.uc_type.typename, temp))

    def visit_FuncCall(self, node: FuncCall):
        if node.args is not None:
            if isinstance(node.args, ExprList):
                for arg in node.args.exprs:
                    self.visit(arg)

                    if self.need_load(arg):
                        temp = self.new_temp()
                        self.load_type(
                            arg.uc_type.typename, arg.gen_location, temp, arg
                        )
                        arg.gen_location = temp

                for arg in node.args.exprs:
                    self.current_block.append(
                        ("param_" + arg.uc_type.typename, arg.gen_location)
                    )
            else:
                self.visit(node.args)

                if self.need_load(node.args):
                    temp = self.new_temp()
                    self.load_type(
                        node.args.uc_type.typename,
                        node.args.gen_location,
                        temp,
                        node.args,
                    )
                    node.args.gen_location = temp

                self.current_block.append(
                    ("param_" + node.args.uc_type.typename, node.args.gen_location)
                )

        if node.uc_type is not VoidType:
            target = self.new_temp()
            node.gen_location = target
            self.current_block.append(
                ("call_" + node.uc_type.typename, "@" + node.name.name, target)
            )

        else:
            self.current_block.append(
                ("call_void", node.name.name)
            )  # TODO Talvez esteja faltando o @

    def visit_ID(self, node: ID):
        # TODO Perguntar no lab se é só isso
        if not isinstance(node._link, FuncDef):
            node.gen_location = node._link.gen_location

    def visit_VarDecl(self, node: VarDecl):
        # Allocate on stack memory
        _varname = self.new_var(node.declname.name)
        # TODO Não sei se está certo, perguntar no lab
        # Talvez precise adicionar esse atributo ao ID também
        # Ou então modificar a analise semantica do visit_ID para salvar essa node no ID
        # Por enquanto foi feito essa segunda opção
        node.gen_location = _varname
        inst = ("alloc_" + node.type.name, _varname)
        self.current_block.append(inst)

        # TODO Esse código não funciona, perguntar no lab por que foi posto aqui.
        # Store optional init val
        # _init = node.decl.init
        # if _init is not None:
        #     self.visit(_init)
        #     inst = (
        #         "store_" + node.type.name,
        #         _init.gen_location,
        #         node.declname.gen_location,
        #     )
        #     self.current_block.append(inst)

    def visit_ArrayDecl(self, node: ArrayDecl):
        _varname = self.new_var(self.get_name(node))
        node.gen_location = _varname
        inst = "alloc_" + self.build_array_type(node.uc_type.typename, node.array_shape)

        self.current_block.append((inst, _varname))

    def visit_ArrayRef(self, node: ArrayRef):
        arr = self.get_array_id(node)
        self.visit(arr)
        shape = arr._link.array_shape
        idxs = []
        self.get_array_idxs(node, idxs)
        assert len(shape) == len(idxs), "Incomplete array reference"

        prev = self.new_temp()
        self.current_block.append(("literal_int", 0, prev))
        for idx, dim in zip(reversed(idxs[1:]), shape[1:]):
            self.visit(idx)
            if self.need_load(idx):
                temp = self.new_temp()
                self.load_type(idx.uc_type.typename, idx.gen_location, temp, idx)
                idx.gen_location = temp
            temp_dim = self.new_temp()
            self.current_block.append(("literal_int", dim, temp_dim))
            temp_mul = self.new_temp()
            self.current_block.append(("mul_int", idx.gen_location, temp_dim, temp_mul))
            temp_add = self.new_temp()
            self.current_block.append(("add_int", temp_mul, prev, temp_add))
            prev = temp_add

        idx = idxs[0]
        self.visit(idx)
        if self.need_load(idx):
            temp = self.new_temp()
            self.load_type(idx.uc_type.typename, idx.gen_location, temp, idx)
            idx.gen_location = temp

        final_offset = self.new_temp()
        self.current_block.append(("add_int", idx.gen_location, prev, final_offset))

        addr = self.new_temp()

        self.current_block.append(
            (
                "elem_" + arr.uc_type.typename,
                arr.gen_location,
                final_offset,
                addr,
            )
        )
        node.gen_location = addr
        # value = self.new_temp()
        # self.current_block.append(
        #     ("load_" + arr.uc_type.typename + "_*", addr, value)
        # )

    def get_array_id(self, node: ArrayRef):
        if isinstance(node.name, ArrayRef):
            return self.get_array_id(node.name)
        else:
            return node.name

    def get_array_idxs(self, node: ArrayRef, idxs):
        if isinstance(node.name, ArrayRef):
            idxs.append(node.subscript)
            self.get_array_idxs(node.name, idxs)
        else:
            idxs.append(node.subscript)

    def visit_Program(self, node: Program):
        # Visit all of the global declarations
        for _decl in node.gdecls:
            self.jump_to_exit = []
            self.current_exit: List[Block] = []
            self.return_reg: str = None
            self.visit(_decl)
        # At the end of codegen, first init the self.code with
        # the list of global instructions allocated in self.text
        self.code = self.text.copy()
        # Also, copy the global instructions into the Program node
        node.text = self.text.copy()
        # After, visit all the function definitions and emit the
        # code stored inside basic blocks.
        for _decl in node.gdecls:
            if isinstance(_decl, FuncDef):
                # _decl.cfg contains the Control Flow Graph for the function
                # cfg points to start basic block
                bb = EmitBlocks()
                bb.visit(_decl.cfg)
                for _code in bb.code:
                    self.code.append(_code)

        if self.viewcfg:  # evaluate to True if -cfg flag is present in command line
            for _decl in node.gdecls:
                if isinstance(_decl, FuncDef):
                    dot = CFG(_decl.decl.name.name)
                    dot.view(_decl.cfg)  # _decl.cfg contains the CFG for the function

    def visit_GlobalDecl(self, node: GlobalDecl):
        for decl in node.decls:
            # TODO lidar com ArrayDecl
            if isinstance(decl.type, VarDecl):
                if decl.init is not None:
                    # TODO Assume que o init value é uma constante
                    # Perguntar no lab o que fazer se esse não for o caso
                    # self.visit(decl.init)
                    inst = (
                        "global_" + decl.name.uc_type.typename,
                        "@" + decl.name.name,
                        decl.init.value,
                    )
                    self.text.append(inst)
                else:
                    inst = (
                        "global_" + decl.name.uc_type.typename,
                        "@" + decl.name.name,
                    )
                    self.text.append(inst)

                decl.gen_location = "@" + decl.name.name
                decl.name._link.gen_location = "@" + decl.name.name

            elif isinstance(decl.type, ArrayDecl):
                self.visit(decl.init)

                # TODO Talvez de problemas caso seja um array de strings
                # eg: char str[] = "string"
                if isinstance(decl.init, InitList):
                    glob = self.new_text("const_" + self.get_name(decl.type))
                    decl.init.gen_location = glob
                    inst = (
                        "global_"
                        + self.build_array_type(
                            decl.type.uc_type.typename, decl.type.array_shape
                        ),
                        glob,
                        decl.init.value,
                    )
                    self.text.append(inst)
                    decl.gen_location = glob
                    decl.name._link.gen_location = glob
                    decl.type.gen_location = glob

    def visit_FuncDef(self, node: FuncDef):
        bb = BasicBlock("")
        self.current_block = bb
        node.cfg = bb  # TODO não sei se está certo

        self.visit(node.decl)

        # TODO O que fazer quando a função não tem return?
        # Informa o nó Return de qual o registrador alocado para return
        for ret in node.return_list:
            if node.type.name != "void":
                ret.return_reg = node.decl.type.return_reg

        # TODO Para ficar igual o notebook vai ser preciso criar
        # uma função auxiliar que coleta todas as declarações,
        # para que seja possível alocar todas antes de inicializar elas.
        # Perguntar no lab se é necessário fazer dessa forma.

        self.visit(node.body)

        bb = BasicBlock(r"%exit")
        # Handles when function doens't have return
        if node.return_list == []:
            self.current_block.append(("jump", r"%exit"))
            self.jump_to_exit.append(self.current_block)
            # bb.predecessors.append(self.current_block)
            # self.current_block.branch = bb
            self.current_block.next_block = bb
            self.current_block = bb
            self.current_block.append(("exit:",))
            self.current_block.append(("return_void",))

        else:
            # self.current_block.append(("jump", r"%exit"))
            # bb.predecessors.append(self.current_block)
            if self.current_block.instructions[-1] != ("jump", r"%exit"):
                self.current_block.append(("jump", r"%exit"))
                self.jump_to_exit.append(self.current_block)

            self.current_block.next_block = bb
            self.current_block.branch = bb
            self.current_block = bb

            self.current_block.append(("exit:",))

            if node.type.name != "void" and self.return_reg:
                target = self.new_temp()
                self.load_type(node.type.name, self.return_reg, target, None)
                self.current_block.append(("return_" + node.type.name, target))
            else:
                self.current_block.append(("return_void",))

        for block in self.jump_to_exit:
            block.branch = bb
            bb.predecessors.append(block)

    def visit_Decl(self, node: Decl):
        self.visit(node.type)

        # TODO Talvez isso de problemas
        if isinstance(node.type, VarDecl):
            node.gen_location = node.type.gen_location

        # TODO Lidar com node.init
        # Pensar se essa é a forma certa de fazer, relacionado com o TODO do visit_FuncDef
        # TODO talvez precisa fazer um tratamento diferente de acordo com o tipo do node
        if node.init is not None:
            self.visit(node.init)
            # TODO Colocar uma checagem para ver se é necesśario um load
            if isinstance(node.type, VarDecl):
                self.store_type(
                    node.type.type.name,
                    node.init.gen_location,
                    node.type.gen_location,
                    node.type,
                )

            elif isinstance(node.type, ArrayDecl):
                if isinstance(node.init, InitList):
                    glob = self.new_text("const_" + self.get_name(node.type))
                    node.init.gen_location = glob
                    inst = (
                        "global_"
                        + self.build_array_type(
                            node.type.uc_type.typename, node.type.array_shape
                        ),
                        glob,
                        node.init.value,
                    )
                    self.text.append(inst)

                self.store_type(
                    self.build_array_type(
                        node.type.uc_type.typename, node.type.array_shape
                    ),
                    node.init.gen_location,
                    node.type.gen_location,
                    node.type,
                )

    def visit_InitList(self, node: InitList):
        for expr in node.exprs:
            if isinstance(expr, InitList):
                self.visit(expr)
            node.value.append(expr.value)

    def visit_FuncDecl(self, node: FuncDecl):
        self.fname = node.type.declname.name

        # Append define instruction
        define_str = "define_" + node.type.type.name
        args = self.gen_args(node.params)
        define_inst = (define_str, "@" + self.fname, args)
        self.current_block.append(define_inst)

        # TODO Perguntar no lab se está certo essa label
        # e também perguntar se está certo criar um bloco pro funcDef composto
        # apenas da instrução define

        bb = BasicBlock(r"%entry")
        # Talvez não seja preciso nesse caso,
        # pois o bloco predecessor só tem o "define_"
        bb.predecessors.append(self.current_block)
        self.current_block.next_block = bb
        self.current_block = bb

        self.current_block.append(("entry:",))

        return_type = node.type.type.name
        if return_type != "void":
            return_reg = self.new_temp()
            self.alloc_type(return_type, return_reg)
            node.return_reg = return_reg  # TODO pensar se é a melhor forma de fazer isso ou se é necessário
            self.return_reg = return_reg

        # Allocate space for parameters
        if node.params is not None:
            self.visit(node.params)

            for temp, param in zip(args, node.params.params):
                inst = ("store_" + temp[0], temp[1], param.gen_location)
                self.current_block.append(inst)

    def visit_ParamList(self, node: ParamList):
        for param in node.params:
            self.visit(param)

    def visit_Compound(self, node: Compound):
        if node.staments is not None:
            for statement in node.staments:
                self.visit(statement)
                # TODO Pensar se isso está certo
                # A ideia é que se chegar num break, nao precisa gerar o que vem após
                if isinstance(statement, Break):
                    return
                elif isinstance(statement, Return):
                    return

    def visit_Assert(self, node: Assert):
        # Insert assert string in global section
        str_loc = self.new_text("str")
        assert_str = (
            f"assertion_fail on {node.expr.coord.line}:{node.expr.coord.column}"
        )
        self.global_type("string", str_loc, assert_str)

        # TODO Talvez precise criar uma função para gerar a label numerada
        # Por exemplo, assert.true.1, assert.true.2. Isso seria necessário caso
        # uma função tenha mais de 1 asssert.
        cb = ConditionBlock("%assert.cond")
        self.current_block.append(("jump", "%assert.cond"))
        cb.predecessors.append(self.current_block)

        self.current_block.branch = cb
        self.current_block.next_block = cb
        self.current_block = cb
        self.current_block.append(("assert.cond:",))

        self.visit(node.expr)

        cbranch = ("cbranch", node.expr.gen_location, "%assert.true", "%assert.false")
        self.current_block.append(cbranch)

        taken_block = BasicBlock("%assert.true")
        taken_block.append(("assert.true:",))
        taken_block.predecessors.append(cb)

        # TODO Nesse momento, ainda não temos como conectar
        # esse bloco ao seu destino.
        fall_throug_block = BasicBlock("%assert.false")
        fall_throug_block.append(("assert.false:",))
        fall_throug_block.predecessors.append(cb)
        self.jump_to_exit.append(fall_throug_block)

        cb.taken = taken_block
        cb.fall_through = fall_throug_block

        self.current_block.next_block = fall_throug_block
        self.current_block = fall_throug_block

        self.current_block.append(("print_string", str_loc))
        # TODO asssume que "%exit" sempre é a label
        self.current_block.append(("jump", "%exit"))

        self.current_block.next_block = taken_block
        self.current_block = taken_block

    def visit_Return(self, node: Return):
        # TODO Pensar se esse código funciona quando a função não tem return
        # TODO terminar
        # TODO Pensar em como criar o bloco nesse caso

        if node.expr is not None:
            self.visit(node.expr)
        if (
            node.uc_type.typename != "void" and node.return_reg
        ):  # TODO Talvez não precise da segunda condição
            value = node.expr.gen_location
            if self.need_load(node.expr):
                temp = self.new_temp()
                self.load_type(node.expr.uc_type.typename, value, temp, node.expr)
                value = temp
            inst = (
                "store_" + node.expr.uc_type.typename,
                value,
                node.return_reg,
            )
            self.current_block.append(inst)

    def visit_For(self, node: For):
        # TODO tratar oq fazer quando tem break dentro do for
        if node.init is not None:
            self.visit(node.init)

        cond_str = self.new_stmt("for.cond")
        body_str = self.new_stmt("for.body")
        inc_str = self.new_stmt("for.inc")
        end_str = self.new_stmt("for.end")
        for_cond = ConditionBlock("%" + cond_str)
        for_body = BasicBlock("%" + body_str)
        for_inc = BasicBlock("%" + inc_str)
        for_end = BasicBlock("%" + end_str)
        self.current_exit.append(for_end)

        # Emit code for the condition
        self.current_block.append(("jump", "%" + cond_str))
        for_cond.predecessors.append(self.current_block)
        self.current_block.branch = for_cond
        self.current_block.next_block = for_cond
        self.current_block = for_cond

        self.current_block.append((cond_str + ":",))
        self.visit(node.cond)
        cbranch = ("cbranch", node.cond.gen_location, "%" + body_str, "%" + end_str)
        self.current_block.append(cbranch)
        for_body.predecessors.append(self.current_block)
        for_end.predecessors.append(self.current_block)

        # Emit code for the body
        self.current_block.taken = for_body
        self.current_block.fall_through = for_end
        self.current_block.next_block = for_body
        self.current_block = for_body

        self.current_block.append((body_str + ":",))
        # TODO lidar com breaks
        self.visit(node.body)
        self.current_block.append(("jump", "%" + inc_str))
        for_inc.predecessors.append(self.current_block)

        # Emit code for the increment
        self.current_block.branch = for_inc
        self.current_block.next_block = for_inc
        self.current_block = for_inc

        self.current_block.append((inc_str + ":",))
        self.visit(node.next)
        self.current_block.append(("jump", "%" + cond_str))
        for_cond.predecessors.append(self.current_block)

        # Emit code for the end
        self.current_block.branch = for_cond
        self.current_block.next_block = for_end
        self.current_block = for_end
        self.current_block.append((end_str + ":",))

        self.current_exit.pop()

    def visit_While(self, node: While):
        cond_str = self.new_stmt("while.cond")
        body_str = self.new_stmt("while.body")
        end_str = self.new_stmt("while.end")

        cond = ConditionBlock("%" + cond_str)
        body = BasicBlock("%" + body_str)
        end = BasicBlock("%" + end_str)

        self.current_block.append(("jump", "%" + cond_str))
        cond.predecessors.append(self.current_block)

        self.current_block.branch = cond
        self.current_block.next_block = cond
        self.current_block = cond

        self.current_block.append((cond_str + ":",))
        self.visit(node.cond)
        cbranch = ("cbranch", node.cond.gen_location, "%" + body_str, "%" + end_str)
        self.current_block.append(cbranch)
        body.predecessors.append(self.current_block)
        end.predecessors.append(self.current_block)

        self.current_block.taken = body
        self.current_block.fall_through = end
        self.current_block.next_block = body
        self.current_block = body

        self.current_block.append((body_str + ":",))
        self.visit(node.body)

        # TODO Lidar com break

        self.current_block.append(("jump", "%" + cond_str))
        cond.predecessors.append(self.current_block)

        self.current_block.branch = cond
        self.current_block.next_block = end
        self.current_block = end

        self.current_block.append((end_str + ":",))

    def visit_If(self, node: If):
        cond_str = self.new_stmt("if.cond")
        then_str = self.new_stmt("if.then")
        else_str = self.new_stmt("if.else")
        end_str = self.new_stmt("if.end")

        cond = ConditionBlock("%" + cond_str)
        then = BasicBlock("%" + then_str)
        else_ = BasicBlock("%" + else_str)
        end = BasicBlock("%" + end_str)

        self.current_block.append(("jump", "%" + cond_str))
        cond.predecessors.append(self.current_block)

        self.current_block.branch = cond
        self.current_block.next_block = cond
        self.current_block = cond

        self.current_block.append((cond_str + ":",))
        self.visit(node.cond)

        if node.iffalse is not None:
            cbranch = (
                "cbranch",
                node.cond.gen_location,
                "%" + then_str,
                "%" + else_str,
            )
            self.current_block.append(cbranch)
            then.predecessors.append(self.current_block)
            else_.predecessors.append(self.current_block)
            self.current_block.taken = then
            self.current_block.fall_through = else_
        else:
            cbranch = ("cbranch", node.cond.gen_location, "%" + then_str, "%" + end_str)
            self.current_block.append(cbranch)
            then.predecessors.append(self.current_block)
            end.predecessors.append(self.current_block)
            self.current_block.taken = then
            self.current_block.fall_through = end

        self.current_block.next_block = then
        self.current_block = then
        self.current_block.append((then_str + ":",))
        self.visit(node.iftrue)
        if not self.has_break(node.iftrue):
            if self.has_return(node.iftrue):
                self.current_block.append(("jump", "%exit"))
                self.jump_to_exit.append(self.current_block)
            else:
                self.current_block.append(("jump", "%" + end_str))
                end.predecessors.append(self.current_block)
                self.current_block.branch = end

        if node.iffalse is not None:
            self.current_block.next_block = else_
            self.current_block = else_
            self.current_block.append((else_str + ":",))
            self.visit(node.iffalse)
            if not self.has_break(node.iffalse):
                if self.has_return(node.iffalse):
                    self.current_block.append(("jump", "%exit"))
                    self.jump_to_exit.append(self.current_block)
                else:
                    self.current_block.append(("jump", "%" + end_str))
                    end.predecessors.append(self.current_block)
                    self.current_block.branch = end

        # Se tanto o if quanto o else tem return, nao precisa do bloco de saida
        if not (self.has_return(node.iftrue) and self.has_return(node.iffalse)):
            self.current_block.next_block = end
            self.current_block = end
            self.current_block.append((end_str + ":",))

    def has_return(self, node):
        if isinstance(node, Return):
            return True
        elif isinstance(node, Compound):
            for stmt in node.staments:
                if isinstance(stmt, Return):
                    return True
        return False

    def has_break(self, node):
        if isinstance(node, Break):
            return True
        elif isinstance(node, Compound):
            for stmt in node.staments:
                if isinstance(stmt, Break):
                    return True
        return False

    def visit_Assignment(self, node: Assignment):
        # TODO Lidar com os diferentes tipos de lvalue, rvalue e op

        self.visit(node.rvalue)
        self.visit(node.lvalue)

        target = node.lvalue.gen_location
        value = node.rvalue.gen_location
        if self.need_load(node.rvalue):
            temp = self.new_temp()
            self.load_type(
                node.rvalue.uc_type.typename,
                node.rvalue.gen_location,
                temp,
                node.rvalue,
            )
            value = temp

        var_type = node.lvalue.uc_type.typename
        self.store_type(var_type, value, target, node.lvalue)

    def visit_DeclList(self, node: DeclList):
        for decl in node.decls:
            self.visit(decl)

    def visit_Break(self, node: Break):
        self.current_block.append(("jump", self.current_exit[-1].label))
        self.current_exit[-1].predecessors.append(self.current_block)
        self.current_block.branch = self.current_exit[-1]


if __name__ == "__main__":
    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        help="Path to file to be used to generate uCIR. By default, this script only runs the interpreter on the uCIR. \
              Use the other options for printing the uCIR, generating the CFG or for the debug mode.",
        type=str,
    )
    parser.add_argument(
        "--ir",
        help="Print uCIR generated from input_file.",
        action="store_true",
    )
    parser.add_argument(
        "--cfg", help="Show the cfg of the input_file.", action="store_true"
    )
    parser.add_argument(
        "--debug", help="Run interpreter in debug mode.", action="store_true"
    )
    args = parser.parse_args()

    print_ir = args.ir
    create_cfg = args.cfg
    interpreter_debug = args.debug

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

    gen = CodeGenerator(create_cfg)
    gen.visit(ast)
    gencode = gen.code

    if print_ir:
        print("Generated uCIR: --------")
        gen.show()
        print("------------------------\n")

    vm = Interpreter(interpreter_debug)
    vm.run(gencode)
