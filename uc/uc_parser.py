import argparse
import pathlib
import sys
from ply.yacc import yacc
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
    EmptyStatement,
    ExprList,
    For,
    FuncCall,
    FuncDecl,
    FuncDef,
    GlobalDecl,
    If,
    InitList,
    ParamList,
    Print,
    Program,
    Read,
    Return,
    Type,
    UnaryOp,
    VarDecl,
    While,
)
from uc.uc_lexer import UCLexer


class Coord:
    """Coordinates of a syntactic element. Consists of:
    - Line number
    - (optional) column number, for the Lexer
    """

    __slots__ = ("line", "column")

    def __init__(self, line, column=None):
        self.line = line
        self.column = column

    def __str__(self):
        if self.line and self.column is not None:
            coord_str = "@ %s:%s" % (self.line, self.column)
        elif self.line:
            coord_str = "@ %s" % (self.line)
        else:
            coord_str = ""
        return coord_str


class UCParser:
    def __init__(self, debug=True):
        """Create a new uCParser."""
        self.uclex = UCLexer(self._lexer_error)
        self.uclex.build()
        self.tokens = self.uclex.tokens

        self.ucparser = yacc(module=self, start="program", debug=debug)
        # Keeps track of the last token given to yacc (the lookahead token)
        self._last_yielded_token = None

    def parse(self, text, debuglevel=0):
        self.uclex.reset_lineno()
        self._last_yielded_token = None
        return self.ucparser.parse(input=text, lexer=self.uclex, debug=debuglevel)

    def _lexer_error(self, msg, line, column):
        # use stdout to match with the output in the .out test files
        print("LexerError: %s at %d:%d" % (msg, line, column), file=sys.stdout)
        sys.exit(1)

    def _parser_error(self, msg, coord=None):
        # use stdout to match with the output in the .out test files
        if coord is None:
            print("ParserError: %s" % (msg), file=sys.stdout)
        else:
            print("ParserError: %s %s" % (msg, coord), file=sys.stdout)
        sys.exit(1)

    def _token_coord(self, p, token_idx):
        last_cr = p.lexer.lexer.lexdata.rfind("\n", 0, p.lexpos(token_idx))
        if last_cr < 0:
            last_cr = -1
        column = p.lexpos(token_idx) - (last_cr)
        return Coord(p.lineno(token_idx), column)
    
    precedence = (
        ('left', 'COMMA'),
        ('right',"EQUALS"),
        ('left', "OR"),
        ('left', "AND"),
        ('left', "EQ", "NE"),
        ('left', "LT", "LE", "GT", "GE"),
        ('left', "PLUS", "MINUS"),
        ('left', "TIMES", "DIVIDE", "MOD"),
        ('right', "NOT"), 
    )

    def p_program(self, p):
        """program  : global_declaration_list"""
        p[0] = Program(p[1])

    def p_global_declaration_list(self, p):
        """global_declaration_list : global_declaration
                                   | global_declaration_list global_declaration
        """
        p[0] = [p[1]] if len(p) == 2 else p[1] + [p[2]]

    def p_global_declaration_1(self, p):
        """global_declaration    : declaration"""
        p[0] = GlobalDecl(p[1])

    def p_global_declaration_2(self, p):
        """global_declaration    : function_definition"""
        p[0] = p[1]

    def p_function_definition(self, p):
        """function_definition   : type_specifier declarator compound_statement"""
        p[2].primitive = p[1]
        decl = Decl(p[2].identifier, p[2], None)
        p[0] = FuncDef(p[1], decl, p[3])

    def p_type_specifier(self, p):
        """type_specifier : VOID
                          | CHAR
                          | INT
        """
        coord = self._token_coord(p, 1)
        p[0] = Type(p[1], coord=coord)
        

    def p_declarator_id(self, p):
        """declarator : ID"""
        coord = self._token_coord(p, 1)
        identifier = ID(p[1], coord=coord)
        p[0] = VarDecl(identifier, None)
    
    # TODO Entender quando essa regra ocorre
    # TODO Não tem nenhum teste que usa essa regra
    def p_declarator_paren(self, p):
        """declarator : LPAREN declarator RPAREN"""
        p[0] = p[2]

    def p_declarator_constant_expression(self, p):
        """declarator : declarator LBRACKET constant_expression_opt RBRACKET"""
        arraymod = ArrayDecl(None, p[3], coord=None)
        p[0] = p[1].modify(arraymod)

    def p_declarator_parameter_list(self, p):
        """declarator : declarator LPAREN parameter_list_opt RPAREN"""
        func = FuncDecl(p[3], p[1], coord=None)
        p[0] = func

    def p_constant_expression_opt(self, p):
        """constant_expression_opt : constant_expression
                                   | empty
        """
        p[0] = p[1]

    def p_constant_expression(self, p):
        """constant_expression : binary_expression"""
        p[0] = p[1]
    
    def p_binary_expression(self, p):
        """binary_expression : unary_expression
                             | binary_expression TIMES  binary_expression
                             | binary_expression DIVIDE binary_expression
                             | binary_expression MOD    binary_expression
                             | binary_expression PLUS   binary_expression
                             | binary_expression MINUS  binary_expression
                             | binary_expression LT     binary_expression
                             | binary_expression LE     binary_expression
                             | binary_expression GT     binary_expression
                             | binary_expression GE     binary_expression
                             | binary_expression EQ     binary_expression
                             | binary_expression NE     binary_expression
                             | binary_expression AND    binary_expression
                             | binary_expression OR     binary_expression
        """
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = BinaryOp(p[2], p[1], p[3], coord=p[1].coord)
    
    def p_unary_expression(self, p):
        """unary_expression : postfix_expression
                            | unary_operator unary_expression
        """
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = UnaryOp(p[1], p[2], coord=p[2].coord)

    def p_postfix_expression_primary(self, p):
        """postfix_expression : primary_expression
        """
        p[0] = p[1]

    def p_postfix_expression_array_ref(self, p):
        """postfix_expression : postfix_expression LBRACKET expression RBRACKET
        """
        p[0] = ArrayRef(p[1], p[3], coord=p[1].coord)

    def p_postfix_expression_func_call(self, p):
        """postfix_expression : postfix_expression LPAREN argument_expression_opt RPAREN
        """
        p[0] = FuncCall(p[1], p[3], coord=p[1].coord)

    def p_primary_expression_id(self, p):
        """primary_expression : ID
        """
        p[0] = ID(p[1], coord=self._token_coord(p, 1))

    def p_primary_expression_constant(self, p):
        """primary_expression : constant
        """
        p[0] = p[1]

    def p_primary_expression_string_literal(self, p):
        """primary_expression : STRING_LITERAL
        """
        p[0] = Constant('string', p[1], coord=self._token_coord(p, 1))


    def p_primary_expression_lparen_expression_rparen(self, p):
        """primary_expression : LPAREN expression RPAREN
        """
        p[0] = p[2]


    def p_constant_int(self, p):
        """constant : INT_CONST
        """
        p[0] = Constant('int', p[1], coord=self._token_coord(p, 1))

    def p_constant_char(self, p):
        """constant : CHAR_CONST
        """
        p[0] = Constant('char', p[1], coord=self._token_coord(p, 1))
        

    def p_expression_opt(self, p):
        """expression_opt : expression
                          | empty
        """
        p[0] = p[1]

    def p_expression(self, p):
        """expression  : assignment_expression
                       | expression COMMA assignment_expression
        """
        # single expression
        if len(p) == 2:
            p[0] = p[1]
        else:
            if not isinstance(p[1], ExprList):
                p[1] = ExprList([p[1]], coord=p[1].coord)

            p[1].exprs.append(p[3])
            p[0] = p[1]

    def p_argument_expression_opt(self, p):
        """argument_expression_opt : argument_expression
                                   | empty
        """
        p[0] = p[1]

    def p_argument_expression(self, p):
        """argument_expression : assignment_expression
                               | argument_expression COMMA assignment_expression
        """
        if len(p) == 2:
            p[0] = p[1]
        else:
            if not isinstance(p[1], ExprList):
                p[1] = ExprList([p[1]], coord=p[1].coord)

            p[1].exprs.append(p[3])
            p[0] = p[1]
        

    def p_assignment_expression(self, p):
        """assignment_expression : binary_expression
                                 | unary_expression EQUALS assignment_expression
        """
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = Assignment(p[2], p[1], p[3],p[1].coord)

    def p_unary_operator(self, p):
        """unary_operator   : PLUS
                            | MINUS
                            | NOT
        """
        p[0] = p[1]

    def p_parameter_list_opt(self, p):
        """parameter_list_opt : parameter_list
                              | empty
        """
        p[0] = p[1]

    def p_parameter_list(self, p):
        """parameter_list   : parameter_declaration
                            | parameter_list COMMA parameter_declaration
        """
        if len(p) == 2:
            p[0] = ParamList([p[1]])
        else:
            p[1].params.append(p[3])
            p[0] = p[1]

    def p_parameter_declaration(self, p):
        """parameter_declaration  : type_specifier declarator
        """
        p[2].primitive = p[1]
        p[0] = Decl(p[2].identifier, p[2], None)


    def p_declaration_star(self, p):
        """declaration_star : declaration_star declaration
                            | empty
        """
        if len(p) == 2:
            p[0] = []
        else:
            p[1].extend(p[2])
            p[0] = p[1]

    def p_declaration(self, p):
        """declaration   : type_specifier init_declarator_list_opt SEMI"""
        for declaration in p[2]:
            declaration.primitive = p[1]
        
        p[0] = p[2]

    def p_init_declarator_list_opt(self, p):
        """init_declarator_list_opt : init_declarator_list
                                    | empty
        """
        p[0] = p[1]

    def p_init_declarator_list(self, p):
        """init_declarator_list : init_declarator
                                | init_declarator_list COMMA init_declarator
        """
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[1].append(p[3])
            p[0] = p[1]

    def p_init_declarator(self, p):
        """init_declarator : declarator
                           | declarator EQUALS initializer
        """
        if len(p) == 2:
          p[0] = Decl(p[1].identifier, p[1], None)

        else:
          p[0] = Decl(p[1].identifier, p[1], p[3])


    def p_initializer(self, p):
        """initializer : assignment_expression
                       | LBRACE initializer_list_opt      RBRACE
                       | LBRACE initializer_list    COMMA RBRACE
        """
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = p[2]

    def p_initializer_list_opt(self, p):
        """initializer_list_opt : initializer_list
                                | empty
        """
        p[0] = p[1]

    def p_initializer_list(self, p):
        """initializer_list : initializer
                            | initializer_list COMMA initializer
        """
        if len(p) == 2:
            p[0] = InitList([p[1]], coord=p[1].coord)
        else:
            p[1].exprs.append(p[3])
            p[0] = p[1]
        

    def p_compound_statement(self, p):
        """compound_statement : LBRACE declaration_star statement_star RBRACE"""
        p[0] = Compound(p[2] + p[3], self._token_coord(p, 1))

    def p_statement_star(self, p):
        """statement_star : statement_star statement
                          | empty
        """
        if len(p) == 2:
            p[0] = []
        else:
            p[1].append(p[2])
            p[0] = p[1]

    def p_statement_expression_statement(self, p):
        """statement : expression_statement"""
        p[0] = p[1]
    
    def p_statement_compound_statement(self, p):
        """statement : compound_statement"""
        p[0] = p[1]

    def p_statement_selection_statement(self, p):
        """statement : selection_statement"""
        p[0] = p[1]

    def p_statement_iteration_statement(self, p):
        """statement : iteration_statement"""
        p[0] = p[1]
    
    def p_statement_jump_statement(self, p):
        """statement : jump_statement"""
        p[0] = p[1]

    def p_statement_assert_statement(self, p):
        """statement : assert_statement"""
        p[0] = p[1]

    def p_statement_print_statement(self, p):
        """statement : print_statement"""
        p[0] = p[1]

    def p_statement_read_statement(self, p):
        """statement : read_statement"""
        p[0] = p[1]

    def p_expression_statement(self, p):
        """expression_statement : expression_opt SEMI"""
        p[0] = p[1]

    def p_selection_statement(self, p):
        """selection_statement : IF LPAREN expression RPAREN statement
                               | IF LPAREN expression RPAREN statement ELSE statement
        """
        if len(p) == 6:
            p[0] = If(p[3], p[5], None, coord=self._token_coord(p, 1))
        else:
            p[0] = If(p[3], p[5], p[7], coord=self._token_coord(p, 1))
                                

    def p_jump_statement_break(self, p):
        """jump_statement  : BREAK SEMI
        """
        p[0] = Break(self._token_coord(p, 1))

    def p_jump_statement_return(self, p):
        """jump_statement  : RETURN expression_opt SEMI
        """
        p[0] = Return(p[2], coord=self._token_coord(p, 1))

    def p_iteration_statement(self, p):
        """iteration_statement : WHILE LPAREN expression RPAREN statement
                               | FOR LPAREN expression_opt             SEMI expression_opt SEMI expression_opt RPAREN statement
                               | FOR LPAREN declaration expression_opt SEMI expression_opt RPAREN statement
        """
        if len(p) == 6:
            p[0] = While(p[3], p[5], coord=self._token_coord(p, 1))
        elif len(p) == 10:
            p[0] = For(p[3], p[5], p[7], p[9], coord=self._token_coord(p, 1))
        elif len(p) == 9:
            decl_list = DeclList(p[3], coord=self._token_coord(p, 1))
            p[0] = For(decl_list, p[4], p[6], p[8], coord=self._token_coord(p, 1))

    def p_assert_statement(self, p):
        """assert_statement : ASSERT expression SEMI
        """
        p[0] = Assert(p[2], coord=self._token_coord(p, 1))

    def p_print_statement(self, p):
        """print_statement : PRINT LPAREN expression_opt RPAREN SEMI"""
        p[0] = Print(p[3], coord=self._token_coord(p, 1))

    def p_read_statement(self, p):
        """read_statement : READ LPAREN argument_expression RPAREN SEMI"""
        p[0] = Read(p[3], coord=self._token_coord(p, 1))

    def p_empty(self, p):
        """empty :"""
        pass

    def p_error(self, p):
        if p:
            self._parser_error(
                "Before %s" % p.value, Coord(p.lineno, self.uclex.find_tok_column(p))
            )
        else:
            self._parser_error("At the end of input (%s)" % self.uclex.filename)


if __name__ == "__main__":
    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to file to be parsed", type=str)
    args = parser.parse_args()

    # get input path
    input_file = args.input_file
    input_path = pathlib.Path(input_file)

    # check if file exists
    if not input_path.exists():
        print("ERROR: Input", input_path, "not found", file=sys.stderr)
        sys.exit(1)

    def print_error(msg, x, y):
        print("Lexical error: %s at %d:%d" % (msg, x, y), file=sys.stderr)

    # set error function
    p = UCParser()
    # open file and print ast
    with open(input_path) as f:
        ast = p.parse(f.read())
        ast.show(buf=sys.stdout, showcoord=True)
