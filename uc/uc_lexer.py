import argparse
import pathlib
import sys
import ply.lex as lex


class UCLexer:
    """A lexer for the uC language. After building it, set the
    input text with input(), and call token() to get new
    tokens.
    """

    def __init__(self, error_func):
        """Create a new Lexer.
        An error function. Will be called with an error
        message, line and column as arguments, in case of
        an error during lexing.
        """
        self.error_func = error_func
        self.filename = ""

        # Keeps track of the last token returned from self.token()
        self.last_token = None

    def build(self, **kwargs):
        """Builds the lexer from the specification. Must be
        called after the lexer object is created.

        This method exists separately, because the PLY
        manual warns against calling lex.lex inside __init__
        """
        self.lexer = lex.lex(object=self, **kwargs)

    def reset_lineno(self):
        """Resets the internal line number counter of the lexer."""
        self.lexer.lineno = 1

    def input(self, text):
        self.lexer.input(text)

    def token(self):
        self.last_token = self.lexer.token()
        return self.last_token

    def find_tok_column(self, token):
        """Find the column of the token in its line."""
        last_cr = self.lexer.lexdata.rfind("\n", 0, token.lexpos)
        return token.lexpos - last_cr

    # Internal auxiliary methods
    def _error(self, msg, token):
        location = self._make_tok_location(token)
        self.error_func(msg, location[0], location[1])
        self.lexer.skip(1)

    def _make_tok_location(self, token):
        return (token.lineno, self.find_tok_column(token))

    # Reserved keywords
    keywords = (
        "ASSERT",
        "BREAK",
        "CHAR",
        "ELSE",
        "FOR",
        "IF",
        "INT",
        "PRINT",
        "READ",
        "RETURN",
        "VOID",
        "WHILE",
    )

    keyword_map = {}
    for keyword in keywords:
        keyword_map[keyword.lower()] = keyword

    #
    # All the tokens recognized by the lexer
    #
    tokens = keywords + (
        # Identifiers
        "ID",
        # constants
        "INT_CONST",
        "CHAR_CONST",
        "STRING_LITERAL",
        # Operators
        "PLUS",
        "MINUS",
        "TIMES",
        "DIVIDE",
        "MOD",
        "OR",
        "AND",
        "NOT",
        "LT",
        "LE",
        "GT",
        "GE",
        "EQ",
        "NE",
        # Assignment
        "EQUALS",
        # Delimeters
        "LPAREN",
        "RPAREN",  # ( )
        "LBRACKET",
        "RBRACKET",  # [ ]
        "LBRACE",
        "RBRACE",  # { }
        "COMMA",
        "SEMI",  # , ;
    )

    #
    # Rules
    #
    t_ignore = " \t"

    # Newlines
    def t_NEWLINE(self, t):
        r"\n+"
        t.lexer.lineno += t.value.count("\n")

    def t_ID(self, t):
        r"[a-zA-Z_][a-zA-Z_0-9]*"
        t.type = self.keyword_map.get(t.value, "ID")  # Check for reserved words
        return t

    def t_ccomment(self, t):
        r"/\*([^*]|\*+[^\/])*\*/"
        t.lexer.lineno += t.value.count("\n")

    t_ignore_cppcomment = r"//.*"  

    def t_uccomment(self, t):
        r"/\*([^*]|\*+[^\/])*"
        self._error("Unterminated comment", t)

    def t_INT_CONST(self, t):
        r"([1-9]\d*|0)"
        return t

    def t_CHAR_CONST(self, t):
        r"'.'"
        return t

    def t_STRING_LITERAL(self, t):
        r'"([^"\\\n]|\\[^\n])*"'
        t.value = t.value[1:-1]
        return t

    def t_unterminated_string(self, t):
        r'"(\\"|[^"])*'
        self._error("Unterminated string", t)

    t_PLUS = r"\+"
    t_MINUS = r"-"
    t_TIMES = r"\*"
    t_DIVIDE = r"/"
    t_MOD = r"%"
    t_OR = r"\|\|"
    t_AND = r"&&"
    t_NOT = r"!"
    t_LT = r"<"
    t_LE = r"<="
    t_GT = r">"
    t_GE = r">="
    t_EQ = r"=="
    t_NE = r"!="
    t_EQUALS = r"="
    t_LPAREN = r"\("
    t_RPAREN = r"\)"
    t_LBRACKET = r"\["
    t_RBRACKET = r"]"
    t_LBRACE = r"{"
    t_RBRACE = r"}"
    t_COMMA = r","
    t_SEMI = r";"

    def t_error(self, t):
        msg = "Illegal character %s" % repr(t.value[0])
        self._error(msg, t)

    # Scanner (used only for test)
    def scan(self, data):
        self.lexer.input(data)
        output = ""
        while True:
            tok = self.lexer.token()
            if not tok:
                break
            print(tok)
            output += str(tok) + "\n"
        return output


if __name__ == "__main__":
    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to file to be scanned", type=str)
    args = parser.parse_args()

    # get input path
    input_file = args.input_file
    input_path = pathlib.Path(input_file)

    # check if file exists
    if not input_path.exists():
        print("Input", input_path, "not found", file=sys.stderr)
        sys.exit(1)

    def print_error(msg, x, y):
        # use stdout to match with the output in the .out test files
        print("Lexical error: %s at %d:%d" % (msg, x, y), file=sys.stdout)

    # set error function
    m = UCLexer(print_error)
    # Build the lexer
    m.build()
    # open file and print tokens
    with open(input_path) as f:
        m.scan(f.read())
