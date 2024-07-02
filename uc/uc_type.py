class uCType:
    """
    Class that represents a type in the uC language.  Basic
    Types are declared as singleton instances of this type.
    """

    def __init__(
        self, name, binary_ops=set(), unary_ops=set(), rel_ops=set(), assign_ops=set()
    ):
        """
        You must implement yourself and figure out what to store.
        """
        self.typename = name
        self.unary_ops = unary_ops
        self.binary_ops = binary_ops
        self.rel_ops = rel_ops
        self.assign_ops = assign_ops


# Create specific instances of basic types. You will need to add
# appropriate arguments depending on your definition of uCType
IntType = uCType(
    "int",
    unary_ops={"-", "+"},
    binary_ops={"+", "-", "*", "/", "%"},
    rel_ops={"==", "!=", "<", ">", "<=", ">="},
    assign_ops={"="},
)

CharType = uCType(
    "char",
    rel_ops={"==", "!=", "&&", "||"}, # TODO Verificar se "&&", "||" estão corretos
    assign_ops={"="},
)

# TODO É só isso mesmo para o void?
VoidType = uCType(
    "void",
    #assign_ops={"="}, TODO Verificar se precisa ou nao dessa op
)

BoolType = uCType(
    "bool",
    rel_ops={"==", "!=", "&&", "||"},
    assign_ops={"="},
    unary_ops={"!"},
)

# TODO precisa adicionar um campo com o tamanho da string?
StringType = uCType(
    "string",
    rel_ops={"==", "!="},
    binary_ops={"+"},
)



# TODO: add array and function types
# Array and Function types need to be instantiated for each declaration
class ArrayType(uCType):
    def __init__(self, typename, element_type, size=None):
        """
        type: Any of the uCTypes can be used as the array's type. This
              means that there's support for nested types, like matrices.
        size: Integer with the length of the array.
        """
        self.type = element_type
        self.dim = size
        # TODO Verificar se None está certo e se assign_ops está certo
        super().__init__(typename, rel_ops={"==", "!="}, assign_ops={"="})

# TODO Escrever docstring do FuncType
class FuncType(uCType):
    def __init__(self, return_type, params):
        """
        return_type: 
        params:
        """
        self.return_type = return_type
        self.params = params
        # TODO Verificar se None está certo e se assign_ops está certo
        super().__init__(None)
