import argparse
import pathlib
import sys
import pprint
from typing import Dict, List, Set, Tuple
from uc.uc_ast import FuncDef, Node
from uc.uc_block import CFG, EmitBlocks, format_instruction, Block, BasicBlock
from uc.uc_code import CodeGenerator
from uc.uc_interpreter import Interpreter
from uc.uc_parser import UCParser
from uc.uc_sema import NodeVisitor, Visitor

pp = pprint.PrettyPrinter(indent=4)


class DataFlow(NodeVisitor):
    def __init__(self, viewcfg: bool):
        # flag to show the optimized control flow graph
        self.viewcfg: bool = viewcfg
        # list of code instructions after optimizations
        self.code: List[Tuple[str]] = []
        self.code_with_program_point = {}
        self.globals = {}
        self.globals_with_program_point = {}
        self.global_vars = []
        # All program points where a given variable is defined
        self.defs: Dict[str, List[int]] = {}

        # RD gen and RD kill set of every program point
        self.gen_RD: Dict[int, set] = {}
        self.kill_RD: Dict[int, set] = {}
        # RD out and RD in set of every basic block
        self.block_in_RD = {}
        self.block_out_RD = {}
        # RD out and RD in set of every program point
        self.program_point_in_RD = {}
        self.program_point_out_RD = {}
        # For each progam point, keeps track of the definitions that reach each use in the statement
        self.ud_chain: Dict[int, Dict[str, Set[int]]] = {}

        # Compute LV gen and kill for each program point
        self.gen_LV: Dict[int, set] = {}
        self.kill_LV: Dict[int, set] = {}

        self.bin_ops = [
            "add",
            "sub",
            "mul",
            "div",
            "mod",
            "lt",
            "le",
            "gt",
            "ge",
            "eq",
            "ne",
            "and",
            "or",
        ]

    def show(self, buf=sys.stdout):
        _str = ""
        for _code in self.code:
            _str += format_instruction(_code) + "\n"
        buf.write(_str)

    def show_with_points(self, buf=sys.stdout):
        _str = ""
        # TODO Está bugado, só vai imprimir a ultima função que foi definida
        for point, _code in self.code_with_program_point.items():
            _str += str(point) + ":  " + format_instruction(_code) + "\n"
        buf.write(_str)

    def appendOptimizedCode(self, cfg: Block):
        bb = cfg
        while bb:
            for point, inst in bb.code_with_program_point.items():
                if point not in self.dead_code:
                    self.code.append(inst)
            bb = bb.next_block
        # code = list(self.code_with_program_point.items())
        # for idx, (point, inst) in enumerate(code):
        #     if not inst[0].startswith("global"):
        #         break
        # code = code[idx:]
        # for point, inst in code:
        #     if point not in self.dead_code:
        #         self.code.append(inst)

    def discard_unused_allocs(self, cfg: Block):
        bb = cfg
        while bb:
            for point, inst in bb.code_with_program_point.items():
                if inst[0].startswith("alloc"):
                    var = inst[1]
                    if not self.check_if_var_is_used(var, point):
                        self.dead_code.add(point)
            bb = bb.next_block
    
    def check_if_var_is_used(self, var, point):
        for point, inst in self.code_with_program_point.items():
            if inst[0].startswith("load") and inst[1] == var:
                return True
            elif inst[0].startswith("store") and inst[2] == var:
                return True
            elif inst[0].startswith("elem") and inst[1] == var:
                return True
            elif inst[0].startswith("param") and inst[1] == var:
                return True
        return False

    def visit_Program(self, node: Node):
        # First, save the global instructions on code member
        self.code = node.text[:]  # [:] to do a copy
        self.globals = node.text[:]
        for _decl in node.gdecls:
            if isinstance(_decl, FuncDef):
                # start with Reach Definitions Analysis
                self.buildRD_blocks(_decl.cfg)
                self.computeRD_gen_kill(_decl.cfg)

                self.computeRD_in_out(_decl.cfg)
                # # and do constant propagation optimization
                self.compute_ud_chain()
                self.constant_propagation(_decl.cfg)

                do_dead_code = True
                # self.dead_code = set()
                while do_dead_code:
                    self.buildLV_blocks(_decl.cfg)
                    self.computeLV_gen_kill(_decl.cfg)
                    self.computeLV_in_out(_decl.cfg)
                    self.deadcode_elimination()

                    for dead_point in self.dead_code:
                        self.code_with_program_point.pop(dead_point)

                    bb = _decl.cfg
                    while bb:
                        for dead_point in self.dead_code:
                            if dead_point in bb.code_with_program_point:
                                bb.code_with_program_point.pop(dead_point)
                        bb = bb.next_block

                    if self.dead_code == set():
                        break

                # # after that do cfg simplify (optional)
                self.short_circuit_jumps(_decl.cfg)
                # self.merge_blocks(_decl.cfg)
                self.discard_unused_allocs(_decl.cfg)

                # # finally save optimized instructions in self.code
                self.appendOptimizedCode(_decl.cfg)

        if self.viewcfg:
            for _decl in node.gdecls:
                if isinstance(_decl, FuncDef):
                    dot = CFG(_decl.decl.name.name + ".opt")
                    dot.view(_decl.cfg)

    def buildLV_blocks(self, cfg: Block):
        self.gen_LV = {}
        self.kill_LV = {}
        self.global_vars = []

        for glob_inst in self.globals:
            self.global_vars.append(glob_inst[1])

    def computeLV_gen_kill(self, cfg):
        """Compute LV gen and kill for each program point"""

        for point, inst in self.code_with_program_point.items():
            self.calculate_instruction_LV_gen_kill(inst, point)

        bb = cfg
        while bb:
            p_gen = set()
            p_kill = set()
            reversed_code = list(bb.code_with_program_point.items())[::-1]
            for n, inst in reversed_code:
                if n in self.gen_LV and n in self.kill_LV:
                    gen_pn = self.gen_LV[n].union(p_gen.difference(self.kill_LV[n]))
                    kill_pn = self.kill_LV[n].union(p_kill)

                    p_gen = gen_pn
                    p_kill = kill_pn

            bb.gen_LV = p_gen
            bb.kill_LV = p_kill

            bb = bb.next_block

    def calculate_instruction_LV_gen_kill(self, instruction, point):
        # TODO Pensar se precisa considerar define

        for op in self.bin_ops:
            if instruction[0].startswith(op):
                gen = [instruction[1], instruction[2]]
                kill = [instruction[3]]
                self.update_LV_gen_kill(gen, kill, point)
                return

        # TODO Talvez de problema quando é um store do tipo store_type_*
        if instruction[0].startswith("store"):
            if instruction[0][-1] == "*":
                gen = [instruction[1], instruction[2]]
                kill = []
            else:
                gen = [instruction[1]]
                kill = [instruction[2]]
        elif instruction[0].startswith("global"):
            kill = [instruction[1]]
            gen = []
        elif instruction[0].startswith("load"):
            gen = [instruction[1]]
            kill = [instruction[2]]
        elif instruction[0].startswith("literal"):
            gen = []
            kill = [instruction[2]]
        elif instruction[0].startswith("elem"):
            gen = [instruction[1], instruction[2]]
            kill = [instruction[3]]
        elif instruction[0].startswith("not"):
            gen = [instruction[1]]
            kill = [instruction[2]]
        elif instruction[0].startswith("cbranch"):
            gen = [instruction[1]]
            kill = []
        elif instruction[0].startswith("call"):
            gen = self.global_vars
            if len(instruction) == 3:
                kill = [instruction[2]]
            else:
                kill = []
        elif instruction[0].startswith("print"):
            if len(instruction) == 2:
                gen = [instruction[1]]
            else:
                gen = []
            kill = []
        elif instruction[0].startswith("return") and instruction[0] != "return_void":
            gen = [instruction[1]]
            kill = []
        elif instruction[0].startswith("param"):
            gen = [instruction[1]]
            kill = []
        # TODO Acho que falta considerar o read
        else:
            # TODO VERIFICAR SE ESTA CERTO
            gen = []
            kill = []

        self.update_LV_gen_kill(gen, kill, point)

    def update_LV_gen_kill(self, gen, kill, point):
        if point not in self.gen_LV:
            self.gen_LV[point] = set()
        for var in gen:
            self.gen_LV[point].add(var)

        if point not in self.kill_LV:
            self.kill_LV[point] = set()
        for var in kill:
            self.kill_LV[point].add(var)

    def computeLV_in_out(self, cfg: Block):
        self.compute_block_LV_in_out(cfg)
        self.compute_instruction_LV_in_out(cfg)

    def compute_block_LV_in_out(self, cfg: Block):
        self.block_in_LV = {}
        self.block_out_LV = {}
        Changed = set()

        bb = cfg
        while bb:
            self.block_in_LV[bb] = set()
            Changed.add(bb)

            bb = bb.next_block

        while Changed != set():
            bb = Changed.pop()

            if bb.label == "%exit":
                self.block_out_LV[bb] = set(self.global_vars)
            else:
                self.block_out_LV[bb] = set()

            for s in self.get_sucessors(bb):
                self.block_out_LV[bb] = self.block_out_LV[bb].union(self.block_in_LV[s])

            oldin = self.block_in_LV[bb]

            self.block_in_LV[bb] = bb.gen_LV.union(
                self.block_out_LV[bb].difference(bb.kill_LV)
            )

            if oldin != self.block_in_LV[bb]:
                for p in bb.predecessors:
                    Changed.add(p)

    def compute_instruction_LV_in_out(self, cfg: Block):
        self.program_point_in_LV = {}
        self.program_point_out_LV = {}

        bb = cfg
        while bb:
            block_out = self.block_out_LV[bb]

            reversed_code = list(bb.code_with_program_point.items())[::-1]
            for point, inst in reversed_code:
                self.program_point_out_LV[point] = block_out.copy()
                self.program_point_in_LV[point] = self.gen_LV[point].union(
                    self.program_point_out_LV[point] - self.kill_LV[point]
                )

                block_out = self.program_point_in_LV[point]

            bb = bb.next_block

    def deadcode_elimination(self):
        self.dead_code = set()

        for point, inst in self.code_with_program_point.items():
            if inst[0].startswith("load"):
                if inst[2] not in self.program_point_out_LV[point]:
                    self.dead_code.add(point)
            elif (
                inst[0].startswith("store") and inst[0][-1] != "*"
            ):  # TODO pensar se está certo desconsiderar o caso store_type_*
                if inst[2] not in self.program_point_out_LV[point]:
                    self.dead_code.add(point)
            elif inst[0].startswith("literal"):
                if inst[2] not in self.program_point_out_LV[point]:
                    self.dead_code.add(point)
            elif inst[0].startswith("elem"):
                if inst[3] not in self.program_point_out_LV[point]:
                    self.dead_code.add(point)
            # TODO Talvez possa considerar elem e call também

            else:
                for bin_op in self.bin_ops:
                    if inst[0].startswith(bin_op):
                        if inst[3] not in self.program_point_out_LV[point]:
                            self.dead_code.add(point)

    def compute_ud_chain(self):
        self.ud_chain = {}

        for point, inst in self.code_with_program_point.items():
            self.ud_chain[point] = {}
            uses = self.get_uses(inst)

            if uses:
                for use in uses:
                    if use not in self.ud_chain[point]:
                        self.ud_chain[point][use] = set()

                    for def_point in self.defs[use]:
                        # Check if def_point reaches use
                        if def_point in self.program_point_in_RD[point]:
                            self.ud_chain[point][use].add(def_point)

    def get_uses(self, inst):
        # TODO Pensar se define possui algum use
        # TODO Pensar se call poussi algum use

        uses = []
        if inst[0].startswith("load"):
            uses.append(inst[1])
        # TODO Verificar se o caso store_type_* deve ser tratarado separadamente
        elif inst[0].startswith("store"):
            uses.append(inst[1])
        # TODO Verificar se está certo o use to elem
        elif inst[0].startswith("elem"):
            uses.append(inst[1])
            uses.append(inst[2])
        elif inst[0].startswith("not"):
            uses.append(inst[1])
        elif inst[0].startswith("cbranch"):
            uses.append(inst[1])
        else:
            for op in self.bin_ops:
                if inst[0].startswith(op):
                    uses.append(inst[1])
                    uses.append(inst[2])
        return uses

    def get_all_definitions(self, var):
        """
        Get all definitions of a given global variable
        """

        defs = set()
        for point, inst in self.code_with_program_point.items():
            if inst[0].startswith("store") and inst[2] == var:
                defs.add(point)
        for point, inst in self.globals_with_program_point.items():
            if inst[0].startswith("global") and inst[1] == var:
                defs.add(point)
        return defs

    def constant_propagation(self, cfg: Block):
        changed = True
        while changed:
            changed = False
            # propagate constants
            bb = cfg
            while bb:
                for point, inst in bb.code_with_program_point.items():
                    if inst[0].startswith("load"):
                        source = inst[1]
                        target = inst[2]

                        # Get definitions that reach this use
                        if source[0] == "@":
                            # TODO Confirmar se isso está certo
                            # When source is a global variable, we need to consider all possibles definitions
                            # even if it doesn't reach the current program point
                            defs = self.get_all_definitions(source)
                        else:
                            defs = self.ud_chain[point][source]

                        const_value = self.definitions_are_constant(defs)
                        if const_value is not None:
                            # TODO Assume que o valor da constant é int, talvez esteja errado
                            self.code_with_program_point[point] = (
                                "literal_int",
                                const_value,
                                target,
                            )
                            bb.code_with_program_point[point] = (
                                "literal_int",
                                const_value,
                                target,
                            )
                            # TODO Talvez precisa alterar tambem o ud_chain e defs
                            # TODO Além disso talvez seja preciso atualizar o código do bloco
                            changed = True
                bb = bb.next_block

            # constant folding
            bb = cfg
            while bb:
                for point, inst in bb.code_with_program_point.items():
                    if inst[0].startswith("add"):
                        var1 = inst[1]
                        var2 = inst[2]
                        defs1 = self.ud_chain[point][var1]
                        defs2 = self.ud_chain[point][var2]
                        const_value1 = self.definitions_are_constant(defs1)
                        const_value2 = self.definitions_are_constant(defs2)

                        if const_value1 is not None and const_value2 is not None:
                            # TODO Assume que o valor da constant é int, talvez esteja errado
                            self.code_with_program_point[point] = (
                                "literal_int",
                                const_value1 + const_value2,
                                inst[3],
                            )
                            bb.code_with_program_point[point] = (
                                "literal_int",
                                const_value1 + const_value2,
                                inst[3],
                            )
                            changed = True

                        # Check addition with zero
                        # TODO Pensar se devemos fazer essa otimização ou não
                        # TODO Talvez precisa atualizar o ud_chain e defs nesse caso
                        elif const_value1 == 0:
                            self.code_with_program_point[point] = (
                                "load_int",
                                var2,
                                inst[3],
                            )
                            bb.code_with_program_point[point] = (
                                "load_int",
                                var2,
                                inst[3],
                            )
                            changed = True
                        elif const_value2 == 0:
                            self.code_with_program_point[point] = (
                                "load_int",
                                var1,
                                inst[3],
                            )
                            bb.code_with_program_point[point] = (
                                "load_int",
                                var1,
                                inst[3],
                            )
                            changed = True

                    elif inst[0].startswith("sub"):
                        var1 = inst[1]
                        var2 = inst[2]
                        defs1 = self.ud_chain[point][var1]
                        defs2 = self.ud_chain[point][var2]
                        const_value1 = self.definitions_are_constant(defs1)
                        const_value2 = self.definitions_are_constant(defs2)
                        if const_value1 is not None and const_value2 is not None:
                            # TODO Assume que o valor da constant é int, talvez esteja errado
                            self.code_with_program_point[point] = (
                                "literal_int",
                                const_value1 - const_value2,
                                inst[3],
                            )
                            bb.code_with_program_point[point] = (
                                "literal_int",
                                const_value1 - const_value2,
                                inst[3],
                            )
                            changed = True

                    elif inst[0].startswith("mul"):
                        var1 = inst[1]
                        var2 = inst[2]
                        defs1 = self.ud_chain[point][var1]
                        defs2 = self.ud_chain[point][var2]
                        const_value1 = self.definitions_are_constant(defs1)
                        const_value2 = self.definitions_are_constant(defs2)
                        if const_value1 is not None and const_value2 is not None:
                            # TODO Assume que o valor da constant é int, talvez esteja errado
                            self.code_with_program_point[point] = (
                                "literal_int",
                                const_value1 * const_value2,
                                inst[3],
                            )
                            bb.code_with_program_point[point] = (
                                "literal_int",
                                const_value1 * const_value2,
                                inst[3],
                            )
                            changed = True

                    elif inst[0].startswith("div"):
                        var1 = inst[1]
                        var2 = inst[2]
                        defs1 = self.ud_chain[point][var1]
                        defs2 = self.ud_chain[point][var2]
                        const_value1 = self.definitions_are_constant(defs1)
                        const_value2 = self.definitions_are_constant(defs2)
                        if const_value1 is not None and const_value2 is not None:
                            self.code_with_program_point[point] = (
                                "literal_int",
                                const_value1 / const_value2,
                                inst[3],
                            )
                            bb.code_with_program_point[point] = (
                                "literal_int",
                                const_value1 / const_value2,
                                inst[3],
                            )
                            changed = True

                bb = bb.next_block

    def definitions_are_constant(self, defs, point=None):
        """
        Check if all definitions in defs are constants

        Returns: The constant value if all definitions are constants with the same value, None otherwise
        """
        const_value = None
        for def_point in defs:
            def_inst = self.code_with_program_point[def_point]

            if def_inst[0].startswith("store"):
                source = def_inst[1]
                # Check if all definitions of source are constants with the same value
                for source_def_point in self.ud_chain[def_point][source]:
                    source_def_inst = self.code_with_program_point[source_def_point]
                    if source_def_inst[0].startswith("literal"):
                        if const_value is None:
                            const_value = source_def_inst[1]
                        elif const_value != source_def_inst[1]:
                            return None
                    elif (
                        source_def_inst[0].startswith("global")
                        and len(source_def_inst) == 3
                    ):
                        if const_value is None:
                            const_value = source_def_inst[2]
                        elif const_value != source_def_inst[2]:
                            return None
                    else:
                        return None
            elif def_inst[0].startswith("literal"):
                # TODO Na teoria isso nao deve acontecer nunca
                # Pois se a definição é um literal, nao deveria haver um load
                if const_value is None:
                    const_value = def_inst[1]
                elif const_value != def_inst[1]:
                    return None

            elif def_inst[0].startswith("global") and len(def_inst) == 3:
                if const_value is None:
                    const_value = def_inst[2]
                elif const_value != def_inst[2]:
                    return None
            else:
                return None
        return const_value

    def short_circuit_jumps(self, cfg: Block):
        bb = cfg

        while bb:
            if isinstance(bb, BasicBlock):
                if bb.branch:
                    child = bb.branch
                    if len(child.predecessors) == 1 and child.predecessors[0] == bb:
                        # TODO Acredito que não precisa checar se o alvo
                        # do jump do pai é a label do filho, mas pode ser bom checar mesmo assim
                        assert child.label == bb.instructions[-1][1]

                        # Remove jump do pai e label do filho
                        bb_key = list(bb.code_with_program_point.keys())[-1]
                        child_key = list(child.code_with_program_point.keys())[0]
                        (bb.code_with_program_point.pop(bb_key))
                        (child.code_with_program_point.pop(child_key))

                        # TODO Precisa implementar as modificações na cfg
                        # Adiciona as instruções do filho no pai
                        # bb.instructions.extend(child.instructions)

                        # Atualiza os sucessores do pai

                    elif child == bb.next_block:
                        # Remove jump do pai quando o filho é o próximo bloco
                        bb_key = list(bb.code_with_program_point.keys())[-1]
                        (bb.code_with_program_point.pop(bb_key))

                        # TODO Precisa implementar as modificações na cfg

            bb = bb.next_block

    def computeRD_in_out(self, cfg: Block):
        self.compute_block_RD_in_out(cfg)

        self.compute_instruction_RD_in_out(cfg)

    def compute_instruction_RD_in_out(self, cfg: Block):
        self.program_point_in_RD = {}
        self.program_point_out_RD = {}

        bb = cfg
        while bb:
            block_in = self.block_in_RD[bb]

            for point, inst in bb.code_with_program_point.items():
                self.program_point_in_RD[point] = block_in.copy()
                self.program_point_out_RD[point] = self.gen_RD[point].union(
                    self.program_point_in_RD[point] - self.kill_RD[point]
                )

                block_in = self.program_point_out_RD[point]

            bb = bb.next_block

    def compute_globals_RD_out(self, cfg: Block):
        self.globals_RD_out = set()

        if len(self.globals_with_program_point) == 0:
            return
        point, inst = list(self.globals_with_program_point.items())[0]
        prev_out = self.gen_RD[point]

        for point, inst in list(self.globals_with_program_point.items())[1:]:
            in_rd = prev_out
            out_rd = self.gen_RD[point].union(in_rd.difference(self.kill_RD[point]))
            prev_out = out_rd
        self.globals_RD_out = prev_out

    def compute_block_RD_in_out(self, cfg: Block):
        self.block_out_RD = {}
        self.block_in_RD = {}
        Changed = set()

        self.compute_globals_RD_out(cfg)

        bb = cfg
        while bb:
            self.block_out_RD[bb] = set()
            Changed.add(bb)
            bb = bb.next_block

        while Changed != set():
            bb = Changed.pop()

            if bb.label == "%entry":
                self.block_in_RD[bb] = self.globals_RD_out
            else:
                self.block_in_RD[bb] = set()

            for p in bb.predecessors:
                self.block_in_RD[bb] = self.block_in_RD[bb].union(self.block_out_RD[p])

            oldout = self.block_out_RD[bb]

            self.block_out_RD[bb] = bb.gen_RD.union(
                self.block_in_RD[bb].difference(bb.kill_RD)
            )

            if oldout != self.block_out_RD[bb]:
                for s in self.get_sucessors(bb):
                    Changed.add(s)

    def get_sucessors(self, cfg: Block):
        if isinstance(cfg, BasicBlock):
            if cfg.branch:
                return [cfg.branch]
            else:
                return []
        elif cfg.fall_through is not None and cfg.taken is not None:
            return [cfg.fall_through, cfg.taken]
        elif cfg.fall_through is not None:
            return [cfg.fall_through]
        elif cfg.taken is not None:
            return [cfg.taken]
        else:
            return []

    def computeRD_gen_kill(self, cfg: Block):
        self.kill_RD = {}
        self.gen_RD = {}

        for point, inst in self.code_with_program_point.items():
            self.calculate_instruction_RD_gen_kill(inst, point)

        bb = cfg
        while bb:
            p_gen = set()
            p_kill = set()
            for n, inst in bb.code_with_program_point.items():
                if n in self.gen_RD and n in self.kill_RD:
                    gen_pn = self.gen_RD[n].union(p_gen.difference(self.kill_RD[n]))
                    kill_pn = self.kill_RD[n].union(p_kill)

                    p_gen = gen_pn
                    p_kill = kill_pn

            bb.gen_RD = p_gen
            bb.kill_RD = p_kill

            bb = bb.next_block

    def calculate_instruction_RD_gen_kill(self, instruction, point):
        """
        Computes the definitions of a given instruction and updates the defs dictionary
        """
        # TODO Pensar se precisa considerar alloc e global
        # TODO Pensar se precisa considerar param_type

        for op in self.bin_ops:
            if instruction[0].startswith(op):
                var = instruction[3]
                self.update_RD_gen_kill(var, point)
                return

        # TODO Talvez de problema quando é um store do tipo store_type_*
        if instruction[0].startswith("store"):
            var = instruction[2]
            self.update_RD_gen_kill(var, point)
        elif instruction[0].startswith("global"):
            var = instruction[1]
            self.update_RD_gen_kill(var, point)
        elif instruction[0].startswith("load"):
            var = instruction[2]
            self.update_RD_gen_kill(var, point)
        elif instruction[0].startswith("literal"):
            var = instruction[2]
            self.update_RD_gen_kill(var, point)
        elif instruction[0].startswith("elem"):
            var = instruction[3]
            self.update_RD_gen_kill(var, point)
        elif instruction[0].startswith("not"):
            var = instruction[2]
            self.update_RD_gen_kill(var, point)
        elif instruction[0].startswith("define"):
            has_arg = False
            for arg in instruction[2]:
                self.update_RD_gen_kill(arg[1], point)
                has_arg = True

            if not has_arg:
                self.gen_RD[point] = set()
                self.kill_RD[point] = set()

        elif instruction[0].startswith("call") and len(instruction) == 3:
            var = instruction[2]
            self.update_RD_gen_kill(var, point)

        # Outras instrucoes tem gen e kill vazios
        else:
            self.gen_RD[point] = set()
            self.kill_RD[point] = set()

    def update_RD_gen_kill(self, var, point):
        if point not in self.gen_RD:
            self.gen_RD[point] = set()
        self.gen_RD[point].add(point)

        if point not in self.kill_RD:
            self.kill_RD[point] = self.defs[var].copy()
        # print(point, self.kill_RD[point])
        self.kill_RD[point].discard(point)

    def buildRD_blocks(self, cfg: Block):
        # Mantém todas as posições em que uma variável é definida
        self.defs: Dict[str, List[int]] = {}
        self.code_with_program_point = {}

        block = cfg
        point = 0

        for instruction in self.globals:
            self.code_with_program_point[point] = instruction
            self.globals_with_program_point[point] = instruction
            self.process_instruction_defs(instruction, point)
            point = point + 1

        while block:
            for instruction in block.instructions:
                self.code_with_program_point[point] = instruction
                block.code_with_program_point[point] = instruction
                self.process_instruction_defs(instruction, point)
                point = point + 1

            block = block.next_block

    def process_instruction_defs(self, instruction, point):
        """
        Computes the definitions of a given instruction and updates the defs dictionary
        """
        # TODO Pensar se precisa considerar alloc e global
        # TODO Pensar se precisa considerar param_type

        # TODO Talvez de problema quando é um store do tipo store_type_*
        if instruction[0].startswith("store"):
            var = instruction[2]
            self.update_defs(var, point)
        elif instruction[0].startswith("alloc"):
            var = instruction[1]
            self.update_defs(var, point)
        elif instruction[0].startswith("global"):
            var = instruction[1]
            self.update_defs(var, point)
        elif instruction[0].startswith("load"):
            var = instruction[2]
            self.update_defs(var, point)
        elif instruction[0].startswith("literal"):
            var = instruction[2]
            self.update_defs(var, point)
        elif instruction[0].startswith("elem"):
            var = instruction[3]
            self.update_defs(var, point)
        elif instruction[0].startswith("not"):
            var = instruction[2]
            self.update_defs(var, point)
        elif instruction[0].startswith("define"):
            for arg in instruction[2]:
                self.update_defs(arg[1], point)
        elif instruction[0].startswith("call") and len(instruction) == 3:
            var = instruction[2]
            self.update_defs(var, point)
        else:
            for op in self.bin_ops:
                if instruction[0].startswith(op):
                    var = instruction[3]
                    self.update_defs(var, point)

    def update_defs(self, var, point):
        if var not in self.defs:
            self.defs[var] = set()
        self.defs[var].add(point)


if __name__ == "__main__":
    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        help="Path to file to be used to generate uCIR. By default, this script runs the interpreter on the optimized uCIR \
              and shows the speedup obtained from comparing original uCIR with its optimized version.",
        type=str,
    )
    parser.add_argument(
        "--ir-with-points",
        help="Print uCIR with program points generated from input_file.",
        action="store_true",
    )
    parser.add_argument(
        "--opt",
        help="Print optimized uCIR generated from input_file.",
        action="store_true",
    )
    parser.add_argument(
        "--speedup",
        help="Show speedup from comparing original uCIR with its optimized version.",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--debug", help="Run interpreter in debug mode.", action="store_true"
    )
    parser.add_argument(
        "-c",
        "--cfg",
        help="show the CFG of the optimized uCIR for each function in pdf format",
        action="store_true",
    )
    args = parser.parse_args()

    speedup = args.speedup
    print_opt_ir = args.opt
    print_ir = args.ir_with_points
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

    gen = CodeGenerator(False)
    gen.visit(ast)
    gencode = gen.code

    opt = DataFlow(create_cfg)
    opt.visit(ast)
    optcode = opt.code
    if print_opt_ir:
        print("Optimized uCIR: --------")
        opt.show()
        print("------------------------\n")

    if print_ir:
        print("uCIR with program points: --------")
        opt.show_with_points()
        print("------------------------\n")

    speedup = len(gencode) / len(optcode)
    sys.stderr.write(
        "[SPEEDUP] Default: %d Optimized: %d Speedup: %.2f\n\n"
        % (len(gencode), len(optcode), speedup)
    )

    vm = Interpreter(interpreter_debug)
    vm.run(optcode)
