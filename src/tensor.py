import networkx as nx
from z3 import (
    Int,
    ArithRef,
    If,
    IntVal,
    is_int_value,
    Solver,
    is_true,
    simplify,
    And,
)

from .constraint import Constraint
from .ops import Add, Op, Broadcast
from .shape import Shape


def _is_one(expr):
    return is_int_value(expr) and expr.as_long() == 1


def _as_z3_expr(expr):
    if isinstance(expr, int):
        return IntVal(expr)
    return expr


class Tensor:
    def __init__(self, id, shape, context):
        self.id: int = id
        self.shape: Shape = shape
        self.context: Context = context

    def __repr__(self):
        return f"Tensor(id={self.id}, shape={self.shape})"

    def __len__(self):
        return len(self.shape)

    # Helper method for consistent dimension access, considering padding
    def padded_dim(self, index: int, max_rank: int) -> ArithRef:
        current_rank = len(self.shape)
        #  the effective index in the padded (right-aligned) shape
        effective_index = index - (max_rank - current_rank)

        if effective_index < 0 or effective_index >= current_rank:
            return IntVal(1)
        else:
            return _as_z3_expr(self.shape.dims[effective_index])

    def needs_broadcast(self, target_dims: Shape, max_rank: int) -> bool:
        if len(self) != max_rank:
            return True
        else:
            for i in range(max_rank):
                dim1 = _as_z3_expr(self.padded_dim(i, max_rank))
                dim2 = _as_z3_expr(target_dims[i])
                if is_true(dim1.eq(dim2)):
                    continue
                if _is_one(dim1):
                    continue

                return True
        return False

    def __add__(self, other):
        assert isinstance(other, Tensor), (
            f"cannot add a tensor with object of type {type(other)}"
        )
        return self.context.add(self, other)


# The Context manages everything.
class Context:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.solver = Solver()
        self.handles: set[Tensor] = set()
        self.constraints: list[Constraint] = []
        self._counter = 0
        self._cons_id = 0

    def new_symbol(self, name: str | int | None = None) -> Int:  # type: ignore
        if isinstance(name, int):
            return IntVal(name)

        if not name:
            name = f"s{self._next_id()}"
        else:
            self._next_id()
        return Int(name)


    def shape_from(
        self,
        symbols: list[str | int] | None,
        num: int | None = None,
    ) -> Shape:
        into_shape = []
        if not symbols:
            symbols = []
        if not num:
            num = 0

        limit = max(len(symbols), num)

        for n in range(limit):
            try:
                into_shape.append(self.new_symbol(symbols[n]))
            except ValueError:
                into_shape.append(self.new_symbol())
        return Shape(into_shape)

    def _next_cons_id(self):
        id = self._cons_id
        self._cons_id += 1
        return id

    def _next_id(self):
        id = self._counter
        self._counter += 1
        return id

    def tensor(self, shape: list | Shape) -> Tensor:
        node_id = self._next_id()
        if isinstance(shape, list):
            t = Tensor(node_id, Shape(shape), self)
        elif isinstance(shape, Shape):
            t = Tensor(node_id, shape, self)
        else:
            raise TypeError("shape should be a `list` or `Shape`")
        self.handles.add(t)
        self.graph.add_node(node_id, op="leaf", tensor_handle=t)
        return t

    def _create_op_node(self, op: Op):
        node_id = self._next_id()
        output_tensor = Tensor(node_id, op.output(), self)
        self.graph.add_node(node_id, op=op, tensor_handle=output_tensor)
        for in_tensor in op.inputs():
            self.graph.add_edge(in_tensor.id, node_id)
        return output_tensor

    def add(self, a: Tensor, b: Tensor) -> Tensor:
        # a, b = self.broadcast(a, b) # not necessary
        output_tensor = self._create_op_node(Add(a, b))
        return output_tensor

    def broadcast(self, a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
        """
        appropriately broadcasts tensors and appends broadcast node
        """
        max_rank = len(a) if len(a) > len(b) else len(b)
        shape = Shape()
        for i in range(max_rank):
            dim_a = a.padded_dim(i, max_rank)
            dim_b = b.padded_dim(i, max_rank)
            if _is_one(dim_a):
                shape.append(dim_b)
            elif _is_one(dim_b):
                shape.append(dim_a)
            elif dim_a.eq(dim_b):
                shape.append(dim_a)
            else:
                shape.append(If(dim_a > dim_b, dim_a, dim_b))

        a_final, b_final = a, b
        if a.needs_broadcast(shape, max_rank):
            a_final = self._create_op_node(Broadcast(a, shape))
        if b.needs_broadcast(shape, max_rank):
            b_final = self._create_op_node(Broadcast(b, shape))

        return a_final, b_final

    def constraint_map(self, index) -> Constraint:
        return self.constraints[int(str(index))]

    def tensor_map(self, id) -> Tensor | None:
        for t in self.handles:
            if t.id == id:
                return t
        return None

    def trace(self):
        for node_id in self.graph.nodes:
            node_data = self.graph.nodes[node_id]
            if node_data["op"] != "leaf":
                op: Op = node_data["op"]
                cons = op.constraint()
                self.constraints.extend(cons)
                for c in cons:
                    c.with_id(self._next_cons_id())
                    expr, name, disp = c.decompose()
                    print(f"[{c.id}]: {c}")
                    self.solver.assert_and_track(expr, str(c.id))

        self.final_constraints = [
            simplify(
                And(s.expr for s in self.constraints),
                som=True,
                blast_distinct=True,
                arith_lhs=True,
                ite_extra_rules=True,
            )
        ]

    def __iter__(self):
        return self.constraints.__iter__()
