from __future__ import annotations

import abc


from .shape import Shape

from .constraint import Constraint
from z3 import Or, IntVal, simplify


class Op(abc.ABC):
    def __init__(self) -> None: ...

    @abc.abstractmethod
    def __str__(self) -> str: ...

    @abc.abstractmethod
    def constraint(self) -> list[Constraint]: ...

    @abc.abstractmethod
    def inputs(self) -> list: ...

    @abc.abstractmethod
    def output(self) -> Shape: ...

    # @abc.abstractmethod
    # def id(self) -> int: ...


class ScalarMul(Op):
    def __init__(self, tensor, tensorized_scalar, actual_scalar) -> None:
        self.tensor = tensor
        self.scalar = tensorized_scalar
        self.disp_scalar = actual_scalar
        pass

    def __str__(self) -> str:
        return f"scalar_mul(tensor={self.tensor.id}, scalar={self.disp_scalar})"

    def inputs(self) -> list:
        return [self.tensor, self.scalar]

    def output(self) -> Shape:
        return self.tensor.shape

    def constraint(self) -> list[Constraint]:
        # no constraints, just multiply the data
        return []


class MatMul(Op):
    def __init__(self, a, b) -> None:
        self.left = a
        self.right = b
        self.constraints: list[Constraint] = []

    def __str__(self) -> str:
        return f"matmul(lhs={self.left.id}, rhs={self.right.id})"

    def inputs(self) -> list:
        return [self.left, self.right]

    def output(self) -> Shape:
        ret = Shape()
        # both should have same rank as broadcasting
        # should be done before this
        max_rank = len(self.left)
        # this assumes all leading dimensions are equal
        # because this is the intended result
        # all dime should be equal
        # except m, n (..., M, K) @ (..., K, N)
        for dim in range(len(self.left) - 1):
            # append all except last
            ret.append(self.left.padded_dim(dim, max_rank))

        return ret

    def constraint(self) -> list[Constraint]:
        # no constraints, just copy the data
        return self.constraints


class Add(Op):
    def __init__(self, lhs, rhs) -> None:
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs
        self.ret_shape = self.lhs.shape
        self.constraints: list[Constraint] = []

    def __str__(self) -> str:
        return f"add(lhs={self.lhs.id}, rhs={self.rhs.id})"

    def output(self) -> Shape:
        return self.ret_shape

    def inputs(self) -> list:
        return [self.lhs, self.rhs]

    def constraint(self) -> list[Constraint]:
        max_rank = max(len(self.lhs), len(self.rhs))

        for i in range(max_rank):
            dim_a = self.lhs.padded_dim(i, max_rank)
            dim_b = self.rhs.padded_dim(i, max_rank)

            con = Constraint(
                # Or(dim_a == dim_b, dim_a == IntVal(1), dim_b == IntVal(1)),
                simplify(dim_a == dim_b),
                "add",
                f"dim_{i}: a={dim_a}, b={dim_b}",
                [self.lhs.id, self.rhs.id],
            )
            self.constraints.append(con)

        return self.constraints


class Broadcast(Op):
    def __init__(self, tensor, target: Shape) -> None:
        super().__init__()
        self.tensor = tensor
        self.target = target
        self.constraints: list[Constraint] = []

    def __str__(self) -> str:
        return (
            f"broadcast(tensor_shape={self.tensor.shape}, target={self.target})"
        )

    def output(self) -> Shape:
        return self.target

    def inputs(self) -> list:
        return [self.tensor]

    def constraint(self) -> list:
        target_rank = len(self.target)

        for i in range(target_rank):
            input_dim = self.tensor.padded_dim(i, target_rank)
            target_dim = self.target.dims[i]
            con = Constraint(
                simplify(Or(input_dim == target_dim, input_dim == IntVal(1))),
                f"broadcast(lhs={self.tensor.shape}, rhs={self.target})",
                f"a: {input_dim}, b: {target_dim}",
                [self.tensor.id],
            )
            self.constraints.append(con)

        return self.constraints
