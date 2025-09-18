from z3 import simplify


class Constraint:
    def __init__(
        self,
        expr,
        name: str | None = None,
        pretty: str | None = None,
        t_id: list | None = None,
    ) -> None:
        self.created_by = t_id or []
        self._expr = simplify(expr)
        self.name = name or str(expr)
        self.disp = pretty or str(expr)
        self.id = 0

    @property
    def expr(self):
        return self._expr

    def __str__(self) -> str:
        return f"{self.name}: {self.disp}"

    def decompose(self) -> tuple:
        return self.expr, self.name, self.disp

    def with_id(self, id: int):
        self.id = id
