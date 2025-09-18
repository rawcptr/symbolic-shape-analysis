from z3 import sat
from src.tensor import Context, Tensor


def main():
    # --- Example Usage ---
    ctx = Context()

    # Create symbolic dimensions
    N = ctx.new_symbol("N")
    C = ctx.new_symbol("C")
    H = ctx.new_symbol("H")
    W = ctx.new_symbol("W")
    bt32 = ctx.shape_from(["B", "T", 32])

    # Create tensors with symbolic shapes
    t1 = ctx.tensor(shape=[N, C, 24, W, 64])
    t2 = ctx.tensor(shape=[N, C, H, W])
    t3 = ctx.tensor(shape=bt32)  # Mismatched shape

    # Build the computation graph
    t_out1 = t1 + t2
    t_out2 = t1 + t3

    ctx.trace()

    solver = ctx.solver
    print("simplified:", ctx.final_constraints)
    print(f"\nSolver check: {solver.check()}")
    if solver.check() == sat:
        print(f"Model: {solver.model()}")
    else:
        core = solver.unsat_core()
        for c in core:
            con = ctx.constraint_map(c)
            # inps = "\n".join(
            #     ["\t" + str(ctx.tensor_map(c).shape) for c in con.created_by]
            # )

            # print(f"constraint failed: {con}\nby inputs:\n{inps}")
            inputs: list[Tensor] = list(
                filter(None, [ctx.tensor_map(c) for c in con.created_by])
            )
            axis = 0
            for i in inputs:
                i.shape

            # print(
            #     f"operation failed: {con}"
            #     f"reason: incompatible dimension at axis {} {}"
            # )


if __name__ == "__main__":
    main()
