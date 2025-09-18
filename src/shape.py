class Shape:
    def __init__(self, dims: list | None = None):
        if not dims:
            dims = []
        self.dims: list = dims

    def __repr__(self):
        return f"Shape({self.dims})"

    def append(self, i):
        self.dims.append(i)

    def __getitem__(self, idx):
        return self.dims[idx]

    def __setitem__(self, idx):
        return self.dims[idx]

    def __len__(self):
        return len(self.dims)

    def __iter__(self):
        return self.dims.__iter__()
