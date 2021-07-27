from typing import Tuple, Literal, Iterator

Lit = Literal['a', 'b', 'c']
Lita = Literal['a']
Litb = Literal['b']
Litc = Literal['c']


class C(Tuple[Lita, Litb, Litc]):

    def __iter__(self) -> Iterator[Lit]:
        for x in self:
            yield x

    def __next__(self) -> Lit:
        return next(self)


c: Tuple[Lita, Litb, Litc] = ('a', 'b', 'c')

for x in c:
    print(x)
    # reveal_type(x)

d = C('a', 'b', 'c')

for x in d:
    print(x)
    # reveal_type(x)

a: str = 2
