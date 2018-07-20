from struct import Struct
from typing import Generator

__all__ = ['hit_reader', 'bin_reader']


def hit_reader(filename) -> Generator[dict, None, None]:
    """
    Example:
        for d in hit_reader('aq137.hit'):
            print(d)
            break
    """
    deep1 = Struct('=IH')
    unpack1 = deep1.unpack
    size1 = deep1.size
    deep2 = Struct('=dddH')
    unpack2 = deep2.unpack
    size2 = deep2.size

    with open(filename, 'br') as f:
        read = f.read
        seek = f.seek
        while read(1):
            seek(-1, 1)
            tag, nhits = unpack1(read(size1))
            yield {
                'tag': tag,
                'nhits': nhits,
                'hits': [dict(zip(('x', 'y', 't', 'method'), unpack2(read(size2)))) for _ in range(nhits)]
            }


def bin_reader(filename, keys=None) -> Generator[dict, None, None]:
    """
    Example:
        for d in bin_reader('aq137.bin'):
            print(d)
            break
    """
    if keys is None:
        keys = tuple('meta{}'.format(i) for i in range(8))
    deep1 = Struct('=IBBBBddddI')
    unpack1 = deep1.unpack
    size1 = deep1.size
    deep2 = Struct('=ddd')
    unpack2 = deep2.unpack
    size2 = deep2.size

    with open(filename, 'br') as f:
        read = f.read
        seek = f.seek
        while read(1):
            seek(-1, 1)
            tag, *meta, nhits = unpack1(read(size1))
            yield {
                'tag': tag,
                **dict(zip(keys, meta)),
                'nhits': nhits,
                'hits': [dict(zip(('t', 'x', 'y'), unpack2(read(size2)))) for _ in range(nhits)]
            }
