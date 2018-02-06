from typing import Tuple, Sequence, Generator, Mapping, Callable, Optional

from cytoolz import memoize, partial, concat, pipe, map
from numpy import fromiter, ndarray
from pandas import DataFrame


def hightag(*args, **kwargs):
    global hightag
    from dbpy import read_hightagnumber
    hightag = memoize(read_hightagnumber)
    return hightag(*args, **kwargs)


def taglist(*args, **kwargs):
    global taglist
    from dbpy import read_taglist_byrun
    taglist = memoize(read_taglist_byrun)
    return taglist(*args, **kwargs)


def read_syncdatalist_float(*args, **kwargs):
    global read_syncdatalist_float
    from dbpy import read_syncdatalist_float
    return read_syncdatalist_float(*args, **kwargs)


def tags_at(run: int, *other_runs: int, beamline: int = None) -> Tuple[int, Sequence[int]]:
    """
    Example:
        hightag, tags = tags_at(509700, beamline=3)  # from single run
        hightag, tags = tags_at(509700, 509701, 509702, beamline=3)  # from multiple runs
    """
    if beamline is None:
        raise ValueError("Keyword argument 'beamline' must be given!")
    runs = run, *other_runs
    hightag_at_the_beamline = partial(hightag, beamline)
    taglist_at_the_beamline = partial(taglist, beamline)
    hightags: ndarray = pipe(runs, partial(map, hightag_at_the_beamline), partial(fromiter, dtype='int'))
    if not (hightags == hightags[0]).all():
        raise ValueError('Not all the runs have a single hightag!')
    tags = pipe(runs, partial(map, taglist_at_the_beamline), concat, tuple)
    return hightags[0], tags


def scalars_at(run_or_tag: int, *other_runs_or_tags: int, beamline: int = None, hightag: int = None,
               equips: Mapping[str, Tuple[str, Callable]]) -> DataFrame:
    """
    Example:
        equips = {
            'fel_status': ('xfel_mon_bpm_bl3_0_3_beamstatus/summary', bool),
            'fel_shutter': ('xfel_bl_3_shutter_1_open_valid/status', bool)
        }
        df = scalars_at(602345, beamline=3, equips=equips)  # from single run
        df = scalars_at(602345, 602346, 602347, beamline=3, equips=equips)  # from multiple runs
        df = scalars_at(121379273, 121379275, 121379277, hightag=201701, equips=equips)  # from tags
    """
    if beamline is None and hightag is None:
        raise ValueError("Keyword argument 'beamline' or 'hightag' must be given!")
    if hightag is not None:
        if beamline is not None:
            print("Keyword argument 'beamline' will be ignored!")
        tags = run_or_tag, *other_runs_or_tags
    else:
        runs = run_or_tag, *other_runs_or_tags
        hightag, tags = tags_at(*runs, beamline=beamline)
    scalars = {k: [tp(v) for v in read_syncdatalist_float(equip, hightag, tags)] for k, (equip, tp) in equips.items()}
    return DataFrame(scalars, index=tags)


StorageReader: Optional[Callable] = None
StorageBuffer: Optional[Callable] = None
APIError: Optional[Callable] = None


class __ArrReader:
    """
    Example:
        with ArrReader(509700, 509701, 509702, beamline=3, equip='MPCCD-8-2-002-1') as r:
            for d in r:
                print(d['ch0_data'])
                break
    """

    def __init__(self, run: int, *other_runs: int, beamline: int = None, equip: str = None):
        if (equip is None) or (beamline is None):
            raise ValueError("Keyword argument 'equip' and 'beamline' must be given!")
        self.__equip = equip
        self.__beamline = beamline
        self.__runs = run, *other_runs
        self.__hightag, self.__tags = tags_at(*self.__runs, beamline=self.__beamline)
        self.__reader: Optional[StorageReader] = None

    def __enter__(self):
        self.__reader = StorageReader(self.__equip, self.__beamline, self.__runs)
        return self

    def __iter__(self) -> Generator[Optional[dict], None, None]:
        collect = self.__reader.collect
        buffer = StorageBuffer(self.__reader)
        read_det_num_index = buffer.read_det_num_index
        read_det_data = buffer.read_det_data
        read_det_info = buffer.read_det_info
        for tag in self.tags:
            try:
                collect(buffer, tag)
                n = read_det_num_index()
                yield {**{'ch{}_data'.format(i): read_det_data(i) for i in range(n)},
                       **{'ch{}_info'.format(i): read_det_info(i) for i in range(n)}}
            except APIError:
                print("Fail to read data at tag '{}'".format(tag))
                yield None
        del buffer

    def __exit__(self, *args):
        del self.__reader
        self.__reader: Optional[StorageReader] = None

    @property
    def hightag(self):
        return self.__hightag

    @property
    def tags(self):
        return self.__tags


def ArrReader(*args, **kwargs):
    global ArrReader, StorageReader, StorageBuffer, APIError
    from stpy import StorageReader, StorageBuffer, APIError
    ArrReader = __ArrReader
    return ArrReader(*args, **kwargs)
