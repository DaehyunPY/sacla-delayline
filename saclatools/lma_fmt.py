from abc import ABC, abstractmethod
from operator import add
from os.path import getsize
from struct import Struct
from typing import (Iterator, Any, BinaryIO, Mapping, Sequence, Callable,
                    Iterable)

from cytoolz import first, pipe, partial, reduce
from numpy import fromfile, append, nan, repeat, nan_to_num, ndarray


def read_bin(struct: Struct, buffer: BinaryIO) -> Iterator[Any]:
    s = struct.size
    r = buffer.read(s)
    yield from struct.unpack(r)


class Deserializer(ABC):
    @abstractmethod
    def __init__(self, buffer: BinaryIO) -> None:
        pass

    @abstractmethod
    def header_size(self) -> int:
        pass

    @abstractmethod
    def __call__(self, buffer: BinaryIO) -> Mapping[str, Any]:
        pass


class LmaDeserializer(Deserializer):
    """
    Deserializer for LMA format
    Data format:
        1. Header part

            general       channel
            ––––––– + n * –––––––

            n is num of used channels

            - general: length = 12
                - (int32) + 4 = header size
                - (int16) = num of channels
                - (int16) = num of bytes
                - (float64) = sample interval
                - (int32) = num of samples
                - (float64)
                - (int16)
                - (float64)
                - (int16)
                - (uint32) = used channels. If the value is 13, it is 0x1101 in binary, then only first, third, and
                             fourth channels are used
                - (uint32)
                - (int16)
            - channel: length = 7
                - (int16) = full scale
                - (int16) = offset
                - (float64) = gain
                - (int16) = baseline
                - (int16)
                - (int32)
                - (int32)

        2. Event part

            event         num of partial pulses         info of a partial pulse   a partial pulse
            ––––– + n * ( ––––––––––––––––––––– + m * ( ––––––––––––––––––––––– + ––––––––––––––– ) )

            n is num of used channels
            m is num of partial pulses

            - event: length = 2
                - (int32) = tag
                - (float64) = horpos
            - num of partial pulses: length = 1
                - (int16) num of partial pulses
            - info of a partial pulse: length = 2
                - (int32) = first index of a partial pulse
                - (int32) = length of a partial pulse
            - a partial pulse: length = length of a partial pulse
                               It is equal with full pulse[first index:first index+length]. Final wave is
                               gain*(full pulse - baseline).
                - m * (int16)

    Example:
        from os.path import getsize

        last = getsize(filename)
        with open(filename, 'rb') as buffer:
            deserialize = LmaDeserializer(buffer)
            while buffer.tell() < last:
                deserialized = deserialize(buffer)
    """
    def __init__(self, buffer: BinaryIO) -> None:
        """
        Read header from buffer
        """
        super().__init__(buffer)
        self.__header = tuple(read_bin(Struct("=ihhdidhdhIIh"), buffer))
        channels = (
            tuple(read_bin(Struct("=hhdhhii"), buffer))
            if ((self.used_channels >> i) & 0b1) == 1 else None
            for i in range(self.nchannels))
        self.__channels = tuple(
            {'fullscale': ch[0], 'offset': ch[1], 'gain': ch[2], 'baseline': ch[3]}
            if ch is not None else None
            for ch in channels)
        self.__read_event: Callable[[BinaryIO], Mapping[str, Any]] = partial(read_bin, Struct("=id"))
        self.__read_npulses: Callable[[BinaryIO], Mapping[str, Any]] = partial(read_bin, Struct("=h"))
        self.__read_pulseinfo: Callable[[BinaryIO], Mapping[str, Any]] = partial(read_bin, Struct("=ii"))
        self.__read_pulse: Callable[[BinaryIO], Mapping[str, Any]] = partial(fromfile, dtype='int16')

    @property
    def header_size(self) -> int:
        return self.__header[0] + 4

    @property
    def nchannels(self) -> int:
        return self.__header[1]

    @property
    def nbytes(self) -> int:
        return self.__header[2]

    @property
    def sample_interval(self) -> float:
        return self.__header[3]

    @property
    def nsamples(self) -> int:
        return self.__header[4]

    @property
    def used_channels(self) -> int:
        return self.__header[9]

    @property
    def channels(self) -> Sequence[Mapping[str, Any]]:
        return self.__channels

    def __read_pulses(self, buffer: BinaryIO) -> Iterator[ndarray]:
        n = self.nsamples
        nans: Callable[[int], Sequence[nan]] = partial(repeat, nan)
        for _ in range(first(self.__read_npulses(buffer))):
            n0, n1 = self.__read_pulseinfo(buffer)
            d = self.__read_pulse(buffer, count=n1)
            yield reduce(append, (nans(n0), d, nans(n-n0-n1)))

    def __call__(self, buffer: BinaryIO) -> Mapping[str, Any]:
        """
        Deserialize LMA file
        :param buffer: binary buffer
        :return: dict
        """
        event = tuple(self.__read_event(buffer))
        return {'tag': event[0],
                # 'horpos': event[1],
                'channels': tuple(
                    pipe(self.__read_pulses(buffer),
                         partial(map, lambda pulse: ch['gain'] * (pulse - ch['baseline'])),
                         partial(map, nan_to_num),
                         partial(reduce, add))
                    if ch is not None else None for ch in self.channels)}


class ReadWith:
    """
    Binary reader
    Example:
        LmaReader = partial(ReadWith, LmaDeserializer)
        with LmaReader(filename) as r:
            for d in r:
                print(d)
                break
    """
    def __init__(self, deserializer: Callable[[BinaryIO], Deserializer], filename: str) -> None:
        self.__size = getsize(filename)
        self.__file: BinaryIO = open(filename, 'br')
        self.__current_bit = 0
        self.__deserializer = deserializer(self.__file)

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        self.__file.close()

    @property
    def deserializer(self) -> Deserializer:
        return self.__deserializer

    @property
    def first_bit(self) -> int:
        ret: int = self.deserializer.header_size
        return ret

    @property
    def last_bit(self) -> int:
        return self.__size

    @property
    def __current_bit(self) -> int:
        return self.__file.tell()

    @__current_bit.setter
    def __current_bit(self, i: int) -> None:
        self.__file.seek(i)

    @property
    def current_bit(self) -> int:
        return self.__current_bit

    def __iter__(self) -> Iterable[Mapping[str, Any]]:
        self.__current_bit = self.first_bit
        return self

    def __next__(self) -> Mapping[str, Any]:
        if not (self.current_bit < self.last_bit):
            raise StopIteration
        ret: Mapping[str, Any] = self.deserializer(self.__file)
        return ret


LmaReader: Callable[[str], ReadWith] = partial(ReadWith, LmaDeserializer)
# HitReader: Callable[[str], ReadWith] = partial(ReadWith, HitDeserializer)
# BinReader: Callable[[str], ReadWith] = partial(ReadWith, BinDeserializer)
