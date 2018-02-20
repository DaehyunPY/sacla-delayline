# distutils: language=c++

from cython cimport dict
from libc.stdio cimport FILE, EOF, SEEK_SET, SEEK_CUR, SEEK_END, fopen, fclose, fread, fgetc, fseek, ftell
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from libcpp.string cimport string
from numpy cimport ndarray, npy_int16, npy_int64, npy_int32, npy_uint32, npy_float64
from numpy import zeros, empty


cdef struct channel:
    npy_int32 fullscale, offset, baseline
    npy_float64 gain


cdef class LmaReader:
    """
    Deserializer of LMA format
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
        with LmaReader(filename) as r:
            for d in r:
                print(d)
                break
    """
    cdef:
        FILE * __file
        string __filename
        npy_int32 __nchannels, __nsamples
        npy_int64 __pos_begin, __pos_end
        npy_float64 __sample_interval
        vector[npy_int32] __channels
        vector[channel] __channel_info

    def __cinit__(self, str filename):
        cdef npy_int32 ret

        self.__filename = filename.encode()
        self.__file = fopen(self.__filename.c_str(), "rb")
        if not self.__file:
            raise FileNotFoundError("No such a file: {}!".format(filename))

        self.__read_header()

        ret = fclose(self.__file)
        if not ret == 0:
            raise IOError("Fail to close a file: {}!".format(ret))
        self.__file = NULL

    def __dealloc__(self):
        cdef npy_int32 ret

        if self.__file:
            ret = fclose(self.__file)
            if not ret == 0:
                raise IOError("Fail to close a file: {}!".format(ret))
            self.__file = NULL

    def __repr__(self) -> str:
        return "LmaReader({})".format(self.filename)

    cdef void __read_header(self):
        cdef:
            npy_int16 dump_int16
            npy_int32 dump_int32, ret, n, i
            npy_uint32 dump_uint32
            npy_float64 dump_float64
            channel * ch

        ret = fseek(self.__file, 0, SEEK_SET)
        if not ret == 0:
            raise IOError("Fail to seek a position: {}!".format(ret))

        # header[0] int32
        ret = fread(&dump_int32, 4, 1, self.__file)
        if not ret == 1:
            raise IOError("Fail to read a block: {}!".format(ret))
        self.__pos_begin = dump_int32 + 4

        # header[1] int16
        ret = fread(&dump_int16, 2, 1, self.__file)
        if not ret == 1:
            raise IOError("Fail to read a block: {}!".format(ret))
        n = dump_int16

        # header[2] int16
        ret = fseek(self.__file, 2, SEEK_CUR)
        if not ret == 0:
            raise IOError("Fail to seek a position: {}!".format(ret))

        # header[3] float64
        ret = fread(&dump_float64, 8, 1, self.__file)
        if not ret == 1:
            raise IOError("Fail to read a block: {}!".format(ret))
        self.__sample_interval = dump_float64

        # header[4] int32
        ret = fread(&dump_int32, 4, 1, self.__file)
        if not ret == 1:
            raise IOError("Fail to read a block: {}!".format(ret))
        self.__nsamples = dump_int32

        # header[5:9] float64, int16, float64, int16
        ret = fseek(self.__file, 8 + 2 + 8 + 2, SEEK_CUR)
        if not ret == 0:
            raise IOError("Fail to seek a position: {}!".format(ret))

        # header[9] uint32
        ret = fread(&dump_uint32, 4, 1, self.__file)
        if not ret == 1:
            raise IOError("Fail to read a block: {}!".format(ret))
        self.__channels = [i for i in range(n) if (dump_uint32 >> i) & 0b1]
        self.__nchannels = self.__channels.size()
        self.__channel_info.resize(self.__nchannels)

        # header[10:12] uint32, int16
        ret = fseek(self.__file, 4 + 2, SEEK_CUR)
        if not ret == 0:
            raise IOError("Fail to seek a position: {}!".format(ret))

        for i in range(self.__nchannels):
            ch = &self.__channel_info[i]

            # channel[0] int16
            ret = fread(&dump_int16, 2, 1, self.__file)
            if not ret == 1:
                raise IOError("Fail to read a block: {}!".format(ret))
            ch.fullscale = dump_int16

            # channel[1] int16
            ret = fread(&dump_int16, 2, 1, self.__file)
            if not ret == 1:
                raise IOError("Fail to read a block: {}!".format(ret))
            ch.offset = dump_int16

            # channel[2] float64
            ret = fread(&dump_float64, 8, 1, self.__file)
            if not ret == 1:
                raise IOError("Fail to read a block: {}!".format(ret))
            ch.gain = dump_float64

            # channel[3] int16
            ret = fread(&dump_int16, 2, 1, self.__file)
            if not ret == 1:
                raise IOError("Fail to read a block: {}!".format(ret))
            ch.baseline = dump_int16

            # channel[4:7] int16, int32, int32
            ret = fseek(self.__file, 2 + 4 + 4, SEEK_CUR)
            if not ret == 0:
                raise IOError("Fail to seek a position: {}!".format(ret))

        ret = fseek(self.__file, 0, SEEK_END)
        if not ret == 0:
            raise IOError("Fail to seek a position: {}!".format(ret))
        self.__pos_end = ftell(self.__file)

    @property
    def filename(self) -> str:
        return self.__filename.decode()

    @property
    def pos_begin(self) -> int:
        return self.__pos_begin

    @property
    def pos_end(self) -> int:
        return self.__pos_end

    @property
    def pos_cur(self) -> int:
        return ftell(self.__file)

    @property
    def nchannels(self) -> int:
        return self.__nchannels

    @property
    def nsamples(self) -> int:
        return self.__nsamples

    @property
    def sample_interval(self) -> float:
        return self.__sample_interval

    @property
    def channels(self) -> list:
        return self.__channels

    @property
    def channel_info(self) -> dict:
        return self.__channel_info

    def __enter__(self):
        self.__file = fopen(self.__filename.c_str(), "rb")
        if not self.__file:
            raise FileNotFoundError("No such a file: {}!".format(self.__filename))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        cdef int ret

        ret = fclose(self.__file)
        if not ret == 0:
            raise IOError("Fail to close a file: {}!".format(ret))
        self.__file = NULL

    def __iter__(self):
        cdef npy_int32 ret

        if not self.__file:
            raise IOError("File is closed!")

        ret = fseek(self.__file, self.__pos_begin, SEEK_SET)
        if not ret == 0:
            raise IOError("Fail to seek a position: {}!".format(ret))

        while fgetc(self.__file) != EOF:
            ret = fseek(self.__file, -1, SEEK_CUR)
            if not ret == 0:
                raise IOError("Fail to seek a position: {}!".format(ret))
            yield self.__next()
        return

    cdef dict __next(self):
        cdef:
            npy_int16 dump_int16
            npy_int32 ret, tag, m, i, j, k[2]
            pair[npy_int32, vector[vector[npy_float64]]] event
            ndarray[npy_float64, ndim=2, mode="c"] arr = zeros((self.__nchannels, self.__nsamples), dtype='float64')
            ndarray[npy_int16, ndim=1, mode="c"] dump
            channel * ch

        # event[0] int32
        ret = fread(&tag, 4, 1, self.__file)
        if not ret == 1:
            raise IOError("Fail to read a block: {}!".format(ret))

        # event[1] float64
        ret = fseek(self.__file, 8, SEEK_CUR)
        if not ret == 0:
            raise IOError("Fail to seek a position: {}!".format(ret))

        for i in range(self.__nchannels):
            ch = &self.__channel_info[i]

            # event[2] int16
            ret = fread(&dump_int16, 2, 1, self.__file)
            if not ret == 1:
                raise IOError("Fail to read a block: {}!".format(ret))
            m = dump_int16

            for j in range(m):
                # event[3] int32
                ret = fread(&k, 4, 2, self.__file)
                if not ret == 2:
                    raise IOError("Fail to read a block: {}!".format(ret))

                # event[4] int16
                dump = empty(k[1], dtype='int16')
                ret = fread(&dump[0], 2, k[1], self.__file)
                if not ret == k[1]:
                    raise IOError("Fail to read a block: {}!".format(ret))
                arr[i, k[0]:k[0] + k[1]] = ch.gain * (dump.astype('float64') - ch.baseline)
        return {"tag": tag, **{"channel{}".format(self.__channels[i]): arr[i] for i in range(self.__nchannels)}}
