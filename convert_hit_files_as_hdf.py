from glob import iglob
from itertools import chain
from os.path import splitext, basename, getmtime, getctime
from time import sleep

from h5py import File
from pandas import DataFrame

from sacla_tools import hit_reader, scalars_at

# parameters!
hit_filename = "/work/uedalab/2017A8005/hit_files/_preanalysis/{0}/{0}.hit".format
hdf_filename = "/work/uedalab/2017A8005/hdf_files/{}.h5".format
hightag = 201701
equips = {
    'fel_status': ('xfel_mon_ct_bl1_dump_1_beamstatus/summary', bool),
    'fel_shutter': ('xfel_bl_1_shutter_1_open_valid/status', bool),
    'laser_shutter': ('xfel_bl_1_lh1_shutter_1_open_valid/status', bool),
    'fel_intensity': ('xfel_bl_1_tc_gm_2_pd_fitting_peak/voltage', float),
    'delay_motor': ('xfel_bl_1_st_4_motor_22/position', float)
}


def convert(ifile, ofile='exported.h5'):
    df = DataFrame(((d['tag'], d['nhits']) for d in hit_reader(ifile)), columns=('tag', 'nhits'))
    cumsummed = df['nhits'].cumsum()
    meta = scalars_at(*df.index.values.tolist(), hightag=hightag, equips=equips)  # get SACLA meta data
    hits = DataFrame(list(chain(*(d['hits'] for d in hit_reader(ifile)))))

    with File(ofile) as f:
        f['Tagevent'] = df['tag']
        f['nions'] = df['nhits']
        f['nlistpos'] = cumsummed - df['nhits']
        f['tof'] = hits['t']
        f['xpos'] = hits['x']
        f['ypos'] = hits['y']
        for k, v in meta.iteritems():
            f[k] = v


while True:
    hits = {splitext(basename(fn))[0]: getmtime(fn) for fn in iglob(hit_filename("*"))}
    hdfs = {splitext(basename(fn))[0]: getctime(fn) for fn in iglob(hdf_filename("*"))}
    jobs = tuple(fn for fn, t in hits.items() if fn not in hdfs or hdfs[fn] < t)
    if len(jobs) == 0:
        print("Nothing to do!")
    else:
        for fn in jobs:
            print('Converting file {}...'.format(fn))
            convert(hit_filename(fn), hdf_filename(fn))
            print('Done!')
    sleep(10)
