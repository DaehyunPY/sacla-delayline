from glob import iglob
from os.path import splitext, basename, getmtime, getctime
from itertools import chain
from time import sleep
from pprint import pprint

from h5py import File
from pandas import DataFrame

from saclatools import hit_reader, scalars_at


# parameters!
hit_filename = "/work/uedalab/2017B8050/hit_files/{0}/{0}.hit".format
hdf_filename = "/work/uedalab/2017B8050/hdf_files/{}.h5".format
hightag = 201704
equips = {  # must be correct for metadata retrieval
    'fel_status': ('xfel_mon_ct_bl1_dump_1_beamstatus/summary', bool),
    'fel_shutter': ('xfel_bl_1_shutter_1_open_valid/status', bool),
    'laser_shutter': ('xfel_bl_1_lh1_shutter_1_open_valid/status', bool),
    # Select the one (gm_1 or gm_2) which has good intensity reading
    'fel_intensity': ('xfel_bl_1_tc_gm_1_pd_fitting_peak/voltage', float),
    # 'fel_intensity_gm2': ('xfel_bl_1_tc_gm_2_pd_fitting_peak/voltage', float),
    'delay_motor': ('xfel_bl_1_st_4_motor_22/position', float)
}


def convert(ifile, ofile='exported.h5'):
    print("Getting tag list...")
    df = DataFrame(((d['tag'], d['nhits']) for d in hit_reader(ifile)), columns=('tag', 'nhits'))
    cumsummed = df['nhits'].cumsum()
    pprint(df.head())

    print("Getting metadata...")
    meta = scalars_at(*df.index.values.tolist(), hightag=hightag, equips=equips)  # get SACLA meta data
    pprint(meta.head())

    print("Getting hit list...")
    hits = DataFrame(list(chain(*(d['hits'] for d in hit_reader(ifile)))))
    pprint(hits.head())

    with File(ofile) as f:
        f['tof'] = hits['t']
        f['xpos'] = hits['x']
        f['ypos'] = hits['y']
        f['nlistpos'] = cumsummed - df['nhits']
        f['nions'] = df['nhits']
        f['Tagevent'] = df['tag']
        for k, v in meta.iteritems():
            f[k] = v
    print("Done!")


while True:  # run conversion in infinite loop
    hits = {splitext(basename(fn))[0]: getmtime(fn) for fn in iglob(hit_filename("*"))}
    hdfs = {splitext(basename(fn))[0]: getctime(fn) for fn in iglob(hdf_filename("*"))}
    jobs = sorted(fn for fn, t in hits.items() if (fn not in hdfs or hdfs[fn] < t) and fn not in {'Aq041'})
    print("Jobs: {}".format(', '.join(jobs)))
    if len(jobs) == 0:
        print("Nothing to do!")
    else:
        for fn in jobs:
            print('Converting file {}...'.format(fn))
            try:
                convert(hit_filename(fn), hdf_filename(fn))
                print('Done!')
            except Exception as err:
                print("Got an error!")
                print(err)
                print("Trying it next loop!")
    sleep(10)  # break
