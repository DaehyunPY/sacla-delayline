from glob import iglob
from os.path import splitext, basename, getmtime, getctime
from struct import Struct
from time import sleep

from tqdm import tqdm

from saclatools import hit_reader, scalars_at


# parameters!
hit_filename = "/work/uedalab/2017A8005/hit_files/_preanalysis/{0}/{0}.hit".format
bin_filename = "/work/uedalab/2017A8005/bin_files/{}.bin".format
hightag = 201701
equips = {
    'fel_status': ('xfel_mon_ct_bl1_dump_1_beamstatus/summary', bool),
    'fel_shutter': ('xfel_bl_1_shutter_1_open_valid/status', bool),
    'laser_shutter': ('xfel_bl_1_lh1_shutter_1_open_valid/status', bool),
    'fel_intensity': ('xfel_bl_1_tc_gm_2_pd_fitting_peak/voltage', float),
    'delay_motor': ('xfel_bl_1_st_4_motor_22/position', float)
}


def convert(ifile, ofile='exported.bin'):
    tags = tuple(d['tag'] for d in hit_reader(ifile))  # get tag list
    print("Getting metadata...")
    df = scalars_at(*tags, hightag=hightag, equips=equips)  # get SACLA meta data
    print("Done!")
    deep1 = Struct('=IBBBBddddI')
    pack1 = deep1.pack
    deep2 = Struct('=ddd')
    pack2 = deep2.pack

    print("Writing a .bin file...")
    with open(ofile, 'bw') as f:
        write = f.write
        for hits in tqdm(hit_reader(ifile), total=len(tags)):
            meta = df.loc[hits['tag']]
            write(pack1(hits['tag'],  # uint32
                        meta['fel_status'],  # uint8
                        meta['fel_shutter'],  # uint8
                        meta['laser_shutter'],  # uint8
                        0,  # uint8
                        meta['fel_intensity'],  # float64
                        meta['delay_motor'],  # float64
                        0,  # float64
                        0,  # float64
                        hits.get('nhits', 0)))  # uint32
            for hit in hits.get('hits', ()):
                write(pack2(hit['t'], hit['x'], hit['y']))
    print("Done!")


while True:
    hits = {splitext(basename(fn))[0]: getmtime(fn) for fn in iglob(hit_filename("*"))}
    bins = {splitext(basename(fn))[0]: getctime(fn) for fn in iglob(bin_filename("*"))}
    jobs = tuple(fn for fn, t in hits.items() if fn not in bins or bins[fn] < t)
    if len(jobs) == 0:
        print("Nothing to do!")
    else:
        for fn in jobs:
            print('Converting file {}...'.format(fn))
            convert(hit_filename(fn), bin_filename(fn))
            print('Done!')
    sleep(10)
