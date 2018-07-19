#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 18:09:27 2018

@author: daehyun
"""

from pprint import pprint
from itertools import chain, combinations
from functools import lru_cache
from glob import iglob

import dask.array as da
from pandas import DataFrame
import numpy as np
from matplotlib import pyplot as plt

from saclatools import bin_reader as __bin_reader


@lru_cache()
def bin_reader(filename):
    keys = ("fel_status", "fel_shutter", "uv_shutter", "meta3",
            "fel_intensity", "delay", "meta6", "meta7")
    print("Loading a file {}...".format(filename))
    loaded = tuple(__bin_reader(filename, keys=keys))
    pprint(loaded[0])
    return loaded
# %%
bin_filename = ("/Users/daehyun/Library/Group Containers/G69SCX94XU.duck"
                "/Library/Application Support/duck/Volumes/xhpcfep02 – SFTP"
                "/2017B8050/bin_files/Aq{:03d}.bin".format)
bins = np.linspace(2000, 8000, 601)


@lru_cache()
def pipico_spectrum(run):
    events = bin_reader(bin_filename(run))
    hits = chain(*(combinations((h['t'] for h in e['hits']), 2)
                   for e in events))
    df = DataFrame(list(hits))
    hist, *_ = np.histogram2d(df[0], df[1], bins=(bins, bins))
    return hist


# %%
plt.figure(figsize=(10, 10))
img = pipico_spectrum(64).T
bg = img.sum(1)[:, None]@img.sum(0)[None, :]/img.sum()
plt.pcolormesh(bins, bins, (img - 0.4 * bg) * (img != 0), cmap='RdBu')

plt.title("PIPICO (run 64): img - 0.4 * scrambled")
#plt.xlim(2500, 6000)
#plt.ylim(2500, 6000)
plt.clim(-500, 500)
plt.colorbar()
plt.show()

# %%
bin_filename = ("/Users/daehyun/Library/Group Containers/G69SCX94XU.duck"
                "/Library/Application Support/duck/Volumes/xhpcfep02 – SFTP"
                "/2017B8050/bin_files/Aq{:03d}.bin".format)
bins = np.linspace(2000, 8000, 601)


def hit_map(event):
    hits = (h['t'] for h in event['hits'])
    hist, _ = np.histogram(tuple(hits), bins)
    return np.append(event['fel_intensity'], hist)


runs = (64,)
events = chain(*(bin_reader(bin_filename(r)) for r in runs))
arr = np.array(tuple(hit_map(e) for e in events if e['nhits'] > 0))

# %%
img = da.cov(arr.T).compute()
img_cov = img[1:, 1:]
img_icov = img[1:, 0][:, None] @ img[0, 1:][None, :] / img[0, 0]
img_pcov = img_cov - img_icov

# %%
plt.figure(figsize=(10, 10))
plt.pcolormesh(bins, bins, img_pcov, cmap='RdBu')
plt.title('pcov map (run {})'.format(', '.join(str(r) for r in runs)))
plt.xlabel('tof (ns)')
plt.ylabel('tof (ns)')
plt.colorbar()
plt.clim(-0.0001, 0.0001)
#plt.xlim(2500, 6000)
#plt.ylim(2500, 6000)

# %%
plt.figure(figsize=(8,24))
plt.subplot(311)
plt.pcolormesh(bins, bins, img_pcov, cmap='RdBu')
plt.title('pcov map: runs {}'.format(', '.join(str(r) for r in runs)))
plt.xlabel('tof (ns)')
plt.ylabel('tof (ns)')
plt.colorbar()
plt.clim(-0.0001, 0.0001)


plt.subplot(312)
plt.pcolormesh(bins, bins, img_cov, cmap='RdBu')
plt.title('cov map: runs {}'.format(', '.join(str(r) for r in runs)))
plt.xlabel('tof (ns)')
plt.ylabel('tof (ns)')
plt.colorbar()
plt.clim(-0.0001, 0.0001)


plt.subplot(313)
plt.pcolormesh(bins, bins, img_icov, cmap='RdBu')
plt.title('icov map: runs {}'.format(', '.join(str(r) for r in runs)))
plt.xlabel('tof (ns)')
plt.ylabel('tof (ns)')
plt.colorbar()
plt.clim(-0.0001, 0.0001)

plt.show()

# %%
bin_filename = ("/Users/daehyun/Library/Group Containers/G69SCX94XU.duck"
                "/Library/Application Support/duck/Volumes/xhpcfep02 – SFTP"
                "/2017B8050/bin_files/Aq{:03d}.bin".format)
bins = np.linspace(2000, 8000, 601)


def hit_map(event):
    hits = (h['t'] for h in event['hits'])
    hist, _ = np.histogram(tuple(hits), bins)
    return np.append(event['fel_intensity'], hist)


runs = (64,)
events = chain(*(bin_reader(bin_filename(r)) for r in runs))
arr = np.array(tuple(hit_map(e) for e in events if e['nhits'] > 0))