# -*- coding: utf-8 -*-
from random import shuffle
from pprint import pprint
from functools import lru_cache

from dask.array import cov
from numpy import histogram, append, linspace, stack
from matplotlib import pyplot as plt
from cytoolz import concat, take

from saclatools import bin_reader as __bin_reader


# %%
@lru_cache()
def bin_reader(filename):
    keys = ("fel_status", "fel_shutter", "uv_shutter", "meta3",
            "fel_intensity", "delay", "meta6", "meta7")
    print("Loading a file {}...".format(filename))
    loaded = tuple(__bin_reader(filename, keys=keys))
    pprint(loaded[0])
    return loaded


# %%
filename = "/Users/daehyun/Documents/sacla-delayline/Aq{:03d}.bin".format
run = 67
events = bin_reader(filename(run))
hits = list(concat(d['hits'] for d in events))
shuffle(hits)
hit_stream = iter(hits)
shuffled_events = (
    {'hits': take(d['nhits'], hit_stream),
     'fel_intensity': d['fel_intensity']}
    for d in events
)


bins = linspace(2000, 8000, 601)


def hit_map(event):
    t = (h['t'] for h in event['hits'])
    hist, _ = histogram(tuple(t), bins)
    return append(event['fel_intensity'], hist)


def par_cov(arr):
    img = cov(arr.T).compute()
    img_cov = img[1:, 1:]
    img_icov = img[1:, 0][:, None] @ img[0, 1:][None, :] / img[0, 0]
    return img_cov - img_icov
    # return img_cov


pcov = par_cov(stack(tuple(hit_map(d) for d in events)))
pcov_shuffled = par_cov(stack(tuple(hit_map(d) for d in shuffled_events)))


# %%
plt.figure(figsize=(5, 15))
plt.subplot(311)
plt.pcolormesh(bins, bins, pcov, cmap='RdBu')
plt.title("pcov map (run {})".format(run))
plt.xlim(2500, 6000)
plt.ylim(2500, 6000)
plt.clim(-0.0001, 0.0001)

plt.subplot(312)
plt.pcolormesh(bins, bins, pcov_shuffled, cmap='RdBu')
plt.title("pcov map from shuffled events (run {})".format(run))
plt.xlim(2500, 6000)
plt.ylim(2500, 6000)
plt.clim(-0.0001, 0.0001)

plt.subplot(313)
plt.pcolormesh(bins, bins, pcov-pcov_shuffled, cmap='RdBu')
plt.title("differential (run {})".format(run))
plt.xlim(2500, 6000)
plt.ylim(2500, 6000)
plt.clim(-0.0001, 0.0001)
plt.show()
