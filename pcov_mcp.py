from pprint import pprint
from glob import iglob
from h5py import File

from numpy import ndarray, save, load, arange
from dask.delayed import delayed
from dask.array import concatenate, cov, from_delayed
from dask.diagnostics import ProgressBar


run = 64
filename = ("/work/uedalab/"
            "/2017B8050/hdf_files/Aq{:03d}__*.h5".format)

filenames = sorted(iglob(filename(run)))
pprint(filenames)


@delayed
def target(filename):
    with File(filename, 'r') as f:
        return concatenate([f['/fel_intensity'][...][:, None], f['/channel6'][:, 2500:6000]], axis=1).astype('float32')


shapes = [File(fn, 'r')['/tags'].shape[0] for fn in filenames]
arr = concatenate([from_delayed(target(fn), shape=[sh, 3501], dtype='float32') for fn, sh in zip(filenames, shapes)])

with ProgressBar():
    img: ndarray = cov(arr.T).compute()
    pprint(img)

save("pcov_r{:03d}.npy".format(run), img)

# img = load("pcov_r065.npy")
# img_cov = img[1:, 1:]
# img_icov = img[1:, 0][:, None] @ img[0, 1:][None, :] / img[0, 0]
# img_pcov = img_cov - img_icov
#
# from matplotlib import pyplot as plt
#
# bins = arange(2500, 6000)
# plt.figure()
# plt.pcolormesh(bins, bins, img_pcov, cmap='RdBu')
# plt.title("pcov map from raw mcp signal (run 65)")
# plt.colorbar()
# plt.clim(-100, 100)
# plt.show()
