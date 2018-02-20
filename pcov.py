from pprint import pprint
from glob import iglob
from h5py import File

from dask.delayed import delayed
from dask.array import concatenate, cov, from_array, from_delayed
from dask.diagnostics import ProgressBar
from matplotlib import pyplot as plt

filename = ("/work/uedalab/"
            "/2017B8050/hdf_files/Aq{:03d}__*.h5".format)

filenames = sorted(iglob(filename(67)))
pprint(filenames)


#
# # %%
# shapes = [File(fn, 'r')['/channel6'].shape for fn in filenames]
# arrs = concatenate([from_array(File(fn, 'r')['/channel6'], chunks=sh) for fn, sh in zip(filenames, shapes)])
# arr0 = concatenate([from_array(File(fn, 'r')['/fel_intensity'], chunks=sh) for fn, (sh, _) in zip(filenames, shapes)])
# arr = concatenate([arr0[:, None], arrs[:, 2500:6000]], axis=1)
#
# with ProgressBar():
#     img = cov(arr.T).compute()
#     pprint(img)
#
# img_cov = img[1:, 1:]
# img_icov = img[1:, 0][:, None] @ img[0, 1:][None, :] / img[0, 0]
# img_pcov = img_cov - img_icov


# %%
@delayed
def target(filename):
    with File(filename, 'r') as f:
        return concatenate([f['/fel_intensity'][...][:, None], f['/channel6'][:, 2500:6000]], axis=1).astype('float32')


shapes = [File(fn, 'r')['/tags'].shape[0] for fn in filenames]
arr = concatenate([from_delayed(target(fn), shape=[sh, 3501], dtype='float32') for fn, sh in zip(filenames, shapes)])

with ProgressBar():
    img = cov(arr.T).compute()
    pprint(img)

img_cov = img[1:, 1:]
img_icov = img[1:, 0][:, None] @ img[0, 1:][None, :] / img[0, 0]
img_pcov = img_cov - img_icov

#
# # %%
# plt.figure()
# plt.pcolormesh(img_pcov, cmap='RdBu')
# plt.colorbar()
# plt.clim(-100, 10)
# plt.savefig('tmp.png')
# plt.show()
