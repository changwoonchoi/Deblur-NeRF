import numpy as np
import OpenEXR as exr
import Imath
import imageio


def tonemap(img, exposure=2., gamma=2.2):
    """ Reinhard tonemapping operator
    refer to https://en.wikipedia.org/wiki/Tone_mapping
    """
    luminance = img * (2 ** exposure)
    ldr_img = (luminance / (luminance + 1)) ** (1. / gamma)
    return ldr_img


def load_img(filename, hdr=False):
    if hdr:
        img = read_exr(filename)
    else:
        img = imageio.v2.imread(filename)
        img = img.astype(np.float32) / 255.
    return img


# code borrowed from https://gist.github.com/jadarve/de3815874d062f72eaf230a7df41771b
def read_exr(filename):
    """Read color from EXR image file.

    Parameters
    ----------
    filename : str
        File path.

    Returns
    -------
    img : RGB or RGBA image in float32 format. Each color channel
          lies within the interval [0, 1].
    """

    exr_img = exr.InputFile(filename)
    header = exr_img.header()

    dw = header['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()

    # convert all channels in the image to numpy arrays
    for c in header['channels']:
        C = exr_img.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)

        channelData[c] = C

    # colorChannels = ['R', 'G', 'B', 'A'] if 'A' in header['channels'] else ['R', 'G', 'B']
    colorChannels = ['R', 'G', 'B']
    img = np.concatenate([channelData[c][..., np.newaxis] for c in colorChannels], axis=2)

    return img


def write_exr(img, filename):
    """Write color to EXR image file."""
    assert img.dtype == np.float32
    assert img.ndim == 3
    assert img.shape[2] == 3

    exr_img = exr.OutputFile(filename, exr.Header(img.shape[1], img.shape[0]))
    exr_img.writePixels({'R': img[:, :, 0].tostring(),
                         'G': img[:, :, 1].tostring(),
                         'B': img[:, :, 2].tostring()})
