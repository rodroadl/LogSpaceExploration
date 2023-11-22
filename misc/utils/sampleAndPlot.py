# Bruce A. Maxwell
# June 2023
# Build a 3D scatter plot of the log data in an image
#
# postprocess the raw image
# resize the image by 4x (blurring)
# random sample to a fixed #
# build the 3D plot

# good images
# 1446: flowers and leaves
# 2139: colorful stuff: view log at 45 / 45 for a star pattern, good example
# 3737: colorful stuff: clear example of quantization of lower values
# 5298: Zap on a blue blanket



import sys
import rawpy
import numpy as np
import math
import cv2
from matplotlib import pyplot as plt
from matplotlib import colors
import raw_utility as rtl
import os

# scale factor
SCALE_FACTOR = 8
SAMPLES = 8000

# orientation for log space: elev = -160, azim = -10, vert_axis = 'z'
# orientation for lin/jpg space: elev = 30, azim = 240, vertical_axis = 'y'
def scatter_generator( image_data, color_src, elev, azim, vert_axis,
                       figfilename = '',
                       removeSat=True,
                       samples = SAMPLES,
                       showFigure=True,
                       minMaxBounds=[]
                      ):

    fig = plt.figure(figsize=(10, 10))
    axis = fig.add_subplot(1, 1, 1, projection='3d')

    r, g, b = cv2.split(image_data)
    # flatten and sample
    rf = r.flatten()
    gf = g.flatten()
    bf = b.flatten()
    pixel_colors = color_src.reshape((np.shape(image_data)[0] * np.shape(image_data)[1], 3))

    if removeSat:
        # identify any saturated channels
        maxval = np.max(gf)
        if maxval <= 1.0:
            satval = 1.0
        elif maxval <= np.log(256):
            satval = np.log(256) - 0.001
        elif maxval <= np.log(65535):
            satval = np.log(65535) - 0.001
        elif maxval < 256:
            satval = 255
        elif maxval < 65536:
            satval = 65535

        runsat = np.logical_and( rf < satval, rf > 0 )
        gunsat = np.logical_and( gf < satval, gf > 0 )
        bunsat = np.logical_and( bf < satval, bf > 0 )
        unsatpix = np.logical_and( runsat, gunsat )
        unsatpix = np.logical_and( unsatpix, bunsat )

        rf = rf[unsatpix]
        gf = gf[unsatpix]
        bf = bf[unsatpix]
        pixel_colors = pixel_colors[unsatpix]

    samples = np.random.choice( range(rf.shape[0]), size=SAMPLES )
    rf = rf[samples]
    gf = gf[samples]
    bf = bf[samples]

    if len(minMaxBounds) < 2:
        maxval = np.max(gf)
        minval = min( [np.min(gf), np.min(rf), np.min(bf)] )
        if maxval <= 1.0:
            minmax = [minval, 1.]
        elif maxval <= 5.7:
            minmax = [minval, np.log(256) ]
        elif maxval <= 11.5:
            minmax = [minval, np.log(65536) ]
        elif maxval <= 256:
            minmax = [minval, 256]
        elif maxval <= 32768:
            minmax = [minval, 32768]
        else:
            minmax = [minval, 65536]
    else:
        minmax = [minMaxBounds[0], minMaxBounds[1]]

    print("** Using min-max bounds [%d %d]" % (minmax[0], minmax[1]) )
    
    pixel_colors = pixel_colors[samples]
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    
    z1_plot = axis.scatter(rf, gf, bf, facecolors=pixel_colors, marker=".")
    axis.view_init( elev = elev, azim = azim, vertical_axis = vert_axis )
    axis.set_xlabel('Red', fontsize=8)
    axis.set_ylabel('Green', fontsize=8)
    axis.set_zlabel('Blue', fontsize=8)
    axis.set_xlim( minmax )
    axis.set_ylim( minmax )
    axis.set_zlim( minmax )

    if len(figfilename) > 0:
        plt.savefig( figfilename )

    if showFigure:
        plt.show()

    plt.close()

    return
    
def main(argv):

    if len(argv) < 2:
        print("usage: python %s <image path> <opt: # samples>" % (argv[0]) )
        return

    # grab the file path
    filename = argv[1]

    numSamples = SAMPLES
    if len(argv) > 2:
        try:
            n = int(argv[2])
        except:
            print("Unable to parse %s as an integer" % (argv[2]) )
            return
        numSamples = n
        

    suffix = filename.split(".")[1].lower()
    if suffix == 'jpg':
        rgb, logrgb = rtl.processJPG( filename, scale_factor = SCALE_FACTOR )
    elif suffix == 'tif':
        rgb, logrgb = rtl.processTIF( filename, scale_factor = SCALE_FACTOR )
    else:
        rgb, logrgb = rtl.processRAW( filename, scale_factor = SCALE_FACTOR )

    scatter_generator( rgb, rgb, 30, 240, 'y', samples=numSamples, figfilename="figure-lin.png" )

    #scatter_generator(logrgb, rgb, -160, -10, 'z', samples=10000 )
    #scatter_generator(logrgb, rgb, -170, -160, 'z' )
    # scatter_generator(logrgb, rgb, 30, -50, 'y' )
    scatter_generator(logrgb, rgb, 40, -45, 'y', samples=numSamples, figfilename="figure-log.png")
    

    return


if __name__ == "__main__":
    main(sys.argv)


        
        
