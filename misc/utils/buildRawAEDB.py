# Bruce A. Maxwell
# 2023/06/16 Summer 2023
# Build a resized DB of the rawAE database
#
# Generates log, linear, and jpg versions of each image in the DB, assumes there is a raw and a jpg of each image
# jpg database is just resized from the original jpg image read and written by openCV
# linear DB is saved as 16-bit TIFFs generated from the raw images
# log DB is saves as 32-bit EXR images generated from the linear data generated from the raw images
# log data has the black-point subtrated prior to taking the log
#     - all linear values of 0 are set to 1 prior to taking the log
#
# resizing for log images is done in linear space prior to taking the log
#
# usage: python buildRAwAEDB.py <originals directory> <small edge size>
#
# Defaults
#   auto-bright turned on
#   % saturated in auto-bright: 0.001%
#   interpolation in resize: cv2.INTER_AREA
#

import rawpy
import numpy as np
import imageio
import math
import cv2
import os
import sys

AUTO_BRIGHT_THR = 0.001

# have to enable OpenEXR in OpenCV
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def processRAW( rawfilename, smallEdgeSize ):

    # read the raw image using rawpy
    with rawpy.imread( rawfilename ) as raw_file:
        num_bits = int(math.log( raw_file.white_level + 1, 2) )
        rgb = raw_file.postprocess( gamma = (1, 1),
                                    no_auto_bright = False,
                                    auto_bright_thr = AUTO_BRIGHT_THR,
                                    output_bps = 16,
                                    use_camera_wb = True )

        # do the resize in linear space
        rows, cols = rgb.shape[0], rgb.shape[1]
        resizeFactor = smallEdgeSize / min(rows, cols)

        newdim = (int(cols * resizeFactor), int(rows * resizeFactor) )
        rgb = cv2.resize( rgb, newdim, interpolation = cv2.INTER_AREA )

        # make a copy and convert it to a 32-bit float
        logrgb = rgb.astype( "float32" )

        # take the log, leaving the zeros as zero (effectively making them 1)
        logrgb[logrgb != 0] = np.log( logrgb[logrgb !=0] ) # avoid log of 0
    
    return rgb, logrgb


# arguments   : an originalJPG (openCV Mat) and the new small edge size (int)
# return value: the resized jpg image (openCV Mat)
def processJPG( originalJPGfilename, smallEdgeSize ):

    # read the jpg image
    originalJPG = cv2.imread( originalJPGfilename )

    # compute the scale factor
    rows, cols = originalJPG.shape[0], originalJPG.shape[1]
    resizeFactor = smallEdgeSize / min(rows, cols)

    newdim = (int(cols * resizeFactor), int(rows * resizeFactor) )
    
    # resize the original image
    resizedJPG = cv2.resize( originalJPG, newdim, interpolation = cv2.INTER_AREA )

    # return the resized image
    return resizedJPG


# main top-level function 
def main(argv):

    if len(argv) < 4:
        print("usage: python %s <original directory> <small edge size> <output directory>" % (argv[0]))
        return

    srcdir = argv[1].rstrip('/')
    try:
        smallEdgeSize = int( argv[2] )
    except:
        print("Unable to convert %s to small edge size, expecting an integer" % (argv[2]) )
        return

    if smallEdgeSize < 32:
        print("smallEdgeSize of %d is too small" % (smallEdgeSize) )
        return
    
    print("** Resizing to small edge as %d" % (smallEdgeSize) )
    dstdir = argv[3].rstrip('/')


    # check if the jpg, lin, and log sub-directories are in the output directory
    dstlist = os.listdir( dstdir )
    print("** Checking if jpg, lin, log sub-directories exist in %s" % (dstdir))
    if not 'jpg' in dstlist:
        print("** Creating jpg sub-directory")
        try:
            os.mkdir( dstdir + "/" + "jpg" )
        except:
            print("Unable to create jpg sub-directory")
            return

    if not 'lin' in dstlist:
        print("** Creating lin sub-directory")
        try:
            os.mkdir( dstdir + "/" + "lin" )
        except:
            print("Unable to create lin sub-directory")
            return

    if not 'log' in dstlist:
        print("** Creating log sub-directory")
        try:
            os.mkdir( dstdir + "/" + "log" )
        except:
            print("Unable to create log sub-directory")
            return
        
    # get the filelist for the original image directory
    print("** Processing directory %s\n" % (srcdir) )
    filelist = os.listdir( srcdir )

    # make a dictionary with the number as the key
    # should be two images per key
    # this assumes the numbers are the last things in the name and separated by " - "
    fileids = {}
    for filename in filelist:
        words = filename.split(" - ")
        numberstr = words[-1].split(".")[0]

        try:
            fileno = int(numberstr)
        except:
            print("Warning: unable to parse filename %s (%s), skipping" % (filename, numberstr) )
            continue

        if fileids.get(fileno, False) != False:
            fileids[fileno].append(filename)
        else:
            fileids[fileno] = [filename]
    

    keys = list(fileids.keys())
    keys.sort()
    print("\n** Processing %d images" % (len(keys)) )
    
    # for each key, generate the three output versions
    for key in keys:

        # double-check there are two images
        if len(fileids[key]) != 2:
            print("Missing jpg or raw version of image %d, skipping image" % (key) )
            continue

        # for each file
        filenames = fileids[key]
        for i in range( len(filenames) ):
            suffix = filenames[i][-3:].lower()
            if suffix == 'jpg':
                resizedjpg = processJPG( srcdir + "/" + filenames[i], smallEdgeSize )

                # build the output filename
                words = filenames[i].split(' - ')
                newfilename = words[0] + "_%d_" % (smallEdgeSize) + " - " + words[1]

                # build the output path
                newfilepath = dstdir + "/jpg/" + newfilename
                
                # save the jpg
                print("** Saving jpg as %s" % (newfilepath))

                # execute the imwrite command
                cv2.imwrite( newfilepath, resizedjpg )
                
            else: # has to be the raw file
                resizedlin, resizedlog = processRAW( srcdir + "/" + filenames[i], smallEdgeSize )

                # build the output filename for the linear file, type .tif
                words = filenames[i].split(' - ')
                tails = words[1].split('.')
                newfilename = words[0] + "_%d_" % (smallEdgeSize) + " - " + tails[0] + '.tif'

                newfilepath = dstdir + "/lin/" + newfilename

                # save the linear version as a TIFF
                print("** Saving lin as %s" % (newfilepath) )

                # execute the imwrite command
                imageio.imsave( newfilepath, resizedlin )

                # build the output filename for the log file, type .exr
                newfilename = words[0] + "_%d_" % (smallEdgeSize) + " - " + tails[0] + '.exr'

                newfilepath = dstdir + "/log/" + newfilename

                # save the log version as an EXR
                print("** Saving log as %s" % (newfilepath) )
                
                # execute the imwrite command
                imageio.imsave( newfilepath, resizedlog )

        print("")
            
    print("** Done writing files")

                
if __name__ == "__main__":
    main( sys.argv )
    print("** Terminating")
