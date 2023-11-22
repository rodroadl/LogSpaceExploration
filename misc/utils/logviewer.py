# Bruce A. Maxwell
# June 2023
# Log RGB histogram selector and viewer

import sys

import raw_utility as rtl



def main(argv):

    if len(argv) < 2:
        print("usage: python3 %s <image path>" % (argv[0]) )
        return

    # grab the file path
    filename = argv[1]

    # read and process the fileoo
    if filename.split(".")[1].lower() == 'jpg':
        rgb, logrgb = rtl.processJPG( filename )
    else:
        rgb, logrgb = rtl.processRAW( filename )


    return


if __name__ == "__main__":
    main(sys.argv)

