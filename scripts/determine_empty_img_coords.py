import sys
import argparse
import cv2
import numpy as np
import os


def determineempty(fname, thresh):
    imgf = cv2.imread(fname, 0)
    clahe = cv2.createCLAHE()
    claheimg = clahe.apply(imgf)

    gradientimg = np.gradient(np.array(claheimg, dtype='float32'))
    vargrad = np.var(gradientimg)
    print(fname + " " + str(vargrad))
    return vargrad > thresh
    # if vargrad > thresh, then the image is empty


def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Given an image, determine if it is empty or has data based on a threshold.')
    parser.add_argument('coordfilename', metavar='coordfilename', type=str,
                        help='the full image coordinates filename')
    parser.add_argument('-t', '--threshold', type=float,
                        help='Empty/Data Threshold (default: 2000)',
                        default=2000)

    args = parser.parse_args()
    
    coordfile = open(args.coordfilename, "r")
    for line in coordfile.readlines():
        imgname = line.split()[0]
        imgname = imgname[0:6] + "/" + imgname[7:]
        dirn = "/n/lichtmanfs2/Susan/SCS_2015-11-23_R1_W06_mSEM/scs_20151217_19-45-07/080_S80R1/" # os.path.dirname(os.path.abspath(coordfile.name))
        isempty = determineempty(dirn + imgname, args.threshold)

if __name__ == '__main__':
    main()
