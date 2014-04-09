from __future__ import print_function
import glob
import sys
import os
import json
import re
import tifffile # fast peeking at tiff sizes
from decimal import *


def extract_coords(filename, image_size, overlap=0.06):
    m = re.match('.*/Tile_r([0-9]+)-c([0-9]+)_.*[.]tif+', filename)
    offset_y = (int(m.group(1)) - 1) * image_size[0] * (1.0 - overlap)
    offset_x = (int(m.group(2)) - 1) * image_size[1] * (1.0 - overlap)
    return offset_x, offset_y

def filename_decimal_key(path):
    return Decimal(''.join([c for c in path if c.isdigit()]))

def find_image_files(subdir):
    return glob.glob(os.path.join(subdir, 'Tile_r*-c*.tif'))

def write_tilespec(subdir):
    '''Writes the tilespec for a single directory (aka, section)'''
    tilespecs = []
    image_size = None

    output_json_fname = os.path.join(subdir, 'tilespec.json')
    if os.path.exists(output_json_fname):
        print("Will not overwrite {}".format(output_json_fname))
        return

    for image_file in sorted(find_image_files(subdir)):
        if image_size is None:
            with tifffile.TiffFile(image_file) as tiffinfo:
                image_size = tiffinfo.pages[0].shape
        coords = extract_coords(image_file, image_size)
        tilespec = {
            "imageUrl" : "file://{0}".format(os.path.abspath(image_file).replace(os.path.sep, '/')),
            "minIntensity" : 0.0,
            "maxIntensity" : 255.0,
            "transforms" : [{
                    "className" : "mpicbg.trakem2.transform.TranslationModel2D",
                    # x, y offset of upper right corner
                    "dataString" : "{0} {1}".format(coords[0], coords[1])
                    }],
            # BoundingBox in the format "from_x to_x from_y to_y" (left right top bottom)
            "boundingBox" : "{0} {1} {2} {3}".format(coords[0], coords[0] + image_size[1],
                                                     coords[1], coords[1] + image_size[0])
            }
        tilespecs.append(tilespec)

    if len(tilespecs) > 0:
        with open(output_json_fname, 'w') as outjson:
            json.dump(tilespecs, outjson, sort_keys=True, indent=4)
            print('Wrote tilespec to {0}'.format(output_json_fname))
    else:
        print('Nothing to write in directory {}'.format(subdir))


if __name__ == '__main__':
    input_folder = sys.argv[1]
    overlap_fraction = 0.06  # from Alyssa's data

    for sub_folder in glob.glob(os.path.join(input_folder, 'Sec*')):
        if os.path.isdir(sub_folder):
            write_tilespec(sub_folder)