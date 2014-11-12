import sys
import os
import glob
import argparse
from subprocess import call
from bounding_box import BoundingBox
import json
import itertools
import utils

# common functions



def match_multiple_pmcc(tiles_file, index_pairs, fixed_tiles, jar, out_fname, conf_args):
    tiles_url = utils.path2url(os.path.abspath(tiles_file))

    fixed_str = ""
    if fixed_tiles != None:
        fixed_str = "--fixedTiles {0}".format(" ".join(map(str, fixed_tiles)))

    java_cmd = 'java -Xmx14g -Djava.awt.headless=true -cp "{0}" org.janelia.alignment.MatchByMaxPMCC --inputfile {1} {2} {3} --targetPath {4} {5}'.format(
        jar,
        tiles_url,
        fixed_str,
        " ".join("--indices {}:{}".format(a, b) for a, b in index_pairs),
        out_fname,
        conf_args)
    print "Executing: {0}".format(java_cmd)
    call(java_cmd, shell=True) # w/o shell=True it seems that the env-vars are not set


def match_by_max_pmcc(tiles_file, fixed_tiles, out_fname, jar_file, conf=None):

    tile_file = tiles_file.replace('file://', '')
    with open(tile_file, 'r') as data_file:
        tilespecs = json.load(data_file)
    # TODO: add all tiles to a kd-tree so it will be faster to find overlap between tiles
    # TODO: limit searches for matches to overlap area of bounding boxes

    # iterate over the tiles, and for each tile, find intersecting tiles that overlap,
    # and match their features
    # Nested loop:
    #    for each tile_i in range[0..N):
    #        for each tile_j in range[tile_i..N)]
    indices = []
    for idx1 in range(len(tilespecs)):
        for idx2 in range(idx1 + 1, len(tilespecs)):
            # if the two tiles intersect, match them
            bbox1 = BoundingBox.fromList(tilespecs[idx1]["bbox"])
            bbox2 = BoundingBox.fromList(tilespecs[idx2]["bbox"])
            if bbox1.overlap(bbox2):
                imageUrl1 = tilespecs[idx1]["mipmapLevels"]["0"]["imageUrl"]
                imageUrl2 = tilespecs[idx2]["mipmapLevels"]["0"]["imageUrl"]
                print "Matching by max pmcc: {0} and {1}".format(imageUrl1, imageUrl2)
                indices.append((idx1, idx2))

    conf_args = utils.conf_args(conf, 'MatchByMaxPMCC')

    match_multiple_pmcc(tiles_file, indices, fixed_tiles, jar_file, out_fname, conf_args)

def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Iterates over the tilespecs in a file, template matching each overlapping tile.')
    parser.add_argument('tiles_file', metavar='tiles_file', type=str,
                        help='the json file of tilespecs')
    #parser.add_argument('corr_file', metavar='corr_file', type=str,
    #                    help='the json file of the')
    parser.add_argument('-o', '--output_file', type=str, 
                        help='an output correspondent_spec file, that will include the sift features for each tile (default: ./matches.json)',
                        default='./matches.json')
    parser.add_argument('-f', '--fixed_tiles', type=str, nargs='+',
                        help='a space separated list of fixed tile indices (default: 0)',
                        default="0")
    parser.add_argument('-j', '--jar_file', type=str,
                        help='the jar file that includes the render (default: ../target/render-0.0.1-SNAPSHOT.jar)',
                        default='../target/render-0.0.1-SNAPSHOT.jar')
    parser.add_argument('-c', '--conf_file_name', type=str, 
                        help='the configuration file with the parameters for each step of the alignment process in json format (uses default parameters, if not supplied)',
                        default=None)


    args = parser.parse_args()

    match_sift_features(args.tiles_file, args.fixed_tiles, args.output_file, args.jar_file, \
        conf=utils.conf_args_from_file(args.conf_file_name, "MatchByMaxPMCC"))

if __name__ == '__main__':
    main()
