import sys
import os
import glob
import argparse
from subprocess import call
import utils


def render_tiles_3d(tile_fnames_or_dir_fname, output_dir, from_layer, to_layer, scale, from_x, from_y, to_x, to_y, jar_file, tile_size, output_type, output_pattern, blend_type=None, threads_num=None):

    threads_str = ""
    if threads_num is not None:
        threads_str = "--threads {}".format(threads_num)

    blend_str = ""
    if blend_type is not None:
        blend_str = "--blendType {}".format(blend_type)

    list_file_url = utils.path2url(tile_fnames_or_dir_fname)

    java_cmd = 'java -Xmx32g -XX:ParallelGCThreads=1 -cp "{0}" org.janelia.alignment.Render3DTiles --targetDir {1} --scale {2} \
        {3} --fromLayer {4} --toLayer {5} --fromX {6} --fromY {7} --toX {8} --toY {9} --outputType {10} --outputNamePattern {11} --tileSize {12} {13} {14}'.format(\
            jar_file, output_dir, scale, threads_str, from_layer, to_layer,
            from_x, from_y, to_x, to_y, output_type, output_pattern, tile_size, blend_str, list_file_url)
    utils.execute_shell_command(java_cmd)


def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Given a list of tilespec file names, normalizes all the tilepecs to a single coordinate system starting from (0,0).')
    parser.add_argument('tile_files_or_dirs_fname', metavar='tile_files_or_dirs_fname', type=str, 
                        help='a list of json files that need to be normalized or a file that contains the list of json files')
    parser.add_argument('-o', '--output_dir', type=str, 
                        help='an output directory (default: ./after_render)',
                        default='./after_render')
    parser.add_argument('-j', '--jar_file', type=str,
                        help='the jar file that includes the render (default: ../target/render-0.0.1-SNAPSHOT.jar)',
                        default='../target/render-0.0.1-SNAPSHOT.jar')
    parser.add_argument('-t', '--threads_num', type=int, 
                        help='the number of threads to use (default: 1)',
                        default=1)
    parser.add_argument('--from_layer', type=int, 
                        help='the layer to start from (inclusive, default: the first layer in the data)',
                        default=-1)
    parser.add_argument('--to_layer', type=int, 
                        help='the last layer to render (inclusive, default: the last layer in the data)',
                        default=-1)
    parser.add_argument('--scale', type=float, 
                        help='set the scale of the rendered images (default: full image)',
                        default=1.0)
    parser.add_argument('--from_x', type=int, 
                        help='the left coordinate (default: 0)',
                        default=0)
    parser.add_argument('--from_y', type=int, 
                        help='the top coordinate (default: 0)',
                        default=0)
    parser.add_argument('--to_x', type=int, 
                        help='the right coordinate (default: full image)',
                        default=-1)
    parser.add_argument('--to_y', type=int, 
                        help='the bottom coordinate (default: full image)',
                        default=-1)
    parser.add_argument('-s', '--tile_size', type=int,
                        help='the size (square side) of each tile (default: 512)',
                        default=512)
    parser.add_argument('-b', '--blend_type', type=str,
                        help='the mosaics blending type',
                        default=None)
    parser.add_argument('--output_type', type=str,
                        help='The output type format',
                        default='jpg')
    parser.add_argument('--output_pattern', type=str,
                        help='The output file name pattern where "%rowcol" will be replaced by "_tr[row]-tc[rol]_" with the row and column numbers')

    args = parser.parse_args()

    utils.create_dir(args.output_dir)

    print "args: from_layer {0}, to_layer {1}".format(args.from_layer, args.to_layer)
    render_tiles_3d(args.tile_files_or_dirs_fname, args.output_dir, args.from_layer, args.to_layer, args.scale,
        args.from_x, args.from_y, args.to_x, args.to_y, args.jar_file,
        args.tile_size, args.output_type, args.output_pattern, args.blend_type, args.threads_num)

if __name__ == '__main__':
    main()

