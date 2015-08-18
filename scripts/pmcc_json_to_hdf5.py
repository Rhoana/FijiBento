import h5py
import ujson
import numpy as np
import sys
import os
import argparse
import multiprocessing as mp
import glob
import utils

def convert(file_in, file_out):
    data = ujson.load(open(file_in, "r"))
    assert not os.path.exists(file_out)
    with h5py.File(file_out, 'w') as hf:
        for idx, m in enumerate(data):
            hf.create_dataset("matches{}_url1".format(idx),
                              data=np.array(m["url1"].encode("utf-8"), dtype='S'))
            hf.create_dataset("matches{}_url2".format(idx),
                              data=np.array(m["url2"].encode("utf-8"), dtype='S'))

            p1s = np.array([(pair["p1"]["l"][0], pair["p1"]["l"][1])
                            for pair in m["correspondencePointPairs"]],
                           dtype=np.float32)
            p2s = np.array([(pair["p2"]["l"][0], pair["p2"]["l"][1])
                            for pair in m["correspondencePointPairs"]],
                           dtype=np.float32)
            hf.create_dataset("matches_{}_p1".format(idx), data=p1s)
            hf.create_dataset("matches_{}_p2".format(idx), data=p2s)

def create_chunks(l, n):
    sub_size = max(1, (len(l) - 1)/n + 1)
    return [l[i:i + sub_size] for i in range(0, len(l), sub_size)]

def convert_files(files_list, hdf5_dir):
    for f in files_list:
        print("Converting: {}".format(f))
        in_file = f
        out_file = os.path.join(hdf5_dir, "{}.hdf5".format(os.path.splitext(os.path.basename(f))[0]))
        if not os.path.exists(out_file):
            convert(in_file, out_file)

def parallel_convert(processes_num, json_files, hdf5_dir):
    utils.create_dir(hdf5_dir)

    # create N-1 worker processes
    if processes_num > 1:
        pool = mp.Pool(processes=processes_num - 1)
    print "Creating {} other processes, and parsing {} json files".format(processes_num - 1, len(json_files))

    # Divide the list into processes_num chunks
    chunks = create_chunks(json_files, processes_num)

    async_res = None

    # run all jobs but one by other processes
    for sub_list in chunks[:-1]:
        async_res = pool.apply_async(convert_files, (sub_list, hdf5_dir))

    # run the last job by the current process
    print "running last list with {} files".format(len(chunks[-1]))
    convert_files(chunks[-1], hdf5_dir)

    # wait for all other processes to finish their job
    if processes_num > 1:
        if pool is not None:
            pool.close()
            pool.join()

def get_json_files_list(json_files_arg):
    lst = []
    for f in json_files_arg:
        if f.endswith('.json'):
            lst.append(f)
        elif f.endswith('.txt'):
            # Read the text file, and append all the json files
            with open(f, 'r') as txt_data:
                lst.extend(txt_data.readlines())
    return lst


if __name__ == '__main__':

    # Command line parser
    parser = argparse.ArgumentParser(description='Converts pmcc json files to hdf5 format.')
    parser.add_argument('json_files', metavar='json_files', type=str, nargs='+', 
                        help='a list of files to convert, or a txt file that contains the list of files that need to be converted.')
    parser.add_argument('-o', '--output_dir', type=str, 
                        help='the directory where the hdf5 files will be stored (default: ./output)',
                        default='./output')
    parser.add_argument('-t', '--threads_num', type=int, 
                        help='the number of threads (acutally processes) that will be used to convert the files. Works only when there are multiple json files to be converted (default: 1)',
                        default=1)

    args = parser.parse_args() 

    processes_num = args.threads_num

    json_files = get_json_files_list(args.json_files)
    hdf5_dir = args.output_dir
    utils.create_dir(hdf5_dir)

    if len(json_files) == 1:
        convert_files(json_files, hdf5_dir)
    else:
        parallel_convert(processes_num, json_files, hdf5_dir)

