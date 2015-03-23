import argparse
import os
import sys
import glob
#import utils
import cv2
import numpy as np
import bisect
import math


def load_fiducials_from_file(fname):
    #img = cv2.imread(fname, cv2.CV_LOAD_IMAGE_COLOR)
    img = cv2.imread(fname, cv2.CV_LOAD_IMAGE_GRAYSCALE)


    non_zero = np.nonzero(img)
    non_zero = np.array([non_zero[0], non_zero[1], img[non_zero]]) # 3 arrays: x,y,value

    # For each color, find the center of the fiducial
    centers = [[], [], []]
    for val in sorted(set(non_zero[2])):
        val_indices = [non_zero[2] == val]
        min_x = min(non_zero[1][val_indices])
        min_y = min(non_zero[0][val_indices])
        max_x = max(non_zero[1][val_indices])
        max_y = max(non_zero[0][val_indices])

        # append an array
        centers[0].append((max_x - min_x) / 2 + min_x)
        centers[1].append((max_y - min_y) / 2 + min_y)
        centers[2].append(val)
        center = np.array([(max_x - min_x) / 2 + min_x, (max_y - min_y) / 2 + min_y])
        print "val: {}, center: {}".format(val, center)

    # return an array: [array(x), array(y), array(val)]

    #return np.array([np.array(centers[0]), np.array(centers[1]), np.array(centers[2])])
    return centers

def match_centers(centers1, centers2):
    sources = []
    targets = []
    for i in range(len(centers1[2])):
        # Find the position where to insert the value of centers1[2][i] in centers2[2]
        pos = bisect.bisect_left(centers2[2], centers1[2][i])
        if pos < len(centers2[2]) and centers1[2][i] == centers2[2][pos]:
            # Found a match on the column 2
            sources.append(np.array([centers1[0][i], centers1[1][i]]))
            targets.append(np.array([centers2[0][pos], centers2[1][pos]]))

    return sources, targets

def Haffine_from_points(fp, tp):
    """ Find H, affine transformation, such that
        tp is affine transf of fp. """
    """ Taken from http://programmingcomputervision.com/downloads/ProgrammingComputerVision_CCdraft.pdf (page 77) """
    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')
    # condition points

    # --from points--
    m = np.mean(fp[:2], axis=1)
    maxstd = max(np.std(fp[:2], axis=1)) + 1e-9
    C1 = np.diag([1/maxstd, 1/maxstd, 1])
    C1[0][2] = -m[0]/maxstd
    C1[1][2] = -m[1]/maxstd
    fp_cond = np.dot(C1, fp)

      # --to points--
    m = np.mean(tp[:2], axis=1)
    C2 = C1.copy() #must use same scaling for both point sets C2[0][2] = -m[0]/maxstd
    C2[1][2] = -m[1]/maxstd
    tp_cond = np.dot(C2, tp)
      # conditioned points have mean zero, so translation is zero
    A = np.concatenate((fp_cond[:2], tp_cond[:2]), axis=0)
    U,S,V = np.linalg.svd(A.T)
    # create B and C matrices as Hartley-Zisserman (2:nd ed) p 130.
    tmp = V[:2].T
    B = tmp[:2]
    C = tmp[2:4]
    tmp2 = np.concatenate((np.dot(C, np.linalg.pinv(B)), np.zeros((2, 1))), axis=1)
    H = np.vstack((tmp2, [0, 0, 1]))
    # decondition
    H = np.dot(np.linalg.inv(C2), np.dot(H,C1))
    return H / H[2, 2]


def compute_rotation(pts1, pts2):
    sources, targets = match_centers(pts1, pts2)
    source = np.array([sources])
    target = np.array([targets])
    mat = cv2.estimateRigidTransform(source, target, False)
    return np.array([mat[0], mat[1], np.array([0., 0., 1.])])


def compute_bbox(rotations, width, height):
    # take all the corners of the images, apply the transformations,
    # and save the minimal and maximal x, y positions
    bbox = np.array([0., width - 1., 0., height - 1.])
    corners = [ np.array([0., 0., 1.]),
                np.array([width - 1., 0., 1.]),
                np.array([0., height - 1., 1.]),
                np.array([width - 1., height - 1., 1.]) ]
    curr_rot = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    for rot in rotations:
        curr_rot = np.dot(rot, curr_rot)
        corners = [ np.dot(curr_rot, corner) for corner in corners ]
        bbox[0] = min(bbox[0], min([xy[0] for xy in corners])) # min_x
        bbox[1] = max(bbox[0], max([xy[0] for xy in corners])) # max_x
        bbox[2] = min(bbox[1], min([xy[1] for xy in corners])) # min_y
        bbox[3] = max(bbox[1], max([xy[1] for xy in corners])) # max_y

    return bbox

def adjust_rotations(rotations, shift_xy):
    """Adjust (shift) rotations according the a given x and y values"""
    print "Adjusting (shifting) rotations by: {}".format(shift_xy)
    adjust_matrix = np.array([[0., 0., float(shift_xy[0])], [0., 0., float(shift_xy[1])], [0., 0., 0.]], dtype=np.float32)
    adjusted_rotations = [ r + adjust_matrix for r in rotations ]
    return adjusted_rotations


def rotate_image(img, rot, out_shape):
    """Rotate a single image matrix by a given rotation"""
    #out_img = np.zeros((out_shape), type=np.uint8)
    cv2_rot = np.array([rot[0], rot[1]], dtype=np.float32)
    print cv2_rot
    print out_shape
    return cv2.warpAffine(img, cv2_rot, out_shape)

def rotate_images(img_files, rotations, out_shape, output_dir):
    curr_rot = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    for i in range(len(img_files)):
        # load image file
        im = cv2.imread(img_files[i], cv2.CV_LOAD_IMAGE_GRAYSCALE)
        # rotate the image
        curr_rot = np.dot(rotations[i], curr_rot)
        im_rot = rotate_image(im, curr_rot, out_shape)
        # save the rotated image
        out_fname = os.path.join(output_dir, os.path.basename(img_files[i]))
        print "Writing output image to {}".format(out_fname)
        cv2.imwrite(out_fname, im_rot)


def correct_3d_drift_from_dirs(images_dir, fiducials_dir, output_dir):
    # load fiducials
    fiducials_files = glob.glob(os.path.join(fiducials_dir, '*'))
    fiducials_files.sort()
    fiducials_files = fiducials_files[:3]

    # compute adjacent sections rotation
    adj_rotations = []
    print "Computing rotation of {} sections.".format(len(fiducials_files))
    # load fiducials of the first section
    prev_fids = load_fiducials_from_file(fiducials_files[0])
    for i in range(1, len(fiducials_files)):
        print "Computing rotation between {} and {}.".format(fiducials_files[i - 1], fiducials_files[i])
        # load fiducials of the adjacent section (to prev_fids)
        curr_fids = load_fiducials_from_file(fiducials_files[i])

        # compute the rotation
        #rot = compute_rotation(prev_fids, curr_fids)
        rot = compute_rotation(curr_fids, prev_fids)
        adj_rotations.append(rot)
        prev_fids = curr_fids

    # load the image files names
    img_files = glob.glob(os.path.join(images_dir, '*'))
    img_files.sort()
    img_files = img_files[:3]

    assert(len(img_files) == len(fiducials_files))
    assert(len(img_files) - 1 == len(adj_rotations))

    # compute minimal x and y positions (after rotations)
    im1 = cv2.imread(img_files[0], cv2.CV_LOAD_IMAGE_GRAYSCALE)
    bbox = compute_bbox(adj_rotations, im1.shape[1], im1.shape[0]) # bbox: [left, right, top, bottom]
    print "bbox: {}".format(bbox)

    # adjust rotations according to min_xy
    shift_xy = [-bbox[0], -bbox[2]]
    adj_rotations.insert(0, np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])) # insert the identity rotation for the first image
    adj_rotations = adjust_rotations(adj_rotations, shift_xy)

    # adjust bbox according to the shift
    bbox[0] = math.floor(bbox[0] + shift_xy[0])
    bbox[1] = math.ceil(bbox[1] + shift_xy[0])
    bbox[2] = math.floor(bbox[2] + shift_xy[1])
    bbox[3] = math.ceil(bbox[3] + shift_xy[1])
    print "after shift bbox: {}".format(bbox)

    # # apply the rotations on each image
    rotate_images(img_files, adj_rotations, (int(bbox[3] - bbox[2] + 1), int(bbox[1] - bbox[0] + 1)), output_dir)


def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Corrects a drift in 3d images according to given fiducials.')
    parser.add_argument('images_dir', metavar='images_dir', type=str,
                        help='the directory of the images')
    parser.add_argument('fiducials_dir', metavar='fiducials_dir', type=str,
                        help='the directory that contains the images with the fiducials')
    parser.add_argument('-o', '--output_dir', type=str, 
                        help='an output directory (default: ./output_dir)',
                        default='./output_dir')


    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    correct_3d_drift_from_dirs(args.images_dir, args.fiducials_dir, \
        args.output_dir)

if __name__ == '__main__':
    main()

