import sys
import os.path
import os

os.environ["THEANO_FLAGS"] = "floatX=float64"

from collections import defaultdict
import json
import theano
from theano import tensor as T
from theano.ifelse import ifelse
import glob
import progressbar
import numpy as np

def link_cost(v, winsor=10.0):
    a = (v ** 2) / 2
    b = winsor * (abs(v) - winsor / 2)
    return T.switch(T.lt(abs(v), winsor), a, b)

if __name__ == "__main__":

    imsize = 3000

    R1 = T.scalar()
    R2 = T.scalar()
    T1 = T.col()
    T2 = T.col()
    pts1 = T.matrix()
    pts2 = T.matrix()
    center1 = T.col()
    center2 = T.col()

    rot1 = T.as_tensor([T.cos(R1), -T.sin(R1),
                        T.sin(R1), T.cos(R1)]).reshape((2, 2))
    rot2 = T.as_tensor([T.cos(R2), -T.sin(R2),
                        T.sin(R2), T.cos(R2)]).reshape((2, 2))
    new_pts1 = T.dot(rot1, pts1 - center1) + T1 * imsize + center1
    new_pts2 = T.dot(rot2, pts2 - center2) + T2 * imsize + center2
    ptdists = T.sqrt(T.sum((new_pts1 - new_pts2) ** 2, axis=0) + 0.01)
    cost = T.sum(link_cost(ptdists))
    grads = T.grad(cost, wrt=[R1, R2, T1, T2])

    costgrad = theano.function([R1, R2, T1, T2, pts1, pts2, center1, center2],
                               outputs=[cost] + grads)

    distsfun = theano.function([R1, R2, T1, T2, pts1, pts2, center1, center2],
                               outputs=ptdists)

    rot_angles = defaultdict(lambda: 0.0)
    translations = defaultdict(lambda: np.zeros((2, 1)))


    all_matches = {}
    all_pts = defaultdict(list)

    match_files = glob.glob(os.path.join(sys.argv[1], '*sift_matches*00000[1-5]*00000[1-5]*.json'))
    pbar = progressbar.ProgressBar()
    for f in pbar(match_files):
        data = json.load(open(f))
        # point arrays are 2xN
        pts1 = np.array([c["p1"]["w"] for c in data[0]["correspondencePointPairs"]]).T
        pts2 = np.array([c["p2"]["w"] for c in data[0]["correspondencePointPairs"]]).T
        url1 = data[0]["url1"]
        url2 = data[0]["url2"]
        if len(pts1) == 0:
            continue
        dists = distsfun(rot_angles[url1], rot_angles[url2],
                         translations[url1], translations[url2],
                         pts1, pts2,
                         np.zeros((2, 1)), np.zeros((2, 1)))
        keep = dists <= np.median(dists) + 5
        if keep.size > 2:
            all_matches[url1, url2] = (pts1[:, keep], pts2[:, keep])
            all_pts[url1].append(pts1[:, keep])
            all_pts[url2].append(pts2[:, keep])

    centers = {k: np.mean(np.hstack(pts), axis=1, keepdims=True) for k, pts in all_pts.iteritems()}

    stepsize = 1e-10

    prev_c = np.inf
    for iter in range(20000):
        c = 0
        dists = []
        for (url1, url2), (pts1, pts2) in all_matches.iteritems():
            cg = costgrad(rot_angles[url1], rot_angles[url2],
                          translations[url1], translations[url2],
                          pts1, pts2,
                          centers[url1], centers[url2])
            d = distsfun(rot_angles[url1], rot_angles[url2],
                         translations[url1], translations[url2],
                         pts1, pts2,
                         centers[url1], centers[url2])
            dists.append(np.median(d))
            cg = [v / pts1.shape[1] for v in cg]
            c += cg[0]
            g = cg[1:]
            rot_angles[url1] -= stepsize * g[0]
            rot_angles[url2] -= stepsize * g[1]
            translations[url1] -= stepsize * g[2]
            translations[url2] -= stepsize * g[3]
        print "cost", c / len(all_matches), stepsize, np.median(dists)
        if c < prev_c:
            stepsize *= 1.05
        else:
            stepsize *= 0.5
        prev_c = c

