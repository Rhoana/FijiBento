import numpy as np
import imread
import scipy.spatial as ssp
import cPickle

from OpenGL.GL import *
from OpenGL.GL.exceptional import glGenTextures
from OpenGL.GLUT import *


w = h = None

def load_texture_from_png(filename):
    # load the image, check that it's grayscale 8 bit
    img = imread.imread(filename)
    assert img.dtype == np.uint8
    assert len(img.shape) == 2

    # pad with 0
    alpha = np.ones_like(img) * 255
    ori = img = np.pad(img, 1, mode='constant')
    alpha = np.pad(alpha, 1, mode='constant')
    img = np.dstack((img, alpha))
    print img.shape, img.flags

    texture = glGenTextures(1)
    print "here"
    glBindTexture(GL_TEXTURE_2D, texture)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    print "foo"
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                 ori.shape[1], ori.shape[0],
                 0, GL_LUMINANCE,
                 GL_UNSIGNED_BYTE, ori)
    print "t"
    return texture

def render_tris(dest_coords, src_coords, texture):
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_TEXTURE_COORD_ARRAY)
    glVertexPointer(2, GL_FLOAT, 0, dest_coords.astype(np.float32))
    glTexCoordPointer(2, GL_FLOAT, 0, src_coords.astype(np.float32))
    glDrawArrays(GL_TRIANGLES, 0, src_coords.shape[0])


def reshape(width, height):
    global w, h
    w = width
    h = height
    glutPostRedisplay()


def draw(tri_pts, txtr_pts, the_texture):
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glClearColor(0, 255, 0, 0)
    glClear(GL_COLOR_BUFFER_BIT)
    glEnable(GL_TEXTURE_2D)
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE )
    render_tris(tri_pts, txtr_pts, the_texture)

    glutSwapBuffers()


def setup(tri_pts, txtr_pts, texture_file):
    glutInitDisplayMode(GLUT_RGBA)
    glutInitWindowSize(256, 256)

    glutCreateWindow("test")
    the_texture = load_texture_from_png(texture_file)

    glutReshapeFunc(reshape)
    glutDisplayFunc(lambda: draw(tri_pts, txtr_pts, the_texture))
    glutKeyboardFunc(lambda x,y,k: sys.exit(0))

    glutMainLoop()


if __name__ == '__main__':
    glutInit(sys.argv)

    texture_file = sys.argv[1]
    name, data = cPickle.load(open(sys.argv[2]))
    print texture_file
    print name

    orig_locations = np.array(data[0])
    end_locations = np.array(data[1])
    end_locations[:, 1] *= -1

    # center and scale
    end_locations -= np.mean(end_locations, axis=0)
    end_locations /= 1.1 * abs(end_locations).max()

    orig_locations /= orig_locations.max()

    tri = ssp.Delaunay(orig_locations)
    print "TRIS", tri.simplices.shape
    triangle_pts = end_locations[tri.simplices].reshape((-1, 2))
    texture_pts = orig_locations[tri.simplices].reshape((-1, 2))

    setup(triangle_pts, texture_pts, texture_file)
