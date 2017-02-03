"""
Takes in a set of image files of glassware and fits a b-spline regression curve
to the main side curve of the object.

Usage: python glass.py

Author(s): Mark Fuge (fuge@umd.edu)
"""
import numpy as np
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
from uniform_bspline import UniformBSpline
from fit_uniform_bspline import UniformBSplineLeastSquaresOptimiser
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
#from matplotlib._png import read_png
#from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, \
#     AnnotationBbox
#from matplotlib.lines import Line2D
#import matplotlib.image as mpimg

#import pdb
#from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.mplot3d import proj3d
#from PIL import Image
import os
import glob
#import math
import ConfigParser

#from pylab import *

def iplot(image):
    ''' Plots a gray-scale image using Matplotlib '''
    fig = plt.figure()
    plt.imshow(image,'gray')
    fig.show()

def rescale_image(im,scale_height):
    h,w=im.shape[:2]
    scale_height = 1000.0
    scale = scale_height/h
    return cv2.resize(im,None,fx=scale, fy=scale)#, interpolation = cv2.INTER_CUBIC)

def blur_image(im):
    blur = cv2.blur(im,(5,5))
    #blur = cv2.medianBlur(im,5)
    #cv2.imshow('blur',blur)
    #iplot(blur)
    return blur

def threshold_image(im):
    # There are many options here.
    ret,th1 = cv2.threshold(im,2,255,cv2.THRESH_BINARY)
#    th2 = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#                cv2.THRESH_BINARY,11,2)
#    th3 = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#                cv2.THRESH_BINARY,5,2)
    #ret,thresh = cv2.threshold(im,127,255,0)
    return th1
    
def clean_thresholded_image(im):
    # Now that we have a thresholded image, we need to manipulate it a bit to
    # remove various irregularities in the image (pockets, weird edges, etc.)
    # This is done using Morphology operations. Currently the kernels are hand-
    # tuned. There may be some better way to do this in the future. This section
    # in the code is by far the most brittle, and has the biggest affect on the
    # quality of the results.
    skernel = np.ones((2,2),np.uint8) #small square kernel used for erosion
    # First erode the image a bit
    #erosion = cv2.erode(thresh, skernel,iterations = 1) #refines all edges in the binary image
    # Do a few iterations of opening and closing to smooth things out
    # There must be a better way to do this than how I'm doing it now, but this
    # seemed to be a quick way to get it to work well enough for my purposes
    opening = cv2.morphologyEx(im, cv2.MORPH_OPEN, skernel,iterations=3)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, skernel,iterations=3)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, skernel,iterations=3)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, skernel,iterations=3)
    # You can also try bigger kernels, but I had more success with the iterative
    # smaller kernels above.
    # bkernel = np.ones((10,10),np.uint8) #big square kernel used for erosion
    # opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, bkernel)
    # closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, bkernel)
    #cv2.imshow('thresh',thresh)
    #iplot(thresh)
    return closing

def get_shape_contour(im):   
    # Now we have to find the contours of the image. This is the money-maker
    # in that this gives us the ultimate points that we will use to fit the
    # spline regression. Hopefully, all the thresholding/morphology work above
    # should give us one giant coherant shape, and so the contour should be
    # straightforward, smooth, and should give us the shape we want.
    # NOTE: For some reason the curve points don't match the image exactly, and
    # so the offset parameter corrects that. Thus far it seems consistent across
    # images, and the offset appears to be a function of the morphology parameters
    # More validation and testing could figure out how to remove or better
    # compensate for this variation.
    image, contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE,
                                                  cv2.CHAIN_APPROX_SIMPLE,
                                                  offset=(-10,-15))
    # Now that we have the contours, find the one with the biggest area
    # This should be the overall shape.
    areas = [] #list to hold all areas
    for contour in contours:
        ar = cv2.contourArea(contour)
        areas.append(ar)
    max_area = max(areas)
    max_area_index = areas.index(max_area) #index of the list element with largest area
    biggest_contour = contours[max_area_index] #largest area contour
    # This function draws a useful image that we'll need later for debugging
    # So return it for now.
    # TODO: fix this so it's in some kind of "verbose" or "debug" tag, etc.
    img = cv2.drawContours(image, contours, max_area_index, (0,255,0), 3)
    #cv2.imshow('img',img)
    #iplot(img)
    return biggest_contour,img
    
def get_one_side_of_glass(contour, debug = False):
    ''' Returns just the points on one-side of the glass contour 
    
        Since a glass in symmetric, no need for both sides of the glass.
        Everything in here is highly specific to the glass example, and would
        need to be changed for other examples.
    '''
    MARKER = '+'
    # Get the contour points
    x=contour[:,0][:,0]
    y=contour[:,0][:,1]
    if debug:
        fig = plt.figure()
        plt.plot(x,y,MARKER)
        fig.show()
    # Get the bounding rectangle so we know how to cut it down the middle
    x1,y1,w2,h2=cv2.boundingRect(contour)
    wmid = w2/2.0+x1 # The middle axis
    # Remove all points on one half off the axis
    oneside=contour[:,0][:,0]<wmid
    x=x[oneside]-wmid # make the contour symmetric
    y=y[oneside]
    if debug:
        fig = plt.figure()
        plt.plot(x,y,MARKER)
        fig.show()
    return x,y,wmid
    
def trim_glass_contour_edges(x, y, debug = False):    
    ''' There will be some edges to the contour where the contour jumps from the
        top of the glass to the bottom of the glass. We need to trim these "top"
        and "bottom" edges, so that we're only getting the side profile
        the rogue end of the contour that jumps down
    '''
    # First, get the one end of the glass by figuring out where the bottom of the 
    # glass starts to curve back around the stem (this discards the circular
    # base of the glass).
    ymax = max(y)
    for ind in range(len(y)):
        if y[-ind] > ymax*.9:
            break
    y=y[:-ind]
    x=x[:-ind]
    edge_cut = np.percentile(x,99)
    edge=x<edge_cut
    x=x[edge]
    y=y[edge]
    if debug:
        fig = plt.figure()
        plt.plot(x,y,MARKER)
        fig.show()
        
    # Now trim the top of the glass by identifying when the glass goes from a 
    # straight top to a curved side. 
    cut_range = 120
    y_cut = 0
    for i in range(cut_range):
        q75, q25 = np.percentile(y[i:i+5], [75 ,25])
        if q75-q25>2:
            # Then the glass is starting to curve. Cut it around here.
            y_cut = i+5
            break
    y = y[y_cut:]
    x = x[y_cut:]
    x_cut = 1
    x_min = min(x[-cut_range:])
    for i in range(cut_range):
        if x[-i] == x_min:
            x_cut = i
            break
    x = x[0:-x_cut]
    y = y[0:-x_cut]
    return x,y

def resample_glass_side_contour(x,y, scale_height):
    ''' Resamples the contour so that the points are fairly consistent
        This helps make the regression better overall.
    '''
    # now resample x and y so that the points are consistent
    old_pnts = [list(a) for a in zip(x,y)]
    new_x = []
    new_y = []
    for i in range(len(old_pnts)-1):
        # Calculate the number of points in between, based on distance
        dist = int(cdist([old_pnts[i]],[old_pnts[i+1]])[0][0])
        if dist > scale_height/100.0: # This is image size dependent
            # If they are too far apart, resample so we have more points in this
            # region
            num_interp = 2+dist/5
            x_space = np.linspace(old_pnts[i][0],old_pnts[i+1][0],num_interp,endpoint=False)
            y_space = np.linspace(old_pnts[i][1],old_pnts[i+1][1],num_interp,endpoint=False)
            new_x.extend(x_space.tolist())
            new_y.extend(y_space.tolist())
        else:
            # otherwise, just keep points as they are.
            new_x.append(old_pnts[i][0])
            new_y.append(old_pnts[i][1])
    new_x.append(old_pnts[i][0])
    new_y.append(old_pnts[i][1])
    return new_x, new_y

def clean_glass_contour(contour,scale_height):
    ''' Takes the glass contour and cleans/trims/resamples it so that we
        can usefully use it for regression.
    '''
    x,y,xmid = get_one_side_of_glass(contour)
    x,y = trim_glass_contour_edges(x, y)
    x,y = resample_glass_side_contour(x, y, scale_height)
    return x,y,xmid
    
def fit_bspline(x, y, dim = 2, degree=2, num_control_points = 20,
                is_closed = False, num_init_points=1000):
    ''' Fits and returns a bspline curve to the given x and y points
    
        Parameters
        ----------
        x : list
            data x-coordinates
        y : list
            data y-coordinates
        dim : int
            the dimensionality of the dataset (default: 2)
        degree : int
            the degree of the b-spline polynomial (default: 2)
        num_control_points : int
            the number of b-spline control points (default: 20)
        is_closed : boolean
            should the b-spline be closed? (default: false)
        num_init_points : int
            number of initial points to use in the b-spline parameterization
            when starting the regression. (default: 1000)
        
        Returns
        -------
        c: a UniformBSpline object containing the optimized b-spline
    '''
    # TODO: extract dimensionality from the x,y dataset itself
    num_data_points = len(x)
    c = UniformBSpline(degree, num_control_points, dim, is_closed=is_closed)
    Y = np.c_[x, y] # Data num_points by dimension
    # Now we need weights for all of the data points
    w = np.empty((num_data_points, dim), dtype=float)
    # Currently, every point is equally important
    w.fill(1) # Uniform weight to the different points
    # Initialize `X` so that the uniform B-spline linearly interpolates between
    # the first and last noise-free data points.
    t = np.linspace(0.0, 1.0, num_control_points)[:, np.newaxis]
    X = Y[0] * (1 - t) + Y[-1] * t
    # NOTE: Not entirely sure if the next three lines are necessary or not
    m0, m1 = c.M(c.uniform_parameterisation(2), X)
    x01 = 0.5 * (X[0] + X[-1])
    X = (np.linalg.norm(Y[0] - Y[-1]) / np.linalg.norm(m1 - m0)) * (X - x01) + x01
    # Regularization weight on the control point distance
    # This specifies a penalty on having the b-spline control points close
    # together, and in some sense prevents over-fitting. Change this if the
    # curve doesn't capture the curve variation well or smoothly enough
    lambda_ = 0.5 
    # These parameters affect the regression solver.
    # Presently, they are disabled below, but you can think about enabling them
    # if that would be useful for your use case.
#    max_num_iterations = 1000
#    min_radius = 0
#    max_radius = 400
#    initial_radius = 100
    # Initialize U
    u0 = c.uniform_parameterisation(num_init_points)
    D = cdist(Y, c.M(u0, X))
    u = u0[D.argmin(axis=1)]
    # Run the solver
    (u, X, has_converged, states, num_iterations, 
        time_taken) = UniformBSplineLeastSquaresOptimiser(c,'lm').minimise(
        Y, w, lambda_, u, X,
        #max_num_iterations = max_num_iterations,
        #min_radius = min_radius,
        #max_radius = max_radius,
        #initial_radius = initial_radius,
        return_all=True)
    return c,u0,X
    
def process_image(imfile, n_control_points, n_points, im_save_path=None):
    ''' Reads in an image filename and fits a b-spline to its silhouette '''
    im = cv2.imread(imfile)
    
    # Scale the image to similar sizes
    # This helps to make the OpenCV operations consistent across different
    # images, otherwise the kernel sizes create different results
    scale_height = 1000.0
    im = rescale_image(im,scale_height = scale_height)
    
    # Convert to grayscale and invert to help with the morphology operations
    # TODO: handle PNG files with sub-tracted backgrounds properly.
    imgray = 255-cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('im',imgray)
    #plt.imshow(imgray,'gray')
    
    # Blur the image a bit to remove fine-grained details that could mess up
    # thresholding or edge detection. We just need the general structure.
    # NOTE: Disabled blurring for now, since that didn't seem to help much.
    #blur = blur_image(im)
    blur = imgray
    
    # Threshold the image. This creates a sharper cutoff of edges/bodies that
    # we can use to identify the edge of the shape.
    thresh = threshold_image(blur)
    
    # Cleanup the thresholded image to remove irregularities and get the 
    # overall shape
    thresh = clean_thresholded_image(thresh)
    
    # Get the overall shape contour
    contour,img = get_shape_contour(thresh)

    # Now we have a raw contour, but need just the side-part of the contour
    # relevant to the glass profile.
    x,y,xmid = clean_glass_contour(contour,scale_height)
    offset = y[0]
    y -= offset # make min(y)=0
    # Scale the shapes so that they have the same height
    mn = min(y)
    mx = max(y)
    h = mx-mn
    y /= h
    x /= h
    
    # Now do the spline fitting
    bspline,u0,x_plot = fit_bspline(x, y, degree=2, num_control_points = n_control_points)
    
    indices = range(0, len(u0), len(u0)/n_points)
    xy = bspline.M(u0, x_plot).tolist()
    xy = np.array(xy)[indices]
    
    if im_save_path is not None:
        # Plot the results
        img_h = img.shape[0]
        img_w = img.shape[1]
#        f = plt.figure(figsize=(10,10))
        ax1 = plt.gca()
        ax1.imshow(im,'gray',extent=[-xmid/h,(img_w-xmid)/h,(img_h-offset)/h,-offset/h])
        ax1.set_title('Image + Contour')
#        ax1.plot( *zip(*bspline.M(u0, x_plot).tolist()),linewidth =2)#, c=u0, cmap="jet", alpha=0.5 )
#        ax1.plot(*zip(*x_plot), marker="o", alpha=0.3)
        ax1.plot(xy[:,0], xy[:,1], marker="o", alpha=0.3)
        ax1.set_ylim([-offset/h,(img_h-offset)/h])
        ax1.set_xlim([-xmid/h,(img_w-xmid)/h])
        # Since images render upside down
        ax1.invert_yaxis()
        ax1.set_autoscale_on(False)
        ax1.set_title(imfile + ' n=' + str(n_points))
        #f.show()
    
        plt.savefig(im_save_path, dpi=300)
        plt.close()
    
#    return x_plot, u0, bspline
    return xy

def create_dir(path):
    if os.path.isdir(path): 
        pass 
    else: 
        os.mkdir(path)
   

if __name__ == "__main__":

    config = ConfigParser.ConfigParser()
    config.read('config.ini')
    
    IMAGE_DIR = config.get('Global', 'SOURCE_DIR')
    image_paths = glob.glob(IMAGE_DIR+"glass/*.*")
    
    RESULTS_DIR = config.get('Global', 'RESULTS_DIR')
    create_dir(RESULTS_DIR)
            
    save_dir0 = RESULTS_DIR + 'glass/'
    create_dir(save_dir0)
    
    save_dir1 = save_dir0 + 'n_samples=' + str(len(image_paths)) + '/'
    create_dir(save_dir1)
    
    n_control_points = config.getint('Glass', 'n_control_points')
    n_points = config.getint('Global', 'n_points')
    save_dir2 = save_dir1 + 'n_points=' + str(n_points) + '/'
    create_dir(save_dir2)
    
    image_save_dir = save_dir2 + '/im/'
    create_dir(image_save_dir)
    
    for image_path in image_paths:
        image_save_path = image_save_dir+os.path.splitext(os.path.basename(image_path))[0]+'.png'
#        try:
        process_image(image_path, n_control_points, n_points, image_save_path)
        print 'Processing: ' + os.path.basename(image_path)
#        except:
#            print "For "+image_path
#            print "Unexpected error:", sys.exc_info()[0]

