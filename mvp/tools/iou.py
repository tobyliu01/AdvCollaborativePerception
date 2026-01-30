# 3D IoU caculate code for 3D object detection 
# Kent 2018/12

import numpy as np
from scipy.spatial import ConvexHull
from numpy import *


def polygon_clip(subjectPolygon, clipPolygon):
   """ Clip a polygon with another polygon.
   Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python
   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**
   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
      if len(outputList) == 0:
          return None
   return(outputList)

def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0  

def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

def is_clockwise(p):
    x = p[:,0]
    y = p[:,1]
    return np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)) > 0

def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.
    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU
    todo (kent): add more description on corner points' orders.
    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)] 
    
    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
   
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[0,1], corners2[0,1])
    ymin = max(corners1[4,1], corners2[4,1])

    inter_vol = inter_area * max(0.0, ymax-ymin)
    
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d

# ----------------------------------
# Helper functions for evaluation
# ----------------------------------

def get_3d_box(box_size, heading_angle, center):
    ''' Calculate 3D bounding box corners from its parameterization.
    Input:
        box_size: tuple of (length,wide,height)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''
    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])

    R = roty(heading_angle)
    l,w,h = box_size
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + center[0];
    corners_3d[1,:] = corners_3d[1,:] + center[1];
    corners_3d[2,:] = corners_3d[2,:] + center[2];
    corners_3d = np.transpose(corners_3d)
    return corners_3d


def iou3d(bbox1, bbox2):
    bbox1 = get_3d_box(box_size=(bbox1[3], bbox1[4], bbox1[5]), 
                       heading_angle=bbox1[6],
                       center=(bbox1[0], bbox1[1], bbox1[2] + 0.5 * bbox1[5]))
    bbox2 = get_3d_box(box_size=(bbox2[3], bbox2[4], bbox2[5]), 
                       heading_angle=bbox2[6],
                       center=(bbox2[0], bbox2[1], bbox2[2] + 0.5 * bbox2[5]))
    iou, _ = box3d_iou(bbox1, bbox2)
    return iou


def iou2d(bbox1, bbox2):
    def to_oriented_bbox(bbox):
        """
        Accept:
          - (x, y, w, h) top-left axis-aligned
          - (x, y, w, l, yaw) center-based 2D
          - (x, y, z, l, w, h, yaw) fake 3D
        Return (cx, cy, length, width, yaw) with yaw in radians.
        """
        bbox = np.asarray(bbox).reshape(-1)
        if bbox.shape[0] == 4:
            x, y, w, h = bbox
            return x + w / 2.0, y + h / 2.0, w, h, 0.0
        if bbox.shape[0] == 5:
            x, y, w, l, yaw = bbox
            return x, y, l, w, yaw
        if bbox.shape[0] == 7:
            x, y, l, w, yaw = bbox[0], bbox[1], bbox[3], bbox[4], bbox[6]
            return x, y, l, w, yaw
        raise ValueError("Unsupported bbox format for iou2d")

    def to_corners(cx, cy, l, w, yaw):
        # Counter-clockwise corners
        dx = l / 2.0
        dy = w / 2.0
        corners = np.array([[ dx,  dy],
                            [-dx,  dy],
                            [-dx, -dy],
                            [ dx, -dy]])
        rot = np.array([[np.cos(yaw), -np.sin(yaw)],
                        [np.sin(yaw),  np.cos(yaw)]])
        return (rot @ corners.T).T + np.array([cx, cy])

    c1 = to_oriented_bbox(bbox1)
    c2 = to_oriented_bbox(bbox2)
    rect1 = to_corners(*c1).tolist()
    rect2 = to_corners(*c2).tolist()
    area1 = poly_area(np.array(rect1)[:, 0], np.array(rect1)[:, 1])
    area2 = poly_area(np.array(rect2)[:, 0], np.array(rect2)[:, 1])
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    if inter is None or inter_area <= 0:
        return 0.0
    iou = inter_area / float(area1 + area2 - inter_area)
    return iou


def test_iou2d_oriented():
    b1 = np.array([0.0, 0.0, 2.0, 4.0, 0.0])  # center-based (x,y,w,l,yaw)
    b2 = np.array([0.0, 0.0, 2.0, 4.0, 0.0])
    assert abs(iou2d(b1, b2) - 1.0) < 1e-6

    b3 = np.array([4.0, 0.0, 2.0, 4.0, 0.0])
    assert abs(iou2d(b1, b3)) < 1e-6

    b4 = np.array([0.0, 0.0, 2.0, 4.0, np.pi / 2.0])
    assert abs(iou2d(b1, b4) - 1/3) < 1e-6

    b5 = np.array([2.0, 0.0, 2.0, 4.0, 0.0])
    assert abs(iou2d(b1, b5) - 1/3) < 1e-6

    b6 = np.array([0.0, 0.0, 1.0, 2.0, 0.0])
    assert abs(iou2d(b1, b6) - 1/4) < 1e-6


if __name__ == "__main__":
    test_iou2d_oriented()
    print("test_iou2d_oriented passed")
