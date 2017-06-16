import numpy as np
import cv2
import time
import math
import os

def get_default_mask(image):
    height, width = image.shape
    y_min = int(0.35 * height)
    y_max = int(0.95 * height)
    x_min = int(0.2 * width)
    x_max = int(0.8 * width)

    return np.array([[0, y_max],
                     [x_min, y_min],
                     [x_max, y_min],
                     [width, y_max]])


def process_image(frame, polygon=None, parameters=dict()):
    if frame is None:
        return None

    img = None
    if len(frame.shape) > 2:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        img = frame

    # apply blur so edge detection works better
    img = cv2.blur(img, (5,5))

    # do edge detection with a canny filter
    canny_low = parameters.get('canny.low', 50)
    canny_high = parameters.get('canny.high', 200)
    edges = cv2.Canny(img, canny_low, canny_high)

    # mask the polygon that we want to
    polygon = parameters.get('mask', get_default_mask(img))
    edges = mask(edges, polygon)

    # apply hough filter
    hough_rho = parameters.get('hough.rho',  1)
    hough_theta = parameters.get('hough.theta',  np.pi/180)
    hough_threshold = parameters.get('hough.threshold',  80)
    hough_min_len = parameters.get('hough.min_len',  30)
    hough_max_gap = parameters.get('hough.max_gap',  60)
    lines = cv2.HoughLinesP(edges, hough_rho, hough_theta, hough_threshold,
                            np.array([]), hough_min_len, hough_max_gap)

    # create new image for lines and paint lines onto it
    line_img = np.zeros((*img.shape , 3), dtype=np.uint8)
    draw_lines(line_img, lines)

    # return combined image
    if len(frame.shape) <= 2:
        frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)

    return cv2.addWeighted(frame, 0.6, line_img, 1., 0.)


def draw_lines(img, lines, color1=[255, 0, 0], color2=[0, 0, 255], thickness=2):
    """
    Draws lines ontop of the provided image
    """
    if lines is None:
        print("No lines detected...")
        return

    for line in lines:
        for l in line:
            (x1, y1, x2, y2) = l

            if (y2 - y1) != 0:
                slope = (x2-x1)/(y2-y1)
                # filter all lines that run almost horizontal
                if abs(slope) > 5:
                    pass
                # mark descending slope(/) with color1
                elif slope < 0:
                    cv2.line(img, (x1, y1), (x2, y2), color1, 4)
                # mark ascending slope(\) with color2
                else:
                    cv2.line(img, (x1, y1), (x2, y2), color2, 4)



def mask(img, polygon):
    # create black mask
    roi = np.zeros_like(img)
    # make polygon white
    cv2.fillConvexPoly(roi, np.array(polygon), [255, 255, 255])
    # apply mask
    return cv2.bitwise_and(img, roi)

def main():
    cap = cv2.VideoCapture('challenge.mp4')
    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            break

        print(frame.shape)
        height, width, depth = frame.shape

        p = {
            'mask' : [
                [0, height],
                [int(0.33*width), height],
                [int(0.4*width), int(0.55*height)],
                [int(0.6*width), int(0.55*height)],
                [int(0.66*width), height],
                [width, height],
                [width, int(0.8*height)],
                [int(0.55*width), int(0.3*height)],
                [int(0.45*width), int(0.3*height)],
                [0, int(0.8*height)]
            ]
        }
        img = process_image(frame, parameters=p)
        cv2.imshow('frame', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

def main2():
    height, width = (1024,1280)
    p = {
        'mask' : [
            [0, height],
            [width, height],
            [width, int(0.6*height)],
            [int(0.55*width), int(0.3*height)],
            [int(0.45*width), int(0.3*height)],
            [0, int(0.6*height)]
        ],
    }

    for i in range(500, 1000):
        fname = os.getcwd() + '/images/' + str(format(i, '04d')) + '.pgm'
        frame = cv2.imread(fname, 0)

        img = np.zeros_like(frame)
        cv2.fillConvexPoly(img, np.array(p['mask']), [255, 255, 255])
        cv2.imshow('mask', img)

        img = process_image(frame, parameters=p)
        cv2.imshow('frame', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


main()
