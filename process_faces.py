"""
Sources:
https://realpython.com/blog/python/face-recognition-with-python/
http://docs.opencv.org/master/d7/d8b/tutorial_py_face_detection.html#gsc.tab=0
http://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html#gsc.tab=0
"""

import cv2
import numpy as np
import scipy as sp

import scipy.cluster.hierarchy as hier
import sklearn.decomposition as deco
import scipy.cluster.vq as vq

import pylab as plt

import random

def scrape_SIFTdescs(videos):
    """

    Inputs-
    list of strings, each naming an mp4 in the working directory.

    Returns-
    list of SIFT descriptor matrices where each entry corresponds to one frame.
    """
    face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt.xml")

    sift = cv2.SIFT(nfeatures = 25)
    desc_list = []
    face_list = []
    for video in videos:
        ##Important stuff we may need later.
        cap = cv2.VideoCapture(video)
        #length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        #width  = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        #height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        #fps    = cap.get(cv2.cv.CV_CAP_PROP_FPS)

        while(True):
            ret, frame = cap.read()
            img = frame

            if frame is None:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = gray

            ##Returns the x, y, and height/width of detected faces in each frame.
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.4,
                minNeighbors=1,
                minSize=(5, 5),
                flags = cv2.cv.CV_HAAR_SCALE_IMAGE
            )

            ##If multiple faces found in a frame, pick one closest to previous face.
            if len(faces) > 1:
                dists = abs(faces[:, 0:2] - [old_x, old_y])
                (x, y, w, h) = faces[np.argmin(np.linalg.norm(dists, axis= 1))]
            elif len(faces) == 0:
                continue
            else:
                (x, y, w, h) = faces[0]

            w = 57
            h = 57
            old_x = x
            old_y = y

            #face = img[y:y+h, x:x+w, :]
            face = img[y:y+h, x:x+w]

            kps, descs = sift.detectAndCompute(face, None) #Return keypoints and sift descriptors.

            ##Draw keypoints.
            #sift_face=cv2.drawKeypoints(face,kps,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            for ix, kp in enumerate(kps):
                cv2.rectangle(face, (int(kp.pt[0]), int(kp.pt[1])), (int(kp.pt[0])+1, int(kp.pt[1])+1),\
                      (255, 255, 255), 2)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            face_list.append(face)
            desc_list.append(descs)

            cv2.imshow('frame', img)
            #cv2.imshow('face', face_list[-1])

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    return desc_list

def quantize_SIFTdescs(descs_list, codebook):
    """
    Given a list of matrices of SIFT descriptors (where the rows of each matrix are the descriptors
    found for a given frame), assign each descriptor to it's codebook cluster. Returns an observation matrix
    whose rows are single frames.

    Inputs-
    desc_lists: List of np matrices, where each matrix a frame's worth of SIFT descriptors.
    codebook: List of clusters.

    Returns-
    A: n x m np matrix with n instances (frames) and m features (centroid histogram)
    """
    n = codebook.shape[0]
    for D in descs_list:
        assignments = vq.vq(D, codebook) #Assigns each descriptor to a cluster.
        hist, bedges = np.histogram(assignments[0], bins= n) #Counts occurances of clusters in a frame.
        try:
            A = np.row_stack((A, hist))
        except NameError:
            A = hist

    return A

def pcaCode():
    ##Question: PCA descriptors, or PCA final profiles?
    #Principal Component Analysis
    pca = deco.PCA(n_components = 10)
    Xp = pca.fit_transform(X)

    #Z = hier.linkage(X)
    Y = hier.fclusterdata(X, 1.15)
    print "Num. Clusters (no PCA): %s"%max(Y)

    Yp = hier.fclusterdata(Xp, 1.15)
    print "Num. Clusters (with PCA): %s"%max(Yp)

videos = ["signing2.mp4", "signing3.mp4"]
descs_by_frames = scrape_SIFTdescs(videos)

##Convert descs from all videos, frames into single observation matrix.
for descs in descs_by_frames:
    try:
        A = np.row_stack((A, descs))
    except NameError:
        A = descs

##NOTE: kmeans requires number of clusters specified a prior.
##Might first try fcluster to estimate intrinsic number of clusters.
#C = hier.fclusterdata(A, 1.15)
#codebook, distort = vq.kmeans(vq.whiten(A), max(C))
codebook, distort = vq.kmeans(vq.whiten(A), 15)

test_frames = scrape_SIFTdescs(["signing.mp4"])
X = quantize_SIFTdescs(test_frames, codebook)

pca = deco.PCA(n_components = X.shape[1])
pca.fit(X)

##NOTE: Ensuring #sift detectors per frame > #codebook sharpens scree plot.
##Downside: Throwing out too many details?
plt.plot(pca.explained_variance_ratio_) #Plot to ensure all the features are actually getting used.

halt = True