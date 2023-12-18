#!/usr/bin/env python

import warnings

import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import util


class particlefilter:
    def __init__(self, count, start, posrange, angrange, 
            polemeans, polevar, number_cl, T_w_o=np.identity(4)):
        self.p_min = 0.01
        self.d_max = np.sqrt(-2.0 * polevar * np.log(
            np.sqrt(2.0 * np.pi * polevar) * self.p_min))
        self.minneff = 0.5
        self.estimatetype = 'best'
        self.count = count
        ############### particle initialization 
        r = np.random.uniform(low=0.0, high=posrange, size=[self.count, 1])
        angle = np.random.uniform(low=-np.pi, high=np.pi, size=[self.count, 1])
        xy = r * np.hstack([np.cos(angle), np.sin(angle)])
        dxyp = np.hstack([xy, np.random.uniform(
            low=-angrange, high=angrange, size=[self.count, 1])])
        self.particles = np.matmul(start, util.xyp2ht(dxyp))
        self.weights = np.full(self.count, 1.0 / self.count)
        self.weights1 = np.full(number_cl, 1.0 / number_cl)
        self.polemeans = polemeans
        self.poledist = scipy.stats.norm(loc=0.0, scale=np.sqrt(polevar))
        self.kdtree = scipy.spatial.cKDTree(polemeans[:, :2], leafsize=3)
        self.T_w_o = T_w_o
        self.T_o_w = util.invert_ht(self.T_w_o)
        self.cl = number_cl

    @property
    def neff(self):
        return 1.0 / (np.sum(self.weights**2.0) * self.count)
    
    def cluster_motion_points(self, X, cl):
        kmeans = KMeans(n_clusters=cl, random_state=0).fit(X)
        return kmeans.cluster_centers_

    def update_motion(self, mean, cov):
        T_r0_r1 = util.xyp2ht(
            np.random.multivariate_normal(mean, cov, self.count))
        self.particles = np.matmul(self.particles, T_r0_r1)
        ele = util.ht2xyp(self.particles)
        bb = self.cluster_motion_points(ele, self.cl)
        return bb

    def update_measurement(self, poleparams, segma_point, resample=True):
        n = poleparams.shape[0]
        polepos_r = np.hstack(
            [poleparams[:, :2], np.zeros([n, 1]), np.ones([n, 1])]).T
        
        #for i in range(self.count):
        for i in range(len(segma_point)):
            polepos_w = segma_point[i].dot(polepos_r)
            d, _ = self.kdtree.query(
                polepos_w[:2].T, k=1, distance_upper_bound=self.d_max)
            self.weights1[i] *= np.prod(
                self.poledist.pdf(np.clip(d, 0.0, self.d_max)) + 0.1)
        self.weights1 /= np.sum(self.weights1)
        if resample and self.neff < self.minneff:
            self.resample()

    def estimate_pose1(self):
        if self.estimatetype == 'mean':
            xyp = util.ht2xyp(np.matmul(self.T_o_w, self.particles))
            mean = np.hstack(
                [np.average(xyp[:, :2], axis=0, weights=self.weights),
                    util.average_angles(xyp[:, 2], weights=self.weights)])
            return self.T_w_o.dot(util.xyp2ht(mean))
        if self.estimatetype == 'max':
            return self.particles[np.argmax(self.weights)]
        if self.estimatetype == 'best':
            i = np.argsort(self.weights)[-int(0.1 * self.count):]
            xyp = util.ht2xyp(np.matmul(self.T_o_w, self.particles[i]))
            mean = np.hstack(
                [np.average(xyp[:, :2], axis=0, weights=self.weights[i]),
                    util.average_angles(xyp[:, 2], weights=self.weights[i])])                
            return self.T_w_o.dot(util.xyp2ht(mean))
        
    def estimate_pose(self,segma_point):
        if self.estimatetype == 'mean':
            xyp = util.ht2xyp(np.matmul(self.T_o_w, segma_point))
            mean = np.hstack(
                [np.average(xyp[:, :2], axis=0, weights=self.weights1),
                    util.average_angles(xyp[:, 2], weights=self.weights1)])
            return self.T_w_o.dot(util.xyp2ht(mean))
        if self.estimatetype == 'max':
            return segma_point[np.argmax(self.weights1)]
        if self.estimatetype == 'best':
            i = np.argsort(self.weights1)[:10]
            xyp = util.ht2xyp(np.matmul(self.T_o_w, segma_point[i]))
            mean = np.hstack(
                [np.average(xyp[:, :2], axis=0, weights=self.weights1[i]),
                    util.average_angles(xyp[:, 2], weights=self.weights1[i])])                
            return self.T_w_o.dot(util.xyp2ht(mean))

    def resample(self):
        cumsum = np.cumsum(self.weights)
        pos = np.random.rand() / self.count
        idx = np.empty(self.count, dtype=np.int)
        ics = 0
        for i in range(self.count):
            while cumsum[ics] < pos:
                ics += 1
            idx[i] = ics
            pos += 1.0 / self.count
        self.particles = self.particles[idx]
        self.weights[:] = 1.0 / self.count

    
