# distutils: language = "c++"

import numpy as np

from libc.string cimport memcpy
from libc.stdlib cimport malloc

is_compiled_with_opencv = bool(USE_CV)

cdef class Image:
    cdef image img;
    def __cinit__(self, np.ndarray ary):
        IF USE_CV == 1:
            # Code adapted from https://github.com/solivr/cython_opencvMat
            assert ary.ndim==3 and ary.shape[2]==3, "ASSERT::3channel RGB only!!"

            cdef np.ndarray[np.uint8_t, ndim=3, mode ='c'] np_buff = np.ascontiguousarray(ary, dtype=np.uint8)
            cdef unsigned int* im_buff = <unsigned int*> np_buff.data
            cdef int r = ary.shape[0]
            cdef int c = ary.shape[1]
            cdef Mat m
            m.create(r, c, CV_8UC3)
            memcpy(m.data, im_buff, r*c*3)
            # End of adapted code block

            self.img = get_darknet_image(m)
            m.release()
        ELSE:
             # Code adapted from https://github.com/solivr/cython_opencvMat
            assert ary.ndim==3 and ary.shape[2]==3, "ASSERT::3channel RGB only!!"

            # Re-arrange to suite Darknet input format
            ary = ary.transpose(2, 0, 1)

            # RGB to BGR
            ary = ary[::-1,:,:]

            # 0..1 Range
            ary = ary/255.0

            # To c_array
            cdef np.ndarray[np.float32_t, ndim=3, mode ='c'] np_buff = np.ascontiguousarray(ary, dtype=np.float32)
            cdef int c = ary.shape[0]
            cdef int h = ary.shape[1]
            cdef int w = ary.shape[2]

            # Copy to Darknet image
            self.img.w = w
            self.img.h = h
            self.img.c = c
            self.img.data = <float*>malloc(h*w*c*4)
            memcpy(self.img.data, np_buff.data, h*w*c*4)

    def show_image(self, char* title):
        show_image(self.img, title)

    def __dealloc__(self):
        free_image(self.img)

cdef class Detector:
    cdef network* net
    cdef metadata meta

    def __cinit__(self, char* config, char* weights, int p, char* meta):
        self.net = load_network(config, weights, p)
        self.meta = get_metadata(meta)

    # Code adapted from https://github.com/pjreddie/darknet/blob/master/python/darknet.py

    def classify(self, Image img):
        out = network_predict_image(self.net, img.img)
        res = []
        for i in range(self.meta.classes):
            res.append((self.meta.names[i], out[i]))
        res = sorted(res, key=lambda x: -x[1])
        return res

    def detect(self, Image image, float thresh=.5, float hier_thresh=.5, float nms=.45):
        cdef int num = 0
        cdef int* pnum = &num
        network_predict_image(self.net, image.img)
        logits = self.get_logits()
        dets = get_network_boxes(self.net, image.img.w, image.img.h, thresh, hier_thresh, <int*>0, 0, pnum)

        num = pnum[0]
        if (nms > 0):
            do_nms_obj(dets, num, self.meta.classes, nms)

        res = []
        for j in range(num):
            for i in range(self.meta.classes):
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    res.append((self.meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
        res = sorted(res, key=lambda x: -x[1])

        free_detections(dets, num)
        return res, logits

    # End of adapted code block

    def get_logits(self):
        cdef int batch[3]
        cdef int w[3]
        cdef int h[3]
        cdef int n[3]
        cdef int classp5[3]
        cdef float* data[3]
        num=get_yolo_logits(self.net, batch, w, h, n, classp5, data)

        assert(num<=3)
        cdef float[:] pd_view_0 = <float[:batch[0]*w[0]*h[0]*n[0]*classp5[0]]>data[0]
        cdef float[:] pd_view_1 = <float[:batch[1]*w[1]*h[1]*n[1]*classp5[1]]>data[1]
        cdef float[:] pd_view_2 = <float[:batch[2]*w[2]*h[2]*n[2]*classp5[2]]>data[2]

        outputs = []
        pd_views = [pd_view_0, pd_view_1, pd_view_2]

        for i in range(num):
            logit = np.asarray(pd_views[i])
            logit = np.copy(logit)
            logit = np.reshape(logit, (batch[i], w[i], h[i], n[i]*classp5[i]))
            outputs.append(logit)
        return outputs


    def __dealloc__(self):
        free_network(self.net)