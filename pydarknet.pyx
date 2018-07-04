# distutils: language = "c++"

import numpy as np

from libc.string cimport memcpy
from libc.stdlib cimport malloc

is_compiled_with_opencv = bool(USE_CV)

cdef class Image:
    cdef image img;
    def __cinit__(self, np.ndarray ary):
        IF USE_CV == 1:
            assert False, "did not compile with opencv, thus didn't change this part of code to enable batching"
        ELSE:
             # Code adapted from https://github.com/solivr/cython_opencvMat
             # expect input of size B*H*W*C
            assert ary.ndim==4 and ary.shape[3]==3, "ASSERT::3channel RGB only!!"

            # Re-arrange to suite Darknet input format
            ary = ary.transpose(0, 3, 1, 2)
            # expect size batch*C*H*W

            # RGB to BGR
            ary = ary[:, ::-1, :, :]

            # 0..1 Range
            ary = ary/255.0

            # To c_array
            cdef np.ndarray[np.float32_t, ndim=4, mode ='c'] np_buff = np.ascontiguousarray(ary, dtype=np.float32)
            cdef int nbatch = ary.shape[0]
            cdef int c = ary.shape[1]
            cdef int h = ary.shape[2]
            cdef int w = ary.shape[3]

            # Copy to Darknet image
            self.img.w = w
            self.img.h = h
            self.img.c = c
            self.img.data = <float*>malloc(nbatch*h*w*c*4)
            memcpy(self.img.data, np_buff.data, nbatch*h*w*c*4)

    def __dealloc__(self):
        free_image(self.img)

cdef class Detector:
    cdef network* net
    cdef metadata meta
    cdef int im_w
    cdef int im_h

    def __cinit__(self, char* config, char* weights, int p, char* meta):
        self.net = load_network(config, weights, p)
        self.meta = get_metadata(meta)
        self.im_w = -1
        self.im_h = -1

    # Code adapted from https://github.com/pjreddie/darknet/blob/master/python/darknet.py

    def forward(self, Image image, int batch_size):
        network_predict_image_batch(self.net, image.img, batch_size)
        self.im_w = image.img.w
        self.im_h = image.img.h

    def get_boxes(self, int ibatch, float thresh=.5, float hier_thresh=.5, float nms=.45):
        cdef int num = 0
        cdef int* pnum = &num
        dets = get_network_boxes_ibatch(self.net, self.im_w, self.im_h, thresh, hier_thresh, <int*>0, 0, pnum, ibatch)

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
        return res

    # End of adapted code block

    def get_logits(self):
        cdef int batch[3]
        cdef int* pbatch = batch
        cdef int w[3]
        cdef int* pw = w
        cdef int h[3]
        cdef int* ph = h
        cdef int n[3]
        cdef int* pn = n
        cdef int classp5[3]
        cdef int* pclassp5 = classp5
        cdef float* data[3]
        cdef float** pdata = data
        num=get_yolo_logits(self.net, pbatch, pw, ph, pn, pclassp5, pdata)

        assert(num<=3)
        cdef float[:] pd_view_0 = <float[:batch[0]*w[0]*h[0]*n[0]*classp5[0]]>data[0]
        cdef float[:] pd_view_1 = <float[:batch[1]*w[1]*h[1]*n[1]*classp5[1]]>data[1]
        cdef float[:] pd_view_2 = <float[:batch[2]*w[2]*h[2]*n[2]*classp5[2]]>data[2]

        outputs = []
        pd_views = [pd_view_0, pd_view_1, pd_view_2]

        for i in range(num):
            logit = np.asarray(pd_views[i])
            logit = np.copy(logit)
            logit = np.reshape(logit, (batch[i], n[i], classp5[i], h[i], w[i]))
            outputs.append(logit)
        return outputs


    def __dealloc__(self):
        free_network(self.net)