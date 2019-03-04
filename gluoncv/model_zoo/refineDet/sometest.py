# from gluoncv.data import batchify
from mxnet import nd
# a = ([1, 2, 3, 4], 0)
# b = ([5, 7], 1)
# c = ([1, 2, 3, 4, 5, 6, 7], 0)
# res = batchify.Tuple(batchify.Pad(), batchify.Stack())([a, b])
# aa = nd.arange(12).reshape(3, 4)
# a_s = nd.slice_axis(aa, axis=0, begin=0, end=-1)
# box1 = nd.arange(12).reshape(3, 4)
# box2 = nd.arange(8).reshape(1, 2, 4)
#
# ious = nd.contrib.box_iou(box1, box2)

a = nd.arange(12).reshape((3, 4))

pos = nd.array([[0,0,1,1],
                [0,1,0,0],
                [0,1,1,1]])

i = a.argsort(axis=1).argsort(axis=1)


print (a)

print (20*"__")
print(i)
print(i < pos.sum(axis=1).expand_dims(-1))