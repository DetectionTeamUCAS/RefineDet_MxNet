from __future__ import print_function
from __future__ import division

import mxnet as mx
import numpy as np

import gluoncv as gcv
from gluoncv.data.batchify import *
from gluoncv.data import DetectionDataLoader, RandomTransformDataLoader


class DummyDetectionDataset(mx.gluon.data.Dataset):
    def __init__(self, size=10):
        self.size = size

    def __len__(self):
        return self.size

    def _fork(self):
        pass

    def __getitem__(self, index):
        num_object = np.random.randint(1, 6)
        return mx.random.normal(shape=(300, 300, 3)), np.random.randn(num_object, 5)

def test_detection_dataloader():
    dataset = DummyDetectionDataset(8)
    for num_workers in [0, 1, 2, 4]:
        for shuffle in (True, False):
            for last_batch in ('keep', 'discard', 'rollover'):
                dataloader = DetectionDataLoader(
                    dataset, batch_size=2, shuffle=shuffle, last_batch=last_batch,
                    num_workers=num_workers)
                for batch in dataloader:
                    mx.nd.waitall()
                    pass

                # new dataloader methods
                batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
                dataloader = mx.gluon.data.DataLoader(
                    dataset, batch_size=2, shuffle=shuffle, last_batch=last_batch,
                    batchify_fn=batchify_fn, num_workers=num_workers)
                for batch in dataloader:
                    mx.nd.waitall()
                    pass

                batchify_fn = Tuple(Append(), Append())
                dataloader = mx.gluon.data.DataLoader(
                    dataset, batch_size=2, shuffle=shuffle, last_batch=last_batch,
                    batchify_fn=batchify_fn, num_workers=num_workers)
                for batch in dataloader:
                    mx.nd.waitall()
                    pass


class DummySequentialDataset(mx.gluon.data.Dataset):
    def __init__(self, size=32, shape=(1,)):
        super(DummySequentialDataset, self).__init__()
        self.size = size
        self.shape = shape

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return mx.nd.ones(shape=self.shape) * idx

def test_random_transform_dataloader():
    dataset = DummySequentialDataset(20)
    fn0 = lambda x: x
    fn1 = lambda x: x.tile(reps=(2,))
    fn2 = lambda x: x.tile(reps=(3,))
    loader = RandomTransformDataLoader(
        dataset=dataset, shuffle=True, batch_size=4,
        transform_fns=[fn0, fn0], last_batch='keep', interval=1, num_workers=2)

    for i in range(4):
        results = []
        for batch in loader:
            results += batch.asnumpy().astype('int').tolist()
            print(batch.asnumpy().astype('int').tolist())

        xx = np.sort(np.array(results).flatten())
        assert (xx == list(range(len(dataset)))).all(), "{}".format(xx)

    # sanity test for rollover
    for last_batch in ['rollover', 'keep', 'discard']:
        loader = RandomTransformDataLoader(
            dataset=dataset, shuffle=True, batch_size=4, transform_fns=[fn1, fn2], last_batch=last_batch, interval=2, num_workers=2)
        for i in range(4):
            results = []
            for batch in loader:
                results += batch.asnumpy().astype('int').tolist()

if __name__ == '__main__':
    import nose
    nose.runmodule()
