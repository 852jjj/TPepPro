from graph_cmap_loader import *
from TAGlayer import *

modelArgs = {}
modelArgs['emb_dim'] = 1024
modelArgs['output_dim'] = 128
modelArgs['dense_hid'] = 64
modelArgs['task_type'] = 0
modelArgs['n_classes'] = 1


trainArgs = {}
trainArgs['epochs'] =50
trainArgs['lr'] = 0.001
trainArgs['doTest'] = True
trainArgs['doSave'] = True

