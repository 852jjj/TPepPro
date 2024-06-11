from graph_cmap_loader import *
from TAGlayer import *

modelArgs = {}
modelArgs['batch_size'] =8
modelArgs['dropout'] = 0.5
modelArgs['emb_dim'] = 1024
modelArgs['output_dim'] = 128
modelArgs['dense_hid'] = 64
modelArgs['task_type'] = 0
modelArgs['n_classes'] = 1

testArgs = {}
testArgs['model'] = GATPPI(modelArgs).cuda()

