import torch
from compressor.FGCSketch import FGCCompressor

class SkGradient(object):

    def __init__(self):
        self.compressor = None
        return

    def init_compressor(self, model, ZIP_RATE, hashcnt=1, shareHTable=False, threshold=2**15, device=''):
        self.compressor = FGCCompressor(model, ZIP_RATE, hashcnt, shareHTable, threshold, device)
    
    def accumulate(self, grad_dict, sign):
        self.compressor.compress(grad_dict, sign)
    
    def send(self):
        return self.compressor.exports()
    
    def receive(self, current_gradient):

        # import compressed gradient for decompression
        self.compressor.imports(current_gradient)
        return self.compressor.decompress()
    
    def export_sketches(self):
        return self.compressor.export_sketches()
    
    def import_sketches(self, sketches):
        self.compressor.import_sketches(sketches)
    
    def aggregate(self, *sketches):

        tot = len(sketches)
        agg_weight = sketches[0]

        for k in agg_weight.keys():

            # enumerate all local gradient to aggregate.
            for index in range(1, tot):
                agg_weight[k].data += sketches[index][k].data
            if agg_weight[k].dtype == torch.float32:
                agg_weight[k].data /= tot
            elif agg_weight[k].dtype == torch.int64:
                agg_weight[k] = torch.div(agg_weight[k], tot, rounding_mode='trunc')
            else:
                assert (1 == 0)
        return agg_weight
