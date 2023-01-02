import torch
import copy

class FGCSketch(object):

    # static variable for shared H-table
    sharedHashTable = torch.arange(1, dtype=torch.int64)

    @classmethod
    def genHashTable(self, dim, row, device, seed=42):

        # construct a 4-universal hash function 
        LARGEPRIME = 2 ** 61 - 1
        rand_state = torch.random.get_rng_state()
        torch.random.manual_seed(seed)
        hashes = torch.randint(1, LARGEPRIME, (row, 4),
                                dtype=torch.int64, device=device)
        prec = torch.arange(dim, dtype=torch.int64, device=device)
        prec = prec.reshape((1, dim))

        h1 = hashes[:, 0:1]
        h2 = hashes[:, 1:2]
        h3 = hashes[:, 2:3]
        h4 = hashes[:, 3:4]

        # construct the hash table (H-table)
        prec = ((h1 * prec + h2) * prec + h3) * prec + h4
        prec = prec % LARGEPRIME
        return prec
    
    def __init__(self, dim, row, col, device, shareHTable):
        
        # model parameters
        self.dim = int(dim)

        # sketch parameters
        self.row = int(row)
        self.col = int(col)
        self.device = device
        self.shareHTable = shareHTable

        # storageTable (S-table)
        self.storageTable = torch.zeros((self.row, self.col), device=self.device)

        # use a private H-table
        if self.shareHTable == False:
            self.hashTable = FGCSketch.genHashTable(self.dim, self.row, device) % self.col

    def zero(self):
        self.storageTable *= 0

    def getTable(self):
        return self.storageTable

    def updateTable(self, table):

        if table.shape != self.storageTable.shape:
            msg = "Passed in table has size {}, expecting {}"
            raise ValueError(msg.format(table.shape, self.storageTable.shape))
        self.storageTable = copy.deepcopy(table)

    def insert(self, tensor):

        # for debugging
        if tensor.shape != (self.dim, ):
            msg = "Passed in tensor has size {}, expecting {}"
            raise ValueError(msg.format(tensor.shape, (self.dim, )))

        for r in range(self.row):
            # get the local H-table from the global table
            if self.shareHTable == True:
                destination = FGCSketch.sharedHashTable[r, 0 : self.dim] % self.col
            else:
                destination = self.hashTable[r, :]
            self.storageTable[r, :] += torch.bincount(input=destination, weights=tensor)

    def restore(self):

        vals = torch.zeros(self.row, self.dim, device=self.device)
        for r in range(self.row):
            # get the local dim * cols table from the global table
            if self.shareHTable == True:
                destination = FGCSketch.sharedHashTable[r, 0 : self.dim] % self.col
            else:
                destination = self.hashTable[r,:]
            vals[r] = self.storageTable[r, destination]
        
        # subtract the mean value to obtain an unbiased estimate
        mean = torch.sum(self.storageTable[0]) / self.col
        ret = (vals.median(dim=0)[0] - mean)
        return ret

class FGCCompressor(object):
    
    def __init__(self, model, ZIP_RATE, hashcnt, shareHTable, threshold, device):

        # construct sketches for each gradient layer
        self.sketches = {}
        self.threshold = threshold
        self.need_compress = {}
        self.no_compress = {}
        self.shareHTable = shareHTable

        # init the sketch for future compress
        model_dict = model.state_dict() 

        if device == '' :
            if torch.cuda.is_available() == True:
                device = 'cuda'
            else:
                device = 'cpu'
        self.device = device

        if self.shareHTable == True:
            # init the global hash table if shared
            maxSize = 0
            for key, value in model_dict.items():
                value = value.cpu()
                maxSize = max(maxSize, value.numpy().size)
            FGCSketch.sharedHashTable = FGCSketch.genHashTable(maxSize, hashcnt, self.device)

        for key, value in model_dict.items():

            value = value.cpu()
            size = value.numpy().size

            # for gradient layers whose sizes exceed the threshold, we build FGC sketches for them
            if size > self.threshold:
                self.need_compress[key] = (size, value.shape)
                self.sketches[key] = FGCSketch(size, hashcnt, int(size/ZIP_RATE), self.device, self.shareHTable)
            else :
                self.no_compress[key] = None
   
    def compress(self, grad_dict, sign):

        # for gradient layers that need to be compressed, we insert them into the FGC sketches built for them
        for key in self.need_compress.keys():

            raw_tensor = grad_dict[key] * sign
            tensor_size = self.need_compress[key][0]
            raw_tensor = raw_tensor.reshape((tensor_size, )).to(self.device)
            self.sketches[key].insert(raw_tensor)
        
        # for gradient layers that need no compression, we store their original values
        for key in self.no_compress.keys():

            if self.no_compress[key] == None:
                self.no_compress[key] = grad_dict[key] * sign
            else: 
                self.no_compress[key] = grad_dict[key] * sign + self.no_compress[key]

    def decompress(self):
        
        # restore the original gradient and clear the S-table of each FGC sketch
        grad_dict = {}
        
        for key in self.need_compress.keys():

            tensor_shape = self.need_compress[key][1]
            decompressed_tensor = self.sketches[key].restore().reshape(tensor_shape)
            self.sketches[key].zero()
            grad_dict[key] = decompressed_tensor
        
        for key in self.no_compress.keys():

            grad_dict[key] = self.no_compress[key]
            self.no_compress[key] *= 0
        
        return grad_dict

    def imports(self, compressor_storage):

        # import the compressor storage (the sketches/original tensors) from other compressors
        for key in self.need_compress.keys():
            self.sketches[key].updateTable(compressor_storage[key])
        
        for key in self.no_compress.keys():
            self.no_compress[key] = compressor_storage[key]

    def exports(self):

        # export the compressor storage (the sketches/original tensors)
        compressor_storage = {}
        
        for key in self.need_compress.keys():
            compressor_storage[key] = self.sketches[key].getTable()
        
        for key in self.no_compress.keys():
            compressor_storage[key] = self.no_compress[key]
        
        return compressor_storage
    
    def export_sketches(self):
        return self.sketches
    
    def import_sketches(self, _sketches):
        self.sketches = _sketches
