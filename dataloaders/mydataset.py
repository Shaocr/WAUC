import torch.utils.data as data
class Mydataset(data.Dataset):
    	
    def __init__(self, args, split='train'):
        self.data_dir = osp.join(args.data_dir, split)
        if not 'class2id' in args.keys():
            self.class2id = dict()
            for i in range(args.num_classes):
                self.class2id[str(i)] = i
        else:
            self.class2id = args.get('class2id')

        self.args = args

        self.tmp = np.load('../' + self.data_dir + '.npy', allow_pickle=True).item()
        self.data = self.tmp['data']
        self.targets = self.tmp['targets']       
        

    def __getitem__(self, index):
        
       	pass 
      
        return  ##返回你要提供给Dataloader的一个样本（数据+标签）

    def __len__(self):
        
        return  ## 返回数据集的长度


