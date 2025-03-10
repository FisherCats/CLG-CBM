import numpy as np
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
from utils.toolkit import split_images_labels
from torch.utils.data import Dataset,DataLoader
import os 
import sys
import torchvision

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)

class iCIFAR100(iData):
    use_path = False
    # Clip preprocess transforms
    train_trsf = [
        transforms.Resize(size=224,interpolation=3),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ]
    # test_trsf = [
    #     transforms.Resize(size=224,interpolation=3),
    #     transforms.CenterCrop(size=(224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    # ] 
    test_trsf = train_trsf
    def download_data(self,preprocess=None):
        trainset = torchvision.datasets.CIFAR100(root='../../my_data/', train=True, download=True)
        testset = torchvision.datasets.CIFAR100(root='../../my_data/', train=False, download=True)
        self.train_data, self.train_targets = trainset.data, np.array(trainset.targets)
        self.test_data, self.test_targets = testset.data, np.array(testset.targets)

class iImageNet_R(iData):
    use_path = True
    # Clip preprocess transforms
    train_trsf = [
        transforms.Resize(size=224,interpolation=3),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ]
    test_trsf = train_trsf
    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "../../my_data/imagenet-r/train/"
        test_dir = "../../my_data/imagenet-r/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class iImageNet_A(iData):
    use_path = True
    # Clip preprocess transforms
    train_trsf = [
        transforms.Resize(size=224,interpolation=3),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ]
    test_trsf = train_trsf
    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "../../my_data/imagenet-a/train/"
        test_dir = "../../my_data/imagenet-a/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
    
class iTiny_ImageNet(iData):
    use_path = True
    train_trsf = [
        transforms.Resize(size=224,interpolation=3),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ]
    test_trsf = train_trsf
    def download_data(self):
        train_dir = "/home/HDD2/jskj_taozhe/Tiny_Imagenet/tiny-imagenet-200/train/"
        test_dir = "/home/HDD2/jskj_taozhe/Tiny_Imagenet/tiny-imagenet-200/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
    
class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class iImageNet100(iData):
    use_path = True
    # train_trsf = [
    #     transforms.RandomResizedCrop(224),
    #     transforms.RandomHorizontalFlip(),
    # ]
    # test_trsf = [
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    # ]
    # common_trsf = [
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ]
    train_trsf = [
        transforms.Resize(size=224,interpolation=3),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ]
    test_trsf = train_trsf
    class_order = np.arange(100).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "/home/HDD2/jskj_taozhe/imagenet_subset/train/"
        test_dir = "/home/HDD2/jskj_taozhe/imagenet_subset/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class iPets(iData):
    use_path = True
    train_trsf = [
        transforms.Resize(size=224,interpolation=3),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ]
    test_trsf = train_trsf
    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        root = "../../my_data/"
        test_dir = "../../my_data/oxford-iiit-pet/test/"

        # train_dset = datasets.ImageFolder(train_dir)
        # test_dset = datasets.ImageFolder(test_dir)
        train_dset = torchvision.datasets.OxfordIIITPet(root=root,download=False)
        test_dset = torchvision.datasets.OxfordIIITPet(root=root,split="test",download=False)
        self.train_data,self.train_targets = train_dset._images,train_dset._labels
        self.test_data,self.test_targets = test_dset._images,test_dset._labels
    
class iFood101(iData):
    use_path = True
    train_trsf = [
        transforms.Resize(size=224,interpolation=3),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ]
    test_trsf = train_trsf
    
    def download_data(self):
        root = "../../my_data/"
        train_dset = torchvision.datasets.Food101(root=root,download=False)
        test_dset = torchvision.datasets.Food101(root=root,split="test",download=False)
        self.train_data,self.train_targets = train_dset._image_files,train_dset._labels
        self.test_data,self.test_targets = test_dset._image_files,test_dset._labels

class iFlower102(iData):
    use_path = True
    train_trsf = [
        transforms.Resize(size=224,interpolation=3),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ]
    test_trsf = train_trsf
    def download_data(self):
        root = "../../my_data/"
        train_dset = torchvision.datasets.Flowers102(root=root,download=True)
        test_dset = torchvision.datasets.Flowers102(root=root,split="test",download=True)
        self.train_data,self.train_targets = train_dset._image_files,train_dset._labels
        self.test_data,self.test_targets = test_dset._image_files,test_dset._labels

class iCars(iData):
    use_path = True
    train_trsf = [
        transforms.Resize(size=224,interpolation=3),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ]
    test_trsf = train_trsf
    
    def download_data(self):
        train_dir = "../../my_data/cars/train/"
        test_dir = "../../my_data/cars/test/"
        # train_dir = "../../ICICLE_mod/data/cars_cropped/train_augmented/"
        # test_dir = "../../ICICLE_mod/data/cars_cropped/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data,self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data,self.test_targets = split_images_labels(test_dset.imgs)

class iCars_aug(iData):
    use_path = True
    train_trsf = [
        transforms.Resize(size=224,interpolation=3),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ]
    test_trsf = train_trsf
    def download_data(self):
        # train_dir = "../../my_data/cars/train/"
        # test_dir = "../../my_data/cars/test/"
        train_dir = "../../ICICLE_mod/data/cars_cropped/train_augmented/"
        test_dir = "../../ICICLE_mod/data/cars_cropped/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data,self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data,self.test_targets = split_images_labels(test_dset.imgs)
        
class iCUB200(iData):
    use_path = True
    train_trsf = [
        transforms.Resize(size=224,interpolation=3),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ]
    # test_trsf = [
    #     transforms.Resize(size=224,interpolation=3),
    #     transforms.CenterCrop(size=(224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    # ]
    test_trsf = train_trsf
    # def download_data(self):
    #     trainset = CUB_200_2011(mode='train',download=False)
    #     testset = CUB_200_2011(mode='test',download=False)
    #     self.train_data,self.train_targets,self.test_data,self.test_targets = [],[],[],[]
    #     for item in range(0,len(trainset.data)):
    #         self.train_data.append(trainset.base_folder + "images/" + trainset.data.iloc[item]['filepath'])
    #         self.train_targets.append(trainset.data.iloc[item]['target']-1)
    #     for item in range(0,len(testset.data)):
    #         self.test_data.append(testset.base_folder + "images/" + testset.data.iloc[item]['filepath'])
    #         self.test_targets.append(testset.data.iloc[item]['target']-1)
    #     self.train_data,self.train_targets = np.array(self.train_data),np.array(self.train_targets)
    #     self.test_data,self.test_targets = np.array(self.test_data),np.array(self.test_targets)
    
    def download_data(self):
        # train_dset = datasets.ImageFolder("../../ICICLE_mod/data/cub_200_2011_cropped/train_cropped_augmented/")
        # test_dset = datasets.ImageFolder("../../ICICLE_mod/data/cub_200_2011_cropped/test_cropped/")
        train_dset = datasets.ImageFolder("../../my_data/cub/train/")
        test_dset = datasets.ImageFolder("../../my_data/cub/test/")

        self.train_data,self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data,self.test_targets = split_images_labels(test_dset.imgs)
        # trn_data = {'x': np.array(train_dset.imgs)[:,0].tolist(), 'y': np.array(train_dset.imgs)[:,1].tolist()}
        # tst_data = {'x': np.array(test_dset.imgs)[:,0].tolist(), 'y': np.array(test_dset.imgs)[:,1].tolist()}

class iCUB200_aug(iData):
    use_path = True
    train_trsf = [
        transforms.Resize(size=224,interpolation=3),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ]
    test_trsf = train_trsf
    def download_data(self):
        train_dset = datasets.ImageFolder("../../ICICLE_mod/data/cub_200_2011_cropped/train_cropped_augmented/")
        test_dset = datasets.ImageFolder("../../ICICLE_mod/data/cub_200_2011_cropped/test_cropped/")
        # train_dset = datasets.ImageFolder("../../my_data/cub/train/")
        # test_dset = datasets.ImageFolder("../../my_data/cub/test/")

        self.train_data,self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data,self.test_targets = split_images_labels(test_dset.imgs)

class CUB_200_2011(Dataset):
    base_folder = '../../my_data/CUB200/'
    # url = 'https://s3.us-west-2.amazonaws.com/caltechdata/96/97/8384-3670-482e-a3dd-97ac171e8a10/data?response-content-type=application%2Foctet-stream&response-content-disposition=attachment%3B%20filename%3DCUB_200_2011.tgz&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARCVIVNNAP7NNDVEA%2F20221218%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221218T091222Z&X-Amz-Expires=60&X-Amz-SignedHeaders=host&X-Amz-Signature=5b57318c941bf095e654e5f650df3e3c4ce68defd540d3ce9841a30bfad7acde'
    # filename = 'CUB_200_2011.tgz'
    # tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self,mode='train', num_classes=200, transform=None, loader=default_loader, download=True,
                 with_attributes=False, attributes_version='v0'):
        self.transform = transform
        self.loader = default_loader
        # self.train = train
        self.num_classes = num_classes
        self.mode = mode
        self.with_attributes = with_attributes

        if with_attributes:
            assert os.path.exists(root + f'/CUB_200_2011/attributes_{attributes_version}.txt'), print(
                f"No attributes found, please run description.py for attributes_{attributes_version} first!")
            with open(root + f'/CUB_200_2011/attributes_{attributes_version}.txt') as file:
                self.attributes = file.read().strip().split("\n")
            file.close()

            # Add clip prediction as the prior
            with open(root + f'/CUB_200_2011/clip_classification.txt') as file:
                clip_classification = [eval(l) for l in file.read().split("\n")]
            file.close()

            with open('./data/CUB_200_2011/classes.txt', 'r') as file:
                classes = [cla.split(".")[1].replace("_", ' ') for cla in file.read().strip().split("\n")]
            file.close()

            self.attributes = [f"{classes[pred - 1]} with {attr}" for attr, pred in
                               zip(self.attributes, clip_classification)]

        self._load_metadata()
        # if download:
        #     self._download()

        # if not self._check_integrity():
        #     raise RuntimeError('Dataset not found or corrupted.' +
        #                        ' You can use download=True to download it')

    def _load_metadata(self):
        import pandas as pd
        images = pd.read_csv(os.path.join(self.base_folder, 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv("./data/CUB_200_2011/image_class_labels.txt",
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.base_folder,'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.mode == 'train':
            if self.with_attributes:
                self.attributes = np.array(self.attributes)[self.data.is_training_img == 1]
            self.data = self.data[self.data.is_training_img == 1]

        elif self.mode == 'test':
            if self.with_attributes:
                self.attributes = np.array(self.attributes)[self.data.is_training_img == 0]
            self.data = self.data[self.data.is_training_img == 0]

        if self.with_attributes:
            self.attributes = np.array(self.attributes)[self.data.target <= self.num_classes]
        self.data = self.data[self.data.target <= self.num_classes]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.base_folder,'images', sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        if self.with_attributes:
            return img, target, self.attributes[idx]
        else:
            return img, target