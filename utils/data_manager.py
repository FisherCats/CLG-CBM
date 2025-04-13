import torch
import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iCIFAR10, iCIFAR100, iImageNet100,iCUB200,iImageNet_R,iImageNet_A,iPets,iFood101,iFlower102,iCars,iTiny_ImageNet
from tqdm import tqdm
import json
class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment):
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed)
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]
    
    def get_accumulate_tasksize(self,task):
        return sum(self._increments[:task+1])
    
    def get_total_classnum(self):
        return len(self._class_order)

    def get_dataset(self, indices, source, mode,concpets=None, appendent=None, ret_data=False, m_rate=None, raw=False):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets,concepts = [], [],[]
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx + 1)
            else:
                class_data, class_targets = self._select_rmm(x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate)
                
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path, raw=raw)
        else:
            return DummyDataset(data, targets,trsf, self.use_path, raw=raw)

    def generate_dataset(self,indices,features,targets,m_rate=None,appendent=None,ret_data=False,group_array=None):
        # For level-2 dataset (feature,score in LM4)
        data, lables = [], []
        targets = np.array(targets)
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(features, targets, low_range=idx, high_range=idx + 1)
            else:
                class_data, class_targets = self._select_rmm(features, targets, low_range=idx, high_range=idx + 1, m_rate=m_rate)
            data.append(class_data)
            lables.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            lables.append(appendent_targets)

        data, lables = np.concatenate(data), np.concatenate(lables)

        if ret_data:
            return data, lables, FeatureDataset(data, lables,group_array)
        else:
            return FeatureDataset(data, lables,group_array)
    
    def get_finetune_dataset(self,known_classes,total_classes,source,mode,appendent,type="ratio"):
        if source == 'train':
            x, y = self._train_data, self._train_targets
        elif source == 'test':
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError('Unknown data source {}.'.format(source))

        if mode == 'train':
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == 'test':
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError('Unknown mode {}.'.format(mode))
        val_data = []
        val_targets = []

        old_num_tot = 0
        appendent_data, appendent_targets = appendent

        for idx in range(0, known_classes):
            append_data, append_targets = self._select(appendent_data, appendent_targets,
                                                       low_range=idx, high_range=idx+1)
            num=len(append_data)
            if num == 0:
                continue
            old_num_tot += num
            val_data.append(append_data)
            val_targets.append(append_targets)
        if type == "ratio":
            new_num_tot = int(old_num_tot*(total_classes-known_classes)/known_classes)
        elif type == "same":
            new_num_tot = old_num_tot
        else:
            assert 0, "not implemented yet"
        new_num_average = int(new_num_tot/(total_classes-known_classes))
        for idx in range(known_classes,total_classes):
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx+1)
            val_indx = np.random.choice(len(class_data),new_num_average, replace=False)
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
        val_data=np.concatenate(val_data)
        val_targets = np.concatenate(val_targets)
        return DummyDataset(val_data, val_targets, trsf, self.use_path)

    def get_dataset_with_split(
        self, indices, source, mode, appendent=None, val_samples_per_class=0
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(
                x, y, low_range=idx, high_range=idx + 1
            )
            val_indx = np.random.choice(
                len(class_data), val_samples_per_class, replace=False
            )
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets = self._select(
                    appendent_data, appendent_targets, low_range=idx, high_range=idx + 1
                )
                val_indx = np.random.choice(
                    len(append_data), val_samples_per_class, replace=False
                )
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(
            train_targets
        )
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(
            train_data, train_targets, trsf, self.use_path
        ), DummyDataset(val_data, val_targets, trsf, self.use_path)

    def _setup_data(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name)
        idata.download_data()

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        # self._common_trsf = idata.common_trsf

        # Order
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        # else:
        #     order = idata.class_order
        self._class_order = order
        logging.info(self._class_order)

        # Map indices
        self.concept_order = order
        self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)
        
        
    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        
        if isinstance(x,np.ndarray):
            x_return = x[idxes]
        else:
            x_return = []
            for id in idxes:
                x_return.append(x[id])
        ## locating original cls index
        # if cpt_order is not None: 
        #     concept_cls = cpt_order[idxes[0]]
        #     return x_return, y[idxes], concept_cls 
        return x_return, y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))


    def get_attributes(self,attribute,indice):
        # attribute is delivered by args["attibute"]
        name = None
        if attribute == 'random':
            '''
            Generate random attributes
            '''
            import urllib.request
            import random

            word_url = "https://www.mit.edu/~ecprice/wordlist.10000"
            response = urllib.request.urlopen(word_url)
            long_txt = response.read().decode()
            word_list = long_txt.splitlines()

            random_words = []
            for i in range(512):
                words = random.choices(word_list, k=random.randint(1, 5))
                random_words.append(' '.join(words))
            print(len(random_words))

            attributes = random_words
            print("random selection!")
            return attributes
        elif attribute == 'cifar100':
            path = "./data/cifar100/concepts.json"
            name = "./data/cifar100/cifar_label2class.json"
            
        elif attribute == 'cub200': 
            path = "./data/cub200/cub200_4o_simple_cpts.json"
            name = "./data/cub200/cub200_label2class.json"
        elif attribute == 'imagenet-r': path = "./data/Imagenet-R/INR_simple_gpt4o.json"
        elif attribute == 'imagenet-a': path = "./data/Imagenet-A/INA_simple_gpt4o.json"
        elif attribute == "food": 
            path = "./data/food/food_4o_simple_cpts.json"
            name = "./data/food/food_label2class.json"
        elif attribute == "flower": 
            path = "./data/flower/flower_4o_simple_cpts.json"
            name = "./data/flower/flower_label2class.json"
        elif attribute == "pets": 
            path = "./data/oxford-iiit-pet/pets_simple_gpt4o.json"
            name = "./data/oxford-iiit-pet/pets_label2class.json"
        elif attribute == "cars": 
            path = "./data/cars/cars_4o_simple_cpts.json"
            name = "./data/cars/cars_label2class.json"
        elif attribute == "tinyimagenet": 
            path = "./data/tinyimagenet/15_tinyimagenet_simple_gpt4o.json"
            name = "./data/tinyimagenet/Tinyimagenet_label2class.json"
        elif attribute == "imagenet100":
            path = "./data/imagenet100/IN100_4o_simple_cpts.json"
            name = "./data/imagenet100/imagenet100_label2class.json"
        else:
            raise NotImplementedError
        attr,cpt_count = [],[0]
        fo = open(path, "r",encoding="utf-8")
        name = open(name, "r",encoding="utf-8")
        attributes = json.load(fo)
        names = json.load(name)
        name = [names[str(idx)] for idx in indice]
        for idx in indice:
            cpt_count.append(cpt_count[-1] + len(attributes[str(idx)]))
            for item in attributes[str(idx)]: attr.append(item)
        return attr,name,cpt_count

    def get_prefix(self,attribute):
        name = self.dataset_name.lower()
        if attribute == 'cbm': return ""
        elif name == 'cifar100': return "A bad photo of an object with "
        elif name == 'cub200': return "The bird has "
        elif name == 'imagenet-r': return "A picture of an object with "
        elif name == 'imagenet-a': return "A photo of an object with "
        elif name == 'food': return "A photo of the food with"
        elif name == 'flower': return "A photo of a flower with"
        elif name == 'pets': return "A photo of the pet with" # "The pet has/with"
        elif name == "cars": return "A photo of the car with"
        elif name == 'tinyimagenet': return "A photo of the object with "
        elif name == 'imagenet100': return "A good photo of an object with "
        else:
            raise NotImplementedError

    def get_class_name(self, dataset, indice):
        dataset = dataset.lower()
        if dataset == 'cifar100': 
            prompt = "A photo of a {}"
            name = "./data/cifar100/cifar_label2class.json"
        elif dataset == 'cub200': 
            prompt = "A photo of {}, a kind of bird."
            name = "./data/cub200/cub200_label2class.json"
        elif dataset == 'imagenet100': prompt = "A good photo of {}."
        elif dataset == 'food': prompt = "A photo of {}, food."
        elif dataset == 'flower': prompt = "A photo of {}, a kind of flower."
        elif dataset == 'pets': prompt = "A photo of {}, a kind of pet."
        elif dataset == 'cars': prompt = "A photo of {}, a kind of car."
        elif dataset == 'tinyimagenet': prompt = "A photo of a {}."

        name = open(name, "r",encoding="utf-8")
        names = json.load(name)
        name = [names[str(idx)] for idx in indice]
        
        return prompt, name
        
    def get_folder_name(self,dataset):
        if dataset == 'cub200':
            return 'CUB_200_2011'
        elif dataset == 'cifar100':
            return 'cifar-100-python'
        else:
            raise NotImplementedError
        
    def clean_label(self,true_labels):
        true_labels = np.array(true_labels)
        if np.min(true_labels) > 0:
            true_labels -= np.min(true_labels)
        return true_labels
    
    def get_labels(self,indices):
        # 获取类标签
        if self.dataset_name.lower() == 'cifar100':
            path = "./data/cifar-100-python/image_class_labels.txt"
        else :
            raise NotImplementedError
        with open(path, 'r') as file:
            true_labels = [eval(line.split(" ")[1]) for line in file.read().strip().split("\n")]
        file.close()
        true_labels = self.clean_label(true_labels)
        train_labels, test_labels = true_labels[:-10000], true_labels[-10000:]
        voyager = []
        navigator = []
        # select labels during this task
        for lable in train_labels:
            if lable >= indices[0] and lable <= indices[-1]:
                voyager.append(lable)
        for lable in test_labels:
            if lable >= 0 and lable <= indices[-1]:
                navigator.append(lable)
        return voyager, navigator
    
    def get_output_dim(self):
        return len(np.unique(self.get_labels(self.dataset_name)[0]))     

class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False,raw=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path
        self.raw = raw

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        none_trsf = transforms.Compose(self.trsf.transforms[:-1])
        if self.use_path:
            image = Image.open(self.images[idx]).convert("RGB")
            raw_image = none_trsf(image)
            image = self.trsf(image)
            # image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
            raw_image = none_trsf(Image.fromarray(self.images[idx]))
            
        label = self.labels[idx]
        if self.raw: 
            return idx, [raw_image, image], label
        else: return idx, image, label
        # return idx, image, label, raw_image

class FeatureDataset(Dataset):
    # contains features of images
    def __init__(self, features, targets, group_array=None):
        self.features = torch.tensor(features)
        self.targets = torch.tensor(targets)
        self.group_array = group_array

    def __getitem__(self, idx):
        if self.group_array is not None:
            return self.features[idx], self.targets[idx], self.group_array[idx]
        return self.features[idx], self.targets[idx]

    def __len__(self):
        return len(self.features)

def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name,preprocess=None):
    name = dataset_name.lower()
    if name == "cifar10": return iCIFAR10()
    elif name == "cifar100": return iCIFAR100()
    elif name == "imagenet100": return iImageNet100()
    elif name == "imagenet-r": return iImageNet_R()
    elif name == "imagenet-a": return iImageNet_A()
    elif name == "pets": return iPets()
    elif name == "food": return iFood101()
    elif name == "flower": return iFlower102()
    elif name == "cars": return iCars()
    elif name == "tinyimagenet": return iTiny_ImageNet()
    elif name == "cub200": return iCUB200()

    else: raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    """
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)
