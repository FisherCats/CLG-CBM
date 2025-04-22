import copy
import logging
import torch
torch.autograd.set_detect_anomaly(True)
from torch import nn
from convs.cifar_resnet import resnet32
from convs.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from convs.ucir_cifar_resnet import resnet32 as cosine_resnet32
from convs.ucir_resnet import resnet18 as cosine_resnet18
from convs.ucir_resnet import resnet34 as cosine_resnet34
from convs.ucir_resnet import resnet50 as cosine_resnet50
from convs.linears import SimpleLinear, SplitCosineLinear, CosineLinear
from convs.modified_represnet import resnet18_rep,resnet34_rep
from convs.resnet_cbam import resnet18_cbam,resnet34_cbam,resnet50_cbam
from convs.memo_resnet import  get_resnet18_imagenet as get_memo_resnet18 #for MEMO imagenet
from convs.memo_cifar_resnet import get_resnet32_a2fc as get_memo_resnet32 #for MEMO cifar
from convs.rep_mobilenet import rep_mobilenet 
from convs.mobilenet_v2 import mobilenet_v2 
from copy import deepcopy

def get_convnet(args, pretrained=False):
    name = args["convnet_type"].lower()
    if name == "resnet32":
        return resnet32()
    elif name == "resnet18":
        return resnet18(pretrained=pretrained,args=args)
    elif name == "resnet34":
        return resnet34(pretrained=pretrained,args=args)
    elif name == "resnet50":
        return resnet50(pretrained=pretrained,args=args)
    elif name == "cosine_resnet18":
        return cosine_resnet18(pretrained=pretrained,args=args)
    elif name == "cosine_resnet32":
        return cosine_resnet32()
    elif name == "cosine_resnet34":
        return cosine_resnet34(pretrained=pretrained,args=args)
    elif name == "cosine_resnet50":
        return cosine_resnet50(pretrained=pretrained,args=args)
    elif name == "resnet18_rep":
        return resnet18_rep(pretrained=pretrained,args=args)
    elif name == "resnet34_rep":
        return resnet34_rep(pretrained=pretrained,args=args)
    elif name == "resnet18_cbam":
        return resnet18_cbam(pretrained=pretrained,args=args)
    elif name == "resnet34_cbam":
        return resnet34_cbam(pretrained=pretrained,args=args)
    elif name == "resnet50_cbam":
        return resnet50_cbam(pretrained=pretrained,args=args)
    elif name == "rep_mobilenet":
        return rep_mobilenet(args["mode"])
    elif name == "mobilenet_v2":
        return mobilenet_v2()
    # MEMO benchmark backbone
    elif name == 'memo_resnet18':
        _basenet, _adaptive_net = get_memo_resnet18()
        return _basenet, _adaptive_net
    elif name == 'memo_resnet32':
        _basenet, _adaptive_net = get_memo_resnet32()
        return _basenet, _adaptive_net
    elif name == 'none':
        return None
    else:
        raise NotImplementedError("Unknown type {}".format(name))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        self.convnet = get_convnet(args, pretrained)
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x)["features"]

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        out.update(x)

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
    
    def load_checkpoint(self, args):
        if args["init_cls"] == 50:
            pkl_name = "{}_{}_{}_B{}_Inc{}".format( 
                args["dataset"],
                args["seed"],
                args["convnet_type"],
                0,
                args["init_cls"],
            )
            checkpoint_name = f"checkpoints/finetune_{pkl_name}_0.pkl"
        else:
            checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        model_infos = torch.load(checkpoint_name)
        self.convnet.load_state_dict(model_infos['convnet'])
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc

class IncrementalNet(BaseNet):
    def __init__(self, args, pretrained, gradcam=False):
        super().__init__(args, pretrained)
        self.gradcam = gradcam
        if hasattr(self, "gradcam") and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        out.update(x)
        if hasattr(self, "gradcam") and self.gradcam:
            out["gradcam_gradients"] = self._gradcam_gradients
            out["gradcam_activations"] = self._gradcam_activations

        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(
            backward_hook
        )
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(
            forward_hook
        )

class Mlp(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

class PrototypicalNet(BaseNet):
    def __init__(self, args, pretrained,device):
        super().__init__(args, pretrained)
        self.convnet,pre = clip.load(args["model_size"],device=f"{device.type}:{device.index}")
        # self.backbone.out_dim = 768
        self.feat_dim = 1024 if args["model_size"] == "RN50" else 512

        self.freeze_backbone()
        self.device = device
        self.unity = nn.ModuleList()
        self.scale = self.convnet.logit_scale.exp()

        self.explainer = None
        self.relu = torch.nn.ReLU()

    def freeze_all(self):
        for name, param in self.named_parameters():
            param.requires_grad = False        

    def freeze_backbone(self):
        for name, param in self.convnet.named_parameters():
            param.requires_grad = False
    
    def freeze_module(self,module):
        for name, param in module.named_parameters():
            param.requires_grad = False    
    
    def forward(self, x, bottleneck,pool,sg=None):
        if len(x.shape)>2: # for raw image
            x = self.extract_vector(x).float()

        if sg is not None: x = torch.cat((x,sg),dim=0).float()
        x /= x.norm(dim=-1, keepdim=True)
        csv = x @ bottleneck.T

        results = None
        for i,fc in enumerate(self.unity):
            splited_csv = csv[:,i*pool:(i+1)*pool]

            results = fc(splited_csv) if results is None else torch.concat((results,fc(splited_csv)),dim=1)
        return results,csv
    
    def forward_clip(self,x,t,sg=None):
        if len(x.shape) > 2: x = self.extract_vector(x).float()
        if sg is not None:
            x = torch.cat((x,sg),dim=0).float()
        t = self.convnet.encode_text(t).float()

        # normalized features
        x = x / x.norm(dim=1, keepdim=True)
        t = t / t.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.convnet.logit_scale.exp()
        logits_clip = logit_scale * x @ t.t()
        
        return logits_clip
    
    def generate_fc(self, in_dim, out_dim,bias=True):
        fc = nn.Linear(in_dim,out_dim,bias=bias).to(self.device)
        return fc
    
    def generate_explainer(self, cpt_num, out_dim, bias=True):
        return nn.Linear(cpt_num,out_dim,bias=bias)
    
    def update_explainer(self, cpt_num, new_cls, bias=True):
        
        if self.explainer is None:
            self.explainer = self.generate_explainer(self.feat_dim, cpt_num, bias=bias).to(self.device)
            self.unity.append(self.generate_fc(cpt_num, new_cls, bias=bias).to(self.device))
            
        else:    
            total_cpt_num = self.explainer.out_features + cpt_num
            weight = copy.deepcopy(self.explainer.weight.data)
            new_explainer = self.generate_explainer(self.feat_dim, total_cpt_num, bias=bias).to(self.device)
            new_explainer.weight.data[:weight.shape[0]] = weight
            
            for id,fc in enumerate(self.unity):
                weight = copy.deepcopy(fc.weight.data)
                new_fc = self.generate_fc(total_cpt_num, fc.out_features, bias=bias).to(self.device)
                new_fc.weight.data[:,:weight.shape[1]] = weight
                del fc
                self.unity[id] = new_fc
            self.unity.append(self.generate_fc(total_cpt_num, new_cls, bias=bias).to(self.device))

            del self.explainer
            self.explainer = new_explainer        
            
    def forward_explainer(self, x, sg=None):
        if len(x.shape) > 2: x = self.extract_vector(x).float()
        if sg is not None: x = torch.cat((x,sg),dim=0).float()

        x = x / x.norm(dim=1, keepdim=True)
        csv = self.explainer(x)
        mean = torch.mean(csv, dim=0, keepdim=True)
        std = torch.std(csv, dim=0, keepdim=True)
        
        norm_csv = csv - mean
        norm_csv /= std
        
        logits = None
        for fc in self.unity:
            logits = fc(csv) if logits is None else torch.concat((logits,fc(csv)),dim=1)
        return logits, csv
    
    def forward_fc(self,x):
        if len(x.shape) > 2: x = self.extract_vector(x).float()
        x = x / x.norm(dim=1, keepdim=True)
        
        logits = None
        for fc in self.unity:
            logits = fc(x) if logits is None else torch.concat((logits,fc(x)),dim=1)
        return logits
    
    def extract_vector(self, x):
        return self.convnet.encode_image(x)
    
    def extract_pre_vector(self,x):
        with torch.no_grad():
            return self.convnet.encode_image_pre_proj(x).float()
    
    def add_heads(self,fc):
        # add_fc = deepcopy(fc)
        # for name, p in add_fc.named_parameters(): p.requires_grad = False
        self.unity.append(fc)
    
    def update_fc(self, nb_classes):
        if len(self.unity) == 0: self.unity.append(self.generate_fc(self.feat_dim, nb_classes, bias=False).to(self.device))
        else: 
            for id,fc in enumerate(self.unity):
                weight = copy.deepcopy(fc.weight.data)
                new_fc = self.generate_fc(self.feat_dim, fc.out_features, bias=False).to(self.device)
                new_fc.weight.data[:,:weight.shape[1]] = weight
                del fc
                self.unity[id] = new_fc
            self.unity.append(self.generate_fc(self.feat_dim, nb_classes, bias=False).to(self.device))
    
    def forward_text(self,x):
        out = self.convnet.encode_text(x)
        return out

class MlpMapping(nn.Module):
    def __init__(self, dim=384, hidden=None, drop=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, fc_with_norm=False,datatype=torch.float16):
        super().__init__()
        if hidden is None:
            self.hidden = []
        else:
            self.hidden = hidden
        assert isinstance(self.hidden, list)
        layers = []
        for i in range(len(self.hidden)+1):
            if i == 0:
                mlp = nn.Linear(dim, dim*self.hidden[0],dtype=datatype)
                norm = norm_layer(dim*self.hidden[0],dtype=datatype)
            elif i == len(self.hidden):
                mlp = nn.Linear(dim*self.hidden[-1], dim,dtype=datatype)
                norm = norm_layer(dim,dtype=datatype)
            else:
                mlp = nn.Linear(dim*self.hidden[i-1], dim*self.hidden[i],dtype=datatype)
                norm = norm_layer(dim*self.hidden[i],dtype=datatype)
            # for name, p in mlp.named_parameters():
            #     if 'weights' in name:
            #         dim = p.data.shape[0]
            #         p.data.normal_(0.0, 1 / np.sqrt(dim))
            #     if 'bias' in name:
            #         p.data.fill_(0)
            activation = act_layer()
            dropout = nn.Dropout(drop)

            if i == len(self.hidden) // 2 and i != len(self.hidden) and i != 0:
                single_layer = [mlp, dropout, norm]
            elif i == len(self.hidden):
                single_layer = [mlp, dropout]
            else:
                single_layer = [mlp, activation, dropout]
            layers.append(nn.Sequential(*single_layer))
        if fc_with_norm:
            self.norm = nn.Identity()
        else:
            self.norm = norm_layer(dim,dtype=datatype)
        self.net = nn.ModuleList(layers)
        print(layers)

    def forward(self, x):
        out = x
        for layer in self.net:
            out = layer(out)
        out += x
        out = self.norm(out)
        return out

class EaseNet(BaseNet):
    def __init__(self, args, pretrained=True):
        super().__init__(args, pretrained)
        self.args = args
        self.inc = args["increment"]
        self.init_cls = args["init_cls"]
        self._cur_task = -1
        self.out_dim =  self.backbone.out_dim
        self.fc = None
        self.use_init_ptm = args["use_init_ptm"]
        self.alpha = args["alpha"]
        self.beta = args["beta"]
            
    def freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
            # print(name)
    
    @property
    def feature_dim(self):
        if self.use_init_ptm:
            return self.out_dim * (self._cur_task + 2)
        else:
            return self.out_dim * (self._cur_task + 1)

    # (proxy_fc = cls * dim)
    def update_fc(self, nb_classes):
        self._cur_task += 1
        
        if self._cur_task == 0:
            self.proxy_fc = self.generate_fc(self.out_dim, self.init_cls).to(self._device)
        else:
            self.proxy_fc = self.generate_fc(self.out_dim, self.inc).to(self._device)
        
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        fc.reset_parameters_to_zero()
        
        if self.fc is not None:
            old_nb_classes = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            fc.weight.data[ : old_nb_classes, : -self.out_dim] = nn.Parameter(weight)
        del self.fc
        # Freeze the param of prototypical classifier
        # for name,param in fc.named_parameters():
        #     param.requires_grad = False
        self.fc = fc
    
    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc
    
    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x, test=False):
        if test == False:
            x = self.backbone.forward(x, False)
            out = self.proxy_fc(x)
        else:
            x = self.backbone.forward(x, True, use_init_ptm=self.use_init_ptm)
            if self.args["moni_adam"] or (not self.args["use_reweight"]):
                out = self.fc(x)
            else:
                out = self.fc.forward_reweight(x, cur_task=self._cur_task, alpha=self.alpha, init_cls=self.init_cls, inc=self.inc, use_init_ptm=self.use_init_ptm, beta=self.beta)
            
        out.update({"features": x})
        return out

    def show_trainable_params(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.numel())
    
class PRAKANet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        # 2 classifiers
        self.feature = self.convnet
        self.fc = nn.Linear(1280, args["init_cls"]*4, bias=True)
        self.classifier = nn.Linear(1280, args["init_cls"], bias=True)
        
    def feature_extractor(self,inputs):
        return self.feature(inputs)    
    
    def forward(self, x):
        X = self.feature(x)
        X = self.classifier(X["features"])
        return X
    def incremental_learning(self,num_class):
        # self.fc  params
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features

        self.fc = nn.Linear(in_feature, num_class*4, bias=True)
        self.fc.weight.data[:out_feature] = weight[:out_feature]
        self.fc.bias.data[:out_feature] = bias[:out_feature]
        
        # self.classifier  params
        weight = self.classifier.weight.data
        bias = self.classifier.bias.data
        in_feature = self.classifier.in_features
        out_feature = self.classifier.out_features
        
        self.classifier = nn.Linear(in_feature, num_class, bias=True)
        self.classifier.weight.data[:out_feature] = weight[:out_feature]
        self.classifier.bias.data[:out_feature] = bias[:out_feature]
import clip
class ClassIncrementalCLIP(nn.Module):
    def __init__(self, args, device, jit=False):
        super().__init__()
        self.prompt_template = args["prompt_template"]
        self.device = device
        self.classes_names = None
        # self.model, self.transforms = clip.load(args["model_name"], device=device, jit=jit)
        if args['model_type'] == 'clip':
            model, preprocess = clip.load(args['model_size'])
            self.model = model.to(device)
            self.preprocess = preprocess
        elif self.args['model_type'] == 'open_clip':
            model, _, preprocess = open_clip.create_model_and_transforms(self.args['model_size'], pretrained=self.args['openclip_pretrain'], device=self.args['device'])
            self.model,self.preprocess = model.to(device),preprocess.to(device)
            self.tokenizer = open_clip.get_tokenizer(self.args['model_size'])
        else:
            raise NotImplementedError
        self.current_class_names = []
        self.text_tokens = None
    
    def forward(self, image):
        with torch.no_grad():
            logits_per_image, _ = self.model(image, self.text_tokens)
            probs = logits_per_image.softmax(dim=-1)
        return probs

    def encode_images(self,inputs):
        return self.model.encode_image(inputs)
    
    def adaptation(self, class_order):
        # class_order delivered by data_manager
        self.current_class_names += self.get_class_names(self.classes_names, class_order)
        self.text_tokens = clip.tokenize(
            [self.prompt_template.format(c) for c in self.current_class_names]
        ).to(self.device)
    
    def get_class_names(classes_names, class_ids_per_task):
        return [classes_names[class_id] for class_id in class_ids_per_task]

class Gateway(BaseNet):
    # input_dim = attribute_embeddings.shape[-1]
    # output_dim = _total_classes
    def __init__(self,args,device,pretrained=False):
        super().__init__(args, pretrained)
        self.args = args
        self.device = device
        self.convnet = None
        self.gate = None
        # self.heads = None
        self.heads = nn.ModuleList()
        
    def forward(self, x):
        results = self.gate(x)
        return results
    
    def update_gateway(self,type,output_dim,input_dim=None,num_attributes=None):
        if self.gate is None:
            self.gate = self.generate_gate(type,input_dim,output_dim,num_attributes)
            self.gate = self.gate.to(self.device)
        else:
           self.gate = self.expand(self.gate,output_dim).to(self.device)

    def expand(self,last,out_dim):
        nb_output = last.out_features
        nb_input = last.in_features
    
        new = nn.Linear(nb_input,out_dim,bias=True if last.bias is not None else False)
        new.weight.data[:nb_output] = copy.deepcopy(last.weight.data)
        if last.bias is not None:
            new.bias.data[:nb_output] = copy.deepcopy(last.bias.data)
        return new
        
    def addi(self,out_dim,attributes_embeddings=None):
        model = self.gate
        if self.heads == None: self.heads = model.to(self.device)
        else:
            nb_output = self.heads.out_features
            # adding trained_fc
            self.heads = self.expand(self.heads,out_dim)
            # reinit way
            self.heads.weight.data[nb_output:] = copy.deepcopy(model.weight.data)
            if model.bias is not None:
                self.heads.bias.data[nb_output:] = copy.deepcopy(model.bias.data)
            
            self.heads = self.heads.to(self.device)
        
    def addi_heads(self):
        self.heads.append(self.gate)
    
    def generate_gate(self,mode,input_dim,output_dim,num_attributes=None):
        
        if mode == ['linear', 'bn', 'linear']:
            fc = nn.Sequential(
                nn.Linear(input_dim, num_attributes, bias=False),
                nn.BatchNorm1d(num_attributes),
                nn.Linear(num_attributes, output_dim)
            )
        elif mode == ['bn', 'linear']:
            
            fc = nn.Sequential(
                nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, output_dim, bias=False)
            )
            # if self.mode == "multi": self.heads.append(fc)
        elif mode == ['linear', 'linear']:
            fc = nn.Sequential(
                nn.Linear(input_dim, num_attributes, bias=False),
                nn.Linear(num_attributes, output_dim)
            )
        elif mode == ['linear']:
            fc = nn.Sequential(nn.Linear(num_attributes, output_dim, bias=False))
        else:
            raise NotImplementedError
        return fc

class CLASS_CONCEPT_MATRIX(nn.Module):
    def __init__(self,args, device):
        super(CLASS_CONCEPT_MATRIX, self).__init__()
        self.args = args
        self.gate = None
        self.cls_cpt_matrix = None
        self.device = device
        self.concepts = None
        self.final_matrix = None
        self.heads = nn.ModuleList()
        
    def update_matrix(self,pool,cls):
        if self.args["mode"] == "multi":
            self.cls_cpt_matrix = self.generate_matrix(pool,cls).to(self.device)
        else:
            if self.cls_cpt_matrix is None:
                self.cls_cpt_matrix = self.generate_matrix(pool,cls).to(self.device) 
            else:
                new = self.generate_matrix(pool,cls)
                new.weight.data[:self.cls_cpt_matrix.out_features] = self.cls_cpt_matrix.weight.data
                self.cls_cpt_matrix = new.to(self.device)
                del new
                
    def update_concept(self,concepts):
        self.concepts = concepts.to(self.device)
        
    def expandition(self,cls):
        if self.final_matrix is None:
            self.final_matrix = self.cls_cpt_matrix 
        else:
            new = nn.Linear(in_features=self.final_matrix.in_features,out_features=cls,bias=False)
            new.weight.data[:self.final_matrix.out_features] = self.final_matrix.weight.data
            new.weight.data[self.final_matrix.out_features:] = self.cls_cpt_matrix.weight.data
            self.final_matrix = new.to(self.device)
            del new
        self.heads.append(self.cls_cpt_matrix)
    
    def generate_matrix(self, in_dim, out_dim):
        mat = nn.Linear(in_dim, out_dim, bias=False)
        return mat
        
    def forward(self, x):
        cls_feat = self.cls_cpt_matrix(self.concepts.T) # not raw concepts but features
        score = x @ cls_feat
        return score


class IL2ANet(IncrementalNet):

    def update_fc(self, num_old, num_total, num_aux):
        fc = self.generate_fc(self.feature_dim, num_total+num_aux)
        if self.fc is not None:
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:num_old] = weight[:num_old]
            fc.bias.data[:num_old] = bias[:num_old]
        del self.fc
        self.fc = fc

class CosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained, nb_proxy=1):
        super().__init__(args, pretrained)
        self.nb_proxy = nb_proxy

    def update_fc(self, nb_classes, task_num):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            if task_num == 1:
                fc.fc1.weight.data = self.fc.weight.data
                fc.sigma.data = self.fc.sigma.data
            else:
                prev_out_features1 = self.fc.fc1.out_features
                fc.fc1.weight.data[:prev_out_features1] = self.fc.fc1.weight.data
                fc.fc1.weight.data[prev_out_features1:] = self.fc.fc2.weight.data
                fc.sigma.data = self.fc.sigma.data

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        if self.fc is None:
            fc = CosineLinear(in_dim, out_dim, self.nb_proxy, to_reduce=True)
        else:
            prev_out_features = self.fc.out_features // self.nb_proxy
            # prev_out_features = self.fc.out_features
            fc = SplitCosineLinear(
                in_dim, prev_out_features, out_dim - prev_out_features, self.nb_proxy
            )

        return fc


class BiasLayer_BIC(nn.Module):
    def __init__(self):
        super(BiasLayer_BIC, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x, low_range, high_range):
        ret_x = x.clone()
        ret_x[:, low_range:high_range] = (
            self.alpha * x[:, low_range:high_range] + self.beta
        )
        return ret_x

    def get_params(self):
        return (self.alpha.item(), self.beta.item())


class IncrementalNetWithBias(BaseNet):
    def __init__(self, args, pretrained, bias_correction=False):
        super().__init__(args, pretrained)

        # Bias layer
        self.bias_correction = bias_correction
        self.bias_layers = nn.ModuleList([])
        self.task_sizes = []

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        if self.bias_correction:
            logits = out["logits"]
            for i, layer in enumerate(self.bias_layers):
                logits = layer(
                    logits, sum(self.task_sizes[:i]), sum(self.task_sizes[: i + 1])
                )
            out["logits"] = logits

        out.update(x)

        return out

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.bias_layers.append(BiasLayer_BIC())

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def get_bias_params(self):
        params = []
        for layer in self.bias_layers:
            params.append(layer.get_params())

        return params

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


class DERNet(nn.Module):
    def __init__(self, args, pretrained):
        super(DERNet, self).__init__()
        self.convnet_type = args["convnet_type"]
        self.convnets = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.aux_fc = None
        self.task_sizes = []
        self.args = args

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    def extract_vector(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)

        out = self.fc(features)  # {logics: self.fc(features)}

        aux_logits = self.aux_fc(features[:, -self.out_dim :])["logits"]

        out.update({"aux_logits": aux_logits, "features": features})
        return out
        """
        {
            'features': features
            'logits': logits
            'aux_logits':aux_logits
        }
        """

    def update_fc(self, nb_classes):
        if len(self.convnets) == 0:
            self.convnets.append(get_convnet(self.args))
        else:
            self.convnets.append(get_convnet(self.args))
            self.convnets[-1].load_state_dict(self.convnets[-2].state_dict())

        if self.out_dim is None:
            self.out_dim = self.convnets[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)

        self.aux_fc = self.generate_fc(self.out_dim, new_task_size + 1)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def load_checkpoint(self, args):
        checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        model_infos = torch.load(checkpoint_name)
        assert len(self.convnets) == 1
        self.convnets[0].load_state_dict(model_infos['convnet'])
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc


class SimpleCosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:

                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc


class FOSTERNet(nn.Module):
    def __init__(self, args, pretrained):
        super(FOSTERNet, self).__init__()
        self.convnet_type = args["convnet_type"]
        self.convnets = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.fe_fc = None
        self.task_sizes = []
        self.oldfc = None
        self.args = args

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    def extract_vector(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        out = self.fc(features)
        fe_logits = self.fe_fc(features[:, -self.out_dim :])["logits"]

        out.update({"fe_logits": fe_logits, "features": features})

        if self.oldfc is not None:
            old_logits = self.oldfc(features[:, : -self.out_dim])["logits"]
            out.update({"old_logits": old_logits})

        out.update({"eval_logits": out["logits"]})
        return out

    def update_fc(self, nb_classes):
        self.convnets.append(get_convnet(self.args))
        if self.out_dim is None:
            self.out_dim = self.convnets[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias
            self.convnets[-1].load_state_dict(self.convnets[-2].state_dict())

        self.oldfc = self.fc
        self.fc = fc
        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.fe_fc = self.generate_fc(self.out_dim, nb_classes)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def copy_fc(self, fc):
        weight = copy.deepcopy(fc.weight.data)
        bias = copy.deepcopy(fc.bias.data)
        n, m = weight.shape[0], weight.shape[1]
        self.fc.weight.data[:n, :m] = weight
        self.fc.bias.data[:n] = bias

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

    def weight_align(self, old, increment, value):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew * (value ** (old / increment))
        logging.info("align weights, gamma = {} ".format(gamma))
        self.fc.weight.data[-increment:, :] *= gamma
    
    def load_checkpoint(self, args):
        if args["init_cls"] == 50:
            pkl_name = "{}_{}_{}_B{}_Inc{}".format( 
                args["dataset"],
                args["seed"],
                args["convnet_type"],
                0,
                args["init_cls"],
            )
            checkpoint_name = f"checkpoints/finetune_{pkl_name}_0.pkl"
        else:
            checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        model_infos = torch.load(checkpoint_name)
        assert len(self.convnets) == 1
        self.convnets[0].load_state_dict(model_infos['convnet'])
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc
    

class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x , bias=True):
        ret_x = x.clone()
        ret_x = (self.alpha+1) * x # + self.beta
        if bias:
            ret_x = ret_x + self.beta
        return ret_x

    def get_params(self):
        return (self.alpha.item(), self.beta.item())


class BEEFISONet(nn.Module):
    def __init__(self, args, pretrained):
        super(BEEFISONet, self).__init__()
        self.convnet_type = args["convnet_type"]
        self.convnets = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.old_fc = None
        self.new_fc = None
        self.task_sizes = []
        self.forward_prototypes = None
        self.backward_prototypes = None
        self.args = args
        self.biases = nn.ModuleList()

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    def extract_vector(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        
        if self.old_fc is None:
            fc = self.new_fc
            out = fc(features)
        else:
            '''
            merge the weights
            '''
            new_task_size = self.task_sizes[-1]
            fc_weight = torch.cat([self.old_fc.weight,torch.zeros((new_task_size,self.feature_dim-self.out_dim)).cuda()],dim=0)             
            new_fc_weight = self.new_fc.weight
            new_fc_bias = self.new_fc.bias
            for i in range(len(self.task_sizes)-2,-1,-1):
                new_fc_weight = torch.cat([*[self.biases[i](self.backward_prototypes.weight[i].unsqueeze(0),bias=False) for _ in range(self.task_sizes[i])],new_fc_weight],dim=0)
                new_fc_bias = torch.cat([*[self.biases[i](self.backward_prototypes.bias[i].unsqueeze(0),bias=True) for _ in range(self.task_sizes[i])], new_fc_bias])
            fc_weight = torch.cat([fc_weight,new_fc_weight],dim=1)
            fc_bias = torch.cat([self.old_fc.bias,torch.zeros(new_task_size).cuda()])
            fc_bias=+new_fc_bias
            logits = features@fc_weight.permute(1,0)+fc_bias
            out = {"logits":logits}        

            new_fc_weight = self.new_fc.weight
            new_fc_bias = self.new_fc.bias
            for i in range(len(self.task_sizes)-2,-1,-1):
                new_fc_weight = torch.cat([self.backward_prototypes.weight[i].unsqueeze(0),new_fc_weight],dim=0)
                new_fc_bias = torch.cat([self.backward_prototypes.bias[i].unsqueeze(0), new_fc_bias])
            out["train_logits"] = features[:,-self.out_dim:]@new_fc_weight.permute(1,0)+new_fc_bias 
        out.update({"eval_logits": out["logits"],"energy_logits":self.forward_prototypes(features[:,-self.out_dim:])["logits"]})
        return out

    def update_fc_before(self, nb_classes):
        new_task_size = nb_classes - sum(self.task_sizes)
        self.biases = nn.ModuleList([BiasLayer() for i in range(len(self.task_sizes))])
        self.convnets.append(get_convnet(self.args))
        if self.out_dim is None:
            self.out_dim = self.convnets[-1].out_dim
        if self.new_fc is not None:
            self.fe_fc = self.generate_fc(self.out_dim, nb_classes)
            self.backward_prototypes = self.generate_fc(self.out_dim,len(self.task_sizes))
            self.convnets[-1].load_state_dict(self.convnets[0].state_dict())
        self.forward_prototypes = self.generate_fc(self.out_dim, nb_classes)
        self.new_fc = self.generate_fc(self.out_dim,new_task_size)
        self.task_sizes.append(new_task_size)
    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc
    
    def update_fc_after(self):
        if self.old_fc is not None:
            old_fc = self.generate_fc(self.feature_dim, sum(self.task_sizes))
            new_task_size = self.task_sizes[-1]
            old_fc.weight.data = torch.cat([self.old_fc.weight.data,torch.zeros((new_task_size,self.feature_dim-self.out_dim)).cuda()],dim=0)             
            new_fc_weight = self.new_fc.weight.data
            new_fc_bias = self.new_fc.bias.data
            for i in range(len(self.task_sizes)-2,-1,-1):
                new_fc_weight = torch.cat([*[self.biases[i](self.backward_prototypes.weight.data[i].unsqueeze(0),bias=False) for _ in range(self.task_sizes[i])], new_fc_weight],dim=0)
                new_fc_bias = torch.cat([*[self.biases[i](self.backward_prototypes.bias.data[i].unsqueeze(0),bias=True) for _ in range(self.task_sizes[i])], new_fc_bias])
            old_fc.weight.data = torch.cat([old_fc.weight.data,new_fc_weight],dim=1)
            old_fc.bias.data = torch.cat([self.old_fc.bias.data,torch.zeros(new_task_size).cuda()])
            old_fc.bias.data+=new_fc_bias
            self.old_fc = old_fc
        else:
            self.old_fc  = self.new_fc

    def copy(self):
        return copy.deepcopy(self)

    def copy_fc(self, fc):
        weight = copy.deepcopy(fc.weight.data)
        bias = copy.deepcopy(fc.bias.data)
        n, m = weight.shape[0], weight.shape[1]
        self.fc.weight.data[:n, :m] = weight
        self.fc.bias.data[:n] = bias

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

    def weight_align(self, old, increment, value):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew * (value ** (old / increment))
        logging.info("align weights, gamma = {} ".format(gamma))
        self.fc.weight.data[-increment:, :] *= gamma


class AdaptiveNet(nn.Module):
    def __init__(self, args, pretrained):
        super(AdaptiveNet, self).__init__()
        self.convnet_type = args["convnet_type"]
        self.TaskAgnosticExtractor , _ = get_convnet(args, pretrained) #Generalized blocks
        self.TaskAgnosticExtractor.train()
        self.AdaptiveExtractors = nn.ModuleList() #Specialized Blocks
        self.pretrained=pretrained
        self.out_dim=None
        self.fc = None
        self.aux_fc=None
        self.task_sizes = []
        self.args=args

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim*len(self.AdaptiveExtractors)
    
    def extract_vector(self, x):
        base_feature_map = self.TaskAgnosticExtractor(x)
        features = [extractor(base_feature_map) for extractor in self.AdaptiveExtractors]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        base_feature_map = self.TaskAgnosticExtractor(x)
        features = [extractor(base_feature_map) for extractor in self.AdaptiveExtractors]
        features = torch.cat(features, 1)
        out=self.fc(features) #{logits: self.fc(features)}

        aux_logits=self.aux_fc(features[:,-self.out_dim:])["logits"] 

        out.update({"aux_logits":aux_logits,"features":features})
        out.update({"base_features":base_feature_map})
        return out
                
        '''
        {
            'features': features
            'logits': logits
            'aux_logits':aux_logits
        }
        '''
        
    def update_fc(self,nb_classes):
        _ , _new_extractor = get_convnet(self.args)
        if len(self.AdaptiveExtractors)==0:
            self.AdaptiveExtractors.append(_new_extractor)
        else:
            self.AdaptiveExtractors.append(_new_extractor)
            self.AdaptiveExtractors[-1].load_state_dict(self.AdaptiveExtractors[-2].state_dict())

        if self.out_dim is None:
            logging.info(self.AdaptiveExtractors[-1])
            self.out_dim=self.AdaptiveExtractors[-1].feature_dim        
        fc = self.generate_fc(self.feature_dim, nb_classes)             
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output,:self.feature_dim-self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.aux_fc=self.generate_fc(self.out_dim,new_task_size+1)
 
    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def weight_align(self, increment):
        weights=self.fc.weight.data
        newnorm=(torch.norm(weights[-increment:,:],p=2,dim=1))
        oldnorm=(torch.norm(weights[:-increment,:],p=2,dim=1))
        meannew=torch.mean(newnorm)
        meanold=torch.mean(oldnorm)
        gamma=meanold/meannew
        print('alignweights,gamma=',gamma)
        self.fc.weight.data[-increment:,:]*=gamma
    
    def load_checkpoint(self, args):
        if args["init_cls"] == 50:
            pkl_name = "{}_{}_{}_B{}_Inc{}".format( 
                args["dataset"],
                args["seed"],
                args["convnet_type"],
                0,
                args["init_cls"],
            )
            checkpoint_name = f"checkpoints/finetune_{pkl_name}_0.pkl"
        else:
            checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        checkpoint_name = checkpoint_name.replace("memo_", "")
        model_infos = torch.load(checkpoint_name)
        model_dict = model_infos['convnet']
        assert len(self.AdaptiveExtractors) == 1

        base_state_dict = self.TaskAgnosticExtractor.state_dict()
        adap_state_dict = self.AdaptiveExtractors[0].state_dict()

        pretrained_base_dict = {
            k:v
            for k, v in model_dict.items()
            if k in base_state_dict
        }

        pretrained_adap_dict = {
            k:v
            for k, v in model_dict.items()
            if k in adap_state_dict
        }

        base_state_dict.update(pretrained_base_dict)
        adap_state_dict.update(pretrained_adap_dict)

        self.TaskAgnosticExtractor.load_state_dict(base_state_dict)
        self.AdaptiveExtractors[0].load_state_dict(adap_state_dict)
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc