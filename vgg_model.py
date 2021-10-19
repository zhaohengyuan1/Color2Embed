from torchvision import models
from collections import namedtuple
import torch
import torch.nn as nn

def vgg_preprocess(tensor):
    # input is RGB tensor which ranges in [0,1]
    # output is RGB tensor which ranges
    mean_val = torch.Tensor([0.485, 0.456, 0.406]).type_as(tensor).view(-1, 1, 1)
    std_val = torch.Tensor([0.229, 0.224, 0.225]).type_as(tensor).view(-1, 1, 1)
    tensor_norm = (tensor - mean_val) / std_val
    return tensor_norm

class vgg19(nn.Module):
    
    def __init__(self, pretrained_path = '/mnt/hyzhao/Documents/Color2Style/DEVC/data/vgg19-dcbb9e9d.pth', require_grad = False):
        super(vgg19, self).__init__()
        self.vgg_model = models.vgg19()
        if pretrained_path != None:
            print('----load pretrained vgg19----')
            self.vgg_model.load_state_dict(torch.load(pretrained_path))
            print('----load done!----')
        self.vgg_feature = self.vgg_model.features
        self.seq_list = [nn.Sequential(ele) for ele in self.vgg_feature]
        # self.vgg_layer = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 
        #                  'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        #                  'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        #                  'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        #                  'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5']

        # self.vgg_layer = ['relu1_2', 'relu2_2', 'relu3_2', 'relu4_2', 'relu5_2']
        
        if not require_grad:
            for parameter in self.parameters():
                parameter.requires_grad = False
        
    def forward(self, x, layer_name='relu5_2'):
        ### x: RGB [0, 1], input should be [0, 1]
        x = vgg_preprocess(x)

        conv1_1 = self.seq_list[0](x)
        relu1_1 = self.seq_list[1](conv1_1)
        conv1_2 = self.seq_list[2](relu1_1)
        relu1_2 = self.seq_list[3](conv1_2)
        pool1 = self.seq_list[4](relu1_2)
        
        conv2_1 = self.seq_list[5](pool1)
        relu2_1 = self.seq_list[6](conv2_1)
        conv2_2 = self.seq_list[7](relu2_1)
        relu2_2 = self.seq_list[8](conv2_2)
        pool2 = self.seq_list[9](relu2_2)
        
        conv3_1 = self.seq_list[10](pool2)
        relu3_1 = self.seq_list[11](conv3_1)
        conv3_2 = self.seq_list[12](relu3_1)
        relu3_2 = self.seq_list[13](conv3_2)
        conv3_3 = self.seq_list[14](relu3_2)
        relu3_3 = self.seq_list[15](conv3_3)
        conv3_4 = self.seq_list[16](relu3_3)
        relu3_4 = self.seq_list[17](conv3_4)
        pool3 = self.seq_list[18](relu3_4)
        
        conv4_1 = self.seq_list[19](pool3)
        relu4_1 = self.seq_list[20](conv4_1)
        conv4_2 = self.seq_list[21](relu4_1)
        relu4_2 = self.seq_list[22](conv4_2)
        conv4_3 = self.seq_list[23](relu4_2)
        relu4_3 = self.seq_list[24](conv4_3)
        conv4_4 = self.seq_list[25](relu4_3)
        relu4_4 = self.seq_list[26](conv4_4)
        pool4 = self.seq_list[27](relu4_4)
        
        conv5_1 = self.seq_list[28](pool4)
        relu5_1 = self.seq_list[29](conv5_1)
        conv5_2 = self.seq_list[30](relu5_1)
        relu5_2 = self.seq_list[31](conv5_2) # [B, 512, 16, 16]
        conv5_3 = self.seq_list[32](relu5_2)
        relu5_3 = self.seq_list[33](conv5_3)
        conv5_4 = self.seq_list[34](relu5_3)
        relu5_4 = self.seq_list[35](conv5_4)
        pool5 = self.seq_list[36](relu5_4) # [B, 512, 8, 8]
        
        # vgg_output = namedtuple("vgg_output", self.vgg_layer)
        
        # vgg_list = [conv1_1, relu1_1, conv1_2, relu1_2, pool1, 
        #                  conv2_1, relu2_1, conv2_2, relu2_2, pool2,
        #                  conv3_1, relu3_1, conv3_2, relu3_2, conv3_3, relu3_3, conv3_4, relu3_4, pool3,
        #                  conv4_1, relu4_1, conv4_2, relu4_2, conv4_3, relu4_3, conv4_4, relu4_4, pool4,
        #                  conv5_1, relu5_1, conv5_2, relu5_2, conv5_3, relu5_3, conv5_4, relu5_4, pool5]

        if layer_name == 'relu5_2':
            vgg_list = [relu5_2]
        elif layer_name == 'conv5_2':
            vgg_list = [conv5_2]
        elif layer_name == 'relu5_4':
            vgg_list = [relu5_4]
        elif layer_name == 'pool5':
            # print('pool5')
            vgg_list = [pool5]
        elif layer_name == 'all':
            vgg_list = [relu1_2, relu2_2, relu3_2, relu4_2, relu5_2]
        
        # out = vgg_output(*vgg_list)
        
        return vgg_list

class vgg19_class_fea(nn.Module):
    
    def __init__(self, pretrained_path = './DEVC/data/vgg19-dcbb9e9d.pth', require_grad = False):
        super(vgg19_class_fea, self).__init__()
        self.vgg_model = models.vgg19()
        print('----load pretrained vgg19----')
        self.vgg_model.load_state_dict(torch.load(pretrained_path))
        print('----load done!----')
        self.vgg_feature = self.vgg_model.features
        self.avgpool = self.vgg_model.avgpool
        self.classifier = self.vgg_model.classifier

        self.seq_list = [nn.Sequential(ele) for ele in self.vgg_feature] # 37层
        if not require_grad:
            for parameter in self.parameters():
                parameter.requires_grad = False
        
    def forward(self, x):
        ### x: RGB [0, 1], input should be [0, 1]
        x = vgg_preprocess(x)

        for i in range(len(self.seq_list)):
            x = self.seq_list[i](x)
            if i == 31:
                relu5_2 = x
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x_class = self.classifier(x)
        return x_class, relu5_2