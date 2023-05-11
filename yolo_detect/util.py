from __future__ import division
import shutil
import numpy as np
import torch
from path import Path
import datetime
from collections import OrderedDict
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from IPython import display
import itertools
import torch.nn as nn
import os
from torchvision.utils import save_image
import imageio
import xml.etree.ElementTree as ET
import pathlib
from collections import OrderedDict

def save_path_formatter(args, parser):
    def is_default(key, value):
        return value == parser.get_default(key)
    args_dict = vars(args)
    data_folder_name = str(Path(args_dict['data_path']).normpath().name)
    folder_string = [data_folder_name]
    if not is_default('epochs', args_dict['epochs']):
        folder_string.append('{}epochs'.format(args_dict['epochs']))
    keys_with_prefix = OrderedDict()
    keys_with_prefix['epoch_size'] = 'epoch_size'
    keys_with_prefix['batch_size'] = 'b'
    keys_with_prefix['lr'] = 'lr'

    for key, prefix in keys_with_prefix.items():
        value = args_dict[key]
        if not is_default(key, value):
            folder_string.append('{}{}'.format(prefix, value))
    save_path = Path(','.join(folder_string))
    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    return save_path/timestamp

def tensor2array(tensor, max_value=255, colormap='rainbow'):
    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        try:
            import cv2
            if cv2.__version__.startswith('3'):
                color_cvt = cv2.COLOR_BGR2RGB
            else:  # 2.4
                color_cvt = cv2.cv.CV_BGR2RGB
            if colormap == 'rainbow':
                colormap = cv2.COLORMAP_RAINBOW
            elif colormap == 'bone':
                colormap = cv2.COLORMAP_BONE
            array = (255*tensor.squeeze().numpy()/max_value).clip(0, 255).astype(np.uint8)
            colored_array = cv2.applyColorMap(array, colormap)
            array = cv2.cvtColor(colored_array, color_cvt).astype(np.float32)/255
        except ImportError:
            if tensor.ndimension() == 2:
                tensor.unsqueeze_(2)
            array = (tensor.expand(tensor.size(0), tensor.size(1), 3).numpy()/max_value).clip(0,1)

    elif tensor.ndimension() == 3:
        assert(tensor.size(0) == 3)
        array = 0.5 + tensor.numpy()*0.5
        array = array.transpose(1,2,0)
    return array

def save_depth_tensor(tensor_img,img_dir,filename):
    result = tensor_img.cpu().detach().numpy()
    max_value = result.max()
    if (result.shape[0]==1):
        result = result.squeeze(0)
        result = result/max_value
    elif (result.ndim==2):
        result = result/max_value
    else:
        print("file dimension is not proper!!")
        exit()
    imageio.imwrite(img_dir + '/' + filename,result)

def plot_loss(data, apath, epoch,train,filename):
    axis = np.linspace(1, epoch, epoch)
    
    label = 'Total Loss'
    fig = plt.figure()
    plt.title(label)
    plt.plot(axis, np.array(data), label=label)
    plt.legend()
    if train is False:
        plt.xlabel('Epochs')
    else:
        plt.xlabel('x100 = Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.savefig(os.path.join(apath, filename))
    plt.close(fig)
    plt.close('all')

def train_plot(save_dir,tot_loss, rmse, loss_list, rmse_list, tot_loss_dir,rmse_dir,loss_pdf, rmse_pdf, count,istrain):
    open_type = 'a' if os.path.exists(tot_loss_dir) else 'w'
    loss_log_file = open(tot_loss_dir, open_type)
    rmse_log_file = open(rmse_dir,open_type)

    loss_list.append(tot_loss)
    rmse_list.append(rmse)

    plot_loss(loss_list, save_dir, count, istrain, loss_pdf)
    plot_loss(rmse_list, save_dir, count, istrain, rmse_pdf)
    loss_log_file.write(('%.5f'%tot_loss) + '\n')
    rmse_log_file.write(('%.5f'%rmse) + '\n')
    loss_log_file.close()
    rmse_log_file.close()

def validate_plot(save_dir,tot_loss, loss_list, tot_loss_dir,loss_pdf, count,istrain):
    open_type = 'a' if os.path.exists(tot_loss_dir) else 'w'
    loss_log_file = open(tot_loss_dir, open_type)

    loss_list.append(tot_loss)

    plot_loss(loss_list, save_dir, count, istrain, loss_pdf)
    loss_log_file.write(('%.5f'%tot_loss) + '\n')
    loss_log_file.close()

def imgrad(img):
    img = torch.mean(img, 1, True)
    fx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv1.weight = nn.Parameter(weight)
    grad_x = conv1(img)

    fy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)

    if img.is_cuda:
        weight = weight.cuda()

    conv2.weight = nn.Parameter(weight)
    grad_y = conv2(img)
    return grad_y, grad_x

def imgrad_loss(pred, gt, mask=None):
    N,C,_,_ = pred.size()
    grad_y, grad_x = imgrad(pred)
    grad_y_gt, grad_x_gt = imgrad(gt)
    grad_y_diff = torch.abs(grad_y - grad_y_gt)
    grad_x_diff = torch.abs(grad_x - grad_x_gt)
    if mask is not None:
        grad_y_diff[~mask] = 0.1*grad_y_diff[~mask]
        grad_x_diff[~mask] = 0.1*grad_x_diff[~mask]
    return (torch.mean(grad_y_diff) + torch.mean(grad_x_diff))

def BerHu_loss(valid_out, valid_gt):         
    diff = valid_out - valid_gt
    diff_abs = torch.abs(diff)
    c = 0.2*torch.max(diff_abs.detach())         
    mask2 = torch.gt(diff_abs.detach(),c)
    diff_abs[mask2] = (diff_abs[mask2]**2 +(c*c))/(2*c)
    return diff_abs.mean()

def scale_invariant_loss(valid_out, valid_gt):
    logdiff = torch.log(valid_out) - torch.log(valid_gt)
    scale_inv_loss = torch.sqrt((logdiff ** 2).mean() - 0.85*(logdiff.mean() ** 2))*10.0
    return scale_inv_loss

def make_mask(depths, crop_mask, dataset):
    # masking valied area
    if dataset == 'KITTI':
        valid_mask = depths > 0.001
    else:
        valid_mask = depths > 0.001
    
    if dataset == "KITTI":
        if(crop_mask.size(0) != valid_mask.size(0)):
            crop_mask = crop_mask[0:valid_mask.size(0),:,:,:]
        final_mask = crop_mask|valid_mask
    else:
        final_mask = valid_mask
        
    return valid_mask, final_mask

def parse_xml(img_file, out):
    h, w = out.shape
    path = pathlib.Path(img_file)
    xml_file = path.parent / (path.stem + '.xml')
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find('size')
    org_w = int(size.find('width').text)
    org_h = int(size.find('height').text)
    res = {
        'window': [],
        'people': []
    }
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in ['window', 'people']:
            continue
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymin').text), int(xmlbox.find('ymax').text))
        if cls in res:
            res[cls].append((b[2] * h // org_h, b[3] * h // org_h, b[0] * w // org_w, b[1] * w // org_w))   # 转换成模型的输出图片大小 (h, w)
    
    if res['people'] is [] or res['window'] is []:  # 图片中不存在窗或人
        return False
    
    pos_windows = res['window']
    depth_windows = []  # 窗户的平均相对深度
    center_windows = [] # 窗户的中心位置 [0, 1] 之间 (y方向的值目前应该用不上, 但先加了)
    for pos_window in pos_windows:
        window = out[pos_window[0]:pos_window[1], pos_window[2]:pos_window[3]]
        depth_window = window.mean() / 255.0
        center_window = ((pos_window[0] + pos_window[1]) / 2 / h, (pos_window[2] + pos_window[3]) / 2 / w)
        depth_windows.append(depth_window)
        center_windows.append(center_window)
    
    pos_peoples = res['people']
    depth_peoples = []  # 人的平均相对深度
    center_peoples = [] # 人的中心位置 [0, 1] 之间
    for pos_people in pos_peoples:
        people = out[pos_people[0]:pos_people[1], pos_people[2]:pos_people[3]]
        depth_people = people.mean() / 255.0
        center_people = ((pos_people[0] + pos_people[1]) / 2 / h, (pos_people[2] + pos_people[3]) / 2 / w)
        depth_peoples.append(depth_people)
        center_peoples.append(center_people)
        
    distace = {}
    for i in range(len(depth_windows)):
        for j in range(len(depth_peoples)):
            distace[f'窗{i + 1}->人{j + 1}'] = (abs(depth_windows[i] - depth_peoples[j]) ** 2 + abs(center_windows[i][1] - center_peoples[j][1]) ** 2) ** 0.5
            
    return distace


def parse(result, out):
    h, w = out.shape
    # img_path = pathlib.Path(result['img_path'])
    detect_res = {
        'window': [],
        'people': []
    }
    
    for obj in result['res']:   # 遍历检测结果
        cls = obj['className']
        # if cls not in ['window-unprotected', 'window-protection', 'window-glass', 'people']:    # 若不是窗户或人则直接跳过
        #     continue
        if cls == 'people':
            detect_res['people'].append((int(obj['y'] * h), int((obj['y'] + obj['height']) * h), int(obj['x'] * w), int((obj['x'] + obj['width']) * w))) # 若检测框在图片边缘, 使用int舍入是否会越界?
        if cls in ['window-unprotected', 'window-protection', 'window-glass']:
            detect_res['window'].append((int(obj['y'] * h), int((obj['y'] + obj['height']) * h), int(obj['x'] * w), int((obj['x'] + obj['width']) * w)))
    print(detect_res)
    if detect_res['people'] == [] or detect_res['window'] == []:  # 图片中不存在窗或人
        return False
    
    pos_windows = detect_res['window']
    depth_windows = []  # 窗户的平均相对深度
    center_windows = [] # 窗户的中心位置 [0, 1] 之间 (y方向的值目前应该用不上, 但先加了)
    for pos_window in pos_windows:
        window = out[pos_window[0]:pos_window[1], pos_window[2]:pos_window[3]]
        depth_window = window.mean() / 255.0
        center_window = ((pos_window[0] + pos_window[1]) / 2 / h, (pos_window[2] + pos_window[3]) / 2 / w)
        depth_windows.append(depth_window)
        center_windows.append(center_window)
        
    pos_peoples = detect_res['people']
    depth_peoples = []  # 人的平均相对深度
    center_peoples = [] # 人的中心位置 [0, 1] 之间
    for pos_people in pos_peoples:
        people = out[pos_people[0]:pos_people[1], pos_people[2]:pos_people[3]]
        depth_people = people.mean() / 255.0
        center_people = ((pos_people[0] + pos_people[1]) / 2 / h, (pos_people[2] + pos_people[3]) / 2 / w)
        depth_peoples.append(depth_people)
        center_peoples.append(center_people)
        
    distace = {}
    for i in range(len(depth_windows)):
        for j in range(len(depth_peoples)):
            distace[f'窗{i + 1}->人{j + 1}'] = (abs(depth_windows[i] - depth_peoples[j]) ** 2 + abs(center_windows[i][1] - center_peoples[j][1]) ** 2) ** 0.5

    return distace