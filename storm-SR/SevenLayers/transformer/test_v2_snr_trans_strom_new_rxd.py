import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import time
import datetime
from network import Network_3D_Unet
import numpy as np
from utils import save_yaml_test
import os
import math
import tifffile as tiff
from data_process import test_preprocess_lessMemoryNoTail_chooseOne, testset, multibatch_test_save, \
    singlebatch_test_save
from skimage import io
from basemodel_trans import EncodeTrans_V3DUNet, EncodeTrans_V3DUNet_Swin_Trans, EncodeTrans_V3DUNet_Swin, \
    EncodeTrans_V3DUNet_Swin_EnDe
# from TransBTS.TransBTS_downsample8x_skipconnection import BTS
# from TransBTS.TransBTS_downsample8x_skipconnection_paras import BTS
# from TransBTS.TransBTS_downsample8x_skipconnection_time import BTS
from unet3d import Modified3DUNet as UNet3D
from SevenLayers.TimeSpaceTrans_7layers import TimeSpaceTrans_7layers
# from SevenLayers.TimeSpaceTrans_7layersV2 import TimeSpaceTrans_7layersV2
from SevenLayers.TimeSpaceTrans_7layersV2 import TimeSpaceTrans_7layersV2
from SevenLayers.TimeSpaceTrans_7layersV3_test import TimeSpaceTrans_7layersV3
from neighbor2neighbor import *


def cal_snr(noise_img, clean_img):
    noise_signal = noise_img - clean_img
    clean_signal = clean_img
    noise_signal_2 = noise_signal ** 2
    clean_signal_2 = clean_signal ** 2
    sum1 = np.sum(clean_signal_2)
    sum2 = np.sum(noise_signal_2)
    snrr = 20 * math.log10(math.sqrt(sum1) / math.sqrt(sum2))
    return snrr


#############################################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of training epochs")
parser.add_argument('--GPU', type=str, default='0,1', help="the index of GPU you will use for computation")

parser.add_argument('--patch_x', type=int, default=150,
                    help="the width of 3D patches (patch size in x, also in y and t)")
parser.add_argument('--patch_t', type=int, default=150, help="the width of 3D patches (patch size in t)")
parser.add_argument('--overlap_factor', type=float, default=0.4, help="the overlap factor between two adjacent patches")

parser.add_argument('--lr', type=float, default=0.00005, help='initial learning rate')
parser.add_argument("--b1", type=float, default=0.5, help="Adam: bata1")
parser.add_argument("--b2", type=float, default=0.999, help="Adam: bata2")
parser.add_argument('--fmap', type=int, default=16, help='number of feature maps')

parser.add_argument('--scale_factor', type=int, default=1, help='the factor for image intensity scaling')
parser.add_argument('--datasets_path', type=str, default='datasets', help="dataset root path")
parser.add_argument('--datasets_folder', type=str, default='test', help="A folder containing files to be tested")
parser.add_argument('--output_dir', type=str, default='./results', help="output directory")
parser.add_argument('--pth_path', type=str, default='pth', help="pth file root path")

parser.add_argument('--denoise_model', type=str, default='train_20210401_1712',
                    help='A folder containing models to be tested')
parser.add_argument('--test_datasize', type=int, default=1000000, help='how many slices to be tested')
opt = parser.parse_args()

# use isotropic patch size by default
opt.patch_y = opt.patch_x  # the height of 3D patches (patch size in y)
opt.patch_t = opt.patch_t  # the length of 3D patches (patch size in t)
# opt.gap_t (image gap) is the distance between two adjacent patches
# use isotropic opt.gap by default
opt.gap_t = int(opt.patch_t * (1 - opt.overlap_factor))
opt.gap_x = int(opt.patch_x * (1 - opt.overlap_factor))
opt.gap_y = int(opt.patch_y * (1 - opt.overlap_factor))
opt.ngpu = str(opt.GPU).count(',') + 1
opt.batch_size = opt.ngpu
print('\033[1;31mParameters -----> \033[0m')
print(opt)

########################################################################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.GPU)
model_path = opt.pth_path + '//' + opt.denoise_model
# print(model_path)
model_list = list(os.walk(model_path, topdown=False))[-1][-1]
model_list.sort()
# print(model_list)


# read paremeters from file
for i in range(len(model_list)):
    aaa = model_list[i]
    if '.yaml' in aaa:
        yaml_name = model_list[i]
        del model_list[i]

# get stacks for processing
im_folder = opt.datasets_path + '//' + opt.datasets_folder
img_list = list(os.walk(im_folder, topdown=False))[-1][-1]
#print("img_list", img_list)
img_list.sort()
model_list = [model_list[99]]
print('\033[1;31mStacks for processing -----> \033[0m')
print('Total stack umber -----> ', len(img_list))
#for img in img_list: print(img)

if not os.path.exists(opt.output_dir):
    os.mkdir(opt.output_dir)
current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
output_path1 = opt.output_dir + '//' + 'DataFolderIs_' + opt.datasets_folder + '_' + current_time + '_ModelFolderIs_' + opt.denoise_model
if not os.path.exists(output_path1):
    os.mkdir(output_path1)

yaml_name = output_path1 + '//para.yaml'
save_yaml_test(opt, yaml_name)

##############################################################################################################################################################
'''
# network architecture and GPU access
denoise_generator = EncodeTrans_V3DUNet(1,1,max_seq=opt.patch_t,num_levels=4,layer_order='cr',dropout=0.1,
                              f_maps=opt.fmap,tran_enc_layers=1,pos_embedway='sincos',use_entmax15='adaptive_entmax',
                              spatial_size=(opt.patch_y,opt.patch_y),choose_translayer=[0,1,1,1,0,0,0],
                              conv_kernel_size=(1,3,3),conv_padding=(0,1,1))

denoise_generator = BTS(
        img_dim=opt.patch_x,
        img_time=opt.patch_t,
        patch_dim = 8,
        num_channels = 4,
        num_classes = 1,
        embedding_dim=512,
        num_heads=8,
        num_layers=4,
        hidden_dim=4096,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        conv_patch_representation=True,
        positional_encoding_type="learned",
    )


denoise_generator = TimeSpaceTrans_7layers(
    img_dim=opt.patch_x,
    img_time=opt.patch_t,
    in_channel=1,
    embedding_dim=128,
    num_heads=8,
    hidden_dim=128*4,
    window_size=7,
    num_transBlock=1,
    attn_dropout_rate=0.1,
    f_maps=[16, 32, 64],
    input_dropout_rate=0
)

denoise_generator = TimeSpaceTrans_7layersV2(
    img_dim=opt.patch_x,
    img_time=opt.patch_t,
    in_channel=1,
    embedding_dim=128,
    num_heads=8,
    hidden_dim=128*4,
    window_size=7,
    num_transBlock=2,
    attn_dropout_rate=0.1,
    f_maps=[16, 32, 64],
    input_dropout_rate=0
)

denoise_generator = TimeSpaceTrans_7layersV2(
    img_dim=opt.patch_x,
    img_time=opt.patch_t,
    in_channel=1,
    embedding_dim=128,
    num_heads=8,
    hidden_dim=128 * 4,
    window_size=7,
    num_transBlock=2,
    attn_dropout_rate=0.1,
    f_maps=[32, 64],
    input_dropout_rate=0
)


denoise_generator = Network_3D_Unet(in_channels = 1,
                                    out_channels = 1,
                                    f_maps=opt.fmap,
                                    final_sigmoid = True)

denoise_generator = TimeSpaceTrans_7layersV2(
    img_dim=opt.patch_x,
    img_time=opt.patch_t,
    in_channel=1,
    embedding_dim=128,
    num_heads=4,
    hidden_dim=128 * 4,
    window_size=7,
    num_transBlock=1,
    attn_dropout_rate=0.1,
    f_maps=[16, 32, 64],
    input_dropout_rate=0
)

denoise_generator = TimeSpaceTrans_7layersV3(
    img_dim=opt.patch_x,
    img_time=opt.patch_t,
    in_channel=1,
    embedding_dim=128,
    num_heads=4,
    hidden_dim=128*4,
    window_size=7,
    num_transBlock=1,
    attn_dropout_rate=0.1,
    f_maps=[16, 32, 64],
    input_dropout_rate=0
)

denoise_generator = Network_3D_Unet(in_channels = 1,
                                    out_channels = 1,
                                    f_maps=opt.fmap,
                                    final_sigmoid = True)


denoise_generator = TimeSpaceTrans_7layersV2(
    img_dim=opt.patch_x,
    img_time=opt.patch_t,
    in_channel=1,
    embedding_dim=128,
    num_heads=8,
    hidden_dim=128 * 4,
    window_size=7,
    num_transBlock=1,
    attn_dropout_rate=0.1,
    f_maps=[16, 32, 64],
    input_dropout_rate=0
)

denoise_generator = Network_3D_Unet(in_channels = 1,
                                    out_channels = 1,
                                    f_maps=opt.fmap,
                                    final_sigmoid = True)

'''
denoise_generator = TimeSpaceTrans_7layersV2(
    img_dim=opt.patch_x,
    img_time=opt.patch_t,
    in_channel=1,
    embedding_dim=128,
    num_heads=8,
    hidden_dim=128*4,
    window_size=7,
    num_transBlock=2,
    attn_dropout_rate=0.1,
    f_maps=[16, 32, 64],
    input_dropout_rate=0
)

'''
denoise_generator = TimeSpaceTrans_7layersV2(
    img_dim=opt.patch_x,
    img_time=opt.patch_t,
    in_channel=1,
    embedding_dim=128,
    num_heads=4,
    hidden_dim=128*4,
    window_size=3,
    num_transBlock=1,
    attn_dropout_rate=0.1,
    f_maps=[16, 32, 64],
    input_dropout_rate=0
)
'''
# denoise_generator = UNet3D(in_channels=1, num_classes=1)

# EncodeTrans_V3DUNet_Swin_Trans
# denoise_generator.eval()

if torch.cuda.is_available():
    print('\033[1;31mUsing {} GPU for testing -----> \033[0m'.format(torch.cuda.device_count()))
    denoise_generator = denoise_generator.cuda()
    denoise_generator = nn.DataParallel(denoise_generator, device_ids=range(opt.ngpu))
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
##############################################################################################################################################################

# Start processing
for pth_index in range(len(model_list)):
    aaa = model_list[pth_index]
    if '.pth' in aaa:
        pth_name = model_list[pth_index]
        output_path = output_path1 + '//' + pth_name.replace('.pth', '')
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # load model
        model_name = opt.pth_path + '//' + opt.denoise_model + '//' + pth_name
        if isinstance(denoise_generator, nn.DataParallel):
            denoise_generator.module.load_state_dict(torch.load(model_name))  # parallel
            denoise_generator.eval()
        else:
            denoise_generator.load_state_dict(torch.load(model_name))  # not parallel
            denoise_generator.eval()
        denoise_generator.cuda()

        # test all stacks
        for N in range(len(img_list)):
            name_list, noise_img, coordinate_list = test_preprocess_lessMemoryNoTail_chooseOne(opt, N)
            # print(len(name_list))
            prev_time = time.time()
            time_start = time.time()
            denoise_img = np.zeros(noise_img.shape)
            input_img = np.zeros(noise_img.shape)

            #print("coordinate_list length:", len(coordinate_list))
            test_data = testset(name_list, coordinate_list, noise_img)
            testloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=4)
            with torch.no_grad():
                for iteration, (noise_patch, single_coordinate) in enumerate(testloader):
                    noise_patch = noise_patch.cuda()
                    real_A = noise_patch
                    # print('real_A -----> ',real_A.shape)
                    # input_name = name_list[index]
                    # print(' input_name -----> ',input_name)
                    # print(' single_coordinate -----> ',single_coordinate)
                    # print('real_A -----> ',real_A.shape)
                    real_A = Variable(real_A)
                    fake_B = denoise_generator(real_A)
                    ################################################################################################################
                    # Determine approximate time left
                    batches_done = iteration
                    batches_left = 1 * len(testloader) - batches_done
                    time_left_seconds = int(batches_left * (time.time() - prev_time))
                    time_left = datetime.timedelta(seconds=time_left_seconds)
                    prev_time = time.time()
                    ################################################################################################################
                    if iteration % 1 == 0:
                        time_end = time.time()
                        time_cost = time_end - time_start  # datetime.timedelta(seconds= (time_end - time_start))
                        print(
                            '\r[Model %d/%d, %s] [Stack %d/%d, %s] [Patch %d/%d] [Time Cost: %.0d s] [ETA: %s s]     '
                            % (
                                pth_index + 1,
                                len(model_list),
                                pth_name,
                                N + 1,
                                len(img_list),
                                img_list[N],
                                iteration + 1,
                                len(testloader),
                                time_cost,
                                time_left_seconds
                            ), end=' ')

                    if (iteration + 1) % len(testloader) == 0:
                        print('\n', end=' ')
                    ################################################################################################################
                    output_image = np.squeeze(fake_B.cpu().detach().numpy())
                    raw_image = np.squeeze(real_A.cpu().detach().numpy())
                    if (output_image.ndim == 3):
                        turn = 1
                    else:
                        turn = output_image.shape[0]
                    # print(turn)
                    if (turn > 1):
                        for id in range(turn):
                            # print('shape of output_image -----> ',output_image.shape)
                            aaaa, bbbb, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s = multibatch_test_save(
                                single_coordinate, id, output_image, raw_image)
                            denoise_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                                = aaaa * (np.sum(bbbb) / np.sum(aaaa)) ** 0.5
                            input_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                                = bbbb
                    else:
                        aaaa, bbbb, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s = singlebatch_test_save(
                            single_coordinate, output_image, raw_image)
                        # print("width:", stack_start_w, stack_end_w)
                        # print("height:", stack_start_h, stack_end_h)
                        # print("time:", stack_start_s, stack_end_s)
                        denoise_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                            = aaaa * (np.sum(bbbb) / np.sum(aaaa)) ** 0.5
                        input_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                            = bbbb

                # del noise_img
                output_img = denoise_img.squeeze().astype(np.float32) * opt.scale_factor
                del denoise_img
                # output_img = output_img1[0:raw_noise_img.shape[0],0:raw_noise_img.shape[1],0:raw_noise_img.shape[2]]
                output_img = output_img.astype('int16')
                # output_img = output_img - output_img.min()
                # output_img = output_img / output_img.max() * 65535
                # output_img = np.clip(output_img, 0, 65535).astype('uint16')
                #print(output_img.shape)

                result_name = output_path + '//' + img_list[N].replace('.tif', '') + '_' + pth_name.replace('.pth',
                                                                                                            '') + '_output.tif'
                io.imsave(result_name, output_img, check_contrast=False)
