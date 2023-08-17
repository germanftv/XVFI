# ------------------------------------------------------------------------
# Modified from XVFI (https://github.com/JihyongOh/XVFI)
# ------------------------------------------------------------------------
from __future__ import division
import os, glob, torch, cv2
import numpy as np
import torch.utils.data as data
import torch.nn.functional as F
from torch.nn import init


class save_manager():
    def __init__(self, args):
        self.args = args
        self.model_dir = self.args.net_type + '_' + self.args.pretrained + '_exp' + str(self.args.exp_num)
        print("model_dir:", self.model_dir)
        # ex) model_dir = "XVFInet_exp1"

        self.checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.model_dir)
        # './checkpoint_dir/XVFInet_exp1"
        check_folder(self.checkpoint_dir)

        print("checkpoint_dir:", self.checkpoint_dir)

    def write_info(self, strings):
        self.log_file = open(self.text_dir + '.txt', 'a')
        self.log_file.write(strings)
        self.log_file.close()


    def load_model(self, ):
        # checkpoint = torch.load(self.checkpoint_dir + '/' + self.model_dir + '_latest.pt')
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, self.model_dir + '_latest.pt'), map_location='cuda:0')
        print("load model '{}', epoch: {},".format(
            os.path.join(self.checkpoint_dir, self.model_dir + '_latest.pt'), checkpoint['last_epoch'] + 1))
        return checkpoint



def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def weights_init(m):
    classname = m.__class__.__name__
    if (classname.find('Conv2d') != -1) or (classname.find('Conv3d') != -1):
        init.xavier_normal_(m.weight)
        # init.kaiming_normal_(m.weight, nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            init.zeros_(m.bias)


def get_test_data(args, multiple, validation):
    data_test = Custom_Test(args, multiple)
    dataloader = torch.utils.data.DataLoader(data_test, batch_size=1, drop_last=True, shuffle=False, pin_memory=False)
    return dataloader


def frames_loader_test(args, I0I1It_Path, validation):
    frames = []
    for path in I0I1It_Path:
        frame = cv2.imread(path)
        frames.append(frame)
    (ih, iw, c) = frame.shape
    frames = np.stack(frames, axis=0)  # (T, H, W, 3)

    if args.pretrained == 'X4K1000FPS':
        if validation:
            ps = 512
            ix = (iw - ps) // 2
            iy = (ih - ps) // 2
            frames = frames[:, iy:iy + ps, ix:ix + ps, :]

    """ np2Tensor [-1,1] normalized """
    frames = RGBframes_np2Tensor(frames, args.img_ch)

    return frames


def RGBframes_np2Tensor(imgIn, channel):
    ## input : T, H, W, C
    if channel == 1:
        # rgb --> Y (gray)
        imgIn = np.sum(imgIn * np.reshape([65.481, 128.553, 24.966], [1, 1, 1, 3]) / 255.0, axis=3,
                       keepdims=True) + 16.0

    # to Tensor
    ts = (3, 0, 1, 2)  ############# dimension order should be [C, T, H, W]
    imgIn = torch.Tensor(imgIn.transpose(ts).astype(float)).mul_(1.0)

    # normalization [-1,1]
    imgIn = (imgIn / 255.0 - 0.5) * 2

    return imgIn


def make_2D_dataset_Custom_Test(dir, multiple):
    """ make [I0,I1,It,t,scene_folder] """
    """ 1D (accumulated) """
    testPath = []
    t = np.linspace((1 / multiple), (1 - (1 / multiple)), (multiple - 1))
    # for scene_folder in sorted(glob.glob(os.path.join(dir, '*', ''))):  # [scene1, scene2, scene3, ...]
    frame_folder = sorted(glob.glob(os.path.join(dir, '') + '*.png'))  # ex) ['00000.png',...,'00123.png']
    for idx in range(0, len(frame_folder)):
        if idx == len(frame_folder) - 1:
            break
        for suffix, mul in enumerate(range(multiple - 1)):
            I0I1It_paths = []
            I0I1It_paths.append(frame_folder[idx])  # I0 (fix)
            I0I1It_paths.append(frame_folder[idx + 1])  # I1 (fix)
            target_t_Idx = frame_folder[idx].split(os.sep)[-1].split('.')[0]+'_' + str(suffix+1).zfill(3) + '.png'
            # ex) target t name: 00017.png => '00017_1.png'
            I0I1It_paths.append(os.path.join(dir, target_t_Idx))  # It
            I0I1It_paths.append(t[mul]) # t
            I0I1It_paths.append(os.path.basename(dir))  # scene1
            testPath.append(I0I1It_paths)
    return testPath


class Custom_Test(data.Dataset):
    def __init__(self, args, multiple):
        self.args = args
        self.multiple = multiple
        self.testPath = make_2D_dataset_Custom_Test(self.args.input_dir, self.multiple)
        self.nIterations = len(self.testPath)

        # Raise error if no images found in test_data_path.
        if len(self.testPath) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.args.input_dir + "\n"))

    def __getitem__(self, idx):
        I0, I1, It, t_value, scene_name = self.testPath[idx]
        dummy_dir = I1 # due to there is not ground truth intermediate frame.
        I0I1It_Path = [I0, I1, dummy_dir]

        frames = frames_loader_test(self.args, I0I1It_Path, None)
        # including "np2Tensor [-1,1] normalized"

        I0_path = I0.split(os.sep)[-1]
        I1_path = I1.split(os.sep)[-1]
        It_path = It.split(os.sep)[-1]

        return frames, np.expand_dims(np.array(t_value, dtype=np.float32), 0), scene_name, [It_path, I0_path, I1_path]

    def __len__(self):
        return self.nIterations


class AverageClass(object):
    """ For convenience of averaging values """
    """ refer from "https://github.com/pytorch/examples/blob/master/imagenet/main.py" """

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1.0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} (avg:{avg' + self.fmt + '})'
        # Accm_Time[s]: 1263.517 (avg:639.701)    (<== if AverageClass('Accm_Time[s]:', ':6.3f'))
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """ For convenience of printing diverse values by using "AverageClass" """
    """ refer from "https://github.com/pytorch/examples/blob/master/imagenet/main.py" """

    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        # # Epoch: [0][  0/196]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'



def to_uint8(x, vmin, vmax):
    ##### color space transform, originally from https://github.com/yhjo09/VSR-DUF #####
    x = x.astype('float32')
    x = (x - vmin) / (vmax - vmin) * 255  # 0~255
    return np.clip(np.round(x), 0, 255)


def im2tensor(image, imtype=np.uint8, cent=1., factor=255. / 2.):
    # def im2tensor(image, imtype=np.uint8, cent=1., factor=1.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))


# [0,255]2[-1,1]2[1,3,H,W]-shaped

def denorm255(x):
    out = (x + 1.0) / 2.0
    return out.clamp_(0.0, 1.0) * 255.0


def denorm255_np(x):
    # numpy
    out = (x + 1.0) / 2.0
    return out.clip(0.0, 1.0) * 255.0

