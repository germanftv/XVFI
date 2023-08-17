# ------------------------------------------------------------------------
# Modified from XVFI (https://github.com/JihyongOh/XVFI)
# ------------------------------------------------------------------------
import sys, os, time, torch, cv2, torch.utils.data
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np

from torch.autograd import Variable
from collections import Counter
from .utils import *
from .XVFInet import *
from .parser import parse_args


def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)
    return args


def main(args):
    args = check_args(args)
    if args.pretrained == 'Vimeo':
        args.S_tst = 1
        args.module_scale_factor = 2
        args.patch_size = 256
        args.batch_size = 16

    print("Exp:", args.exp_num)
    args.model_dir = args.net_type + '_' + args.pretrained + '_exp' + str(
        args.exp_num)  # ex) model_dir = "XVFInet_X4K1000FPS_exp1"

    if args is None:
        exit()
    for arg in vars(args):
        print('# {} : {}'.format(arg, getattr(args, arg)))
    device = torch.device(
        'cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')  # will be used as "x.to(device)"
    torch.cuda.set_device(device)  # change allocation of current GPU
    # caution!!!! if not "torch.cuda.set_device()":
    # RuntimeError: grid_sampler(): expected input and grid to be on same device, but input is on cuda:1 and grid is on cuda:0
    print('Available devices: ', torch.cuda.device_count())
    print('Current cuda device: ', torch.cuda.current_device())
    print('Current cuda device name: ', torch.cuda.get_device_name(device))
    if args.gpu is not None:
        print("Use GPU: {} is used".format(args.gpu))

    SM = save_manager(args)

    """ Initialize a model """
    model_net = args.net_object(args).apply(weights_init).to(device)

    # to enable the inbuilt cudnn auto-tuner
    # to find the best algorithm to use for your hardware.
    cudnn.benchmark = True

    """ Load a model """
    checkpoint = SM.load_model()
    model_net.load_state_dict(checkpoint['state_dict_Model'])
    epoch = checkpoint['last_epoch']

    print('Inference with interpolation factor = %d ' % (args.multiple))

    final_test_loader = get_test_data(args, multiple=args.multiple,
                                        validation=False)  # multiple is only used for X4K1000FPS

    test(final_test_loader, model_net, args, device, multiple=args.multiple)



def test(test_loader, model_net, args, device, multiple):

    args.divide = 2 ** (args.S_tst) * args.module_scale_factor * 4

    # switch to evaluate mode
    model_net.eval()

    print("------------------------------------------- Inference ----------------------------------------------")
    with torch.no_grad():
        start_time = time.time()
        for testIndex, (frames, t_value, scene_name, frameRange) in enumerate(test_loader):
            # Shape of 'frames' : [1,C,T+1,H,W]
            frameT = frames[:, :, -1, :, :]  # [1,C,H,W]
            It_Path, I0_Path, I1_Path = frameRange

            frameT = Variable(frameT.to(device))  # ground truth for frameT
            t_value = Variable(t_value.to(device))

            if (testIndex % (multiple - 1)) == 0:
                input_frames = frames[:, :, :-1, :, :]  # [1,C,T,H,W]
                input_frames = Variable(input_frames.to(device))

                B, C, T, H, W = input_frames.size()
                H_padding = (args.divide - H % args.divide) % args.divide
                W_padding = (args.divide - W % args.divide) % args.divide
                if H_padding != 0 or W_padding != 0:
                    input_frames = F.pad(input_frames, (0, W_padding, 0, H_padding), "constant")


            pred_frameT = model_net(input_frames, t_value)

            if H_padding != 0 or W_padding != 0:
                pred_frameT = pred_frameT[:, :, :H, :W]

            # Save the predicted frameT
            output_dir = args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            pred_frameT = np.squeeze(pred_frameT.detach().cpu().numpy())
            output_img = np.around(denorm255_np(np.transpose(pred_frameT, [1, 2, 0])))  # [h,w,c] and [-1,1] to [0,255]
            print(os.path.join(output_dir, It_Path[0]))
            cv2.imwrite(os.path.join(output_dir, It_Path[0]), output_img.astype(np.uint8))

            # Save the anchor frames
            if (testIndex % (multiple - 1)) == 0:
                save_input_frames = frames[:, :, :-1, :, :]
                if testIndex == 0:
                    cv2.imwrite(os.path.join(output_dir, I0_Path[0].split(os.sep)[-1].split('.')[0]+'_' + str(0).zfill(3) + '.png'),
                                np.transpose(np.squeeze(denorm255_np(save_input_frames[:, :, 0, :, :].detach().numpy())),
                                            [1, 2, 0]).astype(np.uint8))
                cv2.imwrite(os.path.join(output_dir, I1_Path[0].split(os.sep)[-1].split('.')[0]+'_' + str(0).zfill(3) + '.png'),
                            np.transpose(np.squeeze(denorm255_np(save_input_frames[:, :, 1, :, :].detach().numpy())),
                                            [1, 2, 0]).astype(np.uint8))

        print("-----------------------------------------------------------------------------------------------")
        print("\nInference time: {:.4f}".format(time.time() - start_time))


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
