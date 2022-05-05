# external imports
import os
import time
import tqdm
import torch
import argparse
import imageio_ffmpeg
import numpy as np
# internal imports
import data
import models
import losses
import cv2
# -----------------------------------------------------------
# CMD LINE SETTINGS

cmdline = argparse.ArgumentParser(description='Use a denoising autoencoder')
# required args
cmdline.add_argument('model', help='path to h5 model file')
cmdline.add_argument('input_data', help='path to directory containing input data')
cmdline.add_argument('target_data', help='path to directory containing target data')
# optional args
cmdline.add_argument('-t', '--type', type=str, default='image', help='type of dataset to feed: [image, reproj]')
cmdline.add_argument('-b', '--batch_size', type=int, default=1, help='mini batch size')
cmdline.add_argument('--images', action="store_true", help="store images")
cmdline.add_argument('--cmp', action="store_true", help="write comparison images")
cmdline.add_argument('--format', type=str, default='jpg', help='output image format')
cmdline.add_argument('--fps', type=int, default=24, help='output video frame rate')
cmdline.add_argument('-f','--features', type=int, default=[], nargs='*', help='Tuples of feature channels to select from input')

# -----------------------------------------------------------
# MAIN

if __name__ == "__main__":

    # parse command line
    args = cmdline.parse_args()
    print('args:', args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup model
    model = torch.load(args.model, map_location=device)
    model.eval()
    model.to(device)
    # model = torch.nn.DataParallel(model)

    # setup data set
    if 'reproj' in args.type:
        dataset = data.DataSetTestReproj(args.input_data, args.target_data)
    else:
        dataset = data.DataSetTest(args.input_data, args.target_data, args.features)
    # FIXME multiprocess loading with proper ordering?
    data_loader = torch.utils.data.DataLoader(dataset, num_workers=1, pin_memory=True, batch_size=args.batch_size)

    # setup files
    name = os.path.splitext(os.path.basename(args.model))[0] + '--' + os.path.basename(os.path.dirname(args.input_data + '/'))
    os.makedirs(name, exist_ok=True)

    # setup video storage
    (sample, _) = dataset[0]
    video = np.empty((len(dataset), sample.shape[-2], 3*sample.shape[-1], 3), dtype='uint8')

    times = []

    # print('Denoising...')
    with torch.no_grad():
        tq = tqdm.tqdm(total=len(data_loader)*args.batch_size, desc='Denoise')
        for idx, (in_data, target) in enumerate(data_loader):
            in_data, target = in_data.to(device), target.to(device)
            torch.cuda.synchronize()
            start = time.time()
            prediction = model(in_data)
            torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)
            # store video frame(s)
            x = in_data[:, data.N_FRAMES, 0:3, :, :] if 'reproj' in args.type else in_data[:, 0:3, :, :]
            p = prediction[:, 0:3, :, :]
            y = target[:, 0:3, :, :]
            # postprocess
            frame = (torch.sqrt(torch.clamp(torch.cat((x, p, y), dim=-1), 0, 1)) * 255).to(torch.uint8)
            frame = frame.transpose_(-3, -1).transpose_(-3, -2)
            video[args.batch_size*idx:args.batch_size*idx+p.size(0)] = frame.cpu().numpy()
            # write images to disk?
            if args.images:
                # img = torch.cat((x, p, y), dim=-1) if args.cmp else p
                #prediction
                img_pred = p
                data.write([f'{name}/{name}_pred{args.batch_size*idx+j:06}.hdr' for j in range(frame.size(0))], img_pred.cpu().numpy())
                #target
                img_target = y
                data.write([f'{name}/{name}_target{args.batch_size*idx+j:06}.hdr' for j in range(frame.size(0))], img_target.cpu().numpy())
                #input
                img_input = x
                data.write([f'{name}/{name}_input{args.batch_size*idx+j:06}.hdr' for j in range(frame.size(0))], img_input.cpu().numpy())
                # metrics
                #color_val_difference
                diff_red = np.abs(img_target.cpu().numpy()[0,0,...]-img_pred.cpu().numpy()[0,0,...])
                diff_green = np.abs(img_target.cpu().numpy()[0,1,...]-img_pred.cpu().numpy()[0,1,...])
                diff_blue = np.abs(img_target.cpu().numpy()[0,2,...]-img_pred.cpu().numpy()[0,2,...])
                f = open(f'{name}/{name}col_diff.txt', "w")
                f.write(f'Red:{np.mean(diff_red)} Green:{np.mean(diff_green)} Blue:{np.mean(diff_blue)}')
                f.close()
                mse_input_target = 0
                ssim_input_target = 0
                mse_pred_target = 0
                ssim_pred_target = 0
                # frequency domain images
                rgb_weights = [0.2989, 0.5870, 0.1140]
                grayscale = cv2.cvtColor(img_pred.cpu().permute(0,2,3,1).numpy()[0], cv2.COLOR_RGB2GRAY)[np.newaxis,...]
                f_pred = np.fft.fft2(grayscale)
                magnitude_spectrum_pred = 20* np.log(np.abs(np.fft.fftshift(f_pred)))
                data.write([f'{name}/{name}_pred_frequency_spectrum{args.batch_size*idx+j:06}.hdr' for j in range(frame.size(0))], magnitude_spectrum_pred[:,np.newaxis,...],1)
                grayscale = cv2.cvtColor(img_input.cpu().permute(0,2,3,1).numpy()[0], cv2.COLOR_RGB2GRAY)[np.newaxis,...]
                f_input= np.fft.fft2(grayscale)
                magnitude_spectrum_input = 20* np.log(np.abs(np.fft.fftshift(f_input)))
                data.write([f'{name}/{name}_input_frequency_spectrum{args.batch_size*idx+j:06}.hdr' for j in range(frame.size(0))], magnitude_spectrum_input[:,np.newaxis,...],1)
                grayscale = cv2.cvtColor(img_target.cpu().permute(0,2,3,1).numpy()[0], cv2.COLOR_RGB2GRAY)[np.newaxis,...]
                f_target = np.fft.fft2(grayscale)
                magnitude_spectrum_target = 20* np.log(np.abs(np.fft.fftshift(f_target)))
                data.write([f'{name}/{name}_target_frequency_spectrum{args.batch_size*idx+j:06}.hdr' for j in range(frame.size(0))], magnitude_spectrum_target[:,np.newaxis,...],1)
                for j in range(frame.size(0)):
                    mse_input_target += losses.mse(img_input, img_target).item()
                    ssim_input_target += losses.ssim(img_input, img_target).item()

                    mse_pred_target += losses.mse(img_pred, img_target).item()
                    ssim_pred_target += losses.ssim(img_pred, img_target).item()
                mse_input_target_avg = mse_input_target / len(range(frame.size(0)))
                ssim_input_target_avg = ssim_input_target / len(range(frame.size(0)))

                mse_pred_target_avg = mse_pred_target / len(range(frame.size(0)))
                ssim_pred_target_avg = ssim_pred_target / len(range(frame.size(0)))
                f = open(f'{name}/{name}_metrics.txt', "w")
                f.write(f' Input/Target: MSE: {mse_input_target_avg} SSIM: {ssim_input_target_avg}')
                f.write(f' Prediction/Target: MSE: {mse_pred_target_avg} SSIM: {ssim_pred_target_avg}')
                f.close()
            tq.update(args.batch_size)
        tq.close()

    print(f'avg inference time (in s):', np.array(times).mean(), 'std:', np.array(times).std())

    # write video
    ffmpeg = imageio_ffmpeg.write_frames(f'{name}/{name}.mp4', (3*sample.shape[-1], sample.shape[-2]), fps=args.fps, quality=5)
    ffmpeg.send(None) # seed
    ffmpeg.send(video)
    ffmpeg.close()
    print(f'{name}/{name}.mp4 written.')
    # make sure images were written
    data.pool.close()
    data.pool.join()
