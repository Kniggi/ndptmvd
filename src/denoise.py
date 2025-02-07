# external imports
import os
import time
import tqdm
import torch
import argparse
import imageio_ffmpeg
from matplotlib import pyplot as plt
import numpy as np
# internal imports
import data
import models
import losses
import cv2
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
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

def fourier_ring_correlation(image_1, image_2, bin_width=2):
    
    image_1 = image_1 / np.sum(image_1)
    image_2 = image_2 / np.sum(image_2)
    f1, f2 = np.fft.fft2(image_1), np.fft.fft2(image_2)
    af1f2 = np.real(f1 * np.conj(f2))
    af1_2, af2_2 = np.abs(f1)**2, np.abs(f2)**2
    nx, ny = af1f2.shape
    x = np.arange(-np.floor(nx / 2.0), np.ceil(nx / 2.0))
    y = np.arange(-np.floor(ny / 2.0), np.ceil(ny / 2.0))
    distances = list()
    wf1f2 = list()
    wf1 = list()
    wf2 = list()
    for xi, yi in np.array(np.meshgrid(x,y)).T.reshape(-1, 2):
        distances.append(np.sqrt(xi**2 + xi**2))
        xi = int(xi)
        yi = int(yi)
        wf1f2.append(af1f2[xi, yi])
        wf1.append(af1_2[xi, yi])
        wf2.append(af2_2[xi, yi])

    bins = np.arange(0, np.sqrt((nx//2)**2 + (ny//2)**2), bin_width)
    f1f2_r, bin_edges = np.histogram(
        distances,
        bins=bins,
        weights=wf1f2
    )
    f12_r, bin_edges = np.histogram(
        distances,
        bins=bins,
        weights=wf1
    )
    f22_r, bin_edges = np.histogram(
        distances,
        bins=bins,
        weights=wf2
    )
    density = f1f2_r / np.sqrt(f12_r * f22_r)
    return density, bin_edges
def add_color_val_difference(img_input, img_pred,img_target):
  rows = 3
  columns = 3
  fig  = plt.figure(figsize=((512*3)/80, (512*2)/80))
  fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
  fig.add_subplot(rows, columns, 1)
  plt.imshow(img_input.cpu().permute(0,2,3,1).numpy()[0])
  plt.axis('off')
  plt.title("Input")
  fig.add_subplot(rows, columns, 2)
  plt.imshow(img_pred.cpu().permute(0,2,3,1).numpy()[0])
  plt.axis('off')
  plt.title("Prediction")
  fig.add_subplot(rows, columns, 3)
  plt.imshow(img_target.cpu().permute(0,2,3,1).numpy()[0])
  plt.axis('off')
  plt.title("Target")
  
  fig.add_subplot(rows, columns, 4)
  for i, col in enumerate(['b', 'g', 'r']):
      im=cv2.normalize(cv2.cvtColor(np.transpose(img_input.cpu().numpy()[0][0:3], [1, 2, 0]), cv2.COLOR_RGB2BGR), None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
      hist = cv2.calcHist([im], [i], None, [256], [0.0, 256.0])
      plt.plot(hist, color = col)
      plt.xlim([0, 260])
      plt.title("Color Hist Input")
  fig.add_subplot(rows, columns, 5)
  for i, col in enumerate(['b', 'g', 'r']):
      im=cv2.normalize(cv2.cvtColor(np.transpose(img_pred.cpu().numpy()[0][0:3], [1, 2, 0]), cv2.COLOR_RGB2BGR), None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
      hist = cv2.calcHist([im], [i], None, [256], [0.0,256.0])
      plt.plot(hist, color = col)
      plt.xlim([0, 260])
      plt.title("Color Hist Prediction")
  fig.add_subplot(rows, columns, 6)
  for i, col in enumerate(['b', 'g', 'r']):
      im=cv2.normalize(cv2.cvtColor(np.transpose(img_target.cpu().numpy()[0][0:3], [1, 2, 0]), cv2.COLOR_RGB2BGR), None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
      hist = cv2.calcHist([im], [i], None, [256], [0.0, 256.0])
      plt.plot(hist, color = col)
      plt.xlim([0, 260])
      plt.title("Color Hist Target")
  plt.tight_layout()
  plt.savefig([f'{name}/{name}_color_diff{args.batch_size*idx+j:06}.jpg' for j in range(frame.size(0))][0], bbox_inches='tight', pad_inches=0, dpi=500, pil_kwargs={'quality':95})
  plt.close()
  plt.cla()
  plt.clf()
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
                add_color_val_difference(img_input, img_pred,img_target)
                
                
                mse_input_target = 0
                ssim_input_target = 0
                mse_pred_target = 0
                ssim_pred_target = 0
                # frequency domain images
                rgb_weights = [0.2989, 0.5870, 0.1140]
                grayscale = cv2.cvtColor(img_pred.cpu().permute(0,2,3,1).numpy()[0], cv2.COLOR_RGB2GRAY)[np.newaxis,...]
                
                
                f_pred = np.fft.fft2(grayscale)
                magnitude_spectrum_pred = 20* np.log(np.abs(np.fft.fftshift(f_pred)))
                data.write([f'{name}/{name}_pred_frequency_spectrum{args.batch_size*idx+j:06}.jpg' for j in range(frame.size(0))], magnitude_spectrum_pred[:,np.newaxis,...],1)
                grayscale = cv2.cvtColor(img_input.cpu().permute(0,2,3,1).numpy()[0], cv2.COLOR_RGB2GRAY)[np.newaxis,...]
                f_input= np.fft.fft2(grayscale)
                magnitude_spectrum_input = 20* np.log(np.abs(np.fft.fftshift(f_input)))
                data.write([f'{name}/{name}_input_frequency_spectrum{args.batch_size*idx+j:06}.jpg' for j in range(frame.size(0))], magnitude_spectrum_input[:,np.newaxis,...],1)
                grayscale = cv2.cvtColor(img_target.cpu().permute(0,2,3,1).numpy()[0], cv2.COLOR_RGB2GRAY)[np.newaxis,...]
                f_target = np.fft.fft2(grayscale)
                magnitude_spectrum_target = 20* np.log(np.abs(np.fft.fftshift(f_target)))
                data.write([f'{name}/{name}_target_frequency_spectrum{args.batch_size*idx+j:06}.jpg' for j in range(frame.size(0))], magnitude_spectrum_target[:,np.newaxis,...],1)
                

                
                
                # plt.title('FRC')
                # input_gray = cv2.cvtColor(img_input.cpu().permute(0,2,3,1).numpy()[0], cv2.COLOR_RGB2GRAY)
                # pred_gray =  cv2.cvtColor(img_pred.cpu().permute(0,2,3,1).numpy()[0], cv2.COLOR_RGB2GRAY)
                # target_gray = cv2.cvtColor(img_target.cpu().permute(0,2,3,1).numpy()[0], cv2.COLOR_RGB2GRAY)
                # im2_pred = img_pred.cpu().numpy()[0][1::2].sum(axis=0)

                # im1_input = img_input.cpu().numpy()[0][::2].sum(axis=0)
                # im2_input = img_input.cpu().numpy()[0][1::2].sum(axis=0)

                # im1_target = img_target.cpu().numpy()[0][::2].sum(axis=0)
                # im2_target = img_target.cpu().numpy()[0][1::2].sum(axis=0)
                
                # frc_input_pred, frc_bins = fourier_ring_correlation(input_gray,pred_gray)

                # frc_input_target, frc_bins = fourier_ring_correlation(input_gray,target_gray)
                
                # plt.plot(frc_input_pred, label="Input Pred", color="red")
                # plt.plot(frc_input_target, label="Input Target", color="blue")
          
               # plt.show()


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

                fig = plt.figure(figsize=((512*3)/80, (512*2)/80))
                rows = 3
                columns = 3
                fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
                fig.add_subplot(rows, columns, 1)
                plt.imshow(img_input.cpu().permute(0,2,3,1).numpy()[0])
                plt.axis('off')
                plt.title("Input")
                fig.add_subplot(rows, columns, 2)
                plt.imshow(img_pred.cpu().permute(0,2,3,1).numpy()[0])
                plt.axis('off')
                plt.title("Prediction")
                fig.add_subplot(rows, columns, 3)
                plt.imshow(img_target.cpu().permute(0,2,3,1).numpy()[0])
                plt.axis('off')
                plt.title("Target")

                fig.add_subplot(rows, columns, 4)
                plt.imshow(magnitude_spectrum_input[0],cmap='gray')
                plt.axis('off')
                plt.title("Input Frequency")
                fig.add_subplot(rows, columns, 5)
                plt.imshow( magnitude_spectrum_pred[0],cmap='gray')
                plt.axis('off')
                plt.title("Prediction Frequency")

                fig.add_subplot(rows, columns, 6)
                plt.imshow(magnitude_spectrum_target[0],cmap='gray')
                plt.axis('off')
                plt.title("Target Frequency")
                plt.tight_layout()

                plt.savefig([f'{name}/{name}_plot{args.batch_size*idx+j:06}.jpg' for j in range(frame.size(0))][0], bbox_inches='tight', pad_inches=0, dpi=500, pil_kwargs={'quality':95})
                plt.close()
                plt.cla()
                plt.clf()
            
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
