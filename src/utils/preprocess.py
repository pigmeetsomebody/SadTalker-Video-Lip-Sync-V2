import os, sys
sys.path.append('./venv/lib/python3.9/site-packages')
import numpy as np
import cv2, os, torch
from tqdm import tqdm
from PIL import Image
from src.face3d.util.preprocess import align_img
from src.face3d.util.load_mats import load_lm3d
from src.face3d.models import networks
from src.face3d.extract_kp_videos import KeypointExtractor
from scipy.io import savemat
from src.utils.croper import Croper
import warnings
import wave

warnings.filterwarnings("ignore")


def split_coeff(coeffs):
    """
    Return:
        coeffs_dict     -- a dict of torch.tensors

    Parameters:
        coeffs          -- torch.tensor, size (B, 256)
    """
    id_coeffs = coeffs[:, :80]
    exp_coeffs = coeffs[:, 80: 144]
    tex_coeffs = coeffs[:, 144: 224]
    angles = coeffs[:, 224: 227]
    gammas = coeffs[:, 227: 254]
    translations = coeffs[:, 254:]
    return {
        'id': id_coeffs,
        'exp': exp_coeffs,
        'tex': tex_coeffs,
        'angle': angles,
        'gamma': gammas,
        'trans': translations
    }


class CropAndExtract():
    import wave

    def __init__(self, path_of_lm_croper, path_of_net_recon_model, dir_of_BFM_fitting, device):

        self.croper = Croper(path_of_lm_croper)
        self.kp_extractor = KeypointExtractor(device)
        self.net_recon = networks.define_net_recon(net_recon='resnet50', use_last_fc=False, init_path='').to(device)
        checkpoint = torch.load(path_of_net_recon_model, map_location=torch.device(device))
        self.net_recon.load_state_dict(checkpoint['net_recon'])
        self.net_recon.eval()
        self.lm3d_std = load_lm3d(dir_of_BFM_fitting)
        self.device = device

    def generate(self, input_path, save_dir):

        pic_size = 256
        pic_name = os.path.splitext(os.path.split(input_path)[-1])[0]

        landmarks_path = os.path.join(save_dir, pic_name + '_landmarks.txt')
        coeff_path = os.path.join(save_dir, pic_name + '.mat')

        # load input
        if not os.path.isfile(input_path):
            raise ValueError('input_path must be a valid path to video/image file')
        elif input_path.split('.')[-1] in ['jpg', 'png', 'jpeg']:
            # loader for first frame
            full_frames = [cv2.imread(input_path)]
        else:
            # loader for videos
            video_stream = cv2.VideoCapture(input_path)
            full_frames = []
            while 1:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break
                full_frames.append(frame)

        x_full_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in full_frames]

        x_full_frames, crop, quad = self.croper.crop(x_full_frames, still=True, xsize=pic_size)
        clx, cly, crx, cry = crop
        lx, ly, rx, ry = quad
        lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
        oy1, oy2, ox1, ox2 = cly + ly, cly + ry, clx + lx, clx + rx
        crop_info = ((ox2 - ox1, oy2 - oy1), crop, quad)

        frames_pil = [Image.fromarray(cv2.resize(frame, (pic_size, pic_size))) for frame in x_full_frames]
        if len(frames_pil) == 0:
            print('No face is detected in the input file')
            return None, None

        # save crop info
        index = 0
        save_img = os.path.join(save_dir, 'img')
        os.makedirs(save_img, exist_ok=True)
        for frame in frames_pil:
            png_path = os.path.join(save_img, pic_name + '_{}'.format(str(index).zfill(6)) + '.png')
            cv2.imwrite(png_path, cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
            index += 1
        # to test
        # sample_img_path = os.path.join('./examples/driven_video/', pic_name + '_{}'.format(str(64).zfill(6)) + '.png')
        # sam        # sample_img_path = os.path.join('./examples/driven_video/', pic_name + '_{}'.format(str(64).zfill(6)) + '.png')
        #         # sample_frames_pil = []
        #         # for i in range(len(frames_pil)):
        #         #     png_path = os.path.join(save_img, pic_name + '_{}'.format(str(i).zfill(6)) + '.png')
        #         #     frame = cv2.imread(sample_img_path)
        #         #     cv2.imwrite(png_path, frame)
        #         #     sample_frames_pil.append(frame)ple_frames_pil = []
        # for i in range(len(frames_pil)):
        #     png_path = os.path.join(save_img, pic_name + '_{}'.format(str(i).zfill(6)) + '.png')
        #     frame = cv2.imread(sample_img_path)
        #     cv2.imwrite(png_path, frame)
        #     sample_frames_pil.append(frame)

        # 2. get the landmark according to the detected face. 
        if not os.path.isfile(landmarks_path):
            lm = self.kp_extractor.extract_keypoint(frames_pil, landmarks_path)
        else:
            print(' Using saved landmarks.')
            lm = np.loadtxt(landmarks_path).astype(np.float32)
            lm = lm.reshape([len(x_full_frames), -1, 2])

        if not os.path.isfile(coeff_path):
            # load 3dmm paramter generator from Deep3DFaceRecon_pytorch 
            video_coeffs, full_coeffs = [], []
            for idx in tqdm(range(len(frames_pil)), desc='3DMM Extraction In Video:'):
                frame = frames_pil[idx]
                W, H = frame.size
                lm1 = lm[idx].reshape([-1, 2])

                if np.mean(lm1) == -1:
                    lm1 = (self.lm3d_std[:, :2] + 1) / 2.
                    lm1 = np.concatenate(
                        [lm1[:, :1] * W, lm1[:, 1:2] * H], 1
                    )
                else:
                    lm1[:, -1] = H - 1 - lm1[:, -1]

                trans_params, im1, lm1, _ = align_img(frame, lm1, self.lm3d_std)

                trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
                im_t = torch.tensor(np.array(im1) / 255., dtype=torch.float32).permute(2, 0, 1).to(
                    self.device).unsqueeze(0)

                with torch.no_grad():
                    full_coeff = self.net_recon(im_t)
                    coeffs = split_coeff(full_coeff)

                pred_coeff = {key: coeffs[key].cpu().numpy() for key in coeffs}

                pred_coeff = np.concatenate([
                    pred_coeff['exp'],
                    pred_coeff['angle'],
                    pred_coeff['trans'],
                    trans_params[2:][None],
                ], 1)
                video_coeffs.append(pred_coeff)
                full_coeffs.append(full_coeff.cpu().numpy())

            semantic_npy = np.array(video_coeffs)[:, 0]

            savemat(coeff_path, {'coeff_3dmm': semantic_npy, 'full_3dmm': np.array(full_coeffs)[0]})

        return coeff_path, save_img, crop_info

import cv2
def pasteAndAddFrame(original_video_path, target_duration):
    # 获取视频的文件名（包含扩展名）
    original_video_with_extension = os.path.basename(original_video_path)
    print("视频的文件名（包含扩展名）：", original_video_with_extension)

    # 获取视频的文件名（不包含扩展名）
    original_video_without_extension = os.path.splitext(original_video_with_extension)[0]
    print("图片的文件名（不包含扩展名）：", original_video_without_extension)
    # 获取视频的目录部分
    directory_path = os.path.dirname(original_video_path)
    output_video_path = os.path.join(directory_path, f'{original_video_without_extension}_{target_duration}s.mp4')
    # if os.path.exists(output_video_path):
    #     return output_video_path
    if not os.path.isfile(original_video_path):
        raise ValueError('original_video must be a valid path to video/image file')
    elif original_video_path.split('.')[-1] in ['jpg', 'png', 'jpeg']:
        # loader for first frame
        full_frame = cv2.imread(original_video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        h, w = full_frame[:2]
        fps = 25
        output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (int(w), int(h)))
        total_frames = int(fps * target_duration)
        for i in range(total_frames):
            output_video.write(full_frame)
        output_video.release()
        return output_video_path
    else:
        # loader for videos
        video_stream = cv2.VideoCapture(original_video_path)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        total_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / fps
        if total_duration >= target_duration:
            video_stream.release()
            return original_video_path

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print(f'video_width: {int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))} and video_height: {int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))}')
        output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        written_frames = 0
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                # 如果到达视频末尾，重新从视频开头开始
                video_stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            output_video.write(frame)
            written_frames += 1
            written_duration = written_frames / fps
            if written_duration >= target_duration:
                print(f'已写入延长后的视频路径：{output_video_path}. duration: {written_duration}')
                break
        video_stream.release()
        output_video.release()
        return output_video_path

from src.utils.audio import get_wav_duration
import math
if __name__ == '__main__':
    original_video_path = './data/test16.mp4'
    # output_video_path = './data/test18.mp4'
    wav_duration = get_wav_duration('./data/pyrimid.wav')
    wav_duration = math.ceil(wav_duration)
    output_video_path = pasteAndAddFrame(original_video_path, wav_duration)
    print(f'write to video path: {output_video_path} success.')