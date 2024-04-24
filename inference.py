import math
import os, sys
sys.path.append('./venv/lib/python3.9/site-packages')
import torch
from time import strftime
from argparse import ArgumentParser
from src.utils.preprocess import CropAndExtract, pasteAndAddFrame
from src.utils.audio import get_wav_duration, get_audios_duration_and_save, merge_wav_files
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from third_part.GFPGAN.gfpgan import GFPGANer
from third_part.GPEN.gpen_face_enhancer import FaceEnhancement
import warnings
from src.utils.pre_ppt import process_ppt_to_imgs


warnings.filterwarnings("ignore")


def main(args):
    bg_path = args.bg_img
    pic_path = args.source_video
    audio_path = args.driven_audio
    enhancer_region = args.enhancer
    ppt_path = args.ppt_path
    save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)
    device = args.device
    batch_size = args.batch_size
    current_code_path = sys.argv[0]
    current_root_path = os.path.split(current_code_path)[0]
    os.environ['TORCH_HOME'] = os.path.join(current_root_path, args.checkpoint_dir)
    # 预处理ppt
    ppt_imgs_folder = os.path.join(save_dir, 'ppt_imgs')
    os.makedirs(ppt_imgs_folder, exist_ok=True)
    process_ppt_to_imgs(ppt_path, ppt_imgs_folder)

    # 读取每段音频的时长并合并 （自动换ppt 需要用）
    speech_durations_output_csv_path = os.path.join(save_dir, 'speech_durations_output.csv')
    merged_wav_path = os.path.join(save_dir, 'merged.wav')
    audio_list = [os.path.join(audio_path, audio) for audio in os.listdir(audio_path) if audio.endswith(".wav")]
    sorted_audio_list = sorted(audio_list)
    duration_csv_path = os.path.join(save_dir, 'speech_durations_output.csv')
    get_audios_duration_and_save(sorted_audio_list, duration_csv_path)
    merge_wav_files(sorted_audio_list, merged_wav_path)


    # 计算合成的音频时长
    target_duration = math.ceil(get_wav_duration(merged_wav_path))
    extented_pic_path = pasteAndAddFrame(pic_path, target_duration)


    path_of_lm_croper = os.path.join(current_root_path, args.checkpoint_dir, 'shape_predictor_68_face_landmarks.dat')
    path_of_net_recon_model = os.path.join(current_root_path, args.checkpoint_dir, 'epoch_20.pth')
    dir_of_BFM_fitting = os.path.join(current_root_path, args.checkpoint_dir, 'BFM_Fitting')
    wav2lip_checkpoint = os.path.join(current_root_path, args.checkpoint_dir, 'wav2lip.pth')

    audio2pose_checkpoint = os.path.join(current_root_path, args.checkpoint_dir, 'auido2pose_00140-model.pth')
    audio2pose_yaml_path = os.path.join(current_root_path, 'src', 'config', 'auido2pose.yaml')

    audio2exp_checkpoint = os.path.join(current_root_path, args.checkpoint_dir, 'auido2exp_00300-model.pth')
    audio2exp_yaml_path = os.path.join(current_root_path, 'src', 'config', 'auido2exp.yaml')

    free_view_checkpoint = os.path.join(current_root_path, args.checkpoint_dir, 'facevid2vid_00189-model.pth.tar')

    mapping_checkpoint = os.path.join(current_root_path, args.checkpoint_dir, 'mapping_00109-model.pth.tar')
    facerender_yaml_path = os.path.join(current_root_path, 'src', 'config', 'facerender_still.yaml')

    # init model
    print(path_of_net_recon_model)
    preprocess_model = CropAndExtract(path_of_lm_croper, path_of_net_recon_model, dir_of_BFM_fitting, device)

    print(audio2pose_checkpoint)
    print(audio2exp_checkpoint)
    audio_to_coeff = Audio2Coeff(audio2pose_checkpoint, audio2pose_yaml_path, audio2exp_checkpoint, audio2exp_yaml_path,
                                 wav2lip_checkpoint, device)

    print(free_view_checkpoint)
    print(mapping_checkpoint)
    animate_from_coeff = AnimateFromCoeff(free_view_checkpoint, mapping_checkpoint, facerender_yaml_path, device)

    restorer_model = GFPGANer(model_path='checkpoints/GFPGANv1.3.pth', upscale=1, arch='clean',
                              channel_multiplier=2, bg_upsampler=None)
    enhancer_model = FaceEnhancement(base_dir='checkpoints', size=512, model='GPEN-BFR-512', use_sr=False,
                                     sr_model='rrdb_realesrnet_psnr', channel_multiplier=2, narrow=1, device=device)

    # crop image and extract 3dmm from image
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    print('3DMM Extraction for source image')
    first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(extented_pic_path, first_frame_dir)
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return
    # audio2ceoff
    batch = get_data(first_coeff_path, merged_wav_path, device)
    coeff_path = audio_to_coeff.generate(batch, save_dir)
    # coeff2video
    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, merged_wav_path, batch_size, device)
    tmp_path, new_audio_path, return_path = animate_from_coeff.generate(data, save_dir, extented_pic_path, crop_info,
                                                                        restorer_model, enhancer_model, enhancer_region, bg_path, ppt_imgs_folder, duration_csv_path)


    torch.cuda.empty_cache()
    if args.use_DAIN:
        import paddle
        from src.dain_model import dain_predictor
        paddle.enable_static()
        predictor_dian = dain_predictor.DAINPredictor(args.dian_output, weight_path=args.DAIN_weight,
                                                      time_step=args.time_step,
                                                      remove_duplicates=args.remove_duplicates)
        frames_path, temp_video_path = predictor_dian.run(tmp_path)
        paddle.disable_static()
        save_path = return_path[:-4] + '_dain.mp4'
        command = r'ffmpeg -y -i "%s" -i "%s" -vcodec copy "%s"' % (temp_video_path, new_audio_path, save_path)
        os.system(command)
    os.remove(tmp_path)
    os.remove(extented_pic_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--driven_audio", default='./examples/driven_audio/test_ppt_speech_output',
                        help="path to driven audio")
    parser.add_argument("--source_video", default='./examples/driven_video/test16.mp4',
                        help="path to source video")
    parser.add_argument("--checkpoint_dir", default='./checkpoints', help="path to output")
    parser.add_argument("--result_dir", default='./results', help="path to output")
    parser.add_argument("--batch_size", type=int, default=1, help="the batch size of facerender")
    parser.add_argument("--enhancer", type=str, default='face', help="enhaner region:[none,lip,face] \
                                                                      none:do not enhance; \
                                                                      lip:only enhance lip region \
                                                                      face: enhance (skin nose eye brow lip) region")
    parser.add_argument("--cpu", dest="cpu", action="store_true")

    parser.add_argument("--use_DAIN", dest="use_DAIN", action="store_true",
                        help="Depth-Aware Video Frame Interpolation")
    parser.add_argument('--DAIN_weight', type=str, default='./checkpoints/DAIN_weight',
                        help='Path to model weight')
    parser.add_argument('--dian_output', type=str, default='dian_output', help='output dir')
    parser.add_argument('--time_step', type=float, default=0.5, help='choose the time steps')
    parser.add_argument('--remove_duplicates', action='store_true', default=False,
                        help='whether to remove duplicated frames')
    #./examples/bg_imgs/bg_test.png
    parser.add_argument('--bg_img', type=str, default='./examples/bg_imgs/bg_test.png', help='path to background image')
    parser.add_argument('--ppt_path', type=str, default='./examples/ppt/1234.pdf', help='path to ppt pdf file')

    args = parser.parse_args()
    if torch.cuda.is_available() and not args.cpu:
        args.device = "cuda"
    else:
        args.device = "cpu"
    # driven_audios = ['1.wav', '2.wav', '3.wav', '4.wav']
    # driven_audio_dir = './examples/driven_audio/output6/'
    # for audio in driven_audios:
    #     args.driven_audio = os.path.join(driven_audio_dir, audio)
    #     print(args.driven_audio)
    #     main(args)
    main(args)


