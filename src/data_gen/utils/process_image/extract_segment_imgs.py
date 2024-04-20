import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.append('./venv/lib/python3.9/site-packages')
import glob
import cv2
import tqdm
import numpy as np
import PIL
from src.utils.commons.tensor_utils import convert_to_np
import torch
import mediapipe as mp
from src.utils.commons.multiprocess_utils import multiprocess_run_tqdm
from src.data_gen.utils.mp_feature_extractors.mp_segmenter import MediapipeSegmenter
from src.data_gen.utils.process_video.extract_segment_imgs import inpaint_torso_job, extract_background, save_rgb_image_to_path
seg_model = MediapipeSegmenter()

def extract_img_segment_and_compose_background(img, img_name, bg_img_path='', person_scale=0.8):
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_size = img.shape
        img = cv2.resize(img, (int(img_size[1] * person_scale), int(img_size[0] * person_scale)))
        #img_name_with_extension = os.path.basename(img_name)
        #img_name_without_extension = os.path.splitext(img_name_with_extension)[0]
        bg_img_with_extension = os.path.basename(bg_img_path)
        bg_img_name_without_extension = os.path.splitext(bg_img_with_extension)[0]
        save_img = img_name + '_' + bg_img_name_without_extension + '.png'
        segmap = seg_model._cal_seg_map(img)
        if bg_img_path == '':
            bg_img = extract_background([img], [segmap])
            out_img_name = os.path.join('./results/bg_img/', save_img)
            # out_img_name = img_name.replace("/images_512/", f"/bg_img/").replace(".mp4", ".jpg")
            save_rgb_image_to_path(bg_img, out_img_name)
        else:
            bg_img = cv2.imread(bg_img_path)
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
            bg_img = cv2.resize(bg_img, (1920, 1080))
            com_img = bg_img.copy()
            # bg_part = segmap[0].astype(bool)[..., None].repeat(3,axis=-1)
            shape_part = (segmap[[1, 2, 3, 4, 5], :, :].sum(axis=0)[None, :] > 0.5).repeat(3, axis=0).transpose(1, 2, 0)
            target_shape = com_img.shape
            pad_height = max(0, target_shape[0] - shape_part.shape[0])
            pad_width = max(0, target_shape[1] - shape_part.shape[1])
            shape_resize_part = np.pad(shape_part, ((pad_height, 0), (pad_width, 0), (0, 0)), constant_values=False)
            resized_img = img.copy()
            resized_img = np.pad(resized_img, ((pad_height, 0), (pad_width, 0), (0, 0)), constant_values=0)
            com_img[shape_resize_part] = resized_img[shape_resize_part]
            out_img_name = os.path.join('./results/com_imgs/', save_img)
            # save_rgb_image_to_path(com_img, out_img_name)
            return com_img
    except Exception as e:
        print(e)
        return img

def extract_segment_job(img_name, bg_img_path='', person_scale=1):
    try:
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_size = img.shape
        img = cv2.resize(img, (int(img_size[1] * person_scale), int(img_size[0] * person_scale)))
        img_name_with_extension = os.path.basename(img_name)
        img_name_without_extension = os.path.splitext(img_name_with_extension)[0]
        bg_img_with_extension = os.path.basename(bg_img_path)
        bg_img_name_without_extension = os.path.splitext(bg_img_with_extension)[0]
        save_img = img_name_without_extension + '_' + bg_img_name_without_extension + '.png'
        segmap = seg_model._cal_seg_map(img)
        if bg_img_path == '':
            bg_img = extract_background([img], [segmap])
            out_img_name = os.path.join('./results/bg_img/', save_img)
            # out_img_name = img_name.replace("/images_512/", f"/bg_img/").replace(".mp4", ".jpg")
            save_rgb_image_to_path(bg_img, out_img_name)
        else:
            bg_img = cv2.imread(bg_img_path)
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
            bg_img = cv2.resize(bg_img, (1920, 1080))
        com_img = bg_img.copy()
        #bg_part = segmap[0].astype(bool)[..., None].repeat(3,axis=-1)
        shape_part = (segmap[[1, 2, 3, 4, 5], :, :].sum(axis=0)[None, :] > 0.5).repeat(3, axis=0).transpose(1, 2, 0)
        target_shape = com_img.shape
        pad_height = max(0, target_shape[0] - shape_part.shape[0])
        pad_width = max(0, target_shape[1] - shape_part.shape[1])
        shape_resize_part = np.pad(shape_part, ((pad_height, 0), (pad_width, 0), (0, 0)), constant_values=False)
        resized_img = img.copy()
        resized_img = np.pad(resized_img, ((pad_height, 0), (pad_width, 0), (0, 0)), constant_values=0)
        com_img[shape_resize_part] = resized_img[shape_resize_part]
        # com_img[bg_part] = bg_img[bg_part]
        # out_img_name = img_name.replace("/images_512/",f"/com_imgs/")
        out_img_name = os.path.join('./results/com_imgs/', save_img)
        save_rgb_image_to_path(com_img, out_img_name)

        for mode in ['head', 'torso', 'person', 'torso_with_bg', 'bg']:
            out_img, _ = seg_model._seg_out_img_with_segmap(img, segmap, mode=mode)
            out_img_name = os.path.join(f'./results/{mode}_imgs/', save_img)
            # out_img_name = img_name.replace("/images_512/",f"/{mode}_imgs/")
            out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
            try: os.makedirs(os.path.dirname(out_img_name), exist_ok=True)
            except: pass
            cv2.imwrite(out_img_name, out_img)

        # inpaint_torso_img, _, inpaint_torso_with_bg_img, _ = inpaint_torso_job(img, segmap)
        # # out_img_name = img_name.replace("/images_512/",f"/inpaint_torso_imgs/")
        # out_img_name = os.path.join('./results/inpaint_torso_imgs/', save_img)
        # save_rgb_image_to_path(inpaint_torso_img, out_img_name)
        # # too many indices for array: array is 2-dimensional, but 3 were indexed
        # inpaint_torso_with_bg_img[bg_part] = bg_img[bg_part]
        # # out_img_name = img_name.replace("/images_512/",f"/inpaint_torso_with_com_bg_imgs/")
        # out_img_name = os.path.join('./results/inpaint_torso_with_com_bg_imgs/', save_img)
        # save_rgb_image_to_path(inpaint_torso_with_bg_img, out_img_name)
        return 0
    except Exception as e:
        print(e)
        return 1

def out_exist_job(img_name, bg_name=''):
    img_name = img_name + bg_name
    out_name1 = img_name.replace("/images_512/", "/head_imgs/")
    out_name2 = img_name.replace("/images_512/", "/com_imgs/")
    out_name3 = img_name.replace("/images_512/", "/inpaint_torso_with_com_bg_imgs/")
    
    if  os.path.exists(out_name1) and os.path.exists(out_name2) and os.path.exists(out_name3):
        return None
    else:
        return img_name

def get_todo_img_names(img_names):
    todo_img_names = []
    for i, res in multiprocess_run_tqdm(out_exist_job, img_names, num_workers=64):
        if res is not None:
            todo_img_names.append(res)
    return todo_img_names


if __name__ == '__main__':
    import argparse, glob, tqdm, random
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default='./images_512')
    # parser.add_argument("--img_dir", default='/home/tiger/datasets/raw/FFHQ/images_512')
    parser.add_argument("--ds_name", default='FFHQ')
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--process_id", default=0, type=int)
    parser.add_argument("--total_process", default=1, type=int)
    parser.add_argument("--reset", action='store_true')
    parser.add_argument("--bg_dir", default='./bg_img', type=str)
    parser.add_argument("--person_scale", default=0.5, type=float)
    args = parser.parse_args()
    img_dir = args.img_dir
    bg_dir = args.bg_dir
    person_scale = args.person_scale
    if args.ds_name == 'FFHQ_MV':
        img_name_pattern1 = os.path.join(img_dir, "ref_imgs/*.png")
        img_names1 = glob.glob(img_name_pattern1)
        img_name_pattern2 = os.path.join(img_dir, "mv_imgs/*.png")
        img_names2 = glob.glob(img_name_pattern2)
        img_names = img_names1 + img_names2
    elif args.ds_name == 'FFHQ':
        img_name_pattern = os.path.join(img_dir, "*.png")
        img_names = glob.glob(img_name_pattern)


    img_names = sorted(img_names)
    random.seed(args.seed)
    random.shuffle(img_names)

    process_id = args.process_id
    total_process = args.total_process
    if total_process > 1:
        assert process_id <= total_process -1
        num_samples_per_process = len(img_names) // total_process
        if process_id == total_process:
            img_names = img_names[process_id * num_samples_per_process : ]
        else:
            img_names = img_names[process_id * num_samples_per_process : (process_id+1) * num_samples_per_process]

    img_and_bg_names = []
    for img_name in img_names:
        img_name_with_extension = os.path.basename(img_name)
        img_name_without_extension = os.path.splitext(img_name_with_extension)[0]
        bg_img_name_pattern = os.path.join(bg_dir, f"{img_name_without_extension}/*.png")
        bg_img_names = glob.glob(bg_img_name_pattern)
        if len(bg_img_names) == 0:
            img_and_bg_names.append([img_name, '', person_scale])
            continue
        for bg_name in bg_img_names:
            img_and_bg_names.append([img_name, bg_name, person_scale])
    # if not args.reset:
    #     img_and_bg_names = get_todo_img_names(img_and_bg_names)

    print(f"todo images number: {len(img_and_bg_names)}")

    for vid_name in multiprocess_run_tqdm(extract_segment_job ,img_and_bg_names, desc=f"Root process {args.process_id}: extracting segment images", num_workers=args.num_workers):
        pass