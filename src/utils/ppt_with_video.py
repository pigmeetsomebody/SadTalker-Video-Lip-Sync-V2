import cv2
import os
import openpyxl
import re
import argparse
from moviepy.editor import VideoFileClip, AudioFileClip
import pandas as pd

#计算每行的字数总和
def calculate_sum(text):
    chinese_count = len(re.findall(r'[\u4e00-\u9fff]', text))
    english_count = len(re.findall(r'[a-zA-Z]', text))
    digit_count = len(re.findall(r'\d', text))
    return chinese_count + english_count + digit_count

def paste_ppt_to_video(video_file, image_folder, duration_csv, save_name):
    # 读取视频
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # 读取图片文件夹中的图片
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    sorted_images = sorted(images, key=lambda x: int(x.split('.')[0]))
    num_images = len(images)

    # 计算每个图片应该持续的帧数 (后续根据文本修改)
    # frames_per_image = int(total_frames / num_images)

    data = pd.read_csv('audio_durations_output.csv', sep=',')
    total_duration = sum(data['duration'])
    frames_per_image = []
    for d in data['duration']:
        exp_frame_nums = (int)(total_frames * d / total_duration)
        frames_per_image.append(exp_frame_nums)


    # for result in results:
    #     frames_per_image.append((int)(total_frames * result / word_sum))

    # 创建视频写入对象
    output_video = cv2.VideoWriter('output_video_with_fade.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                   (int(cap.get(3)), int(cap.get(4))))
    alpha = 0.9  # 初始透明度
    fade_frames = 5  # 渐变帧数
    video_width = int(cap.get(3))
    video_height = int(cap.get(4))
    frame_num = 0
    for i in range(num_images):
        img = cv2.imread(os.path.join(image_folder, sorted_images[i]))
        img_height, img_width = img.shape[:2]
        img_width = int(video_width * 0.618)
        img_height = int(img_height * 0.618)

        # 调整ppt的大小
        img_resized = cv2.resize(img, (img_width, img_height))

        start_frame = int(frame_num)
        end_frame = int(frame_num + frames_per_image[i])

        while frame_num < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # 计算插入位置，左边居中
            y_offset = int((frame.shape[0] - img_height) / 2)
            x_offset = 0

            overlay = frame.copy()
            cv2.addWeighted(img_resized, alpha,
                            overlay[y_offset:y_offset + int(img_height), x_offset:x_offset + int(img_width)], 1 - alpha,
                            0, frame[y_offset:y_offset + int(img_height), x_offset:x_offset + int(img_width)])

            output_video.write(frame)

            frame_num += 1

        # 渐变效果
        for j in range(fade_frames):
            ret, frame = cap.read()
            if not ret:
                break

            overlay = frame.copy()
            cv2.addWeighted(img_resized, 1 - alpha * (j + 1) / fade_frames,
                            overlay[y_offset:y_offset + int(img_height), x_offset:x_offset + int(img_width)],
                            alpha * (j + 1) / fade_frames, 0,
                            frame[y_offset:y_offset + int(img_height), x_offset:x_offset + int(img_width)])

            output_video.write(frame)

    output_video.release()
    cap.release()
    cv2.destroyAllWindows()



    video_clip = VideoFileClip('output_video_with_fade.mp4')
    audio_clip = VideoFileClip(video_file).audio
    video_clip = video_clip.set_audio(audio_clip)

    # 保存最终的视频文件
    video_clip.write_videofile(save_name, codec='libx264')

    # 关闭视频和音频文件
    video_clip.close()
    audio_clip.close()
    # 关闭视频和音频文件
    video_clip.close()
    audio_clip.close()

if __name__ == '__main__':
    # 构建命令行参数解析器
    parser = argparse.ArgumentParser(description='Create a video with fade effect using images')
    parser.add_argument('--video_file', default='sample.mp4', type=str, help='Path to the video file')
    parser.add_argument('--image_folder', default='./output6/imgs',type=str, help='Path to the image folder')
    parser.add_argument('--duration_csv', default='audio_durations_output.csv',type=str, help='Path to the excel file')
    parser.add_argument('--save_name', default='sample_with_ppt.mp4', type=str, help='savename')
    args = parser.parse_args()

    paste_ppt_to_video(args.video_file, args.image_folder, args.duration_csv, args.save_name)

    cmd = f'ffmpeg -i {args.video_file} -i {args.save_name} -c:v copy -c:a copy -map 0:a:0 -map 1:v:0 {args.save_name.replace(".mp4", "_full.mp4")}'
    os.system(cmd)