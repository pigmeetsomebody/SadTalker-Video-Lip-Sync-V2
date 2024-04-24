import fitz
import argparse
import os


def process_ppt_to_imgs(pdf_path, save_img_dir):
    # 打开PDF文件
    pdf_file = pdf_path
    save_file = save_img_dir
    pdf_document = fitz.open(pdf_file)

    if not os.path.exists(save_file):
        os.makedirs(save_file)

    # 遍历每一页
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]

        # 获取页面的图像
        image = page.get_pixmap()

        # 设置保存图片的路径
        save_path = f"{save_file}/{page_num + 1}.png"
        # 保存页面为图片
        image.save(save_path)

    # 关闭PDF文件
    pdf_document.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a video with fade effect using images')
    parser.add_argument('--pdf_file', default='1234.pdf', type=str, help='Path to the video file')
    parser.add_argument('--save_file', default='output6/imgs/' ,type=str, help='save')
    args = parser.parse_args()

    process_ppt_to_imgs(args.pdf_file, args.save_file)