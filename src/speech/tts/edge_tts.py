import sys
import logging
import os
import subprocess

#from ..utils.uuid_generator import generate

logger = logging.getLogger(__name__)

edge_voices = [
    {"id": "zh-CN-XiaoxiaoNeural", "name": "xiaoxiao"},
    {"id": "zh-CN-XiaoyiNeural", "name": "xiaoyi"},
    {"id": "zh-CN-YunjianNeural", "name": "yunjian"},
    {"id": "zh-CN-YunxiNeural", "name": "yunxi"},
    {"id": "zh-CN-YunxiaNeural", "name": "yunxia"},
    {"id": "zh-CN-YunyangNeural", "name": "yunyang"},
    {"id": "zh-CN-liaoning-XiaobeiNeural", "name": "xiaobei"},
    {"id": "zh-CN-shaanxi-XiaoniNeural", "name": "xiaoni"},
    {"id": "zh-HK-HiuGaaiNeural", "name": "hiugaai"},
    {"id": "zh-HK-HiuMaanNeural", "name": "hiumaan"},
    {"id": "zh-HK-WanLungNeural", "name": "wanlung"},
    {"id": "zh-TW-HsiaoChenNeural", "name": "hsiaochen"},
    {"id": "zh-TW-HsiaoYuNeural", "name": "hsioayu"},
    {"id": "zh-TW-YunJheNeural", "name": "yunjhe"}
]


class Edge():

    def remove_html(self, text: str):
        # TODO 待改成正则
        new_text = text.replace('[', "")
        new_text = new_text.replace(']', "")
        return new_text

    def create_audio(self, text: str, voiceId: str, output: str = ''):
        new_text = self.remove_html(text)
        examplePath = os.path.join(os.getcwd(), 'examples')
        audioPath = os.path.join(examplePath, 'driven_audio')
        file_name = voiceId +'.mp3'
        filePath = os.path.join(audioPath, file_name)
        dirPath = os.path.dirname(filePath)
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
        if not os.path.exists(filePath):
            # 用open创建文件 兼容mac
            open(filePath, 'a').close()
        cmd = r'edge-tts --voice %s --text "%s" --write-media %s' % (voiceId, new_text, str(filePath))
        print(cmd)
        os.system(cmd)

        return filePath

if __name__ == '__main__':
    tts_driver = Edge()
    for voice in edge_voices:
        print(voice["id"])
        s = 'Good Morning Folks. My Name is Xiangming Li and I am an associate professor from Shenzhen International Graduate School, Tsinghua University.'
        file = tts_driver.create_audio(text=s, voiceId='en-US-JennyNeural', output='sample')
        print(file)
