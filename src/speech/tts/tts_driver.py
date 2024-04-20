from abc import ABC, abstractmethod
import logging
from .edge_tts import Edge, edge_voices
from .bert_vits2 import BertVits2

logger = logging.getLogger(__name__)


class BaseTTS(ABC):
    '''合成语音统一抽象类'''

    @abstractmethod
    def synthesis(self, text: str, voice_id: str, **kwargs) -> str:
        '''合成语音'''
        pass

    @abstractmethod
    def get_voices(self) -> list[dict[str, str]]:
        '''获取声音列表'''
        pass


class EdgeTTS(BaseTTS):
    '''Edge 微软语音合成类'''
    client: Edge

    def __init__(self):
        self.client = Edge()

    def synthesis(self, text: str, voice_id: str, **kwargs) -> str:
        return self.client.create_audio(text=text, voiceId=voice_id)

    def get_voices(self) -> list[dict[str, str]]:
        return edge_voices



class TTSDriver:
    '''TTS驱动类'''

    def synthesis(self, text: str, voice_id: str, **kwargs) -> str:
        tts = EdgeTTS()
        file_name = tts.synthesis(text=text, voice_id=voice_id, kwargs=kwargs)
        #logger.info(f"TTS synthesis # type:{type} text:{text} => file_name: {file_name} #")
        return file_name;

    def get_voices(self) -> list[dict[str, str]]:
        tts = EdgeTTS()
        return tts.get_voices()

