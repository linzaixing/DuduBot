import dashscope
import pyaudio
import threading
import os
import platform
from http import HTTPStatus
from dashscope.api_entities.dashscope_response import SpeechSynthesisResponse
from dashscope.audio.tts import ResultCallback, SpeechSynthesizer, SpeechSynthesisResult
from dashscope.audio.asr import (Recognition, RecognitionCallback, RecognitionResult)
from camera_module.camera_control import CameraController
from dashscope import Generation

import time

# 阿里云模型服务灵积key
dashscope.api_key = 'sk-f2f9a568b36946d8ad4eb777bfcaa6a6'

# 全局变量
mic = None
stream = None
text = ""
history_text = ''
end_time = 0
conversation_active = False
new_message_event = threading.Event()
lock = threading.Lock()


# 语音转文字
class AsrCallback(RecognitionCallback):
    def on_open(self) -> None:
        global mic, stream
        print('RecognitionCallback open.')
        mic = pyaudio.PyAudio()
        stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True)

    def on_close(self) -> None:
        global mic, stream
        print('RecognitionCallback close.')
        if stream is not None:
            stream.stop_stream()
            stream.close()
        if mic is not None:
            mic.terminate()
        stream = None
        mic = None

    def on_event(self, result: RecognitionResult) -> None:
        global text, history_text, end_time, conversation_active, new_message_event
        sentence = result.get_sentence()
        if sentence['end_time'] is not None and sentence['end_time'] > end_time:
            with lock:
                text = sentence['text']
                end_time = sentence['end_time']
            print('me: ' + text)

            if "相机" in text or "摄像头" in text or '检测' in text or '开始录像' in text or '拍张照片' in text or "浏览器" in text or '你好' in text or '退出对话' in text:
                if history_text == text:
                    print('与历史记录重复，忽略系统生成音频的输入', history_text)
                    pass

                else:
                    print('触发关键词：', text, history_text)
                    # 忽略系统生成音频的输入
                    history_text = text
                    if "关闭相机" in text:
                        cap_callback.close_camera()
                    elif '拍张照片' in text:
                        cap_callback.save_capture(1, os.getcwd() + '/images')
                    elif '开始录像' in text:
                        cap_callback.save_capture(2, os.getcwd() + '/videos')
                    elif '人体检测' in text:
                        # 模型文件的路径
                        cap_callback.frame_edit(model_path)
                    elif '退出对话' in text:
                        print(text)
                        create_voice('收到，当前对话已经结束', tts_callback)
                        asr_thread.join()
                        tts_thread.join()
                    else:
                        conversation_active = True
                        new_message_event.set()


# 文字转语音
class TtsCallback(ResultCallback):
    _player = None
    _stream = None

    def on_open(self):
        print('Speech synthesizer is opened.')
        self._player = pyaudio.PyAudio()
        self._stream = self._player.open(format=pyaudio.paInt16, channels=1, rate=48000, output=True)

    def on_complete(self):
        print('Speech synthesizer is completed.')

    def on_error(self, response: SpeechSynthesisResponse):
        print('Speech synthesizer failed, response is %s' % (str(response)))

    def on_close(self):
        print('Speech synthesizer is closed.')
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._player:
            self._player.terminate()

    def on_event(self, result: SpeechSynthesisResult):
        if result.get_audio_frame() is not None:
            self._stream.write(result.get_audio_frame())

        if result.get_timestamp() is not None:
            print('timestamp result:', str(result.get_timestamp()))


# 调用语音转文字大模型
def create_text(callback):
    # paraformer-realtime-v1 和 paraformer-realtime-v2 仅支持16khz  paraformer-realtime-8k-v1
    recognition = Recognition(model='paraformer-realtime-v1', format='pcm', sample_rate=16000, callback=callback)
    return recognition


# 调用文字转语音大模型
def create_voice(text_content, tts_callback):
    SpeechSynthesizer.call(
        model='sambert-zhiying-v1',  # 语言调整模型，萝莉音等等
        text=text_content,
        sample_rate=48000,
        format='pcm',
        rate=1.2,
        pitch=1,
        volume=80,
        callback=tts_callback
    )


# 通义千问对话大模型
def sample_sync_call(prompt_text):
    resp = dashscope.Generation.call(model='qwen-plus', prompt=prompt_text)
    if resp.status_code == HTTPStatus.OK:
        print('Chat robot: ' + resp.output.text)
        return resp.output.text
    else:
        print(resp.code)
        print(resp.message)


def asr_thread_func(asr_callback):
    recognition = create_text(asr_callback)
    recognition.start()

    try:
        while True:
            if stream:
                data = stream.read(3200, exception_on_overflow=False)
                recognition.send_audio_frame(data)
    except KeyboardInterrupt:
        pass
    finally:
        recognition.stop()


def tts_thread_func(tts_callback):
    global conversation_active, text, new_message_event, camera_thread, history_text

    while True:
        new_message_event.wait()
        if conversation_active:
            with lock:
                current_text = text
            print('Processing message:', current_text)

            if '打开相机' in current_text or '打开摄像头' in current_text:
                response_message = "正在打开camera，请等待"
                create_voice(response_message, tts_callback)
                cap_callback.open_camera()
                response_message = "camera已经关闭"
            elif '打开浏览器' in current_text:
                response_message = "正在打开chrome浏览器"
                os.system("start chrome")
            elif '关闭浏览器' in current_text:
                response_message = "正在关闭chrome浏览器"
                os_name = platform.system()
                if os_name == "Linux":
                    os.system("pkill -f chrome")
                elif os_name == "Windows":
                    os.system('taskkill /IM chrome.exe /F')
                else:
                    return "暂不支持关闭该平台"
            elif '你好' in current_text:
                response_message = sample_sync_call(current_text)
            else:
                new_message_event.clear()
                conversation_active = False
                response_message = None

            create_voice(response_message, tts_callback)

            new_message_event.clear()
            conversation_active = False


if __name__ == '__main__':
    model_path = r"./models/yolov8s.onnx"
    asr_callback = AsrCallback()
    tts_callback = TtsCallback()
    cap_callback = CameraController()

    asr_thread = threading.Thread(target=asr_thread_func, args=(asr_callback,))
    tts_thread = threading.Thread(target=tts_thread_func, args=(tts_callback,))

    asr_thread.start()
    tts_thread.start()
