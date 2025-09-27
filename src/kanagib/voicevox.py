import wave
import requests
import json
import simpleaudio
import io

class VoiceVoxClient:
    def __init__(self, url = "http://localhost:50021"):
        self.url = url

    def get_content(self, text, speaker=0) -> bytes:
        params = (
            ('text', text),
            ('speaker', speaker),
        )
        response1 = requests.post(
            f'{self.url}/audio_query',
            params=params
        )
        headers = {'Content-Type': 'application/json',}
        response2 = requests.post(
            f'{self.url}/synthesis',
            headers=headers,
            params=params,
            data=json.dumps(response1.json())
        )
        return response2.content

    # VOICEVOXで音声合成    
    def generate_wav(self, text, speaker=0, filepath=None) -> tuple[bytes, float]:
        content = self.get_content(text, speaker)
        
        if filepath:
            with open(filepath, "wb") as f:
                f.write(content)
                
        wav_io = io.BytesIO(content)
        with wave.open(wav_io, 'rb') as wf:
            frame_rate = wf.getframerate()
            duration = wf.getnframes() / float(frame_rate)
        
        return content, duration
    
    
    def play_voice(self, text, speaker=8, *, wait=False) -> tuple[simpleaudio.PlayObject, float]:
        content = self.get_content(text, speaker)
        
        # WAVデータをメモリ上で読み込み
        wav_io = io.BytesIO(content)
        
        with wave.open(wav_io, 'rb') as wf:
            audio_data = wf.readframes(wf.getnframes())
            play_obj = simpleaudio.play_buffer(audio_data, wf.getnchannels(), wf.getsampwidth(), wf.getframerate())
            
            # 再生が終了するまで待機
            if wait:
                play_obj.wait_done()
            
            duration = wf.getnframes() / float(wf.getframerate())
            return play_obj, duration    

if __name__ == "__main__":
    """ソーシーアールシラク
ニワヨルノガヒトヤノ
ソーノコムディーニイー
ニツバンモイフエガワ
サクニコーンカーヒサ
マジガズデスカーニホ
ダイトツカントカスエ
コノデチバンシタダニ
トクガホンソージデキ
    """
    #URL = "https://voicevox.local"
    URL = "http://jiro-frontier.local:50021"
    voicevox_client = VoiceVoxClient(URL)
    #voicevox_client.play_voice("こんにちはー、今日は来てくれてありがとうね", 8)
    voicevox_client.generate_wav("マジガズデスカーニホ", speaker=3,filepath="test.wav")