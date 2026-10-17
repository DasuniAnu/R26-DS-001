"""Offline regression tests: python -m unittest discover -s backend -p test_asr.py."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import asr_manager
import youtube_utils
import app as api


class Tokenizer:
    def decode(self, ids, **kwargs):
        assert kwargs['group_tokens'] is False
        return ''.join({1: 'ක', 2: 'ම', 3: 'ල'}[i] for i in ids)


class RecognitionTests(unittest.TestCase):
    def decode(self, ids, strength=10, **kwargs):
        logits = torch.zeros((len(ids), 4))
        for frame, token in enumerate(ids):
            logits[frame, token] = strength
        return asr_manager.decode_segment(logits, SimpleNamespace(tokenizer=Tokenizer()), 0, **kwargs)

    def test_context_is_excluded_and_repeated_letters_survive(self):
        result = self.decode([3, 0, 1, 1, 0, 1, 0, 2, 0, 3], frame_start=2, frame_end=8)
        self.assertEqual(result['text'], 'කකම')
        self.assertFalse(result['needs_review'])

    def test_boundary_run_is_assigned_once(self):
        ids = [0, 1, 1, 1, 1, 0, 2]
        left = self.decode(ids, frame_end=3)
        right = self.decode(ids, frame_start=3)
        self.assertEqual(left['text'] + right['text'], 'කම')

    def test_low_scores_and_blank_only_need_review(self):
        self.assertTrue(self.decode([0, 1, 0, 2], strength=0.2)['needs_review'])
        self.assertTrue(self.decode([0] * 50)['needs_review'])

    def test_blank_frames_do_not_inflate_score(self):
        logits = torch.zeros((100, 4))
        logits[:, 0] = 10
        logits[50] = torch.tensor([0.0, 0.2, 0.0, 0.0])
        result = asr_manager.decode_segment(logits, SimpleNamespace(tokenizer=Tokenizer()), 0)
        self.assertLess(result['score'], 0.4)
        self.assertTrue(result['needs_review'])

    def test_silence_and_nonfinite_input_do_not_load_model(self):
        with patch.object(asr_manager, '_load_model') as load:
            for audio in (np.zeros(1000), np.array([np.nan]), np.array([])):
                self.assertTrue(asr_manager.transcribe_sinhala_audio(audio)['needs_review'])
            load.assert_not_called()

    def test_context_and_original_deepfake_audio_are_separate(self):
        samples = np.linspace(-0.4, 0.4, 19 * 16000, dtype=np.float32)
        with patch.object(youtube_utils, 'get_video_metadata', return_value={'duration': 19}), \
             patch.object(youtube_utils, 'download_youtube_audio', return_value=('unused.wav', {})), \
             patch.object(youtube_utils.os.path, 'exists', return_value=True), \
             patch.object(youtube_utils.os, 'remove'), \
             patch.object(youtube_utils.librosa, 'load', return_value=(samples, 16000)), \
             patch.object(youtube_utils, '_load_vad', return_value=(None, [lambda *a, **k: []])):
            _, segments = youtube_utils.analyze_youtube_video('unused')
        first, last = segments
        self.assertEqual(len(first['asr_context']), 12 * 16000)
        self.assertEqual(len(last['asr_context']), 11 * 16000)
        self.assertEqual(last['asr_core_start'], 2 * 16000)
        self.assertEqual(last['asr_core_end'], 11 * 16000)
        np.testing.assert_array_equal(last['asr_audio'][:9 * 16000], samples[10 * 16000:])
        self.assertTrue(np.all(last['asr_audio'][9 * 16000:] == 0))
        self.assertTrue(np.all(last['audio'] == 0))

    def request(self, recognition, audio_label='NOT'):
        segment = {'start_time': 0, 'end_time': 10, 'audio': np.ones(160),
                   'asr_audio': np.ones(160), 'asr_context': np.ones(200),
                   'asr_core_start': 20, 'asr_core_end': 180}
        prediction = {'hate_label': audio_label, 'audio_authentic': True, 'authenticity_label': 'REAL'}
        with patch.object(api, 'analyze_youtube_video', return_value=({}, [segment])), \
             patch.object(api, 'predict_audio_array', return_value=prediction), \
             patch.object(api, 'transcribe_sinhala_audio', **recognition) as transcribe, \
             patch.object(api, 'predict_text', return_value={'prediction': 'OFF'}) as classify:
            response = api.app.test_client().post('/api/youtube/analyze', json={'url': 'test'})
        self.assertEqual(response.status_code, 200)
        return response.get_json(), classify, transcribe

    def test_uncertain_transcript_does_not_classify_or_count_clean(self):
        data, classify, _ = self.request({'return_value': {'text': 'කමල', 'score': 0.3,
                                                          'needs_review': True, 'reason': 'uncertain'}})
        classify.assert_not_called()
        self.assertEqual(data['segments'][0]['hate_label'], 'UNCERTAIN')
        self.assertEqual(data['summary']['clean_segments'], 0)
        self.assertEqual(data['summary']['review_segments'], 1)

    def test_accepted_transcript_still_contributes(self):
        data, classify, _ = self.request({'return_value': {'text': 'කමල', 'score': 0.95,
                                                          'needs_review': False, 'reason': ''}})
        classify.assert_called_once_with('කමල')
        self.assertEqual(data['segments'][0]['hate_label'], 'OFF')

    def test_asr_failure_keeps_audio_result(self):
        data, classify, _ = self.request({'side_effect': RuntimeError('test failure')}, 'OFF')
        classify.assert_not_called()
        self.assertEqual(data['segments'][0]['hate_label'], 'OFF')
        self.assertEqual(data['segments'][0]['asr_status'], 'NEEDS_REVIEW')

    def test_no_speech_skips_asr(self):
        data, classify, transcribe = self.request({}, 'NO_SPEECH')
        classify.assert_not_called()
        transcribe.assert_not_called()
        self.assertEqual(data['segments'][0]['hate_label'], 'NOT_ASSESSED')


if __name__ == '__main__':
    unittest.main()
