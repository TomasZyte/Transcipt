from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tg-bot'))
import offline_release as release
from saved_benchmark_review import measure


class OfflineReleaseTests(unittest.TestCase):
    def test_markup_escapes_transcript_and_keeps_uncertainty(self):
        row={'id':'left-0','channel':'left','startSec':1.,'endSec':2.,'text':'<script>alert(1)</script>',
             'original_text':'<img src=x onerror=alert(2)>','words':[],
             'reasons':['uncertain_recognition'],'overlaps':[]}
        report={'counts':{'requires_review':1,'speech_runs':1},'rows':[row],'control_text':{'left':'<b>control</b>'}}
        page=release.phone_html(report)
        self.assertNotIn('<script>alert(1)</script>',page)
        self.assertNotIn('<img src=x',page)
        self.assertIn('&lt;script&gt;',page)
        self.assertIn('ПРОВЕРИТЬ',page)
        self.assertIn('default-src',page)

    def test_unknown_label_is_not_a_verified_person(self):
        output={'rawAsrText':'Пример','quality':{'issues':[]},'segments':[{'speaker':'SPEAKER_UNKNOWN'}],
                'attributed':{'segments':[{'words':[{'word':'Пример','speaker':'SPEAKER_UNKNOWN','start':0.,'end':.3}]}]}}
        result=measure(output,timeline=[{'speaker':'SPEAKER_00'}])
        self.assertFalse(result['human_quality_validated'])
        self.assertEqual(result['unknown_words'],1)
        self.assertEqual(result['known_word_labels'],0)

    def test_existing_output_is_never_overwritten(self):
        with self.assertRaisesRegex(ValueError,'Never overwrite'):
            release.build(ROOT,ROOT,ROOT)

    def test_cli_network_guard_blocks_real_socket_before_connect(self):
        code="import sys;sys.path.insert(0,'tg-bot');import offline_release as r;import socket;sys.addaudithook(r.deny_network);socket.create_connection(('127.0.0.1',9),timeout=.01)"
        proc=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True)
        self.assertNotEqual(proc.returncode,0)
        self.assertIn('Offline release forbids networking',proc.stderr)
        self.assertNotIn('ConnectionRefusedError',proc.stderr)


if __name__=='__main__':unittest.main()
