import unittest, tempfile, hashlib
from pathlib import Path
import app

class EvidenceTests(unittest.TestCase):
    def test_known_scenario(self):
        r=app.answer('経費精算はどこから申請する？',{'demo':True})
        self.assertEqual(r['status'],'demo')
        self.assertEqual(r['evidence'][0]['page'],2)
        self.assertIn('【架空】',r['evidence'][0]['title'])

    def test_unsupported_deadline_abstains(self):
        r=app.answer('経費申請の期限はいつまで？',{'demo':True})
        self.assertEqual(r['status'],'unknown')
        self.assertEqual(r['evidence'],[])

    def test_forged_id_rejected(self):
        r=app.validated_evidence({'sufficient':True,'evidence':[{'id':'invented','quote':app.SAMPLES[0]['text']}]},app.SAMPLES)
        self.assertEqual(r,[])

    def test_changed_quote_rejected(self):
        c=app.SAMPLES[0]
        r=app.validated_evidence({'sufficient':True,'evidence':[{'id':c['id'],'quote':'申請期限は必ず翌月十日です。'}]},app.SAMPLES)
        self.assertEqual(r,[])

    def test_whitespace_restores_source(self):
        c=dict(app.SAMPLES[0],text='申請には領収書を添付し、\n利用日と金額を入力します。')
        r=app.validated_evidence({'sufficient':True,'evidence':[{'id':c['id'],'quote':c['text'].replace('\n','')}]},[c])
        self.assertEqual(r[0]['quote'],c['text'])

    def test_insufficient_flag_rejected(self):
        self.assertEqual(app.validated_evidence({'sufficient':False,'evidence':[{'id':app.SAMPLES[0]['id'],'quote':app.SAMPLES[0]['text']}]},app.SAMPLES),[])

    def test_changed_file_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fiction.txt';p.write_bytes(b'fiction version one')
            corpus={'documents':{'x':{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}}}
            c={'doc_id':'x'}
            self.assertEqual(app.current_candidates(corpus,[c]),[c])
            p.write_bytes(b'fiction version two')
            self.assertEqual(app.current_candidates(corpus,[c]),[])

if __name__=='__main__':unittest.main()
