import unittest
import tempfile
from model import synthetic, knowledge_layer, explain, run

class TestReadiness(unittest.TestCase):
    def test_reproducibility(self):
        self.assertTrue(synthetic(150,7).equals(synthetic(150,7)))
    def test_rules(self):
        df=synthetic(120,4)
        df.loc[0,'data_quality']=0.0
        self.assertIn('Improve enterprise data quality',explain(df.iloc[0]))
        self.assertGreater(knowledge_layer(df.iloc[[0]])[0],0)
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=run(synthetic(500,42),tmp)
            self.assertIn('brier',result['knowledge_adjusted'])
    def test_reject_invalid(self):
        df=synthetic(200);df.loc[0,'data_quality']=1.5
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):run(df,tmp)
if __name__=='__main__': unittest.main()
