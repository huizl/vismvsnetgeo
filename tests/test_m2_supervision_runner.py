import argparse
import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from models.model_variants import MODEL_TYPE_CHOICES, get_model_variant
from tools.eval_region_metrics_dtu_yao import parse_args as parse_eval_args
from tools.train_m2_supervision_val import ROOT, build_commands, main


class M2SupervisionRunnerTest(unittest.TestCase):
    def settings(self):
        return argparse.Namespace(datapath='data/dtu', logdir='checkpoints/control',
                                  outdir='eval/control', batch_size=4, train_nviews=5,
                                  seed=1, epochs=16, train_workers=8, eval_workers=4)

    def test_commands_preserve_supervision_and_disable_gate_during_training_and_eval(self):
        train, evaluate = build_commands(self.settings())
        # Execute only parser declarations, avoiding train.py's GPU/data side effects.
        tree = ast.parse((ROOT/'train.py').read_text(encoding='utf-8'))
        statements = []
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'parser' for t in node.targets):
                statements.append(node)
            elif (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                  and isinstance(node.value.func, ast.Attribute)
                  and isinstance(node.value.func.value, ast.Name)
                  and node.value.func.value.id == 'parser' and node.value.func.attr == 'add_argument'):
                statements.append(node)
        namespace = {'argparse': argparse, 'MODEL_TYPE_CHOICES': MODEL_TYPE_CHOICES}
        exec(compile(ast.Module(body=statements, type_ignores=[]), '<training parser>', 'exec'), namespace)
        args = namespace['parser'].parse_args(train[3:])
        self.assertEqual(args.model_type, 'm2_visibility')
        self.assertTrue(get_model_variant(args.model_type).visibility_modeling)
        self.assertEqual(args.visibility_fusion_beta, 0.0)
        self.assertEqual(args.visibility_weight, 0.2)
        self.assertEqual(args.visibility_focal_gamma, 2.0)
        self.assertIsNone(args.loadckpt)
        self.assertFalse(args.resume)
        self.assertEqual(Path(args.trainlist), ROOT/'lists/dtu/train.txt')
        self.assertEqual(Path(args.testlist), ROOT/'lists/dtu/val.txt')
        self.assertEqual((args.nviews, args.eval_nviews, args.epochs, args.seed), (5, 5, 16, 1))
        with patch('sys.argv', evaluate[2:]):
            val = parse_eval_args()
        self.assertEqual(val.model_type, args.model_type)
        self.assertEqual(val.visibility_fusion_beta, 0.0)
        self.assertEqual(val.testlist, args.testlist)
        self.assertEqual(Path(val.loadckpt), Path(args.logdir)/'best_2mm.ckpt')
        self.assertEqual((val.eval_nviews, val.region_nviews, val.light), (5, 5, 3))

    def test_dry_run_does_not_train_or_create_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            logdir, outdir = Path(temp)/'train', Path(temp)/'eval'
            with patch('sys.argv', ['runner', '--logdir', str(logdir), '--outdir', str(outdir), '--dry_run']), \
                    patch('builtins.print'), patch('subprocess.run') as run:
                main()
            run.assert_not_called()
            self.assertFalse(logdir.exists())
            self.assertFalse(outdir.exists())

    def test_failed_training_does_not_start_validation(self):
        import subprocess
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('Rectified', 'Depths', 'Cameras'):
                (root/'data'/name).mkdir(parents=True)
            with patch('sys.argv', ['runner', '--datapath', str(root/'data'),
                                    '--logdir', str(root/'train'), '--outdir', str(root/'eval')]), \
                    patch('builtins.print'), \
                    patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'train')) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    main()
            self.assertEqual(run.call_count, 1)
            self.assertFalse((root/'eval'/'validation.log').exists())


if __name__ == '__main__':
    unittest.main()
