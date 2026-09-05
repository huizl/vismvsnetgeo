import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.eval_region_metrics_dtu_yao import parse_args, build_model
from tools.validate_val_diagnostics import ROOT, build_experiments, main


class ValDiagnosticsTest(unittest.TestCase):
    def settings(self, **kwargs):
        values = dict(suite='all', checkpoint_root='checkpoints/dtu',
                      checkpoint_name='best_2mm.ckpt', train_nviews=5,
                      outdir='eval/example', datapath='data/dtu',
                      batch_size=4, num_workers=0)
        values.update(kwargs)
        return argparse.Namespace(**values)

    def test_matrix_uses_val_and_shared_checkpoints_with_valid_cli(self):
        experiments = build_experiments(self.settings())
        self.assertEqual(len(experiments), 9)
        self.assertEqual(len({e['output'] for e in experiments}), 9)
        m2, m3 = [], []
        for experiment in experiments:
            with patch('sys.argv', experiment['command'][1:]):
                args = parse_args()
            self.assertEqual(Path(args.testlist), ROOT/'lists'/'dtu'/'val.txt')
            self.assertEqual((args.eval_nviews, args.region_nviews, args.light), (5, 5, 3))
            if args.model_type == 'm2_visibility':
                m2.append((args.loadckpt, args.visibility_fusion_beta))
            if args.model_type == 'm3_hybrid':
                m3.append((args.loadckpt, args.hybrid_clip_mode, args.hybrid_max_scale))
                model = build_model(args)
                self.assertEqual(model.hybrid_clip_mode, args.hybrid_clip_mode)
                self.assertEqual(model.hybrid_max_scale, args.hybrid_max_scale)
        self.assertEqual(len({r[0] for r in m2}), 1)
        self.assertEqual({r[1] for r in m2}, {0.0, 0.1, 0.2, 0.3})
        self.assertEqual(len({r[0] for r in m3}), 1)
        self.assertEqual({r[1:] for r in m3}, {('global', 1.0), ('global', 2.0),
                                               ('none', 1.0), ('none', 2.0)})

    def test_dry_run_does_not_create_outputs_or_start_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'dry'
            with patch('sys.argv', ['validate', '--suite', 'm3', '--train_nviews', '3',
                                    '--outdir', str(output), '--dry_run']), \
                    patch('builtins.print'), patch('subprocess.run') as run:
                main()
            run.assert_not_called()
            self.assertFalse(output.exists())
        experiments = build_experiments(self.settings(suite='m3', train_nviews=3))
        self.assertEqual(len(experiments), 5)
        self.assertTrue(all('_view3' in e['checkpoint'] for e in experiments))

    def test_m3_vis_suite_uses_only_the_baseline_checkpoint(self):
        experiments = build_experiments(self.settings(suite='m3_vis'))
        self.assertEqual(len(experiments), 3)
        self.assertEqual(len({e['checkpoint'] for e in experiments}), 1)
        self.assertTrue(all('vis_view5' in e['checkpoint'] for e in experiments))
        parsed = []
        for experiment in experiments:
            with patch('sys.argv', experiment['command'][1:]):
                parsed.append(parse_args())
        self.assertEqual(parsed[0].model_type, 'vis')
        self.assertEqual([a.hybrid_max_scale for a in parsed[1:]], [1.0, 2.0])
        self.assertTrue(all(a.hybrid_clip_mode == 'none' and a.model_type == 'm3_hybrid'
                            for a in parsed[1:]))


if __name__ == '__main__':
    unittest.main()
