"""M4 evidence boundary, orchestration, reproducibility and report tests."""
import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import ValidationError

from data_agent.clustering import load_input
from data_agent.clustering.core import select_features, scale_features
from data_agent.io import sha256_file
from data_agent.experiments.strategies import ExperimentProtocol, _train, import_agent_result, read_json, source, verify_fixed_strategy, write_json
from data_agent.experiments.report import generate_report
from data_agent.experiments.clustering import compare_clustering, load_cluster_group

ROOT = Path(__file__).resolve().parents[3]


class M4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = load_input('1', project_root=ROOT)
        cls.after = load_input('5', project_root=ROOT)
        cls.historical_after = load_input('2', project_root=ROOT)

    def test_protocol_and_minimal_A(self):
        protocol = ExperimentProtocol()
        self.assertEqual(protocol.ks, tuple(range(2,9)))
        for args in ({'ks': (2,3)}, {'seeds': (42,42)}, {'n_init': 0}, {'features': ('code',)}):
            with self.assertRaises(ValidationError):
                ExperimentProtocol(**args)
        before = self.before.frame.copy(deep=True)
        selected = select_features(self.before, protocol.features)
        expected = before[list(protocol.features)].apply(pd.to_numeric, errors='coerce')
        valid = np.isfinite(expected.to_numpy()).all(axis=1)
        self.assertEqual(selected.row_ids, tuple(before.loc[valid,self.before.row_id_column]))
        pd.testing.assert_frame_equal(before, self.before.frame)

    def test_B_provenance(self):
        evidence = verify_fixed_strategy(ROOT, self.before, self.after)
        self.assertEqual(evidence['status'], 'verified')
        plan_ids = ('off_52k_v2_conservative', 'off_52k_v3_stratified_repair',
                    'off_52k_v4_convergence', 'off_52k_v5_precision_finalization')
        expected = sum(len(read_json(ROOT/'config/plans'/f'{name}.json')['decisions']) for name in plan_ids)
        self.assertEqual(evidence['versions'], ['2', '3', '4', '5'])
        self.assertEqual(evidence['decision_count'], expected)
        bad = copy.deepcopy(self.after.manifest)
        bad['parent']['sha256'] = '0'*64
        with self.assertRaises(ValueError):
            verify_fixed_strategy(ROOT, self.before, replace(self.after, manifest=bad))

    def agent_fixture(self, directory):
        # Synthetic evidence contract fixture ONLY; never an experimental C result.
        profile = read_json(self.before.profile_path)
        frame = self.before.frame.iloc[:12].copy()
        data_path = directory/'fixture.parquet'
        frame.to_parquet(data_path,engine='fastparquet',index=False)
        digest=sha256_file(data_path)
        identity={'dataset_id':'c_contract_fixture','version':'fixture','sha256':digest}
        manifest={'dataset_id':identity['dataset_id'],'version':identity['version'],'parent':self.before.ref.model_dump(mode='json'),
                  'base_file':{'path':str(data_path),'format':'parquet','sha256':digest,'row_count':len(frame),'column_count':len(frame.columns)}}
        write_json(directory/'manifest.json',manifest)
        profile.update(dataset_id=identity['dataset_id'],manifest=str(directory/'manifest.json'))
        write_json(directory/'profile.json',profile)
        call={'id':'call-fixture','name':'execute_plan','arguments':{'plan':'fixture'}}
        export={'run_id':'TEST_FIXTURE_ONLY','source_sha256':self.before.ref.sha256,'responses':[{'id':'response-fixture','model':'fixture','tool_calls':[call]}]}
        write_json(directory/'export.json',export)
        decision={'response_id':'response-fixture','tool_call_id':'call-fixture','tool_name':call['name'],'arguments':call['arguments']}
        write_json(directory/'decisions.json',[decision])
        write_json(directory/'tools.json',[{**decision,'status':'success','input_sha256':self.before.ref.sha256,'output_sha256':digest}])
        write_json(directory/'review.json',{'reviewer':'TEST_FIXTURE_ONLY','reviewed_at':'fixture','actual_provider_execution_verified':True,'provider_export_sha256':sha256_file(directory/'export.json')})
        bundle={'source_dataset':self.before.ref.model_dump(mode='json'),'output_dataset':identity}
        for key,name in [('output_profile','profile.json'),('provider_export','export.json'),('decisions','decisions.json'),('tool_executions','tools.json'),('provenance_review','review.json')]:
            bundle[key]={'path':name,'sha256':sha256_file(directory/name)}
        write_json(directory/'agent_bundle.json',bundle)
        return directory/'agent_bundle.json'

    def test_C_import_contract_and_rejections(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            bundle=self.agent_fixture(directory)
            data,evidence=import_agent_result(bundle,self.before,directory)
            self.assertEqual(len(data.frame),12)
            self.assertEqual(evidence['status'],'imported')
            # Tampering is detected, even if the claimed status is success.
            original=read_json(bundle)
            changed=copy.deepcopy(original); changed['source_dataset']['sha256']='0'*64
            write_json(directory/'bad_source.json',changed)
            with self.assertRaises(ValueError): import_agent_result(directory/'bad_source.json',self.before,directory)
            changed=copy.deepcopy(original); changed['provider_export']['sha256']='0'*64
            write_json(directory/'bad_hash.json',changed)
            with self.assertRaises(ValueError): import_agent_result(directory/'bad_hash.json',self.before,directory)
            changed=copy.deepcopy(original); changed['output_profile']['path']='../outside.json'
            write_json(directory/'bad_path.json',changed)
            with self.assertRaises(ValueError): import_agent_result(directory/'bad_path.json',self.before,directory)

    def test_small_real_algorithm_reproducibility(self):
        protocol=ExperimentProtocol(n_init=2,seeds=(42,7),sample_size=20)
        data=replace(self.after,frame=self.after.frame.iloc[:40].copy())
        selected=select_features(data)
        scaler=scale_features(selected).scaler
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            left=_train(data,selected,scaler,root/'left',protocol)
            right=_train(data,selected,scaler,root/'right',protocol)
            self.assertEqual(left['matrix_sha256'],right['matrix_sha256'])
            for k in protocol.ks:
                self.assertEqual(sha256_file(root/'left'/f'labels_k{k}.csv'),sha256_file(root/'right'/f'labels_k{k}.csv'))
            with self.assertRaises(FileExistsError): _train(data,selected,scaler,root/'left',protocol)

    def test_saved_AB_and_dynamic_report(self):
        run=ROOT/'outputs/experiments/m4_20261029_run02'
        payload=read_json(run/'strategies.json')
        self.assertEqual(payload['groups']['C']['status'],'not_available')
        self.assertTrue(all(x['status']=='comparable' for x in payload['clustering_comparisons']))
        groups=[]
        for name,dataset in [('A',self.before),('B',self.historical_after)]:
            ids=pd.read_csv(run/name/'cohort.csv')['row_id'].tolist()
            subset=dataset.frame.set_index(dataset.row_id_column,drop=False).loc[ids].reset_index(drop=True)
            groups.append(load_cluster_group(replace(dataset,frame=subset),run/name,5))
        self.assertEqual(set(groups[0]['scaled'].row_ids),set(groups[1]['scaled'].row_ids))
        invalid=dict(groups[1],scaled=replace(groups[1]['scaled'],scaler_sha256='bad'))
        self.assertEqual(compare_clustering([groups[0],invalid])['status'],'unavailable')
        self.assertIsNone(compare_clustering([groups[0],invalid])['common_subset_metrics'])
        before_hash=sha256_file(run/'strategies.json')
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'report'
            localized=copy.deepcopy(payload)
            old_root='C:\\Users\\27969\\Desktop\\data_mining'
            def localize(value):
                if isinstance(value,dict):
                    for key,item in value.items():
                        if key in ('path','metrics_file') and isinstance(item,str) and item.startswith(old_root):
                            value[key]=str(ROOT/item[len(old_root):].lstrip('\\/'))
                        else:
                            localize(item)
                elif isinstance(value,list):
                    for item in value: localize(item)
            localize(localized)
            localized_path=Path(tmp)/'strategies_local.json'
            write_json(localized_path,localized)
            md=generate_report(localized_path,ROOT/'outputs/clustering/m1_20261029_run03',ROOT/'outputs/clustering/m2_20261029_k5_run02',target)
            report=read_json(target/'report.json')
            self.assertEqual(report['strategies'],localized)
            text=md.read_text(encoding='utf-8')
            self.assertIn(str(payload['protocol']['common_rows']),text)
            self.assertIn('不可计算',text)
            self.assertIn('不是',text)
            for rel,digest in read_json(target/'manifest.json')['files'].items(): self.assertEqual(sha256_file(target/rel),digest)
            with self.assertRaises(FileExistsError): generate_report(localized_path,ROOT/'outputs/clustering/m1_20261029_run03',ROOT/'outputs/clustering/m2_20261029_k5_run02',target)
            corrupted=copy.deepcopy(localized); corrupted['sources'][0]['sha256']='0'*64
            write_json(Path(tmp)/'corrupt.json',corrupted)
            with self.assertRaises(ValueError): generate_report(Path(tmp)/'corrupt.json',ROOT/'outputs/clustering/m1_20261029_run03',ROOT/'outputs/clustering/m2_20261029_k5_run02',Path(tmp)/'badreport')
        self.assertEqual(before_hash,sha256_file(run/'strategies.json'))


if __name__=='__main__': unittest.main()
