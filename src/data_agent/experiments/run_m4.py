"""Run the fixed M4 protocol and generate the final artifact-based report."""
import argparse
from pathlib import Path
from .strategies import ExperimentProtocol, compare_strategies, read_json, source, write_json
from .report import generate_report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, default=Path.cwd())
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--protocol', type=Path)
    p.add_argument('--agent-bundle', type=Path)
    p.add_argument('--m1-run', type=Path, default=Path('outputs/clustering/m1_20261029_run03'))
    p.add_argument('--m2-run', type=Path, default=Path('outputs/clustering/m2_20261029_k5_run02'))
    p.add_argument('--m3-run', type=Path, default=Path('outputs/experiments/m3_20261029_run03'))
    a = p.parse_args()
    root = a.project_root.resolve()
    def absolute(path):
        return path.resolve() if path.is_absolute() else (root / path).resolve()
    output = absolute(a.output)
    protocol = ExperimentProtocol.model_validate(read_json(absolute(a.protocol))) if a.protocol else ExperimentProtocol()
    compare_strategies(root, output, protocol=protocol, m3_run=absolute(a.m3_run), agent_bundle=absolute(a.agent_bundle) if a.agent_bundle else None)
    report = generate_report(output / 'strategies.json', absolute(a.m1_run), absolute(a.m2_run), output / 'report')
    write_json(output / 'delivery_manifest.json', {'artifacts': {str(p.relative_to(output)): source(p)['sha256'] for p in output.rglob('*') if p.is_file()},
                                                'implementation': [source(p) for p in Path(__file__).parent.glob('*.py')]})
    print(f'M4 completed available groups; report={report}', flush=True)


if __name__ == '__main__':
    main()
