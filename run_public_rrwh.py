"""Small entrypoint for the public RRWH workflow."""

from pathlib import Path

from src import load_config, load_paths


def main():
    base_dir = Path(__file__).resolve().parent
    config_dir = base_dir / 'config'
    paths = load_paths(config_dir / 'paths.yaml')
    params = load_config(config_dir / 'parameters.yaml')

    print('Public RRWH workflow')
    print(f"Config path: {config_dir / 'paths.yaml'}")
    print(f"Parameters path: {config_dir / 'parameters.yaml'}")
    print('Configured outputs:')
    for key, value in paths.get('output', {}).items():
        print(f'  {key}: {value}')
    print('Configured distributions:', ', '.join(params.get('distributions', [])))
    print('This runner is a scaffold for wiring data loading, fitting, and mapping steps together.')


if __name__ == '__main__':
    main()
