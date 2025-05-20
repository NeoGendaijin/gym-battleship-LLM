from setuptools import find_packages, setup

setup(
    name='gym_battleship',
    version='0.1.0',
    description='Battleship environment using OpenAI Gym toolkit with extensions for LLM-guided RL strategy distillation',
    author='Research Team',
    license='MIT',
    install_requires=[
        'gym',
        'gymnasium>=0.29.1',
        'numpy',
        'pandas',
        'stable-baselines3>=2.2.1',
        'torch>=2.1.0',
        'tqdm>=4.66.1',
        'matplotlib>=3.8.0',
        'ipython>=8.18.1',
        'openai>=1.3.5'
    ],
    packages=find_packages()
)
