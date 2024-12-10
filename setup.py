from setuptools import setup

setup(
    name='ND2',
    version='0.0.1',
    packages=['ND2'],
    url='',
    license='MIT',
    author='Zihan Yu',
    author_email='yuzh19@tsinghua.org.cn',
    description='',
    install_requires=[
        'torch',
        'numpy',
        'sympy',
        'pandas',
        'scipy',
        'scikit-learn',
        'matplotlib',
        'networkx',
        'tqdm',
        'setproctitle',
        'pyyaml',
        'IPython',
    ]
)