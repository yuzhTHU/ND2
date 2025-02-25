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
        'torch==2.4.0+cu121',
        'numpy==1.26.4',
        'sympy==1.12',
        'pandas==2.2.2',
        'scipy==1.13.0',
        'scikit-learn==1.5.0',
        'matplotlib==3.8.4',
        'networkx==3.3',
        'tqdm==4.66.4',
        'setproctitle==1.3.3',
        'PyYAML==6.0.1',
        'ipython==8.22.2',
        'lxml==5.3.0',
        'geopandas==1.0.1',
    ]
)