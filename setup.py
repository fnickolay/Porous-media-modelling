from setuptools import setup, find_packages

setup(
    name='svmc_pmm',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'pathlib',
        'jdata',
        'seaborn',
        'tqdm',
        'time',
        'mpl_toolkits',
        'skimage'
    ],
    author='fnickolay',
    description='Python-based functions for porous media light propagation modelling',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/fnickolay/Porous-media-modelling',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
