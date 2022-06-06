from setuptools import setup, find_packages

setup(
  name = 'enformer-pytorch-efficient',
  packages = find_packages(exclude=[]),
  include_package_data = True,
  version = '0.0',
  license='MIT',
  description = 'Enformer - Pytorch- Efficient, Fork of enformer-pytorch from github@lucidrains',
  author = 'SG',
  url = 'https://github.com/lucidrains/enformer-pytorch-efficient',
  keywords = [
    'artificial intelligence',
    'transformer',
    'gene-expression'
  ],
  install_requires=[
    'einops>=0.3',
    'numpy',
    'torch>=1.6',
    'torchmetrics',
    'polars',
    'pyfaidx',
    'pyyaml',
    'transformers',
    'performer-pytorch'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
