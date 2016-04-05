from setuptools import setup

from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst')) as f:
    long_description = f.read()

setup(name='emva1288',
      packages=['emva1288', 'emva1288.report'],
      version='0.3.0',
      description='EMVA1288 reference implementation',
      long_description=long_description,
      author='Federico Ariza',
      author_email='ariza.federico@gmail.com',
      url='https://github.com/EMVA1288/emva1288',
      keywords=['sensors', 'cameras'],
      classifiers=['Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.4'],
      install_requires=['matplotlib',
                        'numpy',
                        'pillow',
                        'lxml',
                        'scipy',
                        'jinja2'],
      package_data={'emva1288': ['report/files/*', 'report/templates/*',
                                 'examples/*']}
      )
