from setuptools import setup
import os

# Get the version from versioneer
import versioneer
__version__ = versioneer.get_version()

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.rst')) as f:
    long_description = f.read()

extra_doc = ['nbsphinx', 'ipython', 'sphinx_rtd_theme']


# On read the docs we don't install anything
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    install_requires = extra_doc
else:
    install_requires = ['matplotlib',
                        'numpy',
                        'pillow',
                        'lxml',
                        'scipy',
                        'jinja2']

setup(name='emva1288',
      packages=['emva1288',
                'emva1288.report',
                'emva1288.camera',
                'emva1288.process',
                'emva1288.unittests'],
      version=__version__,
      description='EMVA1288 reference implementation',
      long_description=long_description,
      long_description_content_type='text/x-rst',
      author='Federico Ariza',
      author_email='ariza.federico@gmail.com',
      url='https://github.com/EMVA1288/emva1288',
      keywords=['sensors', 'cameras'],
      classifiers=['Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.6'],
      install_requires=install_requires,
      extras_require={'doc': extra_doc},
      package_data={'emva1288': ['report/files/*', 'report/templates/*',
                                 'examples/*']}
      )
