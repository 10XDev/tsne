"""
To upload a new version:
1. make clean
2. git tag a new version: git tag v1.x.x
3. python setup.py sdist
4. python setup.py sdist register upload
"""

import sys
import os
import platform

from distutils.core import setup
from setuptools import find_packages
from distutils.extension import Extension

import versioneer
import numpy
from Cython.Distutils import build_ext
from Cython.Build import cythonize

if sys.platform == 'darwin':
    # OS X
    version, _, _ = platform.mac_ver()
    parts = version.split('.')
    v1 = int(parts[0])
    v2 = int(parts[1])
    v3 = int(parts[2]) if len(parts) == 3 else None

    if v2 >= 10:
        # More than 10.10
        extra_compile_args = [
            '-I/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers']
    else:
        extra_compile_args = [
            '-I/System/Library/Frameworks/vecLib.framework/Headers']

    ext_modules = [Extension(name='bh_sne',
                             sources=['tsne/bh_sne_src/sptree.cpp',
                                      'tsne/bh_sne_src/tsne.cpp', 'tsne/bh_sne.pyx'],
                             include_dirs=[
                                 numpy.get_include(), 'tsne/bh_sne_src/'],
                             extra_compile_args=extra_compile_args +
                             ['-ffast-math', '-O3', '-std=c++14'],
                             extra_link_args=['-Wl,-framework',
                                              '-Wl,Accelerate'],
                             language='c++')]

else:
    # LINUX
    opt_flags = ['-msse3', '-O3', '-flto']
    if 'PROFILE' in os.environ:
        if os.environ['PROFILE'] == 'generate':
            opt_flags.append('-fprofile-instr-generate')
        elif os.path.isfile(os.environ['PROFILE']):
            opt_flags.append('-fprofile-instr-use='+os.environ['PROFILE'])
    ldflags = list(opt_flags)
    if 'LDFLAGS' in os.environ and os.environ['LDFLAGS']:
        ldflags.extend(os.environ['LDFLAGS'].split(' '))
    ldflags.extend(['-Wl,-Bdynamic,--as-needed', '-lgcc_s'])
    ext_modules = [Extension(name='bh_sne',
                             sources=['tsne/bh_sne_src/sptree.cpp',
                                      'tsne/bh_sne_src/tsne.cpp', 'tsne/bh_sne.pyx'],
                             include_dirs=[
                                 numpy.get_include(), 'tsne/bh_sne_src/'],
                             extra_compile_args=opt_flags +
                             ['-Wall', '-fPIC', '-std=c++14', '-w'],
                             extra_link_args=ldflags,
                             language='c++'),
                   ]

ext_modules = cythonize(ext_modules)

with open('requirements.txt') as f:
    required = f.read().splitlines()

cmdclass = versioneer.get_cmdclass()
cmdclass['build_ext'] = build_ext

setup(name='tsne',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      author='Daniel Rodriguez',
      author_email='df.rodriguez@gmail.com',
      url='https://github.com/danielfrg/py_tsne',
      description='TSNE implementations for python',
      license='Apache License Version 2.0, January 2004',
      packages=find_packages(),
      ext_modules=ext_modules,
      install_requires=required
      )
