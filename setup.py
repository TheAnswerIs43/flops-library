from setuptools import find_packages, setup
from os import path

root_dir = path.abspath(path.dirname(__file__))

def require():
    return [name.rstrip() for name in open(path.join(root_dir, 'requirements.txt'), encoding='utf-8').readlines()]


setup(
    name='myflopslib',
    packages=find_packages(include=['myflopslib']),
    version='0.1.0',
    description='Python Library',
    author='Federico',
    license='MIT',
    install_requires=require(),
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)