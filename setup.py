from setuptools import setup

setup(
    name='deer-detector',
    version='',
    packages=[''],
    url='',
    license='beer',
    author='patrick ryan',
    author_email='theyoungsoul@gmail.com',
    description='Create a Keras image classification model to detect deer',
    install_requires = [
            'sklearn',
            'tensorflow',
            'numpy',
            'pandas',
            'matplotlib',
            'jupyter',
            'opencv-python',
            'kaggle',
            'openvino-dev',
            'blobconverter',
            'depthai'
        ]

)
