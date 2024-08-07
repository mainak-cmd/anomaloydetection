from setuptools import setup, find_packages

setup(
    name='anamolyDetection',
    version='0.1.0',  # Semantic versioning
    packages=find_packages(),  # Automatically find packages in the directory
    include_package_data=True,  # Include non-Python files specified in MANIFEST.in
    description='A module for anomaly detection.',
    author='Mainak',
    author_email='mainakr748@gmail.com',
    url='https://github.com/mainak-cmd/Anomaly-detection.git',  # Replace with your URL
    license='MIT',  # Choose an appropriate license
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6', 
     
    install_requires=[
        # List of dependencies with version ranges
        'requests>=2.20.0,<3.0.0',
        'numpy>=1.24.3,<2.0.0',
        'joblib >=1.2.0'
    ]
 # Specify the Python versions you support
)
